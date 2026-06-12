#!/usr/bin/env python
"""Two-Step Stage2: Train SceneCo in body denoiser with frozen TrajCo root.

Key design:
  1. Load Stage1 checkpoint (TrajCo in root_model, already trained)
  2. Build model with TrajCo (root) + SceneCo (body)
  3. GT local_root replaces root model prediction (root model bypassed)
  4. Only body features contribute to loss (root loss zeroed)
  5. Root model (including TrajCo) is frozen; SceneCo + VoxelViT trainable
  6. Text features from cache (text_feat in lingo_smplx_cache)

Usage:
  CUDA_VISIBLE_DEVICES=0 python kimodo_scene_project/train/train_twostep_stage2.py \
    kimodo_scene_project/configs/twostep_stage2_sceneco_body.yaml \
    --stage1_ckpt kimodo_scene_project/outputs/twostep_stage1_trajco_root/checkpoints/best_checkpoint.pt
"""

import argparse
import gc
import logging
import os
import sys
import types
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

os.environ["CHECKPOINT_DIR"] = str(PROJECT_ROOT / "kimodo_scene_project/models")

from kimodo_sceneco.model.kimodo_model import KimodoSceneCo
from kimodo.model.load_model import load_model
from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset, collate_fn

log = logging.getLogger(__name__)


class ZeroTextEncoder:
    output_dim = 4096; llm_dim = 4096; max_len = 77
    def __call__(self, texts, device=None):
        B = len(texts)
        feat = torch.zeros(B, 1, self.output_dim)
        length = torch.ones(B, dtype=torch.long)
        if device is not None:
            feat, length = feat.to(device), length.to(device)
        return feat, length
    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self


def build_stage2_model(config, device, stage1_ckpt_path=None):
    """Build KimodoSceneCo model for Stage2: TrajCo (root, frozen) + SceneCo (body, trainable).

    Steps:
      1. Load pretrained Kimodo
      2. Build KimodoSceneCo with TrajCo in root_model + SceneCo in body_model
      3. Load Stage1 checkpoint (TrajCo weights in root_model)
      4. Patch denoiser.forward to use GT local_root (skip root model)
      5. Freeze root_model; only SceneCo + VoxelViT + traj_encoder trainable
    """
    use_in_root = config.get("sceneco", {}).get("use_in_root_model", False)
    use_in_body = config.get("sceneco", {}).get("use_in_body_model", True)

    trajco_cfg = config.get("trajco", {})
    trajco_type = trajco_cfg.get("trajco_type", "cross_attn")
    use_trajco_root = trajco_cfg.get("use_trajco_root", True)
    use_trajco_body = trajco_cfg.get("use_trajco_body", False)

    # ----- Load pretrained Kimodo + SceneCo wrapper -----
    pretrained = load_model("Kimodo-SMPLX-RP-v1", device="cpu",
                             text_encoder=ZeroTextEncoder())
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model

    model = KimodoSceneCo(
        denoiser=inner,
        text_encoder=pretrained.text_encoder,
        num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64),
            "patch_size": (8, 8, 8),
            "d_model": 256,
            "num_layers": 4,
            "use_dual_vit": False,
            "root_voxel_mode": "full",
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=use_in_root,
        use_in_body_model=use_in_body,
        use_trajco=True,  # needed for traj_encoder creation
        use_trajco_root=use_trajco_root,
        use_trajco_body=use_trajco_body,
        traj_dim=5,
        trajco_type=trajco_type,
    ).to(device)

    del pretrained
    gc.collect()

    # ----- Load Stage1 checkpoint (TrajCo weights) -----
    if stage1_ckpt_path and Path(stage1_ckpt_path).exists():
        ckpt = torch.load(stage1_ckpt_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info(f"Loaded Stage1 checkpoint from {stage1_ckpt_path}")
        log.info(f"  Missing keys: {len(missing)} (expected: SceneCo body layers)")
        log.info(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            # Filter to show only SceneCo-related missing keys (expected)
            sceneco_missing = [k for k in missing if "sceneco" in k or "scene_encoder" in k or "scene_null" in k]
            other_missing = [k for k in missing if k not in sceneco_missing]
            if sceneco_missing:
                log.info(f"  SceneCo missing (expected, new layers): {len(sceneco_missing)}")
            if other_missing:
                log.warning(f"  Other missing (unexpected): {other_missing[:10]}")
    else:
        log.warning(f"Stage1 checkpoint not found: {stage1_ckpt_path}")
        log.warning("Training from scratch (TrajCo will be untrained!)")

    # ----- Patch the inner denoiser.forward to use GT local_root -----
    inner_denoiser = model.denoiser.model
    motion_rep = inner_denoiser.motion_rep

    def _patched_forward(_self, x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                         first_heading_angle=None, motion_mask=None, observed_motion=None,
                         scene_feat=None, scene_mask=None,
                         scene_feat_root=None, scene_mask_root=None,
                         scene_feat_body=None, scene_mask_body=None,
                         traj_feats=None, traj_mask=None,
                         cakey_kwargs_root=None, cakey_kwargs_body=None,
                         gt_motion=None):
        """Patched: skip root model, use GT local_root for body stage."""
        rep = _self.motion_rep
        mask_mode = _self.motion_mask_mode

        _feat_body = scene_feat_body if scene_feat_body is not None else scene_feat
        _mask_body = scene_mask_body if scene_mask_body is not None else scene_mask

        if mask_mode == "concat":
            if motion_mask is None or observed_motion is None:
                motion_mask = torch.zeros_like(x)
                observed_motion = torch.zeros_like(x)
            x = x * (1 - motion_mask) + observed_motion * motion_mask

        # Compute GT local_root directly (skip root model)
        lengths = x_pad_mask.sum(-1)
        root_motion_local = rep.global_root_to_local_root(
            gt_motion[..., rep.root_slice], normalized=True, lengths=lengths,
        ).detach()

        body_x = x[..., rep.body_slice]
        x_new = torch.cat([root_motion_local, body_x], axis=-1)

        if mask_mode == "concat":
            x_new_extended = torch.cat([x_new, motion_mask], axis=-1)
        else:
            x_new_extended = x_new

        # Stage 2: body prediction with scene + traj features
        body_has_mods = (
            (hasattr(_self.body_model, 'sceneco_layers') and _self.body_model.sceneco_layers) or
            (hasattr(_self.body_model, 'trajco_layers') and _self.body_model.trajco_layers) or
            (hasattr(_self.body_model, 'cakey_layers') and _self.body_model.cakey_layers)
        )
        if body_has_mods:
            predicted_body = _self.body_model(
                x_new_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle=first_heading_angle,
                scene_feat=_feat_body, scene_mask=_mask_body,
                traj_feats=traj_feats, traj_mask=traj_mask,
                cakey_kwargs=cakey_kwargs_body,
            )
        else:
            predicted_body = _self.body_model(
                x_new_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle=first_heading_angle,
            )

        # Output: GT root + predicted body
        output = x.clone()
        output[..., rep.root_slice] = gt_motion[..., rep.root_slice]
        output[..., rep.body_slice] = predicted_body
        return output

    inner_denoiser.forward = types.MethodType(_patched_forward, inner_denoiser)

    # ----- Freeze: root_model completely (including TrajCo), body pretrained frozen -----
    # Only SceneCo + VoxelViT + scene_null_embed trainable
    for name, param in model.named_parameters():
        if any(kw in name for kw in ("sceneco", "scene_encoder", "scene_null_embed", "voxel_vit")):
            param.requires_grad = True
        elif "root_model" in name:
            param.requires_grad = False  # Freeze root_model including TrajCo
        else:
            param.requires_grad = False  # Freeze body pretrained weights

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Stage2 model: trainable={trainable:,} / total={total:,} ({100*trainable/max(1,total):.1f}%)")

    # Log TrajCo alpha values (should be trained from Stage1)
    for name, param in model.named_parameters():
        if "trajco" in name and "alpha" in name:
            log.info(f"  TrajCo alpha: {name} = {param.item():.5f} (frozen)")

    return model


class Stage2BodyLoss(nn.Module):
    """Diffusion loss that only trains body features with GT root conditioning."""

    def __init__(self, diffusion, loss_mask):
        super().__init__()
        self.diffusion = diffusion
        self.loss_mask = loss_mask  # [1, 1, D] — 0 for root, 1 for body

    def training_losses(self, model, x_start, t, model_kwargs):
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        # Use inner denoiser directly (bypass CFG wrapper)
        inner = model.model
        pred_x0 = inner(
            x_t,
            model_kwargs["x_pad_mask"],
            model_kwargs["text_feat"],
            model_kwargs["text_pad_mask"],
            t,
            first_heading_angle=model_kwargs.get("first_heading_angle"),
            motion_mask=model_kwargs.get("motion_mask"),
            observed_motion=model_kwargs.get("observed_motion"),
            scene_feat_root=model_kwargs.get("scene_feat_root"),
            scene_mask_root=model_kwargs.get("scene_mask_root"),
            scene_feat_body=model_kwargs.get("scene_feat_body"),
            scene_mask_body=model_kwargs.get("scene_mask_body"),
            traj_feats=model_kwargs.get("traj_feats"),
            traj_mask=model_kwargs.get("traj_mask"),
            gt_motion=x_start,
        )

        # Loss only on body features
        mask = model_kwargs["x_pad_mask"].unsqueeze(-1).float()
        mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
        mse = mse * self.loss_mask
        mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        # Monitor root MSE separately (no gradient)
        with torch.no_grad():
            root_mask = 1.0 - self.loss_mask
            root_mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
            root_mse = (root_mse * root_mask).sum(dim=-1) * mask.squeeze(-1)
            root_mse = root_mse.sum() / mask.sum()

        return {"loss": mse, "mse": mse.detach(), "root_mse": root_mse.detach()}


def verify_data_alignment(dataset, n_samples=5):
    """Verify trajectory and scene alignment in the dataset."""
    log.info("=" * 60)
    log.info("DATA ALIGNMENT VERIFICATION")
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        motion = sample["motion_features"]
        voxel = sample["voxel_grid"]
        text = sample["text"]
        scene_name = sample["scene_name"]
        length = sample["length"]

        # Check root trajectory
        root_pos = motion[:length, :3]  # smooth_root_pos (X, Y, Z)
        root_heading = motion[:length, 3:5]  # cos, sin

        # Check scene voxel
        voxel_occupied = (voxel > 0).sum().item()
        voxel_total = voxel.numel()

        log.info(f"  Sample {i}: scene={scene_name}, len={length}, "
                 f"text='{text[:50]}...', "
                 f"root_range=[{root_pos.min():.2f},{root_pos.max():.2f}], "
                 f"heading_range=[{root_heading.min():.2f},{root_heading.max():.2f}], "
                 f"voxel_occupancy={voxel_occupied/voxel_total*100:.1f}%")

        # Check for NaN
        if torch.isnan(motion).any():
            log.warning(f"  WARNING: NaN in motion features!")
        if torch.isnan(voxel).any():
            log.warning(f"  WARNING: NaN in voxel grid!")

        # Check trajectory is not all zeros
        if root_pos.abs().max() < 1e-6:
            log.warning(f"  WARNING: Root trajectory is all zeros!")

        # Check scene is not empty
        if voxel_occupied == 0:
            log.warning(f"  WARNING: Scene voxel is empty!")

    log.info("=" * 60)


def log_first_batch(model, batch, device):
    """Diagnostic logging for first batch."""
    log.info("=" * 60)
    log.info("FIRST BATCH DIAGNOSTIC")
    motion = batch["motion_features"]
    log.info(f"  motion: shape={motion.shape}, range=[{motion.min():.2f}, {motion.max():.2f}]")
    mask = batch["motion_mask"]
    log.info(f"  motion_mask: valid_frames={mask.sum(dim=1).tolist()[:4]}")
    voxel = batch["voxel_grid"]
    log.info(f"  voxel: shape={voxel.shape}, range=[{voxel.min():.4f}, {voxel.max():.4f}], nonzero={(voxel>0).float().mean()*100:.1f}%")
    if "text_feat" in batch:
        tf = batch["text_feat"]
        log.info(f"  text_feat: shape={tf.shape}, range=[{tf.min():.2f}, {tf.max():.2f}]")
    else:
        log.info(f"  text_feat: NOT IN BATCH (will use zeros)")
    log.info(f"  texts (first 3): {batch['texts'][:3]}")
    log.info(f"  lengths: {batch['lengths'].tolist()[:4]}")

    # Check encoded scene features
    voxel_dev = voxel.to(device)
    with torch.no_grad():
        (sf_root, sm_root), (sf_body, sm_body) = model.encode_scene(voxel_dev)
    log.info(f"  scene_feat_root: shape={sf_root.shape}, range=[{sf_root.min():.4f}, {sf_root.max():.4f}]")
    log.info(f"  scene_mask_root: valid={sm_root.float().mean()*100:.1f}%")
    log.info(f"  scene_feat_body: shape={sf_body.shape}, range=[{sf_body.min():.4f}, {sf_body.max():.4f}]")
    log.info(f"  scene_mask_body: valid={sm_body.float().mean()*100:.1f}%")

    root_sceneco = getattr(model.denoiser.model.root_model, 'sceneco_layers', None)
    body_sceneco = getattr(model.denoiser.model.body_model, 'sceneco_layers', None)
    root_trajco = getattr(model.denoiser.model.root_model, 'trajco_layers', None)
    body_trajco = getattr(model.denoiser.model.body_model, 'trajco_layers', None)
    log.info(f"  sceneco_layers: root={len(root_sceneco) if root_sceneco else 0}, body={len(body_sceneco) if body_sceneco else 0}")
    log.info(f"  trajco_layers: root={len(root_trajco) if root_trajco else 0}, body={len(body_trajco) if body_trajco else 0}")

    # Check TrajCo alpha values
    if root_trajco:
        for i, layer in enumerate(root_trajco):
            if hasattr(layer, 'alpha'):
                log.info(f"  root trajco[{i}] alpha = {layer.alpha.item():.5f} (sigmoid={torch.sigmoid(layer.alpha).item():.5f})")
    log.info("=" * 60)


def prepare_batch(model, batch, device, training=True):
    """Process raw batch into model_kwargs."""
    motion = batch["motion_features"].to(device)
    mask = batch["motion_mask"].to(device)
    voxel = batch["voxel_grid"].to(device)
    texts = batch["texts"]
    lengths = batch["lengths"]

    # Encode scene voxel -> patch-wise features
    (scene_feat_root, scene_mask_root), (scene_feat_body, scene_mask_body) = model.encode_scene(voxel)

    # Text features: use cached text_feat if available, otherwise zeros
    B = motion.shape[0]
    if "text_feat" in batch and batch["text_feat"] is not None:
        text_feat = batch["text_feat"].to(device)
        text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    else:
        text_feat = torch.zeros(B, 1, 4096, device=device)
        text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

    # First heading angle from motion features (indices 3:5 = cos/sin of global heading)
    heading = motion[:, 0, 3:5]
    first_heading_angle = torch.atan2(heading[:, 1], heading[:, 0])

    # Trajectory features for TrajCo (root_model, but we still encode for monitoring)
    traj_feats, traj_mask_out = None, None
    has_traj_encoder = hasattr(model, 'traj_encoder') and model.traj_encoder is not None
    if has_traj_encoder:
        root_slice = model.motion_rep.root_slice
        traj_raw = motion[..., root_slice]
        traj_mask_out = mask.clone()
        traj_feats, traj_mask_out = model.encode_traj(traj_raw, traj_mask_out)

    # Loss mask: 0 for root features, 1 for body features
    D = motion.shape[-1]
    root_slice = model.motion_rep.root_slice
    loss_mask = torch.zeros(1, 1, D, device=device)
    loss_mask[..., model.motion_rep.body_slice] = 1.0

    return {
        "x_start": motion,
        "x_pad_mask": mask,
        "scene_feat_root": scene_feat_root,
        "scene_mask_root": scene_mask_root,
        "scene_feat_body": scene_feat_body,
        "scene_mask_body": scene_mask_body,
        "text_feat": text_feat,
        "text_pad_mask": text_pad_mask,
        "first_heading_angle": first_heading_angle,
        "motion_mask": None,
        "observed_motion": None,
        "cfg_type": "nocfg",
        "loss_mask": loss_mask,
        "traj_feats": traj_feats,
        "traj_mask": traj_mask_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                        help="Path to Stage1 checkpoint (TrajCo root)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Stage2 checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_every_epochs", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    trajco_cfg = cfg.get("trajco", {})

    output_dir = Path(args.output_dir or cfg["output_dir"])
    batch_size = args.batch_size or train_cfg.get("batch_size", 4)
    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 40)
    lr = float(train_cfg.get("lr", 1e-4))
    seed = train_cfg.get("seed", 42)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4)
    max_frames = data_cfg.get("max_frames", 196)
    min_frames = data_cfg.get("min_frames", 40)
    fps = data_cfg.get("fps", 30)
    train_ratio = data_cfg.get("train_ratio", 0.9)
    voxel_size = tuple(data_cfg.get("voxel_size", [64, 64, 64]))
    cache_dir = data_cfg.get("cache_dir", "lingo_smplx_cache")
    data_root = data_cfg.get("data_root", "LINGO/dataset")
    log_interval = train_cfg.get("log_interval", 50)
    val_interval = train_cfg.get("val_interval", 500)
    val_max_batches = train_cfg.get("val_max_batches", 10)
    num_base_steps = train_cfg.get("num_base_steps", 1000)
    save_every_epochs = (
        args.save_every_epochs
        if args.save_every_epochs is not None
        else train_cfg.get("save_every_epochs", 50)
    )

    # Stage1 checkpoint path
    stage1_ckpt = args.stage1_ckpt or cfg.get("stage1_checkpoint", None)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ----- Setup logging -----
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(output_dir / "train.log"), logging.StreamHandler()],
    )
    log.info(f"Device: {device}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Stage1 checkpoint: {stage1_ckpt}")

    # ----- Dataset -----
    log.info("Loading dataset ...")
    train_dataset = LINGOSceneMotionDataset(
        data_root=data_root, cache_dir=cache_dir,
        max_frames=max_frames, min_frames=min_frames, fps=fps,
        voxel_size=voxel_size, train_ratio=train_ratio, seed=seed,
        split="train", no_soma_conversion=True,
    )
    val_dataset = LINGOSceneMotionDataset(
        data_root=data_root, cache_dir=cache_dir,
        max_frames=max_frames, min_frames=min_frames, fps=fps,
        voxel_size=voxel_size, train_ratio=train_ratio, seed=seed,
        split="val", no_soma_conversion=True,
    )

    # Verify data alignment
    verify_data_alignment(train_dataset, n_samples=5)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=RandomSampler(train_dataset, replacement=True),
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=RandomSampler(val_dataset, replacement=False),
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    log.info(f"Train: {len(train_dataset)} segments, Val: {len(val_dataset)} segments")

    # ----- Build model -----
    log.info("Building Stage2 model (TrajCo root + SceneCo body) ...")
    model = build_stage2_model(cfg, device, stage1_ckpt)

    # Load Stage2 resume checkpoint if provided
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        log.info(f"Resumed Stage2 from {args.resume}")

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params) / 1e6
    log.info(f"Trainable params: {n_params:.1f}M")

    # ----- Optimizer -----
    opt = AdamW(params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs * len(train_loader), eta_min=lr * 0.01)

    # Pre-compute loss mask
    inner_denoiser = model.denoiser.model
    D = inner_denoiser.motion_rep.motion_rep_dim
    loss_mask = torch.zeros(1, 1, D, device=device)
    loss_mask[..., inner_denoiser.motion_rep.body_slice] = 1.0
    loss_fn = Stage2BodyLoss(model.diffusion, loss_mask)

    # ----- Training loop -----
    step = 0
    best_val = float("inf")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    first_batch_logged = False
    last_grad_norm = 0.0

    log.info(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}")
    log.info(f"Saving epoch checkpoints every {save_every_epochs} epochs")
    log.info(f"SceneCo: root_model={cfg.get('sceneco',{}).get('use_in_root_model',False)}, body_model={cfg.get('sceneco',{}).get('use_in_body_model',True)}")
    log.info(f"TrajCo: root={trajco_cfg.get('use_trajco_root',True)}, body={trajco_cfg.get('use_trajco_body',False)}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_body_mse = 0.0
        epoch_root_mse = 0.0
        n_batches = 0

        for batch in train_loader:
            if not first_batch_logged:
                log_first_batch(model, batch, device)
                first_batch_logged = True

            # Prepare batch
            kwargs = prepare_batch(model, batch, device, training=True)

            B = kwargs["x_start"].shape[0]
            t = torch.randint(0, num_base_steps, (B,), device=device)

            losses = loss_fn.training_losses(model.denoiser, kwargs["x_start"], t, kwargs)

            opt.zero_grad()
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()

            step += 1
            epoch_loss_sum += losses["loss"].item()
            epoch_body_mse += losses["mse"].item()
            epoch_root_mse += losses["root_mse"].item()
            n_batches += 1
            last_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            if step % log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                # Log SceneCo alpha values
                alpha_vals = []
                if hasattr(model.denoiser.model.body_model, 'sceneco_layers') and model.denoiser.model.body_model.sceneco_layers:
                    for layer in model.denoiser.model.body_model.sceneco_layers:
                        if hasattr(layer, 'alpha'):
                            alpha_vals.append(f"{torch.sigmoid(layer.alpha).item():.4f}")
                alpha_str = ",".join(alpha_vals) if alpha_vals else "N/A"

                log.info(
                    f"[Epoch {epoch}/{num_epochs}] step {step} "
                    f"loss={losses['loss'].item():.4f} "
                    f"body_mse={losses['mse'].item():.4f} "
                    f"root_mse={losses['root_mse'].item():.4f} "
                    f"sceneco_alpha=[{alpha_str}] "
                    f"grad_norm={last_grad_norm:.4f} lr={lr_now:.2e}"
                )

            # Validation
            if step % val_interval == 0:
                model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        if n_val >= val_max_batches:
                            break
                        val_kwargs = prepare_batch(model, val_batch, device, training=False)
                        Bv = val_kwargs["x_start"].shape[0]
                        tv = torch.randint(0, num_base_steps, (Bv,), device=device)
                        val_losses = loss_fn.training_losses(
                            model.denoiser, val_kwargs["x_start"], tv, val_kwargs)
                        val_loss += val_losses["loss"].item()
                        n_val += 1

                avg_val = val_loss / max(n_val, 1)
                log.info(f"  VAL step {step}: val_loss={avg_val:.4f}")
                if avg_val < best_val:
                    best_val = avg_val
                    torch.save({
                        "epoch": epoch, "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "val_loss": avg_val,
                    }, ckpt_dir / "best_checkpoint.pt")
                    log.info(f"  Saved best checkpoint")
                model.train()

        # Epoch summary
        avg_loss = epoch_loss_sum / max(n_batches, 1)
        avg_body = epoch_body_mse / max(n_batches, 1)
        avg_root = epoch_root_mse / max(n_batches, 1)
        log.info(f"Epoch {epoch}/{num_epochs}: loss={avg_loss:.4f} body_mse={avg_body:.4f} root_mse={avg_root:.4f} best_val={best_val:.4f}")

        if save_every_epochs > 0 and epoch % save_every_epochs == 0:
            torch.save({
                "epoch": epoch, "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }, ckpt_dir / f"epoch_{epoch:04d}.pt")

    log.info(f"Training complete! Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
