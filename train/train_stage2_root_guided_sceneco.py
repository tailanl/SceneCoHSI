#!/usr/bin/env python
"""Stage2 Root-Guided SceneCo training.

Trains SceneCo body-only adapter with external guided root:
  - Root Stage: NOT predicted; uses external_root (guided_root_5d).
  - Body Stage: SceneCo body-only cross-attention.
  - TrajCo: disabled.
  - Loss: body_slice only (root is external, not learned).
  - Freeze: pretrained backbone, train scene_encoder + body SceneCo adapter.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train/train_stage2_root_guided_sceneco.py \\
    configs/stage2_root_guided_sceneco.yaml \\
    --gpu 0
"""

import argparse
import copy
import gc
import logging
import os
import sys
import types
from pathlib import Path
from typing import Dict, Optional

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
sys.path.insert(0, str(PROJECT_ROOT / "kimodo_scene_project"))  # after kimodo -> higher priority

os.environ["CHECKPOINT_DIR"] = str(PROJECT_ROOT / "kimodo_scene_project/models")

from kimodo.model.load_model import load_model
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo
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


def build_model(config: dict, device, ckpt_path: Optional[str] = None):
    """Build KimodoSceneCo for Stage2 Root-Guided SceneCo training.

    - Root stage: external_root used, root_model skipped entirely.
    - Body stage: SceneCo body-only cross-attention.
    - TrajCo: disabled.
    - Freeze: pretrained backbone; train scene_encoder + body SceneCo only.
    """
    sceneco_cfg = config.get("sceneco", {})

    # Load base Kimodo
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
        },
        device=device,
        cfg_type="scene_separated",
    ).to(device)

    del pretrained
    gc.collect()

    # ----- Patch denoiser.forward: skip root predictor, use external_root for local root -----
    inner_denoiser = model.denoiser.model

    def _patched_forward(_self, x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                         first_heading_angle=None, motion_mask=None, observed_motion=None,
                         scene_feat=None, scene_mask=None,
                         scene_feat_root=None, scene_mask_root=None,
                         scene_feat_body=None, scene_mask_body=None,
                         traj_feats=None, traj_mask=None,
                         cakey_kwargs_root=None, cakey_kwargs_body=None,
                         external_root=None, use_external_root=False,
                         cfg_type=None):
        """Patched forward: skip root model, use external_root for body stage."""
        rep = _self.motion_rep
        mask_mode = _self.motion_mask_mode

        _feat_body = scene_feat_body if scene_feat_body is not None else scene_feat
        _mask_body = scene_mask_body if scene_mask_body is not None else scene_mask

        if mask_mode == "concat":
            if motion_mask is None or observed_motion is None:
                motion_mask = torch.zeros_like(x)
                observed_motion = torch.zeros_like(x)
            x = x * (1 - motion_mask) + observed_motion * motion_mask

        # Use external_root to compute local_root for body conditioning
        if use_external_root and external_root is not None:
            lengths = x_pad_mask.sum(-1)
            root_motion_local = rep.global_root_to_local_root(
                external_root.to(x.dtype), normalized=True, lengths=lengths,
            ).detach()
        else:
            # Fallback: use root from x itself
            root_motion_local = rep.global_root_to_local_root(
                x[..., rep.root_slice], normalized=True, lengths=x_pad_mask.sum(-1),
            ).detach()

        body_x = x[..., rep.body_slice]
        x_new = torch.cat([root_motion_local, body_x], axis=-1)

        if mask_mode == "concat":
            x_new_extended = torch.cat([x_new, motion_mask], axis=-1)
        else:
            x_new_extended = x_new

        # Stage 2: body prediction with scene features
        body_has_mods = (
            (hasattr(_self.body_model, 'sceneco_layers') and _self.body_model.sceneco_layers) or
            (hasattr(_self.body_model, 'trajco_layers') and _self.body_model.trajco_layers) or
            (hasattr(_self.body_model, 'cakey_layers') and _self.body_model.cakey_layers)
        )
        if body_has_mods:
            extra_kwargs = {}
            if hasattr(_self.body_model, 'trajco_layers') and _self.body_model.trajco_layers:
                extra_kwargs.update(traj_feats=traj_feats, traj_mask=traj_mask)
            if hasattr(_self.body_model, 'cakey_layers') and _self.body_model.cakey_layers:
                extra_kwargs.update(cakey_kwargs=cakey_kwargs_body)
            predicted_body = _self.body_model(
                x_new_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle=first_heading_angle,
                scene_feat=_feat_body, scene_mask=_mask_body,
                **extra_kwargs,
            )
        else:
            predicted_body = _self.body_model(
                x_new_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle=first_heading_angle,
            )

        # Output: external_root (or x root) + predicted body
        output = x.clone()
        if use_external_root and external_root is not None:
            output[..., rep.root_slice] = external_root.to(x.dtype)
        output[..., rep.body_slice] = predicted_body
        return output

    inner_denoiser.forward = types.MethodType(_patched_forward, inner_denoiser)

    # ----- Freeze: only body SceneCo + scene_encoder trainable -----
    for p in inner_denoiser.root_model.parameters():
        p.requires_grad = False

    for name, param in model.named_parameters():
        if any(kw in name for kw in ("sceneco", "scene_encoder", "scene_null_embed",
                                       "voxel_vit")):
            param.requires_grad = True
        elif "root_model" in name:
            param.requires_grad = False
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Stage2 Root-Guided SceneCo: trainable={trainable:,} / total={total:,} "
             f"({100*trainable/max(1,total):.1f}%)")

    # ----- Load checkpoint -----
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        log.info(f"Loaded checkpoint from {ckpt_path}")

    return model


class Stage2RootGuidedLoss(nn.Module):
    """Diffusion loss for Stage2 Root-Guided SceneCo training.

    - x_t root is fixed to external_root (if use_external_root)
    - Loss only on body_slice
    - Monitors root MSE separately (no gradient)
    """

    def __init__(self, diffusion, loss_mask, motion_rep):
        super().__init__()
        self.diffusion = diffusion
        self.loss_mask = loss_mask  # (1, 1, D) — 1 at body_slice, 0 at root_slice
        self.motion_rep = motion_rep

    def training_losses(self, model, x_start, t, model_kwargs):
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        # Fix root in x_t if using external_root (training-inference consistency)
        if model_kwargs.get("use_external_root", False) and model_kwargs.get("external_root") is not None:
            root_slice = self.motion_rep.root_slice
            x_t[..., root_slice] = model_kwargs["external_root"].to(x_t.dtype)

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
            external_root=model_kwargs.get("external_root"),
            use_external_root=model_kwargs.get("use_external_root", False),
        )

        # Prior branch (unconditional scene)
        total_prior = None
        if model_kwargs.get("prior_weight", 0.0) > 0:
            pred_x0_null = inner(
                x_t,
                model_kwargs["x_pad_mask"],
                model_kwargs["text_feat"],
                model_kwargs["text_pad_mask"],
                t,
                first_heading_angle=model_kwargs.get("first_heading_angle"),
                motion_mask=model_kwargs.get("motion_mask"),
                observed_motion=model_kwargs.get("observed_motion"),
                scene_feat_root=None,
                scene_mask_root=None,
                scene_feat_body=None,
                scene_mask_body=None,
                traj_feats=None,
                traj_mask=None,
                external_root=model_kwargs.get("external_root"),
                use_external_root=model_kwargs.get("use_external_root", False),
            )
            total_prior = (pred_x0_null - pred_x0.detach()).pow(2).mean()

        # Body-only MSE loss
        mask = model_kwargs["x_pad_mask"].unsqueeze(-1).float()
        mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
        mse = mse * self.loss_mask
        mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        if total_prior is not None:
            prior_weight = model_kwargs["prior_weight"]
            total_loss = mse + prior_weight * total_prior
        else:
            total_loss = mse

        # Monitor root MSE (no gradient)
        with torch.no_grad():
            root_mask = 1.0 - self.loss_mask
            root_mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
            root_mse = (root_mse * root_mask).sum(dim=-1) * mask.squeeze(-1)
            root_mse = root_mse.sum() / mask.sum()

        return {"loss": total_loss, "mse": mse.detach(), "root_mse": root_mse.detach()}


def log_first_batch(model, batch, device):
    """Diagnostic logging for first batch."""
    log.info("=" * 60)
    log.info("FIRST BATCH DIAGNOSTIC")
    motion = batch["motion_features"]
    log.info(f"  motion: shape={motion.shape}, range=[{motion.min():.2f}, {motion.max():.2f}]")
    mask = batch["motion_mask"]
    log.info(f"  motion_mask: valid_frames={mask.sum(dim=1).tolist()[:4]}")
    voxel = batch["voxel_grid"]
    log.info(f"  voxel: shape={voxel.shape}, range=[{voxel.min():.4f}, {voxel.max():.4f}], "
             f"nonzero={(voxel>0).float().mean()*100:.1f}%")
    log.info(f"  texts (first 3): {batch['texts'][:3]}")
    log.info(f"  lengths: {batch['lengths'].tolist()[:4]}")

    if "external_root" in batch:
        ext_root = batch["external_root"]
        log.info(f"  external_root: shape={ext_root.shape}, range=[{ext_root.min():.4f}, {ext_root.max():.4f}]")
        log.info(f"  external_root_sources (first 3): {batch['external_root_source'][:3]}")

    voxel_dev = voxel.to(device)
    with torch.no_grad():
        scene_feat, scene_mask = model.encode_scene(voxel_dev)
    log.info(f"  scene_feat: shape={scene_feat.shape}, range=[{scene_feat.min():.4f}, {scene_feat.max():.4f}]")
    log.info(f"  scene_mask: valid={scene_mask.float().mean()*100:.1f}%")
    body_layers = getattr(model.denoiser.model.body_model, 'sceneco_layers', None)
    log.info(f"  body sceneco_layers: {len(body_layers) if body_layers else 0}")
    log.info("=" * 60)


def prepare_batch(model, batch, device, training=True):
    """Process raw batch into model_kwargs."""
    motion = batch["motion_features"].to(device)
    mask = batch["motion_mask"].to(device)
    voxel = batch["voxel_grid"].to(device)
    texts = batch["texts"]

    # Encode scene voxel (single encoder, share for root and body)
    scene_feat, scene_mask = model.encode_scene(voxel)

    # Text features (zero for scene-conditioned training)
    B = motion.shape[0]
    text_feat = torch.zeros(B, 1, 4096, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

    # First heading angle
    heading = motion[:, 0, 3:5]
    first_heading_angle = torch.atan2(heading[:, 1], heading[:, 0])

    # External root
    external_root = None
    has_external_root = "external_root" in batch
    if has_external_root:
        external_root = batch["external_root"].to(device)

    # Body-only loss mask
    D = motion.shape[-1]
    loss_mask = torch.zeros(1, 1, D, device=device)
    loss_mask[..., model.motion_rep.body_slice] = 1.0

    return {
        "x_start": motion,
        "x_pad_mask": mask,
        "scene_feat_root": scene_feat,
        "scene_mask_root": scene_mask,
        "scene_feat_body": scene_feat,
        "scene_mask_body": scene_mask,
        "text_feat": text_feat,
        "text_pad_mask": text_pad_mask,
        "first_heading_angle": first_heading_angle,
        "motion_mask": None,
        "observed_motion": None,
        "cfg_type": "nocfg",
        "loss_mask": loss_mask,
        "traj_feats": None,
        "traj_mask": None,
        "external_root": external_root,
        "use_external_root": has_external_root,
        "prior_weight": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage2 Root-Guided SceneCo Training")
    parser.add_argument("config", type=str, help="YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    # Overrides for dataset / external root
    parser.add_argument("--external_root_enabled", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--use_external_root", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--path_guided_root_dir", type=str, default=None)
    parser.add_argument("--path_scene_guided_root_dir", type=str, default=None)
    parser.add_argument("--val_root_dir", type=str, default=None)
    parser.add_argument("--root_mix_gt", type=float, default=None)
    parser.add_argument("--root_mix_path", type=float, default=None)
    parser.add_argument("--root_mix_scene", type=float, default=None)

    # Overrides for training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--prior_weight", type=float, default=None)
    parser.add_argument("--scene_dropout", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()

    # ----- Load config -----
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    ext_cfg = cfg.get("external_root", {})
    mix_cfg = cfg.get("root_condition_mix", {})

    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/stage2_root_guided_sceneco"))
    batch_size = args.batch_size or train_cfg.get("batch_size", 4)
    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 400)
    lr = args.lr or float(train_cfg.get("lr", 1e-4))
    prior_weight = args.prior_weight if args.prior_weight is not None else train_cfg.get("prior_weight", 0.0)
    scene_dropout = args.scene_dropout if args.scene_dropout is not None else train_cfg.get("scene_dropout", 0.1)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4)
    seed = train_cfg.get("seed", 42)
    max_frames = data_cfg.get("max_frames", 196)
    min_frames = data_cfg.get("min_frames", 40)
    train_ratio = data_cfg.get("train_ratio", 0.9)
    voxel_size = tuple(data_cfg.get("voxel_size", [64, 64, 64]))
    cache_dir = data_cfg.get("cache_dir", "lingo_smplx_cache")
    data_root = data_cfg.get("data_root", "LINGO/dataset")
    log_interval = train_cfg.get("log_interval", 50)
    val_interval = train_cfg.get("val_interval", 500)
    val_max_batches = train_cfg.get("val_max_batches", 10)
    num_base_steps = train_cfg.get("num_base_steps", 1000)

    # External root config
    external_root_enabled = (args.external_root_enabled if args.external_root_enabled is not None
                             else ext_cfg.get("enabled", True))
    use_external_root = (args.use_external_root if args.use_external_root is not None
                          else ext_cfg.get("use_external_root", True))
    path_guided_root_dir = args.path_guided_root_dir or ext_cfg.get("path_guided_root_dir")
    path_scene_guided_root_dir = args.path_scene_guided_root_dir or ext_cfg.get("path_scene_guided_root_dir")
    val_root_dir = args.val_root_dir or ext_cfg.get("val_root_dir")
    root_mix_gt = args.root_mix_gt if args.root_mix_gt is not None else mix_cfg.get("gt_root", 0.3)
    root_mix_path = args.root_mix_path if args.root_mix_path is not None else mix_cfg.get("path_guided_root", 0.3)
    root_mix_scene = args.root_mix_scene if args.root_mix_scene is not None else mix_cfg.get("path_scene_guided_root", 0.4)

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
    log.info(f"Config: external_root_enabled={external_root_enabled}, use_external_root={use_external_root}")
    log.info(f"Root mix: gt={root_mix_gt}, path={root_mix_path}, scene={root_mix_scene}")

    # ----- Dataset -----
    root_condition_mix = {
        "gt_root": root_mix_gt,
        "path_guided_root": root_mix_path,
        "path_scene_guided_root": root_mix_scene,
    }

    ds_kwargs = dict(
        data_root=data_root, cache_dir=cache_dir,
        max_frames=max_frames, min_frames=min_frames, fps=data_cfg.get("fps", 30),
        voxel_size=voxel_size, train_ratio=train_ratio, seed=seed,
        external_root_enabled=external_root_enabled,
        path_guided_root_dir=path_guided_root_dir,
        path_scene_guided_root_dir=path_scene_guided_root_dir,
        root_condition_mix=root_condition_mix,
        scene_dropout=scene_dropout,
    )

    log.info("Loading train dataset...")
    train_dataset = LINGOSceneMotionDataset(split="train", **ds_kwargs)

    # Val dataset: always use path_scene_guided_root
    val_ds_kwargs = dict(ds_kwargs)
    val_ds_kwargs["split"] = "val"
    val_ds_kwargs["scene_dropout"] = 0.0
    val_ds_kwargs["root_condition_mix"] = {
        "gt_root": 0.0,
        "path_guided_root": 0.0,
        "path_scene_guided_root": 1.0,
    }
    if val_root_dir:
        val_ds_kwargs["path_scene_guided_root_dir"] = val_root_dir

    log.info("Loading val dataset...")
    val_dataset = LINGOSceneMotionDataset(**val_ds_kwargs)

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
    log.info("Building Stage2 Root-Guided SceneCo model...")
    model = build_model(cfg, device, args.resume)

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
    loss_fn = Stage2RootGuidedLoss(model.diffusion, loss_mask, inner_denoiser.motion_rep)

    # ----- Training loop -----
    step = 0
    best_val = float("inf")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    first_batch_logged = False
    last_grad_norm = 0.0

    log.info(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}")
    log.info(f"SceneCo: body_only mode")
    log.info(f"Prior weight: {prior_weight}, Scene dropout: {scene_dropout}")

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

            kwargs = prepare_batch(model, batch, device, training=True)

            if prior_weight > 0:
                kwargs["prior_weight"] = prior_weight

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
                log.info(
                    f"[Epoch {epoch}/{num_epochs}] step {step} "
                    f"loss={losses['loss'].item():.4f} "
                    f"body_mse={losses['mse'].item():.4f} "
                    f"root_mse={losses['root_mse'].item():.4f} "
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
                    log.info(f"  ✓ Saved best checkpoint")
                model.train()

        # Epoch summary
        avg_loss = epoch_loss_sum / max(n_batches, 1)
        avg_body = epoch_body_mse / max(n_batches, 1)
        avg_root = epoch_root_mse / max(n_batches, 1)
        log.info(f"Epoch {epoch}/{num_epochs}: loss={avg_loss:.4f} body_mse={avg_body:.4f} "
                 f"root_mse={avg_root:.4f} best_val={best_val:.4f}")

        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch, "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }, ckpt_dir / f"epoch_{epoch:04d}.pt")

    log.info(f"Training complete! Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
