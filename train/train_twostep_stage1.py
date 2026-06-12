#!/usr/bin/env python
"""Two-Step Stage1: Train TrajCo in root denoiser with root-only data.

Key design:
  1. TrajCo cross_attn injected into root_model only
  2. No SceneCo (neither root nor body) — skip scene encoding entirely
  3. Root trajectory dataset (5-dim global_root_features)
  4. Loss only on root dimensions
  5. Text features from cache (lingo_root_trajectory_smplx has text_feat)
  6. Freeze pretrained, only train TrajCo + TrajEncoder

Usage:
  CUDA_VISIBLE_DEVICES=0 python kimodo_scene_project/train/train_twostep_stage1.py \
    kimodo_scene_project/configs/twostep_stage1_trajco_root.yaml
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


def build_stage1_model(config, device):
    """Build KimodoSceneCo for Stage1: TrajCo in root_model only, no SceneCo.

    Key optimization: skip VoxelViT entirely to save GPU memory.
    """
    trajco_cfg = config.get("trajco", {})
    trajco_type = trajco_cfg.get("trajco_type", "cross_attn")

    # Load pretrained Kimodo
    pretrained = load_model("Kimodo-SMPLX-RP-v1", device="cpu",
                             text_encoder=ZeroTextEncoder())
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model

    # Build KimodoSceneCo with TrajCo in root_model only
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
        use_in_root_model=False,  # No SceneCo in root
        use_in_body_model=False,  # No SceneCo in body
        use_trajco=True,
        use_trajco_root=True,     # TrajCo in root_model
        use_trajco_body=False,    # No TrajCo in body_model
        traj_dim=5,
        trajco_type=trajco_type,
    ).to(device)

    del pretrained
    gc.collect()

    # Freeze: only TrajCo + TrajEncoder trainable
    model.freeze_for_trajco()

    # Move VoxelViT to CPU to save GPU memory (we don't use it for Stage1)
    if hasattr(model, 'scene_encoder'):
        model.scene_encoder.cpu()
    if hasattr(model, 'scene_encoder_root') and model.use_dual_vit:
        model.scene_encoder_root.cpu()
        model.scene_encoder_body.cpu()
    torch.cuda.empty_cache()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Stage1 model: trainable={trainable:,} / total={total:,} ({100*trainable/max(1,total):.1f}%)")

    return model


class Stage1RootLoss(nn.Module):
    """Diffusion loss that only trains root features."""

    def __init__(self, diffusion, loss_mask):
        super().__init__()
        self.diffusion = diffusion
        self.loss_mask = loss_mask  # [1, 1, D] — 1 for root, 0 for body

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
            scene_feat_root=None,  # No scene features for Stage1
            scene_mask_root=None,
            scene_feat_body=None,
            scene_mask_body=None,
            traj_feats=model_kwargs.get("traj_feats"),
            traj_mask=model_kwargs.get("traj_mask"),
        )

        # Loss only on root features
        mask = model_kwargs["x_pad_mask"].unsqueeze(-1).float()
        mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
        mse = mse * self.loss_mask
        mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        # Monitor body MSE separately (no gradient)
        with torch.no_grad():
            body_mask = 1.0 - self.loss_mask
            body_mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
            body_mse = (body_mse * body_mask).sum(dim=-1) * mask.squeeze(-1)
            body_mse = body_mse.sum() / mask.sum()

        return {"loss": mse, "mse": mse.detach(), "body_mse": body_mse.detach()}


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

        root_pos = motion[:length, :3]
        root_heading = motion[:length, 3:5]
        voxel_occupied = (voxel > 0).sum().item()
        voxel_total = voxel.numel()

        log.info(f"  Sample {i}: scene={scene_name}, len={length}, "
                 f"text='{text[:50]}...', "
                 f"root_range=[{root_pos.min():.2f},{root_pos.max():.2f}], "
                 f"heading_range=[{root_heading.min():.2f},{root_heading.max():.2f}], "
                 f"voxel_occupancy={voxel_occupied/voxel_total*100:.1f}%")

        if torch.isnan(motion).any():
            log.warning(f"  WARNING: NaN in motion features!")
        if root_pos.abs().max() < 1e-6:
            log.warning(f"  WARNING: Root trajectory is all zeros!")
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
    if "text_feat" in batch:
        tf = batch["text_feat"]
        log.info(f"  text_feat: shape={tf.shape}, range=[{tf.min():.2f}, {tf.max():.2f}]")
    else:
        log.info(f"  text_feat: NOT IN BATCH (will use zeros)")
    log.info(f"  texts (first 3): {batch['texts'][:3]}")
    log.info(f"  lengths: {batch['lengths'].tolist()[:4]}")

    root_trajco = getattr(model.denoiser.model.root_model, 'trajco_layers', None)
    body_trajco = getattr(model.denoiser.model.body_model, 'trajco_layers', None)
    log.info(f"  trajco_layers: root={len(root_trajco) if root_trajco else 0}, body={len(body_trajco) if body_trajco else 0}")

    if root_trajco:
        for i, layer in enumerate(root_trajco):
            if hasattr(layer, 'alpha'):
                log.info(f"  root trajco[{i}] alpha = {layer.alpha.item():.5f} (sigmoid={torch.sigmoid(layer.alpha).item():.5f})")

    # Test forward pass
    motion_dev = motion.to(device)
    mask_dev = mask.to(device)
    B = motion_dev.shape[0]
    t = torch.randint(0, 1000, (B,), device=device)

    # Text features
    if "text_feat" in batch and batch["text_feat"] is not None:
        text_feat = batch["text_feat"].to(device)
    else:
        text_feat = torch.zeros(B, 1, 4096, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

    # Traj features
    root_slice = model.motion_rep.root_slice
    traj_raw = motion_dev[..., root_slice]
    traj_mask_out = mask_dev.clone()
    traj_feats, traj_mask_out = model.encode_traj(traj_raw, traj_mask_out)
    log.info(f"  traj_feats: shape={traj_feats.shape}, range=[{traj_feats.min():.4f},{traj_feats.max():.4f}]")

    # First heading angle
    heading = motion_dev[:, 0, 3:5]
    first_heading_angle = torch.atan2(heading[:, 1], heading[:, 0])

    with torch.no_grad():
        noise = torch.randn_like(motion_dev)
        x_t = model.diffusion.q_sample(motion_dev, t, noise=noise)
        pred = model.denoiser.model(
            x_t, mask_dev, text_feat, text_pad_mask, t,
            first_heading_angle=first_heading_angle,
            traj_feats=traj_feats, traj_mask=traj_mask_out,
        )
    log.info(f"  pred_x0: shape={pred.shape}, range=[{pred.min():.2f},{pred.max():.2f}]")
    log.info(f"  pred root: range=[{pred[..., :5].min():.2f},{pred[..., :5].max():.2f}]")
    log.info("=" * 60)


def prepare_batch(model, batch, device, training=True):
    """Process raw batch into model_kwargs. No scene encoding for Stage1."""
    motion = batch["motion_features"].to(device)
    mask = batch["motion_mask"].to(device)
    texts = batch["texts"]
    lengths = batch["lengths"]

    B = motion.shape[0]

    # Text features: use cached text_feat if available
    if "text_feat" in batch and batch["text_feat"] is not None:
        text_feat = batch["text_feat"].to(device)
        text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    else:
        text_feat = torch.zeros(B, 1, 4096, device=device)
        text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

    # First heading angle from motion features
    heading = motion[:, 0, 3:5]
    first_heading_angle = torch.atan2(heading[:, 1], heading[:, 0])

    # Trajectory features for TrajCo
    traj_feats, traj_mask_out = None, None
    has_traj_encoder = hasattr(model, 'traj_encoder') and model.traj_encoder is not None
    if has_traj_encoder:
        root_slice = model.motion_rep.root_slice
        traj_raw = motion[..., root_slice]
        traj_mask_out = mask.clone()

        # Traj dropout during training
        if training:
            traj_dropout = 0.1
            drop_traj = torch.rand(B) < traj_dropout
            traj_mask_out[drop_traj] = False

        traj_feats, traj_mask_out = model.encode_traj(traj_raw, traj_mask_out)

    # Loss mask: 1 for root features, 0 for body features
    D = motion.shape[-1]
    root_slice = model.motion_rep.root_slice
    loss_mask = torch.zeros(1, 1, D, device=device)
    loss_mask[..., root_slice] = 1.0

    return {
        "x_start": motion,
        "x_pad_mask": mask,
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
    parser.add_argument("--resume", type=str, default=None)
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
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 800)
    lr = float(train_cfg.get("lr", 1e-4))
    seed = train_cfg.get("seed", 42)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4)
    max_frames = data_cfg.get("max_frames", 196)
    min_frames = data_cfg.get("min_frames", 40)
    fps = data_cfg.get("fps", 30)
    train_ratio = data_cfg.get("train_ratio", 0.9)
    voxel_size = tuple(data_cfg.get("voxel_size", [64, 64, 64]))
    cache_dir = data_cfg.get("traj_data_dir", data_cfg.get("cache_dir", "lingo_root_trajectory_smplx"))
    data_root = data_cfg.get("data_root", "LINGO/dataset")
    log_interval = train_cfg.get("log_interval", 50)
    val_interval = train_cfg.get("val_interval", 500)
    val_max_batches = train_cfg.get("val_max_batches", 10)
    num_base_steps = train_cfg.get("num_base_steps", 1000)
    accum_steps = train_cfg.get("accum_steps", 1)
    save_every_epochs = (
        args.save_every_epochs
        if args.save_every_epochs is not None
        else train_cfg.get("save_every_epochs", 50)
    )

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

    # ----- Dataset -----
    log.info("Loading dataset ...")

    # For root trajectory data, we need motion_rep from the model
    # Build model first to get motion_rep, then build dataset
    log.info("Building Stage1 model (TrajCo root only, no SceneCo) ...")
    model = build_stage1_model(cfg, device)

    # Build dataset with motion_rep from model
    import copy
    cpu_motion_rep = copy.deepcopy(model.motion_rep)
    cpu_motion_rep.skeleton = cpu_motion_rep.skeleton.to("cpu")

    train_dataset = LINGOSceneMotionDataset(
        data_root=data_root, cache_dir=cache_dir,
        max_frames=max_frames, min_frames=min_frames, fps=fps,
        voxel_size=voxel_size, train_ratio=train_ratio, seed=seed,
        split="train", no_soma_conversion=True,
        root_trajectory_data=True,
        motion_rep=cpu_motion_rep,
    )
    val_dataset = LINGOSceneMotionDataset(
        data_root=data_root, cache_dir=cache_dir,
        max_frames=max_frames, min_frames=min_frames, fps=fps,
        voxel_size=voxel_size, train_ratio=train_ratio, seed=seed,
        split="val", no_soma_conversion=True,
        root_trajectory_data=True,
        motion_rep=cpu_motion_rep,
    )

    # Verify data alignment
    verify_data_alignment(train_dataset, n_samples=5)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=RandomSampler(train_dataset, replacement=True),
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        drop_last=True, multiprocessing_context="fork",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=RandomSampler(val_dataset, replacement=False),
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        multiprocessing_context="fork",
    )
    log.info(f"Train: {len(train_dataset)} segments, Val: {len(val_dataset)} segments")

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params) / 1e6
    log.info(f"Trainable params: {n_params:.1f}M")

    # ----- Optimizer -----
    opt = AdamW(params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs * len(train_loader), eta_min=lr * 0.01)

    # Pre-compute loss mask (1 for root, 0 for body)
    inner_denoiser = model.denoiser.model
    D = inner_denoiser.motion_rep.motion_rep_dim
    loss_mask = torch.zeros(1, 1, D, device=device)
    loss_mask[..., inner_denoiser.motion_rep.root_slice] = 1.0
    loss_fn = Stage1RootLoss(model.diffusion, loss_mask)

    # Load resume checkpoint if provided
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info(f"Resumed from epoch {start_epoch}")

    # ----- Training loop -----
    step = 0
    best_val = float("inf")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    first_batch_logged = False
    last_grad_norm = 0.0

    log.info(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}, Accum: {accum_steps}")
    log.info(f"Saving epoch checkpoints every {save_every_epochs} epochs")
    log.info(f"TrajCo: root={trajco_cfg.get('use_trajco_root',True)}, type={trajco_cfg.get('trajco_type','cross_attn')}")

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_root_mse = 0.0
        epoch_body_mse = 0.0
        n_batches = 0
        accum_count = 0

        for batch in train_loader:
            if not first_batch_logged:
                log_first_batch(model, batch, device)
                first_batch_logged = True

            kwargs = prepare_batch(model, batch, device, training=True)
            B = kwargs["x_start"].shape[0]
            t = torch.randint(0, num_base_steps, (B,), device=device)

            losses = loss_fn.training_losses(model.denoiser, kwargs["x_start"], t, kwargs)
            loss = losses["loss"] / accum_steps
            loss.backward()

            accum_count += 1
            step += 1
            epoch_loss_sum += losses["loss"].item()
            epoch_root_mse += losses["mse"].item()
            epoch_body_mse += losses["body_mse"].item()
            n_batches += 1

            if accum_count >= accum_steps:
                last_grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                scheduler.step()
                opt.zero_grad()
                accum_count = 0

            if step % log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                # TrajCo alpha values
                alpha_vals = []
                if hasattr(model.denoiser.model.root_model, 'trajco_layers') and model.denoiser.model.root_model.trajco_layers:
                    for layer in model.denoiser.model.root_model.trajco_layers:
                        if hasattr(layer, 'alpha'):
                            alpha_vals.append(f"{torch.sigmoid(layer.alpha).item():.4f}")
                alpha_str = ",".join(alpha_vals) if alpha_vals else "N/A"

                log.info(
                    f"[Epoch {epoch}/{num_epochs}] step {step} "
                    f"loss={losses['loss'].item():.4f} "
                    f"root_mse={losses['mse'].item():.4f} "
                    f"body_mse={losses['body_mse'].item():.4f} "
                    f"trajco_alpha=[{alpha_str}] "
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

        # Handle remaining accumulation
        if accum_count > 0:
            last_grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            scheduler.step()
            opt.zero_grad()

        # Epoch summary
        avg_loss = epoch_loss_sum / max(n_batches, 1)
        avg_root = epoch_root_mse / max(n_batches, 1)
        avg_body = epoch_body_mse / max(n_batches, 1)
        log.info(f"Epoch {epoch}/{num_epochs}: loss={avg_loss:.4f} root_mse={avg_root:.4f} body_mse={avg_body:.4f} best_val={best_val:.4f}")

        if save_every_epochs > 0 and epoch % save_every_epochs == 0:
            torch.save({
                "epoch": epoch, "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }, ckpt_dir / f"epoch_{epoch:04d}.pt")

    log.info(f"Training complete! Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
