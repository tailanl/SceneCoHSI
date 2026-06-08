# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training script for CaKey + SceneCo two-stage training.

Stage 1: Train CaKey layers only.
  - Freeze all Kimodo pretrained parameters.
  - Randomly sample keyframes from motion data.
  - Train CaKey to stabilize keyframe-based inbetweening.

Stage 2: Train SceneCo on top of frozen CaKey.
  - Load Stage 1 CaKey checkpoint.
  - Freeze Kimodo backbone + CaKey layers.
  - Train VoxelViT scene encoder + SceneCo cross-attention.
"""

import argparse
import gc
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset, collate_fn
from kimodo_sceneco.model.cakey import build_keyframe_mask

log = logging.getLogger(__name__)


def sample_keyframes_for_training(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    stride_choices: list = [20, 30, 40],
    always_first_frame: bool = True,
    always_last_frame: bool = True,
    random_drop: float = 0.1,
    seed: Optional[int] = None,
):
    """Randomly sample keyframes for CaKey Stage 1 training.

    Args:
        motion: [B, N, D] clean motion features (x0).
        lengths: [B] valid frame count per sample.
        stride_choices: List of possible strides for keyframe sampling.
        always_first_frame: Always mark first frame as keyframe.
        always_last_frame: Always mark last frame as keyframe.
        random_drop: Probability of dropping a non-boundary keyframe.

    Returns:
        observed_motion: [B, N, D] motion values at keyframe positions, 0 elsewhere.
        motion_mask: [B, N, D] element-level mask, 1 at keyframe positions.
        keyframe_mask: [B, N, 1] frame-level mask, 1 at keyframe frames.
    """
    B, N, D = motion.shape
    device = motion.device
    dtype = motion.dtype

    motion_mask = torch.zeros(B, N, D, device=device, dtype=dtype)
    keyframe_mask = torch.zeros(B, N, 1, device=device, dtype=dtype)

    for b in range(B):
        T = lengths[b].item()
        stride = random.choice(stride_choices)
        frames = list(range(0, T, stride))

        if always_last_frame and (T - 1) not in frames:
            frames.append(T - 1)

        frames = sorted(set(frames))

        for f_idx in frames:
            drop = (not (f_idx == 0 or f_idx == T - 1)) and random.random() < random_drop
            if not drop:
                keyframe_mask[b, f_idx, 0] = 1.0

        if always_first_frame and T > 0:
            keyframe_mask[b, 0, 0] = 1.0
        if always_last_frame and T > 0:
            keyframe_mask[b, T - 1, 0] = 1.0

    kf_mask_3d = keyframe_mask.expand(-1, -1, D)
    motion_mask = kf_mask_3d

    observed_motion = motion * motion_mask

    return observed_motion, motion_mask, keyframe_mask


class CaKeyDiffusionLoss(nn.Module):
    """Diffusion training loss for Stage 1 CaKey training.

    Uses standard diffusion MSE. CaKey modulates latents inside the denoiser,
    so the predicted x0 is affected by keyframe-based modulation.
    """

    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        cfg_weight = [2.0, 2.0, 2.0]

        pred_x0 = model(
            cfg_weight,
            x_t,
            model_kwargs["x_pad_mask"],
            model_kwargs["text_feat"],
            model_kwargs["text_pad_mask"],
            t,
            first_heading_angle=model_kwargs.get("first_heading_angle"),
            motion_mask=model_kwargs.get("motion_mask"),
            observed_motion=model_kwargs.get("observed_motion"),
            cfg_type="nocfg",
            cakey_kwargs_root=model_kwargs.get("cakey_kwargs_root"),
            cakey_kwargs_body=model_kwargs.get("cakey_kwargs_body"),
        )

        mask = model_kwargs["x_pad_mask"].unsqueeze(-1).float()
        mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
        mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        return {
            "loss": mse,
            "mse": mse.detach(),
        }


class CaKeyTrainer:
    """Stage 1 trainer: train CaKey layers only."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        os.makedirs(args.output_dir, exist_ok=True)
        self.ckpt_dir = Path(args.output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "logs")) if SummaryWriter else None

        self._build_model_stage1()
        self._build_dataset()
        self._build_optimizer()

    def _build_model_stage1(self):
        from kimodo_sceneco.model import KimodoSceneCo
        from kimodo.model import load_model as load_kimodo_model

        cache_dir = getattr(self.args, 'cache_dir', None)
        has_cached_text = cache_dir and Path(cache_dir).exists()

        class _DummyTextEncoder:
            def __call__(self, text):
                raise RuntimeError("Text encoder should not be called in cached mode")
            def to(self, device):
                return self
            def eval(self):
                return self

        log.info("Loading pretrained Kimodo model...")
        if has_cached_text:
            text_encoder = _DummyTextEncoder()
            kimodo_pretrained = load_kimodo_model(
                self.args.pretrained_model, device="cpu",
                text_encoder=text_encoder,
            )
        else:
            kimodo_pretrained = load_kimodo_model(self.args.pretrained_model, device="cpu")
            text_encoder = kimodo_pretrained.text_encoder

        denoiser = kimodo_pretrained.denoiser.model

        scene_encoder_config = {
            "voxel_size": tuple(map(int, self.args.voxel_size.split(","))),
            "patch_size": tuple(map(int, self.args.patch_size.split(","))),
            "in_channels": 1,
            "d_model": 256,
            "num_heads": 4,
            "num_layers": 4,
            "ff_dim": 512,
            "sceneco_dropout": 0.1,
            "use_dual_vit": False,
            "root_voxel_mode": "full",
        }

        self.model = KimodoSceneCo(
            denoiser=denoiser,
            text_encoder=text_encoder,
            num_base_steps=self.args.num_base_steps,
            scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_encoder_config,
            device=self.device,
            cfg_type="nocfg",
            use_in_root_model=False,
            use_in_body_model=False,
            use_cakey_root=self.args.use_cakey_root,
            use_cakey_body=self.args.use_cakey_body,
            cakey_hidden_dim=self.args.cakey_hidden_dim,
        )

        self.model.freeze_for_cakey()

        del kimodo_pretrained
        gc.collect()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info(f"Stage 1 - Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")

        self.loss_fn = CaKeyDiffusionLoss(self.model.diffusion)

        self.motion_dim = self.model.denoiser.model.motion_rep.motion_rep_dim

    def _build_dataset(self):
        import copy

        cpu_motion_rep = copy.deepcopy(self.model.motion_rep)
        cpu_motion_rep.skeleton = cpu_motion_rep.skeleton.to("cpu")

        ds_kwargs = dict(
            data_root=self.args.data_root,
            motion_rep=cpu_motion_rep,
            max_frames=self.args.max_frames,
            min_frames=self.args.min_frames,
            voxel_size=tuple(map(int, self.args.voxel_size.split(","))),
            train_ratio=self.args.train_ratio,
            seed=self.args.seed,
            soma_data_root=getattr(self.args, 'soma_data_root', None),
            cache_dir=getattr(self.args, 'cache_dir', None),
        )

        self.train_dataset = LINGOSceneMotionDataset(
            **ds_kwargs,
            split="train",
            scene_dropout=0.0,
        )
        self.val_dataset = LINGOSceneMotionDataset(
            **ds_kwargs,
            split="val",
            scene_dropout=0.0,
        )

        mp_ctx = "spawn" if self.args.num_workers > 0 else None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            multiprocessing_context=mp_ctx,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
            multiprocessing_context=mp_ctx,
        )

    def _build_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.99),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.total_steps,
            eta_min=self.args.lr * 0.01,
        )

    def _build_t_embed_for_cakey(self, t: torch.Tensor, batch_size: int, d_model: int) -> torch.Tensor:
        """Get timestep embedding for CaKey input.

        Reuses Kimodo's timestep embedder from the root model.
        """
        block = self.model.denoiser.model.root_model
        if hasattr(block, 'embed_timestep') and hasattr(block, 'sequence_pos_encoder'):
            return block.embed_timestep(t)
        else:
            return torch.zeros(batch_size, d_model, device=self.device)

    def _prepare_batch(self, batch: Dict) -> Dict:
        motion = batch["motion_features"].to(self.device)
        mask = batch["motion_mask"].to(self.device)
        texts = batch["texts"]
        lengths = batch["lengths"]

        observed_motion, motion_mask_elem, keyframe_mask = sample_keyframes_for_training(
            motion, lengths,
            stride_choices=self.args.kf_stride_choices,
            always_first_frame=self.args.kf_always_first_frame,
            always_last_frame=self.args.kf_always_last_frame,
            random_drop=self.args.kf_random_drop,
        )

        if "text_feat" in batch:
            text_feat = batch["text_feat"].to(self.device)
            B = text_feat.shape[0]
            text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=self.device)
        else:
            text_feat, text_length = self.model.text_encoder(texts)
            text_feat = text_feat.to(self.device)
            B, maxlen = text_feat.shape[:2]
            text_length_tensor = torch.tensor(text_length, device=self.device)
            text_pad_mask = torch.arange(maxlen, device=self.device).expand(B, maxlen) < text_length_tensor[:, None]

        t_embed = self._build_t_embed_for_cakey(
            torch.zeros(B, dtype=torch.long, device=self.device),
            B, 1024,
        )

        cakey_kwargs_root = {
            "observed_motion": observed_motion,
            "motion_mask": motion_mask_elem,
            "keyframe_mask": keyframe_mask,
            "t_embed": t_embed,
        }
        cakey_kwargs_body = {
            "observed_motion": observed_motion,
            "motion_mask": motion_mask_elem,
            "keyframe_mask": keyframe_mask,
            "t_embed": t_embed,
        }

        first_heading_angle = torch.zeros(B, device=self.device)

        return {
            "x_start": motion,
            "x_pad_mask": mask,
            "text_feat": text_feat,
            "text_pad_mask": text_pad_mask,
            "first_heading_angle": first_heading_angle,
            "motion_mask": motion_mask_elem,
            "observed_motion": observed_motion,
            "cakey_kwargs_root": cakey_kwargs_root,
            "cakey_kwargs_body": cakey_kwargs_body,
            "cfg_type": "nocfg",
            "lengths": lengths,
        }

    @property
    def training(self):
        return self.model.training

    def train_step(self, batch: Dict, accum_count: int = 1) -> Dict[str, float]:
        self.model.train()
        kwargs = self._prepare_batch(batch)

        B, T, D = kwargs["x_start"].shape
        t = torch.randint(0, self.args.num_base_steps, (B,), device=self.device)

        losses = self.loss_fn.training_losses(
            self.model.denoiser,
            kwargs["x_start"],
            t,
            kwargs,
        )

        loss = losses["loss"] / accum_count
        loss.backward()

        return {
            "loss": losses["loss"].item(),
            "mse": losses["mse"].item(),
        }

    def _optimizer_step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.args.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm.item()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_mse = 0
        n_batches = 0

        for batch in self.val_loader:
            kwargs = self._prepare_batch(batch)
            B, T, D = kwargs["x_start"].shape
            t = torch.randint(0, self.args.num_base_steps, (B,), device=self.device)

            losses = self.loss_fn.training_losses(
                self.model.denoiser,
                kwargs["x_start"],
                t,
                kwargs,
            )
            total_loss += losses["loss"].item()
            total_mse += losses["mse"].item()
            n_batches += 1

            if n_batches >= self.args.val_max_batches:
                break

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_mse": total_mse / max(n_batches, 1),
        }

    def save_checkpoint(self, epoch: int):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "args": vars(self.args),
        }
        best_path = self.ckpt_dir / "best_checkpoint.pt"
        torch.save(state, str(best_path))
        log.info(f"Saved best checkpoint (epoch {epoch+1}): {best_path}")

    def _log_first_batch(self, batch):
        log.info("=" * 60)
        log.info("STAGE 1 FIRST BATCH DIAGNOSTIC")
        log.info(f"  motion: shape={batch['motion_features'].shape}")
        log.info(f"  motion_mask: shape={batch['motion_mask'].shape}")
        log.info(f"  texts (first 3): {batch['texts'][:3]}")
        cakey_p = [n for n, p in self.model.named_parameters() if 'cakey' in n and p.requires_grad]
        log.info(f"  cakey params (trainable): {len(cakey_p)}")
        log.info("=" * 60)

    def train(self):
        log.info("=" * 50)
        log.info("Stage 1: Training CaKey Layers")
        log.info(f"  CaKey: root={self.args.use_cakey_root}, body={self.args.use_cakey_body}")
        log.info(f"  Train dataset: {len(self.train_dataset)} samples")
        log.info(f"  Val dataset: {len(self.val_dataset)} samples")
        log.info(f"  Total steps: {self.args.total_steps}")
        log.info(f"  Batch size: {self.args.batch_size}")
        log.info(f"  Accum. steps: {self.args.accum_steps}")
        log.info(f"  LR: {self.args.lr}")
        log.info(f"  Keyframe stride choices: {self.args.kf_stride_choices}")
        log.info(f"  Keyframe random drop: {self.args.kf_random_drop}")
        log.info("=" * 50)

        first_batch_logged = False
        accum_steps = self.args.accum_steps
        self.optimizer.zero_grad()
        last_grad_norm = 0.0
        last_lr = 0.0
        accum_count = 0

        train_iter = iter(self.train_loader)
        self.model.train()

        while self.global_step < self.args.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            if not first_batch_logged:
                self._log_first_batch(batch)
                first_batch_logged = True

            metrics = self.train_step(batch, accum_steps)
            self.global_step += 1
            accum_count += 1

            if accum_count >= accum_steps:
                last_grad_norm = self._optimizer_step()
                last_lr = self.scheduler.get_last_lr()[0]
                accum_count = 0

            if self.global_step % self.args.log_interval == 0:
                log.info(
                    f"[Step {self.global_step}/{self.args.total_steps}] "
                    f"loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}, "
                    f"grad_norm={last_grad_norm:.4f}, lr={last_lr:.2e}"
                )
                if self.writer:
                    self.writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                    self.writer.add_scalar("train/mse", metrics["mse"], self.global_step)
                    self.writer.add_scalar("train/grad_norm", last_grad_norm, self.global_step)
                    self.writer.add_scalar("train/lr", last_lr, self.global_step)

            if self.global_step % self.args.val_interval == 0:
                log.info("Running validation...")
                val_metrics = self.validate()
                for k, v in val_metrics.items():
                    if self.writer:
                        self.writer.add_scalar(f"val/{k}", v, self.global_step)
                log.info(f"  Val loss: {val_metrics['val_loss']:.4f}, Val MSE: {val_metrics['val_mse']:.4f}")

                is_best = val_metrics["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.best_epoch = self.global_step
                    self.save_checkpoint(self.global_step)
                self.model.train()

        if accum_count > 0:
            self._optimizer_step()

        if self.writer:
            self.writer.close()
        log.info(f"Stage 1 complete! Best val_loss={self.best_val_loss:.6f} at step {self.best_epoch}")
        log.info(f"Final checkpoint saved at step {self.global_step}")


class SceneCoStage2Trainer:
    """Stage 2 trainer: train SceneCo + VoxelViT with frozen CaKey."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        os.makedirs(args.output_dir, exist_ok=True)
        self.ckpt_dir = Path(args.output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "logs")) if SummaryWriter else None

        self._build_model_stage2()
        self._build_dataset()
        self._build_optimizer()

    def _build_model_stage2(self):
        from kimodo_sceneco.model import KimodoSceneCo
        from kimodo.model import load_model as load_kimodo_model

        cache_dir = getattr(self.args, 'cache_dir', None)
        has_cached_text = cache_dir and Path(cache_dir).exists()

        class _DummyTextEncoder:
            def __call__(self, text):
                raise RuntimeError("Text encoder should not be called in cached mode")
            def to(self, device):
                return self
            def eval(self):
                return self

        log.info("Loading pretrained Kimodo model...")
        if has_cached_text:
            text_encoder = _DummyTextEncoder()
            kimodo_pretrained = load_kimodo_model(
                self.args.pretrained_model, device="cpu",
                text_encoder=text_encoder,
            )
        else:
            kimodo_pretrained = load_kimodo_model(self.args.pretrained_model, device="cpu")
            text_encoder = kimodo_pretrained.text_encoder

        denoiser = kimodo_pretrained.denoiser.model

        scene_encoder_config = {
            "voxel_size": tuple(map(int, self.args.voxel_size.split(","))),
            "patch_size": tuple(map(int, self.args.patch_size.split(","))),
            "in_channels": 1,
            "d_model": self.args.scene_dim,
            "num_heads": self.args.scene_num_heads,
            "num_layers": self.args.scene_num_layers,
            "ff_dim": self.args.scene_ff_dim,
            "sceneco_dropout": self.args.sceneco_dropout,
            "use_dual_vit": self.args.use_dual_vit,
            "root_voxel_mode": self.args.root_voxel_mode,
        }

        self.model = KimodoSceneCo(
            denoiser=denoiser,
            text_encoder=text_encoder,
            num_base_steps=self.args.num_base_steps,
            scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_encoder_config,
            device=self.device,
            cfg_type="scene_separated",
            use_in_root_model=self.args.use_in_root_model,
            use_in_body_model=self.args.use_in_body_model,
            use_cakey_root=self.args.use_cakey_root,
            use_cakey_body=self.args.use_cakey_body,
            cakey_hidden_dim=self.args.cakey_hidden_dim,
        )

        stage1_path = getattr(self.args, 'stage1_checkpoint', None)
        if stage1_path and Path(stage1_path).exists():
            log.info(f"Loading Stage 1 CaKey checkpoint from: {stage1_path}")
            ckpt = torch.load(stage1_path, map_location="cpu")
            model_state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            log.info(f"Stage 1 checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                log.info(f"  First 5 missing: {missing[:5]}")
        else:
            log.warning("No Stage 1 checkpoint provided. CaKey layers will be random.")

        self.model.freeze_for_sceneco(freeze_cakey=getattr(self.args, 'freeze_cakey', True))

        del kimodo_pretrained
        gc.collect()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info(f"Stage 2 - Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")

        self.motion_dim = self.model.denoiser.model.motion_rep.motion_rep_dim

    def _build_dataset(self):
        import copy

        cache_dir = getattr(self.args, 'cache_dir', None)

        if cache_dir and Path(cache_dir).exists():
            ds_kwargs = dict(
                data_root=self.args.data_root,
                max_frames=self.args.max_frames,
                min_frames=self.args.min_frames,
                voxel_size=tuple(map(int, self.args.voxel_size.split(","))),
                train_ratio=self.args.train_ratio,
                seed=self.args.seed,
                soma_data_root=getattr(self.args, 'soma_data_root', None),
                cache_dir=cache_dir,
            )
        else:
            cpu_motion_rep = copy.deepcopy(self.model.motion_rep)
            cpu_motion_rep.skeleton = cpu_motion_rep.skeleton.to("cpu")

            ds_kwargs = dict(
                data_root=self.args.data_root,
                motion_rep=cpu_motion_rep,
                max_frames=self.args.max_frames,
                min_frames=self.args.min_frames,
                voxel_size=tuple(map(int, self.args.voxel_size.split(","))),
                train_ratio=self.args.train_ratio,
                seed=self.args.seed,
                soma_data_root=getattr(self.args, 'soma_data_root', None),
            )

        self.train_dataset = LINGOSceneMotionDataset(
            **ds_kwargs,
            split="train",
            scene_dropout=self.args.scene_dropout,
        )
        self.val_dataset = LINGOSceneMotionDataset(
            **ds_kwargs,
            split="val",
            scene_dropout=0.0,
        )

        mp_ctx = None
        if self.args.num_workers > 0:
            mp_ctx = "fork" if (cache_dir and Path(cache_dir).exists()) else "spawn"

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            multiprocessing_context=mp_ctx,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
            multiprocessing_context=mp_ctx,
        )

    def _build_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.99),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.total_steps,
            eta_min=self.args.lr * 0.01,
        )

    def _prepare_batch(self, batch: Dict) -> Dict:
        motion = batch["motion_features"].to(self.device)
        mask = batch["motion_mask"].to(self.device)
        voxel = batch["voxel_grid"].to(self.device)
        texts = batch["texts"]
        lengths = batch["lengths"]

        (scene_feat_root, scene_mask_root), (scene_feat_body, scene_mask_body) = self.model.encode_scene(voxel)

        if hasattr(self.args, 'scene_dropout') and self.args.scene_dropout > 0 and self.training:
            drop_mask = torch.rand(scene_feat_root.shape[0]) < self.args.scene_dropout
            scene_feat_root[drop_mask] = 0
            scene_mask_root[drop_mask] = False
            scene_feat_body[drop_mask] = 0
            scene_mask_body[drop_mask] = False

        if "text_feat" in batch:
            text_feat = batch["text_feat"].to(self.device)
            B = text_feat.shape[0]
            text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=self.device)
        else:
            text_feat, text_length = self.model.text_encoder(texts)
            text_feat = text_feat.to(self.device)
            B, maxlen = text_feat.shape[:2]
            text_length_tensor = torch.tensor(text_length, device=self.device)
            text_pad_mask = torch.arange(maxlen, device=self.device).expand(B, maxlen) < text_length_tensor[:, None]

        first_heading_angle = torch.zeros(B, device=self.device)

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
            "cfg_weight": [2.0, 2.0, 2.0],
            "lengths": lengths,
        }

    @property
    def training(self):
        return self.model.training

    def train_step(self, batch: Dict, accum_count: int = 1) -> Dict[str, float]:
        self.model.train()
        kwargs = self._prepare_batch(batch)

        B, T, D = kwargs["x_start"].shape
        t = torch.randint(0, self.args.num_base_steps, (B,), device=self.device)

        noise = torch.randn_like(kwargs["x_start"])
        x_t = self.model.diffusion.q_sample(kwargs["x_start"], t, noise=noise)

        cfg_weight = [2.0, 2.0, 2.0]

        pred_x0 = self.model.denoiser(
            cfg_weight,
            x_t,
            kwargs["x_pad_mask"],
            kwargs["text_feat"],
            kwargs["text_pad_mask"],
            t,
            first_heading_angle=kwargs.get("first_heading_angle"),
            motion_mask=kwargs.get("motion_mask"),
            observed_motion=kwargs.get("observed_motion"),
            scene_feat_root=kwargs.get("scene_feat_root"),
            scene_mask_root=kwargs.get("scene_mask_root"),
            scene_feat_body=kwargs.get("scene_feat_body"),
            scene_mask_body=kwargs.get("scene_mask_body"),
            cfg_type=kwargs.get("cfg_type", "nocfg"),
        )

        mask = kwargs["x_pad_mask"].unsqueeze(-1).float()
        mse = F.mse_loss(pred_x0 * mask, kwargs["x_start"] * mask, reduction="none")
        mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        loss = mse

        prior_loss_val = torch.tensor(0.0, device=self.device)
        if getattr(self.args, 'prior_weight', 0) > 0 and kwargs.get("scene_feat_root") is not None:
            pred_x0_null = self.model.denoiser(
                cfg_weight,
                x_t,
                kwargs["x_pad_mask"],
                kwargs["text_feat"],
                kwargs["text_pad_mask"],
                t,
                first_heading_angle=kwargs.get("first_heading_angle"),
                motion_mask=kwargs.get("motion_mask"),
                observed_motion=kwargs.get("observed_motion"),
                scene_feat_root=None,
                scene_mask_root=None,
                scene_feat_body=None,
                scene_mask_body=None,
                cfg_type="nocfg",
            )
            prior_mse = F.mse_loss(pred_x0_null * mask, kwargs["x_start"] * mask, reduction="none")
            prior_mse = (prior_mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()
            prior_loss_val = prior_mse
            loss = loss + getattr(self.args, 'prior_weight', 0.5) * prior_loss_val

        loss = loss / accum_count
        loss.backward()

        return {
            "loss": loss.item() * accum_count,
            "mse": mse.item(),
            "prior_loss": prior_loss_val.item(),
        }

    def _optimizer_step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.args.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm.item()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_mse = 0
        n_batches = 0

        for batch in self.val_loader:
            kwargs = self._prepare_batch(batch)
            B, T, D = kwargs["x_start"].shape
            t = torch.randint(0, self.args.num_base_steps, (B,), device=self.device)

            noise = torch.randn_like(kwargs["x_start"])
            x_t = self.model.diffusion.q_sample(kwargs["x_start"], t, noise=noise)

            pred_x0 = self.model.denoiser(
                [2.0, 2.0, 2.0],
                x_t,
                kwargs["x_pad_mask"],
                kwargs["text_feat"],
                kwargs["text_pad_mask"],
                t,
                first_heading_angle=kwargs.get("first_heading_angle"),
                scene_feat_root=kwargs.get("scene_feat_root"),
                scene_mask_root=kwargs.get("scene_mask_root"),
                scene_feat_body=kwargs.get("scene_feat_body"),
                scene_mask_body=kwargs.get("scene_mask_body"),
                cfg_type="nocfg",
            )

            mask = kwargs["x_pad_mask"].unsqueeze(-1).float()
            mse = F.mse_loss(pred_x0 * mask, kwargs["x_start"] * mask, reduction="none")
            mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

            total_loss += mse.item()
            total_mse += mse.item()
            n_batches += 1

            if n_batches >= getattr(self.args, 'val_max_batches', 10):
                break

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_mse": total_mse / max(n_batches, 1),
        }

    def save_checkpoint(self, step: int):
        state = {
            "step": step,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "args": vars(self.args),
        }
        best_path = self.ckpt_dir / "best_checkpoint.pt"
        torch.save(state, str(best_path))
        log.info(f"Saved best checkpoint (step {step}): {best_path}")

    def train(self):
        log.info("=" * 50)
        log.info("Stage 2: Training SceneCo (CaKey frozen)")
        log.info(f"  SceneCo: root={self.args.use_in_root_model}, body={self.args.use_in_body_model}")
        log.info(f"  CaKey: root={self.args.use_cakey_root}, body={self.args.use_cakey_body}")
        log.info(f"  Dual ViT: {self.args.use_dual_vit}")
        log.info(f"  Train dataset: {len(self.train_dataset)} samples")
        log.info(f"  Total steps: {self.args.total_steps}")
        log.info(f"  Batch size: {self.args.batch_size}")
        log.info(f"  LR: {self.args.lr}, Prior weight: {getattr(self.args, 'prior_weight', 0.5)}")
        log.info("=" * 50)

        first_batch_logged = False
        accum_steps = self.args.accum_steps
        self.optimizer.zero_grad()
        last_grad_norm = 0.0
        last_lr = 0.0
        accum_count = 0

        train_iter = iter(self.train_loader)
        self.model.train()

        while self.global_step < self.args.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            if not first_batch_logged:
                log.info(f"Stage 2 first batch: motion={batch['motion_features'].shape}, voxel={batch['voxel_grid'].shape}")
                first_batch_logged = True

            metrics = self.train_step(batch, accum_steps)
            self.global_step += 1
            accum_count += 1

            if accum_count >= accum_steps:
                last_grad_norm = self._optimizer_step()
                last_lr = self.scheduler.get_last_lr()[0]
                accum_count = 0

            if self.global_step % self.args.log_interval == 0:
                log.info(
                    f"[Step {self.global_step}/{self.args.total_steps}] "
                    f"loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}, "
                    f"prior_loss={metrics['prior_loss']:.4f}, "
                    f"grad_norm={last_grad_norm:.4f}, lr={last_lr:.2e}"
                )
                if self.writer:
                    self.writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                    self.writer.add_scalar("train/mse", metrics["mse"], self.global_step)
                    self.writer.add_scalar("train/grad_norm", last_grad_norm, self.global_step)
                    self.writer.add_scalar("train/lr", last_lr, self.global_step)

            if self.global_step % self.args.val_interval == 0:
                log.info("Running validation...")
                val_metrics = self.validate()
                for k, v in val_metrics.items():
                    if self.writer:
                        self.writer.add_scalar(f"val/{k}", v, self.global_step)
                log.info(f"  Val loss: {val_metrics['val_loss']:.4f}, Val MSE: {val_metrics['val_mse']:.4f}")

                is_best = val_metrics["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.best_epoch = self.global_step
                    self.save_checkpoint(self.global_step)
                self.model.train()

        if accum_count > 0:
            self._optimizer_step()

        if self.writer:
            self.writer.close()
        log.info(f"Stage 2 complete! Best val_loss={self.best_val_loss:.6f} at step {self.best_epoch}")
        log.info(f"Final checkpoint saved at step {self.global_step}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CaKey + SceneCo")

    parser.add_argument("--data_root", type=str, default="/home/lzsh2025/kimodo-viser/LINGO/dataset")
    parser.add_argument("--soma_data_root", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default="Kimodo-SOMA-RP-v1.1")

    parser.add_argument("--voxel_size", type=str, default="64,64,64")
    parser.add_argument("--patch_size", type=str, default="8,8,8")
    parser.add_argument("--scene_dim", type=int, default=256)
    parser.add_argument("--scene_num_heads", type=int, default=4)
    parser.add_argument("--scene_num_layers", type=int, default=4)
    parser.add_argument("--scene_ff_dim", type=int, default=512)
    parser.add_argument("--num_base_steps", type=int, default=1000)

    parser.add_argument("--use_in_root_model", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--use_in_body_model", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--sceneco_dropout", type=float, default=0.1)
    parser.add_argument("--use_dual_vit", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--root_voxel_mode", type=str, default="full")

    parser.add_argument("--use_cakey_root", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--use_cakey_body", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--cakey_hidden_dim", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=0.5)
    parser.add_argument("--scene_dropout", type=float, default=0.1)

    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--val_max_batches", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--output_dir", type=str, default="./sceneco_output")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--accum_steps", type=int, default=1)

    parser.add_argument("--stage1_cakey", action="store_true", default=False)
    parser.add_argument("--stage2_sceneco", action="store_true", default=False)
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_cakey", type=lambda x: x.lower() == 'true', default=True)

    parser.add_argument("--kf_always_first_frame", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--kf_always_last_frame", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--kf_stride_choices", type=str, default="20,30,40")
    parser.add_argument("--kf_random_drop", type=float, default=0.1)

    args = parser.parse_args()
    args.kf_stride_choices = [int(x) for x in args.kf_stride_choices.split(",")]
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(Path(args.output_dir) / "train.log")),
        ],
    )

    if args.stage1_cakey:
        trainer = CaKeyTrainer(args)
    elif args.stage2_sceneco:
        trainer = SceneCoStage2Trainer(args)
    else:
        raise ValueError("Must specify --stage1_cakey or --stage2_sceneco")

    trainer.train()


if __name__ == "__main__":
    main()
