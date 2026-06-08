import argparse
import gc
import json
import logging
import os
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

log = logging.getLogger(__name__)


class SceneCoDiffusionLoss(nn.Module):

    def __init__(self, diffusion, prior_weight: float = 0.5):
        super().__init__()
        self.diffusion = diffusion
        self.prior_weight = prior_weight

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        if torch.isnan(x_t).any():
            log.error(f"NaN in x_t after q_sample! x_start range=[{x_start.min():.4f}, {x_start.max():.4f}]")

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
            scene_feat=model_kwargs.get("scene_feat"),
            scene_mask=model_kwargs.get("scene_mask"),
            cfg_type=model_kwargs.get("cfg_type", "nocfg"),
        )

        if torch.isnan(pred_x0).any():
            nan_frac = torch.isnan(pred_x0).float().mean().item()
            log.error(f"NaN in pred_x0! {nan_frac*100:.1f}% NaN. x_t range=[{x_t.min():.4f},{x_t.max():.4f}], scene_feat range=[{model_kwargs['scene_feat'].min():.4f},{model_kwargs['scene_feat'].max():.4f}]")

        mask = model_kwargs["x_pad_mask"].unsqueeze(-1).float()
        mse = F.mse_loss(pred_x0 * mask, x_start * mask, reduction="none")
        mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        loss = mse

        prior_loss = torch.tensor(0.0, device=x_start.device)
        if self.prior_weight > 0 and model_kwargs.get("scene_feat") is not None:
            pred_x0_null = model(
                cfg_weight,
                x_t,
                model_kwargs["x_pad_mask"],
                model_kwargs["text_feat"],
                model_kwargs["text_pad_mask"],
                t,
                first_heading_angle=model_kwargs.get("first_heading_angle"),
                motion_mask=model_kwargs.get("motion_mask"),
                observed_motion=model_kwargs.get("observed_motion"),
                scene_feat=None,
                scene_mask=None,
                cfg_type="nocfg",
            )

            prior_mse = F.mse_loss(pred_x0_null * mask, x_start * mask, reduction="none")
            prior_mse = (prior_mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()
            prior_loss = prior_mse
            loss = loss + self.prior_weight * prior_loss

        return {
            "loss": loss,
            "mse": mse.detach(),
            "prior_loss": prior_loss.detach(),
        }


class Trainer:

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

        self._build_model()
        self._build_dataset()
        self._build_optimizer()

    def _build_model(self):
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
            log.info("Cache mode: skipping text_encoder load (using pre-encoded text features)")
            kimodo_pretrained = load_kimodo_model(
                self.args.pretrained_model, device="cpu",
                text_encoder=_DummyTextEncoder(),
            )
        else:
            kimodo_pretrained = load_kimodo_model(self.args.pretrained_model, device="cpu")

        if has_cached_text:
            text_encoder = None
        else:
            text_encoder = kimodo_pretrained.text_encoder

        scene_encoder_config = {
            "voxel_size": tuple(map(int, self.args.voxel_size.split(","))),
            "patch_size": tuple(map(int, self.args.patch_size.split(","))),
            "in_channels": 1,
            "d_model": self.args.scene_dim,
            "num_heads": self.args.scene_num_heads,
            "num_layers": self.args.scene_num_layers,
            "ff_dim": self.args.scene_ff_dim,
        }

        if self.args.exp_type == "exp1":
            from kimodo_sceneco.exp.exp1_monkey_patch import KimodoSceneCoExp1

            denoiser = kimodo_pretrained.denoiser.model

            self.model = KimodoSceneCoExp1(
                denoiser=denoiser,
                text_encoder=text_encoder,
                num_base_steps=self.args.num_base_steps,
                scene_encoder_type="voxel_vit",
                scene_encoder_config=scene_encoder_config,
                device=self.device,
                cfg_type="scene_separated",
            )

        elif self.args.exp_type in ("exp2", "exp3", "exp4"):
            from kimodo_sceneco.exp.exp2_rewrite_layer import KimodoSceneCoExp2
            from kimodo_sceneco.exp.exp2_rewrite_layer.backbone_exp2 import TransformerEncoderBlock
            from kimodo_sceneco.exp.exp2_rewrite_layer.twostage_denoiser_exp2 import TwostageDenoiser as TwostageDenoiserExp2

            pretrained_denoiser = kimodo_pretrained.denoiser.model
            pretrained_root = pretrained_denoiser.root_model
            pretrained_body = pretrained_denoiser.body_model
            motion_rep = pretrained_denoiser.motion_rep
            motion_mask_mode = pretrained_denoiser.motion_mask_mode

            root_use = self.args.exp_type != "exp4"
            body_use = self.args.exp_type != "exp3"

            def _extract_block_config(block, use_sceneco):
                return dict(
                    latent_dim=block.latent_dim,
                    ff_size=block.ff_size,
                    num_layers=block.num_layers,
                    num_heads=block.num_heads,
                    activation=block.activation,
                    dropout=block.dropout,
                    pe_dropout=block.pe_dropout,
                    norm_first=getattr(block, 'norm_first', False),
                    llm_shape=[1, block.embed_text.in_features],
                    use_text_mask=block.use_text_mask,
                    num_text_tokens_override=getattr(block, 'num_text_tokens_override', None),
                    input_first_heading_angle=block.input_first_heading_angle,
                    scene_feat_dim=self.args.scene_dim,
                    use_sceneco=use_sceneco,
                    sceneco_dropout=self.args.sceneco_dropout,
                )

            root_config = _extract_block_config(pretrained_root, root_use)
            root_config["input_dim"] = pretrained_root.input_linear.in_features
            root_config["output_dim"] = pretrained_root.output_linear.out_features
            root_config["skeleton"] = motion_rep.skeleton

            body_config = _extract_block_config(pretrained_body, body_use)
            body_config["input_dim"] = pretrained_body.input_linear.in_features
            body_config["output_dim"] = pretrained_body.output_linear.out_features
            body_config["skeleton"] = motion_rep.skeleton

            new_root = TransformerEncoderBlock(**root_config)
            new_body = TransformerEncoderBlock(**body_config)

            new_denoiser = TwostageDenoiserExp2(
                motion_rep=motion_rep,
                motion_mask_mode=motion_mask_mode,
            )
            new_denoiser.root_model = new_root
            new_denoiser.body_model = new_body

            self.model = KimodoSceneCoExp2(
                denoiser=new_denoiser,
                text_encoder=text_encoder,
                num_base_steps=self.args.num_base_steps,
                scene_encoder_type="voxel_vit",
                scene_encoder_config=scene_encoder_config,
                device=self.device,
                cfg_type="scene_separated",
            )
            self.model._load_and_migrate_pretrained(new_denoiser, pretrained_denoiser)

        else:
            raise ValueError(f"Unknown exp_type: {self.args.exp_type}")

        if self.args.freeze_pretrained:
            self.model.freeze_pretrained()

        del kimodo_pretrained
        gc.collect()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")

        nan_params = [n for n, p in self.model.named_parameters() if torch.isnan(p).any()]
        if nan_params:
            log.error(f"[DIAG] NaN in model PARAMETERS! {nan_params[:5]}")
        else:
            nan_params_t = [n for n, p in self.model.named_parameters() if torch.isnan(p).any()]
            log.info(f"[DIAG] All {total:,} model parameters checked: no NaN")

        self.loss_fn = SceneCoDiffusionLoss(
            self.model.diffusion,
            prior_weight=self.args.prior_weight,
        )

        self.motion_dim = self.model.denoiser.model.motion_rep.motion_rep_dim
        log.info(f"Motion feature dim: {self.motion_dim}")

        self.baseline_model = None
        if self.args.baseline_model:
            log.info("Loading baseline Kimodo for validation...")
            self.baseline_model = load_kimodo_model(
                self.args.baseline_model, device=self.device
            )
            self.baseline_model.eval()

    def _build_dataset(self):
        cache_dir = getattr(self.args, 'cache_dir', None)

        if cache_dir and Path(cache_dir).exists():
            ds_kwargs = dict(
                data_root=self.args.data_root,
                max_frames=self.args.max_frames,
                min_frames=self.args.min_frames,
                voxel_size=tuple(map(int, self.args.voxel_size.split(","))),
                train_ratio=self.args.train_ratio,
                seed=self.args.seed,
                soma_data_root=self.args.soma_data_root,
                cache_dir=cache_dir,
            )
        else:
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
                soma_data_root=self.args.soma_data_root,
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
            if cache_dir and Path(cache_dir).exists():
                mp_ctx = "fork"
            else:
                mp_ctx = "spawn"

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
            T_max=self.args.num_epochs * len(self.train_loader),
            eta_min=self.args.lr * 0.01,
        )

    def _prepare_batch(self, batch: Dict) -> Dict:
        motion = batch["motion_features"].to(self.device)
        mask = batch["motion_mask"].to(self.device)
        voxel = batch["voxel_grid"].to(self.device)
        texts = batch["texts"]
        lengths = batch["lengths"]

        if torch.isnan(motion).any():
            log.error(f"[DIAG] NaN in motion_features INPUT! {torch.isnan(motion).float().mean()*100:.1f}%")
        if torch.isnan(voxel).any():
            log.error(f"[DIAG] NaN in voxel INPUT!")
        if "text_feat" in batch and torch.isnan(batch["text_feat"]).any():
            log.error(f"[DIAG] NaN in text_feat INPUT! {torch.isnan(batch['text_feat']).float().mean()*100:.1f}%")

        scene_feat, scene_mask = self.model.encode_scene(voxel)

        if torch.isnan(scene_feat).any():
            log.error(f"NaN in scene_feat after encode_scene! voxel range=[{voxel.min():.4f}, {voxel.max():.4f}]")

        if self.args.scene_dropout > 0 and self.training:
            drop_mask = torch.rand(scene_feat.shape[0]) < self.args.scene_dropout
            scene_feat[drop_mask] = 0
            scene_mask[drop_mask] = False

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

        bs = B
        first_heading_angle = torch.zeros(bs, device=self.device)

        return {
            "x_start": motion,
            "x_pad_mask": mask,
            "scene_feat": scene_feat,
            "scene_mask": scene_mask,
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
            "prior_loss": losses["prior_loss"].item(),
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

        avg_loss = total_loss / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)

        metrics = {"val_loss": avg_loss, "val_mse": avg_mse}

        if self.baseline_model is not None:
            baseline_mse = self._validate_baseline()
            metrics["baseline_mse"] = baseline_mse
            if avg_mse > 0:
                metrics["degradation_ratio"] = avg_mse / max(baseline_mse, 1e-8)

        return metrics

    @torch.no_grad()
    def _validate_baseline(self) -> float:
        self.baseline_model.eval()
        total_mse = 0
        n_batches = 0

        for batch in self.val_loader:
            motion = batch["motion_features"].to(self.device)
            mask = batch["motion_mask"].to(self.device)
            texts = batch["texts"]
            B, T, D = motion.shape

            text_feat, text_length = self.baseline_model.text_encoder(texts)
            text_feat = text_feat.to(self.device)
            maxlen = text_feat.shape[1]
            text_length_tensor = torch.tensor(text_length, device=self.device)
            text_pad_mask = torch.arange(maxlen, device=self.device).expand(B, maxlen) < text_length_tensor[:, None]

            t = torch.randint(0, self.args.num_base_steps, (B,), device=self.device)
            noise = torch.randn_like(motion)
            x_t = self.baseline_model.diffusion.q_sample(motion, t, noise=noise)

            pred_x0 = self.baseline_model.denoiser(
                [2.0, 2.0],
                x_t,
                mask,
                text_feat,
                text_pad_mask,
                t,
                cfg_type="nocfg",
            )

            mask_f = mask.unsqueeze(-1).float()
            mse = F.mse_loss(pred_x0 * mask_f, motion * mask_f, reduction="none")
            mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()
            total_mse += mse.item()
            n_batches += 1

            if n_batches >= self.args.val_max_batches:
                break

        return total_mse / max(n_batches, 1)

    def _get_alpha_stats(self):
        alphas = []
        if self.args.exp_type == "exp1":
            for block in [self.model.denoiser.model.root_model, self.model.denoiser.model.body_model]:
                if hasattr(block, 'sceneco_layers'):
                    for layer in block.sceneco_layers:
                        alphas.append(layer.alpha.item())
        else:
            for block in [self.model.denoiser.model.root_model, self.model.denoiser.model.body_model]:
                if hasattr(block, 'seqTransEncoder'):
                    for layer in block.seqTransEncoder.layers:
                        if hasattr(layer, 'sceneco') and hasattr(layer.sceneco, 'alpha'):
                            alphas.append(layer.sceneco.alpha.item())
        if not alphas:
            return "N/A"
        return f"mean={sum(alphas)/len(alphas):.5f} min={min(alphas):.5f} max={max(alphas):.5f}"

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
        log.info("FIRST BATCH DIAGNOSTIC")
        log.info(f"  exp_type: {self.args.exp_type}")
        log.info(f"  motion: shape={batch['motion_features'].shape}, range=[{batch['motion_features'].min():.2f}, {batch['motion_features'].max():.2f}]")
        log.info(f"  motion_mask: shape={batch['motion_mask'].shape}, valid_frames={batch['motion_mask'].sum(dim=1).tolist()[:4]}")
        log.info(f"  voxel: shape={batch['voxel_grid'].shape}, range=[{batch['voxel_grid'].min():.4f}, {batch['voxel_grid'].max():.4f}], nonzero={(batch['voxel_grid']>0).float().mean()*100:.1f}%")
        if 'text_feat' in batch:
            log.info(f"  text_feat: shape={batch['text_feat'].shape}, range=[{batch['text_feat'].min():.2f}, {batch['text_feat'].max():.2f}]")
        log.info(f"  texts (first 3): {batch['texts'][:3]}")
        log.info(f"  scene_names (first 3): {batch['scene_names'][:3]}")

        voxel = batch['voxel_grid'].to(self.device)
        with torch.no_grad():
            scene_feat, scene_mask = self.model.encode_scene(voxel)
        log.info(f"  scene_feat (encoded): shape={scene_feat.shape}, range=[{scene_feat.min():.4f}, {scene_feat.max():.4f}]")
        log.info(f"  scene_mask: shape={scene_mask.shape}, valid_patches={scene_mask.float().mean()*100:.1f}%")

        if self.args.exp_type == "exp1":
            log.info(f"  num_sceneco_layers: {len(self.model.denoiser.model.root_model.sceneco_layers)} (per block)")
        else:
            log.info(f"  num_sceneco_layers: {len(self.model.denoiser.model.root_model.seqTransEncoder.layers)} (per block)")

        trainable_p = [n for n, p in self.model.named_parameters() if p.requires_grad]
        nontrainable_but_new = [n for n, p in self.model.named_parameters() if n.startswith(('scene_encoder', 'scene_co', 'sceneco')) and not p.requires_grad]
        log.info(f"  trainable scene params: {len(trainable_p)}")
        log.info(f"  scene params (if any frozen): {nontrainable_but_new}")
        log.info("=" * 60)

    def train(self):
        log.info("Starting training...")
        log.info(f"  Experiment type: {self.args.exp_type}")
        log.info(f"  Train dataset: {len(self.train_dataset)} samples")
        log.info(f"  Val dataset: {len(self.val_dataset)} samples")
        log.info(f"  Epochs: {self.args.num_epochs}")
        log.info(f"  Batch size: {self.args.batch_size}")
        log.info(f"  Accum. steps: {self.args.accum_steps} (effective batch={self.args.batch_size * self.args.accum_steps})")
        log.info(f"  LR: {self.args.lr}, Prior weight: {self.args.prior_weight}, Scene dropout: {self.args.scene_dropout}")

        first_batch_logged = False
        accum_steps = self.args.accum_steps
        self.optimizer.zero_grad()
        last_grad_norm = 0.0
        last_lr = 0.0

        for epoch in range(self.args.num_epochs):
            epoch_loss = 0
            epoch_mse = 0
            n_steps = 0
            accum_count = 0

            self.model.train()
            for batch in self.train_loader:
                if not first_batch_logged:
                    self._log_first_batch(batch)
                    first_batch_logged = True

                metrics = self.train_step(batch, accum_steps)
                epoch_loss += metrics["loss"]
                epoch_mse += metrics["mse"]
                n_steps += 1
                self.global_step += 1
                accum_count += 1

                if accum_count >= accum_steps:
                    last_grad_norm = self._optimizer_step()
                    last_lr = self.scheduler.get_last_lr()[0]
                    accum_count = 0

                if self.global_step % self.args.log_interval == 0:
                    alpha_vals = self._get_alpha_stats()
                    log.info(
                        f"[Epoch {epoch+1}/{self.args.num_epochs}] "
                        f"step {n_steps}/{len(self.train_loader)}: "
                        f"loss={metrics['loss']:.2f}, mse={metrics['mse']:.2f}, "
                        f"prior_loss={metrics['prior_loss']:.4f}, "
                        f"alpha={alpha_vals}, "
                        f"grad_norm={last_grad_norm:.4f}, lr={last_lr:.2e}"
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                        self.writer.add_scalar("train/mse", metrics["mse"], self.global_step)
                        self.writer.add_scalar("train/prior_loss", metrics["prior_loss"], self.global_step)
                        self.writer.add_scalar("train/grad_norm", last_grad_norm, self.global_step)
                        self.writer.add_scalar("train/lr", last_lr, self.global_step)

                if self.global_step % self.args.val_interval == 0:
                    log.info("Running validation...")
                    val_metrics = self.validate()
                    for k, v in val_metrics.items():
                        if self.writer:
                            self.writer.add_scalar(f"val/{k}", v, self.global_step)
                    log.info(
                        f"  Val loss: {val_metrics['val_loss']:.2f} (per-elem: {val_metrics['val_loss']/self.motion_dim:.4f}), "
                        f"Val MSE: {val_metrics['val_mse']:.2f} (per-elem: {val_metrics['val_mse']/self.motion_dim:.4f})"
                    )
                    if "degradation_ratio" in val_metrics:
                        log.info(
                            f"  Degradation ratio: {val_metrics['degradation_ratio']:.3f}x"
                        )

                    is_best = val_metrics["val_loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["val_loss"]
                        self.best_epoch = epoch
                        self.save_checkpoint(epoch)
                    self.model.train()

            if accum_count > 0:
                self._optimizer_step()

            avg_loss = epoch_loss / max(n_steps, 1)
            avg_mse = epoch_mse / max(n_steps, 1)

            alpha_vals = self._get_alpha_stats()

            log.info(
                f"Epoch {epoch+1}/{self.args.num_epochs}: "
                f"loss={avg_loss:.2f} (per-elem: {avg_loss/self.motion_dim:.4f}), "
                f"mse={avg_mse:.2f} (per-elem: {avg_mse/self.motion_dim:.4f}), "
                f"alpha={alpha_vals}"
            )

        if self.writer:
            self.writer.close()
        log.info(f"Training complete! Best val_loss={self.best_val_loss:.6f} at epoch {self.best_epoch+1}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Kimodo-SceneCo (Exp1/Exp2/Exp3/Exp4)")

    parser.add_argument("--exp_type", type=str, choices=["exp1", "exp2", "exp3", "exp4"], required=True)

    parser.add_argument("--data_root", type=str, default="/home/lzsh2025/kimodo-viser/LINGO/dataset")
    parser.add_argument("--soma_data_root", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default="Kimodo-SOMA-RP-v1.1")
    parser.add_argument("--baseline_model", type=str, default=None)

    parser.add_argument("--voxel_size", type=str, default="64,64,64")
    parser.add_argument("--patch_size", type=str, default="8,8,8")
    parser.add_argument("--scene_dim", type=int, default=256)
    parser.add_argument("--scene_num_heads", type=int, default=4)
    parser.add_argument("--scene_num_layers", type=int, default=4)
    parser.add_argument("--scene_ff_dim", type=int, default=512)
    parser.add_argument("--sceneco_dropout", type=float, default=0.1)
    parser.add_argument("--num_base_steps", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=0.5)
    parser.add_argument("--scene_dropout", type=float, default=0.1)
    parser.add_argument("--freeze_pretrained", action="store_true", default=True)
    parser.add_argument("--no_freeze", action="store_true", default=False)

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

    args = parser.parse_args()
    if args.no_freeze:
        args.freeze_pretrained = False
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

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
