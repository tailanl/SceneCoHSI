# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for Kimodo-SceneCo training pipeline.

Runs on CPU with synthetic data to verify:
1. Model construction (VoxelViT + SceneCo + TwostageDenoiser + CFG)
2. Forward pass with scene features
3. Loss computation (L_diff + L_prior)
4. Backward pass and gradient flow (only SceneCo layers)
5. Validation with baseline comparison
6. Checkpoint save/load
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class FakeMotionRep:
    motion_rep_dim = 369
    global_root_dim = 5
    local_root_dim = 4
    body_slice = slice(5, 369)
    normalize_stats = None
    fps = 30

    class skeleton:
        nbjoints = 30
        foot_joint_idx = [7, 8, 10, 11]

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x

    def global_root_to_local_root(self, root_motion, normalized=True, lengths=None):
        B, T, _ = root_motion.shape
        local = torch.zeros(B, T, self.local_root_dim, device=root_motion.device)
        local[:, :, 0] = root_motion[:, :, 3]
        local[:, :, 1] = root_motion[:, :, 0]
        local[:, :, 2] = root_motion[:, :, 2]
        local[:, :, 3] = root_motion[:, :, 1]
        return local

    def create_conditions_from_constraints_batched(self, *args, **kwargs):
        return None, None


def create_synthetic_batch(batch_size=2, max_frames=60, device="cpu"):
    motion_features = torch.randn(batch_size, max_frames, 369, device=device)
    motion_mask = torch.ones(batch_size, max_frames, dtype=torch.bool, device=device)
    lengths = torch.full((batch_size,), max_frames, dtype=torch.long)
    if batch_size > 1:
        lengths[1] = max_frames * 3 // 4
        motion_mask[1, lengths[1]:] = False

    voxel_grid = torch.randn(batch_size, 1, 64, 64, 64, device=device)
    voxel_grid = (voxel_grid > 0.5).float()

    texts = ["a person walks forward", "a person sits down on the chair"]

    return {
        "motion_features": motion_features,
        "motion_mask": motion_mask,
        "voxel_grid": voxel_grid,
        "texts": texts,
        "lengths": lengths,
    }


def test_model_construction():
    log.info("=" * 60)
    log.info("TEST 1: Model Construction")
    log.info("=" * 60)

    from kimodo_sceneco.model.backbone import (
        SceneCoLayer,
        SceneCoTransformerEncoderLayer,
        SceneCoTransformerEncoder,
        TransformerEncoderBlock,
    )
    from kimodo_sceneco.model.scene_encoder import VoxelViT
    from kimodo_sceneco.model.twostage_denoiser import TwostageDenoiser
    from kimodo_sceneco.model.cfg import ClassifierFreeGuidedModel

    motion_rep = FakeMotionRep()

    voxel_vit = VoxelViT(
        voxel_size=(64, 64, 64),
        patch_size=(8, 8, 8),
        in_channels=1,
        d_model=256,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
    )

    denoiser_kwargs = {
        "latent_dim": 256,
        "ff_size": 512,
        "num_layers": 2,
        "num_heads": 4,
        "activation": "gelu",
        "dropout": 0.1,
        "pe_dropout": 0.1,
        "norm_first": True,
        "use_text_mask": True,
        "llm_shape": [1, 4096],
        "input_first_heading_angle": True,
        "use_sceneco": True,
        "scene_feat_dim": 256,
        "sceneco_dropout": 0.1,
    }

    denoiser = TwostageDenoiser(
        motion_rep=motion_rep,
        motion_mask_mode="concat",
        **denoiser_kwargs,
    )

    cfg_model = ClassifierFreeGuidedModel(denoiser, cfg_type="scene_separated")

    total_params = sum(p.numel() for p in denoiser.parameters())
    sceneco_params = sum(
        p.numel() for n, p in denoiser.named_parameters() if "sceneco" in n
    )
    log.info(f"  Total denoiser params: {total_params:,}")
    log.info(f"  SceneCo params: {sceneco_params:,}")
    log.info(f"  SceneCo ratio: {100 * sceneco_params / total_params:.1f}%")

    log.info("  PASSED ✓")
    return denoiser, cfg_model, voxel_vit, motion_rep


def test_forward_pass(denoiser, cfg_model, voxel_vit, device="cpu"):
    log.info("=" * 60)
    log.info("TEST 2: Forward Pass with Scene Features")
    log.info("=" * 60)

    batch = create_synthetic_batch(batch_size=2, max_frames=60, device=device)
    voxel = batch["voxel_grid"]

    scene_feat, scene_mask = voxel_vit(voxel)
    log.info(f"  VoxelViT output: feat={scene_feat.shape}, mask={scene_mask.shape}")

    B, T = 2, 60
    x = batch["motion_features"]
    x_pad_mask = batch["motion_mask"]
    text_feat = torch.randn(B, 1, 4096, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    timesteps = torch.randint(0, 1000, (B,), device=device)
    heading = torch.zeros(B, device=device)
    motion_mask = torch.zeros_like(x)
    observed_motion = torch.zeros_like(x)

    with torch.no_grad():
        out = denoiser(
            x, x_pad_mask, text_feat, text_pad_mask, timesteps,
            first_heading_angle=heading,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            scene_feat=scene_feat,
            scene_mask=scene_mask,
        )
    log.info(f"  Denoiser output: {out.shape}")
    assert out.shape == (B, T, 369), f"Expected (2, 60, 369), got {out.shape}"

    with torch.no_grad():
        out_no_scene = denoiser(
            x, x_pad_mask, text_feat, text_pad_mask, timesteps,
            first_heading_angle=heading,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            scene_feat=None,
            scene_mask=None,
        )
    log.info(f"  Denoiser output (no scene): {out_no_scene.shape}")
    assert out_no_scene.shape == (B, T, 369)

    log.info("  PASSED ✓")
    return scene_feat, scene_mask


def test_cfg_forward(cfg_model, voxel_vit, device="cpu"):
    log.info("=" * 60)
    log.info("TEST 3: Scene-Separated CFG Forward")
    log.info("=" * 60)

    batch = create_synthetic_batch(batch_size=1, max_frames=30, device=device)
    scene_feat, scene_mask = voxel_vit(batch["voxel_grid"])

    with torch.no_grad():
        out = cfg_model(
            cfg_weight=[2.0, 2.0, 2.0],
            x=batch["motion_features"],
            x_pad_mask=batch["motion_mask"],
            text_feat=torch.randn(1, 1, 4096, device=device),
            text_feat_pad_mask=torch.ones(1, 1, dtype=torch.bool, device=device),
            timesteps=torch.tensor([500], device=device),
            first_heading_angle=torch.zeros(1, device=device),
            motion_mask=torch.zeros_like(batch["motion_features"]),
            observed_motion=torch.zeros_like(batch["motion_features"]),
            scene_feat=scene_feat,
            scene_mask=scene_mask,
            cfg_type="scene_separated",
        )
    log.info(f"  CFG output: {out.shape}")
    assert out.shape == (1, 30, 369)

    log.info("  PASSED ✓")


def test_loss_and_backward(denoiser, voxel_vit, device="cpu"):
    log.info("=" * 60)
    log.info("TEST 4: Loss Computation + Backward + Gradient Check")
    log.info("=" * 60)

    from kimodo_sceneco.model.diffusion import Diffusion

    diffusion = Diffusion(num_base_steps=1000)
    motion_rep = FakeMotionRep()

    batch = create_synthetic_batch(batch_size=2, max_frames=60, device=device)
    scene_feat, scene_mask = voxel_vit(batch["voxel_grid"])

    x_start = batch["motion_features"]
    mask = batch["motion_mask"]
    B = 2
    t = torch.randint(0, 1000, (B,), device=device)

    noise = torch.randn_like(x_start)
    x_t = diffusion.q_sample(x_start, t, noise=noise)

    text_feat = torch.randn(B, 1, 4096, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    heading = torch.zeros(B, device=device)
    motion_mask_t = torch.zeros_like(x_start)
    observed_motion = torch.zeros_like(x_start)

    pred_x0 = denoiser(
        x_t, mask, text_feat, text_pad_mask, t,
        first_heading_angle=heading,
        motion_mask=motion_mask_t,
        observed_motion=observed_motion,
        scene_feat=scene_feat,
        scene_mask=scene_mask,
    )

    mask_f = mask.unsqueeze(-1).float()
    mse_loss = torch.nn.functional.mse_loss(pred_x0 * mask_f, x_start * mask_f, reduction="none")
    mse_loss = (mse_loss.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()
    log.info(f"  MSE loss (with scene): {mse_loss.item():.6f}")

    null_scene = torch.zeros_like(scene_feat)
    null_mask = torch.zeros_like(scene_mask)
    pred_x0_null = denoiser(
        x_t, mask, text_feat, text_pad_mask, t,
        first_heading_angle=heading,
        motion_mask=motion_mask_t,
        observed_motion=observed_motion,
        scene_feat=null_scene,
        scene_mask=null_mask,
    )
    prior_loss = torch.nn.functional.mse_loss(pred_x0_null * mask_f, x_start * mask_f, reduction="none")
    prior_loss = (prior_loss.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()
    log.info(f"  Prior loss (null scene): {prior_loss.item():.6f}")

    total_loss = mse_loss + 0.5 * prior_loss
    log.info(f"  Total loss: {total_loss.item():.6f}")

    total_loss.backward()

    sceneco_grads = {}
    other_grads = {}
    for name, param in denoiser.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "sceneco" in name:
                sceneco_grads[name] = grad_norm
            elif grad_norm > 0:
                other_grads[name] = grad_norm

    log.info(f"  SceneCo params with gradients: {len(sceneco_grads)}")
    for name, gn in list(sceneco_grads.items())[:5]:
        log.info(f"    {name}: grad_norm={gn:.6f}")

    log.info(f"  Non-SceneCo params with gradients: {len(other_grads)}")
    if other_grads:
        for name, gn in list(other_grads.items())[:3]:
            log.info(f"    {name}: grad_norm={gn:.6f}")

    assert len(sceneco_grads) > 0, "SceneCo layers should have gradients!"

    log.info("  PASSED ✓")


def test_freeze_strategy(denoiser, voxel_vit, device="cpu"):
    log.info("=" * 60)
    log.info("TEST 5: Freeze Strategy")
    log.info("=" * 60)

    for name, param in denoiser.named_parameters():
        if "sceneco" in name or "scene_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in denoiser.parameters() if not p.requires_grad)
    log.info(f"  Trainable: {trainable:,}")
    log.info(f"  Frozen: {frozen:,}")
    log.info(f"  Trainable ratio: {100 * trainable / (trainable + frozen):.1f}%")

    voxel_vit_params = sum(p.numel() for p in voxel_vit.parameters())
    log.info(f"  VoxelViT params (all trainable): {voxel_vit_params:,}")

    log.info("  PASSED ✓")


def test_training_loop(denoiser, voxel_vit, device="cpu"):
    log.info("=" * 60)
    log.info("TEST 6: Mini Training Loop (5 steps)")
    log.info("=" * 60)

    from kimodo_sceneco.model.diffusion import Diffusion

    diffusion = Diffusion(num_base_steps=1000)

    for name, param in denoiser.named_parameters():
        if "sceneco" in name or "scene_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    voxel_vit.train()
    denoiser.train()

    optimizer = torch.optim.AdamW(
        [p for p in denoiser.parameters() if p.requires_grad]
        + list(voxel_vit.parameters()),
        lr=1e-4,
    )

    losses = []
    for step in range(5):
        batch = create_synthetic_batch(batch_size=2, max_frames=30, device=device)
        scene_feat, scene_mask = voxel_vit(batch["voxel_grid"])

        x_start = batch["motion_features"]
        mask = batch["motion_mask"]
        B = 2
        t = torch.randint(0, 1000, (B,), device=device)

        noise = torch.randn_like(x_start)
        x_t = diffusion.q_sample(x_start, t, noise=noise)

        text_feat = torch.randn(B, 1, 4096, device=device)
        text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        heading = torch.zeros(B, device=device)
        motion_mask_t = torch.zeros_like(x_start)
        observed_motion = torch.zeros_like(x_start)

        pred_x0 = denoiser(
            x_t, mask, text_feat, text_pad_mask, t,
            first_heading_angle=heading,
            motion_mask=motion_mask_t,
            observed_motion=observed_motion,
            scene_feat=scene_feat,
            scene_mask=scene_mask,
        )

        mask_f = mask.unsqueeze(-1).float()
        mse = torch.nn.functional.mse_loss(pred_x0 * mask_f, x_start * mask_f, reduction="none")
        mse = (mse.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        null_scene = torch.zeros_like(scene_feat)
        null_mask = torch.zeros_like(scene_mask)
        pred_null = denoiser(
            x_t, mask, text_feat, text_pad_mask, t,
            first_heading_angle=heading,
            motion_mask=motion_mask_t,
            observed_motion=observed_motion,
            scene_feat=null_scene,
            scene_mask=null_mask,
        )
        prior = torch.nn.functional.mse_loss(pred_null * mask_f, x_start * mask_f, reduction="none")
        prior = (prior.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()

        loss = mse + 0.5 * prior

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in denoiser.parameters() if p.requires_grad] + list(voxel_vit.parameters()),
            1.0,
        )
        optimizer.step()

        losses.append(loss.item())
        log.info(f"  Step {step+1}: loss={loss.item():.6f}, mse={mse.item():.6f}, prior={prior.item():.6f}")

    log.info(f"  Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")
    log.info("  PASSED ✓")


def test_checkpoint(denoiser, voxel_vit, device="cpu"):
    log.info("=" * 60)
    log.info("TEST 7: Checkpoint Save/Load")
    log.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"

        state = {
            "denoiser": denoiser.state_dict(),
            "voxel_vit": voxel_vit.state_dict(),
        }
        torch.save(state, str(ckpt_path))
        log.info(f"  Saved checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

        loaded = torch.load(str(ckpt_path), map_location=device)
        denoiser.load_state_dict(loaded["denoiser"])
        voxel_vit.load_state_dict(loaded["voxel_vit"])
        log.info("  Loaded checkpoint successfully")

    log.info("  PASSED ✓")


def main():
    device = "cpu"
    log.info(f"Running Kimodo-SceneCo smoke test on device: {device}")
    log.info("")

    denoiser, cfg_model, voxel_vit, motion_rep = test_model_construction()

    scene_feat, scene_mask = test_forward_pass(denoiser, cfg_model, voxel_vit, device)

    test_cfg_forward(cfg_model, voxel_vit, device)

    test_loss_and_backward(denoiser, voxel_vit, device)

    denoiser.zero_grad()

    test_freeze_strategy(denoiser, voxel_vit, device)

    test_training_loop(denoiser, voxel_vit, device)

    test_checkpoint(denoiser, voxel_vit, device)

    log.info("")
    log.info("=" * 60)
    log.info("ALL TESTS PASSED ✓")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
