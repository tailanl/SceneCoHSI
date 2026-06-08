"""EnergyGuidance baseline: analytical root energy guidance for path, speed, smoothness, heading, height, and scene avoidance."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class RootGuidanceConfig:
    enabled: bool = True

    # loss weights
    w_path: float = 10.0
    w_goal: float = 20.0
    w_speed: float = 1.0
    w_smooth: float = 2.0
    w_jerk: float = 0.5
    w_heading: float = 2.0
    w_heading_norm: float = 0.5
    w_height: float = 1.0
    w_scene: float = 5.0

    # scene
    scene_margin: float = 0.10

    # guidance scale
    scale: float = 0.03
    max_grad_norm: float = 1.0

    # when to apply guidance (denoising step range)
    start_step: int = 0
    end_step: int = 40


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed angular difference, wrapped to [-pi, pi]."""
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def denormalize_root_5d(root_norm: torch.Tensor, motion_rep, root_slice: slice) -> torch.Tensor:
    """Convert normalized 5D root features back to meter/radian space.

    Args:
        root_norm: (B, T, 5) normalized root [x_norm, y_norm, z_norm, cos_norm, sin_norm].
        motion_rep: KimodoMotionRep instance with mean/std fields.
        root_slice: slice for root features (typically slice(0, 5)).

    Returns:
        root_5d_meter: (B, T, 5) root in meter space [x, y, z, cos, sin].
    """
    if hasattr(motion_rep, "mean") and hasattr(motion_rep, "std"):
        mean = motion_rep.mean[..., root_slice].to(root_norm.device, root_norm.dtype)
        std = motion_rep.std[..., root_slice].to(root_norm.device, root_norm.dtype)
        return root_norm * std + mean

    if hasattr(motion_rep, "stats") and hasattr(motion_rep.stats, "mean"):
        mean = motion_rep.stats.mean[..., root_slice].to(root_norm.device, root_norm.dtype)
        std = motion_rep.stats.std[..., root_slice].to(root_norm.device, root_norm.dtype)
        eps = getattr(motion_rep.stats, "eps", 0.0)
        return root_norm * torch.sqrt(std**2 + eps) + mean

    root_full = torch.zeros(
        *root_norm.shape[:-1],
        getattr(motion_rep, "motion_rep_dim", root_slice.stop),
        device=root_norm.device,
        dtype=root_norm.dtype,
    )
    root_full[..., root_slice] = root_norm
    return motion_rep.unnormalize(root_full)[..., root_slice]


def compute_root_guidance_loss(
    pred_x0: torch.Tensor,
    target_path_xz: torch.Tensor,
    root_slice: slice,
    cfg: RootGuidanceConfig,
    scene_sdf: Optional[torch.Tensor] = None,
    sample_sdf_fn=None,
    motion_rep=None,
    root_is_normalized: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute the hand-written analytical root guidance loss during sampling.

    Args:
        pred_x0: (B, T, D) predicted clean motion (normalized feature space).
        target_path_xz: (B, T, 2) target path in XZ plane (meter space).
        root_slice: slice for root features (typically slice(0, 5)).
        cfg: RootGuidanceConfig with loss weights.
        scene_sdf: optional 2D SDF tensor for scene avoidance.
        sample_sdf_fn: optional callable(scene_sdf, pos_3d) -> sdf_values.
        motion_rep: KimodoMotionRep instance for denormalization.
        root_is_normalized: whether pred_x0 root is in normalized space.

    Returns:
        Dict with 'total' loss and individual components.
    """
    root = pred_x0[..., root_slice]
    if root_is_normalized and motion_rep is not None:
        root = denormalize_root_5d(root, motion_rep, root_slice)

    pos = root[..., 0:3]
    heading = root[..., 3:5]
    xz = pos[..., [0, 2]]

    # 1. path dense loss
    loss_path = ((xz - target_path_xz) ** 2).sum(dim=-1).mean()

    # 2. final goal loss
    loss_goal = ((xz[:, -1] - target_path_xz[:, -1]) ** 2).sum(dim=-1).mean()

    # 3. speed uniformity
    vel = xz[:, 1:] - xz[:, :-1]
    speed = vel.norm(dim=-1)
    loss_speed = speed.var(dim=-1).mean()

    # 4. acceleration smoothness
    acc = xz[:, 2:] - 2 * xz[:, 1:-1] + xz[:, :-2]
    loss_smooth = (acc ** 2).sum(dim=-1).mean()

    # 5. jerk
    if xz.shape[1] >= 4:
        jerk = xz[:, 3:] - 3 * xz[:, 2:-1] + 3 * xz[:, 1:-2] - xz[:, :-3]
        loss_jerk = (jerk ** 2).sum(dim=-1).mean()
    else:
        loss_jerk = pred_x0.new_tensor(0.0)

    # 6. heading-path consistency
    path_theta = torch.atan2(vel[..., 1], vel[..., 0])
    heading_theta = torch.atan2(heading[:, :-1, 1], heading[:, :-1, 0])
    loss_heading = (angle_diff(heading_theta, path_theta) ** 2).mean()

    # 6b. heading unit-norm regularizer: (||heading|| - 1)^2
    loss_heading_norm = ((heading.norm(dim=-1) - 1.0) ** 2).mean()

    # 7. root height stability
    root_y = pos[..., 1]
    loss_height = ((root_y - root_y[:, :1]) ** 2).mean()

    # 8. scene sdf
    if scene_sdf is not None and sample_sdf_fn is not None:
        sdf_value = sample_sdf_fn(scene_sdf, pos)
        loss_scene = F.relu(cfg.scene_margin - sdf_value).pow(2).mean()
    else:
        loss_scene = pred_x0.new_tensor(0.0)

    total = (
        cfg.w_path * loss_path
        + cfg.w_goal * loss_goal
        + cfg.w_speed * loss_speed
        + cfg.w_smooth * loss_smooth
        + cfg.w_jerk * loss_jerk
        + cfg.w_heading * loss_heading
        + cfg.w_heading_norm * loss_heading_norm
        + cfg.w_height * loss_height
        + cfg.w_scene * loss_scene
    )

    return {
        "total": total,
        "path": loss_path,
        "goal": loss_goal,
        "speed": loss_speed,
        "smooth": loss_smooth,
        "jerk": loss_jerk,
        "heading": loss_heading,
        "heading_norm": loss_heading_norm,
        "height": loss_height,
        "scene": loss_scene,
    }
