"""Frame-level feature construction for RootPathSceneClassifier.

Builds a rich per-frame feature vector combining:
  - root position (xz)
  - root height (y)
  - target path
  - positional differences
  - velocities & speeds
  - heading & path direction
  - heading-path angle error
  - scene SDF value
"""

import torch


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def build_root_classifier_features(
    root_5d: torch.Tensor,
    target_path_xz: torch.Tensor,
    scene_sdf=None,
    sample_sdf_fn=None,
) -> torch.Tensor:
    """
    Args:
        root_5d:         (B, T, 5)  meter/canonical [x, y, z, heading_cos, heading_sin]
        target_path_xz:  (B, T, 2)
        scene_sdf:       optional 2D SDF tensor
        sample_sdf_fn:   optional function (scene_sdf, pos) -> (B, T, 1)

    Returns:
        frame_feat: (B, T, 19) when scene_sdf is None, else (B, T, 20)
    """
    pos = root_5d[..., 0:3]          # (B, T, 3)
    heading = root_5d[..., 3:5]      # (B, T, 2)

    root_xz = pos[..., [0, 2]]       # (B, T, 2)
    root_y = pos[..., 1:2]           # (B, T, 1)
    target_xz = target_path_xz       # (B, T, 2)

    # velocity (central difference approximation)
    if root_xz.shape[1] > 1:
        root_vel = root_xz[:, 1:] - root_xz[:, :-1]
        root_vel = torch.cat([root_vel, root_vel[:, -1:]], dim=1)
        target_vel = target_xz[:, 1:] - target_xz[:, :-1]
        target_vel = torch.cat([target_vel, target_vel[:, -1:]], dim=1)
    else:
        root_vel = torch.zeros_like(root_xz)
        target_vel = torch.zeros_like(target_xz)

    root_speed = root_vel.norm(dim=-1, keepdim=True)
    target_speed = target_vel.norm(dim=-1, keepdim=True)

    # path direction
    path_theta = torch.atan2(target_vel[..., 1], target_vel[..., 0])
    path_dir = torch.stack([torch.cos(path_theta), torch.sin(path_theta)], dim=-1)

    # heading-path angle error
    heading_theta = torch.atan2(heading[..., 1], heading[..., 0])
    heading_path_error = angle_diff(heading_theta, path_theta).unsqueeze(-1)

    # distance to target
    root_minus_target = root_xz - target_xz
    dist_to_target = root_minus_target.norm(dim=-1, keepdim=True)

    # core features (19 dims, matching classifier checkpoint input_dim=19)
    feat_parts = [
        root_xz,              # 2
        root_y,               # 1
        target_xz,            # 2
        root_minus_target,    # 2
        dist_to_target,       # 1
        root_vel,             # 2
        target_vel,           # 2
        root_speed,           # 1
        target_speed,         # 1
        heading,              # 2
        path_dir,             # 2
        heading_path_error,   # 1
    ]

    # scene sdf (optional 20th feature, only when SDF is available)
    if scene_sdf is not None and sample_sdf_fn is not None:
        sdf_value = sample_sdf_fn(scene_sdf, pos)
        if sdf_value.dim() == dist_to_target.dim() - 1:
            sdf_value = sdf_value.unsqueeze(-1)
        feat_parts.append(sdf_value)

    frame_feat = torch.cat(feat_parts, dim=-1)

    return frame_feat
