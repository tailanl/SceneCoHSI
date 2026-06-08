"""Path utilities: resampling, smoothing, heading computation."""

import torch
import torch.nn.functional as F


def resample_path_to_length(path_xz: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resample a sparse path to a target length using linear interpolation.

    Args:
        path_xz: (B, K, 2) sparse path in XZ plane.
        target_len: desired number of frames.

    Returns:
        (B, target_len, 2) dense path.
    """
    path_ch = path_xz.transpose(1, 2)  # (B, 2, K)
    dense = F.interpolate(path_ch, size=target_len, mode="linear", align_corners=True)
    return dense.transpose(1, 2)


def smooth_path_xz(path_xz: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Simple moving average smoothing of a path.

    Args:
        path_xz: (B, T, 2) path in XZ plane.
        kernel_size: smoothing kernel size.

    Returns:
        (B, T, 2) smoothed path.
    """
    if kernel_size <= 1:
        return path_xz

    B, T, C = path_xz.shape
    pad = kernel_size // 2
    x = path_xz.transpose(1, 2)  # (B, 2, T)
    x = F.pad(x, (pad, pad), mode="replicate")

    weight = torch.ones(C, 1, kernel_size, device=path_xz.device, dtype=path_xz.dtype)
    weight = weight / kernel_size

    y = F.conv1d(x, weight, groups=C)
    return y.transpose(1, 2)


def heading_from_path_xz(path_xz: torch.Tensor) -> torch.Tensor:
    """
    Compute heading (cos, sin) from path direction in XZ plane.

    Args:
        path_xz: (B, T, 2) path in XZ plane.

    Returns:
        heading: (B, T, 2) with [cos(theta), sin(theta)].
    """
    vel = path_xz[:, 1:] - path_xz[:, :-1]
    theta = torch.atan2(vel[..., 1], vel[..., 0])
    # Repeat last heading for the last frame
    theta = torch.cat([theta, theta[:, -1:]], dim=1)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def path_from_root_5d(root_5d: torch.Tensor) -> torch.Tensor:
    """
    Extract XZ path from 5D root features.

    Args:
        root_5d: (B, T, 5) with [x, y, z, heading_cos, heading_sin].

    Returns:
        (B, T, 2) XZ path.
    """
    return root_5d[..., [0, 2]]
