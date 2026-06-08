"""Scene guidance: 2D SDF construction and sampling for root collision avoidance."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


def build_2d_sdf(
    voxel_grid: torch.Tensor,
    voxel_size: float = 0.1,
    grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    root_height: float = 1.0,
    height_tolerance: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a 2D SDF (Signed Distance Field) from a 3D voxel grid for root-level
    collision avoidance.

    Projects the 3D occupancy to a 2D walkable map at root height,
    then computes the distance transform.

    Args:
        voxel_grid: (X, Y, Z) bool or float occupancy grid.
        voxel_size: size of each voxel in meters.
        grid_origin: (x0, y0, z0) world coordinates of the voxel grid origin.
        root_height: typical root height in meters (default 1.0m).
        height_tolerance: tolerance around root_height to consider obstacles.
        device: torch device for the output SDF.

    Returns:
        sdf_2d: (X, Z) float tensor. Positive = free space, Negative = inside obstacle.
    """
    if device is None:
        device = voxel_grid.device

    if voxel_grid.dtype != torch.bool:
        voxel_grid = voxel_grid > 0.5

    X, Y, Z = voxel_grid.shape

    # Find the Y indices corresponding to root_height ± tolerance
    y0 = grid_origin[1]
    y_low = int((root_height - height_tolerance - y0) / voxel_size)
    y_high = int((root_height + height_tolerance - y0) / voxel_size)
    y_low = max(0, y_low)
    y_high = min(Y, y_high)

    # Project to 2D: any occupied voxel in the root height band = obstacle
    if y_low >= y_high:
        # No voxels in this height range, assume free
        occ_2d = torch.zeros(X, Z, dtype=torch.bool, device=device)
    else:
        occ_2d = voxel_grid[:, y_low:y_high, :].any(dim=1)

    # Compute distance transform using scipy
    occ_np = occ_2d.cpu().numpy()
    # Distance from obstacles: inside obstacle = negative, outside = positive
    dist_outside = distance_transform_edt(~occ_np).astype(np.float32) * voxel_size
    dist_inside = distance_transform_edt(occ_np).astype(np.float32) * voxel_size
    sdf_np = dist_outside - dist_inside

    sdf_2d = torch.from_numpy(sdf_np).to(device)
    return sdf_2d


def sample_sdf_2d(
    sdf_2d: torch.Tensor,
    pos: torch.Tensor,
    voxel_size: float = 0.1,
    grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """
    Sample 2D SDF values at given 3D positions (uses XZ coordinates).

    Args:
        sdf_2d: (X, Z) SDF tensor.
        pos: (B, T, 3) positions in world coordinates.
        voxel_size: size of each voxel in meters.
        grid_origin: (x0, y0, z0) world coordinates of the SDF grid origin.

    Returns:
        sdf_values: (B, T) SDF values at the given positions.
    """
    X, Z = sdf_2d.shape

    # Convert world XZ to grid indices
    ix = (pos[..., 0] - grid_origin[0]) / voxel_size
    iz = (pos[..., 2] - grid_origin[2]) / voxel_size

    # Normalize to [-1, 1] for grid_sample
    ix_norm = 2.0 * ix / (X - 1) - 1.0
    iz_norm = 2.0 * iz / (Z - 1) - 1.0

    # grid_sample expects (N, C, H, W) input and (N, 2, ...) grid
    # sdf_2d: (X, Z) -> (1, 1, X, Z)
    sdf_4d = sdf_2d.unsqueeze(0).unsqueeze(0)

    # grid: (B*T, 1, 1, 2) with (x, z) = (iz_norm, ix_norm)
    # grid_sample uses (y, x) ordering = (z, x) in our case
    B, T, _ = pos.shape
    grid = torch.stack([iz_norm, ix_norm], dim=-1)  # (B, T, 2)
    grid = grid.reshape(B * T, 1, 1, 2)

    # Expand sdf_4d to match batch size
    sdf_4d_expanded = sdf_4d.expand(B * T, -1, -1, -1)

    values = F.grid_sample(
        sdf_4d_expanded.float(),
        grid.float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return values.reshape(B, T)
