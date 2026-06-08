"""Voxelization: convert 3D scene mesh / point cloud into occupancy grids.

Supports multiple input formats: mesh (.obj/.ply/.glb), point cloud (.ply/.npy).
Outputs .npz files with occupancy, metadata, and transformation matrices.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def compute_transform_world_to_voxel(
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Compute [4, 4] homogeneous transform matrix: world coords → voxel indices.

    voxel_idx = floor((world_xyz - origin) / voxel_size)

    Args:
        origin: [3] min corner of voxel grid.
        voxel_size: physical size of one voxel side.

    Returns:
        T_w2v: [4, 4] homogeneous matrix.
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.eye(3) / voxel_size
    T[:3, 3] = -origin / voxel_size
    return T


def compute_transform_voxel_to_world(
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Compute [4, 4] inverse transform: voxel indices → world coordinates.

    world_xyz = voxel_idx * voxel_size + origin

    Args:
        origin: [3] min corner.
        voxel_size: physical voxel side length.

    Returns:
        T_v2w: [4, 4] homogeneous matrix.
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.eye(3) * voxel_size
    T[:3, 3] = origin
    return T


def mesh_to_voxel(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_size: Tuple[int, int, int] = (64, 64, 64),
    scene_bounds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """Convert a triangle mesh to occupancy voxel grid.

    Uses a KDTree-based approach for fast voxelization.

    Args:
        vertices: [V, 3] vertex positions.
        faces: [F, 3] triangle face indices.
        grid_size: (X, Y, Z) output grid dimensions.
        scene_bounds: [2, 3] optional (min, max) bounds override.

    Returns:
        occupancy: [X, Y, Z] binary occupancy grid.
        meta: dict with voxel_size, origin, scene_bounds, transform matrices.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    gx, gy, gz = grid_size

    if scene_bounds is None:
        min_b = vertices.min(axis=0) - 0.1
        max_b = vertices.max(axis=0) + 0.1
    else:
        min_b, max_b = np.asarray(scene_bounds[0]), np.asarray(scene_bounds[1])

    voxel_size_arr = (max_b - min_b) / np.array([gx, gy, gz])
    voxel_size_val = float(voxel_size_arr.max())

    occupancy = np.zeros((gx, gy, gz), dtype=np.float32)

    from scipy.spatial import KDTree
    tree = KDTree(vertices)

    tri_centers = vertices[faces].mean(axis=1)

    for cx in range(gx):
        x = min_b[0] + (cx + 0.5) * voxel_size_arr[0]
        for cy in range(gy):
            y = min_b[1] + (cy + 0.5) * voxel_size_arr[1]
            for cz in range(gz):
                z = min_b[2] + (cz + 0.5) * voxel_size_arr[2]
                d, _ = tree.query([x, y, z])
                if d < voxel_size_val:
                    occupancy[cx, cy, cz] = 1.0

    origin = min_b.astype(np.float32)

    meta = {
        "voxel_size": voxel_size_val,
        "origin": origin.tolist(),
        "scene_bounds": np.array([min_b, max_b], dtype=np.float32),
        "grid_size": list(grid_size),
        "transform_world_to_voxel": compute_transform_world_to_voxel(origin, voxel_size_val),
        "transform_voxel_to_world": compute_transform_voxel_to_world(origin, voxel_size_val),
    }

    return occupancy, meta


def pointcloud_to_voxel(
    points: np.ndarray,
    grid_size: Tuple[int, int, int] = (64, 64, 64),
    voxel_size: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """Convert point cloud to occupancy voxel grid.

    Args:
        points: [N, 3] point positions.
        grid_size: (X, Y, Z) output grid dimensions.
        voxel_size: Optional float; if None, computed from bounds.

    Returns:
        occupancy: [X, Y, Z] binary occupancy grid.
        meta: dict with voxel_size, origin, scene_bounds, transform matrices.
    """
    points = np.asarray(points, dtype=np.float64)
    gx, gy, gz = grid_size

    min_b = points.min(axis=0) - 0.05
    max_b = points.max(axis=0) + 0.05

    if voxel_size is None:
        span = max_b - min_b
        voxel_size = float(max(span / np.array([gx, gy, gz])))

    occupancy = np.zeros((gx, gy, gz), dtype=np.float32)

    coords = ((points - min_b) / voxel_size).astype(np.int32)
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < gx)
        & (coords[:, 1] >= 0) & (coords[:, 1] < gy)
        & (coords[:, 2] >= 0) & (coords[:, 2] < gz)
    )
    coords = coords[valid]
    occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

    origin = min_b.astype(np.float32)

    meta = {
        "voxel_size": voxel_size,
        "origin": origin.tolist(),
        "scene_bounds": np.array([min_b, max_b], dtype=np.float32),
        "grid_size": list(grid_size),
        "transform_world_to_voxel": compute_transform_world_to_voxel(origin, voxel_size),
        "transform_voxel_to_world": compute_transform_voxel_to_world(origin, voxel_size),
    }

    return occupancy, meta


def save_voxel_scene(
    output_path: str,
    occupancy: np.ndarray,
    meta: dict,
    scene_id: str = "unknown",
):
    """Save voxelized scene to .npz file.

    Args:
        output_path: Path to output .npz file.
        occupancy: [X, Y, Z] binary occupancy grid.
        meta: Metadata dict from voxelization (must contain transform matrices).
        scene_id: Scene identifier string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "occupancy": occupancy.astype(np.float32),
        "scene_id": scene_id,
        "voxel_size": meta["voxel_size"],
        "origin": np.array(meta["origin"], dtype=np.float32),
        "scene_bounds": meta["scene_bounds"],
        "grid_size": np.array(meta["grid_size"], dtype=np.int32),
        "transform_world_to_voxel": meta.get(
            "transform_world_to_voxel",
            compute_transform_world_to_voxel(
                np.array(meta["origin"], dtype=np.float32),
                float(meta["voxel_size"]),
            ),
        ),
        "transform_voxel_to_world": meta.get(
            "transform_voxel_to_world",
            compute_transform_voxel_to_world(
                np.array(meta["origin"], dtype=np.float32),
                float(meta["voxel_size"]),
            ),
        ),
    }

    np.savez_compressed(str(output_path), **save_dict)


def load_voxel_scene(path: str) -> Tuple[np.ndarray, dict]:
    """Load voxelized scene from .npz file.

    Returns:
        occupancy: [X, Y, Z] binary occupancy grid.
        meta: dict with all metadata including transform matrices.
    """
    data = np.load(path, allow_pickle=True)
    occupancy = data["occupancy"]
    meta = {
        "voxel_size": float(data["voxel_size"]),
        "origin": data["origin"],
        "scene_bounds": data["scene_bounds"],
        "grid_size": data["grid_size"],
        "scene_id": str(data.get("scene_id", Path(path).stem)),
        "transform_world_to_voxel": data.get(
            "transform_world_to_voxel",
            compute_transform_world_to_voxel(
                data["origin"], float(data["voxel_size"])
            ),
        ),
        "transform_voxel_to_world": data.get(
            "transform_voxel_to_world",
            compute_transform_voxel_to_world(
                data["origin"], float(data["voxel_size"])
            ),
        ),
    }
    return occupancy, meta


def load_voxel_scene_lingo(
    scene_npy_path: str,
    target_size: Tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """Load LINGO-format scene .npy file and downsample to target voxel size.

    The LINGO scene files are pre-computed voxel grids at various resolutions.
    This function loads and downsamples them to a consistent size.

    Args:
        scene_npy_path: Path to scene .npy file.
        target_size: (X, Y, Z) target resolution.

    Returns:
        voxel: [target_size] binary occupancy grid.
    """
    import scipy.ndimage

    voxel = np.load(scene_npy_path).astype(np.float32)
    sx, sy, sz = voxel.shape
    tx, ty, tz = target_size

    if sx == tx and sy == ty and sz == tz:
        return voxel

    zoom_factors = (tx / sx, ty / sy, tz / sz)
    voxel = scipy.ndimage.zoom(voxel, zoom_factors, order=0) > 0.5
    voxel = voxel.astype(np.float32)

    result = np.zeros(target_size, dtype=np.float32)
    cx, cy, cz = min(tx, voxel.shape[0]), min(ty, voxel.shape[1]), min(tz, voxel.shape[2])
    result[:cx, :cy, :cz] = voxel[:cx, :cy, :cz]
    return result


__all__ = [
    "compute_transform_world_to_voxel",
    "compute_transform_voxel_to_world",
    "mesh_to_voxel",
    "pointcloud_to_voxel",
    "save_voxel_scene",
    "load_voxel_scene",
    "load_voxel_scene_lingo",
]
