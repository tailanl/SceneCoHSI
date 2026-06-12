"""Raw LINGO scene utilities for root-level feasibility.

This module uses the same raw scene convention as eval/eval_sceneadapt_metrics.py
and the official LINGO datasets/lingo.py:
Scene/{scene}.npy has shape (X, Y, Z), voxel size is 0.02m, X/Z are centered
at world zero, and the floor/contact layer below floor_ignore_height is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


RAW_SCENE_VOXEL_SIZE = 0.02
DEFAULT_FLOOR_IGNORE_HEIGHT = 0.08


@dataclass(frozen=True)
class RawSceneInfo:
    scene_name: str
    path: Path
    voxel_size: float
    floor_ignore_height: float
    depth: int
    height: int
    width: int

    @property
    def x_min(self) -> float:
        return -self.width * self.voxel_size / 2.0

    @property
    def x_max(self) -> float:
        return self.width * self.voxel_size / 2.0

    @property
    def z_min(self) -> float:
        return -self.depth * self.voxel_size / 2.0

    @property
    def z_max(self) -> float:
        return self.depth * self.voxel_size / 2.0


@dataclass
class RawScene2D:
    """Top-down occupancy and nearest-free projection data."""

    occ_xz: np.ndarray
    free_xz: np.ndarray
    dist_to_obstacle_m: np.ndarray
    nearest_free_ix: np.ndarray
    nearest_free_iz: np.ndarray
    info: RawSceneInfo


def find_raw_scene_path(scene_dir: str | Path, scene_name: str) -> Path | None:
    """Find Scene/{scene}.npy, handling mirrored scene suffixes."""
    scene_dir = Path(scene_dir)
    candidates = [scene_name]
    base_name = scene_name.split("-")[0]
    no_mirror = scene_name.replace("_mirror", "")
    for candidate in (base_name, no_mirror):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        path = scene_dir / f"{candidate}.npy"
        if path.exists():
            return path
    return None


def load_raw_scene(scene_dir: str | Path, scene_name: str) -> tuple[np.ndarray, RawSceneInfo] | tuple[None, None]:
    path = find_raw_scene_path(scene_dir, scene_name)
    if path is None:
        return None, None
    raw = np.load(str(path))
    width, height, depth = raw.shape
    info = RawSceneInfo(
        scene_name=scene_name,
        path=path,
        voxel_size=RAW_SCENE_VOXEL_SIZE,
        floor_ignore_height=DEFAULT_FLOOR_IGNORE_HEIGHT,
        depth=depth,
        height=height,
        width=width,
    )
    return raw, info


def make_raw_scene_info(
    raw_scene: np.ndarray,
    scene_name: str,
    path: str | Path,
    voxel_size: float = RAW_SCENE_VOXEL_SIZE,
    floor_ignore_height: float = DEFAULT_FLOOR_IGNORE_HEIGHT,
) -> RawSceneInfo:
    width, height, depth = raw_scene.shape
    return RawSceneInfo(
        scene_name=scene_name,
        path=Path(path),
        voxel_size=voxel_size,
        floor_ignore_height=floor_ignore_height,
        depth=depth,
        height=height,
        width=width,
    )


def xz_to_grid(xz: np.ndarray, info: RawSceneInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map world XZ points to raw scene ix/iz and in-bounds mask."""
    pts = np.asarray(xz, dtype=np.float32).reshape(-1, 2)
    ix = np.floor(pts[:, 0] / info.voxel_size + info.width / 2.0).astype(np.int32)
    iz = np.floor(pts[:, 1] / info.voxel_size + info.depth / 2.0).astype(np.int32)
    in_bounds = (ix >= 0) & (ix < info.width) & (iz >= 0) & (iz < info.depth)
    return ix, iz, in_bounds


def grid_to_xz(ix: np.ndarray, iz: np.ndarray, info: RawSceneInfo) -> np.ndarray:
    """Map raw scene ix/iz cell centers to world XZ."""
    x = (np.asarray(ix, dtype=np.float32) + 0.5 - info.width / 2.0) * info.voxel_size
    z = (np.asarray(iz, dtype=np.float32) + 0.5 - info.depth / 2.0) * info.voxel_size
    return np.stack([x, z], axis=-1)


def build_topdown_occupancy(
    raw_scene: np.ndarray,
    info: RawSceneInfo,
    floor_ignore_height: float | None = None,
) -> np.ndarray:
    """Collapse raw 3D occupancy to XZ, ignoring floor/contact voxels."""
    if floor_ignore_height is None:
        floor_ignore_height = info.floor_ignore_height
    y_start = int(np.ceil(floor_ignore_height / info.voxel_size))
    y_start = max(0, min(info.height, y_start))
    occ_xyz = raw_scene[:, y_start:, :] > 0
    return occ_xyz.any(axis=1).copy()  # (X, Z)


def build_raw_scene_2d(
    raw_scene: np.ndarray,
    info: RawSceneInfo,
    floor_ignore_height: float | None = None,
) -> RawScene2D:
    """Build top-down occupancy, distance, and nearest-free maps."""
    occ_xz = build_topdown_occupancy(raw_scene, info, floor_ignore_height=floor_ignore_height)
    free_xz = ~occ_xz

    # Distance to nearest obstacle for free-space clearance.
    dist_to_obstacle = distance_transform_edt(free_xz).astype(np.float32) * info.voxel_size

    # For any cell, nearest free cell indices. Used to project obstacles/out-of-bounds.
    _, nearest = distance_transform_edt(~free_xz, return_indices=True)
    nearest_free_ix = nearest[0].astype(np.int32)
    nearest_free_iz = nearest[1].astype(np.int32)

    return RawScene2D(
        occ_xz=occ_xz,
        free_xz=free_xz,
        dist_to_obstacle_m=dist_to_obstacle,
        nearest_free_ix=nearest_free_ix,
        nearest_free_iz=nearest_free_iz,
        info=info,
    )


def root_feasibility(xz: np.ndarray, scene2d: RawScene2D, clearance_m: float = 0.0) -> dict[str, float]:
    """Compute root-only feasibility statistics for XZ path."""
    pts = np.asarray(xz, dtype=np.float32).reshape(-1, 2)
    ix, iz, in_bounds = xz_to_grid(pts, scene2d.info)

    valid_ix = np.clip(ix, 0, scene2d.info.width - 1)
    valid_iz = np.clip(iz, 0, scene2d.info.depth - 1)
    occupied = np.ones(len(pts), dtype=bool)
    clearance_bad = np.ones(len(pts), dtype=bool)
    if len(pts):
        occupied[in_bounds] = scene2d.occ_xz[valid_ix[in_bounds], valid_iz[in_bounds]]
        clearance_bad[in_bounds] = (
            scene2d.dist_to_obstacle_m[valid_ix[in_bounds], valid_iz[in_bounds]] < clearance_m
        )

    invalid = (~in_bounds) | occupied | clearance_bad
    return {
        "frames": float(len(pts)),
        "out_of_bounds_rate": float((~in_bounds).mean()) if len(pts) else 0.0,
        "occupied_rate": float(occupied.mean()) if len(pts) else 0.0,
        "clearance_violation_rate": float(clearance_bad.mean()) if len(pts) else 0.0,
        "invalid_rate": float(invalid.mean()) if len(pts) else 0.0,
        "min_clearance_m": float(scene2d.dist_to_obstacle_m[valid_ix[in_bounds], valid_iz[in_bounds]].min())
        if in_bounds.any() else float("nan"),
    }


def project_xz_to_free(
    xz: np.ndarray,
    scene2d: RawScene2D,
    clearance_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project XZ points outside free space to nearest free/clearance-valid cells.

    Returns:
        corrected_xz: same shape as input
        changed: boolean mask with one value per point
    """
    original_shape = np.asarray(xz).shape
    pts = np.asarray(xz, dtype=np.float32).reshape(-1, 2)
    ix, iz, in_bounds = xz_to_grid(pts, scene2d.info)
    clipped_ix = np.clip(ix, 0, scene2d.info.width - 1)
    clipped_iz = np.clip(iz, 0, scene2d.info.depth - 1)

    occupied = np.ones(len(pts), dtype=bool)
    clearance_bad = np.ones(len(pts), dtype=bool)
    if len(pts):
        occupied[in_bounds] = scene2d.occ_xz[clipped_ix[in_bounds], clipped_iz[in_bounds]]
        clearance_bad[in_bounds] = (
            scene2d.dist_to_obstacle_m[clipped_ix[in_bounds], clipped_iz[in_bounds]] < clearance_m
        )
    changed = (~in_bounds) | occupied | clearance_bad

    target_free = scene2d.free_xz
    if clearance_m > 0:
        target_free = target_free & (scene2d.dist_to_obstacle_m >= clearance_m)
    if not target_free.any():
        target_free = scene2d.free_xz

    # Recompute nearest free if clearance changes the target set.
    _, nearest = distance_transform_edt(~target_free, return_indices=True)
    nearest_ix = nearest[0].astype(np.int32)
    nearest_iz = nearest[1].astype(np.int32)

    corrected = pts.copy()
    if changed.any():
        src_ix = clipped_ix[changed]
        src_iz = clipped_iz[changed]
        dst_ix = nearest_ix[src_ix, src_iz]
        dst_iz = nearest_iz[src_ix, src_iz]
        corrected[changed] = grid_to_xz(dst_ix, dst_iz, scene2d.info)

    return corrected.reshape(original_shape), changed.reshape(original_shape[:-1])


def smooth_xz_path(xz: np.ndarray, window: int = 5, keep_endpoints: bool = True) -> np.ndarray:
    """Small moving-average smoother for corrected root paths."""
    if window <= 1:
        return np.asarray(xz, dtype=np.float32).copy()
    path = np.asarray(xz, dtype=np.float32)
    if path.shape[0] < 3:
        return path.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(path, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.stack([
        np.convolve(padded[:, dim], kernel, mode="valid")
        for dim in range(path.shape[1])
    ], axis=-1).astype(np.float32)
    if keep_endpoints:
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
    return smoothed
