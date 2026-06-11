"""Evaluate SceneAdapt-style scene metrics.

Default mode uses the original LINGO scene grid in true coordinates:
  - Raw scene: 300x100x400 bool grid = (Z depth, Y height, X width)
  - Voxel size: 0.02m
  - World mapping: X and Z are centered at 0, Y starts at the floor
  - Floor/contact layer below --floor_ignore_height is ignored

The previous 64^3 motion-extent projection is still available as
``--metric_mode legacy2d`` for reproducing older reports. It is not the
default because GT motions can receive high CFR under that proxy.

Usage:
    python eval/eval_sceneadapt_metrics.py \
        --pred_dir outputs/e7_gt_root_stage2_sceneco/val_gen \
        --cache_dir lingo_smplx_cache \
        --output_csv outputs/e7_gt_root_stage2_sceneco/scene_metrics.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.ndimage import distance_transform_edt

log = logging.getLogger(__name__)

RAW_SCENE_VOXEL_SIZE = 0.02
DEFAULT_FLOOR_IGNORE_HEIGHT = 0.08
DEFAULT_RAW_SCENE_DIR = PROJECT_ROOT / "LINGO" / "dataset" / "dataset" / "Scene"

METRIC_KEYS = [
    "CollisionFrameRate",
    "NonWalkableRootRate",
    "PenetrationRate",
    "PenetrationMean",
    "PenetrationMax",
    "SceneSDFPenalty",
]


def nan_metrics(extra=None):
    metrics = {key: float("nan") for key in METRIC_KEYS}
    if extra:
        metrics.update(extra)
    return metrics


def load_scene_cache(cache_dir, scene_name):
    """Load the cached voxel_grid (64^3) for a given scene_name.

    Scans lingo_smplx_cache for the first NPZ whose scene_name matches.
    Returns (voxel_grid_64, scene_name) or (None, None).
    """
    cache_dir = Path(cache_dir)
    for npz_file in sorted(cache_dir.glob("seg_*.npz")):
        try:
            data = np.load(str(npz_file), allow_pickle=True)
            if str(data.get("scene_name", "")) == scene_name:
                return data["voxel_grid"].astype(np.float32), scene_name
        except Exception:
            continue
    return None, None


def build_scene_cache_index(cache_dir):
    """Map scene_name to a representative cache file with voxel_grid."""
    cache_dir = Path(cache_dir)
    index = {}
    for npz_file in sorted(cache_dir.glob("seg_*.npz")):
        try:
            data = np.load(str(npz_file), allow_pickle=True)
            scene_name = str(data.get("scene_name", ""))
            if scene_name and scene_name not in index and "voxel_grid" in data.files:
                index[scene_name] = npz_file
        except Exception:
            continue
    log.info("Indexed %d scene cache entries from %s", len(index), cache_dir)
    return index


def load_scene_cache_from_index(scene_index, scene_name):
    """Load a scene voxel_grid using a prebuilt scene_name -> cache file index."""
    npz_file = scene_index.get(scene_name)
    if npz_file is None:
        return None, None
    try:
        data = np.load(str(npz_file), allow_pickle=True)
        return data["voxel_grid"].astype(np.float32), scene_name
    except Exception:
        return None, None


def build_2d_sdf_from_cache(voxel_64, motion_x_range, motion_z_range, margin=0.5):
    """Build 2D SDF from cached 64^3 voxel grid using motion extent.
    
    Args:
        voxel_64: (64, 64, 64) float32 occupancy density grid (0-1)
        motion_x_range: (x_min, x_max) in meters
        motion_z_range: (z_min, z_max) in meters
        margin: extra padding around motion extent (meters)
    
    Returns:
        sdf_2d: (64, 64) SDF in XZ plane (negative=occupied, positive=free)
        grid_info: dict with x_min, x_max, z_min, z_max, nx, nz
    """
    x_min, x_max = motion_x_range
    z_min, z_max = motion_z_range
    gx_min = x_min - margin
    gx_max = x_max + margin
    gz_min = z_min - margin
    gz_max = z_max + margin
    
    # Collapse Y dimension: take mean occupancy at each XZ
    occ_2d = voxel_64.mean(axis=1)  # (64, 64)
    
    # Invert: 1 = occupied, 0 = free for SDF computation
    binary_occ = occ_2d > 0.5
    dist_out = distance_transform_edt(~binary_occ).astype(np.float32)
    dist_in = distance_transform_edt(binary_occ).astype(np.float32)
    
    # Scale to physical size
    phys_w = gx_max - gx_min
    phys_h = gz_max - gz_min
    voxel_w = phys_w / 64
    voxel_h = phys_h / 64
    sdf_2d = (dist_out - dist_in) * max(voxel_w, voxel_h)
    
    grid_info = {
        "x_min": gx_min, "x_max": gx_max,
        "z_min": gz_min, "z_max": gz_max,
        "nx": 64, "nz": 64,
    }
    return sdf_2d, grid_info


def compute_legacy2d_scene_metrics(gen_root, gen_joints, scene_name, cache_dir, scene_index=None):
    """Compute the legacy 64^3 motion-extent 2D scene proxy.
    
    Args:
        gen_root: (T, 3) generated root positions in meters
        gen_joints: (T, 22, 3) generated joint positions in meters
        scene_name: scene identifier string
        cache_dir: path to lingo_smplx_cache
    
    Returns:
        Dict of metrics.
    """
    # Load cached voxel_grid for this scene
    if scene_index is not None:
        voxel_64, _ = load_scene_cache_from_index(scene_index, scene_name)
    else:
        voxel_64, _ = load_scene_cache(cache_dir, scene_name)
    if voxel_64 is None:
        return nan_metrics({
            "MetricMode": "legacy2d",
            "OutOfSceneOrFloorIgnoredJointRate": float("nan"),
            "FloorIgnoreHeight": float("nan"),
        })
    
    # Compute motion extent
    x_min, x_max = float(gen_root[:, 0].min()), float(gen_root[:, 0].max())
    z_min, z_max = float(gen_root[:, 2].min()), float(gen_root[:, 2].max())
    
    # Build SDF aligned to motion extent
    sdf_2d, grid = build_2d_sdf_from_cache(voxel_64, (x_min, x_max), (z_min, z_max))
    
    T = gen_root.shape[0]
    nx, nz = grid["nx"], grid["nz"]
    gx_min, gx_max = grid["x_min"], grid["x_max"]
    gz_min, gz_max = grid["z_min"], grid["z_max"]
    
    def world_to_grid(x_world, z_world):
        ix = int((x_world - gx_min) / (gx_max - gx_min) * nx)
        iz = int((z_world - gz_min) / (gz_max - gz_min) * nz)
        return max(0, min(nx - 1, ix)), max(0, min(nz - 1, iz))
    
    # NonWalkableRootRate
    root_sdf = np.array([sdf_2d[world_to_grid(gen_root[t, 0], gen_root[t, 2])]
                         for t in range(T)])
    non_walkable_rate = float((root_sdf < 0).mean())
    
    # CollisionFrameRate: any joint inside obstacle
    collision_frames = np.zeros(T, dtype=bool)
    penetration_depths = []
    
    for t in range(T):
        for j in range(gen_joints.shape[1]):
            ix, iz = world_to_grid(gen_joints[t, j, 0], gen_joints[t, j, 2])
            jsdf = sdf_2d[ix, iz]
            if jsdf < 0:
                collision_frames[t] = True
                penetration_depths.append(-jsdf)
    
    collision_frame_rate = float(collision_frames.mean())
    
    if penetration_depths:
        penetration_rate = len(penetration_depths) / (T * gen_joints.shape[1])
        penetration_mean = float(np.mean(penetration_depths))
        penetration_max = float(np.max(penetration_depths))
    else:
        penetration_rate = 0.0
        penetration_mean = 0.0
        penetration_max = 0.0
    
    # SceneSDFPenalty
    margin = 0.10
    sdf_penalty = np.maximum(margin - root_sdf, 0) ** 2
    scene_sdf_penalty = float(sdf_penalty.mean())
    
    return {
        "CollisionFrameRate": collision_frame_rate,
        "NonWalkableRootRate": non_walkable_rate,
        "PenetrationRate": penetration_rate,
        "PenetrationMean": penetration_mean,
        "PenetrationMax": penetration_max,
        "SceneSDFPenalty": scene_sdf_penalty,
        "MetricMode": "legacy2d",
        "OutOfSceneOrFloorIgnoredJointRate": float("nan"),
        "FloorIgnoreHeight": float("nan"),
    }


def find_raw_scene_path(scene_dir, scene_name):
    """Find a raw LINGO Scene/{scene}.npy file for a scene name."""
    scene_dir = Path(scene_dir)
    candidates = [scene_name]
    base_name = scene_name.split("-")[0]
    if base_name not in candidates:
        candidates.append(base_name)
    no_mirror = scene_name.replace("_mirror", "")
    if no_mirror not in candidates:
        candidates.append(no_mirror)

    for candidate in candidates:
        path = scene_dir / f"{candidate}.npy"
        if path.exists():
            return path
    return None


def load_raw_scene(scene_dir, scene_name, raw_scene_cache=None):
    """Load a raw LINGO scene grid, optionally using a caller-owned cache."""
    path = find_raw_scene_path(scene_dir, scene_name)
    if path is None:
        return None

    key = str(path)
    if raw_scene_cache is not None:
        if key not in raw_scene_cache:
            raw_scene_cache[key] = np.load(key)
        return raw_scene_cache[key]
    return np.load(key)


def _raw_scene_hits(points, raw_scene, floor_ignore_height, scene_voxel_size):
    """Return point hits against raw scene occupancy, excluding floor contact."""
    points = np.asarray(points, dtype=np.float32)
    original_shape = points.shape[:-1]
    pts = points.reshape(-1, 3)
    depth, height, width = raw_scene.shape

    ix = np.floor(pts[:, 0] / scene_voxel_size + width / 2.0).astype(np.int32)
    iy = np.floor(pts[:, 1] / scene_voxel_size).astype(np.int32)
    iz = np.floor(pts[:, 2] / scene_voxel_size + depth / 2.0).astype(np.int32)

    valid = (
        (ix >= 0) & (ix < width)
        & (iy >= 0) & (iy < height)
        & (iz >= 0) & (iz < depth)
        & (pts[:, 1] >= floor_ignore_height)
    )

    hits = np.zeros(pts.shape[0], dtype=bool)
    valid_idx = np.where(valid)[0]
    if valid_idx.size:
        hits[valid_idx] = raw_scene[iz[valid_idx], iy[valid_idx], ix[valid_idx]] > 0

    return hits.reshape(original_shape), valid.reshape(original_shape)


def compute_raw3d_scene_metrics(
    gen_root,
    gen_joints,
    scene_name,
    scene_dir=DEFAULT_RAW_SCENE_DIR,
    floor_ignore_height=DEFAULT_FLOOR_IGNORE_HEIGHT,
    scene_voxel_size=RAW_SCENE_VOXEL_SIZE,
    raw_scene_cache=None,
):
    """Compute point-sampled 3D collision metrics in raw LINGO coordinates.

    This intentionally ignores the floor/contact layer so standing and foot
    contact are not counted as obstacle collisions. Penetration depth is a
    voxel-level proxy because this raw grid has occupancy, not a true SDF.
    """
    raw_scene = load_raw_scene(scene_dir, scene_name, raw_scene_cache=raw_scene_cache)
    if raw_scene is None:
        return nan_metrics({
            "MetricMode": "raw3d_floor_filtered",
            "OutOfSceneOrFloorIgnoredJointRate": float("nan"),
            "FloorIgnoreHeight": floor_ignore_height,
        })

    T = min(gen_root.shape[0], gen_joints.shape[0])
    root = np.asarray(gen_root[:T], dtype=np.float32)
    joints = np.asarray(gen_joints[:T], dtype=np.float32)

    joint_hits, joint_valid = _raw_scene_hits(
        joints,
        raw_scene,
        floor_ignore_height=floor_ignore_height,
        scene_voxel_size=scene_voxel_size,
    )
    root_hits, root_valid = _raw_scene_hits(
        root,
        raw_scene,
        floor_ignore_height=floor_ignore_height,
        scene_voxel_size=scene_voxel_size,
    )

    collision_frame_rate = float(joint_hits.any(axis=1).mean()) if T else 0.0
    penetration_rate = float(joint_hits.mean()) if joint_hits.size else 0.0
    non_walkable_rate = float(root_hits.mean()) if root_hits.size else 0.0

    if joint_hits.any():
        penetration_mean = float(scene_voxel_size)
        penetration_max = float(scene_voxel_size)
    else:
        penetration_mean = 0.0
        penetration_max = 0.0

    return {
        "CollisionFrameRate": collision_frame_rate,
        "NonWalkableRootRate": non_walkable_rate,
        "PenetrationRate": penetration_rate,
        "PenetrationMean": penetration_mean,
        "PenetrationMax": penetration_max,
        "SceneSDFPenalty": float("nan"),
        "MetricMode": "raw3d_floor_filtered",
        "OutOfSceneOrFloorIgnoredJointRate": float((~joint_valid).mean()) if joint_valid.size else 0.0,
        "FloorIgnoreHeight": floor_ignore_height,
    }


def compute_scene_metrics(
    gen_root,
    gen_joints,
    scene_name,
    cache_dir,
    scene_index=None,
    metric_mode="raw3d",
    scene_dir=DEFAULT_RAW_SCENE_DIR,
    floor_ignore_height=DEFAULT_FLOOR_IGNORE_HEIGHT,
    raw_scene_cache=None,
):
    """Compute scene metrics.

    ``raw3d`` is the default corrected metric. ``legacy2d`` reproduces the
    old 64^3 motion-extent projection for comparison.
    """
    if metric_mode == "legacy2d":
        return compute_legacy2d_scene_metrics(
            gen_root,
            gen_joints,
            scene_name,
            cache_dir,
            scene_index=scene_index,
        )
    if metric_mode != "raw3d":
        raise ValueError(f"Unknown metric_mode: {metric_mode}")
    return compute_raw3d_scene_metrics(
        gen_root,
        gen_joints,
        scene_name,
        scene_dir=scene_dir,
        floor_ignore_height=floor_ignore_height,
        raw_scene_cache=raw_scene_cache,
    )


def main():
    parser = argparse.ArgumentParser(description="SceneAdapt metrics evaluation")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="lingo_smplx_cache")
    parser.add_argument("--scene_dir", type=str, default=str(DEFAULT_RAW_SCENE_DIR))
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--metric_mode", choices=["raw3d", "legacy2d"], default="raw3d")
    parser.add_argument("--floor_ignore_height", type=float, default=DEFAULT_FLOOR_IGNORE_HEIGHT)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    pred_dir = Path(args.pred_dir)
    npz_files = sorted(pred_dir.glob("sample_*.npz"))
    if not npz_files:
        npz_files = sorted(pred_dir.glob("seg_*.npz"))
    if not npz_files:
        npz_files = sorted(pred_dir.glob("*.npz"))
    log.info(f"Found {len(npz_files)} samples in {pred_dir}")
    scene_index = build_scene_cache_index(args.cache_dir) if args.metric_mode == "legacy2d" else None
    raw_scene_cache = {} if args.metric_mode == "raw3d" else None

    all_metrics = []
    for npz_file in npz_files:
        data = np.load(str(npz_file), allow_pickle=True)
        gen_root = np.array(data["gen_root"], dtype=np.float32)
        gen_joints = np.array(data["gen_joints"], dtype=np.float32)
        scene_name = str(data.get("scene_name", ""))
        
        if not scene_name or scene_name == "None":
            log.warning(f"  SKIP {npz_file.name}: no scene_name")
            continue
        
        metrics = compute_scene_metrics(
            gen_root,
            gen_joints,
            scene_name,
            args.cache_dir,
            scene_index=scene_index,
            metric_mode=args.metric_mode,
            scene_dir=args.scene_dir,
            floor_ignore_height=args.floor_ignore_height,
            raw_scene_cache=raw_scene_cache,
        )
        metrics["sample_id"] = npz_file.stem
        metrics["method"] = args.method
        all_metrics.append(metrics)
    
    # Write CSV
    if all_metrics:
        fieldnames = [
            "sample_id",
            "method",
            "MetricMode",
            "CollisionFrameRate",
            "NonWalkableRootRate",
            "PenetrationRate",
            "PenetrationMean",
            "PenetrationMax",
            "SceneSDFPenalty",
            "OutOfSceneOrFloorIgnoredJointRate",
            "FloorIgnoreHeight",
        ]
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        
        for key in ["CollisionFrameRate", "NonWalkableRootRate", "PenetrationRate",
                     "PenetrationMean", "PenetrationMax", "SceneSDFPenalty",
                     "OutOfSceneOrFloorIgnoredJointRate"]:
            vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if vals:
                log.info(f"  {key}: {np.mean(vals):.4f}")
    
    log.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
