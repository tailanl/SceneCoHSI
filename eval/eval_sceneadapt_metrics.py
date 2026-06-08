"""Evaluate SceneAdapt-style metrics: collision, penetration, non-walkable root.

NOTE: This is a joint-level 2D proxy evaluation.
      For formal penetration metrics, a mesh-level (SMPL-X vertices) 3D SDF
      evaluation is needed as a future extension.

Usage:
    python eval/eval_sceneadapt_metrics.py \
        --pred_dir outputs/guidance_path_scene_body \
        --scene_dir LINGO/dataset/dataset/Scene \
        --output_csv outputs/guidance_path_scene/scene_metrics.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

log = logging.getLogger(__name__)


def load_scene_voxel(scene_name, scene_dir):
    """Load scene voxel grid from two possible paths.

    Priority: Scene/{scene_name}.npy > Scene/{scene_name}/voxel_grid.npy
              > Scene/{scene_name}/semantic_voxel_grid.npy
    """
    scene_dir_path = Path(scene_dir)

    # Path 1: Scene/{scene_name}.npy (flat)
    flat_path = scene_dir_path / f"{scene_name}.npy"
    if flat_path.exists():
        return np.load(str(flat_path))

    # Path 2: Scene/{scene_name}/voxel_grid.npy (directory)
    scene_path = scene_dir_path / scene_name
    if not scene_path.exists():
        return None

    for fname in ["voxel_grid.npy", "semantic_voxel_grid.npy"]:
        fpath = scene_path / fname
        if fpath.exists():
            return np.load(str(fpath))
    return None


def build_2d_occupancy(voxel_grid, voxel_size=0.1, grid_origin=(0.0, 0.0, 0.0), root_height=1.0, height_tol=0.5):
    """Build 2D occupancy map from 3D voxel grid at root height."""
    if voxel_grid.dtype != bool:
        voxel_grid = voxel_grid > 0.5

    X, Y, Z = voxel_grid.shape
    y_low = max(0, int((root_height - height_tol) / voxel_size))
    y_high = min(Y, int((root_height + height_tol) / voxel_size))

    if y_low >= y_high:
        return np.zeros((X, Z), dtype=bool)

    return voxel_grid[:, y_low:y_high, :].any(axis=1)


def compute_scene_metrics(gen_root, gen_joints, scene_name, scene_dir,
                          voxel_size=0.1, grid_origin=(0.0, 0.0, 0.0)):
    """
    Compute scene collision metrics.

    Args:
        gen_root: (T, 3) generated root positions.
        gen_joints: (T, 22, 3) generated joint positions.
        scene_name: scene identifier.
        scene_dir: path to scene directory.

    Returns:
        Dict of metrics.
    """
    voxel_grid = load_scene_voxel(scene_name, scene_dir)
    if voxel_grid is None:
        return {
            "CollisionFrameRate": float("nan"),
            "NonWalkableRootRate": float("nan"),
            "PenetrationRate": float("nan"),
            "PenetrationMean": float("nan"),
            "PenetrationMax": float("nan"),
            "SceneSDFPenalty": float("nan"),
        }

    # Build 2D occupancy and SDF
    occ_2d = build_2d_occupancy(voxel_grid, voxel_size)
    dist_outside = distance_transform_edt(~occ_2d).astype(np.float32) * voxel_size
    dist_inside = distance_transform_edt(occ_2d).astype(np.float32) * voxel_size
    sdf_2d = dist_outside - dist_inside

    X, Z = sdf_2d.shape
    T = gen_root.shape[0]

    # Sample SDF at root positions
    root_xz = gen_root[:, [0, 2]]  # (T, 2)
    root_sdf = np.zeros(T)
    for t in range(T):
        ix = int((root_xz[t, 0] - grid_origin[0]) / voxel_size)
        iz = int((root_xz[t, 1] - grid_origin[2]) / voxel_size)
        if 0 <= ix < X and 0 <= iz < Z:
            root_sdf[t] = sdf_2d[ix, iz]
        else:
            root_sdf[t] = -1.0  # Outside grid = obstacle

    # NonWalkableRootRate: fraction of frames where root is inside obstacle
    non_walkable = root_sdf < 0
    non_walkable_rate = non_walkable.mean()

    # CollisionFrameRate: fraction of frames where any joint is inside obstacle
    collision_frames = np.zeros(T, dtype=bool)
    penetration_depths = []

    for t in range(T):
        joints_xz = gen_joints[t, :, [0, 2]]  # (22, 2)
        for j in range(joints_xz.shape[0]):
            ix = int((joints_xz[j, 0] - grid_origin[0]) / voxel_size)
            iz = int((joints_xz[j, 1] - grid_origin[2]) / voxel_size)
            if 0 <= ix < X and 0 <= iz < Z:
                joint_sdf = sdf_2d[ix, iz]
                if joint_sdf < 0:
                    collision_frames[t] = True
                    penetration_depths.append(-joint_sdf)

    collision_frame_rate = collision_frames.mean()

    if penetration_depths:
        penetration_rate = len(penetration_depths) / (T * 22)
        penetration_mean = np.mean(penetration_depths)
        penetration_max = np.max(penetration_depths)
    else:
        penetration_rate = 0.0
        penetration_mean = 0.0
        penetration_max = 0.0

    # SceneSDFPenalty: mean of relu(margin - sdf) at root
    margin = 0.10
    sdf_penalty = np.maximum(margin - root_sdf, 0) ** 2
    scene_sdf_penalty = sdf_penalty.mean()

    return {
        "CollisionFrameRate": collision_frame_rate,
        "NonWalkableRootRate": non_walkable_rate,
        "PenetrationRate": penetration_rate,
        "PenetrationMean": penetration_mean,
        "PenetrationMax": penetration_max,
        "SceneSDFPenalty": scene_sdf_penalty,
    }


def main():
    parser = argparse.ArgumentParser(description="SceneAdapt metrics evaluation")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--method", type=str, default="path_scene_guidance")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pred_dir = Path(args.pred_dir)
    npz_files = sorted(pred_dir.glob("sample_*.npz"))
    log.info(f"Found {len(npz_files)} samples in {pred_dir}")

    all_metrics = []
    for npz_file in npz_files:
        data = np.load(str(npz_file), allow_pickle=True)
        gen_root = data["gen_root"]  # (T, 3)
        gen_joints = data["gen_joints"]  # (T, 22, 3)
        scene_name = str(data.get("scene_name", ""))

        if not scene_name:
            log.warning(f"  SKIP {npz_file.name}: no scene_name")
            continue

        metrics = compute_scene_metrics(gen_root, gen_joints, scene_name, args.scene_dir)
        metrics["sample_id"] = npz_file.stem
        metrics["method"] = args.method
        all_metrics.append(metrics)

    # Write CSV
    if all_metrics:
        fieldnames = ["sample_id", "method", "CollisionFrameRate", "NonWalkableRootRate",
                      "PenetrationRate", "PenetrationMean", "PenetrationMax", "SceneSDFPenalty"]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

        # Print summary
        for key in ["CollisionFrameRate", "NonWalkableRootRate", "PenetrationRate",
                     "PenetrationMean", "PenetrationMax", "SceneSDFPenalty"]:
            vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if vals:
                log.info(f"  {key}: {np.mean(vals):.4f}")

    log.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
