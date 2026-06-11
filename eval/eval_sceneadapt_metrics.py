"""LINGO-standard scene collision evaluation.

Based on LINGO dataset (SIGGRAPH Asia 2024):
  https://github.com/mileret/lingo-release

Scene grid definition from datasets/lingo.py:
  train: [-3, 0, -4, 3, 2, 4, 300, 100, 400]  → X∈[-3,3], Z∈[-4,4], voxel=0.02m
  test:  [-4, 0, -6, 4, 2, 6, 400, 100, 600]  → X∈[-4,4], Z∈[-6,6], voxel=0.02m

Critical: points outside grid are treated as OCCUPIED (collision).
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt

PROJECT_DIR = Path(__file__).resolve().parent.parent

# LINGO scene grid definitions (from datasets/lingo.py)
# Format: [x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz]
LINGO_GRIDS = {
    (300, 100, 400): np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400]),
    (400, 100, 600): np.array([-4, 0, -6, 4, 2, 6, 400, 100, 600]),
}


def get_grid_params(scene_shape):
    """Get LINGO grid parameters for a given scene shape."""
    key = tuple(int(s) for s in scene_shape)
    if key in LINGO_GRIDS:
        g = LINGO_GRIDS[key]
        return {
            "x_min": float(g[0]), "y_min": float(g[1]), "z_min": float(g[2]),
            "x_max": float(g[3]), "y_max": float(g[4]), "z_max": float(g[5]),
            "nx": int(g[6]), "ny": int(g[7]), "nz": int(g[8]),
        }
    # Fallback: compute from shape assuming voxel_size=0.02
    vs = 0.02
    return {
        "x_min": -scene_shape[0] * vs / 2, "y_min": 0.0, "z_min": -scene_shape[2] * vs / 2,
        "x_max": scene_shape[0] * vs / 2, "y_max": scene_shape[1] * vs,
        "z_max": scene_shape[2] * vs / 2,
        "nx": int(scene_shape[0]), "ny": int(scene_shape[1]), "nz": int(scene_shape[2]),
    }


def load_scene_occ(scene_name, scene_dir=None):
    """Load raw scene occupancy grid from LINGO Scene directory."""
    if scene_dir is None:
        scene_dir = PROJECT_DIR / "LINGO" / "dataset" / "dataset" / "Scene"
    scene_dir = Path(scene_dir)
    
    for suffix in [scene_name, scene_name.split("-")[0] if "-" in scene_name else scene_name]:
        path = scene_dir / f"{suffix}.npy"
        if path.exists():
            return np.load(str(path))
    return None


def compute_lingo_scene_metrics(gen_root, gen_joints, scene_name, scene_dir=None, floor_ignore_height=0.08):
    """Compute collision metrics using LINGO's scene grid definition.
    
    Key rules (from datasets/lingo.py get_occ_for_points):
    - Points inside grid: check occupancy at 3D voxel position
    - Points outside grid: treated as OCCUPIED (collision)
    - floor_ignore_height: ignore joints below this Y (feet on ground = false positive)
    - Scene voxel: True = occupied, False = free
    
    Args:
        gen_root: (T, 3) root positions in meters
        gen_joints: (T, J, 3) joint positions in meters
        scene_name: scene identifier
        scene_dir: path to Scene directory
    
    Returns:
        dict with CollisionFrameRate, PenetrationRate, etc.
    """
    occ = load_scene_occ(scene_name, scene_dir)
    if occ is None:
        return {k: float("nan") for k in ["CollisionFrameRate", "NonWalkableRootRate",
                 "PenetrationRate", "PenetrationMean", "PenetrationMax", "SceneSDFPenalty"]}
    
    grid = get_grid_params(occ.shape)
    x_min, y_min, z_min = grid["x_min"], grid["y_min"], grid["z_min"]
    x_max, y_max, z_max = grid["x_max"], grid["y_max"], grid["z_max"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    
    T = gen_root.shape[0]
    J = gen_joints.shape[1]
    
    vs_x = (x_max - x_min) / nx
    vs_y = (y_max - y_min) / ny
    vs_z = (z_max - z_min) / nz
    
    def world_to_voxel_3d(x, y, z):
        """Convert world coords to voxel indices. Returns (ix, iy, iz, in_bounds)."""
        ix = int((x - x_min) / vs_x)
        iy = int((y - y_min) / vs_y)
        iz_ = int((z - z_min) / vs_z)
        inb = (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz_ < nz)
        return max(0, min(nx-1, ix)), max(0, min(ny-1, iy)), max(0, min(nz-1, iz_)), inb
    
    # Root collision (2D check for root since root is at ground level)
    occ_2d = occ[:, 0:1, :].any(axis=1)  # Only check Y=0 (floor level) for root
    def world_to_voxel_2d(x_w, z_w):
        ix = int((x_w - x_min) / vs_x)
        iz_ = int((z_w - z_min) / vs_z)
        inb = 0 <= ix < nx and 0 <= iz_ < nz
        return max(0, min(nx-1, ix)), max(0, min(nz-1, iz_)), inb
    
    # Root collision: 3D voxel lookup at actual root Y height
    root_in_obstacle = np.zeros(T, dtype=bool)
    for t in range(T):
        ix, iy, iz, inb = world_to_voxel_3d(gen_root[t, 0], gen_root[t, 1], gen_root[t, 2])
        if not inb:
            root_in_obstacle[t] = True
        else:
            root_in_obstacle[t] = occ[ix, iy, iz]
    non_walkable_rate = float(root_in_obstacle.mean())
    
    # Joint collision: 3D voxel lookup (LINGO-style), skip floor-level joints
    collision_frames = np.zeros(T, dtype=bool)
    penetration_depths = []
    ignored_floor = 0
    total_joints = 0
    
    for t in range(T):
        for j in range(J):
            x, y, z = gen_joints[t, j, 0], gen_joints[t, j, 1], gen_joints[t, j, 2]
            total_joints += 1
            # Floor contact filter: ignore joints very close to ground
            if y < floor_ignore_height:
                ignored_floor += 1
                continue
            ix, iy, iz, inb = world_to_voxel_3d(x, y, z)
            if not inb:
                collision_frames[t] = True
            elif occ[ix, iy, iz]:
                collision_frames[t] = True
                penetration_depths.append(1.0)
    
    cfr = float(collision_frames.mean())
    
    if penetration_depths:
        pen_rate = len(penetration_depths) / max(1, total_joints - ignored_floor)
        pen_mean = float(np.mean(penetration_depths))
        pen_max = float(np.max(penetration_depths))
    else:
        pen_rate = 0.0
        pen_mean = 0.0
        pen_max = 0.0
    
    # SDF penalty at root (3D)
    root_sdf_vals = []
    for t in range(T):
        ix, iy, iz, inb = world_to_voxel_3d(gen_root[t, 0], gen_root[t, 1], gen_root[t, 2])
        if inb:
            root_sdf_vals.append(-1.0 if occ[ix, iy, iz] else 1.0)
        else:
            root_sdf_vals.append(-1.0)
    root_sdf = np.array(root_sdf_vals)
    margin = 0.10
    sdf_penalty = float(np.maximum(margin - root_sdf, 0).mean() ** 2)
    
    return {
        "CollisionFrameRate": cfr,
        "NonWalkableRootRate": non_walkable_rate,
        "PenetrationRate": pen_rate,
        "PenetrationMean": pen_mean,
        "PenetrationMax": pen_max,
        "SceneSDFPenalty": sdf_penalty,
    }

def main():
    import argparse, csv, logging
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="LINGO Scene metrics evaluation")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--method", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pred_dir = Path(args.pred_dir)
    npz_files = sorted(pred_dir.glob("sample_*.npz"))
    if not npz_files:
        npz_files = sorted(pred_dir.glob("seg_*.npz"))
    if not npz_files:
        npz_files = sorted(pred_dir.glob("*.npz"))
    log.info(f"Found {len(npz_files)} samples in {pred_dir}")

    all_metrics = []
    for npz_file in npz_files:
        data = np.load(str(npz_file), allow_pickle=True)
        gen_root = np.array(data["gen_root"], dtype=np.float32)
        gen_joints = np.array(data["gen_joints"], dtype=np.float32)
        scene_name = str(data.get("scene_name", ""))

        if not scene_name or scene_name == "None":
            continue

        metrics = compute_lingo_scene_metrics(gen_root, gen_joints, scene_name, args.scene_dir)
        metrics["sample_id"] = npz_file.stem
        metrics["method"] = args.method
        all_metrics.append(metrics)

    if all_metrics:
        fieldnames = ["sample_id", "method", "CollisionFrameRate", "NonWalkableRootRate",
                      "PenetrationRate", "PenetrationMean", "PenetrationMax", "SceneSDFPenalty"]
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

        for key in ["CollisionFrameRate", "NonWalkableRootRate", "PenetrationRate",
                     "PenetrationMean", "PenetrationMax", "SceneSDFPenalty"]:
            vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if vals:
                log.info(f"  {key}: {np.mean(vals):.4f}")

    log.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
