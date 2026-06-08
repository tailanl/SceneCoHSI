"""Visualize guided root trajectories vs target path + scene occupancy.

For each sample, generates:
1. top-down view: target path (dashed), generated root (solid), scene SDF contour
2. heading arrows every N frames
3. non-walkable frames marked red
4. root XZ and heading time series

Usage:
    python scripts/visualize_guided_root.py \
        --pred_dir outputs/guidance_path_only \
        --scene_dir LINGO/dataset/dataset/Scene \
        --output_dir outputs/guided_root_viz \
        --max_samples 10
"""

import argparse, logging, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

log = logging.getLogger(__name__)


def build_scene_sdf(scene_name, scene_dir, voxel_size=0.1, grid_origin=(0.0, 0.0, 0.0),
                    root_height=1.0, height_tol=0.5):
    """Build 2D SDF from scene voxel grid."""
    scene_path = Path(scene_dir) / scene_name
    if not scene_path.exists():
        return None, None

    for fname in ["semantic_voxel_grid.npy", "voxel_grid.npy"]:
        fpath = scene_path / fname
        if fpath.exists():
            voxel_grid = np.load(str(fpath))
            break
    else:
        return None, None

    if voxel_grid.dtype != bool:
        voxel_grid = voxel_grid > 0.5

    X, Y, Z = voxel_grid.shape
    y_low = max(0, int((root_height - height_tol - grid_origin[1]) / voxel_size))
    y_high = min(Y, int((root_height + height_tol - grid_origin[1]) / voxel_size))

    if y_low >= y_high:
        occ_2d = np.zeros((X, Z), dtype=bool)
    else:
        occ_2d = voxel_grid[:, y_low:y_high, :].any(axis=1)

    dist_out = distance_transform_edt(~occ_2d).astype(np.float32) * voxel_size
    dist_in = distance_transform_edt(occ_2d).astype(np.float32) * voxel_size
    sdf = dist_out - dist_in

    return sdf, occ_2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--arrow_every", type=int, default=10, help="Show heading arrow every N frames")
    parser.add_argument("--plot_time_series", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.pred_dir)
    npz_files = sorted(pred_dir.glob("sample_*.npz"))[:args.max_samples]

    for npz_file in npz_files:
        data = np.load(str(npz_file), allow_pickle=True)
        gen_root = data["gen_root"]           # (T, 3)
        gt_root_xz = data.get("gt_root_xz")   # (T, 2)
        scene_name = str(data.get("scene_name", ""))
        text = str(data.get("text", ""))

        T = gen_root.shape[0]

        # Build scene SDF
        scene_sdf, scene_occ = None, None
        if scene_name:
            scene_sdf, scene_occ = build_scene_sdf(scene_name, args.scene_dir)

        gen_xz = gen_root[:, [0, 2]]  # (T, 2)

        # Determine non-walkable frames
        non_walkable = np.zeros(T, dtype=bool)
        if scene_sdf is not None:
            X, Z = scene_sdf.shape
            for t in range(T):
                ix = int(gen_xz[t, 0] / 0.1)
                iz = int(gen_xz[t, 1] / 0.1)
                if 0 <= ix < X and 0 <= iz < Z:
                    if scene_sdf[ix, iz] < 0:
                        non_walkable[t] = True

        # Compute heading from gen_root XZ
        vel = gen_xz[1:] - gen_xz[:-1]
        heading_theta = np.arctan2(vel[:, 1], vel[:, 0])
        heading_theta = np.concatenate([heading_theta, heading_theta[-1:]])

        fig, axes = plt.subplots(1, 2 if args.plot_time_series else 1,
                                figsize=(16 if args.plot_time_series else 8, 7),
                                squeeze=False)
        ax = axes[0, 0]

        # Scene background
        if scene_occ is not None:
            extent = [0, scene_occ.shape[1] * 0.1, 0, scene_occ.shape[0] * 0.1]
            ax.imshow(scene_occ.T, origin="lower", extent=extent, cmap="gray_r", alpha=0.3)
            if scene_sdf is not None:
                # SDF contour at 0
                cs = ax.contour(scene_sdf.T, levels=[0], origin="lower",
                               extent=extent, colors="red", linewidths=1.0, alpha=0.5)

        # Target path (GT)
        if gt_root_xz is not None:
            ax.plot(gt_root_xz[:, 0], gt_root_xz[:, 1], "b--", linewidth=1.5, alpha=0.7, label="GT target")

        # Generated root - normal frames (blue) and non-walkable (red)
        normal = ~non_walkable
        if normal.any():
            ax.plot(gen_xz[normal, 0], gen_xz[normal, 1], "g-", linewidth=2.0, label="Gen root")
        if non_walkable.any():
            ax.plot(gen_xz[non_walkable, 0], gen_xz[non_walkable, 1], "r.", markersize=4, label="Non-walkable")
            ax.scatter(gen_xz[non_walkable, 0], gen_xz[non_walkable, 1], c="red", s=10, zorder=5)

        # Heading arrows
        arrow_idx = np.arange(0, T, args.arrow_every)
        for ai in arrow_idx:
            dx = np.cos(heading_theta[ai]) * 0.3
            dy = np.sin(heading_theta[ai]) * 0.3
            ax.arrow(gen_xz[ai, 0], gen_xz[ai, 1], dx, dy,
                    head_width=0.15, head_length=0.2, fc="orange", ec="orange", alpha=0.6)

        # Start/end markers
        ax.scatter(gen_xz[0, 0], gen_xz[0, 1], c="green", s=80, marker="o", zorder=6, label="Start")
        ax.scatter(gen_xz[-1, 0], gen_xz[-1, 1], c="red", s=80, marker="x", zorder=6, label="End")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{npz_file.stem}: {text}\nnon-walkable={non_walkable.sum()}/{T}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Time series
        if args.plot_time_series:
            ax2 = axes[0, 1]
            ax2.plot(np.arange(T), gen_xz[:, 0], "b-", alpha=0.7, label="X")
            ax2.plot(np.arange(T), gen_xz[:, 1], "r-", alpha=0.7, label="Z")
            if gt_root_xz is not None:
                ax2.plot(np.arange(T), gt_root_xz[:, 0], "b--", alpha=0.4, label="GT X")
                ax2.plot(np.arange(T), gt_root_xz[:, 1], "r--", alpha=0.4, label="GT Z")
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Position (m)")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(output_dir / f"{npz_file.stem}.png"), dpi=150)
        plt.close(fig)

    log.info(f"Saved {len(npz_files)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
