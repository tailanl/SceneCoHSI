import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "kimodo"))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.definitions import SMPLXSkeleton22

SMPLX_22_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

SPINE_BONES = {(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)}
LEFT_LEG_BONES = {(0, 1), (1, 4), (4, 7), (7, 10)}
RIGHT_LEG_BONES = {(0, 2), (2, 5), (5, 8), (8, 11)}
LEFT_ARM_BONES = {(9, 13), (13, 16), (16, 18), (18, 20)}
RIGHT_ARM_BONES = {(9, 14), (14, 17), (17, 19), (19, 21)}

BONE_COLORS = {
    "spine": "#FF8C00",
    "left_leg": "#4169E1",
    "right_leg": "#DC143C",
    "left_arm": "#32CD32",
    "right_arm": "#FF69B4",
    "head": "#FFD700",
}


def get_bone_color(i, j):
    bone = (i, j)
    if bone in SPINE_BONES: return BONE_COLORS["spine"]
    if bone in LEFT_LEG_BONES: return BONE_COLORS["left_leg"]
    if bone in RIGHT_LEG_BONES: return BONE_COLORS["right_leg"]
    if bone in LEFT_ARM_BONES: return BONE_COLORS["left_arm"]
    if bone in RIGHT_ARM_BONES: return BONE_COLORS["right_arm"]
    return "#AAAAAA"


def load_motion_rep(checkpoint_dir):
    skeleton = SMPLXSkeleton22()
    stats_path = Path(checkpoint_dir) / "Kimodo-SMPLX-RP-v1" / "stats" / "motion"
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=30, stats_path=str(stats_path))
    return motion_rep


def load_global_root_stats(checkpoint_dir):
    stats_dir = Path(checkpoint_dir) / "Kimodo-SMPLX-RP-v1" / "stats" / "motion" / "global_root"
    mean = np.load(stats_dir / "mean.npy")
    std = np.load(stats_dir / "std.npy")
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def load_sample(cache_dir, idx):
    cache_path = Path(cache_dir) / f"seg_{idx:05d}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    return np.load(str(cache_path), allow_pickle=True)


def denormalize_root(normalized, mean, std):
    return normalized * std + mean


def extract_scene_pts(voxel_grid, voxel_size, cell_size=0.125):
    V = voxel_size[0]
    occupied = np.argwhere(voxel_grid > 0.5)
    if len(occupied) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    pts = occupied.astype(np.float32) * cell_size
    pts[:, 0] -= (V * cell_size) / 2
    pts[:, 2] -= (V * cell_size) / 2
    heights = pts[:, 1]
    return pts, heights


def main():
    parser = argparse.ArgumentParser(description="Visualize motion + root trajectory + scene")
    parser.add_argument("--smplx_cache", type=str,
                        default="lingo_smplx_cache",
                        help="Path to lingo_smplx_cache directory")
    parser.add_argument("--traj_cache", type=str,
                        default="lingo_root_trajectory_smplx",
                        help="Path to lingo_root_trajectory_smplx directory")
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/viz_motion_traj_v3")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--voxel_size", type=int, nargs=3, default=[64, 64, 64])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading motion representation...")
    motion_rep = load_motion_rep(args.checkpoint_dir)

    mean_root, std_root = load_global_root_stats(args.checkpoint_dir)
    print(f"Root stats: mean={mean_root[:3]}, std={std_root[:3]}")

    for sample_idx in range(args.n_samples):
        seg_idx = args.idx + sample_idx

        print(f"\n=== Sample {seg_idx} ===")

        try:
            data_full = load_sample(args.smplx_cache, seg_idx)
        except FileNotFoundError:
            print(f"  smplx_cache seg_{seg_idx:05d} not found, skipping")
            continue

        try:
            data_traj = load_sample(args.traj_cache, seg_idx)
        except FileNotFoundError:
            print(f"  traj_cache seg_{seg_idx:05d} not found, skipping")
            continue

        motion_features = torch.from_numpy(data_full["motion_features"]).float().unsqueeze(0)
        T = min(motion_features.shape[1], args.max_frames)
        motion_features = motion_features[:, :T]

        traj_norm = data_traj["global_root_features"][:T, :3]
        traj_raw = denormalize_root(traj_norm, mean_root[:3], std_root[:3])

        voxel = data_full.get("voxel_grid", data_traj.get("voxel_grid", None))

        with torch.no_grad():
            inv = motion_rep.inverse(
                motion_features, is_normalized=True,
                posed_joints_from="positions", return_numpy=True,
            )

        posed_joints = inv["posed_joints"][0].copy()
        smooth_root_pos = inv["smooth_root_pos"][0]

        posed_joints[..., 1] += smooth_root_pos[..., None, 1]

        root_positions = smooth_root_pos.copy()
        trail_root = root_positions.copy()

        pelvis_y = posed_joints[:, 0, 1]
        print(f"  Pelvis Y: [{pelvis_y.min():.2f}, {pelvis_y.max():.2f}]")
        print(f"  Smooth Y: [{smooth_root_pos[:, 1].min():.2f}, {smooth_root_pos[:, 1].max():.2f}]")
        print(f"  Y diff mean: {np.abs(smooth_root_pos[:, 1] - pelvis_y).mean():.4f}")

        scene = str(data_full.get("scene_name", "unknown"))
        text = str(data_full.get("text", ""))
        print(f"  Scene: {scene}")
        print(f"  Text: {text}")
        print(f"  Frames: {T}")

        if voxel is not None:
            scene_pts, scene_heights = extract_scene_pts(voxel, tuple(args.voxel_size))
            print(f"  Scene voxel occupied: {len(scene_pts)} points")
        else:
            scene_pts, scene_heights = None, None

        T_ds = T // args.downsample

        joints_ds = posed_joints[::args.downsample]
        all_x = joints_ds[..., 0].flatten()
        all_z = joints_ds[..., 2].flatten()
        all_y = joints_ds[..., 1].flatten()

        if scene_pts is not None and len(scene_pts) > 0:
            all_x = np.concatenate([all_x, scene_pts[:, 0]])
            all_z = np.concatenate([all_z, scene_pts[:, 1]])
            all_y = np.concatenate([all_y, scene_pts[:, 2]])

        all_x = all_x[np.isfinite(all_x)]
        all_z = all_z[np.isfinite(all_z)]
        all_y = all_y[np.isfinite(all_y)]
        if len(all_x) == 0:
            continue
        x_range = all_x.max() - all_x.min() + 0.5
        z_range = all_z.max() - all_z.min() + 0.5
        x_mid = (all_x.max() + all_x.min()) / 2
        z_mid = (all_z.max() + all_z.min()) / 2
        y_min = all_y.min() - 0.2
        y_max = all_y.max() + 0.5
        view_range = max(x_range, z_range)
        if not np.isfinite(view_range) or view_range <= 0:
            view_range = 2.0

        print(f"  Rendering animation ({T_ds} frames)...")

        fig = plt.figure(figsize=(14, 10), dpi=args.dpi, facecolor="white")
        ax = fig.add_subplot(111, projection="3d", facecolor="white")

        ax.grid(False)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.xaxis.line.set_color("black")
        ax.yaxis.line.set_color("black")
        ax.zaxis.line.set_color("black")

        scene_artists = None
        scene_colors = None
        scene_alphas = None
        if scene_pts is not None and len(scene_pts) > 0:
            step = max(1, len(scene_pts) // 8000)
            scene_sample = scene_pts[::step]
            h_sample = scene_heights[::step]
            colors = np.zeros((len(scene_sample), 3))
            alphas = np.zeros(len(scene_sample))
            floor_mask = h_sample < 0.15
            colors[floor_mask] = [0.70, 0.65, 0.60]
            alphas[floor_mask] = 0.08
            mid_mask = (h_sample >= 0.15) & (h_sample < 1.5)
            colors[mid_mask] = [0.75, 0.72, 0.68]
            alphas[mid_mask] = 0.06
            high_mask = h_sample >= 1.5
            colors[high_mask] = [0.85, 0.83, 0.80]
            alphas[high_mask] = 0.04
            scene_artists = (scene_sample[:, 0], scene_sample[:, 1], scene_sample[:, 2])
            scene_colors = np.concatenate([colors, alphas[:, None]], axis=1)

        def draw_frame(frame_idx, ax):
            f = frame_idx * args.downsample
            joints = posed_joints[f]

            if scene_artists is not None:
                ax.scatter(
                    scene_artists[0], scene_artists[1], scene_artists[2],
                    c=scene_colors, s=2, rasterized=True,
                )

            for (i, j) in SMPLX_22_CONNECTIONS:
                color = get_bone_color(i, j)
                ax.plot(
                    [joints[i, 0], joints[j, 0]],
                    [joints[i, 2], joints[j, 2]],
                    [joints[i, 1], joints[j, 1]],
                    color=color, linewidth=2.5, alpha=0.85,
                )

            ax.scatter(
                joints[:, 0], joints[:, 2], joints[:, 1],
                c="#ffffff", s=8, alpha=0.7,
            )

            trail_x = trail_root[:f+1, 0]
            trail_z = trail_root[:f+1, 2]
            trail_y = trail_root[:f+1, 1]

            ax.plot(trail_x, trail_z, trail_y,
                    color="#f39c12", linewidth=2.0, alpha=0.85, label="Root Trail")

            ax.scatter(
                trail_root[f, 0], trail_root[f, 2], trail_root[f, 1],
                c="#e74c3c", s=60, marker="o", edgecolors="white", linewidth=0.8,
                zorder=10, label="Current",
            )

            ax.scatter(
                trail_x[0], trail_z[0], trail_y[0],
                c="#2ecc71", s=100, marker="*", edgecolors="white", linewidth=0.8,
                zorder=11, label="Start",
            )

        def update(frame):
            f = frame * args.downsample
            ax.clear()
            ax.set_facecolor("white")
            ax.grid(False)
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
            draw_frame(frame, ax)
            ax.set_xlim(x_mid - view_range / 2 - 0.3, x_mid + view_range / 2 + 0.3)
            ax.set_zlim(z_mid - view_range / 2 - 0.3, z_mid + view_range / 2 + 0.3)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("X (m)", color="black")
            ax.set_ylabel("Z (m)", color="black")
            ax.set_zlabel("Y (up, m)", color="black")
            ax.tick_params(colors="black")
            label = f"Scene: {scene} | {text[:70]}"
            ax.set_title(label, fontsize=10, color="black")
            ax.view_init(elev=30, azim=-55)
            ax.legend(loc="upper left", fontsize=8, facecolor="white", labelcolor="black")
            ax.text2D(0.02, 0.02, f"Frame {f}/{T}",
                      transform=ax.transAxes, fontsize=9, color="black")
            return []

        ani = FuncAnimation(fig, update, frames=T_ds, interval=40, blit=False)

        out_path = output_dir / f"sample_{seg_idx:05d}.mp4"
        ani.save(str(out_path), writer="ffmpeg", fps=25, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone! Outputs in {output_dir}")


if __name__ == "__main__":
    main()
