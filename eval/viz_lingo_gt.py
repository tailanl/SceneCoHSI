#!/usr/bin/env python
"""Visualize LINGO original motion data directly from human_joints_aligned.npy.
Uses SMPL skeleton connections to render 3D human poses from ground truth.
No SOMA, no Kimodo FK - raw joint positions from the dataset.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent

# SMPL body skeleton connections (22 main body joints)
SMPL_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),           # pelvis → hips + spine
    (1, 4), (4, 7), (7, 10),          # left leg
    (2, 5), (5, 8), (8, 11),          # right leg
    (3, 6), (6, 9), (9, 12),          # spine → neck
    (12, 13), (13, 16), (16, 18), (18, 20),  # left arm
    (12, 14), (14, 17), (17, 19), (19, 21),  # right arm
    (12, 15),                          # neck → head
]

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]


def draw_skeleton(ax, joints, fi, color, s=40):
    pos = joints[fi]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c=color, s=s, depthshade=False, zorder=10,
               edgecolors="white", linewidths=0.5)
    for a, b in SMPL_CONNECTIONS:
        if a < pos.shape[0] and b < pos.shape[0]:
            ax.plot([pos[a, 0], pos[b, 0]],
                    [pos[a, 1], pos[b, 1]],
                    [pos[a, 2], pos[b, 2]],
                    color=color, linewidth=2.5, zorder=8)


def main():
    # Load aligned joints
    joints_all = np.load(
        str(PROJECT_ROOT / "LINGO/dataset/dataset/human_joints_aligned.npy"),
        mmap_mode="r"
    )
    start_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/end_idx.npy")).flatten()

    import pickle
    with open(str(PROJECT_ROOT / "LINGO/dataset/dataset/scene_name.pkl"), "rb") as f:
        scene_names = pickle.load(f)
    with open(str(PROJECT_ROOT / "LINGO/dataset/dataset/text_aug.pkl"), "rb") as f:
        text_aug = pickle.load(f)

    n_segments = len(start_idx)
    print(f"Total segments: {n_segments}")
    print(f"Joints shape: {joints_all.shape}")

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_segments, size=5, replace=False)

    output_dir = PROJECT_ROOT / "kimodo_scene_project/outputs/dataset_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    for vi, si in enumerate(sample_indices):
        s = start_idx[si]; e = end_idx[si]
        n_frames = e - s
        joints = joints_all[s:e, :22]  # First 22 SMPL body joints
        scene_name = scene_names[si]

        texts = text_aug[si] if si < len(text_aug) else ["motion"]
        text = texts[0] if isinstance(texts, list) and texts else str(texts)

        print(f"\nSample {vi}: scene={scene_name}, frames={n_frames}, text={text[:60]}")

        # Check proportions
        pelv_y = joints[0, 0, 1]
        head_y = joints[0, 15, 1]
        l_foot_y = joints[0, 10, 1]
        height = max(head_y, joints[0, 15, 1]) - min(pelv_y, l_foot_y,
                                                       joints[0, 11, 1])
        print(f"  Pelvis Y={pelv_y:.2f}, Head Y={head_y:.2f}, Height≈{height:.2f}m")

        # Scale to visualization units
        j_viz = joints * 100  # METER_TO_UNIT

        frames_to_show = min(10, n_frames)
        step = max(1, n_frames // frames_to_show)
        fig, axes = plt.subplots(2, 5, figsize=(22, 10),
                                 subplot_kw={"projection": "3d"}, facecolor="white")
        axes = axes.flatten()

        all_pts = j_viz.reshape(-1, 3)
        center = np.mean(all_pts, axis=0)
        spread = max(np.max(np.abs(all_pts - center)), 50)

        for pi in range(frames_to_show):
            fi = min(pi * step, n_frames - 1)
            ax = axes[pi]
            ax.set_facecolor("#FAFAFA")
            draw_skeleton(ax, j_viz, fi, "#4CAF50", s=30)
            ax.set_xlim(center[0] - spread, center[0] + spread)
            ax.set_ylim(center[1] - spread * 0.4, center[1] + spread * 0.8)
            ax.set_zlim(center[2] - spread * 0.6, center[2] + spread * 0.6)
            ax.view_init(elev=15, azim=-45)
            ax.set_title(f"t={fi}/{n_frames-1}", fontsize=9)
            ax.set_axis_off()

        fig.suptitle(f"LINGO GT: {scene_name}\n{text}\n{n_frames} frames, SMPL skeleton",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        out_path = output_dir / f"lingo_gt_{vi}_{scene_name}.png"
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()