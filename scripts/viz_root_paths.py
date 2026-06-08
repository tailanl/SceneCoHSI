#!/usr/bin/env python
"""Visualize root paths from lingo_smplx_cache segments.

Generates PNG images showing:
1. Top-down XZ view of root trajectory
2. Y-height over time
3. Heading angle over time
"""

import argparse, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import einops

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton import SMPLXSkeleton22


def load_segments(cache_indices):
    cache_dir = PROJECT_ROOT / "lingo_smplx_cache"
    joints_file = PROJECT_ROOT / "LINGO/dataset/dataset/human_joints_aligned.npy"
    start_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/end_idx.npy")).flatten()

    count = 0
    seg_ranges = {}
    for i in range(len(start_idx)):
        si, ei = int(start_idx[i]), int(end_idx[i])
        if 40 <= ei - si <= 196:
            seg_ranges[count] = (si, ei)
            count += 1

    samples = []
    joints_all = np.load(str(joints_file), mmap_mode="r")
    for ci in cache_indices:
        cache_file = cache_dir / f"seg_{ci:05d}.npz"
        if not cache_file.exists():
            continue
        data = np.load(str(cache_file), allow_pickle=True)
        T = int(data["length"])
        s, e = seg_ranges.get(ci, (0, T))
        samples.append({
            "cache_idx": ci,
            "text": str(data.get("text", "")),
            "num_frames": T,
            "motion_features": data["motion_features"][:T].copy(),
            "scene_name": str(data.get("scene_name", "")),
            "gt_joints": joints_all[s:e, :22, :].copy(),
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_idx", type=int, nargs="+", default=[0, 2, 5, 8, 11])
    parser.add_argument("--output_dir", type=str, default="kimodo_scene_project/outputs/exp_root_paths")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skel = SMPLXSkeleton22()
    motion_rep = KimodoMotionRep(
        fps=30, stats_path="models/Kimodo-SMPLX-RP-v1/stats/motion", skeleton=skel)

    samples = load_segments(args.cache_idx)
    print(f"Loaded {len(samples)} samples")

    for si, sample in enumerate(samples):
        ci = sample["cache_idx"]
        text = sample["text"]
        T = sample["num_frames"]
        features = sample["motion_features"]
        feat_t = torch.from_numpy(features).float().unsqueeze(0)

        # Unnormalize to get real root
        unnorm = motion_rep.unnormalize(feat_t)
        s_root, g_heading, *_ = einops.unpack(unnorm, motion_rep.ps, "b t *")
        root_np = s_root[0].cpu().numpy()  # [T, 3] X,Y,Z
        heading_np = g_heading[0].cpu().numpy()  # [T]

        # Root trail XZ top-down
        fig = plt.figure(figsize=(16, 5), dpi=150, facecolor="white")

        # Panel 1: XZ top-down
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(root_np[:, 0], root_np[:, 2], linewidth=2, color="#2196F3", label="Root XZ")
        ax1.scatter(root_np[0, 0], root_np[0, 2], c="#4CAF50", s=120, marker="o", label="Start", zorder=5, edgecolors="white", linewidth=1.5)
        ax1.scatter(root_np[-1, 0], root_np[-1, 2], c="#F44336", s=120, marker="s", label="End", zorder=5, edgecolors="white", linewidth=1.5)
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
        ax1.set_title("Root Trajectory (XZ Top-Down)")
        ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        # Panel 2: Y height
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(np.arange(T), root_np[:, 1], linewidth=2, color="#FF9800")
        ax2.set_xlabel("Frame"); ax2.set_ylabel("Y (m)")
        ax2.set_title("Root Height Over Time")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Heading
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(np.arange(T), np.degrees(heading_np), linewidth=2, color="#9C27B0")
        ax3.set_xlabel("Frame"); ax3.set_ylabel("Heading (deg)")
        ax3.set_title("Root Heading Over Time")
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f"seg_{ci:05d}: \"{text}\"  (T={T})", fontsize=11, fontweight="bold")
        fig.tight_layout()
        out_path = out_dir / f"root_{ci:05d}.png"
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")

    # Now also show: what features are constrained in root+init_pose mode
    print(f"\nRoot slice: {motion_rep.root_slice}")
    print(f"Body slice: {motion_rep.body_slice}")
    print(f"  global_root_dim={motion_rep.global_root_dim}")
    print(f"  body_dim={motion_rep.body_dim}")
    print(f"  local_root_dim={motion_rep.local_root_dim}")

    # Show: with root+init_pose, only frame 0 body features are constrained
    # That means: all frames have root constrained, only frame 0 has body constrained
    print(f"\nConstraint pattern:")
    print(f"  Root (all frames): dims {motion_rep.root_slice} = {motion_rep.global_root_dim} dims (smooth_root_pos X,Y,Z + heading)")
    print(f"  Body (frame 0 only): dims {motion_rep.body_slice} = {motion_rep.body_dim} dims (local_root + body_pose)")

    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
