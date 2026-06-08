#!/usr/bin/env python
"""可视化 lingo_smplx_cache 中的骨骼动作

从缓存的 .npz 文件中反归一化 motion features，还原关节位置，
使用 matplotlib 渲染 3D 骨骼动画并保存为视频。

用法:
    python kimodo_scene_project/scripts/visualize_smplx_cache.py
    python kimodo_scene_project/scripts/visualize_smplx_cache.py --segment 6
    python kimodo_scene_project/scripts/visualize_smplx_cache.py --segments 0,5,10,20
    python kimodo_scene_project/scripts/visualize_smplx_cache.py --segment 6 --output_dir viz_output
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")

SMPLX_22_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]

SMPLX_22_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

BONE_COLORS = {
    "spine": "#FF8C00",
    "left_leg": "#4169E1",
    "right_leg": "#DC143C",
    "left_arm": "#32CD32",
    "right_arm": "#FF69B4",
    "head": "#FFD700",
}

SPINE_BONES = {(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)}
LEFT_LEG_BONES = {(0, 1), (1, 4), (4, 7), (7, 10)}
RIGHT_LEG_BONES = {(0, 2), (2, 5), (5, 8), (8, 11)}
LEFT_ARM_BONES = {(9, 13), (13, 16), (16, 18), (18, 20)}
RIGHT_ARM_BONES = {(9, 14), (14, 17), (17, 19), (19, 21)}
HEAD_BONE = {(12, 15)}


def get_bone_color(i, j):
    bone = (i, j)
    if bone in SPINE_BONES:
        return BONE_COLORS["spine"]
    if bone in LEFT_LEG_BONES:
        return BONE_COLORS["left_leg"]
    if bone in RIGHT_LEG_BONES:
        return BONE_COLORS["right_leg"]
    if bone in LEFT_ARM_BONES:
        return BONE_COLORS["left_arm"]
    if bone in RIGHT_ARM_BONES:
        return BONE_COLORS["right_arm"]
    if bone in HEAD_BONE:
        return BONE_COLORS["head"]
    return "#AAAAAA"


def load_motion_rep():
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton import SMPLXSkeleton22

    model_dir = PROJECT_ROOT / "models" / "Kimodo-SMPLX-RP-v1"
    stats_path = model_dir / "stats" / "motion"
    skel = SMPLXSkeleton22()
    motion_rep = KimodoMotionRep(fps=30, stats_path=str(stats_path), skeleton=skel)
    return motion_rep, skel


def decode_joints_from_cache(npz_path: str, motion_rep):
    data = np.load(str(npz_path), allow_pickle=True)
    motion_features = torch.from_numpy(data["motion_features"]).float().unsqueeze(0)
    length = int(data["length"])

    with torch.no_grad():
        result = motion_rep.inverse(motion_features, is_normalized=True, posed_joints_from="positions")

    posed_joints = result["posed_joints"].squeeze(0).numpy()
    root_positions = result["root_positions"].squeeze(0).numpy()
    smooth_root_pos = result["smooth_root_pos"].squeeze(0).numpy()

    scene_name = str(data["scene_name"]) if "scene_name" in data else "unknown"
    text = str(data["text"]) if "text" in data else ""

    return {
        "posed_joints": posed_joints[:length],
        "root_positions": root_positions[:length],
        "smooth_root_pos": smooth_root_pos[:length],
        "length": length,
        "scene_name": scene_name,
        "text": text,
    }


def render_segment_video(seg: dict, output_path: str, fps: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    joints = seg["posed_joints"]
    root_pos = seg["smooth_root_pos"]
    num_frames = seg["length"]

    all_joints = joints.reshape(-1, 3)
    x_min, y_min, z_min = all_joints.min(axis=0)
    x_max, y_max, z_max = all_joints.max(axis=0)
    padding = 0.3
    x_min -= padding; x_max += padding
    y_min -= padding; y_max += padding
    z_min -= padding; z_max += padding

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range, z_max - z_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    fig = plt.figure(figsize=(12, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=100):
        for frame_idx in range(num_frames):
            ax.cla()

            pos = joints[frame_idx]

            for i, j in SMPLX_22_CONNECTIONS:
                color = get_bone_color(i, j)
                lw = 3.5 if (i, j) in SPINE_BONES else 2.5
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 2], pos[j, 2]],
                    [pos[i, 1], pos[j, 1]],
                    color=color, linewidth=lw, zorder=5,
                )

            ax.scatter(
                pos[:, 0], pos[:, 2], pos[:, 1],
                c="white", s=25, depthshade=False, zorder=10,
                edgecolors="#333333", linewidths=0.5,
            )

            ax.scatter(
                [pos[0, 0]], [pos[0, 2]], [pos[0, 1]],
                c="red", s=80, depthshade=False, zorder=15,
                edgecolors="white", linewidths=1.5,
            )

            trail_start = max(0, frame_idx - 50)
            trail = root_pos[trail_start:frame_idx + 1]
            if len(trail) >= 2:
                ax.plot(
                    trail[:, 0], trail[:, 2], trail[:, 1],
                    color="cyan", linewidth=1.5, alpha=0.7, zorder=3,
                )

            ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
            ax.set_ylim(z_min - 0.1, z_min + max_range - 0.1)
            ax.set_zlim(0, max_range)

            ax.set_xlabel("X", color="gray", fontsize=8)
            ax.set_ylabel("Z", color="gray", fontsize=8)
            ax.set_zlabel("Y (up)", color="gray", fontsize=8)

            ax.tick_params(colors="gray", labelsize=6)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("gray")
            ax.yaxis.pane.set_edgecolor("gray")
            ax.zaxis.pane.set_edgecolor("gray")
            ax.grid(True, alpha=0.2)

            ax.set_title(
                f"Scene: {seg['scene_name']}  |  \"{seg['text']}\"  |  "
                f"Frame {frame_idx}/{num_frames-1}",
                color="white", fontsize=11, pad=10,
            )

            ax.view_init(elev=25, azim=-60 + frame_idx * 0.3)

            writer.grab_frame()

            if frame_idx % 20 == 0:
                print(f"    Frame {frame_idx}/{num_frames-1}")

    plt.close(fig)


def render_segment_gif(seg: dict, output_path: str, fps: int = 15, skip: int = 2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    joints = seg["posed_joints"]
    root_pos = seg["smooth_root_pos"]
    num_frames = seg["length"]

    all_joints = joints.reshape(-1, 3)
    x_min, y_min, z_min = all_joints.min(axis=0)
    x_max, y_max, z_max = all_joints.max(axis=0)
    padding = 0.3
    x_min -= padding; x_max += padding
    y_min -= padding; y_max += padding
    z_min -= padding; z_max += padding

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_min + x_max) / 2

    fig = plt.figure(figsize=(10, 7), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    frames_to_render = list(range(0, num_frames, skip))
    writer = PillowWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=80):
        for render_idx, frame_idx in enumerate(frames_to_render):
            ax.cla()

            pos = joints[frame_idx]

            for i, j in SMPLX_22_CONNECTIONS:
                color = get_bone_color(i, j)
                lw = 3.5 if (i, j) in SPINE_BONES else 2.5
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 2], pos[j, 2]],
                    [pos[i, 1], pos[j, 1]],
                    color=color, linewidth=lw, zorder=5,
                )

            ax.scatter(
                pos[:, 0], pos[:, 2], pos[:, 1],
                c="white", s=20, depthshade=False, zorder=10,
                edgecolors="#333333", linewidths=0.5,
            )

            ax.scatter(
                [pos[0, 0]], [pos[0, 2]], [pos[0, 1]],
                c="red", s=60, depthshade=False, zorder=15,
                edgecolors="white", linewidths=1.0,
            )

            trail_start = max(0, frame_idx - 50)
            trail = root_pos[trail_start:frame_idx + 1]
            if len(trail) >= 2:
                ax.plot(
                    trail[:, 0], trail[:, 2], trail[:, 1],
                    color="cyan", linewidth=1.5, alpha=0.7, zorder=3,
                )

            ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
            ax.set_ylim(z_min - 0.1, z_min + max_range - 0.1)
            ax.set_zlim(0, max_range)

            ax.tick_params(colors="gray", labelsize=6)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("gray")
            ax.yaxis.pane.set_edgecolor("gray")
            ax.zaxis.pane.set_edgecolor("gray")
            ax.grid(True, alpha=0.2)

            ax.set_title(
                f"Scene: {seg['scene_name']}  |  \"{seg['text']}\"  |  "
                f"Frame {frame_idx}/{num_frames-1}",
                color="white", fontsize=10, pad=10,
            )

            ax.view_init(elev=25, azim=-60 + frame_idx * 0.3)
            writer.grab_frame()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="可视化 lingo_smplx_cache 骨骼动作")
    parser.add_argument("--cache_dir", type=str, default="lingo_smplx_cache")
    parser.add_argument("--segment", type=int, default=None,
                        help="指定单个片段索引")
    parser.add_argument("--segments", type=str, default=None,
                        help="指定多个片段索引，逗号分隔 (e.g., '0,5,10')")
    parser.add_argument("--num_segments", type=int, default=5,
                        help="加载的片段数量（默认5）")
    parser.add_argument("--output_dir", type=str, default="kimodo_scene_project/viz_output")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4"],
                        help="输出格式 (gif 或 mp4)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--skip", type=int, default=2,
                        help="GIF 模式下每隔几帧渲染一帧")
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=196)
    args = parser.parse_args()

    print("正在加载 motion representation...")
    motion_rep, skel = load_motion_rep()

    print(f"  SMPLX22: {skel.nbjoints} joints, motion_rep_dim={motion_rep.motion_rep_dim}")
    print(f"  root_slice: {motion_rep.root_slice} (global_root_dim={motion_rep.global_root_dim})")
    print(f"  body_slice: {motion_rep.body_slice} (body_dim={motion_rep.body_dim})")
    print(f"  local_root_dim: {motion_rep.local_root_dim}")
    print()

    cache_dir = Path(args.cache_dir)
    npz_files = sorted(cache_dir.glob("seg_*.npz"))
    print(f"找到 {len(npz_files)} 个缓存文件")

    target_indices = None
    if args.segment is not None:
        target_indices = [args.segment]
    elif args.segments is not None:
        target_indices = [int(x.strip()) for x in args.segments.split(",")]

    if target_indices is not None:
        selected_files = []
        for idx in target_indices:
            target_file = cache_dir / f"seg_{idx:05d}.npz"
            if target_file.exists():
                selected_files.append(target_file)
            else:
                print(f"  警告: 片段文件 {target_file} 不存在")
        npz_files = selected_files

    segments = []
    for i, f in enumerate(npz_files):
        if len(segments) >= (len(target_indices) if target_indices else args.num_segments):
            break
        try:
            seg = decode_joints_from_cache(str(f), motion_rep)
            if seg["length"] >= args.min_frames and seg["length"] <= args.max_frames:
                segments.append(seg)
                print(f"  [{len(segments)-1}] {f.name}: {seg['length']}帧, "
                      f"场景={seg['scene_name']}, 动作=\"{seg['text'][:50]}\"")
        except Exception as e:
            print(f"  跳过 {f.name}: {e}")

    if not segments:
        print("没有加载到任何片段!")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始渲染 {len(segments)} 个片段...")
    for i, seg in enumerate(segments):
        ext = "gif" if args.format == "gif" else "mp4"
        out_name = f"seg_{i:03d}_{seg['scene_name']}_{seg['text'][:20].replace(' ', '_')}.{ext}"
        out_path = output_dir / out_name

        print(f"\n[{i+1}/{len(segments)}] 渲染: {out_name}")
        print(f"  场景={seg['scene_name']}, 动作=\"{seg['text']}\", 帧数={seg['length']}")

        if args.format == "gif":
            render_segment_gif(seg, str(out_path), fps=args.fps, skip=args.skip)
        else:
            render_segment_video(seg, str(out_path), fps=args.fps)

        print(f"  -> 已保存: {out_path}")

    print(f"\n{'='*60}")
    print(f"  完成! {len(segments)} 个片段已渲染到 {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
