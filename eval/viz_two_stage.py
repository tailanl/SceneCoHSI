"""Visualize two-stage inference results: MP4 video + root trajectory comparison.

Loads npz outputs from two_stage_inference.py and renders:
  1. Side-by-side 3D skeleton animation MP4 (Stage1 | Stage2)
  2. 2D top-down root trajectory comparison PNG

Usage:
    python kimodo_scene_project/eval/viz_two_stage.py \
        --input_dir kimodo_scene_project/outputs/two_stage_inference \
        --output_dir kimodo_scene_project/outputs/two_stage_viz \
        --fps 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kimodo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SOMA"))
os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = PROJECT_ROOT / "LINGO" / "scene_mesh" / "Scene_mesh"
LINGO_DATASET_DIR = PROJECT_ROOT / "LINGO" / "dataset" / "dataset"

METER_TO_UNIT = 100.0

SOMA77_TO_SMPL22 = {
    0: 0,     # Hips → pelvis
    67: 1,    # LeftLeg → left_hip
    72: 2,    # RightLeg → right_hip
    1: 3,     # Spine1 → spine1
    68: 4,    # LeftShin → left_knee
    73: 5,    # RightShin → right_knee
    2: 6,     # Spine2 → spine2
    69: 7,    # LeftFoot → left_ankle
    74: 8,    # RightFoot → right_ankle
    3: 9,     # Chest → spine3
    70: 10,   # LeftToeBase → left_foot
    75: 11,   # RightToeBase → right_foot
    4: 12,    # Neck1 → neck
    11: 13,   # LeftShoulder → left_collar
    39: 14,   # RightShoulder → right_collar
    6: 15,    # Head → head
    12: 16,   # LeftArm → left_shoulder
    40: 17,   # RightArm → right_shoulder
    13: 18,   # LeftForeArm → left_elbow
    41: 19,   # RightForeArm → right_elbow
    14: 20,   # LeftHand → left_wrist
    42: 21,   # RightHand → right_wrist
}

SMPL_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (12, 15),
]

SOMA30_TO_SMPL22 = {
    0: 0, 22: 1, 26: 2, 1: 3, 23: 4, 27: 5,
    2: 6, 24: 7, 28: 8, 3: 9, 25: 10, 29: 11,
    4: 12, 10: 13, 16: 14, 6: 15, 11: 16, 17: 17,
    12: 18, 18: 19, 13: 20, 19: 21,
}


def soma77_to_smpl22(joints77):
    """SOMA77 [T, 77, 3] → SMPL22 [T, 22, 3]"""
    T = joints77.shape[0]
    smpl = np.zeros((T, 22, 3), dtype=np.float32)
    for s77, s22 in SOMA77_TO_SMPL22.items():
        smpl[:, s22] = joints77[:, s77]
    return smpl


def soma30_to_smpl22(soma_joints):
    """SOMA30 [T, 30, 3] → SMPL22 [T, 22, 3]"""
    T = soma_joints.shape[0]
    smpl = np.zeros((T, 22, 3), dtype=np.float32)
    for s30, s22 in SOMA30_TO_SMPL22.items():
        smpl[:, s22] = soma_joints[:, s30]
    return smpl


def load_scene_pts(scene_name, n_points=3000):
    """Load scene mesh vertices (trimesh loads OBJ as Y-up, same as joints)"""
    base_name = scene_name.replace("_mirror", "")
    mesh_path = None
    for candidate in [scene_name, base_name]:
        p = MESH_DIR / candidate / "mesh_low.obj"
        if p.exists():
            mesh_path = p
            break
    if mesh_path is None:
        print(f"  Warning: no mesh found for scene {scene_name}")
        return None
    scene_obj = trimesh.load(str(mesh_path), force="scene")
    verts_list = []
    for name, geom in scene_obj.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            verts = np.array(geom.vertices)
            transform = scene_obj.graph.get(name)[0]
            if transform is not None:
                verts = trimesh.transform_points(verts, transform)
            verts_list.append(verts.astype(np.float32))
    if not verts_list:
        return None
    vv = np.concatenate(verts_list, axis=0)
    if len(vv) > n_points:
        vv = vv[np.random.choice(len(vv), n_points, replace=False)]
    return vv * METER_TO_UNIT


def load_raw_gt_smpl22(seg_idx, max_frames=None):
    """Load raw SMPL22 GT joints from human_joints_aligned.npy"""
    start_arr = np.load(str(LINGO_DATASET_DIR / "start_idx.npy"))
    end_arr = np.load(str(LINGO_DATASET_DIR / "end_idx.npy"))
    s, e = int(start_arr[seg_idx]), int(end_arr[seg_idx])
    if max_frames is not None:
        e = min(e, s + max_frames)
    joints_path = LINGO_DATASET_DIR / "human_joints_aligned.npy"
    raw = np.load(str(joints_path), mmap_mode="r")
    gt = raw[s:e].astype(np.float32).copy()
    gt = gt[:, :22, :]  # LINGO 28 → SMPL 22
    gt *= METER_TO_UNIT
    return gt


def _prepare_3d(posed_joints, root_positions):
    j3d = posed_joints * METER_TO_UNIT
    r3d = root_positions * METER_TO_UNIT
    return j3d.astype(np.float32), r3d.astype(np.float32)


def _draw_skeleton(ax, joints, roots, fi, color, root_color, scope):
    pos = joints[fi]
    for a, b in SMPL_CONNECTIONS:
        if a < pos.shape[0] and b < pos.shape[0]:
            ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], [pos[a, 2], pos[b, 2]],
                    color=color, linewidth=3, zorder=8, alpha=0.85)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=40,
               depthshade=False, zorder=10, edgecolors='white', linewidths=0.8)
    if fi > 0:
        s = max(0, fi - 30)
        t = roots[s:fi + 1]
        if t.shape[1] == 1:
            t = t[:, 0, :]
        if len(t) > 1:
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=root_color, linewidth=2.5, alpha=0.75, zorder=6)
    ax.set_xlim(scope["x_min"], scope["x_max"])
    ax.set_ylim(scope["y_min"], scope["y_max"])
    ax.set_zlim(scope.get("z_min", 0), scope.get("z_max", 250))
    ax.set_axis_off()
    ax.grid(False)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False


def render_video(data_s2, scene_pts, scene_name, prompt, output_path, fps=20):
    j2_raw = np.squeeze(data_s2["posed_joints"].astype(np.float32))
    if j2_raw.ndim == 4:
        j2_raw = j2_raw.reshape(j2_raw.shape[1], -1, j2_raw.shape[-1])
    r2_raw = np.squeeze(data_s2["root_positions"].astype(np.float32))
    if r2_raw.ndim == 1:
        r2_raw = r2_raw[:, None]

    j2, r2 = _prepare_3d(j2_raw, r2_raw)
    if j2.shape[1] == 30:
        j2 = soma30_to_smpl22(j2)

    nf = j2.shape[0]
    j2 = j2[:nf]
    r2 = r2[:nf]

    def _compute_scope(joints):
        pts = joints.reshape(-1, 3)
        c = np.mean(pts, axis=0)
        s = max(np.max(np.abs(pts - c)) + 50, 80)
        return {
            "x_min": c[0] - s, "x_max": c[0] + s,
            "y_min": c[1] - s * 0.6, "y_max": c[1] + s * 0.6,
            "z_min": c[2] - s * 0.5, "z_max": c[2] + s,
        }

    scope_s2 = _compute_scope(j2)

    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    writer = FFMpegWriter(fps=fps)
    writer.setup(fig, str(output_path), dpi=100)
    print(f"  Rendering video: {nf} frames → {Path(output_path).name} ...")

    try:
        for fi in tqdm(range(nf), desc="  Video"):
            ax.cla()
            ax.set_facecolor("white")

            if scene_pts is not None:
                ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                             c="silver", s=5, alpha=0.6, marker=".")

            _draw_skeleton(ax, j2, r2, fi, "#E65100", "#BF360C", scope_s2)

            ax.view_init(elev=20, azim=-115)
            ax.set_title(f'"{prompt}"  |  Scene {scene_name}', color="#333333", fontsize=11, pad=8)
            writer.grab_frame()
    finally:
        writer.finish()

    plt.close(fig)
    print(f"  Video saved: {output_path}")


def render_comparison_video(data_s1, data_s2, scene_pts, scene_name, prompt, output_path, fps=20):
    j1_raw = np.squeeze(data_s1["posed_joints"].astype(np.float32))
    if j1_raw.ndim == 4:
        j1_raw = j1_raw.reshape(j1_raw.shape[1], -1, j1_raw.shape[-1])
    r1_raw = np.squeeze(data_s1["root_positions"].astype(np.float32))
    if r1_raw.ndim == 1:
        r1_raw = r1_raw[:, None]

    j1, r1 = _prepare_3d(j1_raw, r1_raw)
    j1 = soma77_to_smpl22(j1)

    j2_raw = np.squeeze(data_s2["posed_joints"].astype(np.float32))
    if j2_raw.ndim == 4:
        j2_raw = j2_raw.reshape(j2_raw.shape[1], -1, j2_raw.shape[-1])
    r2_raw = np.squeeze(data_s2["root_positions"].astype(np.float32))
    if r2_raw.ndim == 1:
        r2_raw = r2_raw[:, None]

    j2, r2 = _prepare_3d(j2_raw, r2_raw)
    if j2.shape[1] == 30:
        j2 = soma30_to_smpl22(j2)

    nf = min(j1.shape[0], j2.shape[0])
    j1 = j1[:nf]
    r1 = r1[:nf]
    j2 = j2[:nf]
    r2 = r2[:nf]

    def _cs(joints):
        pts = joints.reshape(-1, 3)
        c = np.mean(pts, axis=0)
        s = max(np.max(np.abs(pts - c)) + 50, 80)
        return {"x_min": c[0]-s, "x_max": c[0]+s, "y_min": c[1]-s*0.6, "y_max": c[1]+s*0.6,
                "z_min": c[2]-s*0.5, "z_max": c[2]+s}

    scope_s1 = _cs(j1)
    scope_s2 = _cs(j2)

    fig = plt.figure(figsize=(20, 10), facecolor="white")
    ax_l = fig.add_subplot(121, projection="3d", facecolor="white")
    ax_r = fig.add_subplot(122, projection="3d", facecolor="white")
    fig.subplots_adjust(wspace=0.02)

    writer = FFMpegWriter(fps=fps)
    writer.setup(fig, str(output_path), dpi=80)
    print(f"  Rendering comparison video: {nf} frames → {Path(output_path).name} ...")

    try:
        for fi in tqdm(range(nf), desc="  Comparison"):
            ax_l.cla()
            ax_r.cla()
            ax_l.set_facecolor("white")
            ax_r.set_facecolor("white")

            if scene_pts is not None:
                ax_l.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                             c="silver", s=5, alpha=0.6, marker=".")
                ax_r.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                             c="silver", s=5, alpha=0.6, marker=".")

            _draw_skeleton(ax_l, j1, r1, fi, "#1565C0", "#0D47A1", scope_s1)
            _draw_skeleton(ax_r, j2, r2, fi, "#E65100", "#BF360C", scope_s2)

            ax_l.view_init(elev=20, azim=-115)
            ax_r.view_init(elev=20, azim=-115)

            ax_l.set_title("Stage 1: SceneCo root_only", color="#1565C0", fontsize=12, pad=8, fontweight="bold")
            ax_r.set_title("Stage 2: Kimodo + root constraint", color="#E65100", fontsize=12, pad=8, fontweight="bold")

            fig.suptitle(f'"{prompt}"  |  Frame {fi}/{nf}  |  Scene {scene_name}',
                         color="#333333", fontsize=10, y=0.02)
            writer.grab_frame()
    finally:
        writer.finish()

    plt.close(fig)
    print(f"  Comparison video saved: {output_path}")


def render_trajectory_plot(data_s1, data_s2, prompt, output_path):
    root1 = data_s1["smooth_root_pos"]
    root2 = data_s2["smooth_root_pos"]
    nf = min(len(root1), len(root2))
    root1, root2 = root1[:nf], root2[:nf]

    t = np.arange(nf) / 30.0

    def compute_speed(roots):
        v = np.diff(roots, axis=0)
        return np.linalg.norm(v, axis=-1)

    def compute_heading(roots):
        v = np.diff(roots, axis=0)
        h = np.arctan2(v[:, 2], v[:, 0])
        dh = np.abs(np.diff(h))
        dh[dh > np.pi] = 2 * np.pi - dh[dh > np.pi]
        cum = np.concatenate([[0], np.cumsum(dh)])
        return cum

    spd1 = compute_speed(root1)
    spd2 = compute_speed(root2)
    hdg1 = compute_heading(root1)
    hdg2 = compute_heading(root2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor="white")
    fig.suptitle(f'Motion Analysis — "{prompt}"', fontsize=14, fontweight="bold", y=0.98)

    ax_spd = axes[0, 0]
    t_spd = t[1:]
    ax_spd.plot(t_spd, spd1, '-', color="#00DDFF", linewidth=2, label="Stage1: SceneCo root_only", alpha=0.9)
    ax_spd.plot(t_spd, spd2, '-', color="#FF8800", linewidth=2, label="Stage2: Kimodo + root constr.", alpha=0.9)
    ax_spd.set_xlabel("Time (s)", fontsize=11)
    ax_spd.set_ylabel("Speed (m/s)", fontsize=11)
    ax_spd.set_title("Root Translation Speed", fontsize=12, fontweight="bold")
    ax_spd.legend(fontsize=9)
    ax_spd.grid(True, alpha=0.3)
    ax_spd.text(0.02, 0.98,
                f"Mean S1={np.mean(spd1):.3f} m/s\nMean S2={np.mean(spd2):.3f} m/s",
                transform=ax_spd.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax_hdg = axes[0, 1]
    ax_hdg.plot(t, hdg1, '-', color="#00DDFF", linewidth=2, label="Stage1", alpha=0.9)
    ax_hdg.plot(t, hdg2, '-', color="#FF8800", linewidth=2, label="Stage2", alpha=0.9)
    ax_hdg.set_xlabel("Time (s)", fontsize=11)
    ax_hdg.set_ylabel("Cumulative Heading Change (rad)", fontsize=11)
    ax_hdg.set_title("Cumulative Heading / Orientation Change", fontsize=12, fontweight="bold")
    ax_hdg.legend(fontsize=9)
    ax_hdg.grid(True, alpha=0.3)
    ax_hdg.text(0.02, 0.98,
                f"Total Δθ S1={hdg1[-1]:.2f} rad ({np.degrees(hdg1[-1]):.1f}°)\nTotal Δθ S2={hdg2[-1]:.2f} rad ({np.degrees(hdg2[-1]):.1f}°)",
                transform=ax_hdg.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax_x = axes[1, 0]
    ax_x.plot(t, root1[:, 0], '-', color="#00DDFF", linewidth=2, label="Stage1 X")
    ax_x.plot(t, root2[:, 0], '-', color="#FF8800", linewidth=2, label="Stage2 X")
    ax_x.set_xlabel("Time (s)", fontsize=11)
    ax_x.set_ylabel("X (m)", fontsize=11)
    ax_x.set_title("X Position over Time", fontsize=12, fontweight="bold")
    ax_x.legend(fontsize=9)
    ax_x.grid(True, alpha=0.3)

    ax_z = axes[1, 1]
    ax_z.plot(t, root1[:, 2], '-', color="#00DDFF", linewidth=2, label="Stage1 Z")
    ax_z.plot(t, root2[:, 2], '-', color="#FF8800", linewidth=2, label="Stage2 Z")
    ax_z.set_xlabel("Time (s)", fontsize=11)
    ax_z.set_ylabel("Z (m)", fontsize=11)
    ax_z.set_title("Z Position over Time", fontsize=12, fontweight="bold")
    ax_z.legend(fontsize=9)
    ax_z.grid(True, alpha=0.3)

    diff_3d = np.sqrt(np.sum((root1 - root2) ** 2, axis=-1))
    ax_z_right = ax_z.twinx()
    ax_z_right.plot(t, diff_3d, ':', color="red", linewidth=1.5, alpha=0.6, label="|Δ| 3D")
    ax_z_right.set_ylabel("|Δ| 3D (m)", color="red", fontsize=9)
    ax_z_right.tick_params(axis="y", labelcolor="red")
    ax_z_right.legend(loc="upper right", fontsize=8)

    stats_text = (
        f"3D Error:\n"
        f"  Mean: {np.mean(diff_3d):.4f} m\n"
        f"  Max:  {np.max(diff_3d):.4f} m\n"
        f"  RMSE: {np.sqrt(np.mean(diff_3d ** 2)):.4f} m"
    )
    ax_x.text(0.98, 0.98, stats_text, transform=ax_x.transAxes,
               fontsize=9, verticalalignment="top", horizontalalignment="right",
               fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  Trajectory plot saved: {output_path}")



def find_motion_files(input_dir):
    input_dir = Path(input_dir)
    s1_files = sorted(input_dir.glob("stage1_sceneco_*_motion.npz"))

    pairs = []
    for s1_path in s1_files:
        stem = s1_path.stem
        suffix = stem.replace("stage1_sceneco_", "")
        s2_path = input_dir / f"stage2_kimodo_{suffix}.npz"
        if s2_path.exists():
            pairs.append((s1_path, s2_path, suffix))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Visualize two-stage inference results")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing stage1/stage2 npz files")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/two_stage_viz")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_motion_files(input_dir)
    if not pairs:
        print(f"No stage1/stage2 pairs found in {input_dir}")
        return

    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset
    ds = LINGOSceneMotionDataset(
        data_root=str(PROJECT_ROOT / "LINGO" / "dataset"),
        max_frames=196, min_frames=40, voxel_size=(64, 64, 64),
        train_ratio=0.9, seed=42, split="val",
        scene_dropout=0.0,
        cache_dir=str(PROJECT_ROOT / "kimodo/kimodo_sceneco/cached_data"),
    )

    scene_cache = {}
    sn_cache = {}

    def get_scene(sample_idx):
        if sample_idx in scene_cache:
            return scene_cache[sample_idx], sn_cache[sample_idx]
        seg_meta = ds.segments[sample_idx]
        if "cache_path" in seg_meta:
            actual = int(Path(seg_meta["cache_path"]).stem.split("_")[-1])
        else:
            actual = sample_idx
        sn = ds[sample_idx].get("scene_name", seg_meta.get("scene_name", "026"))
        if hasattr(sn, '__iter__') and not isinstance(sn, str):
            sn = sn[0] if len(sn) > 0 else "026"
        scene_cache[sample_idx] = load_scene_pts(sn)
        sn_cache[sample_idx] = sn
        return scene_cache[sample_idx], sn_cache[sample_idx]

    print(f"Found {len(pairs)} motion pair(s) in {input_dir}")

    for s1_path, s2_path, suffix in pairs:
        import re
        m = re.search(r'_s(\d+)_motion', suffix)
        sample_idx = int(m.group(1)) if m else 0

        scene_pts, scene_name = get_scene(sample_idx)
        prompt_from_file = suffix.replace("_s" + str(sample_idx) + "_motion", "").replace("_", " ")

        print(f"\nProcessing: {suffix}  (sample={sample_idx}, scene={scene_name})")

        data_s2 = dict(np.load(s2_path))

        base_name = suffix.replace("_motion", "")
        video_path = output_dir / f"{base_name}_stage2.mp4"

        render_video(data_s2, scene_pts, scene_name, prompt_from_file, video_path, fps=args.fps)

    print(f"\nDone. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
