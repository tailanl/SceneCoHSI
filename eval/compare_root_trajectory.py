#!/usr/bin/env python
"""Compare Stage1 SceneCo vs GT: MP4 video + heading PNG.

For LINGO val segments, reconstructs motion from noisy GT and compares:
  - GT (ground truth)
  - Stage1 SceneCo (with scene input)
  - Original Kimodo (pretrained, no SceneCo)

Output: MP4 video (GT | Stage1 w/ scene) and heading PNG.

Usage:
    conda activate kimodo && cd /home/lzsh2025/kimodo-viser && \
    CUDA_VISIBLE_DEVICES=7 python kimodo_scene_project/eval/compare_root_trajectory.py --gpu 0
"""

import argparse
import io
import os
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_ROOT / "SOMA"))

os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
os.environ.setdefault("PYTHONHASHSEED", "0")

METER_TO_UNIT = 100.0
MESH_DIR = PROJECT_ROOT / "LINGO" / "scene_mesh" / "Scene_mesh"

SOMA30_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 7), (6, 8), (6, 9),
    (3, 10), (10, 11), (11, 12), (12, 13), (13, 14), (13, 15),
    (3, 16), (16, 17), (17, 18), (18, 19), (19, 20), (19, 21),
    (0, 22), (22, 23), (23, 24), (24, 25),
    (0, 26), (26, 27), (27, 28), (28, 29),
]


def load_segmented_scene_pts(scene_name, n_points=12000):
    import trimesh

    mesh_path = None
    base_name = scene_name.replace("_mirror", "")
    for candidate in [scene_name, base_name]:
        p = MESH_DIR / candidate / "mesh_low.obj"
        if p.exists():
            mesh_path = p
            break
    if mesh_path is None:
        return []

    scene_obj = trimesh.load(str(mesh_path), force="scene")
    all_verts, all_faces, all_normals = [], [], []
    offset = 0
    for name, geom in scene_obj.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            verts = np.array(geom.vertices)
            transform = scene_obj.graph.get(name)[0]
            if transform is not None:
                verts = trimesh.transform_points(verts, transform)
            faces = np.array(geom.faces) + offset
            fn = np.array(geom.face_normals)
            if transform is not None:
                R = transform[:3, :3]
                fn = (R @ fn.T).T
                fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
            all_verts.append(verts)
            all_faces.append(faces)
            all_normals.append(fn)
            offset += len(verts)

    if not all_verts:
        return []

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    fn_all = np.concatenate(all_normals, axis=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    point_heights = points[:, 1]
    pn = fn_all[face_idx]

    up_mask = pn[:, 1] > 0.5
    down_mask = pn[:, 1] < -0.5
    vert_mask = ~up_mask & ~down_mask

    segments = []
    floor_mask = up_mask & (point_heights < 0.15)
    ceiling_mask = down_mask & (point_heights > 1.5)
    furniture_mask = up_mask & (point_heights >= 0.15) & (point_heights < 1.5)
    wall_mask = vert_mask

    if floor_mask.any():
        segments.append({"pts": points[floor_mask], "color": (0.75, 0.75, 0.70)})
    if wall_mask.any():
        segments.append({"pts": points[wall_mask], "color": (0.60, 0.55, 0.50)})
    if ceiling_mask.any():
        segments.append({"pts": points[ceiling_mask], "color": (0.90, 0.90, 0.88)})
    if furniture_mask.any():
        furn_pts = points[furniture_mask]
        furn_h = point_heights[furniture_mask]
        for (lo, hi), col in [
            ((0.15, 0.5), (0.72, 0.53, 0.04)),
            ((0.5, 0.8), (0.55, 0.27, 0.07)),
            ((0.8, 1.1), (0.41, 0.41, 0.41)),
            ((1.1, 1.5), (0.25, 0.41, 0.88)),
        ]:
            m = (furn_h >= lo) & (furn_h < hi)
            if m.any():
                segments.append({"pts": furn_pts[m], "color": col})

    remaining = ~(floor_mask | ceiling_mask | wall_mask | furniture_mask)
    if remaining.any():
        segments.append({"pts": points[remaining], "color": (0.65, 0.65, 0.65)})

    if not segments:
        segments = [{"pts": points, "color": (0.7, 0.7, 0.7)}]
    return segments


def build_stage1_model(device):
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=str(device))
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model

    model = KimodoSceneCo(
        denoiser=inner,
        text_encoder=pretrained.text_encoder,
        num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64),
            "patch_size": (8, 8, 8),
            "d_model": 256,
            "num_layers": 4,
            "use_dual_vit": True,
            "root_voxel_mode": "floor",
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=True,
        use_in_body_model=False,
    )
    model = model.to(device)
    model.eval()

    ckpt_path = PROJECT_ROOT / "kimodo_scene_project/outputs/stage1_root_only/checkpoints/checkpoint_step150000_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    print(f"  Stage1 SceneCo loaded: step={ckpt.get('global_step', ckpt.get('step', '?'))}")
    return model


def build_original_kimodo(device):
    from kimodo.model import load_model
    model = load_model("Kimodo-SOMA-RP-v1.1", device=str(device))
    model.eval()
    print(f"  Original Kimodo loaded")
    return model


def run_reconstruction_sceneco(model, motion_gt, mask, voxel, text_feat, text_pad_mask, noise_level, device):
    B, T, D = motion_gt.shape
    t = torch.full((B,), noise_level, device=device)
    noise = torch.randn_like(motion_gt)
    x_t = model.diffusion.q_sample(motion_gt, t, noise=noise)
    x_pad_mask = mask.bool() if mask.dtype != torch.bool else mask
    fha = torch.zeros(B, device=device)

    with torch.no_grad():
        (sfr, smr), (sfb, smb) = model.encode_scene(voxel)

        pred_with = model.denoiser(
            [2.0, 2.0, 2.0],
            x_t, x_pad_mask, text_feat, text_pad_mask, t,
            first_heading_angle=fha,
            scene_feat_root=sfr, scene_mask_root=smr,
            scene_feat_body=sfb, scene_mask_body=smb,
            cfg_type="nocfg",
        )

    return pred_with


def run_reconstruction_kimodo(model, motion_gt, mask, text_feat, text_pad_mask, noise_level, device):
    B, T, D = motion_gt.shape
    t = torch.full((B,), noise_level, device=device)
    noise = torch.randn_like(motion_gt)
    x_t = model.diffusion.q_sample(motion_gt, t, noise=noise)

    x_pad_mask = mask.bool() if mask.dtype != torch.bool else mask
    fha = torch.zeros(B, device=device)

    with torch.no_grad():
        pred = model.denoiser.model(
            x_t, x_pad_mask, text_feat, text_pad_mask, t,
            first_heading_angle=fha,
        )

    return pred


def features_to_dict(features, motion_rep, is_normalized=True):
    with torch.no_grad():
        out = motion_rep.inverse(features, is_normalized=is_normalized, return_numpy=False)
    return {
        "posed_joints": out["posed_joints"].cpu().numpy(),
        "root_positions": out["root_positions"].cpu().numpy(),
        "global_root_heading": out["global_root_heading"].cpu().numpy(),
        "smooth_root_pos": out["smooth_root_pos"].cpu().numpy(),
    }


def _fill_nan_joints(joints):
    """Fill NaN frames by linear interpolation from nearest valid frames."""
    filled = joints.copy()
    T, J, D = filled.shape
    for j in range(J):
        for d in range(D):
            col = filled[:, j, d]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue
            valid = np.where(~nan_mask)[0]
            if len(valid) == 0:
                continue  # all NaN, can't fill
            # Linear interp
            col[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid, col[valid]
            )
            filled[:, j, d] = col
    return filled


def _prepare_3d(joints, roots):
    j = np.squeeze(joints.astype(np.float32))
    r = np.squeeze(roots.astype(np.float32))
    if j.ndim == 4:
        j = j.reshape(j.shape[1], -1, j.shape[-1])
        r = r.reshape(r.shape[1], -1, r.shape[-1])
    j = _fill_nan_joints(j)
    jr = j * METER_TO_UNIT
    rr = r * METER_TO_UNIT
    if rr.ndim == 2:
        rr = rr[:, None, :]
    # NOTE: inverse() already outputs Y-up coordinates; do NOT swap Y/Z
    return jr, rr


def _draw_skeleton(ax, joints, roots, fi, n_joints, color, root_color):
    pos = joints[fi]
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=color, s=60, depthshade=False, zorder=10, edgecolors="white", linewidths=0.8,
    )
    for a, b in SOMA30_CONNECTIONS:
        if a < pos.shape[0] and b < pos.shape[0]:
            ax.plot(
                [pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], [pos[a, 2], pos[b, 2]],
                color=color, linewidth=3.5, zorder=8,
            )
    rp = roots[fi, 0]
    ax.scatter(
        [rp[0]], [rp[1]], [rp[2]],
        c=root_color, s=100, depthshade=False, zorder=11, marker="s", edgecolors="white",
    )


def compute_heading_angle_2d(heading_2d):
    return np.arctan2(heading_2d[..., 1], heading_2d[..., 0]) * 180 / np.pi


def render_comparison_video(dict_gt, dict_s1, scene_segments, output_path, fps=20):
    j_gt, r_gt = _prepare_3d(dict_gt["posed_joints"], dict_gt["root_positions"])
    j_s1, r_s1 = _prepare_3d(dict_s1["posed_joints"], dict_s1["root_positions"])
    nf = min(j_gt.shape[0], j_s1.shape[0])

    char_pts = np.concatenate([j_gt[:nf].reshape(-1, 3), j_s1[:nf].reshape(-1, 3)], axis=0)
    char_pts = char_pts[np.isfinite(char_pts).all(axis=1)]
    char_center = np.mean(char_pts, axis=0) if len(char_pts) > 0 else np.zeros(3)
    char_spread = np.max(np.abs(char_pts - char_center)) + 40 if len(char_pts) > 0 else 80

    scope = {
        "x_min": char_center[0] - char_spread,
        "x_max": char_center[0] + char_spread,
        "y_min": char_center[1] - char_spread,
        "y_max": char_center[1] + char_spread,
        "z_min": char_center[2] - char_spread * 0.4,
        "z_max": char_center[2] + char_spread,
    }

    fig = plt.figure(figsize=(10, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, facecolor="white", edgecolor="none", format="png")
    buf.seek(0)
    test_frame = np.array(Image.open(buf))
    buf.close()
    h, w = test_frame.shape[:2]
    h -= h % 2
    w -= w % 2

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "medium"}

    for fi in tqdm(range(nf), desc=f"  MP4", leave=False):
        ax.cla()
        ax.set_facecolor("#FAFAFA")

        for seg in scene_segments:
            pts = seg["pts"].copy()
            pts_r = np.zeros_like(pts)
            pts_r[:, 0] = pts[:, 0] * METER_TO_UNIT
            pts_r[:, 1] = pts[:, 2] * METER_TO_UNIT
            pts_r[:, 2] = pts[:, 1] * METER_TO_UNIT
            ax.scatter(pts_r[:, 0], pts_r[:, 1], pts_r[:, 2],
                       c=[seg["color"]], s=0.3, alpha=0.12, depthshade=True, zorder=1)

        _draw_skeleton(ax, j_gt, r_gt, fi, j_gt.shape[1], "#4CAF50", "#1B5E20")
        _draw_skeleton(ax, j_s1, r_s1, fi, j_s1.shape[1], "#FF5722", "#BF360C")

        ax.set_xlim(scope["x_min"], scope["x_max"])
        ax.set_ylim(scope["y_min"], scope["y_max"])
        ax.set_zlim(scope["z_min"], scope["z_max"])
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-45)

        ax.set_title(f"GT (green)  vs  Stage1 Root-Only (orange)  —  {nf}fr",
                     fontsize=12, pad=-2)

        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, facecolor="white", edgecolor="none", format="png")
        buf.seek(0)
        frame = np.array(Image.open(buf))
        buf.close()

        av_frame = av.VideoFrame.from_ndarray(frame[:h, :w, :3], format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    plt.close(fig)
    container.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--noise_level", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset

    cache_dir = str(PROJECT_ROOT / "kimodo/kimodo_sceneco/cached_data")
    ds_val = LINGOSceneMotionDataset(
        data_root=str(PROJECT_ROOT / "LINGO" / "dataset"),
        max_frames=196, min_frames=40,
        voxel_size=(64, 64, 64),
        train_ratio=0.9, seed=42,
        split="val",
        scene_dropout=0.0,
        cache_dir=cache_dir,
    )

    rng = np.random.RandomState(42)
    indices = rng.choice(len(ds_val), size=min(args.num_samples, len(ds_val)), replace=False)

    print(f"Loading Stage1 SceneCo model...")
    model_s1 = build_stage1_model(device)
    mr_s1 = model_s1.denoiser.model.motion_rep

    print(f"Loading Original Kimodo model...")
    model_orig = build_original_kimodo(device)
    mr_orig = model_orig.motion_rep

    output_dir = PROJECT_ROOT / "kimodo_scene_project/outputs/eval_lingo/root_trajectory_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    for vi, idx in enumerate(sorted(indices)):
        seg = ds_val[idx]
        scene_name = seg.get("scene_name", f"unk_{vi}")
        text = seg.get("text", "no-text")
        n_frames = int(seg["length"])
        print(f"\n[{vi}] scene={scene_name}, frames={n_frames}, text={text[:50]}")

        motion_gt = seg["motion_features"].unsqueeze(0).to(device)
        T_full = motion_gt.shape[1]
        mask = torch.zeros(1, T_full, device=device)
        mask[0, :n_frames] = 1.0
        voxel = seg["voxel_grid"].unsqueeze(0).to(device)

        text_feat_1d = seg["text_feat"].reshape(1, 1, -1).to(device)
        text_pad_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

        # Stage1 SceneCo reconstruction
        pred_s1 = run_reconstruction_sceneco(
            model_s1, motion_gt, mask, voxel, text_feat_1d, text_pad_mask, args.noise_level, device
        )

        # Original Kimodo reconstruction
        pred_orig = run_reconstruction_kimodo(
            model_orig, motion_gt[:, :n_frames, :], mask[:, :n_frames],
            text_feat_1d, text_pad_mask, args.noise_level, device
        )

        # Decode
        data_gt = features_to_dict(motion_gt[:, :n_frames, :], mr_s1, is_normalized=True)
        data_s1 = features_to_dict(pred_s1[:, :n_frames, :], mr_s1, is_normalized=True)
        data_orig = features_to_dict(pred_orig, mr_orig, is_normalized=True)

        # Root-only: Stage1 root_model output + GT body features
        pred_s1_ro = torch.cat([
            pred_s1[:, :n_frames, mr_s1.root_slice],
            motion_gt[:, :n_frames, mr_s1.body_slice],
        ], dim=-1)
        data_s1_ro = features_to_dict(pred_s1_ro, mr_s1, is_normalized=True)

        # Debug: print root positions
        rp_gt = data_gt["root_positions"][0, :, :]
        rp_s1 = data_s1["root_positions"][0, :, :]
        rp_ro = data_s1_ro["root_positions"][0, :, :]
        print(f"    [DIAG] GT  XZ range: x=[{rp_gt[:,0].min():.3f},{rp_gt[:,0].max():.3f}] "
              f"z=[{rp_gt[:,2].min():.3f},{rp_gt[:,2].max():.3f}]")
        print(f"    [DIAG] S1  XZ range: x=[{rp_s1[:,0].min():.3f},{rp_s1[:,0].max():.3f}] "
              f"z=[{rp_s1[:,2].min():.3f},{rp_s1[:,2].max():.3f}]")
        print(f"    [DIAG] RO  XZ range: x=[{rp_ro[:,0].min():.3f},{rp_ro[:,0].max():.3f}] "
              f"z=[{rp_ro[:,2].min():.3f},{rp_ro[:,2].max():.3f}]")

        safe_name = f"sample{vi}_{scene_name}_{text[:20].replace(' ','_')}"

        # Load scene
        scene_segments = load_segmented_scene_pts(scene_name, n_points=12000)

        # ===== MP4 Video: GT vs Stage1 Root-Only in same scene =====
        out_mp4 = output_dir / f"root_only_{safe_name}.mp4"
        print(f"  Generating overlay video (GT green + Stage1 orange)...")
        render_comparison_video(data_gt, data_s1_ro, scene_segments, out_mp4, fps=20)
        print(f"  Saved: {out_mp4}")

        # ===== Heading Angle PNG =====
        fig2, ax_head = plt.subplots(figsize=(14, 5), facecolor="white")
        ax_head.set_facecolor("white")
        time_axis = np.arange(n_frames) / 20.0

        head_gt = compute_heading_angle_2d(data_gt["global_root_heading"][0])
        head_s1 = compute_heading_angle_2d(data_s1["global_root_heading"][0])
        head_or = compute_heading_angle_2d(data_orig["global_root_heading"][0])

        ax_head.plot(time_axis, head_gt, color="#4CAF50", linewidth=2.5, label="GT")
        ax_head.plot(time_axis[:len(head_s1)], head_s1, color="#FF5722", linewidth=2,
                     label="Stage1 (w/ scene)")
        ax_head.plot(time_axis[:len(head_or)], head_or, color="#9C27B0", linewidth=1.5,
                     linestyle=":", label="Original Kimodo")

        mse_h_s1 = np.mean((head_gt[:len(head_s1)] - head_s1) ** 2)
        mse_h_or = np.mean((head_gt[:len(head_or)] - head_or) ** 2)

        ax_head.set_xlabel("Time (s)", fontsize=11)
        ax_head.set_ylabel("Heading (degrees)", fontsize=11)
        ax_head.set_title(f"Heading Angle — {scene_name}  |  {text[:60]}\n"
                          f"MSE: Stage1(w)= {mse_h_s1:.1f}  |  Kimodo= {mse_h_or:.1f}",
                          fontsize=11)
        ax_head.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax_head.grid(True, alpha=0.3)

        fig2.tight_layout()
        out_head = output_dir / f"heading_{safe_name}.png"
        fig2.savefig(str(out_head), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved heading: {out_head}")

    del model_s1, model_orig
    torch.cuda.empty_cache()

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
