#!/usr/bin/env python
"""LINGO test-set evaluation + visualization for all 5 experiments.

Evaluates each model on the LINGO val split by computing diffusion reconstruction MSE.
Compares with and without scene conditioning.
Visualizes selected segments: MP4 video (GT | With Scene | No Scene) + 2D trajectory.

Usage:
    CUDA_VISIBLE_DEVICES=2 python kimodo_scene_project/eval/batch_eval_lingo.py --gpu 0
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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
SCENE_DIR = PROJECT_ROOT / "LINGO" / "dataset" / "dataset" / "Scene"
MESH_DIR = PROJECT_ROOT / "LINGO" / "scene_mesh" / "Scene_mesh"

SOMA30_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 7), (6, 8), (6, 9),
    (3, 10), (10, 11), (11, 12), (12, 13), (13, 14), (13, 15),
    (3, 16), (16, 17), (17, 18), (18, 19), (19, 20), (19, 21),
    (0, 22), (22, 23), (23, 24), (24, 25),
    (0, 26), (26, 27), (27, 28), (28, 29),
]

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
    4: 12, 10: 13, 16: 14, 6: 15,
    11: 16, 17: 17, 12: 18, 18: 19, 13: 20, 19: 21,
}

LINGO_DATASET_DIR = PROJECT_ROOT / "LINGO" / "dataset" / "dataset"


def find_latest_ckpt(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = PROJECT_ROOT / ckpt_dir

    steps = []
    for f in ckpt_dir.glob("checkpoint_step*.pt"):
        if "_final" in f.name:
            continue
        try:
            steps.append((int(f.stem.replace("checkpoint_step", "")), f))
        except ValueError:
            pass

    best_path = ckpt_dir / "best_checkpoint.pt"
    if best_path.exists():
        steps.append((999999, best_path))

    if not steps:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    steps.sort()
    return steps[-1][1]


EXPERIMENTS = {
    "single_vit": {
        "label": "Base (shared encoder)",
        "ckpt_dir": "kimodo_scene_project/outputs/single_vit_gpu1/checkpoints",
        "use_in_root_model": True,
        "use_in_body_model": False,
        "dual_vit": False,
        "root_voxel_mode": "full",
    },
    "dual_vit": {
        "label": "Dual ViT (full)",
        "ckpt_dir": "kimodo_scene_project/outputs/dual_vit_gpu2/checkpoints",
        "use_in_root_model": True,
        "use_in_body_model": False,
        "dual_vit": True,
        "root_voxel_mode": "full",
    },
    "dual_vit_floor": {
        "label": "Dual ViT (floor)",
        "ckpt_dir": "kimodo_scene_project/outputs/dual_vit_floor_gpu3/checkpoints",
        "use_in_root_model": True,
        "use_in_body_model": False,
        "dual_vit": True,
        "root_voxel_mode": "floor",
    },
    "root_only": {
        "label": "Root-Only SceneCo",
        "ckpt_dir": "kimodo_scene_project/outputs/root_only_sceneco/checkpoints",
        "use_in_root_model": True,
        "use_in_body_model": False,
        "dual_vit": True,
        "root_voxel_mode": "full",
    },
    "body_only": {
        "label": "Body-Only SceneCo",
        "ckpt_dir": "kimodo_scene_project/outputs/body_only_sceneco/checkpoints",
        "use_in_root_model": False,
        "use_in_body_model": True,
        "dual_vit": True,
        "root_voxel_mode": "full",
    },
    "stage1_root_only": {
        "label": "2-Stage Stage1 (root, floor)",
        "ckpt_dir": "kimodo_scene_project/outputs/stage1_root_only/checkpoints",
        "use_in_root_model": True,
        "use_in_body_model": False,
        "dual_vit": True,
        "root_voxel_mode": "floor",
    },
    "root_body_merged": {
        "label": "Merged Root+Body SceneCo",
        "ckpt_dir": "kimodo_scene_project/outputs/root_body_merged/checkpoints",
        "use_in_root_model": True,
        "use_in_body_model": True,
        "dual_vit": True,
        "root_voxel_mode": "full",
    },
}


def build_model(exp_cfg, device):
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
            "use_dual_vit": exp_cfg["dual_vit"],
            "root_voxel_mode": exp_cfg["root_voxel_mode"],
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=exp_cfg["use_in_root_model"],
        use_in_body_model=exp_cfg["use_in_body_model"],
    )
    model = model.to(device)
    model.eval()

    ckpt_path = find_latest_ckpt(exp_cfg["ckpt_dir"])
    print(f"  Loading {exp_cfg['label']} from {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if not exp_cfg["dual_vit"]:
        renamed = {}
        for k, v in state_dict.items():
            if k.startswith("scene_encoder."):
                renamed[k.replace("scene_encoder.", "scene_encoder_root.")] = v
                renamed[k.replace("scene_encoder.", "scene_encoder_body.")] = v
            else:
                renamed[k] = v
        state_dict = renamed

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    Missing keys: {len(missing)}")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")
    print(f"    Epoch/Step: {ckpt.get('epoch', ckpt.get('global_step', ckpt.get('step', '?')))}")

    return model


def features_to_joints(motion_rep, features, is_normalized=True):
    with torch.no_grad():
        out = motion_rep.inverse(features, is_normalized=is_normalized, return_numpy=False)
    return {
        "posed_joints": out["posed_joints"].cpu().numpy(),
        "root_positions": out["root_positions"].cpu().numpy(),
    }


def _fill_nan_joints(joints):
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
                continue
            col[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid, col[valid]
            )
            filled[:, j, d] = col
    return filled


def soma30_to_smpl22(soma_joints):
    """SOMA30 [T, 30, 3] → SMPL22 [T, 22, 3]"""
    T = soma_joints.shape[0]
    smpl = np.zeros((T, 22, 3), dtype=np.float32)
    for s30, s22 in SOMA30_TO_SMPL22.items():
        smpl[:, s22] = soma_joints[:, s30]
    return smpl


def load_raw_gt_smpl22(seg_or_meta, max_frames=None):
    """Load raw SMPL22 GT joints from human_joints_aligned.npy"""
    if "start" in seg_or_meta:
        s, e = int(seg_or_meta["start"]), int(seg_or_meta["end"])
    else:
        seg_idx = int(Path(seg_or_meta["cache_path"]).stem.split("_")[-1])
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


def _draw_skeleton(ax, joints, roots, fi, n_joints, color, root_color, connections=SOMA30_CONNECTIONS):
    pos = joints[fi]
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=color, s=30, depthshade=False, zorder=10, edgecolors="white", linewidths=0.5,
    )
    for a, b in connections:
        if a < pos.shape[0] and b < pos.shape[0]:
            ax.plot(
                [pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], [pos[a, 2], pos[b, 2]],
                color=color, linewidth=2.0, zorder=8,
            )
    rp = roots[fi, 0]
    ax.scatter(
        [rp[0]], [rp[1]], [rp[2]],
        c=root_color, s=60, depthshade=False, zorder=11, marker="s", edgecolors="white",
    )


def render_reconstruction_video(dict_gt, dict_with, scene_name, scene_segments, output_path,
                                connections=SOMA30_CONNECTIONS, label_gt="GT", label_with="Pred", fps=20):
    import io
    import av
    from PIL import Image

    j_gt, r_gt = _prepare_3d(dict_gt["posed_joints"], dict_gt["root_positions"])
    j_w, r_w = _prepare_3d(dict_with["posed_joints"], dict_with["root_positions"])
    nf = min(j_gt.shape[0], j_w.shape[0])

    all_pts = [j_gt[:nf].reshape(-1, 3), j_w[:nf].reshape(-1, 3)]
    for seg in scene_segments:
        pts = seg["pts"].copy()
        pts_r = pts * METER_TO_UNIT
        all_pts.append(pts_r)
    all_pts = np.concatenate(all_pts, axis=0)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if len(all_pts) == 0:
        center = np.zeros(3)
        spread = 120
    else:
        center = np.mean(all_pts, axis=0)
        spread = np.max(np.abs(all_pts - center)) + 30

    scope = {
        "x_min": center[0] - spread, "x_max": center[0] + spread,
        "y_min": center[1] - spread, "y_max": center[1] + spread,
        "z_min": center[2] - spread * 0.5, "z_max": center[2] + spread,
    }

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    ax_gt = fig.add_subplot(121, projection="3d", facecolor="white")
    ax_w = fig.add_subplot(122, projection="3d", facecolor="white")
    fig.subplots_adjust(wspace=0.02)

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

    for fi in tqdm(range(nf), desc=f"  MP4 {(scene_name or '?')[:10]}", leave=False):
        for ax in [ax_gt, ax_w]:
            ax.cla()
            ax.set_facecolor("white")

        for seg in scene_segments:
            pts = seg["pts"].copy()
            pts_r = pts * METER_TO_UNIT
            for ax in [ax_gt, ax_w]:
                ax.scatter(pts_r[:, 0], pts_r[:, 1], pts_r[:, 2],
                           c=[seg["color"]], s=0.6, alpha=0.4, depthshade=True, zorder=2)

        _draw_skeleton(ax_gt, j_gt, r_gt, fi, j_gt.shape[1], "#4CAF50", "#1B5E20", connections)
        _draw_skeleton(ax_w, j_w, r_w, fi, j_w.shape[1], "#FF5722", "#BF360C", connections)

        for ax in [ax_gt, ax_w]:
            ax.set_xlim(scope["x_min"], scope["x_max"])
            ax.set_ylim(scope["y_min"], scope["y_max"])
            ax.set_zlim(scope["z_min"], scope["z_max"])
            ax.set_axis_off()
            ax.view_init(elev=60, azim=-60)

        ax_gt.set_title(label_gt, fontsize=12, fontweight="bold", color="#4CAF50")
        ax_w.set_title(label_with, fontsize=12, fontweight="bold", color="#FF5722")

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


def render_reconstruction_trajectory(dict_gt, dict_with, dict_no, scene_name, scene_segments, output_path):
    j_gt, r_gt = _prepare_3d(dict_gt["posed_joints"], dict_gt["root_positions"])
    j_w, r_w = _prepare_3d(dict_with["posed_joints"], dict_with["root_positions"])
    j_n, r_n = _prepare_3d(dict_no["posed_joints"], dict_no["root_positions"])

    roots_gt = r_gt[:min(j_gt.shape[0], j_w.shape[0], j_n.shape[0]), 0, :2].copy()
    roots_w = r_w[:min(j_gt.shape[0], j_w.shape[0], j_n.shape[0]), 0, :2].copy()
    roots_n = r_n[:min(j_gt.shape[0], j_w.shape[0], j_n.shape[0]), 0, :2].copy()

    all_x = [roots_gt[:, 0], roots_w[:, 0], roots_n[:, 0]]
    all_y = [roots_gt[:, 1], roots_w[:, 1], roots_n[:, 1]]
    for seg in scene_segments:
        pts = seg["pts"]
        all_x.append(pts[:, 0] * METER_TO_UNIT)
        all_y.append(pts[:, 1] * METER_TO_UNIT)
    all_x = np.concatenate([a.ravel() for a in all_x])
    all_y = np.concatenate([a.ravel() for a in all_y])
    mask_xy = np.isfinite(all_x) & np.isfinite(all_y)
    all_x = all_x[mask_xy]
    all_y = all_y[mask_xy]
    if len(all_x) == 0:
        center = np.array([0.0, 0.0])
        spread = 100
    else:
        center = np.array([np.mean(all_x), np.mean(all_y)])
        spread = np.max(np.abs(np.column_stack([all_x, all_y]) - center)) + 30
    xlim = (center[0] - spread, center[0] + spread)
    ylim = (center[1] - spread, center[1] + spread)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")

    for seg in scene_segments:
        x2d = seg["pts"][:, 0] * METER_TO_UNIT
        y2d = seg["pts"][:, 2] * METER_TO_UNIT
        ax.scatter(x2d, y2d, c=[seg["color"]], s=0.5, alpha=0.4)

    ax.plot(roots_gt[:, 0], roots_gt[:, 1], color="#4CAF50", linewidth=2, label="GT", zorder=10)
    ax.plot(roots_w[:, 0], roots_w[:, 1], color="#FF5722", linewidth=2, label="WITH Scene", zorder=10)
    ax.plot(roots_n[:, 0], roots_n[:, 1], color="#2196F3", linewidth=2, label="NO Scene", zorder=10)
    ax.scatter(roots_gt[0, 0], roots_gt[0, 1], color="#4CAF50", s=80, marker="o", zorder=11)
    ax.scatter(roots_gt[-1, 0], roots_gt[-1, 1], color="#1B5E20", s=100, marker="*", zorder=11)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title(f"Trajectory — Scene {scene_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_viz_segments(ds_val, num_samples=3, seed=123):
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds_val), size=min(num_samples, len(ds_val)), replace=False)
    segments = []
    for idx in sorted(indices):
        item = ds_val[idx]
        item["_viz_idx"] = idx
        item["_seg_meta"] = ds_val.segments[idx]
        segments.append(item)
    return segments


def evaluate_experiment(exp_cfg, device, val_loader, num_batches):
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_cfg['label']}")
    print(f"{'='*60}")

    model = build_model(exp_cfg, device)
    D = model.denoiser.model.motion_rep.motion_rep_dim
    root_dim = model.denoiser.model.motion_rep.global_root_dim
    motion_rep = model.denoiser.model.motion_rep
    print(f"  motion_dim={D}, root_dim={root_dim}")

    total_mse_with = 0.0
    total_mse_no = 0.0
    total_root_mse_with = 0.0
    total_root_mse_no = 0.0
    total_body_mse_with = 0.0
    total_body_mse_no = 0.0
    total_elements = 0.0
    n_batches = 0

    for batch in tqdm(val_loader, total=num_batches, desc=f"  {exp_cfg['label']}"):
        motion = batch["motion_features"].to(device)
        mask = batch["motion_mask"].to(device)
        voxel = batch["voxel_grid"].to(device)
        B, T, D = motion.shape
        total_elements += mask.sum().item() * D

        if "text_feat" in batch:
            text_feat = batch["text_feat"].to(device)
            text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        else:
            texts = batch["texts"]
            text_feat, text_length = model.text_encoder(texts)
            text_feat = text_feat.to(device)
            maxlen = text_feat.shape[1]
            text_pad_mask = torch.arange(maxlen, device=device).expand(B, maxlen) < text_length[:, None].to(device)

        with torch.no_grad():
            (scene_feat_root, scene_mask_root), (scene_feat_body, scene_mask_body) = model.encode_scene(voxel)

            t = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(motion)
            x_t = model.diffusion.q_sample(motion, t, noise=noise)

            x_pad_mask = mask.bool() if mask.dtype != torch.bool else mask

            fha = torch.zeros(B, device=device)

            pred_with = model.denoiser(
                [2.0, 2.0, 2.0],
                x_t, x_pad_mask, text_feat, text_pad_mask, t,
                first_heading_angle=fha,
                scene_feat_root=scene_feat_root,
                scene_mask_root=scene_mask_root,
                scene_feat_body=scene_feat_body,
                scene_mask_body=scene_mask_body,
                cfg_type="nocfg",
            )

            pred_no = model.denoiser(
                [2.0, 2.0],
                x_t, x_pad_mask, text_feat, text_pad_mask, t,
                first_heading_angle=fha,
                cfg_type="nocfg",
            )

        mask_f = x_pad_mask.unsqueeze(-1).float()

        mse_with = F.mse_loss(pred_with * mask_f, motion * mask_f, reduction="none")
        mse_no = F.mse_loss(pred_no * mask_f, motion * mask_f, reduction="none")

        total_mse_with += mse_with.sum().item()
        total_mse_no += mse_no.sum().item()

        total_root_mse_with += mse_with[:, :, :root_dim].sum().item()
        total_root_mse_no += mse_no[:, :, :root_dim].sum().item()

        total_body_mse_with += mse_with[:, :, root_dim:].sum().item()
        total_body_mse_no += mse_no[:, :, root_dim:].sum().item()

        n_batches += 1
        if n_batches >= num_batches:
            break

    total_frames = total_elements / D
    metrics = {
        "num_batches": n_batches,
        "total_frames": int(total_frames),
        "mse_with_scene": round(total_mse_with / max(total_elements, 1), 6),
        "mse_no_scene": round(total_mse_no / max(total_elements, 1), 6),
        "root_mse_with": round(total_root_mse_with / max(total_frames * root_dim, 1), 6),
        "root_mse_no": round(total_root_mse_no / max(total_frames * root_dim, 1), 6),
        "body_mse_with": round(total_body_mse_with / max(total_frames * (D - root_dim), 1), 6),
        "body_mse_no": round(total_body_mse_no / max(total_frames * (D - root_dim), 1), 6),
        "delta_mse": round((total_mse_with - total_mse_no) / max(total_elements, 1), 6),
        "delta_root": round((total_root_mse_with - total_root_mse_no) / max(total_frames * root_dim, 1), 6),
        "delta_body": round((total_body_mse_with - total_body_mse_no) / max(total_frames * (D - root_dim), 1), 6),
    }

    return model, motion_rep, metrics


def render_viz_samples(model, motion_rep, exp_cfg, viz_segments, output_dir, device):
    print(f"\n  Rendering visualization samples for {exp_cfg['label']}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_level = 100

    for vi, seg in enumerate(viz_segments):
        scene_name = seg.get("scene_name", f"unknown_{vi}")
        text = seg.get("text", "no-text")

        motion_gt = seg["motion_features"].unsqueeze(0).to(device)
        n_frames = int(seg["length"]) if "length" in seg else int(seg["motion_mask"].sum().item())
        T_full = motion_gt.shape[1]
        mask = torch.zeros(1, T_full, device=device)
        mask[0, :n_frames] = 1.0
        voxel = seg["voxel_grid"].unsqueeze(0).to(device)
        B, T, D_full = motion_gt.shape

        if "text_feat" in seg:
            text_feat = seg["text_feat"].reshape(1, 1, -1).to(device)
            text_pad_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        else:
            text_feat, text_length = model.text_encoder([text])
            text_feat = text_feat.to(device)
            maxlen = text_feat.shape[1]
            text_pad_mask = torch.arange(maxlen, device=device).expand(1, maxlen) < text_length.to(device)

        with torch.no_grad():
            (sfr, smr), (sfb, smb) = model.encode_scene(voxel)

            t = torch.full((1,), noise_level, device=device)
            noise = torch.randn_like(motion_gt)
            x_t = model.diffusion.q_sample(motion_gt, t, noise=noise)

            x_pad_mask = mask.bool() if mask.dtype != torch.bool else mask
            fha = torch.zeros(1, device=device)

            pred_with = model.denoiser(
                [2.0, 2.0, 2.0],
                x_t, x_pad_mask, text_feat, text_pad_mask, t,
                first_heading_angle=fha,
                scene_feat_root=sfr, scene_mask_root=smr,
                scene_feat_body=sfb, scene_mask_body=smb,
                cfg_type="nocfg",
            )
            pred_no = model.denoiser(
                [2.0, 2.0],
                x_t, x_pad_mask, text_feat, text_pad_mask, t,
                first_heading_angle=fha,
                cfg_type="nocfg",
            )

        j_gt = features_to_joints(motion_rep, motion_gt[:, :n_frames, :], is_normalized=True)
        j_w = features_to_joints(motion_rep, pred_with[:, :n_frames, :], is_normalized=True)
        j_n = features_to_joints(motion_rep, pred_no[:, :n_frames, :], is_normalized=True)

        # --- SMPL22: GT from raw human_joints_aligned.npy, Pred converted from SOMA30 ---
        gt_meta = seg.get("_seg_meta", {})
        gt_raw = load_raw_gt_smpl22(gt_meta, max_frames=n_frames)
        n_gt = min(gt_raw.shape[0], n_frames)
        gt_raw = gt_raw[:n_gt]

        j_smpl_w = soma30_to_smpl22(np.squeeze(j_w["posed_joints"].astype(np.float32)))
        if j_smpl_w.ndim == 4:
            j_smpl_w = j_smpl_w.reshape(j_smpl_w.shape[1], -1, j_smpl_w.shape[-1])
        r_smpl_w = j_smpl_w[:, 0:1, :]  # pelvis as root

        j_smpl_n = soma30_to_smpl22(np.squeeze(j_n["posed_joints"].astype(np.float32)))
        if j_smpl_n.ndim == 4:
            j_smpl_n = j_smpl_n.reshape(j_smpl_n.shape[1], -1, j_smpl_n.shape[-1])
        r_smpl_n = j_smpl_n[:, 0:1, :]

        dict_gt = {"posed_joints": gt_raw / METER_TO_UNIT, "root_positions": gt_raw[:, 0:1, :] / METER_TO_UNIT}
        dict_w = {"posed_joints": j_smpl_w, "root_positions": r_smpl_w}
        dict_n = {"posed_joints": j_smpl_n, "root_positions": r_smpl_n}

        scene_segments = load_segmented_scene_pts(scene_name, n_points=8000)

        safe_name = f"sample{vi}_{scene_name}_{text[:25].replace(' ','_').replace('/','-')}"

        mp4_path = output_dir / f"recon_{safe_name}.mp4"
        if scene_segments:
            render_reconstruction_video(dict_gt, dict_w, scene_name, scene_segments, mp4_path,
                                        connections=SMPL_CONNECTIONS,
                                        label_gt="GT (SMPL22)", label_with="Pred (SMPL22)", fps=20)

        traj_path = output_dir / f"traj_{safe_name}.png"
        if scene_segments:
            render_reconstruction_trajectory(dict_gt, dict_w, dict_n, scene_name, scene_segments, traj_path)

        print(f"    [{vi}] scene={scene_name} | text={text[:40]} | frames={n_frames}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_viz_samples", type=int, default=3)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--exps", type=str, nargs="*", default=None,
                        help="Only run these experiments (keys), e.g. --exps stage1_root_only")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading LINGO val dataset...")
    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset, collate_fn
    from kimodo.model import load_model

    cache_dir = str(PROJECT_ROOT / "kimodo/kimodo_sceneco/cached_data")

    pretrained_cpu = load_model("Kimodo-SOMA-RP-v1.1", device="cpu")
    motion_dim = pretrained_cpu.denoiser.model.motion_rep.motion_rep_dim
    root_dim = pretrained_cpu.denoiser.model.motion_rep.global_root_dim
    print(f"  motion_dim={motion_dim}, root_dim={root_dim}")
    del pretrained_cpu

    ds_val = LINGOSceneMotionDataset(
        data_root=str(PROJECT_ROOT / "LINGO" / "dataset"),
        max_frames=196, min_frames=40,
        voxel_size=(64, 64, 64),
        train_ratio=0.9, seed=42,
        split="val",
        scene_dropout=0.0,
        cache_dir=cache_dir,
    )
    print(f"  Val dataset: {len(ds_val)} segments")

    viz_segments = sample_viz_segments(ds_val, num_samples=args.num_viz_samples, seed=123)
    print(f"  Viz samples: {len(viz_segments)} segments")
    for vi, vs in enumerate(viz_segments):
        sn = vs.get("scene_name", "?")
        tx = vs.get("text", "?")
        nf = int(vs["length"]) if "length" in vs else int(vs["motion_mask"].sum().item())
        print(f"    [{vi}] scene={sn}, frames={nf}, text={tx[:60]}")

    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )

    num_batches = min(args.num_batches, len(val_loader))
    print(f"  Evaluating {num_batches} batches ({num_batches * args.batch_size} samples)")

    output_root = PROJECT_ROOT / "kimodo_scene_project" / "outputs" / "eval_lingo"
    output_root.mkdir(parents=True, exist_ok=True)

    exps_to_run = list(EXPERIMENTS.keys()) if args.exps is None else args.exps

    all_metrics = {}
    for exp_key in exps_to_run:
        exp_cfg = EXPERIMENTS[exp_key]
        if not args.skip_metrics:
            model, motion_rep, metrics = evaluate_experiment(exp_cfg, device, val_loader, num_batches)
            all_metrics[exp_key] = metrics

            print(f"\n  [{exp_cfg['label']}]")
            print(f"    Total batches: {metrics['num_batches']}, frames: {metrics['total_frames']}")
            print(f"    MSE with scene:     {metrics['mse_with_scene']}")
            print(f"    MSE no scene:       {metrics['mse_no_scene']}")
            print(f"    Δ MSE (with - no):  {metrics['delta_mse']}")
            print(f"    Root MSE with:      {metrics['root_mse_with']}")
            print(f"    Root MSE no:        {metrics['root_mse_no']}")
            print(f"    Δ Root:             {metrics['delta_root']}")
            print(f"    Body MSE with:      {metrics['body_mse_with']}")
            print(f"    Body MSE no:        {metrics['body_mse_no']}")
            print(f"    Δ Body:             {metrics['delta_body']}")
        else:
            model = build_model(exp_cfg, device)
            motion_rep = model.denoiser.model.motion_rep

        viz_dir = output_root / exp_key / "viz"
        render_viz_samples(model, motion_rep, exp_cfg, viz_segments, viz_dir, device)

        del model
        torch.cuda.empty_cache()
        print(f"  Completed {exp_cfg['label']}")

    if not args.skip_metrics:
        metrics_path = output_root / "lingo_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"{'Experiment':<25s} {'MSE(with)':>10s} {'MSE(no)':>10s} {'Δ MSE':>10s} {'Δ Root':>10s} {'Δ Body':>10s}")
        print("-" * 75)
        for exp_key, exp_cfg in EXPERIMENTS.items():
            m = all_metrics[exp_key]
            print(f"{exp_cfg['label']:<25s} {m['mse_with_scene']:>10.4f} {m['mse_no_scene']:>10.4f} {m['delta_mse']:>10.4f} {m['delta_root']:>10.4f} {m['delta_body']:>10.4f}")

        print(f"\nMetrics: {metrics_path}")

    print(f"Viz:      {output_root}/*/viz/")
    print(f"Done.")


if __name__ == "__main__":
    main()
