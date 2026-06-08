#!/usr/bin/env python
"""实验：LINGO 原始 SMPL24/SMPLX22 关节点 与 SOMA30 关节点映射对比可视化。

LINGO 数据集中的原始 motion 使用 SMPL/SMPLX 骨架，而 Kimodo 的 inverse
解码出的是 SOMA30 关节。本脚本将两者同时可视化并生成 MP4 对比视频。

Usage:
  cd kimodo-viser
  PYTHONPATH=kimodo_scene_project:kimodo:$PYTHONPATH \
    python kimodo_scene_project/eval/exp_root_body_soma_smpl.py --gpu 0
"""

import argparse
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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
CACHE_DIR = PROJECT_ROOT / "kimodo" / "kimodo_sceneco" / "cached_data"
OUTPUT_DIR = Path("exp_root_to_body_output")

# ===== SMPL 24 关节连接 (从 SMPL parent hierarchy) =====
# SMPL parents: [ -1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,15,15 ]
# Names: 0=pelvis, 1=Lhip, 2=Rhip, 3=spine1, 4=Lknee, 5=Rknee,
#        6=spine2, 7=Lankle, 8=Rankle, 9=spine3, 10=Lfoot, 11=Rfoot,
#        12=neck, 13=Lcollar, 14=Rcollar, 15=head, 16=Lshoulder, 17=Rshoulder,
#        18=Lelbow, 19=Relbow, 20=Lwrist, 21=Rwrist, 22=jaw, 23=leye
SMPL24_CONNECTIONS = [
    (0, 1), (1, 4), (4, 7), (7, 10),       # left leg chain
    (0, 2), (2, 5), (5, 8), (8, 11),       # right leg chain
    (0, 3), (3, 6), (6, 9), (9, 12),       # spine chain
    (12, 15), (15, 22), (15, 23),           # head chain
    (9, 13), (13, 16), (16, 18), (18, 20),  # left arm (from spine3)
    (9, 14), (14, 17), (17, 19), (19, 21),  # right arm (from spine3)
]

# ===== SMPL 22 关节连接 (无 jaw/leye) =====
SMPL22_CONNECTIONS = [
    (0, 1), (1, 4), (4, 7), (7, 10),       # left leg chain
    (0, 2), (2, 5), (5, 8), (8, 11),       # right leg chain
    (0, 3), (3, 6), (6, 9), (9, 12),       # spine chain
    (12, 15),                               # head
    (9, 13), (13, 16), (16, 18), (18, 20),  # left arm (from spine3)
    (9, 14), (14, 17), (17, 19), (19, 21),  # right arm (from spine3)
]

# ===== SOMA30 连接 (从 bone_order_names_with_parents 推导) =====
# 0=Hips, 1=Spine1, 2=Spine2, 3=Chest, 4=Neck1, 5=Neck2, 6=Head, 7=Jaw, 8=LeftEye,
# 9=RightEye, 10=LeftShoulder, 11=LeftArm, 12=LeftForeArm, 13=LeftHand,
# 14=LeftHandThumbEnd, 15=LeftHandMiddleEnd, 16=RightShoulder, 17=RightArm,
# 18=RightForeArm, 19=RightHand, 20=RightHandThumbEnd, 21=RightHandMiddleEnd,
# 22=LeftLeg, 23=LeftShin, 24=LeftFoot, 25=LeftToeBase,
# 26=RightLeg, 27=RightShin, 28=RightFoot, 29=RightToeBase
SOMA30_CONNECTIONS = [
    # spine: Hips → Spine1 → Spine2 → Chest
    (0, 1), (1, 2), (2, 3),
    # neck/head: Chest → Neck1 → Neck2 → Head
    (3, 4), (4, 5), (5, 6),
    # head extras: Head → Jaw, Head → LeftEye, Head → RightEye
    (6, 7), (6, 8), (6, 9),
    # left arm: Chest → LShoulder → LArm → LForeArm → LHand
    (3, 10), (10, 11), (11, 12), (12, 13),
    # left hand extras: LHand → LThumbEnd, LHand → LMiddleEnd
    (13, 14), (13, 15),
    # right arm: Chest → RShoulder → RArm → RForeArm → RHand
    (3, 16), (16, 17), (17, 18), (18, 19),
    # right hand extras: RHand → RThumbEnd, RHand → RMiddleEnd
    (19, 20), (19, 21),
    # left leg: Hips → LLeg → LShin → LFoot → LToeBase
    (0, 22), (22, 23), (23, 24), (24, 25),
    # right leg: Hips → RLeg → RShin → RFoot → RToeBase
    (0, 26), (26, 27), (27, 28), (28, 29),
]

# ===== SMPL22 -> SOMA30 映射 =====
# SMPL22 joints → SOMA30 joints
# SMPL22: 0=pelvis, 1=Lhip, 2=Rhip, 3=spine1, 4=Lknee, 5=Rknee,
#          6=spine2, 7=Lankle, 8=Rankle, 9=spine3, 10=Lfoot, 11=Rfoot,
#          12=neck, 13=Lcollar, 14=Rcollar, 15=head, 16=Lshoulder, 17=Rshoulder,
#          18=Lelbow, 19=Relbow, 20=Lwrist, 21=Rwrist
# SOMA30: 0=Hips, 1=Spine1, 2=Spine2, 3=Chest, 4=Neck1, 5=Neck2, 6=Head,
#          10=LeftShoulder, 11=LeftArm, 12=LeftForeArm, 13=LeftHand,
#          16=RightShoulder, 17=RightArm, 18=RightForeArm, 19=RightHand,
#          22=LeftLeg, 23=LeftShin, 24=LeftFoot, 25=LeftToeBase,
#          26=RightLeg, 27=RightShin, 28=RightFoot, 29=RightToeBase
SMPL22_TO_SOMA30 = {
    0: 0,    # pelvis → Hips
    1: 22,   # Lhip → LeftLeg
    2: 26,   # Rhip → RightLeg
    3: 1,    # spine1 → Spine1
    4: 23,   # Lknee → LeftShin
    5: 27,   # Rknee → RightShin
    6: 2,    # spine2 → Spine2
    7: 24,   # Lankle → LeftFoot
    8: 28,   # Rankle → RightFoot
    9: 3,    # spine3 → Chest
    10: 25,  # Lfoot → LeftToeBase
    11: 29,  # Rfoot → RightToeBase
    12: 4,   # neck → Neck1
    13: 10,  # Lcollar → LeftShoulder
    14: 16,  # Rcollar → RightShoulder
    15: 6,   # head → Head
    16: 11,  # Lshoulder → LeftArm
    17: 17,  # Rshoulder → RightArm
    18: 12,  # Lelbow → LeftForeArm
    19: 18,  # Relbow → RightForeArm
    20: 13,  # Lwrist → LeftHand
    21: 19,  # Rwrist → RightHand
}


def load_pretrained_denoiser(device):
    """加载预训练 Kimodo-SOMA-RP-v1.1, 返回 TwostageDenoiser 和 motion_rep."""
    from kimodo.model import load_model
    print("Loading pretrained Kimodo-SOMA-RP-v1.1 ...")
    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=str(device))
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model
    denoiser = inner
    motion_rep = denoiser.motion_rep
    from kimodo_sceneco.model.diffusion import Diffusion
    diffusion = Diffusion(num_base_steps=1000)
    print(f"  motion_rep_dim={motion_rep.motion_rep_dim}, "
          f"global_root_dim={motion_rep.global_root_dim}")
    return denoiser, motion_rep, diffusion


def load_lingo_sample(seg_name=None, device="cpu"):
    """加载单个 LINGO 缓存样本."""
    if seg_name:
        path = CACHE_DIR / seg_name
    else:
        available = sorted(CACHE_DIR.glob("seg_*.npz"))
        path = available[len(available) // 3]
        seg_name = path.name

    if not path.exists():
        raise FileNotFoundError(f"LINGO sample not found: {path}")

    data = np.load(str(path), allow_pickle=True)
    motion = torch.from_numpy(data["motion_features"]).float().unsqueeze(0).to(device)
    length = int(data["length"])
    text = str(data["text"])
    text_feat = torch.from_numpy(data["text_feat"]).float().to(device)
    scene_name = str(data["scene_name"])
    print(f"  Loaded: {seg_name}, T={length}, text='{text}', scene={scene_name}")
    return motion[:, :length], length, text, text_feat, scene_name


def run_full_forward(denoiser, motion, text_feat, timestep, device):
    """完整 TwostageDenoiser 前向传播."""
    denoiser.eval()
    B, T, D = motion.shape
    x_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    fha = torch.zeros(B, device=device)
    motion_mask = torch.zeros_like(motion)
    observed_motion = torch.zeros_like(motion)
    with torch.no_grad():
        output = denoiser(
            motion, x_pad_mask, text_feat, text_pad_mask, timestep,
            first_heading_angle=fha,
            motion_mask=motion_mask, observed_motion=observed_motion,
        )
    return output


def run_body_only_from_gt_root(denoiser, x_noisy, motion_gt, text_feat, timestep, device):
    """用 GT root 驱动 body_model."""
    denoiser.eval()
    B, T, D = x_noisy.shape
    motion_rep = denoiser.motion_rep
    x_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    fha = torch.zeros(B, device=device)
    lengths = x_pad_mask.sum(-1)

    root_gt = motion_gt[:, :, :motion_rep.global_root_dim].clone()
    with torch.no_grad():
        root_motion_local = motion_rep.global_root_to_local_root(
            root_gt, normalized=True, lengths=lengths,
        )

    body_x = x_noisy[:, :, motion_rep.body_slice]
    x_new = torch.cat([root_motion_local, body_x], axis=-1)
    motion_mask = torch.zeros_like(x_noisy)
    x_new_extended = torch.cat([x_new, motion_mask], axis=-1)

    with torch.no_grad():
        predicted_body = denoiser.body_model(
            x_new_extended, x_pad_mask, text_feat, text_pad_mask,
            timestep, first_heading_angle=fha,
        )
    return predicted_body


def features_to_joints(motion_rep, features):
    """将 motion features 解码为 posed_joints + root_positions."""
    with torch.no_grad():
        out = motion_rep.inverse(features, is_normalized=True, return_numpy=False)
    return {
        "posed_joints": out["posed_joints"].cpu().numpy(),
        "root_positions": out["root_positions"].cpu().numpy(),
    }


def _fill_nan_joints(joints):
    """Per-joint per-channel NaN fill using linear interpolation."""
    filled = joints.copy()
    T, J, D = filled.shape
    for j_idx in range(J):
        for d_idx in range(D):
            col = filled[:, j_idx, d_idx]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue
            valid = np.where(~nan_mask)[0]
            if len(valid) == 0:
                continue
            col[nan_mask] = np.interp(np.where(nan_mask)[0], valid, col[valid])
            filled[:, j_idx, d_idx] = col
    return filled


def _prepare_3d(joints, roots):
    """与 batch_eval_lingo.py 完全一致的 3D 数据准备。"""
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
    return jr, rr


def map_soma30_to_smpl22(soma30_joints):
    """将 SOMA30 关节 (T, 30, 3) 映射到 SMPL22 关节 (T, 22, 3).
    
    只取 SOMA30 中与 SMPL22 有对应关系的关节。
    """
    T = soma30_joints.shape[0]
    smpl22_joints = np.zeros((T, 22, 3), dtype=np.float32)
    for smpl_idx, soma_idx in SMPL22_TO_SOMA30.items():
        if soma_idx < soma30_joints.shape[1]:
            smpl22_joints[:, smpl_idx, :] = soma30_joints[:, soma_idx, :]
    return smpl22_joints


def _draw_skeleton(ax, joints, roots, fi, color, root_color, connections):
    pos = joints[fi]
    n_joints = pos.shape[0]
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=[color] * n_joints, s=30, depthshade=False, zorder=10,
        edgecolors="white", linewidths=0.5,
    )
    for a, b in connections:
        if a < n_joints and b < n_joints:
            ax.plot(
                [pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], [pos[a, 2], pos[b, 2]],
                color=color, linewidth=2.0, zorder=8,
            )
    if roots.ndim == 2:
        rp = roots[fi]
    else:
        rp = roots[fi, 0]
    ax.scatter(
        [rp[0]], [rp[1]], [rp[2]],
        c=[root_color], s=60, depthshade=False, zorder=11, marker="s",
        edgecolors="white",
    )


def render_soma_smpl_comparison(dict_gt_soma, dict_body_soma, dict_full_soma,
                                output_path, fps=20):
    """渲染 SOMA30 vs 映射到 SMPL22 的并排视频。
    
    左列: GT SOMA30 (原始)
    中列: Full Model SOMA30 → 映射到 SMPL22 (用 SMPL 连接绘制)
    右列: Body(GT root) SOMA30 → 映射到 SMPL22 (用 SMPL 连接绘制)
    """
    import av
    from PIL import Image
    from tqdm import tqdm

    # 使用与 batch_eval_lingo.py 一致的 _prepare_3d
    j_gt, r_gt = _prepare_3d(
        dict_gt_soma["posed_joints"], dict_gt_soma["root_positions"],
    )
    j_full_soma, r_full = _prepare_3d(
        dict_full_soma["posed_joints"], dict_full_soma["root_positions"],
    )
    j_body_soma, r_body = _prepare_3d(
        dict_body_soma["posed_joints"], dict_body_soma["root_positions"],
    )

    # 映射到 SMPL22
    j_full_smpl = map_soma30_to_smpl22(j_full_soma)
    j_body_smpl = map_soma30_to_smpl22(j_body_soma)

    nf = min(j_gt.shape[0], j_full_smpl.shape[0], j_body_smpl.shape[0])

    # 计算视口 (用 GT SOMA30)
    all_pts = j_gt[:nf].reshape(-1, 3)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if len(all_pts) == 0:
        center = np.zeros(3)
        spread = 200
    else:
        center = np.mean(all_pts, axis=0)
        spread = np.max(np.abs(all_pts - center)) + 30

    scope = {
        "x_min": center[0] - spread, "x_max": center[0] + spread,
        "y_min": center[1] - spread, "y_max": center[1] + spread,
        "z_min": center[2] - spread * 0.5, "z_max": center[2] + spread,
    }

    fig = plt.figure(figsize=(21, 7), facecolor="white")
    ax1 = fig.add_subplot(131, projection="3d", facecolor="white")
    ax2 = fig.add_subplot(132, projection="3d", facecolor="white")
    ax3 = fig.add_subplot(133, projection="3d", facecolor="white")
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

    for fi in tqdm(range(nf), desc="  Rendering MP4"):
        for ax in [ax1, ax2, ax3]:
            ax.cla()
            ax.set_facecolor("white")

        # 左: GT SOMA30
        _draw_skeleton(ax1, j_gt, r_gt, fi, "#4CAF50", "#1B5E20", SOMA30_CONNECTIONS)
        # 中: Full Model → SMPL22
        _draw_skeleton(ax2, j_full_smpl, r_full, fi, "#2196F3", "#0D47A1", SMPL22_CONNECTIONS)
        # 右: Body(GT root) → SMPL22
        _draw_skeleton(ax3, j_body_smpl, r_body, fi, "#FF5722", "#BF360C", SMPL22_CONNECTIONS)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(scope["x_min"], scope["x_max"])
            ax.set_ylim(scope["y_min"], scope["y_max"])
            ax.set_zlim(scope["z_min"], scope["z_max"])
            ax.set_axis_off()
            ax.view_init(elev=60, azim=-60)

        ax1.set_title("GT (SOMA30)", fontsize=12, fontweight="bold", color="#4CAF50")
        ax2.set_title("Full Model (SMPL22)", fontsize=12, fontweight="bold", color="#2196F3")
        ax3.set_title("Body(GT root) (SMPL22)", fontsize=12, fontweight="bold", color="#FF5722")

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


def render_soma_only_comparison(dict_gt_soma, dict_body_soma, dict_full_soma,
                                output_path, fps=20):
    """渲染纯 SOMA30 三列对比视频 (与原来相同)."""
    import av
    from PIL import Image
    from tqdm import tqdm

    j_gt, r_gt = _prepare_3d(
        dict_gt_soma["posed_joints"], dict_gt_soma["root_positions"],
    )
    j_body, r_body = _prepare_3d(
        dict_body_soma["posed_joints"], dict_body_soma["root_positions"],
    )
    j_full, r_full = _prepare_3d(
        dict_full_soma["posed_joints"], dict_full_soma["root_positions"],
    )

    nf = min(j_gt.shape[0], j_body.shape[0], j_full.shape[0])

    all_pts = j_gt[:nf].reshape(-1, 3)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if len(all_pts) == 0:
        center = np.zeros(3)
        spread = 200
    else:
        center = np.mean(all_pts, axis=0)
        spread = np.max(np.abs(all_pts - center)) + 30

    scope = {
        "x_min": center[0] - spread, "x_max": center[0] + spread,
        "y_min": center[1] - spread, "y_max": center[1] + spread,
        "z_min": center[2] - spread * 0.5, "z_max": center[2] + spread,
    }

    fig = plt.figure(figsize=(21, 7), facecolor="white")
    ax_gt = fig.add_subplot(131, projection="3d", facecolor="white")
    ax_body = fig.add_subplot(132, projection="3d", facecolor="white")
    ax_full = fig.add_subplot(133, projection="3d", facecolor="white")
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

    for fi in tqdm(range(nf), desc="  Rendering MP4"):
        for ax in [ax_gt, ax_body, ax_full]:
            ax.cla()
            ax.set_facecolor("white")

        _draw_skeleton(ax_gt, j_gt, r_gt, fi, "#4CAF50", "#1B5E20", SOMA30_CONNECTIONS)
        _draw_skeleton(ax_body, j_body, r_body, fi, "#FF5722", "#BF360C", SOMA30_CONNECTIONS)
        _draw_skeleton(ax_full, j_full, r_full, fi, "#2196F3", "#0D47A1", SOMA30_CONNECTIONS)

        for ax in [ax_gt, ax_body, ax_full]:
            ax.set_xlim(scope["x_min"], scope["x_max"])
            ax.set_ylim(scope["y_min"], scope["y_max"])
            ax.set_zlim(scope["z_min"], scope["z_max"])
            ax.set_axis_off()
            ax.view_init(elev=60, azim=-60)

        ax_gt.set_title("GT (SOMA30)", fontsize=12, fontweight="bold", color="#4CAF50")
        ax_body.set_title("Body(GT root) (SOMA30)", fontsize=12, fontweight="bold", color="#FF5722")
        ax_full.set_title("Full Model (SOMA30)", fontsize=12, fontweight="bold", color="#2196F3")

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


def render_root_trajectory(dict_gt, dict_body, dict_full, output_path):
    """渲染 root 2D 轨迹对比图."""
    _, r_gt = _prepare_3d(
        dict_gt["posed_joints"], dict_gt["root_positions"],
    )
    _, r_body = _prepare_3d(
        dict_body["posed_joints"], dict_body["root_positions"],
    )
    _, r_full = _prepare_3d(
        dict_full["posed_joints"], dict_full["root_positions"],
    )

    nf = min(r_gt.shape[0], r_body.shape[0], r_full.shape[0])
    roots_gt = r_gt[:nf, 0, :2]
    roots_body = r_body[:nf, 0, :2]
    roots_full = r_full[:nf, 0, :2]

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(roots_gt[:, 0], roots_gt[:, 1], color="#4CAF50", linewidth=2, label="GT", zorder=10)
    ax.plot(roots_body[:, 0], roots_body[:, 1], color="#FF5722", linewidth=2,
            label="Body(GT root)", zorder=10)
    ax.plot(roots_full[:, 0], roots_full[:, 1], color="#2196F3", linewidth=2,
            label="Full Model", zorder=10)
    ax.scatter(roots_gt[0, 0], roots_gt[0, 1], color="#4CAF50", s=80, marker="o", zorder=11)
    ax.scatter(roots_gt[-1, 0], roots_gt[-1, 1], color="#1B5E20", s=100, marker="*", zorder=11)

    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("Root Trajectory — GT vs Body(GT root) vs Full Model",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seg", type=str, default=None,
                        help="LINGO cached segment name (auto-selects if None)")
    parser.add_argument("--timestep", type=int, default=500,
                        help="Diffusion timestep for denoising (0=no noise, 500=mid)")
    parser.add_argument("--output_dir", type=str, default="exp_root_to_body_output")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("实验: GT root → local_root → body_model (Stage2) - SMPL/SOMA 映射")
    print(f"Device: {device}, Timestep: {args.timestep}")
    print("=" * 60)

    # ===== 1. 加载模型 =====
    denoiser, motion_rep, diffusion = load_pretrained_denoiser(device)

    # ===== 2. 加载 LINGO 数据 =====
    motion, T, text, text_feat, scene_name = load_lingo_sample(args.seg, device)
    print(f"  motion shape: {motion.shape}")

    # ===== 3. 固定 seed 加噪 =====
    torch.manual_seed(42)
    timestep = torch.tensor([args.timestep], device=device)
    noise = torch.randn_like(motion)
    x_t = diffusion.q_sample(motion, timestep, noise=noise)

    # ===== 4. 完整 forward =====
    print("Running full forward (both stages)...")
    output_full = run_full_forward(denoiser, x_t, text_feat, timestep, device)

    root_full = output_full[:, :, :motion_rep.global_root_dim]
    body_full = output_full[:, :, motion_rep.global_root_dim:]

    # ===== 5. body_only (GT root) =====
    print("Running body_only with GT root...")
    body_pred = run_body_only_from_gt_root(
        denoiser, x_t, motion, text_feat, timestep, device,
    )
    del x_t

    # 组合: GT root + body_model(GT root) 的 body
    root_gt_feature = motion[:, :, :motion_rep.global_root_dim]
    output_body = torch.cat([root_gt_feature, body_pred], axis=-1)

    # ===== 6. 解码到关节 =====
    print("Decoding to joints...")
    joints_gt = features_to_joints(motion_rep, motion)
    joints_body = features_to_joints(motion_rep, output_body)
    joints_full = features_to_joints(motion_rep, output_full)

    # ===== 7. 计算指标 =====
    print("\n" + "=" * 60)
    print("指标对比")
    print("=" * 60)

    B, T_dim, D = motion.shape
    mask_f = torch.ones(1, T_dim, 1, device=device)

    root_gt = motion[:, :, :motion_rep.global_root_dim]
    mse_root_full = F.mse_loss(root_full * mask_f, root_gt * mask_f).item()
    body_gt = motion[:, :, motion_rep.global_root_dim:]
    mse_body_body = F.mse_loss(body_pred * mask_f, body_gt * mask_f).item()
    mse_body_full = F.mse_loss(body_full * mask_f, body_gt * mask_f).item()

    print(f"  Root MSE (full model vs GT):    {mse_root_full:.6f}")
    print(f"  Body MSE (body-only vs GT):      {mse_body_body:.6f}")
    print(f"  Body MSE (full model vs GT):     {mse_body_full:.6f}")
    print(f"  Body diff (body_only - full):    {mse_body_body - mse_body_full:.6f}")

    # ===== 8. 可视化 =====
    print("\n" + "=" * 60)
    print("渲染可视化...")
    print("=" * 60)

    safe_label = scene_name.replace('/', '_').replace('.', '_')

    # Root 轨迹图
    traj_path = output_dir / f"root_trajectory_{safe_label}_t{args.timestep}.png"
    render_root_trajectory(joints_gt, joints_body, joints_full, traj_path)
    print(f"  Root trajectory: {traj_path}")

    # SOMA30 三列对比视频
    soma_video = output_dir / f"soma30_{safe_label}_t{args.timestep}.mp4"
    render_soma_only_comparison(joints_gt, joints_body, joints_full, soma_video)
    print(f"  SOMA30 video: {soma_video}")

    # SOMA30 vs SMPL22 映射对比视频
    smpl_video = output_dir / f"smpl22_{safe_label}_t{args.timestep}.mp4"
    render_soma_smpl_comparison(joints_gt, joints_body, joints_full, smpl_video)
    print(f"  SMPL22 video: {smpl_video}")

    print("\n" + "=" * 60)
    print("实验完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
