#!/usr/bin/env python
"""实验：从 LINGO 数据集中提取 GT root → local_root → 传入 body_model (Stage2)。

验证用 GT root 驱动 body_model 生成的结果与原始 GT 身体的差异。
输出：
  - 终端: MSE 指标对比
  - PNG: root 轨迹 2D 图
  - MP4: GT vs Body-Only (GT root) vs Full Model 并排视频

Usage:
  cd kimodo-viser
  PYTHONPATH=kimodo_scene_project:kimodo:$PYTHONPATH \
    python kimodo_scene_project/eval/exp_root_to_body.py --gpu 0
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

SOMA30_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 7), (6, 8), (6, 9),
    (3, 10), (10, 11), (11, 12), (12, 13), (13, 14), (13, 15),
    (3, 16), (16, 17), (17, 18), (18, 19), (19, 20), (19, 21),
    (0, 22), (22, 23), (23, 24), (24, 25),
    (0, 26), (26, 27), (27, 28), (28, 29),
]

OUTPUT_DIR = Path("exp_root_to_body_output")
CACHE_DIR = PROJECT_ROOT / "kimodo" / "kimodo_sceneco" / "cached_data"


# ==================== 模型加载 ====================

def load_pretrained_denoiser(device):
    """加载预训练 Kimodo-SOMA-RP-v1.1, 返回 TwostageDenoiser 和 motion_rep."""
    from kimodo.model import load_model
    print("Loading pretrained Kimodo-SOMA-RP-v1.1 ...")
    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=str(device))
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model
    # 确保是 TwostageDenoiser
    denoiser = inner
    motion_rep = denoiser.motion_rep

    # 加载 diffusion
    from kimodo_sceneco.model.diffusion import Diffusion
    diffusion = Diffusion(num_base_steps=1000)

    print(f"  motion_rep_dim={motion_rep.motion_rep_dim}, "
          f"global_root_dim={motion_rep.global_root_dim}, "
          f"local_root_dim={motion_rep.local_root_dim}")
    return denoiser, motion_rep, diffusion


# ==================== 数据加载 ====================

def load_lingo_sample(seg_name=None, device="cpu"):
    """加载单个 LINGO 缓存样本. 返回 (motion, length, text, text_feat, scene_name)."""
    if seg_name:
        path = CACHE_DIR / seg_name
    else:
        available = sorted(CACHE_DIR.glob("seg_*.npz"))
        path = available[len(available) // 3]  # 取中间位置的样本
        seg_name = path.name

    if not path.exists():
        raise FileNotFoundError(f"LINGO sample not found: {path}")

    data = np.load(str(path), allow_pickle=True)
    motion = torch.from_numpy(data["motion_features"]).float().unsqueeze(0).to(device)  # [1, T, 369]
    length = int(data["length"])
    text = str(data["text"])
    text_feat = torch.from_numpy(data["text_feat"]).float().to(device)
    scene_name = str(data["scene_name"])
    print(f"  Loaded: {seg_name}, T={length}, text='{text}', scene={scene_name}")
    return motion[:, :length], length, text, text_feat, scene_name


# ==================== 前向推理 ====================

def run_full_forward(denoiser, motion, text_feat, timestep, device):
    """完整 TwostageDenoiser 前向传播 (eval 模式)."""
    denoiser.eval()
    B, T, D = motion.shape

    # 加噪
    x_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    fha = torch.zeros(B, device=device)

    motion_mask = torch.zeros_like(motion)
    observed_motion = torch.zeros_like(motion)

    with torch.no_grad():
        output = denoiser(
            motion,
            x_pad_mask,
            text_feat,
            text_pad_mask,
            timestep,
            first_heading_angle=fha,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
        )
    return output  # [B, T, 369]


def run_body_only_from_gt_root(denoiser, x_noisy, motion_gt, text_feat, timestep, device):
    """只用 body_model, 输入 GT root (转 local_root) + noisy body, 得到 body 预测.

    Args:
        denoiser: TwostageDenoiser
        x_noisy: [B, T, D] 加噪后的运动特征, 取其 body 部分
        motion_gt: [B, T, D] 干净的 GT 运动特征, 取其 root 部分
    """
    denoiser.eval()
    B, T, D = x_noisy.shape
    motion_rep = denoiser.motion_rep

    x_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    fha = torch.zeros(B, device=device)
    lengths = x_pad_mask.sum(-1)

    # GT global root: 前 5 维 (从干净的 GT motion 中提取!)
    root_gt = motion_gt[:, :, :motion_rep.global_root_dim].clone()

    # 转为 local_root (normalized, 因为 GT motion 是 normalized 的)
    with torch.no_grad():
        root_motion_local = motion_rep.global_root_to_local_root(
            root_gt, normalized=True, lengths=lengths,
        )

    # noisy body 特征 (从 x_noisy 中提取)
    body_x = x_noisy[:, :, motion_rep.body_slice]  # [B, T, 364]

    # 拼接: GT local_root + noisy body_x
    x_new = torch.cat([root_motion_local, body_x], axis=-1)

    # concat motion_mask
    motion_mask = torch.zeros_like(x_noisy)
    x_new_extended = torch.cat([x_new, motion_mask], axis=-1)

    with torch.no_grad():
        predicted_body = denoiser.body_model(
            x_new_extended,
            x_pad_mask,
            text_feat,
            text_pad_mask,
            timestep,
            first_heading_angle=fha,
        )

    return predicted_body  # [B, T, 364]


# ==================== 解码到关节 ====================

def features_to_joints(motion_rep, features):
    """将 motion features 解码为 posed_joints + root_positions (numpy)."""
    with torch.no_grad():
        out = motion_rep.inverse(features, is_normalized=True, return_numpy=False)
    return {
        "posed_joints": out["posed_joints"].cpu().numpy(),
        "root_positions": out["root_positions"].cpu().numpy(),
    }


# ==================== 可视化 ====================

def _draw_skeleton(ax, joints, roots, fi, color, root_color, connections=SOMA30_CONNECTIONS):
    pos = joints[fi]
    n_joints = pos.shape[0]
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=[color] * n_joints, s=30, depthshade=False, zorder=10, edgecolors="white", linewidths=0.5,
    )
    for a, b in connections:
        if a < n_joints and b < n_joints:
            ax.plot(
                [pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], [pos[a, 2], pos[b, 2]],
                color=color, linewidth=2.0, zorder=8,
            )
    rp = roots[fi, 0]
    ax.scatter(
        [rp[0]], [rp[1]], [rp[2]],
        c=[root_color], s=60, depthshade=False, zorder=11, marker="s", edgecolors="white",
    )


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
            col[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid, col[valid]
            )
            filled[:, j_idx, d_idx] = col
    return filled


def render_comparison_video(dict_gt, dict_body, dict_full, output_path, label_gt="GT",
                            label_body="Body(GT root)", label_full="Full Model", fps=20):
    """渲染 GT vs Body-Only vs Full Model 三列视频."""
    import av
    from PIL import Image
    from tqdm import tqdm

    j_gt = np.squeeze(dict_gt["posed_joints"].astype(np.float32)) * METER_TO_UNIT
    r_gt = np.squeeze(dict_gt["root_positions"].astype(np.float32)) * METER_TO_UNIT
    j_body = np.squeeze(dict_body["posed_joints"].astype(np.float32)) * METER_TO_UNIT
    r_body = np.squeeze(dict_body["root_positions"].astype(np.float32)) * METER_TO_UNIT
    j_full = np.squeeze(dict_full["posed_joints"].astype(np.float32)) * METER_TO_UNIT
    r_full = np.squeeze(dict_full["root_positions"].astype(np.float32)) * METER_TO_UNIT

    # 用线性插值填充 NaN
    j_gt = _fill_nan_joints(j_gt)
    j_body = _fill_nan_joints(j_body)
    j_full = _fill_nan_joints(j_full)

    if r_gt.ndim == 2:
        r_gt = r_gt[:, None, :]
        r_body = r_body[:, None, :]
        r_full = r_full[:, None, :]

    nf = min(j_gt.shape[0], j_body.shape[0], j_full.shape[0])

    all_pts = j_gt[:nf].reshape(-1, 3)
    # 过滤 NaN/Inf
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

        _draw_skeleton(ax_gt, j_gt, r_gt, fi, "#4CAF50", "#1B5E20")
        _draw_skeleton(ax_body, j_body, r_body, fi, "#FF5722", "#BF360C")
        _draw_skeleton(ax_full, j_full, r_full, fi, "#2196F3", "#0D47A1")

        for ax in [ax_gt, ax_body, ax_full]:
            ax.set_xlim(scope["x_min"], scope["x_max"])
            ax.set_ylim(scope["y_min"], scope["y_max"])
            ax.set_zlim(scope["z_min"], scope["z_max"])
            ax.set_axis_off()
            ax.view_init(elev=60, azim=-60)

        ax_gt.set_title(label_gt, fontsize=12, fontweight="bold", color="#4CAF50")
        ax_body.set_title(label_body, fontsize=12, fontweight="bold", color="#FF5722")
        ax_full.set_title(label_full, fontsize=12, fontweight="bold", color="#2196F3")

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
    r_gt = np.squeeze(dict_gt["root_positions"].astype(np.float32)) * METER_TO_UNIT
    r_body = np.squeeze(dict_body["root_positions"].astype(np.float32)) * METER_TO_UNIT
    r_full = np.squeeze(dict_full["root_positions"].astype(np.float32)) * METER_TO_UNIT

    if r_gt.ndim == 2:
        r_gt = r_gt[:, None, :]
        r_body = r_body[:, None, :]
        r_full = r_full[:, None, :]

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


# ==================== 主逻辑 ====================

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
    print("实验: GT root → local_root → body_model (Stage2)")
    print(f"Device: {device}, Timestep: {args.timestep}")
    print("=" * 60)

    # ===== 1. 加载模型 =====
    denoiser, motion_rep, diffusion = load_pretrained_denoiser(device)

    # ===== 2. 加载 LINGO 数据 =====
    motion, T, text, text_feat, scene_name = load_lingo_sample(args.seg, device)
    seg_label = args.seg if args.seg else scene_name
    print(f"  motion shape: {motion.shape}")

    # ===== 3. 固定 seed 加噪 (保证可复现) =====
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
    # 使用与完整 forward 完全相同的 x_t，只是 body_model 用 GT root 而非 root_model 的预测
    print("Running body_only with GT root...")
    body_pred = run_body_only_from_gt_root(
        denoiser, x_t, motion, text_feat, timestep, device,
    )
    del x_t

    # 组合 body_only 输出: GT root + body_model(GT root) 的 body
    # 这样可视化时 "Body(GT root)" 的 root 才是 GT root，而非 full model 的 root
    root_gt_feature = motion[:, :, :motion_rep.global_root_dim]
    output_body = torch.cat([root_gt_feature, body_pred], axis=-1)

    # ===== 6. 解码到关节 =====
    print("Decoding to joints...")
    joints_gt = features_to_joints(motion_rep, motion)          # GT (clean)
    joints_body = features_to_joints(motion_rep, output_body)    # body_only
    joints_full = features_to_joints(motion_rep, output_full)    # full model

    # ===== 7. 计算指标 =====
    print("\n" + "=" * 60)
    print("指标对比")
    print("=" * 60)

    B, T_dim, D = motion.shape
    mask_f = torch.ones(1, T_dim, 1, device=device)

    # Root MSE
    root_gt = motion[:, :, :motion_rep.global_root_dim]
    mse_root_body = F.mse_loss(root_full * mask_f, root_gt * mask_f).item()
    mse_root_full = F.mse_loss(root_full * mask_f, root_gt * mask_f).item()

    # Body MSE (GT body vs predicted body)
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

    # Root 轨迹图 (使用 scene_name 作为文件名)
    safe_label = scene_name.replace('/', '_').replace('.', '_')
    traj_path = output_dir / f"root_trajectory_{safe_label}_t{args.timestep}.png"
    render_root_trajectory(joints_gt, joints_body, joints_full, traj_path)
    print(f"  Root trajectory: {traj_path}")

    # 三列对比视频
    video_path = output_dir / f"comparison_{safe_label}_t{args.timestep}.mp4"
    render_comparison_video(joints_gt, joints_body, joints_full, video_path, fps=20)
    print(f"  Comparison video: {video_path}")

    print("\n" + "=" * 60)
    print("实验完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
