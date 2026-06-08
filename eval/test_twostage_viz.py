#!/usr/bin/env python
"""Test original Kimodo two-stage outputs: visualize both root trajectory and body motion.

Stage 1 (root_model): 5D global root trajectory [smooth_root_pos(x,y,z), heading(cos,sin)]
  -> Visualized as wide 2D top-down trajectory plot (PNG) + heading arrows
Stage 2 (body_model): 364D body features + root -> full body motion
  -> Visualized as single-panel top-down MP4: full skeleton + Stage 1 trajectory overlay

Usage:
    CUDA_VISIBLE_DEVICES=7 python kimodo_scene_project/eval/test_twostage_viz.py \
        --prompt "a person walks forward" --gpu 0
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
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

SCALE = 100.0

TEST_PROMPTS = [
    "a person walks forward in a straight line",
    "a person turns around and walks back",
    "a person walks in a circle",
    "a person runs then stops",
    "a person crouches and moves sideways",
]


class InterceptOutput:
    def __init__(self):
        self.root_pred = None
        self.body_pred = None
        self.motion_rep = None


def build_hooked_forward(original_forward, capture: InterceptOutput):
    def hooked_forward(self, x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                       first_heading_angle=None, motion_mask=None, observed_motion=None,
                       scene_feat=None, scene_mask=None):
        if self.motion_mask_mode == "concat":
            if motion_mask is None or observed_motion is None:
                motion_mask = torch.zeros_like(x)
                observed_motion = torch.zeros_like(x)
            x = x * (1 - motion_mask) + observed_motion * motion_mask
            x_extended = torch.cat([x, motion_mask], axis=-1)
        else:
            x_extended = x

        has_sceneco = hasattr(self.root_model, 'sceneco_layers') and self.root_model.sceneco_layers
        if has_sceneco:
            root_motion_pred = self.root_model(
                x_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle,
                scene_feat=scene_feat, scene_mask=scene_mask,
            )
        else:
            root_motion_pred = self.root_model(
                x_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle,
            )

        capture.root_pred = root_motion_pred.detach().clone()

        lengths = x_pad_mask.sum(-1)

        convert_ctx = torch.no_grad() if self.training else contextlib.nullcontext()
        with convert_ctx:
            root_motion_local = self.motion_rep.global_root_to_local_root(
                root_motion_pred, normalized=True, lengths=lengths,
            )

        body_x = x[..., self.motion_rep.body_slice]
        x_new = torch.cat([root_motion_local, body_x], axis=-1)

        if self.motion_mask_mode == "concat":
            x_new_extended = torch.cat([x_new, motion_mask], axis=-1)
        else:
            x_new_extended = x_new

        body_has_sceneco = hasattr(self.body_model, 'sceneco_layers') and self.body_model.sceneco_layers
        if body_has_sceneco:
            predicted_body = self.body_model(
                x_new_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle,
                scene_feat=scene_feat, scene_mask=scene_mask,
            )
        else:
            predicted_body = self.body_model(
                x_new_extended, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle,
            )

        capture.body_pred = predicted_body.detach().clone()
        capture.motion_rep = self.motion_rep

        output = torch.cat([root_motion_pred, predicted_body], axis=-1)
        return output

    return hooked_forward


def load_kimodo_with_hook(device):
    from kimodo.model import load_model

    model = load_model("Kimodo-SOMA-RP-v1.1", device=str(device))
    model.eval()

    inner_denoiser = model.denoiser
    if hasattr(inner_denoiser, "model"):
        inner_denoiser = inner_denoiser.model

    capture = InterceptOutput()

    import types
    inner_denoiser.forward = types.MethodType(
        build_hooked_forward(inner_denoiser.forward, capture), inner_denoiser
    )

    return model, capture


def get_skeleton_connections(model):
    skel = model.output_skeleton
    if hasattr(skel, 'joint_parents'):
        p = skel.joint_parents.cpu().numpy()
        return [(int(p[i]), i) for i in range(len(p)) if p[i] >= 0]
    return []


def generate_motion_with_hook(model, capture, prompt, num_frames, num_denoising_steps, device):
    with torch.no_grad():
        output = model(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=num_denoising_steps,
            return_numpy=True,
        )

    root_pred = capture.root_pred
    body_pred = capture.body_pred
    motion_rep = capture.motion_rep

    if root_pred is not None:
        root_pred = root_pred.cpu()
    if body_pred is not None:
        body_pred = body_pred.cpu()

    return output, root_pred, body_pred, motion_rep


def decode_root_trajectory(root_pred, motion_rep):
    if root_pred is None or motion_rep is None:
        return None
    root_pred_np = root_pred[0].numpy()
    return {
        "smooth_root_pos": root_pred_np[:, 0:3],
        "heading_angle": np.arctan2(root_pred_np[:, 4], root_pred_np[:, 3]),
    }


# ============================================================
#  Stage 1: Root Trajectory Plot
# ============================================================

def render_root_trajectory(root_data, output_path, prompt, arrow_step=10):
    if root_data is None:
        print("  No root trajectory data to render")
        return

    pos = root_data["smooth_root_pos"]
    heading = root_data["heading_angle"]

    fig, (ax_traj, ax_heading) = plt.subplots(1, 2, figsize=(20, 8),
                                               facecolor="white",
                                               gridspec_kw={"width_ratios": [2, 1]})

    # ---- Left: wide top-down trajectory ----
    x, z = pos[:, 0], pos[:, 2]
    x_pad = max((x.max() - x.min()) * 0.25, 0.3)
    z_pad = max((z.max() - z.min()) * 0.25, 0.3)

    ax_traj.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax_traj.set_ylim(z.min() - z_pad, z.max() + z_pad)
    ax_traj.set_aspect("equal")
    ax_traj.set_facecolor("#fafafa")

    # trajectory line with gradient color
    n = len(x)
    for i in range(n - 1):
        alpha = 0.4 + 0.6 * i / n
        ax_traj.plot(x[i:i + 2], z[i:i + 2], color="#1a73e8", linewidth=3, alpha=alpha)

    # heading arrows
    for i in range(0, n, arrow_step):
        dx = np.cos(heading[i]) * 0.06
        dz = np.sin(heading[i]) * 0.06
        ax_traj.arrow(x[i], z[i], dx, dz,
                      head_width=0.03, head_length=0.04,
                      fc="#ff6d01", ec="#ff6d01", alpha=0.75, zorder=9)

    # start / end markers
    ax_traj.scatter(x[0], z[0], color="#34a853", s=200, marker="o",
                    zorder=12, edgecolors="white", linewidths=2, label="Start")
    ax_traj.scatter(x[-1], z[-1], color="#ea4335", s=250, marker="*",
                    zorder=12, edgecolors="white", linewidths=2, label="End")

    ax_traj.set_xlabel("X (m)", fontsize=13)
    ax_traj.set_ylabel("Z (m)", fontsize=13)
    ax_traj.set_title("Stage 1 — Root Trajectory (Top-Down)", fontsize=15, fontweight="bold", pad=12)
    ax_traj.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax_traj.grid(True, alpha=0.25, linestyle="--")

    # ---- Right: heading angle ----
    frames = np.arange(len(heading))
    deg = np.degrees(np.unwrap(heading))
    ax_heading.fill_between(frames, 0, deg, alpha=0.12, color="#1a73e8")
    ax_heading.plot(frames, deg, color="#1a73e8", linewidth=2.5)
    ax_heading.axhline(y=0, color="#ccc", linestyle="--", alpha=0.6)
    ax_heading.set_xlabel("Frame", fontsize=13)
    ax_heading.set_ylabel("Heading (°)", fontsize=13)
    ax_heading.set_title("Root Heading Angle", fontsize=15, fontweight="bold", pad=12)
    ax_heading.grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(f'Root Trajectory — "{prompt[:70]}"',
                 fontsize=12, color="#666", y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Stage 1 saved: {output_path}")


# ============================================================
#  Stage 2: Motion MP4 — Top-Down with Trajectory Overlay
# ============================================================

def _draw_skeleton_xz(ax, joints_xz, connections, color, linewidth=2.5, joint_size=20):
    for a, b in connections:
        if a < joints_xz.shape[0] and b < joints_xz.shape[0]:
            ax.plot([joints_xz[a, 0], joints_xz[b, 0]],
                    [joints_xz[a, 1], joints_xz[b, 1]],
                    color=color, linewidth=linewidth, alpha=0.92, zorder=8,
                    solid_capstyle="round")
    ax.scatter(joints_xz[:, 0], joints_xz[:, 1],
               c=color, s=joint_size, zorder=10, edgecolors="white", linewidths=0.4)


def render_motion_mp4(output, root_data, output_path, prompt, connections, fps=30):
    posed_joints = np.squeeze(output["posed_joints"].astype(np.float32))
    root_positions = np.squeeze(output["root_positions"].astype(np.float32))

    if posed_joints.ndim == 4:
        posed_joints = posed_joints[0]
    if root_positions.ndim == 3:
        root_positions = root_positions[0]

    j3d = posed_joints * SCALE
    r3d = root_positions * SCALE
    j3d[..., 1], j3d[..., 2] = j3d[..., 2].copy(), j3d[..., 1].copy()
    r3d[..., 1], r3d[..., 2] = r3d[..., 2].copy(), r3d[..., 1].copy()

    nf = j3d.shape[0]

    if root_data is not None:
        traj_pos = root_data["smooth_root_pos"] * SCALE
        heading = root_data["heading_angle"]
    else:
        traj_pos = r3d.copy()
        heading = np.zeros(nf)
    traj_x, traj_z = traj_pos[:, 0], traj_pos[:, 2]
    arrow_step = max(1, nf // 15)

    # ---- scope: union of full skeleton spread + trajectory bounds ----
    # skeleton XZ extent (all frames, all joints)
    all_j_xz = j3d.reshape(-1, 3)[:, [0, 2]]
    x_min_s = all_j_xz[:, 0].min()
    x_max_s = all_j_xz[:, 0].max()
    z_min_s = all_j_xz[:, 1].min()
    z_max_s = all_j_xz[:, 1].max()

    x_min = min(x_min_s, traj_x.min())
    x_max = max(x_max_s, traj_x.max())
    z_min = min(z_min_s, traj_z.min())
    z_max = max(z_max_s, traj_z.max())

    spread_x = x_max - x_min
    spread_z = z_max - z_min
    x_pad = max(spread_x * 0.15, 0.5)
    z_pad = max(spread_z * 0.15, 0.5)
    scope = {
        "x_min": x_min - x_pad, "x_max": x_max + x_pad,
        "z_min": z_min - z_pad, "z_max": z_max + z_pad,
    }

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
    ax.set_facecolor("#fafafa")

    writer = FFMpegWriter(fps=fps)
    writer.setup(fig, str(output_path), dpi=110)
    print(f"  Stage 2 MP4: {nf} frames")

    for fi in tqdm(range(nf), desc="  MP4"):
        ax.cla()
        ax.set_facecolor("#fafafa")

        # ---- trajectory background ----
        ax.plot(traj_x[:fi + 1], traj_z[:fi + 1],
                color="#1a73e8", linewidth=3.5, alpha=0.7, zorder=3)
        if fi < nf - 1:
            ax.plot(traj_x[fi:], traj_z[fi:],
                    color="#bbb", linewidth=2, alpha=0.2, linestyle="--", zorder=2)

        for i in range(0, nf, arrow_step):
            dx = np.cos(heading[i]) * 0.08
            dz = np.sin(heading[i]) * 0.08
            c = "#1a73e8" if i <= fi else "#ccc"
            a = 0.55 if i <= fi else 0.2
            ax.arrow(traj_x[i], traj_z[i], dx, dz,
                     head_width=0.03, head_length=0.04,
                     fc=c, ec=c, alpha=a, zorder=4)

        ax.scatter(traj_x[0], traj_z[0], color="#34a853", s=200, marker="o",
                   zorder=15, edgecolors="white", linewidths=2)
        ax.scatter(traj_x[-1], traj_z[-1], color="#ea4335", s=250, marker="*",
                   zorder=15, edgecolors="white", linewidths=2)

        # ---- skeleton XZ projection ----
        joints_xz = j3d[fi][:, [0, 2]]
        _draw_skeleton_xz(ax, joints_xz, connections, "#1a73e8", linewidth=2.8, joint_size=25)

        # ---- root marker ----
        rp = r3d[fi]
        ax.scatter(rp[0], rp[1], c="#ea4335", s=120, marker="s",
                   zorder=14, edgecolors="white", linewidths=1.5)

        # ---- root trail ----
        trail_s = max(0, fi - 50)
        trail = r3d[trail_s:fi + 1]
        if len(trail) >= 2:
            ax.plot(trail[:, 0], trail[:, 1],
                    color="#ff6d01", linewidth=3, alpha=0.45, zorder=5)

        # ---- layout ----
        ax.set_xlim(scope["x_min"], scope["x_max"])
        ax.set_ylim(scope["z_min"], scope["z_max"])
        ax.set_aspect("equal")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Z", fontsize=12)
        ax.grid(True, alpha=0.15, linestyle="--")
        ax.set_title(
            f'Stage 2 — Top-Down Motion with Stage 1 Trajectory\n"{prompt[:65]}"',
            fontsize=13, fontweight="bold", pad=10, color="#333",
        )

        pct = (fi + 1) / nf * 100
        ax.text(0.98, 0.02, f"Frame {fi+1}/{nf} ({pct:.0f}%)",
                transform=ax.transAxes, fontsize=10, color="#888",
                ha="right", va="bottom")

        writer.grab_frame()

    writer.finish()
    plt.close(fig)
    print(f"  Stage 2 saved: {output_path}")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Kimodo two-stage visualization")
    parser.add_argument("--prompt", type=str, default="a person walks forward in a straight line")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--denoising-steps", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="kimodo_scene_project/outputs/twostage_viz")
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frames: {args.num_frames}  Steps: {args.denoising_steps}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading Kimodo ...")
    model, capture = load_kimodo_with_hook(device)

    # Get correct skeleton connections from model
    connections = get_skeleton_connections(model)
    print(f"  Skeleton connections: {len(connections)}")

    prompts_to_run = TEST_PROMPTS if args.all_prompts else [args.prompt]

    for prompt in prompts_to_run:
        safe_name = prompt[:40].replace(" ", "_").replace(".", "")
        print(f"\n{'=' * 60}")
        print(f'  "{prompt}"')
        print(f"{'=' * 60}")

        output, root_pred, body_pred, motion_rep = generate_motion_with_hook(
            model, capture, prompt, args.num_frames, args.denoising_steps, device
        )

        print(f"  Output joints: {output['posed_joints'].shape}  "
              f"root: {output['root_positions'].shape}")

        # Stage 1: decode + render root trajectory
        root_data = decode_root_trajectory(root_pred, motion_rep)
        if root_data is not None:
            xr = root_data["smooth_root_pos"]
            print(f"  [Stage 1] Trajectory X:[{xr[:,0].min():.3f},{xr[:,0].max():.3f}]  "
                  f"Z:[{xr[:,2].min():.3f},{xr[:,2].max():.3f}]")
            render_root_trajectory(root_data, output_dir / f"stage1_traj_{safe_name}.png", prompt)
        else:
            print("  [Stage 1] WARNING: no root data captured!")

        # Stage 2: render motion MP4
        render_motion_mp4(output, root_data, output_dir / f"stage2_motion_{safe_name}.mp4",
                          prompt, connections, fps=args.fps)

    print(f"\n{'=' * 60}")
    print(f"  Done → {output_dir}")
    print(f"  stage1_traj_*.png   — root trajectory")
    print(f"  stage2_motion_*.mp4 — top-down skeleton on trajectory")


if __name__ == "__main__":
    main()
