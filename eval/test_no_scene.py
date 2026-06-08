#!/usr/bin/env python
"""Test SceneCo model without scene input vs original Kimodo.

Loads both the trained SceneCo checkpoint and original Kimodo model,
generates motions with the same text prompts (NO scene input),
compares outputs numerically, and renders MP4 videos side-by-side.

Usage:
    python kimodo_scene_project/eval/test_no_scene.py \
        --ckpt kimodo_scene_project/outputs/root_only_sceneco_gpu3/checkpoints/checkpoint_step110000.pt \
        --gpu 3
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
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

NVIDIA_FPS = 30
METER_TO_UNIT = 100

SKELETON_CONNECTIONS = [
    (0, 2), (0, 3), (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (3, 15), (15, 18), (18, 21),
    (2, 16), (16, 19), (19, 22),
    (16, 17), (19, 20), (22, 23),
    (15, 24), (24, 27), (27, 29),
    (18, 25), (25, 28), (28, 30),
    (24, 26), (27, 31),
    (25, 21), (28, 22),
]

TEST_PROMPTS = [
    ("walk forward in a straight line", 120),
    ("turn around and look back", 100),
    ("sit down on a chair", 100),
    ("run quickly then stop", 100),
    ("raise hands and wave", 80),
    ("bend down to pick up something", 100),
    ("jump up and down", 60),
    ("walk in a circle", 120),
]


def load_sceneco_model(ckpt_path: str, device: str):
    """Load trained SceneCo model from checkpoint."""
    from kimodo.model import load_model

    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=device)
    inner_denoiser = pretrained.denoiser
    if hasattr(inner_denoiser, "model"):
        inner_denoiser = inner_denoiser.model

    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    model = KimodoSceneCo(
        denoiser=inner_denoiser,
        text_encoder=pretrained.text_encoder,
        num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64),
            "patch_size": (8, 8, 8),
            "d_model": 256,
            "num_layers": 4,
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=True,
        use_in_body_model=False,
    )
    model = model.to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded SceneCo checkpoint: {ckpt_path}")
    print(f"  Step: {ckpt.get('global_step', 'N/A')}")

    alphas = {}
    for name, param in model.named_parameters():
        if "alpha" in name and param.numel() == 1:
            alphas[name.split(".")[-1]] = f"{param.item():.6f}"
    print(f"  Alphas: {alphas}")
    return model


def load_original_kimodo(device: str):
    """Load original Kimodo model (no SceneCo)."""
    from kimodo.model import load_model

    model = load_model("Kimodo-SOMA-RP-v1.1", device=device)
    model.eval()
    print("Loaded original Kimodo model")
    return model


def generate_motion_orig(model, prompt: str, num_frames: int, device: str):
    """Generate motion with original Kimodo (no scene support)."""
    with torch.no_grad():
        output = model(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=50,
            cfg_weight=[2.0, 2.0],
            return_numpy=True,
        )
    return output


def generate_motion_sceneco(model, prompt: str, num_frames: int, device: str):
    """Generate motion with SceneCo model WITHOUT scene input."""
    with torch.no_grad():
        output = model(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=50,
            cfg_weight=[2.0, 2.0, 2.0],
            scene_input=None,
            return_numpy=True,
        )
    return output


def prepare_3d_data(posed_joints, root_positions):
    num_joints = posed_joints.shape[1]
    joints_3d = np.zeros_like(posed_joints)
    joints_3d[:, :, 0] = posed_joints[:, :, 0] * METER_TO_UNIT
    joints_3d[:, :, 1] = posed_joints[:, :, 2] * METER_TO_UNIT
    joints_3d[:, :, 2] = posed_joints[:, :, 1] * METER_TO_UNIT

    root_3d = np.zeros_like(root_positions)
    root_3d[:, 0] = root_positions[:, 0] * METER_TO_UNIT
    root_3d[:, 1] = root_positions[:, 2] * METER_TO_UNIT
    root_3d[:, 2] = root_positions[:, 1] * METER_TO_UNIT
    return joints_3d, root_3d, num_joints


def _draw_skeleton_on_ax(ax, joints_3d, root_3d, frame_idx, num_joints,
                         bone_color, joint_color, root_trail_color, scope):
    j3d = joints_3d[frame_idx]

    for a, b in SKELETON_CONNECTIONS:
        if a < num_joints and b < num_joints:
            ax.plot([j3d[a, 0], j3d[b, 0]], [j3d[a, 1], j3d[b, 1]], [j3d[a, 2], j3d[b, 2]],
                    color=bone_color, linewidth=3, zorder=8, alpha=0.85)

    ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2],
               c=joint_color, s=40, depthshade=False, zorder=10,
               edgecolors='white', linewidths=0.8)

    if frame_idx > 0:
        trail_start = max(0, frame_idx - 30)
        trail = root_3d[trail_start:frame_idx + 1]
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                    color=root_trail_color, linewidth=2.5, alpha=0.75, zorder=6)

    ax.set_xlim(scope["x_min"], scope["x_max"])
    ax.set_ylim(scope["y_min"], scope["y_max"])
    ax.set_zlim(scope.get("z_min", 0), scope.get("z_max", 250))
    ax.set_axis_off()
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False


def render_comparison_video(out_orig, out_sceneco, output_path: str,
                            prompt: str, joint_rmse: float, root_rmse: float,
                            fps: int = 30):
    """Render side-by-side comparison MP4: Original Kimodo (left) vs SceneCo (right)."""
    joints_o, root_o, _ = prepare_3d_data(out_orig["posed_joints"], out_orig["root_positions"])
    joints_s, root_s, _ = prepare_3d_data(out_sceneco["posed_joints"], out_sceneco["root_positions"])

    num_frames = min(joints_o.shape[0], joints_s.shape[0])
    margin = 50
    x_min = min(joints_o[:, :, 0].min(), joints_s[:, :, 0].min()) - margin
    x_max = max(joints_o[:, :, 0].max(), joints_s[:, :, 0].max()) + margin
    y_min = min(joints_o[:, :, 1].min(), joints_s[:, :, 1].min()) - margin
    y_max = max(joints_o[:, :, 1].max(), joints_s[:, :, 1].max()) + margin
    z_min = min(joints_o[:, :, 2].min(), joints_s[:, :, 2].min()) - 20
    z_max = max(joints_o[:, :, 2].max(), joints_s[:, :, 2].max()) + 20
    scope = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
             "z_min": z_min, "z_max": z_max}

    fig = plt.figure(figsize=(22, 10), facecolor="black")
    ax_left = fig.add_subplot(121, projection="3d", facecolor="#0a0a1a")
    ax_right = fig.add_subplot(122, projection="3d", facecolor="#0a0a1a")
    fig.subplots_adjust(wspace=0.02)

    writer = FFMpegWriter(fps=fps)
    print(f"  Rendering comparison: {num_frames} frames → {output_path} ...")

    with writer.saving(fig, str(output_path), dpi=80):
        for frame_idx in tqdm(range(num_frames), desc="  rendering"):
            ax_left.cla()
            ax_right.cla()
            ax_left.set_facecolor("#0a0a1a")
            ax_right.set_facecolor("#0a0a1a")

            _draw_skeleton_on_ax(ax_left, joints_o, root_o, frame_idx, joints_o.shape[1],
                                 bone_color="#00DDFF", joint_color="#00DDFF",
                                 root_trail_color="#00DDFF", scope=scope)
            _draw_skeleton_on_ax(ax_right, joints_s, root_s, frame_idx, joints_s.shape[1],
                                 bone_color="#FF8800", joint_color="#FF8800",
                                 root_trail_color="#FF8800", scope=scope)

            ax_left.view_init(elev=20, azim=-55 + frame_idx * 0.4)
            ax_right.view_init(elev=20, azim=-55 + frame_idx * 0.4)

            fig.suptitle(f'"{prompt}"  |  joints RMSE={joint_rmse:.4f}  root RMSE={root_rmse:.4f}',
                         color="white", fontsize=13, y=0.02, fontweight="normal")
            ax_left.set_title("Original Kimodo", color="#00DDFF", fontsize=12, pad=8, fontweight="bold")
            ax_right.set_title("SceneCo (gate=0)", color="#FF8800", fontsize=12, pad=8, fontweight="bold")

            writer.grab_frame()

    plt.close()
    print(f"  Saved: {output_path}")
    return str(output_path)


def compare_numerically(out_orig: dict, out_sceneco: dict, prompt: str):
    """Compare two motion outputs numerically."""
    report = {"prompt": prompt}

    for key in ["posed_joints", "root_positions"]:
        if key in out_orig and key in out_sceneco:
            orig = out_orig[key]
            sc = out_sceneco[key]
            if orig.shape != sc.shape:
                report[f"{key}_shape_mismatch"] = f"orig={list(orig.shape)} scene={list(sc.shape)}"
                continue
            diff = np.abs(orig - sc)
            report[f"{key}_max_abs_diff"] = float(diff.max())
            report[f"{key}_mean_abs_diff"] = float(diff.mean())
            report[f"{key}_rmse"] = float(np.sqrt((diff ** 2).mean()))

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="SceneCo checkpoint path")
    parser.add_argument("--gpu", type=int, default=3, help="GPU device ID")
    parser.add_argument("--output", type=str, default="kimodo_scene_project/outputs/test_no_scene",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="DDIM denoising steps")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load models
    print("\n" + "=" * 60)
    print("  LOADING MODELS")
    print("=" * 60)
    model_sceneco = load_sceneco_model(str(ckpt_path), device)
    model_orig = load_original_kimodo(device)

    # Generate + compare
    results = []
    print("\n" + "=" * 60)
    print("  GENERATING MOTIONS (NO SCENE INPUT)")
    print("=" * 60)

    for prompt, num_frames in TEST_PROMPTS:
        print(f"\n--- Prompt: \"{prompt}\" ({num_frames} frames) ---")

        out_orig = generate_motion_orig(model_orig, prompt, num_frames, device)
        out_sceneco = generate_motion_sceneco(model_sceneco, prompt, num_frames, device)

        comparison = compare_numerically(out_orig, out_sceneco, prompt)
        print(f"  Posed joints RMSE: {comparison.get('posed_joints_rmse', 'N/A'):.6f}" if 'posed_joints_rmse' in comparison else f"  Key: {[k for k in comparison if 'rmse' in k]}")
        print(f"  Root positions RMSE: {comparison.get('root_positions_rmse', 'N/A'):.6f}" if 'root_positions_rmse' in comparison else "")

        safe_name = prompt.replace(" ", "_")[:40]
        vid_path = str(output_dir / f"comparison_{safe_name}.mp4")

        render_comparison_video(
            out_orig, out_sceneco,
            vid_path, prompt,
            joint_rmse=comparison.get("posed_joints_rmse", 0),
            root_rmse=comparison.get("root_positions_rmse", 0),
            fps=args.fps,
        )

        comparison.update({
            "prompt": prompt,
            "num_frames": num_frames,
            "video_comparison": vid_path,
        })
        results.append(comparison)

    # Summary report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    all_joint_rmse = []
    all_root_rmse = []
    for r in results:
        j_rmse = r.get("posed_joints_rmse")
        r_rmse = r.get("root_positions_rmse")
        if j_rmse is not None:
            all_joint_rmse.append(j_rmse)
        if r_rmse is not None:
            all_root_rmse.append(r_rmse)
        print(f"  \"{r['prompt']}\": joints_rmse={j_rmse:.6f}, root_rmse={r_rmse:.6f}" if j_rmse else f"  \"{r['prompt']}\": N/A")

    if all_joint_rmse:
        print(f"\n  Avg joints RMSE:  {np.mean(all_joint_rmse):.6f} ± {np.std(all_joint_rmse):.6f}")
    if all_root_rmse:
        print(f"  Avg root RMSE:    {np.mean(all_root_rmse):.6f} ± {np.std(all_root_rmse):.6f}")

    print(f"\n  Report:  {report_path}")
    print(f"  Videos:  {output_dir}/")
    print("=" * 60)
    print("✅ Test complete!")


if __name__ == "__main__":
    main()
