#!/usr/bin/env python
"""2D root trajectory visualization for SMPLX rtdata models.

Compares predicted root trajectory (top-down X-Z, height X-Y, heading) against
ground truth from lingo_root_trajectory_smplx. Includes scene floor plan.

Fixes:
  - GT is z-scored; denormalized via global_root stats before plotting.
  - Scene voxel floor overlay on top-down view.
  - Handles both root_only and root_body checkpoints.

Usage:
    CUDA_VISIBLE_DEVICES=7 python kimodo_scene_project/eval/viz_root_trajectory_2d.py \
        --gpu 0 --num_samples 20
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_ROOT / "SOMA"))

os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
os.environ.setdefault("PYTHONHASHSEED", "0")


def load_global_root_stats():
    stats_dir = PROJECT_ROOT / "models" / "Kimodo-SMPLX-RP-v1" / "stats" / "motion" / "global_root"
    mean = np.load(stats_dir / "mean.npy")
    std = np.load(stats_dir / "std.npy")
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def denormalize_root(gt_normalized, mean, std):
    return gt_normalized * std + mean


def extract_scene_outline(voxel_grid, gt_root, pred_root, scene_phys_size=8.0):
    """Extract 2D floor outline from voxel grid.

    Maps voxel indices to world coordinates using a fixed physical extent
    centered at the origin. Both GT and pred trajectories are plotted in
    this same world frame — no per-sample squashing/stretching.

    Args:
        voxel_grid: [X, Y, Z] or [1, X, Y, Z] occupancy grid.
        gt_root: [T, 3] denormalized GT root positions.
        pred_root: [T, 3] predicted root positions.
        scene_phys_size: physical side length (meters) of the voxel cube.
    """
    v = voxel_grid.squeeze()
    if v.ndim != 3:
        return None, (0, 0, 0, 0)

    vx, vy, vz = v.shape

    floor_mask = np.zeros((vx, vz), dtype=bool)
    for zi in range(vz):
        for xi in range(vx):
            col = v[xi, :, zi]
            if (col > 0.5).any():
                floor_mask[xi, zi] = True

    if not floor_mask.any():
        return None, (0, 0, 0, 0)

    half = scene_phys_size / 2.0
    vox_half = vx / 2.0

    def voxel_to_world(xi, zi):
        wx = (xi - vox_half) * (half / vox_half)
        wz = (zi - vox_half) * (half / vox_half)
        return wx, wz

    outline_pts = []
    step = max(1, vx // 80)
    for xi in range(0, vx, step):
        for zi in range(0, vz, step):
            if floor_mask[xi, zi]:
                wx, wz = voxel_to_world(xi, zi)
                outline_pts.append([wx, wz])
    if not outline_pts:
        return None, (0, 0, 0, 0)

    all_x = np.concatenate([gt_root[:, 0], pred_root[:, 0],
                            np.array([p[0] for p in outline_pts])])
    all_z = np.concatenate([gt_root[:, 2], pred_root[:, 2],
                            np.array([p[1] for p in outline_pts])])
    xmin, xmax = all_x.min() - 0.5, all_x.max() + 0.5
    zmin, zmax = all_z.min() - 0.5, all_z.max() + 0.5
    bounds = (xmin, xmax, zmin, zmax)

    return np.array(outline_pts), bounds


def build_smplx_model(ckpt_path, device, use_in_root_model, use_in_body_model, dual_vit=True):
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    print(f"  Loading SMPLX base model (Kimodo-SMPLX-RP-v1)...")
    pretrained = load_model("Kimodo-SMPLX-RP-v1", device=str(device))
    inner_denoiser = pretrained.denoiser
    if hasattr(inner_denoiser, "model"):
        inner_denoiser = inner_denoiser.model

    print(f"  Building KimodoSceneCo (root={use_in_root_model}, body={use_in_body_model}, dual_vit={dual_vit})...")
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
            "use_dual_vit": dual_vit,
            "root_voxel_mode": "full",
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=use_in_root_model,
        use_in_body_model=use_in_body_model,
    )
    model = model.to(device)
    model.eval()

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    Missing keys: {len(missing)}")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")
    print(f"    Epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')}")
    return model


def generate_motion(model, prompt, num_frames, voxel_grid):
    with torch.no_grad():
        voxel_grid = voxel_grid.to(model.device)
        output = model(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=50,
            cfg_weight=[2.0, 2.0, 2.0],
            cfg_type="scene_separated",
            scene_input=voxel_grid,
            return_numpy=True,
        )
    root_pos = output["smooth_root_pos"]
    if root_pos.ndim == 3:
        root_pos = root_pos[0]
    return root_pos


def load_val_samples(max_samples=20):
    cache_dir = PROJECT_ROOT / "lingo_root_trajectory_smplx"
    npz_files = sorted(cache_dir.glob("seg_*.npz"))

    rng = np.random.RandomState(42)
    indices = list(range(len(npz_files)))
    rng.shuffle(indices)
    n_train = int(len(indices) * 0.9)
    val_indices = sorted(indices[n_train:])

    if len(val_indices) > max_samples:
        val_indices = val_indices[:max_samples]

    samples = []
    for idx in val_indices:
        f = npz_files[idx]
        data = np.load(str(f), allow_pickle=True)
        samples.append({
            "idx": idx,
            "path": str(f),
            "global_root_features": data["global_root_features"].astype(np.float32),
            "voxel_grid": data["voxel_grid"].astype(np.float32),
            "length": int(data["length"]),
            "scene_name": str(data["scene_name"]),
            "text": str(data.get("text", "motion")),
        })
    return samples


def plot_trajectory_2d(ax_xz, ax_xy, gt_root, pred_root, scene_name, text,
                        title_prefix, scene_outline, bounds):
    gt_x, gt_y, gt_z = gt_root[:, 0], gt_root[:, 1], gt_root[:, 2]
    pd_x, pd_y, pd_z = pred_root[:, 0], pred_root[:, 1], pred_root[:, 2]

    if scene_outline is not None:
        sx = scene_outline[:, 0]
        sy = scene_outline[:, 1]
        ax_xz.scatter(sx, sy, c="lightgray", s=0.3, alpha=0.4, rasterized=True, label="Scene")

    ax_xz.plot(gt_x, gt_z, "b-", linewidth=2.5, alpha=0.9, label="GT", zorder=5)
    ax_xz.plot(pd_x, pd_z, "r--", linewidth=2.5, alpha=0.9, label="Pred", zorder=5)
    ax_xz.scatter(gt_x[0], gt_z[0], c="blue", s=80, marker="o", edgecolors="white",
                  zorder=10, label="GT start")
    ax_xz.scatter(pd_x[0], pd_z[0], c="red", s=80, marker="s", edgecolors="white",
                  zorder=10, label="Pred start")
    ax_xz.scatter(gt_x[-1], gt_z[-1], c="darkblue", s=80, marker="D", edgecolors="white",
                  zorder=10, label="GT end")
    ax_xz.scatter(pd_x[-1], pd_z[-1], c="darkred", s=80, marker="D", edgecolors="white",
                  zorder=10, label="Pred end")
    ax_xz.set_xlabel("X (meters)")
    ax_xz.set_ylabel("Z (meters)")
    ax_xz.set_title(f"{title_prefix}\nTop-Down (X-Z) + Scene floor")
    ax_xz.legend(fontsize=7, loc="lower left", ncol=2)
    ax_xz.set_aspect("equal")
    ax_xz.grid(True, alpha=0.3)
    if bounds != (0, 0, 0, 0):
        ax_xz.set_xlim(bounds[0], bounds[1])
        ax_xz.set_ylim(bounds[2], bounds[3])

    ax_xy.plot(gt_x, gt_y, "b-", linewidth=2.5, alpha=0.9, label="GT")
    ax_xy.plot(pd_x, pd_y, "r--", linewidth=2.5, alpha=0.9, label="Pred")
    ax_xy.set_xlabel("X (meters)")
    ax_xy.set_ylabel("Y / Height (meters)")
    ax_xy.set_title("Side View (X-Y)")
    ax_xy.legend(fontsize=7, loc="upper right")
    ax_xy.grid(True, alpha=0.3)


def plot_heading_angle(ax, gt_root, pred_root):
    def compute_heading(root_pos):
        diff = root_pos[1:] - root_pos[:-1]
        heading = np.arctan2(diff[:, 2], diff[:, 0])
        heading = np.concatenate([heading[:1], heading])
        return np.degrees(heading)

    gt_heading = compute_heading(gt_root)
    pd_heading = compute_heading(pred_root)
    t = np.arange(len(gt_heading))
    ax.plot(t, gt_heading, "b-", linewidth=2, alpha=0.9, label="GT heading")
    ax.plot(t, pd_heading, "r--", linewidth=2, alpha=0.9, label="Pred heading")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Heading (°)")
    ax.set_title("Heading Angle")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_error_over_time(ax, gt_root, pred_root):
    errors = np.sqrt(np.sum((gt_root - pred_root) ** 2, axis=1))
    t = np.arange(len(errors))
    ax.fill_between(t, 0, errors, alpha=0.3, color="red")
    ax.plot(t, errors, "r-", linewidth=1.5, alpha=0.8)
    mean_err = np.mean(errors)
    ax.axhline(mean_err, color="darkred", linestyle=":", linewidth=1.5,
               label=f"mean={mean_err:.3f}m")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (meters)")
    ax.set_title(f"Root Position Error (mean={mean_err:.3f}m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="2D root trajectory visualization for SMPLX rtdata")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/viz_root_traj_2d")
    parser.add_argument("--ckpt_root_only", type=str,
                        default="kimodo_scene_project/outputs/smplx_root_only_rtdata/checkpoints/best_checkpoint.pt")
    parser.add_argument("--ckpt_root_body", type=str,
                        default="kimodo_scene_project/outputs/smplx_root_body_rtdata/checkpoints/best_checkpoint.pt")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated: root_only,root_body or all")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading global_root stats for denormalization...")
    mean_root, std_root = load_global_root_stats()

    print("\n" + "=" * 60)
    print("  Loading validation samples...")
    print("=" * 60)
    samples = load_val_samples(max_samples=args.num_samples)
    print(f"  Loaded {len(samples)} validation samples")

    requested = set(args.models.split(",")) if args.models != "all" else {"root_only", "root_body"}

    model_defs = []
    ckpt_root_only = Path(args.ckpt_root_only)
    if not ckpt_root_only.is_absolute():
        ckpt_root_only = PROJECT_ROOT / ckpt_root_only
    if "root_only" in requested and ckpt_root_only.exists():
        model_defs.append({
            "name": "root_only",
            "ckpt": ckpt_root_only,
            "use_in_root_model": True,
            "use_in_body_model": False,
            "dual_vit": True,
        })

    ckpt_root_body = Path(args.ckpt_root_body)
    if not ckpt_root_body.is_absolute():
        ckpt_root_body = PROJECT_ROOT / ckpt_root_body
    if "root_body" in requested and ckpt_root_body.exists():
        model_defs.append({
            "name": "root_body",
            "ckpt": ckpt_root_body,
            "use_in_root_model": True,
            "use_in_body_model": True,
            "dual_vit": True,
        })

    if not model_defs:
        print("ERROR: No checkpoints found!")
        return
    print(f"  Models to evaluate: {[m['name'] for m in model_defs]}")

    all_errors = {}

    for model_cfg in model_defs:
        print("\n" + "=" * 60)
        print(f"  Model: {model_cfg['name']}")
        print("=" * 60)

        model = build_smplx_model(
            model_cfg["ckpt"], str(device),
            model_cfg["use_in_root_model"], model_cfg["use_in_body_model"],
            model_cfg["dual_vit"],
        )

        model_output_dir = output_dir / model_cfg["name"]
        model_output_dir.mkdir(parents=True, exist_ok=True)

        err_list = []

        for si, sample in enumerate(tqdm(samples, desc=f"  {model_cfg['name']}")):
            scene_name = sample["scene_name"]
            text = sample["text"]
            num_frames = sample["length"]
            gt_norm = sample["global_root_features"][:, :3]
            voxel_grid = torch.from_numpy(sample["voxel_grid"]).unsqueeze(0).unsqueeze(0)

            gt_root = denormalize_root(gt_norm, mean_root[:3], std_root[:3])

            try:
                pred_root = generate_motion(model, text, num_frames, voxel_grid)

                n = min(gt_root.shape[0], pred_root.shape[0])
                gt_root_plot = gt_root[:n]
                pred_root_plot = pred_root[:n]

                scene_outline, bounds = extract_scene_outline(
                    sample["voxel_grid"], gt_root_plot, pred_root_plot,
                    scene_phys_size=8.0,
                )

                errors = np.sqrt(np.sum((gt_root_plot - pred_root_plot) ** 2, axis=1))
                err_list.append(np.mean(errors))

                fig, axes = plt.subplots(2, 2, figsize=(16, 13), facecolor="white")
                plot_trajectory_2d(axes[0, 0], axes[0, 1], gt_root_plot, pred_root_plot,
                                   scene_name, text, model_cfg["name"].upper(),
                                   scene_outline, bounds)
                plot_heading_angle(axes[1, 0], gt_root_plot, pred_root_plot)
                plot_error_over_time(axes[1, 1], gt_root_plot, pred_root_plot)

                safe_name = scene_name.replace("/", "_").replace(" ", "_")[:30]
                title = f'[{model_cfg["name"].upper()}]  {scene_name}  |  "{text[:60]}"'
                if len(text) > 60:
                    title += "..."
                fig.suptitle(title, fontsize=11, fontweight="bold")
                fig.tight_layout()

                out_path = model_output_dir / f"{si:02d}_{safe_name}.png"
                fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
                plt.close(fig)

            except Exception as e:
                print(f"\n  [ERROR] sample {si} ({scene_name}): {e}")
                import traceback
                traceback.print_exc()
                continue

        if err_list:
            all_errors[model_cfg["name"]] = (np.mean(err_list), np.std(err_list))
            print(f"\n  {model_cfg['name']}: mean_err={np.mean(err_list):.4f}m ± {np.std(err_list):.4f}m")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for name, (m, s) in sorted(all_errors.items()):
        print(f"  {name:20s}  {m:.4f}m ± {s:.4f}m")
    print(f"\n  Outputs: {output_dir}")
    print(f"  root_only: {output_dir}/root_only/")
    print(f"  root_body: {output_dir}/root_body/")
    print("=" * 60)


if __name__ == "__main__":
    main()
