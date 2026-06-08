#!/usr/bin/env python
"""2D Top-Down Trajectory Visualization for SceneCo experiments (exp1-4).

Overlays root trajectory on XY-projected scene voxel occupancy,
comparing WITH scene vs WITHOUT scene vs original KiMoDo.
Usage: python -m kimodo_sceneco.exp.vis_trajectory_2d --gpu 6
"""

import logging
import os
import argparse
import glob

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("vis_traj2d")


def setup_env():
    os.environ.setdefault("CHECKPOINT_DIR",
                          "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_original_kimodo(device="cpu"):
    from kimodo.model import load_model as load_kimodo_model
    return load_kimodo_model("Kimodo-SOMA-RP-v1.1", device=device)


def load_sceneco_model(exp_type, checkpoint_path, text_encoder=None, device="cpu"):
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location="cpu")
    exp_type = ckpt.get("exp_type", exp_type)
    from kimodo.model import load_model as load_kimodo_model

    class _DummyTE:
        def __call__(self, t): raise RuntimeError()
        def to(self, d): return self
        def eval(self): return self

    kimodo_pretrained = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device="cpu",
                                          text_encoder=_DummyTE())
    scene_config = {
        'voxel_size': (64, 64, 64), 'patch_size': (8, 8, 8),
        'in_channels': 1, 'd_model': 256, 'num_heads': 4, 'num_layers': 4, 'ff_dim': 512,
    }

    if exp_type == "exp1":
        from kimodo_sceneco.exp.exp1_monkey_patch import KimodoSceneCoExp1
        pretrained_denoiser = kimodo_pretrained.denoiser.model
        model = KimodoSceneCoExp1(
            denoiser=pretrained_denoiser, text_encoder=text_encoder,
            num_base_steps=1000, scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_config, device=_torch.device("cpu"),
            cfg_type="scene_separated",
        )
    else:
        from kimodo_sceneco.exp.exp2_rewrite_layer import KimodoSceneCoExp2
        from kimodo_sceneco.exp.exp2_rewrite_layer.backbone_exp2 import TransformerEncoderBlock
        from kimodo_sceneco.exp.exp2_rewrite_layer.twostage_denoiser_exp2 import TwostageDenoiser as TDE2

        pretrained_denoiser = kimodo_pretrained.denoiser.model
        pretrained_root = pretrained_denoiser.root_model
        pretrained_body = pretrained_denoiser.body_model
        motion_rep = pretrained_denoiser.motion_rep
        motion_mask_mode = pretrained_denoiser.motion_mask_mode

        root_use = exp_type != "exp4"
        body_use = exp_type != "exp3"

        def _extract(block, use_sc):
            return dict(
                latent_dim=block.latent_dim, ff_size=block.ff_size,
                num_layers=block.num_layers, num_heads=block.num_heads,
                activation=block.activation, dropout=block.dropout,
                pe_dropout=block.pe_dropout,
                norm_first=getattr(block, 'norm_first', False),
                llm_shape=[1, block.embed_text.in_features],
                use_text_mask=block.use_text_mask,
                num_text_tokens_override=getattr(block, 'num_text_tokens_override', None),
                input_first_heading_angle=block.input_first_heading_angle,
                scene_feat_dim=256, use_sceneco=use_sc, sceneco_dropout=0.1,
            )

        root_config = _extract(pretrained_root, root_use)
        root_config["input_dim"] = pretrained_root.input_linear.in_features
        root_config["output_dim"] = pretrained_root.output_linear.out_features
        root_config["skeleton"] = motion_rep.skeleton

        body_config = _extract(pretrained_body, body_use)
        body_config["input_dim"] = pretrained_body.input_linear.in_features
        body_config["output_dim"] = pretrained_body.output_linear.out_features
        body_config["skeleton"] = motion_rep.skeleton

        new_root = TransformerEncoderBlock(**root_config)
        new_body = TransformerEncoderBlock(**body_config)
        new_denoiser = TDE2(motion_rep=motion_rep, motion_mask_mode=motion_mask_mode)
        new_denoiser.root_model = new_root
        new_denoiser.body_model = new_body

        model = KimodoSceneCoExp2(
            denoiser=new_denoiser, text_encoder=text_encoder,
            num_base_steps=1000, scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_config, device=_torch.device("cpu"),
            cfg_type="scene_separated",
        )
        model._load_and_migrate_pretrained(new_denoiser, pretrained_denoiser)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    target_device = _torch.device(device)
    model.to(target_device)
    model.device = target_device
    model.eval()
    return model


@torch.no_grad()
def generate_original(model, texts, num_frames, num_steps=50, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return model(
        prompts=texts, num_frames=num_frames,
        num_denoising_steps=num_steps, cfg_weight=[2.0, 2.0],
        return_numpy=False, progress_bar=lambda x: x,
    )


@torch.no_grad()
def generate_sceneco(model, texts, num_frames, scene_input=None,
                     num_steps=50, cfg_weight=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg_weight is None:
        cfg_weight = [2.0, 2.0, 2.0]
    return model(
        prompts=texts, num_frames=num_frames,
        num_denoising_steps=num_steps, cfg_weight=cfg_weight,
        return_numpy=False, progress_bar=lambda x: x,
        scene_input=scene_input,
    )


def extract_root_trajectory(out, max_frames=None):
    pj = out["posed_joints"].cpu().numpy()
    if pj.ndim == 4:
        pj = pj[0]
    root = pj[:, 0, :].copy()
    if max_frames and root.shape[0] > max_frames:
        root = root[:max_frames]
    return root


def voxel_xy_occupancy(voxel_grid, voxel_size=0.1):
    vox = voxel_grid.squeeze().cpu().numpy()
    D = vox.shape[0]
    occ = np.max(vox, axis=0)
    H, W = occ.shape
    extent = [0, W * voxel_size, 0, H * voxel_size]
    return occ, extent


def plot_one_trajectory(ax, root_xy, scene_occ, extent, title,
                        color="#1a73e8", lw=2.5):
    ax.imshow(scene_occ, origin='lower', extent=extent,
              cmap='Greys', alpha=0.25, interpolation='nearest')
    ax.contourf(np.linspace(extent[0], extent[1], scene_occ.shape[1]),
                np.linspace(extent[2], extent[3], scene_occ.shape[0]),
                scene_occ, levels=[0.5, 1.0], colors=['#cccccc'], alpha=0.4)

    x = root_xy[:, 0]
    y = root_xy[:, 1]

    displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

    x_mid = (x.min() + x.max()) / 2
    y_mid = (y.min() + y.max()) / 2
    x_span = max(np.ptp(x), 0.3)
    y_span = max(np.ptp(y), 0.3)
    view_range = max(x_span, y_span) * 1.8

    ax.plot(x, y, color=color, linewidth=lw, alpha=0.9, zorder=5)
    ax.scatter([x[0]], [y[0]], color='green', s=120, zorder=6,
               edgecolors='black', linewidth=1.5, label='Start')
    ax.scatter([x[-1]], [y[-1]], color='red', s=120, zorder=6,
               edgecolors='black', linewidth=1.5, label='End')

    for i in range(0, len(x), max(1, len(x) // 12)):
        ax.scatter([x[i]], [y[i]], color=color, s=30, alpha=0.7, zorder=5,
                   edgecolors='white', linewidth=0.5)

    ax.set_xlim(x_mid - view_range, x_mid + view_range)
    ax.set_ylim(y_mid - view_range, y_mid + view_range)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    ax.text(0.02, 0.96, f"Δ = {displacement:.3f}m", transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=color,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor=color))

    return displacement


def load_scene_samples(cache_dir, device, num_samples=6):
    all_files = sorted(glob.glob(os.path.join(cache_dir, "seg_*.npz")))
    import random
    random.seed(42)
    selected = sorted(random.sample(all_files, min(num_samples, len(all_files))))
    samples = []
    for fp in selected:
        data = np.load(fp, allow_pickle=True)
        voxel = torch.from_numpy(data["voxel_grid"]).float()
        if voxel.ndim == 3:
            voxel = voxel.unsqueeze(0)
        samples.append({
            "voxel_grid": voxel.to(device),
            "text": str(data["text"]),
            "length": max(int(data["length"]), 40),
            "scene_name": str(data.get("scene_name", os.path.basename(fp))),
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="../trajectory_plots")
    parser.add_argument("--num_samples", type=int, default=6)
    args = parser.parse_args()

    setup_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    cache_dir = "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/cached_data"

    checkpoints = {
        "exp1": "./exp1_monkey_patch_output/checkpoints/best_checkpoint.pt",
        "exp2": "./exp2_rewrite_layer_output/checkpoints/best_checkpoint.pt",
        "exp3": "./exp3_root_only_output/checkpoints/best_checkpoint.pt",
        "exp4": "./exp4_body_only_output/checkpoints/best_checkpoint.pt",
    }

    log.info("Loading scene samples...")
    samples = load_scene_samples(cache_dir, device="cpu", num_samples=args.num_samples)
    for i, s in enumerate(samples):
        log.info(f"  [{i+1}] {s['scene_name']}: '{s['text']}' len={s['length']}")

    log.info("Loading original KiMoDo...")
    orig = load_original_kimodo(device)

    all_summaries = {}

    for exp_name, checkpoint_path in checkpoints.items():
        log.info(f"\n{'='*60}")
        log.info(f"Processing {exp_name}...")
        log.info(f"{'='*60}")

        try:
            model = load_sceneco_model(exp_name, checkpoint_path,
                                       text_encoder=orig.text_encoder, device=device)
            log.info(f"  {exp_name} loaded OK")
        except Exception as e:
            log.error(f"  FAILED to load {exp_name}: {e}")
            continue

        ncols = 3
        nrows = len(samples)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        col_titles = ["NO Scene (original KiMoDo)", f"{exp_name} WITH Scene",
                       f"{exp_name} WITHOUT Scene"]
        for ci, title in enumerate(col_titles):
            axes[0, ci].set_title(title, fontsize=12, fontweight='bold')

        summary = []

        for row, sample in enumerate(samples):
            text = sample["text"]
            n_frames = sample["length"]
            scene_name = sample["scene_name"]
            voxel_grid = sample["voxel_grid"].to(device).unsqueeze(0)

            log.info(f"  [{row+1}/{len(samples)}] {scene_name}: '{text}'")

            out_orig = generate_original(orig, text, n_frames, seed=42)
            out_yes = generate_sceneco(model, text, n_frames,
                                       scene_input=voxel_grid, seed=42)
            out_no = generate_sceneco(model, text, n_frames,
                                      scene_input=None, seed=42)

            root_orig = extract_root_trajectory(out_orig)
            root_yes = extract_root_trajectory(out_yes)
            root_no = extract_root_trajectory(out_no)

            scene_occ, extent = voxel_xy_occupancy(sample["voxel_grid"])

            disp_orig = plot_one_trajectory(
                axes[row, 0], root_orig, scene_occ, extent,
                f"[{scene_name}]\n{text[:40]}", color="#1a73e8")
            disp_yes = plot_one_trajectory(
                axes[row, 1], root_yes, scene_occ, extent,
                f"[{scene_name}] WITH scene\n{text[:40]}", color="#d93025")
            disp_no = plot_one_trajectory(
                axes[row, 2], root_no, scene_occ, extent,
                f"[{scene_name}] NO scene\n{text[:40]}", color="#e37400")

            log.info(f"    Displacement: orig={disp_orig:.2f}m  "
                     f"with_scene={disp_yes:.2f}m  no_scene={disp_no:.2f}m")

            summary.append({
                "scene": scene_name, "text": text,
                "disp_orig": disp_orig, "disp_yes": disp_yes,
                "disp_no": disp_no,
                "ratio": disp_yes / max(disp_no, 0.01),
            })

        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor("white")

        plt.tight_layout(pad=2.0)
        out_path = os.path.join(args.output_dir, f"{exp_name}_trajectory_2d.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        log.info(f"  Saved → {out_path}")

        all_summaries[exp_name] = summary

        del model
        torch.cuda.empty_cache()

    del orig
    torch.cuda.empty_cache()

    for exp_name, summary in all_summaries.items():
        log.info(f"\n{'='*90}")
        log.info(f"SUMMARY: {exp_name}")
        log.info(f"{'='*90}")
        log.info(f"{'Scene':<18} {'disp_orig':>10} {'disp_with':>10} "
                 f"{'disp_no':>10} {'with/no':>8}")
        log.info(f"{'-'*90}")
        for s in summary:
            log.info(f"{s['scene']:<18} {s['disp_orig']:10.2f} "
                     f"{s['disp_yes']:10.2f} {s['disp_no']:10.2f} "
                     f"{s['ratio']:8.2f}x")
        mean_orig = np.mean([s["disp_orig"] for s in summary])
        mean_yes = np.mean([s["disp_yes"] for s in summary])
        mean_no = np.mean([s["disp_no"] for s in summary])
        log.info(f"{'MEAN':<18} {mean_orig:10.2f} {mean_yes:10.2f} "
                 f"{mean_no:10.2f} {mean_yes/max(mean_no, 0.01):8.2f}x")
        log.info(f"{'='*90}")



if __name__ == "__main__":
    main()
