#!/usr/bin/env python
"""Test exp3 (Root-Only SceneCo) with real scene data.

Loads cached scene samples, generates motion WITH scene vs WITHOUT scene
(same text, same seed), compares outputs, and renders videos.
Usage: python -m kimodo_sceneco.exp.test_exp3_scene --gpu 6
"""

import logging
import os
import sys
import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

log = logging.getLogger("test_exp3")


def setup_env():
    os.environ.setdefault("CHECKPOINT_DIR", "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_original_kimodo(device="cpu"):
    from kimodo.model import load_model as load_kimodo_model
    kimodo = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device=device)
    kimodo.eval()
    return kimodo


def load_sceneco_model(exp_type, checkpoint_path, text_encoder=None, device="cpu"):
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location="cpu")
    exp_type = ckpt.get("exp_type", exp_type)

    from kimodo.model import load_model as load_kimodo_model

    class _DummyTE:
        def __call__(self, text): raise RuntimeError()
        def to(self, d): return self
        def eval(self): return self

    kimodo_pretrained = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device="cpu", text_encoder=_DummyTE())

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


def get_skeleton_connections(model):
    skeleton = model.skeleton
    if hasattr(skeleton, 'joint_parents'):
        parents = skeleton.joint_parents.cpu().numpy()
    else:
        return []
    connections = []
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx >= 0:
            connections.append((int(parent_idx), child_idx))
    return connections


@torch.no_grad()
def generate_motion(model, texts, num_frames, scene_input=None, num_steps=50, cfg_weight=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_sceneco = hasattr(model, 'scene_null_embed')

    if not is_sceneco:
        if cfg_weight is None:
            cfg_weight = [2.0, 2.0]
        return model(
            prompts=texts, num_frames=num_frames,
            num_denoising_steps=num_steps, cfg_weight=cfg_weight,
            return_numpy=False, progress_bar=lambda x: x,
        )
    else:
        if cfg_weight is None:
            cfg_weight = [2.0, 2.0, 2.0]
        return model(
            prompts=texts, num_frames=num_frames,
            num_denoising_steps=num_steps, cfg_weight=cfg_weight,
            return_numpy=False, progress_bar=lambda x: x,
            scene_input=scene_input,
        )


def render_video(posed_joints, output_path, title, connections=None,
                 scene_voxel=None, fps=30, dpi=100):
    T, J, D = posed_joints.shape
    log.info(f"  Rendering: {T} frames, {J} joints → {os.path.basename(output_path)}")

    has_scene = scene_voxel is not None and scene_voxel.sum() > 0
    if has_scene:
        scene_pts = _voxel_to_scene_points(scene_voxel, stride=4, voxel_size=0.1)
        log.info(f"  Scene: {len(scene_pts)} surface voxel points")
    else:
        scene_pts = np.zeros((0, 3))

    all_coords = posed_joints.reshape(-1, 3)
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    z_range = all_coords[:, 2].max() - all_coords[:, 2].min()
    max_range = max(x_range, y_range, z_range, 0.5)
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) / 2
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) / 2
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) / 2

    fig = plt.figure(figsize=(12, 9), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    def init_ax():
        ax.clear()
        r = max_range * 1.3
        ax.set_xlim(mid_x - r, mid_x + r)
        ax.set_ylim(mid_y - r, mid_y + r)
        ax.set_zlim(mid_z - r, mid_z + r)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=25, azim=-55)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if len(scene_pts) > 0:
            z_norm = (scene_pts[:, 2] - scene_pts[:, 2].min()) / max(np.ptp(scene_pts[:, 2]), 0.01)
            colors = np.stack([0.2 + 0.5 * z_norm, 0.8 - 0.4 * z_norm, 0.3 * z_norm], axis=1)
            ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                       c=colors, s=1, alpha=0.25, rasterized=True)

    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, dpi):
        for t in range(T):
            init_ax()
            xyz = posed_joints[t]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="#ff4444", s=20, alpha=0.95, zorder=5)
            if connections:
                for parent, child in connections:
                    ax.plot(
                        [xyz[parent, 0], xyz[child, 0]],
                        [xyz[parent, 1], xyz[child, 1]],
                        [xyz[parent, 2], xyz[child, 2]],
                        c="#ffffff", linewidth=2.5, alpha=0.85, zorder=4,
                    )
            ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], c="#00e5ff", s=50, alpha=1.0, zorder=6, edgecolors='white', linewidth=0.5)
            writer.grab_frame()

    plt.close(fig)


def _voxel_to_scene_points(voxel_grid, stride=2, voxel_size=0.1):
    vox = voxel_grid.squeeze().cpu().numpy()
    D, H, W = vox.shape
    pts = []
    for z in range(stride // 2, D, stride):
        for y in range(stride // 2, H, stride):
            for x in range(stride // 2, W, stride):
                if vox[z, y, x] > 0:
                    if (z == 0 or z == D - 1 or y == 0 or y == H - 1 or x == 0 or x == W - 1 or
                        vox[z - 1, y, x] == 0 or vox[z + 1, y, x] == 0 or
                        vox[z, y - 1, x] == 0 or vox[z, y + 1, x] == 0 or
                        vox[z, y, x - 1] == 0 or vox[z, y, x + 1] == 0):
                        world_x = x * voxel_size
                        world_y = y * voxel_size
                        world_z = z * voxel_size
                        pts.append([world_x, world_y, world_z])
    return np.array(pts) if pts else np.zeros((0, 3)).reshape(0, 3)


def compare_motions(out_no, out_yes):
    results = {}
    if "posed_joints" in out_no and "posed_joints" in out_yes:
        pj_no = out_no["posed_joints"].cpu()
        pj_yes = out_yes["posed_joints"].cpu()
        if pj_no.ndim == 4:
            pj_no = pj_no[0]
            pj_yes = pj_yes[0]
        if pj_no.shape == pj_yes.shape:
            jl2 = float(torch.norm(pj_no - pj_yes, dim=-1).mean().item())
            mse = float(F.mse_loss(pj_no, pj_yes).item())
            results["joint_l2"] = jl2
            results["joint_mse"] = mse

            cos_sims = []
            for ji in range(pj_no.shape[1]):
                a = pj_no[:, ji, :] - pj_no[:, ji, :].mean(0, keepdim=True)
                b = pj_yes[:, ji, :] - pj_yes[:, ji, :].mean(0, keepdim=True)
                cs = F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
                cos_sims.append(cs)
            results["traj_cos"] = float(np.mean(cos_sims))

    return results


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
        text = str(data["text"])
        length = int(data["length"])
        scene_name = str(data.get("scene_name", os.path.basename(fp)))
        samples.append({
            "voxel_grid": voxel.to(device),
            "text": text,
            "length": max(length, 40),
            "scene_name": scene_name,
        })
        log.info(f"  [{len(samples)}] {scene_name}: '{text}' len={length} voxel={list(voxel.shape)}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="../exp3_scene_videos")
    parser.add_argument("--num_samples", type=int, default=6)
    args = parser.parse_args()

    setup_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    cache_dir = "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/cached_data"
    checkpoint_path = "./exp3_root_only_output/checkpoints/best_checkpoint.pt"

    log.info("Loading original KiMoDo...")
    orig = load_original_kimodo(device)
    connections = get_skeleton_connections(orig)
    log.info(f"  Skeleton: {len(connections)} bones")

    log.info(f"Loading exp3 from {checkpoint_path}...")
    model = load_sceneco_model("exp3", checkpoint_path, text_encoder=orig.text_encoder, device=device)
    log.info("  exp3 loaded OK")

    log.info(f"Loading {args.num_samples} scene samples from {cache_dir}...")
    samples = load_scene_samples(cache_dir, device, args.num_samples)

    log.info("")
    log.info("=" * 70)
    log.info("TESTING exp3: WITH scene vs WITHOUT scene (same text, same seed)")
    log.info("=" * 70)

    all_results = []

    for idx, sample in enumerate(samples):
        text = sample["text"]
        n_frames = sample["length"]
        scene_name = sample["scene_name"]
        voxel_grid = sample["voxel_grid"].unsqueeze(0)

        log.info(f"\n--- Sample {idx+1}/{len(samples)} [{scene_name}] ---")
        log.info(f"  Text: '{text}'")
        log.info(f"  Frames: {n_frames}")
        log.info(f"  Voxel: {list(voxel_grid.shape)}")
        log.info(f"  Voxel occupancy: {(voxel_grid > 0).float().mean().item()*100:.1f}%")

        log.info("  Generating WITHOUT scene (seed=42)...")
        out_no = generate_motion(model, text, n_frames, scene_input=None, seed=42)

        log.info("  Generating WITH scene (seed=42)...")
        out_yes = generate_motion(model, text, n_frames, scene_input=voxel_grid, seed=42)

        comp = compare_motions(out_no, out_yes)
        log.info(f"  Joint L2 (with vs without): {comp.get('joint_l2', 0):.4f}m")
        log.info(f"  Joint MSE:                  {comp.get('joint_mse', 0):.4f}")
        log.info(f"  Trajectory cosine:          {comp.get('traj_cos', 0):.4f}")
        all_results.append(comp)

        tag = f"{idx+1:02d}_{scene_name.replace(' ', '_')}"
        pj_no = out_no["posed_joints"].cpu().numpy()
        pj_yes = out_yes["posed_joints"].cpu().numpy()
        if pj_no.ndim == 4:
            pj_no = pj_no[0]
            pj_yes = pj_yes[0]

        render_video(pj_no, os.path.join(args.output_dir, f"{tag}_noscene.mp4"),
                     f"[{scene_name}] NO scene: {text[:60]}", connections)
        render_video(pj_yes, os.path.join(args.output_dir, f"{tag}_withscene.mp4"),
                     f"[{scene_name}] WITH scene: {text[:60]}", connections,
                     scene_voxel=voxel_grid)

    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    l2_mean = np.mean([r["joint_l2"] for r in all_results])
    l2_std = np.std([r["joint_l2"] for r in all_results])
    cos_mean = np.mean([r["traj_cos"] for r in all_results])
    log.info(f"  Mean Joint L2 (with scene vs without): {l2_mean:.4f}m ± {l2_std:.4f}")
    log.info(f"  Mean Trajectory Cosine:                {cos_mean:.4f}")
    log.info("")
    log.info("Interpretation:")
    log.info("  Joint L2 > 0.05m → scene significantly changes the motion")
    log.info("  Traj Cosine < 0.95 → motion pattern differs with scene context")
    log.info(f"  Videos saved to: {os.path.abspath(args.output_dir)}/")

    del model, orig
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
