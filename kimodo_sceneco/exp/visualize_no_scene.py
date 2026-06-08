#!/usr/bin/env python
"""Visualize motion from all 4 SceneCo experiments + original KiMoDo (no scene).

Generates MP4 videos for the same prompt+seed across all models.
Usage: python -m kimodo_sceneco.exp.visualize_no_scene --gpu 6
"""

import logging
import os
import sys
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

log = logging.getLogger("visualize")


def setup_env():
    os.environ.setdefault("CHECKPOINT_DIR", "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_original_kimodo(device="cpu"):
    from kimodo.model import load_model as load_kimodo_model
    log.info("Loading original KiMoDo...")
    kimodo = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device=device)
    kimodo.eval()
    return kimodo


def load_sceneco_model(exp_type, checkpoint_path, text_encoder=None, device="cpu"):
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location="cpu")
    exp_type = ckpt.get("exp_type", exp_type)

    from kimodo.model import load_model as load_kimodo_model

    class _DummyTE:
        def __call__(self, text):
            raise RuntimeError()
        def to(self, d):
            return self
        def eval(self):
            return self

    log.info(f"Loading pretrained KiMoDo base for {exp_type}...")
    kimodo_pretrained = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device="cpu", text_encoder=_DummyTE())

    scene_config = {
        'voxel_size': (64, 64, 64), 'patch_size': (8, 8, 8),
        'in_channels': 1, 'd_model': 256, 'num_heads': 4,
        'num_layers': 4, 'ff_dim': 512,
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
        log.info("Loading checkpoint weights...")
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
        log.warning("No joint_parents found; using scatter-only rendering")
        return []
    connections = []
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx >= 0:
            connections.append((int(parent_idx), child_idx))
    return connections


@torch.no_grad()
def generate_motion(model, texts, num_frames, num_steps=50, cfg_weight=None, seed=42):
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
            scene_input=None,
        )


def render_video(posed_joints, output_path, title, connections=None, fps=30, dpi=100):
    T, J, D = posed_joints.shape
    log.info(f"  Rendering: {T} frames, {J} joints → {output_path}")

    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    joints_np = posed_joints

    all_coords = joints_np.reshape(-1, 3)
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    z_range = all_coords[:, 2].max() - all_coords[:, 2].min()
    max_range = max(x_range, y_range, z_range, 0.5)
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) / 2
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) / 2
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) / 2

    def init_ax():
        ax.clear()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.view_init(elev=20, azim=-60)

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi):
        for t in range(T):
            init_ax()
            xyz = joints_np[t]

            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="red", s=15, alpha=0.9)

            if connections:
                for parent, child in connections:
                    ax.plot(
                        [xyz[parent, 0], xyz[child, 0]],
                        [xyz[parent, 1], xyz[child, 1]],
                        [xyz[parent, 2], xyz[child, 2]],
                        c="white", linewidth=2, alpha=0.8,
                    )

            root_pos = xyz[0]
            ax.scatter([root_pos[0]], [root_pos[1]], [root_pos[2]], c="cyan", s=40, alpha=1.0)

            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#1a1a2e")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            writer.grab_frame()

    plt.close(fig)
    log.info(f"  Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./no_scene_videos")
    args = parser.parse_args()

    setup_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    orig = load_original_kimodo(device)

    connections = get_skeleton_connections(orig)
    log.info(f"Skeleton connections: {len(connections)} bones")

    checkpoints = {
        "exp1": "./exp1_monkey_patch_output/checkpoints/best_checkpoint.pt",
        "exp2": "./exp2_rewrite_layer_output/checkpoints/best_checkpoint.pt",
        "exp3": "./exp3_root_only_output/checkpoints/best_checkpoint.pt",
        "exp4": "./exp4_body_only_output/checkpoints/best_checkpoint.pt",
    }

    models = {"original": orig}
    for exp_name, ckpt_path in checkpoints.items():
        try:
            models[exp_name] = load_sceneco_model(
                exp_name, ckpt_path, text_encoder=orig.text_encoder, device=device
            )
            log.info(f"  {exp_name} loaded OK")
        except Exception as e:
            log.error(f"  FAILED to load {exp_name}: {e}")

    test_prompts = [
        ("A person walks forward in a straight line.", 60, "walk"),
        ("A person turns around and walks back.", 80, "turn"),
        ("A person crouches down and stands up.", 50, "crouch"),
        ("A person performs a jumping jack.", 40, "jumping_jack"),
    ]

    for text, num_frames, tag in test_prompts:
        log.info(f"\n{'='*60}")
        log.info(f"Prompt: '{text}' ({num_frames}f)")
        log.info(f"{'='*60}")

        for model_name, model in models.items():
            log.info(f"  Generating {model_name}...")
            out = generate_motion(model, text, num_frames, num_steps=50, seed=42)

            if "posed_joints" not in out:
                log.error(f"  {model_name}: no posed_joints in output!")
                continue

            posed_joints = out["posed_joints"].cpu().numpy()
            if posed_joints.ndim == 4:
                posed_joints = posed_joints[0]

            video_path = os.path.join(args.output_dir, f"{tag}_{model_name}.mp4")
            render_video(
                posed_joints, video_path,
                title=f"{model_name}: {text[:50]}...",
                connections=connections,
            )

    log.info(f"\nDone! Videos saved to {os.path.abspath(args.output_dir)}/")

    for m in models.values():
        if m is not orig:
            del m
    del orig
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
