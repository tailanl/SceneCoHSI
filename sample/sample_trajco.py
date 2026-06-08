"""Sample inference for TrajCo: trajectory-conditioned motion generation.

Demonstrates Plan B (TrajCo) usage: given a root trajectory, generate full body motion.

Usage:
    python kimodo_scene_project/sample/sample_trajco.py \
        --checkpoint kimodo_scene_project/outputs/trajco_smplx/checkpoints/best_checkpoint.pt \
        --text "a person walks forward" \
        --num_frames 120 \
        --output_dir outputs/samples
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))


def load_model(checkpoint_path: str, device: str = "cuda"):
    from kimodo_sceneco.model import KimodoSceneCo
    from kimodo.model import load_model as load_kimodo_model

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = state.get("args", {})

    pretrained_model = ckpt_args.get("pretrained_model", "Kimodo-SMPLX-RP-v1")
    kimodo_pretrained = load_kimodo_model(pretrained_model, device="cpu")

    scene_encoder_config = {
        "voxel_size": (64, 64, 64),
        "patch_size": (8, 8, 8),
        "in_channels": 1,
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 4,
        "ff_dim": 512,
        "sceneco_dropout": 0.1,
        "use_dual_vit": False,
        "root_voxel_mode": "full",
    }

    model = KimodoSceneCo(
        denoiser=kimodo_pretrained.denoiser.model,
        text_encoder=kimodo_pretrained.text_encoder,
        num_base_steps=ckpt_args.get("num_base_steps", 1000),
        scene_encoder_type="voxel_vit",
        scene_encoder_config=scene_encoder_config,
        device="cpu",
        cfg_type="scene_separated",
        use_in_root_model=False,
        use_in_body_model=False,
        use_trajco=True,
        traj_dim=ckpt_args.get("traj_dim", 5),
        trajco_dropout=ckpt_args.get("trajco_dropout", 0.1),
    )

    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    log.info(f"Loaded TrajCo model from {checkpoint_path}")
    return model


def build_trajectory(num_frames: int, traj_type: str = "line") -> np.ndarray:
    """Build a simple root trajectory for demonstration.

    Args:
        num_frames: number of frames
        traj_type: one of "line", "circle", "wave"

    Returns:
        np.ndarray of shape (num_frames, 5) — smooth_root_pos(3) + heading(2)
    """
    t = np.linspace(0, 1, num_frames)

    if traj_type == "line":
        x = t * 5.0
        z = np.zeros_like(t)
        y = np.zeros_like(t)
        heading = np.zeros_like(t)
    elif traj_type == "circle":
        radius = 2.0
        x = radius * np.sin(t * 2 * np.pi)
        z = radius * (1 - np.cos(t * 2 * np.pi))
        y = np.zeros_like(t)
        heading = t * 2 * np.pi
    elif traj_type == "wave":
        x = t * 5.0
        z = np.sin(t * 4 * np.pi) * 1.5
        y = np.zeros_like(t)
        heading = np.arctan2(np.gradient(z), np.gradient(x))
    else:
        raise ValueError(f"Unknown traj_type: {traj_type}")

    heading_vec = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    traj = np.stack([x, y, z], axis=-1)
    return np.concatenate([traj, heading_vec], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default="a person walks forward")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--traj_type", type=str, default="line",
                        choices=["line", "circle", "wave"])
    parser.add_argument("--cfg_weight", type=float, nargs="+", default=[2.0, 2.0])
    parser.add_argument("--num_denoising_steps", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/samples")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, args.device)

    traj_np = build_trajectory(args.num_frames, args.traj_type)
    traj_input = torch.from_numpy(traj_np).unsqueeze(0).to(args.device)

    output = model(
        prompts=args.text,
        num_frames=args.num_frames,
        num_denoising_steps=args.num_denoising_steps,
        cfg_weight=args.cfg_weight,
        num_samples=args.num_samples,
        return_numpy=True,
        traj_input=traj_input,
    )

    save_path = os.path.join(args.output_dir, f"trajco_{args.traj_type}.npz")
    np.savez(save_path, **output)
    log.info(f"Saved generated motion to {save_path}")

    traj_path = os.path.join(args.output_dir, f"traj_{args.traj_type}.npy")
    np.save(traj_path, traj_np)
    log.info(f"Saved input trajectory to {traj_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import logging as log
    main()
