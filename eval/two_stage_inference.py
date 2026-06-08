"""Two-stage inference pipeline: SceneCo root_only → root trajectory → Kimodo body.

Stage 1: SceneCo (root_only) generates scene-aware root path from text + scene.
Stage 2: Original Kimodo generates full-body motion conditioned on that root path.

Usage:
    PYTHONPATH="kimodo:SOMA:$PYTHONPATH" CHECKPOINT_DIR=models/Kimodo-SOMA-RP-v1.1 \
    HF_HOME=.hf_cache TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local \
    TEXT_ENCODER_DEVICE=cpu python kimodo_scene_project/eval/two_stage_inference.py \
    --ckpt_path kimodo_scene_project/outputs/root_only_sceneco/checkpoints/best_checkpoint.pt \
    --prompt "walk forward" --num_frames 120 --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_sceneco_root_only(ckpt_path: str, device: str, dual_vit: bool = True, root_voxel_mode: str = "full",
                            pretrained_name: str = "Kimodo-SOMA-RP-v1.1"):
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    print(f"Loading pretrained Kimodo base ({pretrained_name})...")
    pretrained = load_model(pretrained_name, device=device)
    inner_denoiser = pretrained.denoiser
    if hasattr(inner_denoiser, "model"):
        inner_denoiser = inner_denoiser.model

    print("Building KimodoSceneCo (root_only)...")
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
            "root_voxel_mode": root_voxel_mode,
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=True,
        use_in_body_model=False,
    )
    model = model.to(device)
    model.eval()

    print(f"Loading SceneCo checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    print(f"  Checkpoint step: {ckpt.get('global_step', ckpt.get('step', '?'))}")
    for name, param in model.named_parameters():
        if "alpha" in name and param.numel() == 1:
            print(f"  {name.rsplit('.', 1)[0].rsplit('.', 1)[-1]:20s}: {param.item():.6f}")

    return model


def build_original_kimodo(device: str, pretrained_name: str = "Kimodo-SOMA-RP-v1.1"):
    from kimodo.model import load_model

    print(f"Loading original Kimodo model ({pretrained_name})...")
    model = load_model(pretrained_name, device=device)
    model.eval()
    return model


def load_voxel_grid_from_npz(npz_path: str, voxel_size=(64, 64, 64)):
    data = np.load(npz_path)
    grid = data.get("voxel_grid", data.get("arr_0"))
    if grid is None:
        raise KeyError(f"No voxel_grid found in {npz_path}, keys: {list(data.keys())}")

    grid = grid.astype(np.float32)
    if grid.ndim == 4:
        grid = grid[0]
    if grid.shape != tuple(voxel_size):
        import scipy.ndimage
        zoom = [vs / gs for vs, gs in zip(voxel_size, grid.shape)]
        grid = scipy.ndimage.zoom(grid, zoom, order=1)
        grid = (grid > 0.5).astype(np.float32)
    return torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)


def stage1_generate_root_path(model, prompt: str, num_frames: int, voxel_grid=None):
    print(f"\n[Stage 1] SceneCo root_only: '{prompt}', {num_frames} frames")
    with torch.no_grad():
        if voxel_grid is not None:
            voxel_grid = voxel_grid.to(model.device)

        output = model(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=50,
            cfg_weight=[3.0, 1.5, 2.0],
            cfg_type="scene_separated",
            scene_input=voxel_grid,
            return_numpy=True,
        )

    print(f"  root_positions shape: {output['root_positions'].shape}")
    print(f"  smooth_root_pos shape: {output['smooth_root_pos'].shape}")
    return output


def stage2_generate_body(model, prompt: str, num_frames: int, root_trajectory: np.ndarray):
    print(f"\n[Stage 2] Original Kimodo with root constraint: '{prompt}', {num_frames} frames")
    from kimodo.constraints import Root2DConstraintSet

    root_2d = root_trajectory[:, [0, 2]]
    frame_indices = torch.arange(num_frames, dtype=torch.long, device=model.device)

    constraint = Root2DConstraintSet(
        skeleton=model.skeleton,
        frame_indices=frame_indices,
        smooth_root_2d=torch.from_numpy(root_2d).float().to(model.device),
        to_crop=False,
    )

    with torch.no_grad():
        output = model(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=50,
            constraint_lst=[constraint],
            cfg_weight=[2.0, 2.0],
            cfg_type="separated",
            return_numpy=True,
        )

    print(f"  root_positions shape: {output['root_positions'].shape}")
    return output


def save_motion(output: dict, out_path: Path, prefix: str):
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path / f"{prefix}_motion.npz",
        posed_joints=output.get("posed_joints"),
        root_positions=output.get("root_positions"),
        local_rot_mats=output.get("local_rot_mats"),
        global_rot_mats=output.get("global_rot_mats"),
        foot_contacts=output.get("foot_contacts"),
        global_root_heading=output.get("global_root_heading"),
        smooth_root_pos=output.get("smooth_root_pos", output.get("root_positions")),
    )


def compute_root_trajectory_error(traj_a: np.ndarray, traj_b: np.ndarray) -> dict:
    t_a = np.squeeze(traj_a)
    t_b = np.squeeze(traj_b)
    n = min(len(t_a), len(t_b))
    t_a, t_b = t_a[:n], t_b[:n]

    diff = t_a - t_b
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    per_frame = float(np.mean(np.sqrt(np.sum(diff ** 2, axis=-1))))
    return {"mse": mse, "rmse": rmse, "mean_per_frame_l2": per_frame}


def main():
    parser = argparse.ArgumentParser(description="Two-stage SceneCo root → Kimodo body inference")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to SceneCo root_only checkpoint")
    parser.add_argument("--prompt", type=str, default="walk forward",
                        help="Text prompt for motion generation")
    parser.add_argument("--num_frames", type=int, default=120,
                        help="Number of frames to generate")
    parser.add_argument("--num_denoising_steps", type=int, default=50,
                        help="DDIM denoising steps")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--voxel_npz", type=str, default=None,
                        help="Path to .npz file containing voxel_grid (optional)")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/two_stage_inference")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dual_vit", action="store_true", default=True,
                        help="Use dual VoxelViT encoders")
    parser.add_argument("--no_dual_vit", action="store_true",
                        help="Use single shared VoxelViT encoder")
    parser.add_argument("--root_voxel_mode", type=str, default="full",
                        choices=["full", "floor"],
                        help="Voxel mode for root encoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_sample", type=int, default=None,
                        help="Sample index from LINGO val set (overrides --voxel_npz and --prompt)")
    parser.add_argument("--stage2_model", type=str, default="Kimodo-SOMA-RP-v1.1",
                        help="Pretrained Kimodo model for Stage 2 body generation "
                             "(e.g. Kimodo-SOMA-RP-v1.1, Kimodo-SMPLX-RP-v1)")
    args = parser.parse_args()

    if args.no_dual_vit:
        args.dual_vit = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt
    num_frames = args.num_frames
    voxel_grid = None

    if args.dataset_sample is not None:
        from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset

        print(f"\nLoading dataset sample {args.dataset_sample} from LINGO val set...")
        ds = LINGOSceneMotionDataset(
            data_root=str(PROJECT_ROOT / "LINGO" / "dataset"),
            max_frames=196, min_frames=40,
            voxel_size=(64, 64, 64),
            train_ratio=0.9, seed=42,
            split="val",
            scene_dropout=0.0,
            cache_dir=str(PROJECT_ROOT / "kimodo/kimodo_sceneco/cached_data"),
        )
        seg = ds[args.dataset_sample]
        prompt = seg.get("text", "no-text")
        num_frames = int(seg["length"])
        voxel_grid = seg["voxel_grid"].unsqueeze(0)
        print(f"  scene: {seg.get('scene_name', '?')}, text: '{prompt}', frames: {num_frames}")
    elif args.voxel_npz:
        print(f"\nLoading voxel grid from: {args.voxel_npz}")
        voxel_grid = load_voxel_grid_from_npz(args.voxel_npz)

    model_s1 = build_sceneco_root_only(
        args.ckpt_path, str(device),
        dual_vit=args.dual_vit,
        root_voxel_mode=args.root_voxel_mode,
    )

    model_orig = build_original_kimodo(str(device), pretrained_name=args.stage2_model)

    for sample_idx in range(args.num_samples):
        sample_seed = args.seed + sample_idx
        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)

        print(f"\n{'='*60}")
        print(f"Sample {sample_idx} (seed={sample_seed})")
        print(f"{'='*60}")

        stage1_output = stage1_generate_root_path(
            model_s1, prompt, num_frames, voxel_grid=voxel_grid,
        )

        root_trajectory = stage1_output["smooth_root_pos"]
        if root_trajectory.ndim == 3:
            root_trajectory = root_trajectory[0]

        stage2_output = stage2_generate_body(
            model_orig, prompt, num_frames,
            root_trajectory=root_trajectory,
        )

        safe_prompt = prompt.replace(" ", "_")[:40]
        prefix_s1 = f"stage1_sceneco_{safe_prompt}_s{sample_idx}"
        prefix_s2 = f"stage2_kimodo_{safe_prompt}_s{sample_idx}"

        save_motion(stage1_output, out_dir, prefix_s1)
        save_motion(stage2_output, out_dir, prefix_s2)

        root_s2 = stage2_output["smooth_root_pos"]
        if root_s2.ndim == 3:
            root_s2 = root_s2[0]
        root_error = compute_root_trajectory_error(root_trajectory, root_s2)
        print(f"\n  Root trajectory error (Stage1 vs Stage2):")
        print(f"    MSE:  {root_error['mse']:.6f}")
        print(f"    RMSE: {root_error['rmse']:.6f}")
        print(f"    Mean per-frame L2: {root_error['mean_per_frame_l2']:.6f}")

    config = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_frames": num_frames,
        "num_denoising_steps": args.num_denoising_steps,
        "prompt": prompt,
        "ckpt_path": args.ckpt_path,
        "dual_vit": args.dual_vit,
        "root_voxel_mode": args.root_voxel_mode,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
