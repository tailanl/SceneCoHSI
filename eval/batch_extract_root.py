"""Batch extract Stage1 root trajectories for all LINGO val samples.

Usage (split across tmux sessions):
    # GPU 0, samples 0-387
    python batch_extract_root.py --start 0 --end 388 --gpu 0 --output_dir outputs/root_trajectories

    # GPU 0, samples 388-776
    python batch_extract_root.py --start 388 --end 776 --gpu 0 --output_dir outputs/root_trajectories

    # GPU 0, samples 776-1164
    python batch_extract_root.py --start 776 --end 1164 --gpu 0 --output_dir outputs/root_trajectories

    # GPU 0, samples 1164-1551
    python batch_extract_root.py --start 1164 --end 1551 --gpu 0 --output_dir outputs/root_trajectories
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_sceneco_root_only(ckpt_path, device, dual_vit=True, root_voxel_mode="full"):
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    print(f"Loading pretrained Kimodo base (Kimodo-SOMA-RP-v1.1)...")
    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=device)
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
    return model


def generate_root_path(model, prompt, num_frames, voxel_grid):
    with torch.no_grad():
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
    root = output["smooth_root_pos"]
    if root.ndim == 3:
        root = root[0]
    return root


def main():
    parser = argparse.ArgumentParser(description="Batch extract Stage1 root trajectories")
    parser.add_argument("--start", type=int, required=True, help="Start sample index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End sample index (exclusive)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str,
                        default="kimodo_scene_project/outputs/root_only_sceneco/checkpoints/best_checkpoint.pt")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/root_trajectories")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples: {args.start} → {args.end} ({args.end - args.start} total)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset

    print("Loading LINGO val dataset...")
    ds = LINGOSceneMotionDataset(
        data_root=str(PROJECT_ROOT / "LINGO" / "dataset"),
        max_frames=196, min_frames=40,
        voxel_size=(64, 64, 64),
        train_ratio=0.9, seed=42,
        split="val",
        scene_dropout=0.0,
        cache_dir=str(PROJECT_ROOT / "kimodo/kimodo_sceneco/cached_data"),
    )

    model = build_sceneco_root_only(args.ckpt_path, str(device))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    success = 0
    failed = 0
    start_time = time.time()

    for idx in tqdm(range(args.start, args.end), desc=f"GPU{args.gpu}"):
        seg = ds[idx]
        prompt = seg.get("text", "no-text")
        num_frames = int(seg["length"])
        scene_name = seg.get("scene_name", "unknown")
        if hasattr(scene_name, '__iter__') and not isinstance(scene_name, str):
            scene_name = scene_name[0] if len(scene_name) > 0 else "unknown"
        voxel_grid = seg["voxel_grid"].unsqueeze(0)

        safe_prompt = prompt.replace(" ", "_")[:50]
        out_path = out_dir / f"{idx:04d}_{safe_prompt}_root.npz"

        if out_path.exists():
            success += 1
            continue

        try:
            root_traj = generate_root_path(model, prompt, num_frames, voxel_grid)
            np.savez_compressed(
                out_path,
                smooth_root_pos=root_traj.astype(np.float32),
                prompt=prompt,
                scene_name=scene_name,
                num_frames=num_frames,
                sample_idx=idx,
            )
            success += 1
        except Exception as e:
            print(f"\n  [FAIL] sample {idx}: {e}")
            failed += 1
            continue

    elapsed = time.time() - start_time
    print(f"\nDone! {success} success, {failed} failed in {elapsed:.0f}s ({elapsed/(success+failed):.1f}s/sample)")


if __name__ == "__main__":
    main()
