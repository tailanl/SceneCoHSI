#!/usr/bin/env python
"""Merge root_only (SceneCo in root) + body_only (SceneCo in body) into a combined model.

Usage:
    python kimodo_scene_project/eval/merge_root_body.py
    # Output: kimodo_scene_project/outputs/root_body_merged/best_checkpoint.pt
"""

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_ROOT / "SOMA"))

os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
os.environ.setdefault("PYTHONHASHSEED", "0")


def main():
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    device = "cpu"

    p_root = PROJECT_ROOT / "kimodo_scene_project/outputs/root_only_sceneco/checkpoints/best_checkpoint.pt"
    p_body = PROJECT_ROOT / "kimodo_scene_project/outputs/body_only_sceneco/checkpoints/best_checkpoint.pt"
    out_dir = PROJECT_ROOT / "kimodo_scene_project/outputs/root_body_merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained Kimodo...")
    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=device)
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model

    print("Creating combined model (use_in_root_model=True, use_in_body_model=True)...")
    model = KimodoSceneCo(
        denoiser=inner,
        text_encoder=pretrained.text_encoder,
        num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64),
            "patch_size": (8, 8, 8),
            "d_model": 256,
            "num_layers": 4,
            "use_dual_vit": True,
            "root_voxel_mode": "full",
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=True,
        use_in_body_model=True,
    )

    print("Loading root_only checkpoint...")
    ckpt_root = torch.load(p_root, map_location=device)
    sd_root = ckpt_root.get("model_state_dict", ckpt_root)

    print("Loading body_only checkpoint...")
    ckpt_body = torch.load(p_body, map_location=device)
    sd_body = ckpt_body.get("model_state_dict", ckpt_body)

    merged = {}

    from_root = 0
    from_body = 0
    from_either = 0

    for key in sd_root.keys():
        if "root_model.sceneco_layers" in key:
            merged[key] = sd_root[key]
            from_root += 1
        elif "scene_encoder_root" in key and "scene_encoder_body" not in key:
            merged[key] = sd_root[key]
            from_root += 1
        elif key in sd_body:
            merged[key] = sd_body[key]
            from_either += 1
        else:
            merged[key] = sd_root[key]
            from_either += 1

    for key in sd_body.keys():
        if key in merged:
            continue
        if "body_model.sceneco_layers" in key:
            merged[key] = sd_body[key]
            from_body += 1
        elif "scene_encoder_body" in key:
            merged[key] = sd_body[key]
            from_body += 1
        else:
            merged[key] = sd_body[key]
            from_body += 1

    print(f"\nMerge stats:")
    print(f"  From root_only: {from_root} keys (root SceneCo + scene_encoder_root)")
    print(f"  From body_only: {from_body} keys (body SceneCo + scene_encoder_body)")
    print(f"  From either:    {from_either} keys (shared backbone)")

    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(f"\nLoad result: missing={len(missing)}, unexpected={len(unexpected)}")

    for name, param in model.named_parameters():
        if "alpha" in name and "sceneco" in name.lower():
            gate = torch.sigmoid(param)
            print(f"  {name}: {param.item():.4f} (gate={gate.item():.4f})")

    out_path = out_dir / "best_checkpoint.pt"
    torch.save({"model_state_dict": merged, "description": "Merged root_only + body_only"}, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
