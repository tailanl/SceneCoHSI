"""Generate root trajectories with trained RootPathSceneClassifier guidance."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

FILE = Path(__file__).resolve()
PROJECT_DIR = FILE.parent.parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_DIR))

from kimodo.model.load_model import load_model

from kimodo_sceneco.critic.root_path_scene_classifier import RootPathSceneClassifier
from kimodo_sceneco.guidance.root_guidance import RootGuidanceConfig
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

log = logging.getLogger(__name__)


def load_cfg(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


def load_sceneco_model(model_ckpt: str, device: torch.device) -> KimodoSceneCo:
    pretrained = load_model(model_ckpt, device="cpu")
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model

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
        },
        device=device,
        cfg_type="scene_separated",
    )
    model.eval()
    return model


def load_classifier(ckpt_path: str, cfg: dict, device: torch.device) -> RootPathSceneClassifier:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classifier_cfg = cfg.get("root_classifier", {})
    model = RootPathSceneClassifier(
        input_dim=ckpt.get("input_dim", classifier_cfg.get("input_dim", 20)),
        hidden_dim=ckpt.get("hidden_dim", classifier_cfg.get("hidden_dim", 256)),
        num_layers=ckpt.get("num_layers", classifier_cfg.get("num_layers", 4)),
        num_heads=ckpt.get("num_heads", classifier_cfg.get("num_heads", 4)),
        dropout=ckpt.get("dropout", classifier_cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def find_cache_files(cache_dir: str | None) -> list[Path]:
    candidates = []
    if cache_dir:
        candidates.append(Path(cache_dir))
    candidates.extend(
        [
            PROJECT_DIR / "LINGO/dataset/dataset/lingo_smplx_cache",
            REPO_ROOT / "LINGO/dataset/dataset/lingo_smplx_cache",
            REPO_ROOT / "lingo_smplx_cache",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            files = sorted(candidate.glob("*.pt"))
            if files:
                return files
    return []


def get_motion_tensor(data: dict, file: Path) -> torch.Tensor:
    motion = data.get("motion", data.get("beta_motion", data.get("data", data.get("motion_features"))))
    if motion is None:
        raise KeyError(f"{file} has no motion, beta_motion, data, or motion_features entry")
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    return motion.float()


def get_text(data: dict, fallback: str = "") -> str:
    for key in ("text", "caption", "prompt", "description"):
        value = data.get(key)
        if value is not None:
            return str(value)
    return fallback


def make_root_guidance_cfg(cfg: dict, enabled: bool) -> RootGuidanceConfig:
    energy = cfg.get("energy_guidance", cfg.get("path_guidance", {}))
    classifier = cfg.get("classifier_guidance", {})
    return RootGuidanceConfig(
        enabled=enabled,
        w_path=energy.get("w_path", 10.0),
        w_goal=energy.get("w_goal", 20.0),
        w_speed=energy.get("w_speed", 1.0),
        w_smooth=energy.get("w_smooth", 2.0),
        w_jerk=energy.get("w_jerk", 0.5),
        w_heading=energy.get("w_heading", 2.0),
        w_heading_norm=energy.get("w_heading_norm", 0.5),
        w_height=energy.get("w_height", 1.0),
        w_scene=energy.get("w_scene", 0.0),
        scene_margin=energy.get("scene_margin", 0.10),
        scale=energy.get("scale", classifier.get("scale", 0.05)),
        max_grad_norm=energy.get("max_grad_norm", classifier.get("max_grad_norm", 1.0)),
        start_step=energy.get("start_step", classifier.get("start_step", 0)),
        end_step=energy.get("end_step", classifier.get("end_step", 40)),
    )


def root_5d_from_output(output: dict) -> np.ndarray:
    pos = output["smooth_root_pos"]
    heading = output["global_root_heading"]
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(heading, torch.Tensor):
        heading = heading.detach().cpu().numpy()
    if pos.ndim == 3:
        pos = pos[0]
    if heading.ndim == 3:
        heading = heading[0]
    return np.concatenate([pos, heading], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/root_classifier_guidance.yaml")
    parser.add_argument("--classifier_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--num_denoising_steps", type=int, default=None)
    parser.add_argument("--cfg_weight", type=float, nargs=2, default=None)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--classifier_guidance_scale", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_cfg(args.config)

    gen_cfg = cfg.get("generation", {})
    classifier_cfg = cfg.get("classifier_guidance", {})
    hybrid_cfg = cfg.get("hybrid", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    gpu = args.gpu if args.gpu is not None else gen_cfg.get("gpu", 0)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.classifier_ckpt or cfg.get("root_classifier", {}).get(
        "checkpoint", "outputs/root_path_scene_classifier/best.pt"
    )
    num_steps = args.num_denoising_steps or gen_cfg.get("num_denoising_steps", 50)
    cfg_weight = args.cfg_weight or gen_cfg.get("cfg_weight", [2.0, 2.0])
    guidance_scale = args.classifier_guidance_scale or classifier_cfg.get("scale", 0.05)
    max_grad_norm = args.max_grad_norm or classifier_cfg.get("max_grad_norm", 1.0)
    hybrid_enabled = args.hybrid or hybrid_cfg.get("enabled", False)

    log.info("Loading KimodoSceneCo model")
    model = load_sceneco_model(model_cfg.get("checkpoint", "Kimodo-SMPLX-RP-v1"), device)

    log.info("Loading RootPathSceneClassifier from %s", ckpt_path)
    root_classifier = load_classifier(ckpt_path, cfg, device)

    cache_files = find_cache_files(args.cache_dir or data_cfg.get("cache_dir"))
    if not cache_files:
        raise FileNotFoundError("No .pt cache files found for target paths")
    log.info("Found %d cache files", len(cache_files))

    root_guidance_cfg = make_root_guidance_cfg(cfg, enabled=hybrid_enabled)
    metadata = []

    for sample_idx, cache_file in enumerate(cache_files[: args.num_samples]):
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        motion = get_motion_tensor(data, cache_file)
        T = min(int(motion.shape[0]), gen_cfg.get("num_frames", 196))
        if T < 2:
            continue

        target_path_xz = motion[:T, [0, 2]].unsqueeze(0).to(device)
        text = get_text(data, fallback="")

        output = model(
            prompts=[text],
            num_frames=[T],
            num_denoising_steps=num_steps,
            cfg_weight=cfg_weight,
            num_samples=1,
            root_classifier=root_classifier,
            classifier_guidance_scale=guidance_scale,
            classifier_max_grad_norm=max_grad_norm,
            root_classifier_start_step=classifier_cfg.get("start_step", 0),
            root_classifier_end_step=classifier_cfg.get("end_step", num_steps),
            hybrid=hybrid_enabled,
            root_guidance_cfg=root_guidance_cfg if hybrid_enabled else None,
            w_classifier=hybrid_cfg.get("w_classifier", 1.0),
            w_energy=hybrid_cfg.get("w_energy", 0.3),
            target_path_xz=target_path_xz,
            return_numpy=False,
        )

        guided_root_5d = root_5d_from_output(output)
        out_file = output_dir / f"{cache_file.stem}_{sample_idx:04d}.npy"
        np.save(out_file, guided_root_5d)

        metadata.append(
            {
                "file": str(out_file),
                "source_cache": str(cache_file),
                "text": text,
                "num_frames": int(T),
                "hybrid": bool(hybrid_enabled),
                "classifier_guidance_scale": float(guidance_scale),
                "max_grad_norm": float(max_grad_norm),
            }
        )
        if (sample_idx + 1) % 10 == 0:
            log.info("Generated %d/%d", sample_idx + 1, args.num_samples)

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Done. Saved %d guided roots to %s", len(metadata), output_dir)


if __name__ == "__main__":
    main()
