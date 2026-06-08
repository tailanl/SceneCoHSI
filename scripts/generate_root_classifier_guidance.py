"""Generate root trajectories with trained RootPathSceneClassifier guidance."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import types
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
from kimodo_sceneco.critic.root_classifier_dataset import (
    extract_root_5d_meter,
    load_motion_features,
)
from kimodo_sceneco.guidance.root_guidance import RootGuidanceConfig
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

log = logging.getLogger(__name__)


class ZeroTextEncoder:
    def __init__(self, dim: int = 4096):
        self.dim = dim

    def __call__(self, texts):
        feat = torch.zeros(len(texts), 1, self.dim)
        lengths = [0 for _ in texts]
        return feat, lengths


def load_cfg(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


def load_sceneco_model(
    model_ckpt: str,
    device: torch.device,
    use_zero_text_encoder: bool = True,
) -> KimodoSceneCo:
    model_name = model_ckpt
    model_path = Path(model_ckpt)
    if model_path.exists() and model_path.is_dir():
        os.environ.setdefault("CHECKPOINT_DIR", str(model_path.parent))
        model_name = model_path.name
    text_encoder = ZeroTextEncoder() if use_zero_text_encoder else None
    pretrained = load_model(model_name, device="cpu", text_encoder=text_encoder)
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
        cfg_type="separated",
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
            PROJECT_DIR / "lingo_smplx_cache",
            PROJECT_DIR / "LINGO/dataset/dataset/lingo_smplx_cache",
            REPO_ROOT / "LINGO/dataset/dataset/lingo_smplx_cache",
            REPO_ROOT / "lingo_smplx_cache",
        ]
    )
    for candidate in candidates:
        npz_files: list[Path] = []
        pt_files: list[Path] = []
        if candidate.exists():
            for subdir in [candidate, candidate / "train", candidate / "val"]:
                if subdir.exists():
                    npz_files.extend(
                        path for path in sorted(subdir.glob("*.npz")) if ".tmp" not in path.name
                    )
                    pt_files.extend(
                        path for path in sorted(subdir.glob("*.pt")) if ".tmp" not in path.name
                    )
        files = sorted(set(npz_files)) + sorted(set(pt_files))
        if files:
            return files
    return []


def load_cache_data(path: Path) -> dict:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"Unsupported cache file: {path}")


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


def get_scene_name(data: dict) -> str:
    value = data.get("scene_name", "")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    return str(value)


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


def root_5d_norm_from_output(output: dict, model: KimodoSceneCo, length: int) -> np.ndarray:
    local_rot_mats = output["local_rot_mats"]
    root_positions = output["root_positions"]
    if not isinstance(local_rot_mats, torch.Tensor):
        local_rot_mats = torch.from_numpy(local_rot_mats).to(model.device)
    if not isinstance(root_positions, torch.Tensor):
        root_positions = torch.from_numpy(root_positions).to(model.device)
    if local_rot_mats.dim() == 4:
        local_rot_mats = local_rot_mats.unsqueeze(0)
    if root_positions.dim() == 2:
        root_positions = root_positions.unsqueeze(0)
    lengths = torch.tensor([length], device=local_rot_mats.device)
    features_norm = model.motion_rep(
        local_rot_mats,
        root_positions,
        to_normalize=True,
        lengths=lengths,
    )
    root_norm = features_norm[..., model.motion_rep.root_slice]
    return root_norm[0].detach().cpu().numpy().astype(np.float32)


def install_guidance_logger(model: KimodoSceneCo, rows: list[dict], sample_id_fn):
    original = model.denoising_step_with_root_classifier_guidance

    def wrapped(self, *args, **kwargs):
        t = kwargs.get("t")
        if t is None and len(args) >= 5:
            t = args[4]
        x_tm1, logs = original(*args, **kwargs)
        step = int(t[0].detach().cpu().item()) if isinstance(t, torch.Tensor) else -1
        row = {
            "sample_id": sample_id_fn(),
            "step": step,
            "loss_cls": logs.get("loss_cls", ""),
            "score_valid": logs.get("score_valid", ""),
            "loss_total": logs.get("loss_total", ""),
            "grad_norm": logs.get("grad_norm", ""),
        }
        rows.append(row)
        log.info(
            "sample_id=%s step=%s loss_cls=%s score_valid=%s loss_total=%s grad_norm=%s",
            row["sample_id"],
            row["step"],
            row["loss_cls"],
            row["score_valid"],
            row["loss_total"],
            row["grad_norm"],
        )
        return x_tm1, logs

    model.denoising_step_with_root_classifier_guidance = types.MethodType(wrapped, model)


def write_guidance_log(path: Path, rows: list[dict]) -> None:
    fieldnames = ["sample_id", "step", "loss_cls", "score_valid", "loss_total", "grad_norm"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    parser.add_argument(
        "--real_text_encoder",
        action="store_true",
        help="Use the model-configured text encoder instead of the local zero encoder fallback.",
    )
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
        "checkpoint", "outputs/root_path_classifier/best.pt"
    )
    num_steps = args.num_denoising_steps or gen_cfg.get("num_denoising_steps", 50)
    cfg_weight = args.cfg_weight or gen_cfg.get("cfg_weight", [2.0, 2.0])
    guidance_scale = args.classifier_guidance_scale or classifier_cfg.get("scale", 0.05)
    max_grad_norm = args.max_grad_norm or classifier_cfg.get("max_grad_norm", 1.0)
    hybrid_enabled = args.hybrid or hybrid_cfg.get("enabled", False)

    log.info("Loading KimodoSceneCo model")
    model = load_sceneco_model(
        model_cfg.get("checkpoint", "Kimodo-SMPLX-RP-v1"),
        device,
        use_zero_text_encoder=not args.real_text_encoder,
    )

    log.info("Loading RootPathSceneClassifier from %s", ckpt_path)
    root_classifier = load_classifier(ckpt_path, cfg, device)

    cache_files = find_cache_files(args.cache_dir or data_cfg.get("cache_dir"))
    if not cache_files:
        raise FileNotFoundError("No .npz or .pt cache files found for target paths")
    log.info("Found %d cache files", len(cache_files))

    root_guidance_cfg = make_root_guidance_cfg(cfg, enabled=hybrid_enabled)
    metadata = []
    guidance_rows: list[dict] = []
    current_sample_id = {"value": -1}
    install_guidance_logger(model, guidance_rows, lambda: current_sample_id["value"])

    for sample_idx, cache_file in enumerate(cache_files[: args.num_samples]):
        current_sample_id["value"] = sample_idx
        data = load_cache_data(cache_file)
        motion = get_motion_tensor(data, cache_file)
        T = min(int(motion.shape[0]), gen_cfg.get("num_frames", 196))
        if T < 2:
            continue

        motion_features = load_motion_features(cache_file)
        root_5d_meter = extract_root_5d_meter(model.motion_rep, motion_features, device=device)[:T]
        target_path_xz = torch.from_numpy(root_5d_meter[:, [0, 2]]).float().unsqueeze(0).to(device)
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

        guided_root_5d_meter = root_5d_from_output(output)[:T].astype(np.float32)
        guided_root_5d_norm = root_5d_norm_from_output(output, model, T)[:T]
        target_path_np = target_path_xz[0].detach().cpu().numpy().astype(np.float32)
        scene_name = get_scene_name(data)

        out_file = output_dir / f"{cache_file.stem}_{sample_idx:04d}.npz"
        np.savez(
            out_file,
            guided_root_5d_norm=guided_root_5d_norm,
            guided_root_5d_meter=guided_root_5d_meter,
            target_path_xz=target_path_np,
            text=np.asarray(text),
            scene_name=np.asarray(scene_name),
            source_file=np.asarray(str(cache_file)),
        )

        metadata.append(
            {
                "file": str(out_file),
                "source_cache": str(cache_file),
                "text": text,
                "scene_name": scene_name,
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
    write_guidance_log(output_dir / "guidance_log.csv", guidance_rows)
    log.info("Done. Saved %d guided roots to %s", len(metadata), output_dir)


if __name__ == "__main__":
    main()
