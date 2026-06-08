"""Build or validate cache data for RootPathSceneClassifier training.

The classifier dataset can read Kimodo/LINGO cache files directly. This script
adds a reproducible preflight step and can optionally export compact files with
only root_5d, target_path_xz, and metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def get_motion_tensor(data: dict, file: Path) -> torch.Tensor | None:
    motion = data.get("motion", data.get("beta_motion", data.get("data", data.get("motion_features"))))
    if motion is None:
        print(f"SKIP {file}: no motion-like key")
        return None
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    if motion.ndim != 2 or motion.shape[-1] < 5:
        print(f"SKIP {file}: expected (T,D>=5), got {tuple(motion.shape)}")
        return None
    return motion.float()


def get_text(data: dict) -> str:
    for key in ("text", "caption", "prompt", "description"):
        if key in data and data[key] is not None:
            return str(data[key])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="LINGO/dataset/dataset/lingo_smplx_cache")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default="outputs/root_classifier_dataset_manifest.json")
    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--min_frames", type=int, default=2)
    parser.add_argument("--export_compact", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.export_compact:
        if output_dir is None:
            raise ValueError("--output_dir is required with --export_compact")
        output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    files = sorted(cache_dir.glob("*.pt"))
    for idx, file in enumerate(files):
        data = torch.load(file, map_location="cpu", weights_only=False)
        motion = get_motion_tensor(data, file)
        if motion is None:
            continue
        T = min(int(motion.shape[0]), args.max_frames)
        if T < args.min_frames:
            continue

        root_5d = motion[:T, :5].clone()
        target_path_xz = root_5d[:, [0, 2]].clone()
        entry = {
            "source": str(file),
            "num_frames": T,
            "text": get_text(data),
            "scene": str(data.get("scene", data.get("scene_name", ""))),
        }

        if args.export_compact and output_dir is not None:
            out_file = output_dir / f"root_classifier_{idx:06d}.pt"
            torch.save(
                {
                    "motion": root_5d,
                    "target_path_xz": target_path_xz,
                    "text": entry["text"],
                    "scene": entry["scene"],
                    "source": str(file),
                },
                out_file,
            )
            entry["compact_file"] = str(out_file)

        entries.append(entry)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        json.dump(
            {
                "cache_dir": str(cache_dir),
                "num_source_files": len(files),
                "num_valid": len(entries),
                "max_frames": args.max_frames,
                "entries": entries,
            },
            f,
            indent=2,
        )

    print(f"Valid samples: {len(entries)}/{len(files)}")
    print(f"Wrote manifest: {manifest}")
    if args.export_compact and output_dir is not None:
        print(f"Wrote compact dataset: {output_dir}")


if __name__ == "__main__":
    main()
