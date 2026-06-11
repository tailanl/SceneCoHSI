"""Export GT root from lingo_smplx_cache for Stage2.

This is the v3-compatible exporter:
- Uses CACHE-BASED index (same as dataset._load_cached_index)
- Decodes cached normalized motion_features directly, without re-normalizing
- Adds guided_root_5d_meter via motion_rep inverse
- Complete schema for README_E4_E7_FIX compliance
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(1, str(PROJECT_DIR.parent / "kimodo"))

import os
os.environ["CHECKPOINT_DIR"] = str(PROJECT_DIR / "models")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import random
import numpy as np
import torch
from tqdm.auto import tqdm

from kimodo.model.load_model import load_model

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export GT root for Stage2")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = PROJECT_DIR / "lingo_smplx_cache"

    # ---- Same index as dataset._load_cached_index ----
    npz_files = sorted(cache_dir.glob("seg_*.npz"))
    segments = []
    for f in npz_files:
        data = np.load(str(f), allow_pickle=True)
        length = int(data["length"])
        if 40 <= length <= 196:
            segments.append({"cache_path": str(f), "length": length,
                             "stem": f.stem})
    log.info(f"Total cache segments (40-196 frames): {len(segments)}")

    rng = random.Random(args.split_seed)
    indices = list(range(len(segments)))
    rng.shuffle(indices)
    n_train = int(len(indices) * args.split_ratio)
    if args.split == "train":
        chosen = sorted(indices[:n_train])
    else:
        chosen = sorted(indices[n_train:])
    chosen_segs = [segments[i] for i in chosen]
    if args.start_idx:
        chosen_segs = chosen_segs[args.start_idx:]
    if args.max_samples >= 0:
        chosen_segs = chosen_segs[:args.max_samples]
    log.info(f"Split '{args.split}': {len(chosen_segs)} segments")

    # ---- Load motion_rep for meter conversion ----
    log.info("Loading Kimodo model for root meter conversion...")
    model = load_model("Kimodo-SMPLX-RP-v1", device=device)
    model.eval()
    mr = model.motion_rep

    count = 0
    for seg in tqdm(chosen_segs, desc="Exporting GT root"):
        cache_file = Path(seg["cache_path"])
        data = np.load(str(cache_file), allow_pickle=True)
        T = seg["length"]
        feat = data["motion_features"][:T].astype(np.float32)
        text = str(data.get("text", ""))
        scene_name = str(data.get("scene_name", ""))
        source_id = cache_file.stem

        # guided_root_5d_norm: normalized root from features
        guided_root_5d_norm = feat[:, :5].astype(np.float32)

        # guided_root_5d_meter: convert to meter space
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            output = mr.inverse(feat_t, is_normalized=True, return_numpy=True)
        root_meter = output["smooth_root_pos"][0].astype(np.float32)
        heading = output["global_root_heading"][0].astype(np.float32)
        if root_meter.ndim != 2 or root_meter.shape[1] < 3:
            raise ValueError(f"{cache_file}: smooth_root_pos shape is {root_meter.shape}")
        if heading.ndim != 2 or heading.shape[1] != 2:
            raise ValueError(f"{cache_file}: global_root_heading shape is {heading.shape}")
        if root_meter.shape[0] != heading.shape[0]:
            raise ValueError(
                f"{cache_file}: root/heading length mismatch "
                f"{root_meter.shape} vs {heading.shape}"
            )
        guided_root_5d_meter = np.concatenate([root_meter[:, :3], heading], axis=-1)

        # target_path_xz
        target_path_xz = root_meter[:, [0, 2]].astype(np.float32)

        np.savez(
            str(output_dir / f"{source_id}.npz"),
            guided_root_5d_norm=guided_root_5d_norm,
            guided_root_5d_meter=guided_root_5d_meter.astype(np.float32),
            target_path_xz=target_path_xz,
            text=np.asarray(text),
            scene_name=np.asarray(scene_name),
            source_file=np.asarray(str(cache_file)),
        )
        count += 1

    log.info(f"Done. Exported {count} GT root files to {output_dir}")


if __name__ == "__main__":
    main()
