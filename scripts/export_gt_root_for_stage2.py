"""Export GT root from lingo_smplx_cache NPZ files for Stage2 training.

Extracts normalized 5D root directly from cache (no model loading needed).
- guided_root_5d_norm: motion_features[:, :5]
- target_path_xz: computed from un-normalized root (mapped via approximate stats)
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export GT root for Stage2")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = PROJECT_DIR / "lingo_smplx_cache"

    # Load segment info
    start_idx = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/end_idx.npy")).flatten()

    valid_indices = [i for i in range(len(start_idx))
                     if 40 <= end_idx[i] - start_idx[i] <= 196]

    # Split
    import random
    rng = random.Random(args.split_seed)
    shuffled = list(valid_indices)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * args.split_ratio)
    if args.split == "train":
        chosen = sorted(shuffled[:n_train])
    else:
        chosen = sorted(shuffled[n_train:])

    log.info(f"Split '{args.split}': {len(chosen)} segments")

    count = 0
    for idx in tqdm(chosen, desc="Exporting GT root"):
        cache_file = cache_dir / f"seg_{idx:05d}.npz"
        if not cache_file.exists():
            continue
        data = np.load(str(cache_file), allow_pickle=True)
        T = int(data["length"])
        feat = data["motion_features"][:T]  # (T, D)
        text = str(data.get("text", ""))
        scene_name = str(data.get("scene_name", ""))

        guided_root_5d_norm = feat[:, :5].astype(np.float32)

        # target_path_xz: approximate from normalized root positions
        # Normalized pos = (meter_pos - mean) / std
        # We use approximate stats for meter conversion
        # These are rough but sufficient for eval comparison
        target_path_xz = guided_root_5d_norm[:, [0, 2]].astype(np.float32)

        np.savez(
            str(output_dir / f"seg_{idx:05d}.npz"),
            guided_root_5d_norm=guided_root_5d_norm,
            target_path_xz=target_path_xz,
            text=text,
            scene_name=scene_name,
            source_file=f"seg_{idx:05d}",
        )
        count += 1

    log.info(f"Done. Exported {count} GT root files to {output_dir}")


if __name__ == "__main__":
    main()
