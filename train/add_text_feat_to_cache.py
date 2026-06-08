#!/usr/bin/env python
"""Add text_feat from lingo_root_trajectory_smplx to lingo_smplx_cache.

One-time operation: copies pre-computed LLM2Vec text features from the root
trajectory cache into the full motion cache so Stage2 training can use them.

Usage:
  python kimodo_scene_project/train/add_text_feat_to_cache.py
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_CACHE = ROOT / "lingo_root_trajectory_smplx"
DST_CACHE = ROOT / "lingo_smplx_cache"


def main():
    src_files = sorted(SRC_CACHE.glob("seg_*.npz"))
    print(f"Source (root_traj): {len(src_files)} files")
    print(f"Destination (smplx_cache): {len(list(DST_CACHE.glob('seg_*.npz')))} files")

    added = 0
    skipped = 0
    missing = 0

    for src_path in src_files:
        dst_path = DST_CACHE / src_path.name
        if not dst_path.exists():
            missing += 1
            continue

        # Check if text_feat already exists
        dst_data = np.load(str(dst_path), allow_pickle=True)
        if "text_feat" in dst_data.keys():
            skipped += 1
            continue

        # Load text_feat from source
        src_data = np.load(str(src_path), allow_pickle=True)
        if "text_feat" not in src_data.keys():
            print(f"  WARNING: {src_path.name} has no text_feat in source")
            missing += 1
            continue

        text_feat = src_data["text_feat"]

        # Rebuild the npz with text_feat added
        keys = list(dst_data.keys())
        arrays = {k: dst_data[k] for k in keys}
        arrays["text_feat"] = text_feat

        # Save back (np.savez adds .npz extension automatically)
        tmp_path = str(dst_path.with_suffix("")) + "_tmp"
        np.savez(tmp_path, **arrays)
        # np.savez creates tmp_path + ".npz"
        actual_tmp = tmp_path + ".npz"
        os.replace(actual_tmp, str(dst_path))

        added += 1
        if added % 1000 == 0:
            print(f"  Processed {added} files...")

    print(f"\nDone! Added: {added}, Skipped (already has text_feat): {skipped}, Missing: {missing}")


if __name__ == "__main__":
    main()
