"""Patch existing E4 root NPZ files to add missing fields:
  guided_root_5d_meter, target_path_xz, source_file

Reads existing files, computes missing fields from gen_root/gt_root_xz,
and overwrites with complete schema.
"""

import argparse, logging, sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def heading_from_path(root_xz):
    """Compute heading [cos, sin] from XZ path."""
    if isinstance(root_xz, np.ndarray):
        from numpy import asarray
        xz = asarray(root_xz, dtype=np.float32)
        vel = xz[1:] - xz[:-1]
        theta = np.arctan2(vel[:, 1], vel[:, 0])
        theta = np.concatenate([theta, theta[-1:]])
        return np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    import torch
    vel = root_xz[1:] - root_xz[:-1]
    theta = torch.atan2(vel[:, 1], vel[:, 0])
    theta = torch.cat([theta, theta[-1:]])
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="lingo_smplx_cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    root_dir = Path(args.root_dir)
    npz_files = sorted(root_dir.glob("*.npz"))
    log.info(f"Patching {len(npz_files)} files in {root_dir}")

    patched = 0
    skipped = 0
    for f in tqdm(npz_files, desc="Patching"):
        data = np.load(str(f), allow_pickle=True)
        keys = set(data.files)
        needs_fix = "guided_root_5d_meter" not in keys or "source_file" not in keys or "target_path_xz" not in keys
        if not needs_fix:
            skipped += 1
            continue

        gen_root = np.array(data["gen_root"], dtype=np.float32)
        gt_root_xz = np.array(data.get("gt_root_xz", gen_root[:, [0, 2]]), dtype=np.float32)
        guided_root_5d_norm = np.array(data.get("guided_root_5d_norm", np.zeros((gen_root.shape[0], 5))), dtype=np.float32)
        text = str(data.get("text", ""))
        scene_name = str(data.get("scene_name", ""))

        heading = heading_from_path(gt_root_xz)
        guided_root_5d_meter = np.concatenate([gen_root[:, :3], heading], axis=-1).astype(np.float32)
        target_path_xz = gt_root_xz.astype(np.float32) if gt_root_xz.shape[-1] == 2 else gen_root[:, [0, 2]].astype(np.float32)
        source_file = f"lingo_smplx_cache/{f.stem}.npz" if "_" in f.stem else str(f)

        np.savez(
            str(f),
            gen_root=gen_root,
            gt_root_xz=gt_root_xz,
            gen_joints=np.array(data.get("gen_joints", np.zeros((gen_root.shape[0], 22, 3))), dtype=np.float32),
            gt_joints=np.array(data.get("gt_joints", np.zeros((gen_root.shape[0], 22, 3))), dtype=np.float32),
            text=np.asarray(text),
            scene_name=np.asarray(scene_name),
            guided_root_5d_norm=guided_root_5d_norm,
            guided_root_5d_meter=guided_root_5d_meter,
            target_path_xz=target_path_xz,
            source_file=np.asarray(source_file),
        )
        patched += 1

    log.info(f"Patched {patched}, skipped {skipped} (already complete)")


if __name__ == "__main__":
    main()
