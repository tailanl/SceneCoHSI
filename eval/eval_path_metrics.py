"""Evaluate path consistency metrics: PathADE, PathFDE, HeadingError, SpeedStd, etc.

Usage:
    python eval/eval_path_metrics.py \
        --pred_dir outputs/guidance_path_only \
        --output_csv outputs/guidance_path_only/path_metrics.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

import numpy as np
import torch

log = logging.getLogger(__name__)


def compute_path_metrics(root_5d, target_path_xz):
    """
    Compute path consistency metrics.

    Args:
        root_5d: (T, 5) or (T, 3) root features. If 5D, uses [x,y,z,cos,sin].
        target_path_xz: (T, 2) target path in XZ plane.

    Returns:
        Dict of metrics.
    """
    if isinstance(root_5d, np.ndarray):
        root_5d = torch.from_numpy(root_5d).float()
    if isinstance(target_path_xz, np.ndarray):
        target_path_xz = torch.from_numpy(target_path_xz).float()

    if root_5d.dim() == 3:
        root_5d = root_5d[0]
    if target_path_xz.dim() == 3:
        target_path_xz = target_path_xz[0]

    if root_5d.shape[-1] >= 3:
        pos = root_5d[..., :3]
        xz = pos[..., [0, 2]]
    else:
        xz = root_5d[..., :2]

    heading = root_5d[..., 3:5] if root_5d.shape[-1] >= 5 else None

    # PathADE
    path_ade = ((xz - target_path_xz) ** 2).sum(-1).sqrt().mean()

    # PathFDE
    path_fde = ((xz[-1] - target_path_xz[-1]) ** 2).sum(-1).sqrt()

    # Speed metrics
    vel = xz[1:] - xz[:-1]
    speed = vel.norm(dim=-1)
    speed_mean = speed.mean()
    speed_std = speed.std() if speed.numel() > 1 else torch.tensor(0.0)

    # Acceleration
    if xz.shape[0] >= 3:
        acc = xz[2:] - 2 * xz[1:-1] + xz[:-2]
        root_accel = acc.norm(dim=-1).mean()
    else:
        root_accel = torch.tensor(0.0)

    # Jerk
    if xz.shape[0] >= 4:
        jerk = xz[3:] - 3 * xz[2:-1] + 3 * xz[1:-2] - xz[:-3]
        root_jerk = jerk.norm(dim=-1).mean()
    else:
        root_jerk = torch.tensor(0.0)

    # Heading error
    heading_error = torch.tensor(0.0)
    if heading is not None and vel.shape[0] > 0:
        path_theta = torch.atan2(vel[..., 1], vel[..., 0])
        heading_theta = torch.atan2(heading[:-1, 1], heading[:-1, 0])
        diff = torch.atan2(
            torch.sin(heading_theta - path_theta),
            torch.cos(heading_theta - path_theta),
        )
        heading_error = diff.abs().mean()

    # Root Y smoothness
    if root_5d.shape[-1] >= 3:
        root_y = root_5d[:, 1]
        root_y_smooth = ((root_y[1:] - root_y[:-1]) ** 2).mean()
    else:
        root_y_smooth = torch.tensor(0.0)

    return {
        "PathADE": path_ade.item(),
        "PathFDE": path_fde.item(),
        "SpeedMean": speed_mean.item(),
        "SpeedStd": speed_std.item(),
        "RootAccel": root_accel.item(),
        "RootJerk": root_jerk.item(),
        "HeadingError": heading_error.item(),
        "RootYSmooth": root_y_smooth.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Path metrics evaluation")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--method", type=str, default="path_guidance")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pred_dir = Path(args.pred_dir)
    npz_files = sorted(pred_dir.glob("sample_*.npz"))
    if not npz_files:
        npz_files = sorted(pred_dir.glob("seg_*.npz"))
    if not npz_files:
        npz_files = sorted(pred_dir.glob("*.npz"))
    log.info(f"Found {len(npz_files)} samples in {pred_dir}")

    all_metrics = []
    for npz_file in npz_files:
        data = np.load(str(npz_file), allow_pickle=True)
        gen_root = np.array(data["gen_root"], dtype=np.float32)  # (T, 3)
        gt_root_xz = np.array(data["gt_root_xz"], dtype=np.float32)  # (T, 2)

        metrics = compute_path_metrics(gen_root, gt_root_xz)
        metrics["sample_id"] = npz_file.stem
        metrics["method"] = args.method
        all_metrics.append(metrics)

    # Write CSV
    if all_metrics:
        fieldnames = ["sample_id", "method", "PathADE", "PathFDE", "SpeedMean", "SpeedStd",
                      "RootAccel", "RootJerk", "HeadingError", "RootYSmooth"]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

        # Print summary
        avg = {}
        for key in ["PathADE", "PathFDE", "SpeedMean", "SpeedStd", "RootAccel", "RootJerk",
                     "HeadingError", "RootYSmooth"]:
            vals = [m[key] for m in all_metrics]
            avg[key] = np.mean(vals)
            log.info(f"  {key}: {avg[key]:.4f}")

    log.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
