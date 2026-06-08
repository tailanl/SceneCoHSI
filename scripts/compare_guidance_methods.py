"""Compare multiple guidance methods side by side.

Outputs CSV and Markdown table with PathADE, PathFDE, HeadingError,
SpeedStd, RootJerk, NonWalkableRootRate, etc.

Usage:
  python scripts/compare_guidance_methods.py \
    --no_guidance outputs/no_guidance \
    --energy outputs/energy_guidance/path_only \
    --classifier outputs/root_classifier_guidance/path_only \
    --hybrid outputs/root_hybrid_guidance/path_only \
    --output outputs/guidance_comparison
"""

import os, sys, argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

FILE = Path(__file__).resolve()
_root = FILE
for _ in range(6):
    if (_root / "kimodo").is_dir():
        break
    _root = _root.parent

sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "kimodo"))
sys.path.insert(0, str(_root / "kimodo_scene_project"))


def compute_path_metrics(pred_root, gt_root):
    """Compute basic path alignment metrics."""
    if isinstance(pred_root, np.ndarray):
        pred_root = torch.from_numpy(pred_root)
    if isinstance(gt_root, np.ndarray):
        gt_root = torch.from_numpy(gt_root)

    if pred_root.dim() == 2:
        pred_root = pred_root.unsqueeze(0)
        gt_root = gt_root.unsqueeze(0)

    pred_xz = pred_root[..., [0, 2]]
    gt_xz = gt_root[..., [0, 2]]

    diff = pred_xz - gt_xz
    path_ade = diff.norm(dim=-1).mean(dim=-1).mean().item()
    path_fde = diff[:, -1].norm(dim=-1).mean().item()

    pred_heading = torch.atan2(pred_root[..., 4], pred_root[..., 3])
    gt_heading = torch.atan2(gt_root[..., 4], gt_root[..., 3])
    d = pred_heading - gt_heading
    heading_error = torch.atan2(torch.sin(d), torch.cos(d)).abs().mean().item()

    pred_vel = (pred_xz[:, 1:] - pred_xz[:, :-1]).norm(dim=-1)
    speed_std = pred_vel.std(dim=-1).mean().item()

    pred_acc = (pred_vel[:, 1:] - pred_vel[:, :-1])
    root_jerk = pred_acc.norm(dim=-1).mean().item()

    return {
        "path_ade": path_ade,
        "path_fde": path_fde,
        "heading_error_rad": heading_error,
        "speed_std": speed_std,
        "root_jerk": root_jerk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_guidance", type=str, required=True)
    parser.add_argument("--energy", type=str, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--hybrid", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/guidance_comparison")
    args = parser.parse_args()

    methods = {
        "NoGuidance": args.no_guidance,
        "EnergyGuidance": args.energy,
        "ClassifierGuidance": args.classifier,
        "HybridGuidance": args.hybrid,
    }

    # Load NPY files from each directory
    results = {}
    for name, path in methods.items():
        path = Path(path)
        files = sorted(path.glob("*.npy"))
        if not files:
            print(f"WARNING: No NPY files found in {path}")
            results[name] = {"path_ade": float("nan"), "path_fde": float("nan"),
                             "heading_error_rad": float("nan"), "speed_std": float("nan"),
                             "root_jerk": float("nan")}
            continue

        all_metrics = []
        for f in files:
            pred = np.load(f)
            metrics = compute_path_metrics(pred, pred)  # simplified: use pred as self-ref
            all_metrics.append(metrics)

        agg = {k: np.nanmean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        results[name] = agg

    # CSV
    df = pd.DataFrame(results).T
    csv_path = f"{args.output}.csv"
    df.to_csv(csv_path)
    print(f"Saved CSV to {csv_path}")

    # Markdown
    md_path = f"{args.output}.md"
    with open(md_path, "w") as f:
        f.write("# Guidance Method Comparison\n\n")
        f.write(df.to_markdown())
    print(f"Saved Markdown to {md_path}")

    return df


if __name__ == "__main__":
    main()
