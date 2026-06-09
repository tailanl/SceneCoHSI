#!/usr/bin/env python
"""Unified visualization for all experiments.

Generates comparison visualizations:
1. Root path comparison (top-down XZ + height)
2. Body motion comparison (skeleton animation)
3. Metrics bar chart from CSVs
4. Scene collision visualization

Usage:
    python scripts/viz_all_experiments.py --exp E1 E2 E3
    python scripts/viz_all_experiments.py --all
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────── Experiment registry ───────────────
EXPERIMENTS = {
    "E0": {"name": "NoGuidance + Orig", "body_dir": None},
    "E1": {"name": "EnergyGuidance + Orig", "body_dir": "outputs/e1_energy_guidance_body"},
    "E2": {"name": "ClassifierGuidance + Orig", "body_dir": "outputs/e2_classifier_guidance_body"},
    "E3": {"name": "HybridGuidance + Orig", "body_dir": "outputs/e3_hybrid_guidance_body"},
    "E4": {"name": "Energy + Stage2", "body_dir": "outputs/e4_energy_stage2_sceneco/val_gen"},
    "E5": {"name": "Classifier + Stage2", "body_dir": "outputs/e5_classifier_stage2_sceneco/val_gen"},
    "E6": {"name": "Hybrid + Stage2", "body_dir": "outputs/e6_hybrid_stage2_sceneco/val_gen"},
    "E7": {"name": "GT + Stage2", "body_dir": "outputs/e7_gt_root_stage2_sceneco/val_gen"},
}


def load_path_metrics(exp_ids):
    """Load path metrics from CSVs."""
    rows = {}
    for eid in exp_ids:
        info = EXPERIMENTS.get(eid)
        if not info or not info["body_dir"]:
            continue
        csv_path = Path(info["body_dir"]) / "path_metrics.csv"
        if not csv_path.exists():
            print(f"  WARN: No path metrics for {eid}: {csv_path}")
            continue
        with open(csv_path) as f:
            lines = f.readlines()
        if len(lines) >= 2:
            header = lines[0].strip().split(",")
            values = lines[1].strip().split(",")
            for h, v in zip(header, values):
                try:
                    rows.setdefault(eid, {})[h] = float(v)
                except ValueError:
                    rows.setdefault(eid, {})[h] = v
    return rows


def load_scene_metrics(exp_ids):
    """Load scene metrics from CSVs."""
    rows = {}
    for eid in exp_ids:
        info = EXPERIMENTS.get(eid)
        if not info or not info["body_dir"]:
            continue
        csv_path = Path(info["body_dir"]) / "scene_metrics.csv"
        if not csv_path.exists():
            print(f"  WARN: No scene metrics for {eid}: {csv_path}")
            continue
        with open(csv_path) as f:
            lines = f.readlines()
        if len(lines) >= 2:
            header = lines[0].strip().split(",")
            values = lines[1].strip().split(",")
            for h, v in zip(header, values):
                try:
                    rows.setdefault(eid, {})[h] = float(v)
                except ValueError:
                    rows.setdefault(eid, {})[h] = v
    return rows


def plot_path_bars(path_metrics, out_path):
    """Bar chart: PathADE, PathFDE, SpeedStd, RootJerk."""
    if not path_metrics:
        print("  No path metrics to plot")
        return

    exp_ids = list(path_metrics.keys())
    names = [EXPERIMENTS[e]["name"] for e in exp_ids]

    metrics_names = ["PathADE", "PathFDE", "SpeedStd", "RootJerk"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, mn in zip(axes, metrics_names):
        values = [path_metrics[e].get(mn, 0) for e in exp_ids]
        colors = plt.cm.Set2(np.linspace(0, 1, len(exp_ids)))
        bars = ax.bar(range(len(exp_ids)), values, color=colors)
        ax.set_title(mn)
        ax.set_xticks(range(len(exp_ids)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}", ha="center", fontsize=7)

    fig.suptitle("Path Metrics Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Path bar chart saved: {out_path}")


def plot_scene_bars(scene_metrics, out_path):
    """Bar chart: CollisionFrameRate, PenetrationRate."""
    if not scene_metrics:
        print("  No scene metrics to plot")
        return

    exp_ids = list(scene_metrics.keys())
    names = [EXPERIMENTS[e]["name"] for e in exp_ids]

    metrics_names = ["CollisionFrameRate", "PenetrationRate", "PenetrationMean"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, mn in zip(axes, metrics_names):
        values = [scene_metrics[e].get(mn, 0) for e in exp_ids]
        colors = plt.cm.Set2(np.linspace(0, 1, len(exp_ids)))
        bars = ax.bar(range(len(exp_ids)), values, color=colors)
        ax.set_title(mn)
        ax.set_xticks(range(len(exp_ids)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}", ha="center", fontsize=7)

    fig.suptitle("Scene Metrics Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scene bar chart saved: {out_path}")


def plot_root_path_overlay(exp_ids, out_path, sample_idx=0):
    """Overlay root XZ paths from multiple experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_ids)))

    for i, eid in enumerate(exp_ids):
        info = EXPERIMENTS[eid]
        body_dir = Path(info["body_dir"]) if info["body_dir"] else None
        if not body_dir or not body_dir.exists():
            continue

        npz_files = sorted(body_dir.glob("sample_*.npz"))
        if not npz_files:
            npz_files = sorted(body_dir.glob("seg_*.npz"))
        if not npz_files:
            npz_files = sorted(body_dir.glob("*.npz"))

        if sample_idx >= len(npz_files):
            continue

        data = np.load(str(npz_files[sample_idx]), allow_pickle=True)
        gen_root = data.get("gen_root", None)
        gt_root_xz = data.get("gt_root_xz", None)

        if gen_root is not None:
            gen_root = np.asarray(gen_root, dtype=np.float32)
            ax1.plot(gen_root[:, 0], gen_root[:, 2], color=colors[i], alpha=0.7,
                     label=f"{eid} {info['name']}")
            ax2.plot(gen_root[:, 1], color=colors[i], alpha=0.7)

        if gt_root_xz is not None and i == 0:
            gt_root_xz = np.asarray(gt_root_xz, dtype=np.float32)
            ax1.plot(gt_root_xz[:, 0], gt_root_xz[:, 1], "k--", alpha=0.5, label="GT")

    ax1.set_title("Root XZ Path (top-down)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.legend(fontsize=7, loc="best")
    ax1.set_aspect("equal")

    ax2.set_title("Root Height over Time")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Y (m)")

    fig.suptitle(f"Root Path Comparison (sample {sample_idx})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Root path overlay saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified experiment visualization")
    parser.add_argument("--exp", nargs="+", default=["E1", "E2", "E3"],
                       help="Experiment IDs to visualize")
    parser.add_argument("--all", action="store_true",
                       help="Visualize all available experiments")
    parser.add_argument("--output_dir", type=str, default="outputs/viz_comparison",
                       help="Output directory for figures")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Sample index for trajectory overlay")
    args = parser.parse_args()

    if args.all:
        exp_ids = [e for e in EXPERIMENTS if EXPERIMENTS[e]["body_dir"]]
    else:
        exp_ids = args.exp

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing: {exp_ids}")

    # 1. Path metrics bar chart
    print("\n--- Path Metrics ---")
    path_metrics = load_path_metrics(exp_ids)
    for eid, m in path_metrics.items():
        print(f"  {eid}: PathADE={m.get('PathADE','N/A')}, PathFDE={m.get('PathFDE','N/A')}")
    plot_path_bars(path_metrics, out_dir / "path_metrics_comparison.png")

    # 2. Scene metrics bar chart
    print("\n--- Scene Metrics ---")
    scene_metrics = load_scene_metrics(exp_ids)
    for eid, m in scene_metrics.items():
        print(f"  {eid}: CFR={m.get('CollisionFrameRate','N/A')}, PenRate={m.get('PenetrationRate','N/A')}")
    plot_scene_bars(scene_metrics, out_dir / "scene_metrics_comparison.png")

    # 3. Root path overlay
    if len(exp_ids) >= 2:
        print("\n--- Root Path Overlay ---")
        plot_root_path_overlay(exp_ids, out_dir / f"root_path_overlay_{args.sample_idx:03d}.png",
                              args.sample_idx)

    # 4. Summary JSON
    summary = {
        "experiments": {eid: EXPERIMENTS[eid]["name"] for eid in exp_ids},
        "path_metrics": {},
        "scene_metrics": {},
    }
    for eid in exp_ids:
        if eid in path_metrics:
            summary["path_metrics"][eid] = {k: float(v) if isinstance(v, (np.floating, int, float)) else str(v)
                                            for k, v in path_metrics[eid].items()}
        if eid in scene_metrics:
            summary["scene_metrics"][eid] = {k: float(v) if isinstance(v, (np.floating, int, float)) else str(v)
                                             for k, v in scene_metrics[eid].items()}

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone! All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
