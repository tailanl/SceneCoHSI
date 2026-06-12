#!/usr/bin/env python
"""Summarize latest checkpoint prediction metrics with explicit Pene column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EXPS = ["E4", "E5", "E6", "E7", "E8", "E9", "E10"]


def read_mean(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for col in df.columns:
        if col in {"sample_id", "method"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.isna().all():
            out[col] = float(values.mean())
    return out


def summarize(eval_root: Path, exps: list[str]) -> pd.DataFrame:
    rows = []
    for exp in exps:
        pred_dir = eval_root / exp / "pred"
        path_metrics = read_mean(pred_dir / "path_metrics.csv")
        scene_metrics = read_mean(pred_dir / "scene_metrics.csv")
        row = {
            "experiment": exp,
            "pred_dir": str(pred_dir),
            **path_metrics,
            **scene_metrics,
        }
        if "PenetrationMean" in row:
            row["Pene"] = row["PenetrationMean"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "PathADE",
        "PathFDE",
        "CollisionFrameRate",
        "NonWalkableRootRate",
        "PenetrationRate",
        "Pene",
    ]
    metrics = [m for m in metrics if m in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    labels = df["experiment"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for ax, metric in zip(axes, metrics):
        values = pd.to_numeric(df[metric], errors="coerce").fillna(0.0).to_numpy()
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_title(metric)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ymax = max(float(values.max()), 1e-6)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.015,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Latest Checkpoint Prediction Metrics")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_analysis(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Latest Checkpoint Prediction Analysis",
        "",
        "Definitions:",
        "",
        "- `PenetrationRate`: fraction of non-floor joints inside occupied scene voxels.",
        "- `Pene`: alias of `PenetrationMean`, kept as a separate report column for penetration depth/amount.",
        "",
        "## Summary",
        "",
        df.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    for metric in ["PathADE", "CollisionFrameRate", "PenetrationRate", "Pene"]:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce")
            if not values.isna().all():
                best_idx = values.idxmin()
                worst_idx = values.idxmax()
                lines.append(
                    f"- Lowest {metric}: {df.loc[best_idx, 'experiment']} ({values.loc[best_idx]:.6f}); "
                    f"highest: {df.loc[worst_idx, 'experiment']} ({values.loc[worst_idx]:.6f})."
                )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="outputs/retrain_mirrorfix50/latest_ckpt_eval")
    parser.add_argument("--exps", nargs="+", default=DEFAULT_EXPS)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    df = summarize(eval_root, args.exps)
    out_dir = eval_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "latest_metrics_summary.csv", index=False)
    (out_dir / "latest_metrics_summary.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    plot_summary(df, out_dir / "latest_metrics_summary.png")
    write_analysis(df, out_dir / "ANALYSIS.md")
    print(json.dumps({"summary_dir": str(out_dir), "experiments": args.exps}, indent=2))


if __name__ == "__main__":
    main()
