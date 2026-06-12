#!/usr/bin/env python
"""Prepare unified test registry and structure visualization for retrain_mirrorfix50.

This script does not train or generate motions. It only:
  1. Links no-train baselines E1-E3 into the current run root.
  2. Writes a machine-readable experiment registry.
  3. Renders a compact experiment-structure diagram and no-train root overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


EXPERIMENTS = [
    {
        "id": "E1",
        "family": "no_train_baseline",
        "root_method": "energy_guidance",
        "body_method": "original_kimodo",
        "train_required": False,
        "source_body_dir": "outputs/e1_energy_guidance_body",
        "source_root_dir": "outputs/e1_energy_guidance_root",
        "run_body_dir": "no_train/e1_energy_guidance_body",
        "notes": "No SceneCo body training; existing generated body samples.",
    },
    {
        "id": "E2",
        "family": "no_train_baseline",
        "root_method": "classifier_guidance",
        "body_method": "original_kimodo",
        "train_required": False,
        "source_body_dir": "outputs/e2_classifier_guidance_body",
        "source_root_dir": "outputs/e2_classifier_guidance_root",
        "run_body_dir": "no_train/e2_classifier_guidance_body",
        "notes": "No SceneCo body training; existing generated body samples.",
    },
    {
        "id": "E3",
        "family": "no_train_baseline",
        "root_method": "hybrid_guidance",
        "body_method": "original_kimodo",
        "train_required": False,
        "source_body_dir": "outputs/e3_hybrid_guidance_body",
        "source_root_dir": "outputs/e3_hybrid_guidance_root",
        "run_body_dir": "no_train/e3_hybrid_guidance_body",
        "notes": "No SceneCo body training; existing generated body samples.",
    },
    {
        "id": "E4",
        "family": "stage2_sceneco",
        "root_method": "energy_guidance",
        "body_method": "stage2_sceneco",
        "train_required": True,
        "run_body_dir": "e4_energy_stage2",
        "train_root_dir": "outputs/e4_energy_guidance_train/path_only",
        "val_root_dir": "outputs/e4_energy_guidance_val/path_only",
    },
    {
        "id": "E5",
        "family": "stage2_sceneco",
        "root_method": "classifier_guidance",
        "body_method": "stage2_sceneco",
        "train_required": True,
        "run_body_dir": "e5_classifier_stage2",
        "train_root_dir": "outputs/e5_classifier_guidance_train/path_only",
        "val_root_dir": "outputs/e5_classifier_guidance_val/path_only",
    },
    {
        "id": "E6",
        "family": "stage2_sceneco",
        "root_method": "hybrid_guidance",
        "body_method": "stage2_sceneco",
        "train_required": True,
        "run_body_dir": "e6_hybrid_stage2",
        "train_root_dir": "outputs/e6_hybrid_guidance_train/path_only",
        "val_root_dir": "outputs/e6_hybrid_guidance_val/path_only",
    },
    {
        "id": "E7",
        "family": "stage2_sceneco",
        "root_method": "gt_root",
        "body_method": "stage2_sceneco",
        "train_required": True,
        "run_body_dir": "e7_gt_stage2",
        "train_root_dir": "outputs/e7_gt_root_v3_train",
        "val_root_dir": "outputs/e7_gt_root_v3_val",
    },
    {
        "id": "E8",
        "family": "raw3d_sceneco",
        "root_method": "classifier_guidance_raw3d",
        "body_method": "stage2_sceneco",
        "train_required": True,
        "run_body_dir": "e8_classifier_raw3d_stage2",
        "train_root_dir": "outputs/e8_classifier_raw3d_train",
        "val_root_dir": "outputs/e8_classifier_raw3d_val",
    },
    {
        "id": "E9",
        "family": "raw3d_sceneco",
        "root_method": "hybrid_guidance_raw3d",
        "body_method": "stage2_sceneco",
        "train_required": True,
        "run_body_dir": "e9_hybrid_raw3d_stage2",
        "train_root_dir": "outputs/e9_hybrid_raw3d_train",
        "val_root_dir": "outputs/e9_hybrid_raw3d_val",
    },
    {
        "id": "E10",
        "family": "raw3d_sceneco",
        "root_method": "gt_root_projected",
        "body_method": "stage2_sceneco",
        "train_required": True,
        "run_body_dir": "e10_gt_projected_stage2",
        "train_root_dir": "outputs/e10_gt_projected_train",
        "val_root_dir": "outputs/e10_gt_projected_val",
    },
    {
        "id": "T1",
        "family": "trajco_comparison",
        "root_method": "root_trajco_stage1",
        "body_method": "stage2_sceneco_body",
        "train_required": True,
        "run_body_dir": "root_trajco_stage2_sceneco_body",
        "stage1_dir": "root_trajco_stage1",
        "notes": "Two-step comparison: root TrajCo Stage1, then SceneCo body Stage2.",
    },
]


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def count_npz(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.npz"))


def safe_symlink(target: Path, link: Path) -> str:
    if not target.exists():
        return "missing_target"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        current = link.resolve()
        if current == target.resolve():
            return "exists"
        link.unlink()
    elif link.exists():
        return "exists_non_symlink"
    rel_target = os.path.relpath(target, start=link.parent)
    link.symlink_to(rel_target, target_is_directory=target.is_dir())
    return "created"


def build_registry(run_root: Path, link_no_train: bool) -> list[dict]:
    rows = []
    for exp in EXPERIMENTS:
        row = dict(exp)
        run_body = run_root / exp["run_body_dir"]

        if exp["family"] == "no_train_baseline":
            source_body = resolve(exp["source_body_dir"])
            source_root = resolve(exp["source_root_dir"])
            row["source_body_dir_abs"] = str(source_body)
            row["source_root_dir_abs"] = str(source_root)
            row["body_link_status"] = safe_symlink(source_body, run_body) if link_no_train else "disabled"
            root_link = run_root / "no_train" / f"{exp['id'].lower()}_root"
            row["root_link_status"] = safe_symlink(source_root, root_link) if link_no_train else "disabled"
            row["body_npz_count"] = count_npz(run_body if run_body.exists() else source_body)
            row["checkpoint_exists"] = False
        else:
            checkpoint = run_body / "checkpoints" / "best_checkpoint.pt"
            row["body_npz_count"] = count_npz(run_body)
            row["checkpoint"] = str(checkpoint)
            row["checkpoint_exists"] = checkpoint.exists()
            if "train_root_dir" in exp:
                row["train_root_count"] = count_npz(resolve(exp["train_root_dir"]))
            if "val_root_dir" in exp:
                row["val_root_count"] = count_npz(resolve(exp["val_root_dir"]))
            if "stage1_dir" in exp:
                stage1_ckpt = run_root / exp["stage1_dir"] / "checkpoints" / "best_checkpoint.pt"
                row["stage1_checkpoint"] = str(stage1_ckpt)
                row["stage1_checkpoint_exists"] = stage1_ckpt.exists()

        row["run_body_dir_abs"] = str(run_body)
        rows.append(row)
    return rows


def write_registry(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_registry.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    keys = sorted({key for row in rows for key in row})
    with (out_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_structure(rows: list[dict], out_path: Path) -> None:
    colors = {
        "no_train_baseline": "#B7E3C0",
        "stage2_sceneco": "#BFD7F6",
        "raw3d_sceneco": "#F7D8A8",
        "trajco_comparison": "#D8C2F0",
    }
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    ax.text(0.25, 11.45, "Root source / method", fontsize=12, weight="bold")
    ax.text(3.6, 11.45, "Body stage", fontsize=12, weight="bold")
    ax.text(6.25, 11.45, "Run-root output", fontsize=12, weight="bold")
    ax.text(8.7, 11.45, "Test status", fontsize=12, weight="bold")

    y = 10.8
    for row in rows:
        color = colors.get(row["family"], "#DDDDDD")
        eid = row["id"]
        root_label = f"{eid}: {row['root_method']}"
        body_label = row["body_method"]
        output_label = row["run_body_dir"]
        if row["train_required"]:
            status = "ckpt: yes" if row.get("checkpoint_exists") else "ckpt: pending"
        else:
            status = f"npz: {row.get('body_npz_count', 0)}"

        for x, text, width in [
            (0.2, root_label, 2.65),
            (3.35, body_label, 2.25),
            (6.0, output_label, 2.25),
            (8.65, status, 1.15),
        ]:
            ax.add_patch(
                plt.Rectangle((x, y - 0.38), width, 0.55, facecolor=color, edgecolor="#555555", linewidth=0.8)
            )
            ax.text(x + 0.08, y - 0.1, text, fontsize=8, va="center", ha="left")
        ax.annotate("", xy=(3.3, y - 0.1), xytext=(2.9, y - 0.1), arrowprops={"arrowstyle": "->", "lw": 0.8})
        ax.annotate("", xy=(5.95, y - 0.1), xytext=(5.65, y - 0.1), arrowprops={"arrowstyle": "->", "lw": 0.8})
        ax.annotate("", xy=(8.6, y - 0.1), xytext=(8.3, y - 0.1), arrowprops={"arrowstyle": "->", "lw": 0.8})
        y -= 0.85

    legend_y = 0.85
    for i, (family, color) in enumerate(colors.items()):
        x = 0.25 + i * 2.3
        ax.add_patch(plt.Rectangle((x, legend_y), 0.25, 0.25, facecolor=color, edgecolor="#555555"))
        ax.text(x + 0.33, legend_y + 0.13, family, fontsize=8, va="center")

    ax.set_title("retrain_mirrorfix50 Experiment Structure", fontsize=15, weight="bold", pad=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_no_train_overlay(run_root: Path, out_path: Path, sample_idx: int) -> None:
    baseline_dirs = [
        ("E1", run_root / "no_train" / "e1_energy_guidance_body"),
        ("E2", run_root / "no_train" / "e2_classifier_guidance_body"),
        ("E3", run_root / "no_train" / "e3_hybrid_guidance_body"),
    ]
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"E1": "#1F77B4", "E2": "#2CA02C", "E3": "#D62728"}
    plotted = 0
    for eid, body_dir in baseline_dirs:
        files = sorted(body_dir.glob("sample_*.npz")) or sorted(body_dir.glob("*.npz"))
        if sample_idx >= len(files):
            continue
        data = np.load(files[sample_idx], allow_pickle=True)
        gen_root = np.asarray(data["gen_root"], dtype=np.float32)
        ax.plot(gen_root[:, 0], gen_root[:, 2], color=colors[eid], linewidth=2.0, label=eid)
        if plotted == 0 and "gt_root_xz" in data.files:
            gt = np.asarray(data["gt_root_xz"], dtype=np.float32)
            ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=1.5, label="GT path")
        plotted += 1

    ax.set_title(f"No-train baseline root overlay (sample_idx={sample_idx})")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# retrain_mirrorfix50 Test Registry",
        "",
        "All experiment test artifacts for this run should be written under `outputs/retrain_mirrorfix50`.",
        "No-train baselines are linked into `outputs/retrain_mirrorfix50/no_train` instead of copied.",
        "",
        "| ID | Family | Root | Body | Train? | Run output | Status |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        if row["train_required"]:
            status = "checkpoint ready" if row.get("checkpoint_exists") else "checkpoint pending"
        else:
            status = f"{row.get('body_npz_count', 0)} npz"
        lines.append(
            f"| {row['id']} | {row['family']} | {row['root_method']} | {row['body_method']} | "
            f"{str(row['train_required']).lower()} | `{row['run_body_dir']}` | {status} |"
        )
    lines += [
        "",
        "Generated files:",
        "",
        "- `eval_viz/experiment_registry.json`",
        "- `eval_viz/experiment_registry.csv`",
        "- `eval_viz/figures/experiment_structure.png`",
        "- `eval_viz/figures/no_train_root_overlay.png`",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="outputs/retrain_mirrorfix50")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--no_link_no_train", action="store_true")
    args = parser.parse_args()

    run_root = resolve(args.run_root)
    out_dir = run_root / "eval_viz"
    fig_dir = out_dir / "figures"

    rows = build_registry(run_root, link_no_train=not args.no_link_no_train)
    write_registry(rows, out_dir)
    plot_structure(rows, fig_dir / "experiment_structure.png")
    plot_no_train_overlay(run_root, fig_dir / "no_train_root_overlay.png", args.sample_idx)
    write_markdown(rows, out_dir / "README.md")

    payload = {
        "run_root": str(run_root),
        "registry": str(out_dir / "experiment_registry.json"),
        "structure_png": str(fig_dir / "experiment_structure.png"),
        "no_train_overlay_png": str(fig_dir / "no_train_root_overlay.png"),
        "experiments": len(rows),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
