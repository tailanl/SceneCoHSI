#!/usr/bin/env python
"""Build per-sample metrics, anomaly tables, and top-down scene videos.

The renderer is intentionally top-down: X/Z are screen axes, Y is ignored except
through the generated joints. Scene occupancy is shown using the same cache-based
alignment used by eval_sceneadapt_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from eval.eval_path_metrics import compute_path_metrics
from eval.eval_sceneadapt_metrics import (
    DEFAULT_FLOOR_IGNORE_HEIGHT,
    DEFAULT_RAW_SCENE_DIR,
    RAW_SCENE_VOXEL_SIZE,
    build_2d_sdf_from_cache,
    build_scene_cache_index,
    compute_scene_metrics,
    load_raw_scene,
    load_scene_cache_from_index,
)


SMPLX_22_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]


EXPERIMENTS = OrderedDict(
    [
        ("E1", {
            "name": "EnergyGuidance + Original Body",
            "body_dir": "outputs/e1_energy_guidance_body",
        }),
        ("E2", {
            "name": "ClassifierGuidance + Original Body",
            "body_dir": "outputs/e2_classifier_guidance_body",
        }),
        ("E3", {
            "name": "HybridGuidance + Original Body",
            "body_dir": "outputs/e3_hybrid_guidance_body",
        }),
        ("E5", {
            "name": "ClassifierGuidance + Stage2 SceneCo",
            "body_dir": "outputs/e5_v3_stage2/val_gen",
        }),
        ("E7", {
            "name": "GTRoot + Stage2 SceneCo",
            "body_dir": "outputs/e7_v3_stage2/val_gen",
        }),
    ]
)


PATH_KEYS = [
    "PathADE",
    "PathFDE",
    "SpeedMean",
    "SpeedStd",
    "RootAccel",
    "RootJerk",
    "HeadingError",
    "RootYSmooth",
]
SCENE_KEYS = [
    "CollisionFrameRate",
    "NonWalkableRootRate",
    "PenetrationRate",
    "PenetrationMean",
    "PenetrationMax",
    "SceneSDFPenalty",
]
ANOMALY_KEYS = [
    "PathADE",
    "PathFDE",
    "CollisionFrameRate",
    "PenetrationRate",
    "RootJerk",
]

RAW_SCENE_CACHE = {}


def find_npz_files(body_dir: Path) -> list[Path]:
    files = sorted(body_dir.glob("sample_*.npz"))
    if not files:
        files = sorted(body_dir.glob("seg_*.npz"))
    if not files:
        files = sorted(body_dir.glob("*.npz"))
    return files


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def compute_all_metrics(cache_dir: str, max_samples: int | None = None) -> pd.DataFrame:
    scene_index = build_scene_cache_index(cache_dir)
    rows = []

    for exp_id, info in EXPERIMENTS.items():
        body_dir = Path(info["body_dir"])
        files = find_npz_files(body_dir) if body_dir.exists() else []
        if max_samples is not None:
            files = files[:max_samples]
        print(f"[metrics] {exp_id}: {len(files)} files from {body_dir}")

        for path in files:
            data = np.load(path, allow_pickle=True)
            gen_root = np.asarray(data["gen_root"], dtype=np.float32)
            gt_root_xz = np.asarray(data["gt_root_xz"], dtype=np.float32)
            gen_joints = np.asarray(data["gen_joints"], dtype=np.float32)
            scene_name = scalar_str(data.get("scene_name", ""))
            text = scalar_str(data.get("text", ""))

            path_metrics = compute_path_metrics(gen_root, gt_root_xz)
            scene_metrics = compute_scene_metrics(
                gen_root,
                gen_joints,
                scene_name,
                cache_dir,
                scene_index=scene_index,
            )
            rows.append({
                "experiment": exp_id,
                "model": info["name"],
                "sample_id": path.stem,
                "body_file": str(path),
                "scene_name": scene_name,
                "text": text,
                "num_frames": int(gen_root.shape[0]),
                **path_metrics,
                **scene_metrics,
            })

    return pd.DataFrame(rows)


def add_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score = np.zeros(len(df), dtype=np.float32)
    for key in ANOMALY_KEYS:
        if key not in df:
            continue
        values = pd.to_numeric(df[key], errors="coerce")
        ranks = values.rank(pct=True).fillna(0.0).to_numpy(dtype=np.float32)
        score += ranks
        df[f"{key}_rank"] = ranks
    df["anomaly_score"] = score / max(1, len([k for k in ANOMALY_KEYS if k in df]))
    return df.sort_values(["anomaly_score"], ascending=False)


def aggregate_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in PATH_KEYS + SCENE_KEYS if c in df.columns]
    grouped = (
        df.groupby(["experiment", "model"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("experiment")
    )
    grouped.insert(2, "samples", df.groupby(["experiment", "model"]).size().values)
    return grouped


def plot_metric_summary(summary: pd.DataFrame, out_path: Path) -> None:
    candidate_metrics = [
        "PathADE",
        "PathFDE",
        "CollisionFrameRate",
        "NonWalkableRootRate",
        "PenetrationRate",
        "SceneSDFPenalty",
    ]
    metrics = [
        m for m in candidate_metrics
        if m in summary.columns and not pd.to_numeric(summary[m], errors="coerce").isna().all()
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    labels = summary["experiment"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(summary)))

    for ax, metric in zip(axes, metrics):
        values = summary[metric].to_numpy(dtype=np.float32)
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_title(metric)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Completed Model Metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def scene_background(scene_name: str, scene_index: dict, gen_root: np.ndarray):
    raw_scene = load_raw_scene(DEFAULT_RAW_SCENE_DIR, scene_name, raw_scene_cache=RAW_SCENE_CACHE)
    if raw_scene is not None:
        depth, height, width = raw_scene.shape
        y0 = max(1, int(DEFAULT_FLOOR_IGNORE_HEIGHT / RAW_SCENE_VOXEL_SIZE))
        y1 = min(height, int(2.0 / RAW_SCENE_VOXEL_SIZE))
        occ_zx = raw_scene[:, y0:y1, :].any(axis=1)
        occ_xz = occ_zx.T
        extent = [
            -width * RAW_SCENE_VOXEL_SIZE / 2.0,
            width * RAW_SCENE_VOXEL_SIZE / 2.0,
            -depth * RAW_SCENE_VOXEL_SIZE / 2.0,
            depth * RAW_SCENE_VOXEL_SIZE / 2.0,
        ]
        return occ_xz, None, extent

    voxel_64, _ = load_scene_cache_from_index(scene_index, scene_name)
    if voxel_64 is None:
        return None, None, None

    x_min, x_max = float(gen_root[:, 0].min()), float(gen_root[:, 0].max())
    z_min, z_max = float(gen_root[:, 2].min()), float(gen_root[:, 2].max())
    sdf, grid = build_2d_sdf_from_cache(voxel_64, (x_min, x_max), (z_min, z_max))
    occ = sdf < 0
    extent = [grid["x_min"], grid["x_max"], grid["z_min"], grid["z_max"]]
    return occ, sdf, extent


def compute_limits(gen_root: np.ndarray, gt_root_xz: np.ndarray, gen_joints: np.ndarray, extent):
    xs = [gen_root[:, 0], gen_joints[..., 0].reshape(-1)]
    zs = [gen_root[:, 2], gen_joints[..., 2].reshape(-1)]
    if gt_root_xz is not None:
        xs.append(gt_root_xz[:, 0])
        zs.append(gt_root_xz[:, 1])
    if extent is not None:
        xs.extend([np.asarray([extent[0], extent[1]])])
        zs.extend([np.asarray([extent[2], extent[3]])])
    x_all = np.concatenate(xs)
    z_all = np.concatenate(zs)
    margin = max(0.5, 0.1 * max(x_all.max() - x_all.min(), z_all.max() - z_all.min()))
    return (
        float(x_all.min() - margin),
        float(x_all.max() + margin),
        float(z_all.min() - margin),
        float(z_all.max() + margin),
    )


def render_video(row: pd.Series, scene_index: dict, output_dir: Path, fps: int, frame_stride: int) -> Path:
    body_file = Path(row["body_file"])
    data = np.load(body_file, allow_pickle=True)
    gen_root = np.asarray(data["gen_root"], dtype=np.float32)
    gen_joints = np.asarray(data["gen_joints"], dtype=np.float32)
    gt_root_xz = np.asarray(data["gt_root_xz"], dtype=np.float32) if "gt_root_xz" in data.files else None
    scene_name = scalar_str(data.get("scene_name", ""))
    text = scalar_str(data.get("text", ""))

    occ, sdf, extent = scene_background(scene_name, scene_index, gen_root)
    x0, x1, z0, z1 = compute_limits(gen_root, gt_root_xz, gen_joints, extent)

    safe_sample = str(row["sample_id"]).replace("/", "_")
    out_path = output_dir / f"{row['experiment']}_{safe_sample}_topdown.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = list(range(0, gen_root.shape[0], max(1, frame_stride)))
    if frames[-1] != gen_root.shape[0] - 1:
        frames.append(gen_root.shape[0] - 1)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)

    metric_text = (
        f"PathADE={row.get('PathADE', np.nan):.3f}  "
        f"CFR={row.get('CollisionFrameRate', np.nan):.3f}  "
        f"PenRate={row.get('PenetrationRate', np.nan):.3f}"
    )

    with writer.saving(fig, str(out_path), dpi=140):
        for frame_idx in frames:
            ax.clear()
            if occ is not None:
                ax.imshow(
                    occ.T,
                    origin="lower",
                    extent=extent,
                    cmap="Greys",
                    alpha=0.28,
                    interpolation="nearest",
                )
                if sdf is not None:
                    ax.contour(
                        sdf.T,
                        levels=[0],
                        origin="lower",
                        extent=extent,
                        colors="#B22222",
                        linewidths=0.8,
                        alpha=0.65,
                    )

            if gt_root_xz is not None:
                ax.plot(
                    gt_root_xz[:, 0],
                    gt_root_xz[:, 1],
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.7,
                    label="target path",
                )
                ax.scatter(
                    gt_root_xz[0, 0],
                    gt_root_xz[0, 1],
                    c="black",
                    s=38,
                    marker="o",
                    label="target start",
                    zorder=7,
                )
                ax.scatter(
                    gt_root_xz[-1, 0],
                    gt_root_xz[-1, 1],
                    c="black",
                    s=82,
                    marker="*",
                    label="target end",
                    zorder=7,
                )

            ax.plot(
                gen_root[: frame_idx + 1, 0],
                gen_root[: frame_idx + 1, 2],
                color="#1F77B4",
                linewidth=2.0,
                label="generated root",
            )
            ax.scatter(gen_root[0, 0], gen_root[0, 2], c="#2CA02C", s=42, label="generated start", zorder=5)
            ax.scatter(gen_root[-1, 0], gen_root[-1, 2], c="#D62728", s=58, marker="x", label="generated end", zorder=5)

            joints = gen_joints[frame_idx]
            for parent, child in SMPLX_22_CONNECTIONS:
                if parent < joints.shape[0] and child < joints.shape[0]:
                    ax.plot(
                        [joints[parent, 0], joints[child, 0]],
                        [joints[parent, 2], joints[child, 2]],
                        color="#E15759",
                        linewidth=2.0,
                        alpha=0.9,
                    )
            ax.scatter(joints[:, 0], joints[:, 2], c="#E15759", s=12, zorder=6)

            ax.set_xlim(x0, x1)
            ax.set_ylim(z0, z1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            title = f"{row['experiment']} {row['sample_id']} | scene={scene_name} | frame {frame_idx + 1}/{gen_root.shape[0]}"
            ax.set_title(title, fontsize=10)
            ax.text(
                0.01,
                0.99,
                metric_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
            if text:
                ax.text(
                    0.01,
                    0.02,
                    f"{text[:82]}\nblack star=target end, red x=generated end",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
                )
            ax.legend(loc="upper right", fontsize=6, framealpha=0.75)
            writer.grab_frame()

    plt.close(fig)
    return out_path


def write_analysis(summary: pd.DataFrame, anomalies: pd.DataFrame, out_path: Path) -> None:
    best_path = summary.sort_values("PathADE").iloc[0]
    best_root_scene = summary.sort_values("NonWalkableRootRate").iloc[0]
    best_penalty = summary.sort_values("SceneSDFPenalty").iloc[0]
    worst_collision = summary.sort_values("CollisionFrameRate", ascending=False).iloc[0]

    lines = [
        "# Top-Down Video Metrics Report",
        "",
        "## Completed Models",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Preliminary Analysis",
        "",
        f"- Best path alignment by PathADE: {best_path['experiment']} ({best_path['PathADE']:.6f}).",
        f"- Lowest root non-walkable rate: {best_root_scene['experiment']} ({best_root_scene['NonWalkableRootRate']:.6f}).",
        f"- Lowest SceneSDFPenalty: {best_penalty['experiment']} ({best_penalty['SceneSDFPenalty']:.6f}).",
        f"- Highest body CollisionFrameRate: {worst_collision['experiment']} ({worst_collision['CollisionFrameRate']:.6f}).",
        "- E7 uses GT root, so its PathADE/PathFDE are expected to be near zero; body-scene collision can still be high.",
        "- The anomaly score ranks samples with high path error, collision rate, penetration rate, SDF penalty, and root jerk.",
        "",
        "## Selected Anomaly Samples",
        "",
        anomalies[
            [
                "experiment",
                "sample_id",
                "anomaly_score",
                "PathADE",
                "PathFDE",
                "CollisionFrameRate",
                "PenetrationRate",
                "SceneSDFPenalty",
                "RootJerk",
                "video_file",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/topdown_scene_video_report")
    parser.add_argument("--cache_dir", type=str, default="lingo_smplx_cache")
    parser.add_argument("--videos_per_model", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_videos", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    video_dir = out_dir / "videos"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = compute_all_metrics(args.cache_dir, max_samples=args.max_samples)
    if df.empty:
        raise RuntimeError("No samples found for completed experiments.")
    df = add_anomaly_scores(df)
    df.to_csv(out_dir / "all_sample_metrics.csv", index=False)

    summary = aggregate_model_metrics(df)
    summary.to_csv(out_dir / "model_metrics_summary.csv", index=False)
    plot_metric_summary(summary, fig_dir / "model_metrics_summary.png")

    selected = (
        df.sort_values("anomaly_score", ascending=False)
        .groupby("experiment", as_index=False, group_keys=False)
        .head(args.videos_per_model)
        .copy()
    )

    scene_index = build_scene_cache_index(args.cache_dir)
    video_files = []
    if not args.skip_videos:
        for _, row in selected.iterrows():
            print(f"[video] {row['experiment']} {row['sample_id']}")
            video_files.append(
                str(render_video(row, scene_index, video_dir, args.fps, args.frame_stride))
            )
    else:
        video_files = ["" for _ in range(len(selected))]

    selected["video_file"] = video_files
    selected.to_csv(out_dir / "anomaly_samples.csv", index=False)
    write_analysis(summary, selected, out_dir / "PRELIMINARY_ANALYSIS.md")

    payload = {
        "output_dir": str(out_dir),
        "num_samples": int(len(df)),
        "num_videos": int(len(video_files)),
        "experiments": list(EXPERIMENTS.keys()),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
