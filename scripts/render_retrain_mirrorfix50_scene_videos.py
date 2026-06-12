#!/usr/bin/env python
"""Render in-scene action videos from the unified retrain_mirrorfix50 registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SCENE_VOXEL_SIZE = 0.02
FLOOR_IGNORE_HEIGHT = 0.08

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


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def scene_grid(shape: tuple[int, int, int]) -> tuple[float, float, float, float]:
    if tuple(shape) == (300, 100, 400):
        return -3.0, 3.0, -4.0, 4.0
    if tuple(shape) == (400, 100, 600):
        return -4.0, 4.0, -6.0, 6.0
    x = shape[0] * RAW_SCENE_VOXEL_SIZE / 2.0
    z = shape[2] * RAW_SCENE_VOXEL_SIZE / 2.0
    return -x, x, -z, z


def load_scene(scene_dir: Path, scene_name: str) -> tuple[np.ndarray | None, str]:
    names = [scene_name]
    if scene_name.endswith("_mirror"):
        names.append(scene_name.removesuffix("_mirror"))
    if "-" in scene_name:
        names.append(scene_name.split("-")[0])
    for name in names:
        path = scene_dir / f"{name}.npy"
        if path.exists():
            return np.load(path), name
    return None, ""


def scene_occupancy_xz(scene: np.ndarray) -> tuple[np.ndarray, list[float]]:
    x_min, x_max, z_min, z_max = scene_grid(tuple(scene.shape))
    y0 = max(1, int(FLOOR_IGNORE_HEIGHT / RAW_SCENE_VOXEL_SIZE))
    y1 = min(scene.shape[1], int(2.0 / RAW_SCENE_VOXEL_SIZE))
    occ_xz = scene[:, y0:y1, :].any(axis=1)
    return occ_xz, [x_min, x_max, z_min, z_max]


def find_npz_files(body_dir: Path) -> list[Path]:
    files = sorted(body_dir.glob("sample_*.npz"))
    if not files:
        files = sorted(body_dir.glob("seg_*.npz"))
    if not files:
        files = sorted(body_dir.glob("*.npz"))
    return files


def limits_from_motion(gen_root: np.ndarray, gen_joints: np.ndarray, gt_root_xz, extent) -> tuple[float, float, float, float]:
    xs = [gen_root[:, 0], gen_joints[..., 0].reshape(-1)]
    zs = [gen_root[:, 2], gen_joints[..., 2].reshape(-1)]
    if gt_root_xz is not None:
        xs.append(gt_root_xz[:, 0])
        zs.append(gt_root_xz[:, 1])
    if extent is not None:
        xs.append(np.asarray([extent[0], extent[1]], dtype=np.float32))
        zs.append(np.asarray([extent[2], extent[3]], dtype=np.float32))
    x_all = np.concatenate(xs)
    z_all = np.concatenate(zs)
    margin = max(0.4, 0.08 * max(float(np.ptp(x_all)), float(np.ptp(z_all))))
    return (
        float(x_all.min() - margin),
        float(x_all.max() + margin),
        float(z_all.min() - margin),
        float(z_all.max() + margin),
    )


def render_video(
    npz_path: Path,
    exp_id: str,
    out_path: Path,
    scene_dir: Path,
    fps: int,
    frame_stride: int,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    gen_root = np.asarray(data["gen_root"], dtype=np.float32)
    gen_joints = np.asarray(data["gen_joints"], dtype=np.float32)
    gt_root_xz = np.asarray(data["gt_root_xz"], dtype=np.float32) if "gt_root_xz" in data.files else None
    scene_name = scalar_str(data.get("scene_name", ""))
    text = scalar_str(data.get("text", ""))

    scene, loaded_scene = load_scene(scene_dir, scene_name)
    occ_xz = None
    extent = None
    if scene is not None:
        occ_xz, extent = scene_occupancy_xz(scene)
    x0, x1, z0, z1 = limits_from_motion(gen_root, gen_joints, gt_root_xz, extent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(range(0, gen_root.shape[0], max(1, frame_stride)))
    if frames[-1] != gen_root.shape[0] - 1:
        frames.append(gen_root.shape[0] - 1)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2200)

    with writer.saving(fig, str(out_path), dpi=150):
        for frame_idx in frames:
            ax.clear()
            if occ_xz is not None:
                ax.imshow(
                    occ_xz.T,
                    origin="lower",
                    extent=extent,
                    cmap="Greys",
                    alpha=0.30,
                    interpolation="nearest",
                )

            if gt_root_xz is not None:
                ax.plot(gt_root_xz[:, 0], gt_root_xz[:, 1], "k--", linewidth=1.2, alpha=0.75, label="target path")

            ax.plot(
                gen_root[: frame_idx + 1, 0],
                gen_root[: frame_idx + 1, 2],
                color="#1F77B4",
                linewidth=2.0,
                label="generated root",
            )
            ax.scatter(gen_root[0, 0], gen_root[0, 2], c="#2CA02C", s=34, zorder=5, label="start")
            ax.scatter(gen_root[-1, 0], gen_root[-1, 2], c="#D62728", s=48, marker="x", zorder=5, label="end")

            joints = gen_joints[frame_idx]
            for parent, child in SMPLX_22_CONNECTIONS:
                if parent < joints.shape[0] and child < joints.shape[0]:
                    ax.plot(
                        [joints[parent, 0], joints[child, 0]],
                        [joints[parent, 2], joints[child, 2]],
                        color="#E15759",
                        linewidth=2.0,
                        alpha=0.95,
                    )
            ax.scatter(joints[:, 0], joints[:, 2], c="#E15759", s=12, zorder=6)

            ax.set_xlim(x0, x1)
            ax.set_ylim(z0, z1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.18)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_title(
                f"{exp_id} | {npz_path.stem} | scene={scene_name}"
                f"{' -> ' + loaded_scene if loaded_scene and loaded_scene != scene_name else ''}"
                f" | frame {frame_idx + 1}/{gen_root.shape[0]}",
                fontsize=9,
            )
            if text:
                ax.text(
                    0.01,
                    0.02,
                    text[:90],
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none"},
                )
            ax.legend(loc="upper right", fontsize=7, framealpha=0.78)
            writer.grab_frame()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="outputs/retrain_mirrorfix50")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--eval_root", default=None)
    parser.add_argument("--include", nargs="+", default=["E1", "E2", "E3"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--videos_per_exp", type=int, default=1)
    parser.add_argument("--scene_dir", default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame_stride", type=int, default=2)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    registry_path = Path(args.registry) if args.registry else run_root / "eval_viz" / "experiment_registry.json"
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    scene_dir = Path(args.scene_dir)
    if not scene_dir.is_absolute():
        scene_dir = PROJECT_ROOT / scene_dir
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "eval_viz" / "videos" / "scene_actions"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = [row for row in registry if row["id"] in set(args.include)]
    eval_root = Path(args.eval_root) if args.eval_root else None
    if eval_root is not None and not eval_root.is_absolute():
        eval_root = PROJECT_ROOT / eval_root
    outputs = []

    for row in selected:
        if eval_root is not None:
            body_dir = eval_root / row["id"] / "pred"
        else:
            body_dir = Path(row.get("run_body_dir_abs", run_root / row["run_body_dir"]))
        files = find_npz_files(body_dir)
        if not files:
            print(f"[skip] {row['id']}: no npz in {body_dir}")
            continue
        start = min(args.sample_idx, max(0, len(files) - 1))
        for npz_path in files[start : start + args.videos_per_exp]:
            out_path = output_dir / row["id"] / f"{npz_path.stem}_scene_action.mp4"
            print(f"[render] {row['id']} {npz_path} -> {out_path}")
            render_video(npz_path, row["id"], out_path, scene_dir, args.fps, args.frame_stride)
            outputs.append(str(out_path))

    summary = {"output_dir": str(output_dir), "videos": outputs}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
