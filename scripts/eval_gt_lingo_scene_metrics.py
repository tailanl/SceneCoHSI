"""Evaluate GT LINGO joints against raw LINGO scene occupancy."""

from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fix_lingo_mirror_data import build_corrected_map


def load_scene(scene_dir: Path, scene_name: str, cache: dict[str, np.ndarray]) -> tuple[np.ndarray | None, str]:
    candidates = [scene_name]
    base_name = scene_name.split("-")[0]
    no_mirror = scene_name.replace("_mirror", "")
    for candidate in (base_name, no_mirror):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        path = scene_dir / f"{candidate}.npy"
        if path.exists():
            key = str(path)
            if key not in cache:
                cache[key] = np.load(str(path))
            return cache[key], path.name
    return None, ""


def scene_grid(shape: tuple[int, int, int]) -> tuple[float, float, float, float, float, float]:
    nx, ny, nz = shape
    if shape == (300, 100, 400):
        return -3.0, 0.0, -4.0, 3.0, 2.0, 4.0
    if shape == (400, 100, 600):
        return -4.0, 0.0, -6.0, 4.0, 2.0, 6.0
    vs = 0.02
    return -nx * vs / 2.0, 0.0, -nz * vs / 2.0, nx * vs / 2.0, ny * vs, nz * vs / 2.0


def compute_metrics(joints: np.ndarray, scene: np.ndarray, floor_ignore_height: float) -> dict:
    nx, ny, nz = scene.shape
    x_min, y_min, z_min, x_max, y_max, z_max = scene_grid(scene.shape)
    vs_x = (x_max - x_min) / nx
    vs_y = (y_max - y_min) / ny
    vs_z = (z_max - z_min) / nz

    frames, num_joints, _ = joints.shape
    collision_frames = np.zeros(frames, dtype=bool)
    penetrating = 0
    denom = 0
    ignored_floor = 0
    out_of_bounds = 0

    for t in range(frames):
        for j in range(num_joints):
            x, y, z = (float(v) for v in joints[t, j])
            if y < floor_ignore_height:
                ignored_floor += 1
                continue
            denom += 1
            ix = int((x - x_min) / vs_x)
            iy = int((y - y_min) / vs_y)
            iz = int((z - z_min) / vs_z)
            in_bounds = 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz
            if not in_bounds:
                collision_frames[t] = True
                out_of_bounds += 1
            elif bool(scene[ix, iy, iz]):
                collision_frames[t] = True
                penetrating += 1

    total = frames * num_joints
    return {
        "CollisionFrameRate": float(collision_frames.mean()),
        "PenetrationRate": float(penetrating / denom) if denom else float("nan"),
        "PenetrationMean": 0.02 if penetrating else 0.0,
        "PenetrationMax": 0.02 if penetrating else 0.0,
        "OutOfSceneOrFloorIgnoredJointRate": float((out_of_bounds + ignored_floor) / total),
        "IgnoredFloorJointRate": float(ignored_floor / total),
        "OutOfBoundsJointRate": float(out_of_bounds / total),
        "PenetratingJointCount": int(penetrating),
        "DenomJointCount": int(denom),
        "FrameCount": int(frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GT LINGO scene metrics.")
    parser.add_argument("--dataset_dir", type=Path, default=Path("LINGO/dataset/dataset"))
    parser.add_argument("--cache_dir", type=Path, default=Path("lingo_smplx_cache"))
    parser.add_argument("--scene_dir", type=Path, default=Path("LINGO/dataset/dataset/Scene"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/gt_scene_metrics"))
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--floor_ignore_height", type=float, default=0.08)
    args = parser.parse_args()

    start = np.load(str(args.dataset_dir / "start_idx.npy")).flatten()
    end = np.load(str(args.dataset_dir / "end_idx.npy")).flatten()
    joints_all = np.load(str(args.dataset_dir / "human_joints_aligned.npy"), mmap_mode="r")
    with open(args.dataset_dir / "scene_name.pkl", "rb") as f:
        scene_per_frame = pickle.load(f)

    segments = []
    for idx, (s_raw, e_raw) in enumerate(zip(start, end)):
        s = int(s_raw)
        e = int(e_raw)
        length = e - s
        if length < args.min_frames or length > args.max_frames or s >= len(scene_per_frame):
            continue
        segments.append(
            {
                "raw_idx": idx,
                "compact_idx": len(segments),
                "start": s,
                "end": e,
                "length": length,
            }
        )

    indices = list(range(len(segments)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    if args.split == "train":
        selected = indices[:n_train]
    elif args.split == "val":
        selected = indices[n_train:]
    else:
        selected = indices

    corrected_map = build_corrected_map(args.cache_dir, args.min_frames, args.max_frames)
    scene_cache = {}
    rows = []
    for local_idx in selected:
        seg = segments[local_idx]
        sample_id = f"seg_{seg['compact_idx']:05d}"
        scene_name = corrected_map.get(sample_id)
        if scene_name is None:
            continue
        scene, loaded_scene = load_scene(args.scene_dir, scene_name, scene_cache)
        if scene is None:
            continue
        joints = np.asarray(joints_all[seg["start"]:seg["end"]], dtype=np.float32)
        metrics = compute_metrics(joints, scene, args.floor_ignore_height)
        row = {
            "sample_id": sample_id,
            "compact_idx": seg["compact_idx"],
            "raw_idx": seg["raw_idx"],
            "scene_name": scene_name,
            "loaded_scene": loaded_scene,
            "start": seg["start"],
            "end": seg["end"],
            "length": seg["length"],
        }
        row.update(metrics)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / f"gt_{args.split}_scene_metrics_corrected.csv"
    with open(metrics_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"split": args.split, "Samples": len(rows), "CSV": str(metrics_csv)}
    for key in [
        "CollisionFrameRate",
        "PenetrationRate",
        "PenetrationMean",
        "PenetrationMax",
        "OutOfSceneOrFloorIgnoredJointRate",
        "IgnoredFloorJointRate",
        "OutOfBoundsJointRate",
    ]:
        summary[key] = float(np.nanmean([row[key] for row in rows]))
    summary["PooledPenetrationRate"] = float(
        sum(row["PenetratingJointCount"] for row in rows)
        / sum(row["DenomJointCount"] for row in rows)
    )

    summary_csv = args.output_dir / f"gt_{args.split}_scene_metrics_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
