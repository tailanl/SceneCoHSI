#!/usr/bin/env python
"""Evaluate generated NPZ files from the unified retrain_mirrorfix50 registry.

This is intended as test-code scaffolding. It can run now on no-train baselines
E1-E3, and later on trained experiments after their generated body NPZ files
are written under the same run root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from eval.eval_path_metrics import compute_path_metrics
from eval.eval_sceneadapt_metrics import compute_lingo_scene_metrics


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def find_npz_files(body_dir: Path) -> list[Path]:
    files = sorted(body_dir.glob("sample_*.npz"))
    if not files:
        files = sorted(body_dir.glob("seg_*.npz"))
    if not files:
        files = sorted(body_dir.glob("*.npz"))
    return files


def resolve_body_dir(run_root: Path, row: dict) -> Path:
    run_body = Path(row["run_body_dir_abs"]) if "run_body_dir_abs" in row else run_root / row["run_body_dir"]
    if run_body.exists():
        return run_body
    if "source_body_dir_abs" in row:
        return Path(row["source_body_dir_abs"])
    return run_body


def evaluate_registry(
    registry_path: Path,
    run_root: Path,
    output_dir: Path,
    include_ids: set[str] | None,
    families: set[str] | None,
    max_samples: int | None,
    scene_dir: str,
    skip_scene: bool,
) -> pd.DataFrame:
    rows = json.loads(registry_path.read_text(encoding="utf-8"))
    metric_rows = []

    for exp in rows:
        if include_ids and exp["id"] not in include_ids:
            continue
        if families and exp["family"] not in families:
            continue

        body_dir = resolve_body_dir(run_root, exp)
        files = find_npz_files(body_dir)
        if max_samples is not None:
            files = files[:max_samples]
        print(f"[eval] {exp['id']}: {len(files)} files from {body_dir}")

        for path in files:
            data = np.load(path, allow_pickle=True)
            if "gen_root" not in data.files or "gt_root_xz" not in data.files:
                continue

            gen_root = np.asarray(data["gen_root"], dtype=np.float32)
            gt_root_xz = np.asarray(data["gt_root_xz"], dtype=np.float32)
            path_metrics = compute_path_metrics(gen_root, gt_root_xz)

            scene_metrics = {}
            if not skip_scene and "gen_joints" in data.files:
                scene_name = scalar_str(data.get("scene_name", ""))
                gen_joints = np.asarray(data["gen_joints"], dtype=np.float32)
                scene_metrics = compute_lingo_scene_metrics(
                    gen_root,
                    gen_joints,
                    scene_name,
                    scene_dir=scene_dir,
                )

            metric_rows.append(
                {
                    "experiment": exp["id"],
                    "family": exp["family"],
                    "root_method": exp["root_method"],
                    "body_method": exp["body_method"],
                    "sample_id": path.stem,
                    "body_file": str(path),
                    "scene_name": scalar_str(data.get("scene_name", "")),
                    "text": scalar_str(data.get("text", "")),
                    "num_frames": int(gen_root.shape[0]),
                    **path_metrics,
                    **scene_metrics,
                }
            )

    df = pd.DataFrame(metric_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_sample_metrics.csv", index=False)

    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep = [
            col
            for col in numeric_cols
            if col not in {"num_frames"} and not col.endswith("_rank")
        ]
        summary = (
            df.groupby(["experiment", "family", "root_method", "body_method"], as_index=False)[keep]
            .mean(numeric_only=True)
            .sort_values("experiment")
        )
        counts = df.groupby("experiment").size().rename("samples").reset_index()
        summary = summary.merge(counts, on="experiment", how="left")
        summary.to_csv(output_dir / "summary_metrics.csv", index=False)
        (output_dir / "summary_metrics.md").write_text(
            summary.to_markdown(index=False, floatfmt=".6f"),
            encoding="utf-8",
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="outputs/retrain_mirrorfix50")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=["no_train_baseline"])
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--scene_dir", default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--skip_scene", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    registry = Path(args.registry) if args.registry else run_root / "eval_viz" / "experiment_registry.json"
    if not registry.is_absolute():
        registry = PROJECT_ROOT / registry
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "eval_viz" / "test_smoke"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    df = evaluate_registry(
        registry_path=registry,
        run_root=run_root,
        output_dir=output_dir,
        include_ids=set(args.include) if args.include else None,
        families=set(args.families) if args.families else None,
        max_samples=args.max_samples,
        scene_dir=args.scene_dir,
        skip_scene=args.skip_scene,
    )

    print(json.dumps({"output_dir": str(output_dir), "rows": int(len(df))}, indent=2))


if __name__ == "__main__":
    main()
