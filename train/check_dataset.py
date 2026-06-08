#!/usr/bin/env python3
"""Dataset Format Diagnostic Checker.

Verifies that the LINGO-SOMA dataset conforms to expected format:
  - Scene voxel files: shape, value range, occupancy ratio
  - Motion files: joint count, frame count, NaN/Inf checks
  - Motion-scene pairing: coordinate alignment sanity

Usage:
    python check_dataset.py \
        --scene_dir LINGO/dataset/dataset/Scene \
        --motion_dir soma_converted_all/lingo \
        --output_dir kimodo_scene_project/outputs/reports
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def check_scene_file(path: Path) -> Dict:
    result = {
        "path": str(path),
        "valid": True,
        "issues": [],
    }
    try:
        data = np.load(path, allow_pickle=True)
        if not isinstance(data, np.ndarray):
            result["issues"].append(f"not a numpy array, got {type(data)}")
            result["valid"] = False
            return result

        result["shape"] = list(data.shape)
        result["dtype"] = str(data.dtype)

        if len(data.shape) != 3:
            result["issues"].append(f"expected 3D, got {data.shape}")
            result["valid"] = False

        if np.any(np.isnan(data)):
            result["issues"].append("contains NaN")
            result["valid"] = False
        if np.any(np.isinf(data)):
            result["issues"].append("contains Inf")
            result["valid"] = False

        occ_ratio = float(data.mean())
        result["occupancy_ratio"] = round(occ_ratio, 6)

        if occ_ratio == 0:
            result["issues"].append("all zeros (empty scene)")
        elif occ_ratio == 1.0:
            result["issues"].append("all ones (fully occupied)")

        min_v = float(data.min())
        max_v = float(data.max())
        result["value_range"] = [min_v, max_v]

        if not set(data.flatten()[:1000]).issubset({0, 1, 0.0, 1.0, True, False}):
            result["issues"].append("non-binary values detected")

    except Exception as e:
        result["issues"].append(str(e))
        result["valid"] = False

    return result


def check_motion_file(path: Path) -> Dict:
    result = {
        "path": str(path),
        "valid": True,
        "issues": [],
    }
    try:
        data = dict(np.load(path, allow_pickle=True))
        result["keys"] = sorted(data.keys())

        expected_key_groups = [
            ["posed_joints", "soma77_joints", "smpl22_joints"],
            ["root_positions", "soma_root_transl"],
        ]
        for group in expected_key_groups:
            found = any(k in data for k in group)
            if not found:
                result["issues"].append(f"missing motion key group: {group}")

        if "posed_joints" in data:
            pj = data["posed_joints"]
            if isinstance(pj, np.ndarray):
                result["posed_joints_shape"] = list(pj.shape)
                result["posed_joints_dtype"] = str(pj.dtype)
                if np.any(np.isnan(pj)):
                    result["issues"].append("posed_joints contains NaN")
                    result["valid"] = False
                if np.any(np.isinf(pj)):
                    result["issues"].append("posed_joints contains Inf")
                    result["valid"] = False

        if "root_positions" in data:
            rp = data["root_positions"]
            if isinstance(rp, np.ndarray):
                result["root_positions_shape"] = list(rp.shape)
                if np.any(np.isnan(rp)):
                    result["issues"].append("root_positions contains NaN")
                    result["valid"] = False

        for key in data:
            val = data[key]
            if isinstance(val, np.ndarray):
                is_floating = np.issubdtype(val.dtype, np.floating)
                result.setdefault("arrays", {})[key] = {
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                    "min": float(val.min()) if is_floating else None,
                    "max": float(val.max()) if is_floating else None,
                    "mean": float(val.mean()) if is_floating else None,
                    "has_nan": bool(np.any(np.isnan(val))) if is_floating else False,
                    "has_inf": bool(np.any(np.isinf(val))) if is_floating else False,
                }
                if is_floating:
                    if np.any(np.isnan(val)):
                        result["issues"].append(f"{key} contains NaN")
                        result["valid"] = False
                    if np.any(np.isinf(val)):
                        result["issues"].append(f"{key} contains Inf")
                        result["valid"] = False

    except Exception as e:
        result["issues"].append(str(e))
        result["valid"] = False

    return result


def check_motion_coord_range(motion_files: List[Path]) -> Dict:
    joint_ranges = []
    root_ranges = []
    for f in motion_files:
        try:
            data = dict(np.load(f, allow_pickle=True))
            pj = data.get("posed_joints") or data.get("soma77_joints") or data.get("smpl22_joints")
            rp = data.get("root_positions") or data.get("soma_root_transl")

            if pj is not None and isinstance(pj, np.ndarray):
                if pj.ndim == 4:
                    pj = pj[0]
                if pj.ndim >= 3:
                    joint_ranges.append((float(pj.min()), float(pj.max())))

            if rp is not None and isinstance(rp, np.ndarray):
                if rp.ndim == 3:
                    rp = rp[0]
                if rp.ndim >= 2:
                    root_ranges.append((
                        float(rp[:, 0].min()), float(rp[:, 0].max()),
                        float(rp[:, 1].min()), float(rp[:, 1].max()),
                        float(rp[:, 2].min()) if rp.shape[1] > 2 else 0.0,
                        float(rp[:, 2].max()) if rp.shape[1] > 2 else 0.0,
                    ))
        except Exception:
            pass

    if not joint_ranges:
        return {"checked": False, "reason": "no valid joint data"}

    return {
        "checked": True,
        "num_files_checked": len(joint_ranges),
        "joint_global_min": float(np.min([jr[0] for jr in joint_ranges])),
        "joint_global_max": float(np.max([jr[1] for jr in joint_ranges])),
        "root_x_range": [float(np.min([rr[0] for rr in root_ranges])), float(np.max([rr[1] for rr in root_ranges]))] if root_ranges else None,
        "root_y_range": [float(np.min([rr[2] for rr in root_ranges])), float(np.max([rr[3] for rr in root_ranges]))] if root_ranges else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset format diagnostic checker")
    parser.add_argument("--scene_dir", type=str,
                        default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--motion_dir", type=str,
                        default="soma_converted_all/lingo")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/reports")
    parser.add_argument("--max_scenes", type=int, default=10)
    parser.add_argument("--max_motions", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "scene_dir": args.scene_dir,
            "motion_dir": args.motion_dir,
        },
        "scenes": {},
        "motions": {},
        "summary": {},
    }

    scene_dir = Path(args.scene_dir)
    if scene_dir.exists():
        scene_files = sorted(scene_dir.glob("*.npy"))[:args.max_scenes]
        print(f"Checking {len(scene_files)} scene files...")
        for sf in scene_files:
            report["scenes"][sf.name] = check_scene_file(sf)

    motion_dir = Path(args.motion_dir)
    if motion_dir.exists():
        motion_files = sorted(motion_dir.glob("*.npz"))[:args.max_motions]
        print(f"Checking {len(motion_files)} motion files...")
        for mf in motion_files:
            report["motions"][mf.name] = check_motion_file(mf)

        if motion_files:
            print("Checking motion coordinate ranges...")
            all_motions = sorted(motion_dir.glob("*.npz"))[:50]
            report["coordinate_ranges"] = check_motion_coord_range(all_motions)

    scene_valid = sum(1 for s in report["scenes"].values() if s.get("valid"))
    scene_total = len(report["scenes"])
    motion_valid = sum(1 for m in report["motions"].values() if m.get("valid"))
    motion_total = len(report["motions"])

    scene_shape_counts = {}
    for s in report["scenes"].values():
        sh = tuple(s.get("shape", []))
        scene_shape_counts[str(sh)] = scene_shape_counts.get(str(sh), 0) + 1

    report["summary"] = {
        "scenes_checked": scene_total,
        "scenes_valid": scene_valid,
        "scene_pass_rate": f"{scene_valid}/{scene_total}",
        "scene_shape_distribution": scene_shape_counts,
        "motions_checked": motion_total,
        "motions_valid": motion_valid,
        "motion_pass_rate": f"{motion_valid}/{motion_total}",
    }

    with open(out_dir / "dataset_check.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== Dataset Check Summary ===")
    print(f"Scenes:  {scene_valid}/{scene_total} valid")
    print(f"Motions: {motion_valid}/{motion_total} valid")
    print(f"Scene shapes: {scene_shape_counts}")
    print(f"\nReport saved to {out_dir / 'dataset_check.json'}")


if __name__ == "__main__":
    main()
