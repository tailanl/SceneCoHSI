"""Post-process generated root NPZ files with raw3d floor-filtered scene feasibility.

This is intended for E8/E9/E10-style experiments:
1. generate root files with the existing Energy/Classifier/Hybrid generator;
2. run this script to project target/root XZ onto raw-scene free space;
3. feed the corrected output directory to Stage2 external_root training/generation.

By default the script writes additional corrected keys and leaves the original
root untouched. Use --overwrite_root_keys when the output directory should be
consumed by Stage2.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(1, str(REPO_ROOT / "kimodo"))

from kimodo_sceneco.guidance.raw_scene_root import (  # noqa: E402
    DEFAULT_FLOOR_IGNORE_HEIGHT,
    RAW_SCENE_VOXEL_SIZE,
    build_raw_scene_2d,
    find_raw_scene_path,
    load_raw_scene,
    make_raw_scene_info,
    project_xz_to_free,
    root_feasibility,
    smooth_xz_path,
)

log = logging.getLogger(__name__)


def _scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def load_kimodo_motion_rep(checkpoint: str, device: torch.device):
    os.environ.setdefault("CHECKPOINT_DIR", str(PROJECT_DIR / "models"))
    from kimodo.model.load_model import load_model

    model = load_model(checkpoint, device=device)
    model.eval()
    return model.motion_rep


def normalize_root_5d(root_5d_meter: np.ndarray, motion_rep, device: torch.device) -> np.ndarray:
    """Normalize meter-space root 5D using Kimodo motion_rep root statistics."""
    root = torch.from_numpy(root_5d_meter).float().to(device)
    root_slice = motion_rep.root_slice
    if hasattr(motion_rep, "mean") and hasattr(motion_rep, "std"):
        mean = motion_rep.mean[..., root_slice].to(device=device, dtype=root.dtype).reshape(-1)
        std = motion_rep.std[..., root_slice].to(device=device, dtype=root.dtype).reshape(-1)
        return ((root - mean) / std.clamp_min(1e-8)).detach().cpu().numpy().astype(np.float32)

    if hasattr(motion_rep, "stats") and hasattr(motion_rep.stats, "mean"):
        mean = motion_rep.stats.mean[..., root_slice].to(device=device, dtype=root.dtype).reshape(-1)
        std = motion_rep.stats.std[..., root_slice].to(device=device, dtype=root.dtype).reshape(-1)
        eps = getattr(motion_rep.stats, "eps", 0.0)
        denom = torch.sqrt(std**2 + eps).clamp_min(1e-8)
        return ((root - mean) / denom).detach().cpu().numpy().astype(np.float32)

    raise RuntimeError("motion_rep does not expose mean/std root statistics")


def load_npz_dict(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def build_corrected_scene_name_map(cache_dir: Path) -> dict[str, str]:
    """Build source_id -> corrected scene_name for LINGO mirrored segments.

    The released LINGO metadata labels every mirrored segment as 005_mirror.
    The actual paired scene is the mirror of the corresponding non-mirrored
    segment in the valid cache ordering. This mirrors the correction used by
    LINGOSceneMotionDataset.
    """
    if not cache_dir.exists():
        return {}

    rows = []
    for f in sorted(cache_dir.glob("seg_*.npz")):
        data = np.load(str(f), allow_pickle=True)
        length = int(data["length"])
        if 40 <= length <= 196:
            rows.append(
                {
                    "stem": f.stem,
                    "scene_name": str(data.get("scene_name", "")),
                }
            )

    non_mirror = [row for row in rows if "_mirror" not in row["scene_name"]]
    mirror = [row for row in rows if "_mirror" in row["scene_name"]]

    corrected = {row["stem"]: row["scene_name"] for row in rows}
    for idx in range(min(len(non_mirror), len(mirror))):
        corrected[mirror[idx]["stem"]] = f"{non_mirror[idx]['scene_name']}_mirror"
    return corrected


def infer_source_id(npz_path: Path, data: dict) -> str:
    source_file = data.get("source_file")
    if source_file is not None:
        source = Path(_scalar_str(source_file))
        if source.stem:
            return source.stem
    return npz_path.stem


def pick_root_5d(data: dict, path: Path) -> np.ndarray:
    if "guided_root_5d_meter" in data:
        root = np.asarray(data["guided_root_5d_meter"], dtype=np.float32)
        if root.ndim == 2 and root.shape[1] >= 5:
            return root[:, :5].copy()
    if "gen_root" in data:
        gen_root = np.asarray(data["gen_root"], dtype=np.float32)
        if gen_root.ndim == 2 and gen_root.shape[1] >= 3:
            heading = np.zeros((gen_root.shape[0], 2), dtype=np.float32)
            heading[:, 0] = 1.0
            return np.concatenate([gen_root[:, :3], heading], axis=-1)
    raise ValueError(f"{path} has neither guided_root_5d_meter nor gen_root")


def copy_with_updates(data: dict, updates: dict, overwrite_root_keys: bool) -> dict:
    out = dict(data)
    out.update(updates)
    if overwrite_root_keys:
        out["guided_root_5d_meter"] = updates["corrected_root_5d_meter"]
        out["target_path_xz"] = updates.get("corrected_target_path_xz", data.get("target_path_xz", ""))
        if "corrected_guided_root_5d_norm" in updates:
            out["guided_root_5d_norm"] = updates["corrected_guided_root_5d_norm"]
    return out


def save_npz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **data)


def process_file(
    npz_path: Path,
    output_path: Path,
    scene_dir: Path,
    clearance_m: float,
    smooth_window: int,
    project_target_path: bool,
    overwrite_root_keys: bool,
    motion_rep,
    device: torch.device,
    scene_name_map: dict[str, str],
) -> dict:
    data = load_npz_dict(npz_path)
    source_id = infer_source_id(npz_path, data)
    original_scene_name = _scalar_str(data.get("scene_name", ""))
    scene_name = scene_name_map.get(source_id, original_scene_name)
    raw_scene, info = load_raw_scene(scene_dir, scene_name)
    if raw_scene is None or info is None:
        out = dict(data)
        out["raw3d_postprocess_status"] = np.asarray("missing_scene")
        out["original_scene_name"] = np.asarray(original_scene_name)
        out["scene_name"] = np.asarray(scene_name)
        save_npz(output_path, out)
        return {
            "file": npz_path.name,
            "source_id": source_id,
            "original_scene_name": original_scene_name,
            "scene_name": scene_name,
            "status": "missing_scene",
        }

    info = make_raw_scene_info(
        raw_scene,
        scene_name=scene_name,
        path=find_raw_scene_path(scene_dir, scene_name) or "",
        voxel_size=RAW_SCENE_VOXEL_SIZE,
        floor_ignore_height=DEFAULT_FLOOR_IGNORE_HEIGHT,
    )
    scene2d = build_raw_scene_2d(raw_scene, info)

    root_5d = pick_root_5d(data, npz_path)
    before = root_feasibility(root_5d[:, [0, 2]], scene2d, clearance_m=clearance_m)

    original_xz = root_5d[:, [0, 2]]
    corrected_xz, root_changed = project_xz_to_free(original_xz, scene2d, clearance_m=clearance_m)
    if root_changed.any():
        corrected_xz = smooth_xz_path(corrected_xz, window=smooth_window, keep_endpoints=True)
        # A second projection after smoothing guarantees no smoothed point moves back into an obstacle.
        corrected_xz, root_changed_after_smooth = project_xz_to_free(
            corrected_xz,
            scene2d,
            clearance_m=clearance_m,
        )
    else:
        root_changed_after_smooth = np.zeros_like(root_changed, dtype=bool)
    root_shifted = np.linalg.norm(corrected_xz - original_xz, axis=-1) > 1e-6

    corrected_root = root_5d.copy()
    corrected_root[:, 0] = corrected_xz[:, 0]
    corrected_root[:, 2] = corrected_xz[:, 1]
    after = root_feasibility(corrected_root[:, [0, 2]], scene2d, clearance_m=clearance_m)

    updates = {
        "corrected_root_5d_meter": corrected_root.astype(np.float32),
        "raw3d_root_changed_mask": (root_changed | root_changed_after_smooth | root_shifted).astype(np.bool_),
        "raw3d_postprocess_status": np.asarray("ok"),
        "raw3d_clearance_m": np.asarray(float(clearance_m), dtype=np.float32),
        "raw3d_floor_ignore_height": np.asarray(float(DEFAULT_FLOOR_IGNORE_HEIGHT), dtype=np.float32),
        "original_scene_name": np.asarray(original_scene_name),
        "scene_name": np.asarray(scene_name),
    }

    if motion_rep is not None:
        updates["corrected_guided_root_5d_norm"] = normalize_root_5d(
            corrected_root.astype(np.float32),
            motion_rep,
            device,
        )

    target_before = {}
    target_after = {}
    if project_target_path and "target_path_xz" in data:
        target = np.asarray(data["target_path_xz"], dtype=np.float32)
        if target.ndim == 2 and target.shape[1] >= 2:
            target_before = root_feasibility(target[:, :2], scene2d, clearance_m=clearance_m)
            corrected_target, target_changed = project_xz_to_free(
                target[:, :2],
                scene2d,
                clearance_m=clearance_m,
            )
            target_after = root_feasibility(corrected_target[:, :2], scene2d, clearance_m=clearance_m)
            updates["corrected_target_path_xz"] = corrected_target.astype(np.float32)
            updates["raw3d_target_changed_mask"] = target_changed.astype(np.bool_)

    if overwrite_root_keys and "guided_root_5d_norm" in data and motion_rep is None:
        raise RuntimeError(
            "--overwrite_root_keys on files with guided_root_5d_norm requires --update_norm"
        )

    out = copy_with_updates(data, updates, overwrite_root_keys=overwrite_root_keys)
    save_npz(output_path, out)

    changed_mask = root_changed | root_changed_after_smooth | root_shifted
    max_shift = float(np.linalg.norm(corrected_root[:, [0, 2]] - root_5d[:, [0, 2]], axis=-1).max())
    return {
        "file": npz_path.name,
        "source_id": source_id,
        "original_scene_name": original_scene_name,
        "scene_name": scene_name,
        "scene_name_changed": scene_name != original_scene_name,
        "status": "ok",
        "root_invalid_before": before["invalid_rate"],
        "root_invalid_after": after["invalid_rate"],
        "root_occupied_before": before["occupied_rate"],
        "root_occupied_after": after["occupied_rate"],
        "root_out_of_bounds_before": before["out_of_bounds_rate"],
        "root_out_of_bounds_after": after["out_of_bounds_rate"],
        "root_min_clearance_before": before["min_clearance_m"],
        "root_min_clearance_after": after["min_clearance_m"],
        "root_changed_frames": int(changed_mask.sum()),
        "root_max_shift_m": max_shift,
        "target_invalid_before": target_before.get("invalid_rate", ""),
        "target_invalid_after": target_after.get("invalid_rate", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--scene_dir",
        default=str(PROJECT_DIR / "LINGO" / "dataset" / "dataset" / "Scene"),
    )
    parser.add_argument("--clearance_m", type=float, default=0.04)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--max_files", type=int, default=-1)
    parser.add_argument("--project_target_path", action="store_true")
    parser.add_argument(
        "--overwrite_root_keys",
        action="store_true",
        help="Replace guided_root_5d_meter/target_path_xz so output can feed Stage2.",
    )
    parser.add_argument(
        "--update_norm",
        action="store_true",
        help="Load Kimodo and update guided_root_5d_norm after correction.",
    )
    parser.add_argument("--kimodo_checkpoint", default="Kimodo-SMPLX-RP-v1")
    parser.add_argument("--cache_dir", default=str(PROJECT_DIR / "lingo_smplx_cache"))
    parser.add_argument(
        "--no_fix_mirror_scene_names",
        action="store_true",
        help="Trust scene_name stored in root npz files instead of correcting LINGO mirrors.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    if args.max_files >= 0:
        files = files[:args.max_files]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    motion_rep = None
    if args.update_norm:
        log.info("Loading Kimodo motion_rep for root normalization")
        motion_rep = load_kimodo_motion_rep(args.kimodo_checkpoint, device=device)

    scene_name_map = {}
    if not args.no_fix_mirror_scene_names:
        scene_name_map = build_corrected_scene_name_map(Path(args.cache_dir))
        changed = sum(
            1
            for stem, scene_name in scene_name_map.items()
            if scene_name.endswith("_mirror") and scene_name != "005_mirror"
        )
        log.info(
            "Loaded corrected scene-name map for %d cache files (%d non-005 mirrors)",
            len(scene_name_map),
            changed,
        )

    rows = []
    for idx, npz_path in enumerate(files, start=1):
        output_path = output_dir / npz_path.name
        row = process_file(
            npz_path=npz_path,
            output_path=output_path,
            scene_dir=Path(args.scene_dir),
            clearance_m=args.clearance_m,
            smooth_window=args.smooth_window,
            project_target_path=args.project_target_path,
            overwrite_root_keys=args.overwrite_root_keys,
            motion_rep=motion_rep,
            device=device,
            scene_name_map=scene_name_map,
        )
        rows.append(row)
        if idx % 100 == 0:
            log.info("Processed %d/%d", idx, len(files))

    summary_path = output_dir / "raw3d_root_postprocess_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "source_id",
        "original_scene_name",
        "scene_name",
        "scene_name_changed",
        "status",
        "root_invalid_before",
        "root_invalid_after",
        "root_occupied_before",
        "root_occupied_after",
        "root_out_of_bounds_before",
        "root_out_of_bounds_after",
        "root_min_clearance_before",
        "root_min_clearance_after",
        "root_changed_frames",
        "root_max_shift_m",
        "target_invalid_before",
        "target_invalid_after",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        before = np.mean([float(row["root_invalid_before"]) for row in ok_rows])
        after = np.mean([float(row["root_invalid_after"]) for row in ok_rows])
        changed = np.mean([float(row["root_changed_frames"]) for row in ok_rows])
        log.info("Root invalid rate: %.6f -> %.6f", before, after)
        log.info("Mean changed frames: %.2f", changed)
    log.info("Saved corrected roots to %s", output_dir)
    log.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
