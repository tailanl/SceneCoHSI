"""Fix LINGO mirrored scene names in cached and derived NPZ data.

The released LINGO metadata labels mirrored segments as ``005_mirror`` even
when the mirrored segment belongs to a different scene.  This script pairs
mirrored segments with their non-mirrored counterpart in the valid cache order,
rewrites ``scene_name``, and refreshes ``voxel_grid`` when present.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom as scipy_zoom


def load_npz_dict(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def downsample_voxel(voxel: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    voxel = np.asarray(voxel)
    if tuple(voxel.shape) == tuple(target_shape):
        return voxel.astype(np.float32)
    zoom_factors = tuple(t / s for t, s in zip(target_shape, voxel.shape))
    resized = scipy_zoom(voxel, zoom_factors, order=0) > 0.5
    return resized.astype(np.float32)


def load_scene_voxel(
    scene_dir: Path,
    scene_name: str,
    target_shape: tuple[int, int, int],
    scene_cache: dict[tuple[str, tuple[int, int, int]], np.ndarray],
) -> np.ndarray | None:
    cache_key = (scene_name, target_shape)
    if cache_key in scene_cache:
        return scene_cache[cache_key]

    candidates = [scene_name]
    base_name = scene_name.split("-")[0]
    no_mirror = scene_name.replace("_mirror", "")
    for candidate in (base_name, no_mirror):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        path = scene_dir / f"{candidate}.npy"
        if path.exists():
            voxel = downsample_voxel(np.load(str(path)), target_shape)
            scene_cache[cache_key] = voxel
            return voxel
    return None


def build_corrected_map(cache_dir: Path, min_frames: int, max_frames: int) -> dict[str, str]:
    rows = []
    for path in sorted(cache_dir.glob("seg_*.npz")):
        data = np.load(str(path), allow_pickle=True)
        if "length" not in data or "scene_name" not in data:
            continue
        length = int(data["length"])
        if min_frames <= length <= max_frames:
            rows.append({"stem": path.stem, "scene_name": scalar_str(data["scene_name"])})

    non_mirror = [row for row in rows if "_mirror" not in row["scene_name"]]
    mirror = [row for row in rows if "_mirror" in row["scene_name"]]

    corrected = {row["stem"]: row["scene_name"] for row in rows}
    for idx in range(min(len(non_mirror), len(mirror))):
        corrected[mirror[idx]["stem"]] = f"{non_mirror[idx]['scene_name']}_mirror"
    return corrected


def infer_source_stem(path: Path, data: dict) -> str:
    source = data.get("source_file")
    if source is not None:
        stem = Path(scalar_str(source)).stem
        if stem:
            return stem
    return path.stem


def repair_dir(npz_dir: Path, scene_dir: Path, corrected_map: dict[str, str], dry_run: bool) -> list[dict]:
    rows = []
    scene_cache: dict[tuple[str, tuple[int, int, int]], np.ndarray] = {}
    for path in sorted(npz_dir.glob("*.npz")):
        meta = np.load(str(path), allow_pickle=True)
        if "scene_name" not in meta.files:
            continue
        source_stem = path.stem
        if "source_file" in meta.files:
            source = Path(scalar_str(meta["source_file"]))
            if source.stem:
                source_stem = source.stem
        if source_stem not in corrected_map:
            continue

        old_scene = scalar_str(meta["scene_name"])
        new_scene = corrected_map[source_stem]
        changed_scene = old_scene != new_scene
        has_voxel = "voxel_grid" in meta.files
        changed_voxel = bool(has_voxel and changed_scene)

        if not changed_scene and not changed_voxel:
            continue

        if not dry_run:
            data = load_npz_dict(path)
            data["scene_name"] = np.asarray(new_scene)
            if has_voxel:
                target_shape = tuple(int(x) for x in np.asarray(data["voxel_grid"]).shape)
                voxel = load_scene_voxel(scene_dir, new_scene, target_shape, scene_cache)
                if voxel is not None:
                    data["voxel_grid"] = voxel
            np.savez(str(path), **data)

        rows.append(
            {
                "path": str(path),
                "source_stem": source_stem,
                "old_scene_name": old_scene,
                "new_scene_name": new_scene,
                "changed_scene": int(changed_scene),
                "changed_voxel": int(changed_voxel),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix LINGO mirror scene names in NPZ dirs.")
    parser.add_argument("--cache_dir", type=Path, default=Path("lingo_smplx_cache"))
    parser.add_argument("--scene_dir", type=Path, default=Path("LINGO/dataset/dataset/Scene"))
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--report_csv", type=Path, default=Path("outputs/mirror_fix_report.csv"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    corrected_map = build_corrected_map(args.cache_dir, args.min_frames, args.max_frames)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for npz_dir in args.dirs:
        if not npz_dir.exists():
            continue
        rows = repair_dir(npz_dir, args.scene_dir, corrected_map, args.dry_run)
        all_rows.extend(rows)
        print(f"{npz_dir}: repaired {len(rows)} files")

    with open(args.report_csv, "w", newline="") as f:
        fieldnames = [
            "path",
            "source_stem",
            "old_scene_name",
            "new_scene_name",
            "changed_scene",
            "changed_voxel",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"report: {args.report_csv} rows={len(all_rows)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
