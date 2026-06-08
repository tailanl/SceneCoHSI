#!/usr/bin/env python
"""Extract root trajectory dataset from LINGO motion data.

Reads existing cache (.npz) or raw SOMA/joints data, extracts root trajectory
features (smooth_root_pos, global_root_heading, local_root_motion), and saves
them as a new dataset for root-only training or analysis.

Usage:
    python kimodo_scene_project/scripts/extract_root_trajectory.py \
        --source smplx_cache \
        --output_dir lingo_root_trajectory_smplx \
        --split both

    python kimodo_scene_project/scripts/extract_root_trajectory.py \
        --source soma_cache \
        --output_dir lingo_root_trajectory_soma \
        --split both

    python kimodo_scene_project/scripts/extract_root_trajectory.py \
        --source raw_joints \
        --output_dir lingo_root_trajectory_raw \
        --split both
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_ROOT / "SOMA"))

os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")


def extract_root_from_smplx_cache(cache_dir, output_dir, split, min_frames, max_frames):
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton import SMPLXSkeleton22

    model_dir = PROJECT_ROOT / "models" / "Kimodo-SMPLX-RP-v1"
    stats_path = model_dir / "stats" / "motion"
    skel = SMPLXSkeleton22()
    motion_rep = KimodoMotionRep(fps=30, stats_path=str(stats_path), skeleton=skel)

    root_slice = motion_rep.root_slice
    global_root_dim = motion_rep.global_root_dim
    local_root_dim = motion_rep.local_root_dim

    print(f"SMPLX skeleton: {skel.nbjoints} joints, motion_rep_dim={motion_rep.motion_rep_dim}")
    print(f"  root_slice: {root_slice} (global_root_dim={global_root_dim})")
    print(f"  local_root_dim: {local_root_dim}")
    print(f"  root features: smooth_root_pos(3) + global_root_heading(2) = 5")
    print(f"  local root features: local_root_rot_vel(1) + local_root_vel(2) + global_root_y(1) = 4")

    npz_files = sorted(Path(cache_dir).glob("seg_*.npz"))
    print(f"Found {len(npz_files)} segments in {cache_dir}")

    segments = []
    for f in npz_files:
        data = np.load(str(f), allow_pickle=True)
        length = int(data["length"])
        if length < min_frames or length > max_frames:
            continue
        segments.append((f, data, length))

    rng = np.random.RandomState(42)
    indices = list(range(len(segments)))
    rng.shuffle(indices)
    n_train = int(len(indices) * 0.9)

    if split == "train":
        selected = set(indices[:n_train])
    elif split == "val":
        selected = set(indices[n_train:])
    else:
        selected = set(indices)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    for i, (f, data, length) in enumerate(tqdm(segments, desc="Extracting root (SMPLX)")):
        if i not in selected:
            continue

        motion_features = torch.from_numpy(data["motion_features"]).float().unsqueeze(0)

        global_root_features = motion_features[:, :, root_slice].squeeze(0).numpy()

        with torch.no_grad():
            local_root_features = motion_rep.global_root_to_local_root(
                motion_features[:, :, root_slice],
                normalized=True,
                lengths=torch.tensor([length]),
            ).squeeze(0).numpy()

        voxel = data["voxel_grid"].copy() if "voxel_grid" in data else np.zeros((64, 64, 64), dtype=np.float32)
        scene_name = str(data["scene_name"]) if "scene_name" in data else "unknown"
        text = str(data["text"]) if "text" in data else ""

        text_feat = data["text_feat"].copy() if "text_feat" in data else None

        save_dict = {
            "global_root_features": global_root_features.astype(np.float32),
            "local_root_features": local_root_features.astype(np.float32),
            "voxel_grid": voxel.astype(np.float32),
            "length": np.array(length, dtype=np.int64),
            "scene_name": np.array(scene_name),
            "text": np.array(text),
            "source_file": np.array(f.name),
        }
        if text_feat is not None:
            save_dict["text_feat"] = text_feat.astype(np.float32)

        out_path = output_dir / f"seg_{count:05d}.npz"
        np.savez_compressed(str(out_path), **save_dict)
        count += 1

    _save_manifest(output_dir, count, global_root_dim, local_root_dim, "smplx_cache", split)
    print(f"Done! {count} segments saved, {skipped} skipped -> {output_dir}")
    return count


def extract_root_from_soma_cache(cache_dir, output_dir, split, min_frames, max_frames):
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton import SOMASkeleton30

    model_dir = PROJECT_ROOT / "models" / "Kimodo-SOMA-RP-v1.1"
    stats_path = model_dir / "stats" / "motion"
    skel = SOMASkeleton30()
    motion_rep = KimodoMotionRep(fps=30, stats_path=str(stats_path), skeleton=skel)

    root_slice = motion_rep.root_slice
    global_root_dim = motion_rep.global_root_dim
    local_root_dim = motion_rep.local_root_dim

    print(f"SOMA30 skeleton: {skel.nbjoints} joints, motion_rep_dim={motion_rep.motion_rep_dim}")
    print(f"  root_slice: {root_slice} (global_root_dim={global_root_dim})")
    print(f"  local_root_dim: {local_root_dim}")

    npz_files = sorted(Path(cache_dir).glob("seg_*.npz"))
    print(f"Found {len(npz_files)} segments in {cache_dir}")

    segments = []
    for f in npz_files:
        data = np.load(str(f), allow_pickle=True)
        length = int(data["length"])
        if length < min_frames or length > max_frames:
            continue
        segments.append((f, data, length))

    rng = np.random.RandomState(42)
    indices = list(range(len(segments)))
    rng.shuffle(indices)
    n_train = int(len(indices) * 0.9)

    if split == "train":
        selected = set(indices[:n_train])
    elif split == "val":
        selected = set(indices[n_train:])
    else:
        selected = set(indices)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, (f, data, length) in enumerate(tqdm(segments, desc="Extracting root (SOMA)")):
        if i not in selected:
            continue

        motion_features = torch.from_numpy(data["motion_features"]).float().unsqueeze(0)

        global_root_features = motion_features[:, :, root_slice].squeeze(0).numpy()

        with torch.no_grad():
            local_root_features = motion_rep.global_root_to_local_root(
                motion_features[:, :, root_slice],
                normalized=True,
                lengths=torch.tensor([length]),
            ).squeeze(0).numpy()

        voxel = data["voxel_grid"].copy() if "voxel_grid" in data else np.zeros((64, 64, 64), dtype=np.float32)
        scene_name = str(data["scene_name"]) if "scene_name" in data else "unknown"
        text = str(data["text"]) if "text" in data else ""
        text_feat = data["text_feat"].copy() if "text_feat" in data else None

        save_dict = {
            "global_root_features": global_root_features.astype(np.float32),
            "local_root_features": local_root_features.astype(np.float32),
            "voxel_grid": voxel.astype(np.float32),
            "length": np.array(length, dtype=np.int64),
            "scene_name": np.array(scene_name),
            "text": np.array(text),
            "source_file": np.array(f.name),
        }
        if text_feat is not None:
            save_dict["text_feat"] = text_feat.astype(np.float32)

        out_path = output_dir / f"seg_{count:05d}.npz"
        np.savez_compressed(str(out_path), **save_dict)
        count += 1

    _save_manifest(output_dir, count, global_root_dim, local_root_dim, "soma_cache", split)
    print(f"Done! {count} segments saved -> {output_dir}")
    return count


def extract_root_from_raw_joints(data_root, output_dir, split, min_frames, max_frames):
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton import SMPLXSkeleton22
    from kimodo.motion_rep.smooth_root import get_smooth_root_pos

    model_dir = PROJECT_ROOT / "models" / "Kimodo-SMPLX-RP-v1"
    stats_path = model_dir / "stats" / "motion"
    skel = SMPLXSkeleton22()
    motion_rep = KimodoMotionRep(fps=30, stats_path=str(stats_path), skeleton=skel)

    global_root_dim = motion_rep.global_root_dim
    local_root_dim = motion_rep.local_root_dim

    print(f"Raw joints -> SMPLX: {skel.nbjoints} joints")
    print(f"  global_root_dim={global_root_dim}, local_root_dim={local_root_dim}")

    data_dir = Path(data_root) / "dataset"
    with open(data_dir / "scene_name.pkl", "rb") as f:
        scene_names = pickle.load(f)
    start_idx = np.load(str(data_dir / "start_idx.npy")).flatten()
    end_idx = np.load(str(data_dir / "end_idx.npy")).flatten()
    with open(data_dir / "text_aug.pkl", "rb") as f:
        text_aug = pickle.load(f)

    joints_path = data_dir / "human_joints_aligned.npy"
    joints_all = np.load(str(joints_path), mmap_mode="r")
    print(f"Joints shape: {joints_all.shape}, segments: {len(start_idx)}")

    scene_dir = data_dir / "Scene"
    scene_cache = {}
    all_scenes = set()
    valid_segments = []
    for i in range(len(start_idx)):
        s, e = int(start_idx[i]), int(end_idx[i])
        length = e - s
        if length < min_frames or length > max_frames:
            continue
        if i >= len(scene_names):
            break
        scene_name = scene_names[s]
        all_scenes.add(scene_name)
        valid_segments.append((i, s, e, length, scene_name))

    for scene_name in tqdm(all_scenes, desc="Loading scenes"):
        base_name = scene_name.split("-")[0]
        for suffix in [scene_name, base_name]:
            path = scene_dir / f"{suffix}.npy"
            if path.exists():
                from scipy.ndimage import zoom as scipy_zoom
                voxel = np.load(str(path)).astype(np.float32)
                sx, sy, sz = voxel.shape
                tx, ty, tz = 64, 64, 64
                if sx != tx or sy != ty or sz != tz:
                    factors = (tx / sx, ty / sy, tz / sz)
                    voxel = scipy_zoom(voxel, factors, order=1)
                scene_cache[scene_name] = voxel
                break

    rng = np.random.RandomState(42)
    indices = list(range(len(valid_segments)))
    rng.shuffle(indices)
    n_train = int(len(indices) * 0.9)

    if split == "train":
        selected = set(indices[:n_train])
    elif split == "val":
        selected = set(indices[n_train:])
    else:
        selected = set(indices)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx_i, (i, s, e, length, scene_name) in enumerate(tqdm(valid_segments, desc="Extracting root (raw)")):
        if idx_i not in selected:
            continue

        joints = joints_all[s:e]
        joints_t = torch.from_numpy(joints.copy()).float()
        if joints_t.shape[1] > skel.nbjoints:
            joints_t = joints_t[:, :skel.nbjoints, :]

        root_pos = joints_t[:, 0, :].unsqueeze(0)

        with torch.no_grad():
            smooth_root = get_smooth_root_pos(root_pos).squeeze(0)

        diff = smooth_root[1:] - smooth_root[:-1]
        heading = torch.atan2(diff[:, 2], diff[:, 0])
        heading = torch.cat([heading[:1], heading])
        global_root_heading = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

        global_root_features = torch.cat([smooth_root, global_root_heading], dim=-1).numpy()

        with torch.no_grad():
            local_root_features = motion_rep.global_root_to_local_root(
                torch.cat([smooth_root, global_root_heading], dim=-1).unsqueeze(0),
                normalized=False,
                lengths=torch.tensor([length]),
            ).squeeze(0).numpy()

        voxel = scene_cache.get(scene_name, np.zeros((64, 64, 64), dtype=np.float32))
        text = text_aug[i][0] if i < len(text_aug) and len(text_aug[i]) > 0 else "motion"
        if not isinstance(text, str):
            text = str(text)

        save_dict = {
            "global_root_features": global_root_features.astype(np.float32),
            "local_root_features": local_root_features.astype(np.float32),
            "voxel_grid": voxel.astype(np.float32),
            "length": np.array(length, dtype=np.int64),
            "scene_name": np.array(scene_name),
            "text": np.array(text),
            "source_seg_idx": np.array(i, dtype=np.int64),
        }

        out_path = output_dir / f"seg_{count:05d}.npz"
        np.savez_compressed(str(out_path), **save_dict)
        count += 1

    _save_manifest(output_dir, count, global_root_dim, local_root_dim, "raw_joints", split)
    print(f"Done! {count} segments saved -> {output_dir}")
    return count


def extract_root_from_soma_raw(soma_dir, output_dir, split, min_frames, max_frames):
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton import SOMASkeleton30
    from kimodo.motion_rep.smooth_root import get_smooth_root_pos

    model_dir = PROJECT_ROOT / "models" / "Kimodo-SOMA-RP-v1.1"
    stats_path = model_dir / "stats" / "motion"
    skel = SOMASkeleton30()
    motion_rep = KimodoMotionRep(fps=30, stats_path=str(stats_path), skeleton=skel)

    global_root_dim = motion_rep.global_root_dim
    local_root_dim = motion_rep.local_root_dim

    print(f"SOMA raw -> SOMA30: {skel.nbjoints} joints")
    print(f"  global_root_dim={global_root_dim}, local_root_dim={local_root_dim}")

    soma_dir = Path(soma_dir)
    npz_files = sorted(soma_dir.glob("seg_*_soma.npz"))
    if not npz_files:
        npz_files = sorted(soma_dir.glob("seg_*.npz"))
    print(f"Found {len(npz_files)} SOMA files in {soma_dir}")

    valid_segments = []
    for f in npz_files:
        data = np.load(str(f), allow_pickle=True)
        if "soma_root_transl" in data:
            root_transl_data = data["soma_root_transl"]
        elif "transl" in data:
            root_transl_data = data["transl"]
        else:
            continue
        T = root_transl_data.shape[0]
        if T < min_frames or T > max_frames:
            continue
        valid_segments.append((f, data, T))

    rng = np.random.RandomState(42)
    indices = list(range(len(valid_segments)))
    rng.shuffle(indices)
    n_train = int(len(indices) * 0.9)

    if split == "train":
        selected = set(indices[:n_train])
    elif split == "val":
        selected = set(indices[n_train:])
    else:
        selected = set(indices)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, (f, data, T) in enumerate(tqdm(valid_segments, desc="Extracting root (SOMA raw)")):
        if i not in selected:
            continue

        if "soma_root_transl" in data:
            root_transl = torch.from_numpy(data["soma_root_transl"].copy()).float().unsqueeze(0)
        elif "transl" in data:
            root_transl = torch.from_numpy(data["transl"].copy()).float().unsqueeze(0)
        else:
            continue

        with torch.no_grad():
            smooth_root = get_smooth_root_pos(root_transl).squeeze(0)

        diff = smooth_root[1:] - smooth_root[:-1]
        heading = torch.atan2(diff[:, 2], diff[:, 0])
        heading = torch.cat([heading[:1], heading])
        global_root_heading = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

        global_root_features = torch.cat([smooth_root, global_root_heading], dim=-1).numpy()

        with torch.no_grad():
            local_root_features = motion_rep.global_root_to_local_root(
                torch.cat([smooth_root, global_root_heading], dim=-1).unsqueeze(0),
                normalized=False,
                lengths=torch.tensor([T]),
            ).squeeze(0).numpy()

        scene_name = str(data.get("scene_name", "unknown")) if "scene_name" in data else "unknown"

        save_dict = {
            "global_root_features": global_root_features.astype(np.float32),
            "local_root_features": local_root_features.astype(np.float32),
            "length": np.array(T, dtype=np.int64),
            "scene_name": np.array(scene_name),
            "source_file": np.array(f.name),
        }

        out_path = output_dir / f"seg_{count:05d}.npz"
        np.savez_compressed(str(out_path), **save_dict)
        count += 1

    _save_manifest(output_dir, count, global_root_dim, local_root_dim, "soma_raw", split)
    print(f"Done! {count} segments saved -> {output_dir}")
    return count


def _save_manifest(output_dir, count, global_root_dim, local_root_dim, source, split):
    manifest = {
        "num_segments": np.array(count, dtype=np.int64),
        "global_root_dim": np.array(global_root_dim, dtype=np.int64),
        "local_root_dim": np.array(local_root_dim, dtype=np.int64),
        "source": np.array(source),
        "split": np.array(split),
        "feature_description": np.array(
            "global_root_features: [T, 5] = smooth_root_pos(3) + global_root_heading(2); "
            "local_root_features: [T, 4] = local_root_rot_vel(1) + local_root_vel(2) + global_root_y(1)"
        ),
    }
    np.savez_compressed(str(output_dir / "manifest.npz"), **manifest)


def main():
    parser = argparse.ArgumentParser(description="Extract root trajectory dataset from LINGO data")
    parser.add_argument("--source", type=str, required=True,
                        choices=["smplx_cache", "soma_cache", "raw_joints", "soma_raw"],
                        help="Data source type")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for root trajectory dataset")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "val", "both"],
                        help="Which split to extract")
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=196)
    args = parser.parse_args()

    print(f"=== Root Trajectory Extraction ===")
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output_dir}")
    print(f"  Split: {args.split}")
    print(f"  Frame range: [{args.min_frames}, {args.max_frames}]")
    print()

    if args.source == "smplx_cache":
        input_dir = args.input_dir or str(PROJECT_ROOT / "lingo_smplx_cache")
        extract_root_from_smplx_cache(input_dir, args.output_dir, args.split, args.min_frames, args.max_frames)

    elif args.source == "soma_cache":
        input_dir = args.input_dir or str(PROJECT_ROOT / "kimodo" / "kimodo_sceneco" / "cached_data")
        extract_root_from_soma_cache(input_dir, args.output_dir, args.split, args.min_frames, args.max_frames)

    elif args.source == "raw_joints":
        input_dir = args.input_dir or str(PROJECT_ROOT / "LINGO" / "dataset")
        extract_root_from_raw_joints(input_dir, args.output_dir, args.split, args.min_frames, args.max_frames)

    elif args.source == "soma_raw":
        input_dir = args.input_dir or str(PROJECT_ROOT / "soma_converted_all" / "lingo")
        extract_root_from_soma_raw(input_dir, args.output_dir, args.split, args.min_frames, args.max_frames)


if __name__ == "__main__":
    main()
