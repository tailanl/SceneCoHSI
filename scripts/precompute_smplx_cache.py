#!/usr/bin/env python
"""
Precompute LINGO SMPLX cache v2.
Uses SMPL-X pose parameters (axis-angle) for accurate rotation data
and human_joints_aligned for correct joint positions.
Saves as .npz files for fast training.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "kimodo"))


def _downsample_voxel(voxel, target_size=(64, 64, 64)):
    sx, sy, sz = voxel.shape
    tx, ty, tz = target_size
    if sx == tx and sy == ty and sz == tz:
        return voxel
    factors = (tx / sx, ty / sy, tz / sz)
    return scipy_zoom(voxel.astype(np.float32), factors, order=1)


def compute_motion_features(joints, pose, orient, transl, motion_rep):
    """Produce features in KimodoMotionRep layout using:
    - joints (human_joints_aligned [T, 22, 3]) for positions/velocities
    - pose (human_pose [T, 63]) + orient (human_orient [T, 3]) for accurate rotations via FK

    Layout: [smooth_root_pos(3), heading(2), local_joints(66),
             global_rot_data(132), velocities(66), foot_contacts(4)]
    """
    from kimodo.geometry import axis_angle_to_matrix, matrix_to_cont6d
    from kimodo.skeleton.kinematics import fk as kimodo_fk

    num_joints = motion_rep.skeleton.nbjoints
    skel = motion_rep.skeleton
    T = joints.shape[0]
    joints_t = torch.from_numpy(joints.copy()).float()

    if joints_t.shape[1] > num_joints:
        joints_t = joints_t[:, :num_joints, :]

    # ── root & heading ──
    root_positions = joints_t[:, 0, :].clone()
    smooth_root_pos = root_positions.clone()

    diff = root_positions[1:] - root_positions[:-1]
    heading = torch.atan2(diff[:, 2], diff[:, 0])
    heading = torch.cat([heading[:1], heading])
    global_root_heading = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

    # ── local_joints (matching KimodoMotionRep.__call__) ──
    # X, Z: relative to smooth_root; Y: absolute world Y
    local_joints = joints_t - smooth_root_pos[:, None, :]  # X, Z correct
    local_joints[..., 1] = joints_t[..., 1]                # Y = absolute world Y

    # ── global_rot_data from SMPL-X pose params via FK ──
    pose_t = torch.from_numpy(pose.copy()).float()    # [T, 63]
    orient_t = torch.from_numpy(orient.copy()).float()  # [T, 3]
    transl_t = torch.from_numpy(transl.copy()).float()  # [T, 3]

    body_pose_aa = pose_t.reshape(T, 21, 3)             # 21 body joints in axis-angle
    root_orient_aa = orient_t                            # root orientation

    body_rots = axis_angle_to_matrix(body_pose_aa)       # [T, 21, 3, 3]
    root_rot = axis_angle_to_matrix(root_orient_aa)      # [T, 3, 3]

    local_rot_mats = torch.zeros(T, 22, 3, 3)
    local_rot_mats[:, 0] = root_rot
    local_rot_mats[:, 1:] = body_rots

    global_rots, _, _ = kimodo_fk(local_rot_mats, transl_t, skel)
    global_rot_data = matrix_to_cont6d(global_rots).reshape(T, num_joints * 6)

    # ── velocities ──
    vel_global = torch.zeros_like(joints_t)
    vel_global[:-1] = (joints_t[1:] - joints_t[:-1]) * motion_rep.fps
    vel_global[-1] = vel_global[-2]
    velocities = vel_global.reshape(T, -1)

    # ── foot_contacts ──
    foot_contacts = torch.zeros(T, 4)
    for j_idx, col in [(7, 0), (10, 1), (8, 2), (11, 3)]:
        if j_idx < joints_t.shape[1]:
            foot_contacts[:, col] = (joints_t[:, j_idx, 1] < 0.1).float()

    # ── pack in KimodoMotionRep order ──
    features = torch.cat([
        smooth_root_pos,             # 0:3
        global_root_heading,         # 3:5
        local_joints.reshape(T, -1), # 5:71
        global_rot_data,             # 71:203
        velocities,                  # 203:269
        foot_contacts,               # 269:273
    ], dim=-1)

    if hasattr(motion_rep, 'stats') and motion_rep.stats is not None:
        features = motion_rep.normalize(features.unsqueeze(0)).squeeze(0)

    return features.numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="LINGO/dataset")
    parser.add_argument("--model_dir", default="models/Kimodo-SMPLX-RP-v1")
    parser.add_argument("--output_dir", default="lingo_smplx_cache")
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--voxel_size", type=int, nargs=3, default=[64, 64, 64])
    args = parser.parse_args()

    data_dir = Path(args.data_root) / "dataset"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton import SMPLXSkeleton22

    stats_path = Path(args.model_dir) / "stats" / "motion"
    skel = SMPLXSkeleton22()
    motion_rep = KimodoMotionRep(fps=30, stats_path=stats_path, skeleton=skel)
    print(f"SMPLX joints: {skel.nbjoints}, motion_rep_dim: {motion_rep.motion_rep_dim}")

    print(f"Loading metadata from {data_dir}")
    with open(data_dir / "scene_name.pkl", "rb") as f:
        scene_names = pickle.load(f)
    start_idx = np.load(str(data_dir / "start_idx.npy")).flatten()
    end_idx = np.load(str(data_dir / "end_idx.npy")).flatten()
    with open(data_dir / "text_aug.pkl", "rb") as f:
        text_aug = pickle.load(f)

    print(f"Total segments: {len(start_idx)}")

    print(f"Loading data files (memory-mapped)...")
    joints_all = np.load(str(data_dir / "human_joints_aligned.npy"), mmap_mode="r")
    pose_all = np.load(str(data_dir / "human_pose.npy"), mmap_mode="r")
    orient_all = np.load(str(data_dir / "human_orient.npy"), mmap_mode="r")
    transl_all = np.load(str(data_dir / "transl_aligned.npy"), mmap_mode="r")
    print(f"  joints: {joints_all.shape}, pose: {pose_all.shape}")
    print(f"  orient: {orient_all.shape}, transl: {transl_all.shape}")

    print("Preloading scenes...")
    scene_cache = {}
    scene_dir = data_dir / "Scene"
    all_scenes = set()
    for i in range(len(start_idx)):
        s, e = int(start_idx[i]), int(end_idx[i])
        if e - s < args.min_frames or e - s > args.max_frames:
            continue
        if i >= len(scene_names):
            break
        scene_name = scene_names[s]
        all_scenes.add(scene_name)

    for scene_name in all_scenes:
        base_name = scene_name.split("-")[0]
        found = False
        for suffix in [scene_name, base_name]:
            path = scene_dir / f"{suffix}.npy"
            if path.exists():
                voxel = np.load(str(path)).astype(np.float32)
                voxel = _downsample_voxel(voxel, tuple(args.voxel_size))
                scene_cache[scene_name] = voxel
                found = True
                break
        if not found:
            print(f"  WARNING: missing scene file for {scene_name}")

    print(f"Scenes loaded: {len(scene_cache)}")

    seg_count = 0
    skipped = 0
    for i in tqdm(range(len(start_idx)), desc="Caching"):
        s = int(start_idx[i])
        e = int(end_idx[i])
        length = e - s

        if length < args.min_frames or length > args.max_frames:
            skipped += 1
            continue

        if i >= len(scene_names):
            break

        scene_name = scene_names[s]

        joints = joints_all[s:e]
        pose = pose_all[s:e]
        orient = orient_all[s:e]
        transl = transl_all[s:e]

        motion_features = compute_motion_features(joints, pose, orient, transl, motion_rep)

        voxel = scene_cache.get(scene_name, np.zeros(tuple(args.voxel_size), dtype=np.float32))

        text = text_aug[i][0] if i < len(text_aug) and len(text_aug[i]) > 0 else "motion"
        if isinstance(text, (str,)):
            pass
        elif hasattr(text, "decode"):
            text = text.decode("utf-8")
        else:
            text = str(text)

        out_path = output_dir / f"seg_{seg_count:05d}.npz"
        np.savez_compressed(
            str(out_path),
            motion_features=motion_features,
            voxel_grid=voxel,
            length=np.array(length, dtype=np.int64),
            scene_name=np.array(scene_name),
            text=np.array(text),
        )
        seg_count += 1

    print(f"\nDone! {seg_count} segments cached, {skipped} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
