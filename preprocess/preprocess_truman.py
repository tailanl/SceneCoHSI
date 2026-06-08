"""Pre-process TRUMAN data directly (no SOMA conversion) into cached Kimodo .npz format.

TRUMAN provides 24 SMPL joint positions. This script:
1. Reads TRUMAN human_joints.npy (24 joints)
2. Segments by seg_name.npy
3. Converts joint positions → KimodoMotionRep features (root + heading + local_joints + velocities + foot_contacts)
4. Creates voxel grids from TRUMAN object meshes
5. Uses action_label.npy as text descriptions
6. Saves as .npz cache files compatible with existing training pipeline

Usage:
    conda activate kimodo
    PYTHONPATH="kimodo:SOMA:$PYTHONPATH" \
    CHECKPOINT_DIR=models HF_HOME=.hf_cache \
    TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu \
    python kimodo_scene_project/preprocess/preprocess_truman.py \
        --truman_dir TRUMAN \
        --output_dir kimodo/kimodo_sceneco/cached_data_truman \
        --voxel_size 64,64,64 \
        --max_frames 196 \
        --min_frames 40 \
        --encode_text
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)


def load_motion_rep():
    from kimodo.model import load_model
    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device="cpu")
    motion_rep = pretrained.motion_rep
    motion_rep.skeleton = motion_rep.skeleton.to("cpu")
    return motion_rep


def truman_joints_to_features_direct(joints_24, motion_rep):
    """Convert TRUMAN 24-joint positions to KimodoMotionRep features directly.

    Builds features from: root trajectory, heading angle, local joints,
    joint velocities, and foot contacts. No SOMA conversion needed.
    """
    T = joints_24.shape[0]
    target_dim = motion_rep.motion_rep_dim

    joints_t = torch.from_numpy(joints_24).float()

    root_pos = joints_t[:, 0, :]
    smooth_root_pos = root_pos.clone()

    diff = smooth_root_pos[1:] - smooth_root_pos[:-1]
    heading = torch.atan2(diff[:, 2], diff[:, 0])
    heading = torch.cat([heading[:1], heading])
    global_root_heading = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

    local_joints = joints_t - smooth_root_pos[:, None, :]

    velocities = torch.zeros_like(joints_t)
    velocities[1:] = joints_t[1:] - joints_t[:-1]

    foot_contacts = torch.zeros(T, 4)
    l_ankle_idx = 7
    r_ankle_idx = 8
    l_foot_idx = 10
    r_foot_idx = 11
    h_thresh = 0.1
    foot_contacts[:, 0] = (joints_t[:, l_ankle_idx, 1] < h_thresh).float()
    foot_contacts[:, 1] = (joints_t[:, r_ankle_idx, 1] < h_thresh).float()
    foot_contacts[:, 2] = (joints_t[:, l_foot_idx, 1] < h_thresh).float()
    foot_contacts[:, 3] = (joints_t[:, r_foot_idx, 1] < h_thresh).float()

    features = torch.cat(
        [
            smooth_root_pos,
            global_root_heading,
            local_joints.reshape(T, -1),
            velocities.reshape(T, -1),
            foot_contacts,
        ],
        dim=-1,
    )

    if features.shape[-1] < target_dim:
        pad = torch.zeros(T, target_dim - features.shape[-1])
        features = torch.cat([features, pad], dim=-1)
    elif features.shape[-1] > target_dim:
        features = features[:, :target_dim]

    if hasattr(motion_rep, 'stats') and motion_rep.stats is not None:
        features = motion_rep.normalize(features.unsqueeze(0)).squeeze(0)

    return features


def load_object_meshes(obj_mesh_dir):
    """Load all TRUMAN object meshes and return their vertices.

    Returns:
        dict: object_name → [V, 3] vertex array
    """
    meshes = {}
    try:
        import trimesh
        for obj_path in sorted(Path(obj_mesh_dir).glob("*.obj")):
            obj_name = obj_path.stem
            mesh = trimesh.load(str(obj_path), force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                meshes[obj_name] = np.array(mesh.vertices, dtype=np.float32)
            log.info(f"  Loaded mesh: {obj_name} ({meshes[obj_name].shape[0]} vertices)")
    except ImportError:
        log.warning("trimesh not available, voxelization will be skipped")

    return meshes


def create_voxel_from_objects(object_names, object_poses, obj_meshes, grid_size=(64, 64, 64)):
    """Create occupancy voxel grid from TRUMAN object poses and meshes.

    Applies per-object transformations from the pose dict, then samples
    vertex positions into a shared voxel grid.

    Args:
        object_names: list of object name strings
        object_poses: dict object_name → per-frame transform
        obj_meshes: dict object_name → vertices [V, 3]
        grid_size: (X, Y, Z) output grid dimensions

    Returns:
        voxel: [X, Y, Z] float32 occupancy grid
    """
    all_points = []
    for obj_name in object_names:
        if obj_name not in obj_meshes:
            continue
        verts = obj_meshes[obj_name].copy()

        if object_poses is not None and obj_name in object_poses:
            pose = object_poses[obj_name]
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                verts_h = np.hstack([verts, np.ones((len(verts), 1))])
                verts_t = (pose @ verts_h.T).T[:, :3]
                verts = verts_t

        all_points.append(verts)

    if not all_points:
        return np.zeros(grid_size, dtype=np.float32)

    points = np.concatenate(all_points, axis=0)

    gx, gy, gz = grid_size
    min_b = points.min(axis=0) - 0.1
    max_b = points.max(axis=0) + 0.1
    span = max_b - min_b
    voxel_size = float(max(span / np.array([gx, gy, gz])))

    occupancy = np.zeros((gx, gy, gz), dtype=np.float32)
    coords = ((points - min_b) / voxel_size).astype(np.int32)
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < gx)
        & (coords[:, 1] >= 0) & (coords[:, 1] < gy)
        & (coords[:, 2] >= 0) & (coords[:, 2] < gz)
    )
    occupancy[coords[valid, 0], coords[valid, 1], coords[valid, 2]] = 1.0
    return occupancy


def downsample_voxel(voxel, target_size):
    tx, ty, tz = target_size
    sx, sy, sz = voxel.shape
    if sx == tx and sy == ty and sz == tz:
        return voxel.astype(np.float32)
    from scipy.ndimage import zoom as scipy_zoom
    zoom_factors = (tx / sx, ty / sy, tz / sz)
    voxel_resized = scipy_zoom(voxel, zoom_factors, order=0) > 0.5
    voxel_resized = voxel_resized.astype(np.float32)
    result = np.zeros((tx, ty, tz), dtype=np.float32)
    cx, cy, cz = min(tx, voxel_resized.shape[0]), min(ty, voxel_resized.shape[1]), min(tz, voxel_resized.shape[2])
    result[:cx, :cy, :cz] = voxel_resized[:cx, :cy, :cz]
    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess TRUMAN data for Kimodo-SceneCo training")
    parser.add_argument("--truman_dir", type=str, required=False,
                        help="Path to TRUMAN dataset directory")
    parser.add_argument("--output_dir", type=str, default="kimodo/kimodo_sceneco/cached_data_truman",
                        help="Output directory for cached .npz files")
    parser.add_argument("--voxel_size", type=str, default="64,64,64")
    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--encode_text", action="store_true", default=False,
                        help="Encode action labels with LLM2Vec and cache text features")
    parser.add_argument("--add_text_feat_only", action="store_true", default=False,
                        help="Only add text_feat to existing cached npz files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.add_text_feat_only:
        _add_text_feat_to_cache(output_dir)
        return

    voxel_size = tuple(map(int, args.voxel_size.split(",")))
    truman_dir = Path(args.truman_dir)

    log.info("Loading motion_rep...")
    motion_rep = load_motion_rep()

    text_encoder = None
    if args.encode_text:
        log.info("Loading LLM2Vec text encoder...")
        from kimodo_sceneco.model.llm2vec import LLM2VecEncoder
        text_encoder = LLM2VecEncoder(
            base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            dtype="float32",
            llm_dim=4096,
            device="cpu",
        )

    seg_name = np.load(truman_dir / "seg_name.npy", mmap_mode="r")
    human_joints = np.load(truman_dir / "human_joints.npy", mmap_mode="r")
    total_frames = len(seg_name)
    log.info(f"TRUMAN: {total_frames:,} frames, joints shape={human_joints.shape}")

    action_label = np.load(truman_dir / "action_label.npy", mmap_mode="r")
    scene_list = np.load(truman_dir / "scene_list.npy", mmap_mode="r")
    log.info(f"Action labels: {action_label.shape}, Scene labels: {scene_list.shape}")

    obj_pose_dir = truman_dir / "Object_all-20260331T024226Z-3-001" / "Object_all" / "Object_pose"
    obj_mesh_dir = truman_dir / "Object_all-20260331T024226Z-3-001" / "Object_all" / "Object_mesh"

    obj_meshes = {}
    if obj_mesh_dir.exists():
        obj_meshes = load_object_meshes(obj_mesh_dir)
    else:
        log.warning(f"Object mesh directory not found: {obj_mesh_dir}")

    unique_segs, seg_starts, seg_counts = np.unique(seg_name, return_index=True, return_counts=True)

    valid_indices = []
    for i, (seg, count) in enumerate(zip(unique_segs, seg_counts)):
        if count >= args.min_frames:
            valid_indices.append(i)

    log.info(f"Valid segments (>= {args.min_frames} frames): {len(valid_indices)} / {len(unique_segs)}")

    success = 0
    skipped = 0
    failed = 0
    first_errors = []

    for si, seg_idx in enumerate(tqdm(valid_indices, desc="Preprocessing TRUMAN")):
        try:
            seg_id = str(unique_segs[seg_idx])
            start = int(seg_starts[seg_idx])
            count = int(seg_counts[seg_idx])

            actual_end = min(start + args.max_frames, start + count)
            actual_count = actual_end - start

            joints_24 = human_joints[start:actual_end].copy()
            features = truman_joints_to_features_direct(joints_24, motion_rep)

            base_seg = seg_id.split("_augment")[0]
            obj_pose_path = obj_pose_dir / f"{base_seg}.npy"

            voxel = np.zeros(voxel_size, dtype=np.float32)
            if obj_pose_path.exists() and obj_meshes:
                obj_data = np.load(str(obj_pose_path), allow_pickle=True)
                obj_dict = obj_data.item() if obj_data.ndim == 0 else obj_data
                if isinstance(obj_dict, dict):
                    object_names = list(obj_dict.keys())
                    first_pose = obj_dict[object_names[0]]
                    if isinstance(first_pose, np.ndarray) and first_pose.ndim >= 3:
                        sample_pose = {k: v[0] for k, v in obj_dict.items()}
                    else:
                        sample_pose = obj_dict
                    voxel = create_voxel_from_objects(
                        object_names, sample_pose, obj_meshes, grid_size=voxel_size
                    )

            text = str(action_label[start]) if action_label is not None else "motion"

            seg_name_clean = seg_id.replace("_augment", "_").replace("@", "_").replace(":", "-")
            out_path = output_dir / f"seg_{seg_name_clean}.npz"

            save_dict = dict(
                motion_features=features.numpy() if isinstance(features, torch.Tensor) else features,
                voxel_grid=voxel.astype(np.float32),
                text=text,
                length=actual_count,
                scene_name=f"seg_{seg_name_clean}",
            )
            if text_encoder is not None:
                with torch.no_grad():
                    text_feat, _ = text_encoder([text])
                save_dict["text_feat"] = text_feat.cpu().numpy()

            np.savez_compressed(out_path, **save_dict)
            success += 1

        except Exception as e:
            failed += 1
            if len(first_errors) < 5:
                first_errors.append(f"{seg_id}: {type(e).__name__}: {e}")
            continue

    log.info(f"Done! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    if first_errors:
        log.info("First errors:")
        for err in first_errors:
            log.info(f"  {err}")
    log.info(f"Output: {output_dir}")


def _add_text_feat_to_cache(output_dir):
    """Add pre-encoded text features to existing cached npz files."""
    output_dir = Path(output_dir)
    npz_files = sorted(output_dir.glob("seg_*.npz"))
    if not npz_files:
        log.error(f"No cached npz files found in {output_dir}")
        return
    log.info(f"Found {len(npz_files)} cached files. Collecting unique texts...")

    text_to_indices = {}
    file_data = {}
    for npz_path in npz_files:
        data = dict(np.load(str(npz_path), allow_pickle=True))
        if "text_feat" in data:
            continue
        text = str(data.get("text", "motion"))
        if text not in text_to_indices:
            text_to_indices[text] = []
        text_to_indices[text].append(npz_path)
        file_data[npz_path] = data

    if not text_to_indices:
        log.info("All files already have text_feat. Nothing to do.")
        return

    unique_texts = list(text_to_indices.keys())
    log.info(f"Found {len(unique_texts)} unique texts across {len(file_data)} files. Loading LLM2Vec...")

    from kimodo_sceneco.model.llm2vec import LLM2VecEncoder
    text_encoder = LLM2VecEncoder(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        dtype="float32",
        llm_dim=4096,
        device="cpu",
    )

    log.info(f"Encoding {len(unique_texts)} unique texts...")
    text_feat_cache = {}
    encode_batch_size = 16
    for i in tqdm(range(0, len(unique_texts), encode_batch_size), desc="Encoding text batches"):
        batch_texts = unique_texts[i:i + encode_batch_size]
        with torch.no_grad():
            text_feat, _ = text_encoder(batch_texts)
        text_feat_np = text_feat.cpu().numpy()
        for j, text in enumerate(batch_texts):
            text_feat_cache[text] = text_feat_np[j:j+1]

    log.info(f"Writing text features to {len(file_data)} files...")
    for npz_path, data in tqdm(file_data.items(), desc="Saving"):
        try:
            text = str(data.get("text", "motion"))
            data["text_feat"] = text_feat_cache[text]
            np.savez_compressed(str(npz_path), **data)
        except Exception:
            continue

    log.info("Text encoding done!")


if __name__ == "__main__":
    main()
