"""Pre-process SOMA-converted LINGO data into cached Kimodo motion features.

Uses soma77_joints from seg_XXXXX.npz directly (already FK'd),
then converts joint positions + root positions to KimodoMotionRep features.

Optionally encodes text with LLM2Vec and caches text features to avoid
loading the text encoder during training (fixes CUDA OOM).

Usage:
    PYTHONPATH="kimodo:SOMA:$PYTHONPATH" \
    CHECKPOINT_DIR=models HF_HOME=.hf_cache \
    TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu \
    python -m kimodo_sceneco.train.preprocess \
        --soma_lingo_dir soma_converted_all/lingo \
        --scene_dir LINGO/dataset/dataset/Scene \
        --output_dir kimodo/kimodo_sceneco/cached_data \
        --voxel_size 64,64,64 \
        --max_frames 196 \
        --min_frames 40 \
        --method joints_direct \
        --encode_text
"""

import argparse
import logging
import os
import pickle
import random
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


def joints_to_features_via_fk(joints77, root_transl, motion_rep, skel77, skel30):
    """Convert 77-joint positions to KimodoMotionRep features via inverse FK -> FK.

    Since we have joint positions (not rotations), we use the skeleton's
    inverse kinematics to recover local rotations, then forward kinematics
    to get the proper KimodoMotionRep features.
    """
    T = joints77.shape[0]
    device = joints77.device

    local_rot_mats_77 = skel77.ik(joints77, root_transl)

    global_rot_mats_77 = skel77.fk(local_rot_mats_77, root_transl)

    from kimodo.skeleton.transforms import global_rots_to_local_rots
    local_rot_mats_77_clean = global_rots_to_local_rots(global_rot_mats_77, skel77)

    local_rot_mats_30 = skel30.from_SOMASkeleton77(local_rot_mats_77_clean)

    local_rot_mats_30 = local_rot_mats_30.unsqueeze(0)
    root_positions = root_transl.unsqueeze(0)

    features = motion_rep(
        local_rot_mats_30,
        root_positions,
        to_normalize=True,
        to_canonicalize=True,
    )

    return features.squeeze(0)


def joints_to_features_direct(joints77, root_transl, motion_rep):
    """Direct conversion: use joint positions to build approximate features.

    This bypasses FK/IK entirely and constructs features from joint positions,
    root trajectory, and foot contacts.
    """
    T = joints77.shape[0]
    target_dim = motion_rep.motion_rep_dim

    root_pos = root_transl
    smooth_root_pos = root_pos.clone()

    diff = smooth_root_pos[1:] - smooth_root_pos[:-1]
    heading = torch.atan2(diff[:, 2], diff[:, 0])
    heading = torch.cat([heading[:1], heading])
    global_root_heading = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

    local_joints = joints77 - smooth_root_pos[:, None, :]

    velocities = torch.zeros_like(joints77)
    velocities[1:] = joints77[1:] - joints77[:-1]

    foot_contacts = torch.zeros(T, 4)
    l_foot_idx = 7
    r_foot_idx = 8
    l_toe_idx = 10
    r_toe_idx = 11
    if joints77.shape[1] > max(l_toe_idx, r_toe_idx):
        h_thresh = 0.1
        foot_contacts[:, 0] = (joints77[:, l_foot_idx, 1] < h_thresh).float()
        foot_contacts[:, 1] = (joints77[:, l_toe_idx, 1] < h_thresh).float()
        foot_contacts[:, 2] = (joints77[:, r_foot_idx, 1] < h_thresh).float()
        foot_contacts[:, 3] = (joints77[:, r_toe_idx, 1] < h_thresh).float()

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


def convert_segment_soma_rotvec(seg_path, motion_rep, converter):
    """Convert using soma_poses_rotvec + joint_orient from _soma.npz."""
    data = np.load(str(seg_path), allow_pickle=True)

    poses_rotvec = data["soma_poses_rotvec"]
    root_transl = data["soma_root_transl"]
    parents_soma = data["parents_soma"]

    soma_path = seg_path.parent / seg_path.name.replace(".npz", "_soma.npz")
    if not soma_path.exists():
        return None

    soma_data = np.load(str(soma_path), allow_pickle=True)
    joint_orient = soma_data["joint_orient"]

    T = poses_rotvec.shape[0]
    num_joints = poses_rotvec.shape[1]

    rotvec_to_matrix = converter["rotvec_to_matrix"]
    precompute_joint_orient = converter["precompute_joint_orient"]
    apply_joint_orient_local = converter["apply_joint_orient_local"]
    skel77 = converter["skel77"]
    skel30 = converter["skel30"]
    global_rots_to_local_rots = converter["global_rots_to_local_rots"]

    poses_t = torch.from_numpy(poses_rotvec.copy()).float()
    root_transl_t = torch.from_numpy(root_transl.copy()).float()

    rel_rotmats = rotvec_to_matrix(poses_t)

    joint_orient_t = torch.from_numpy(joint_orient).float()
    if joint_orient_t.shape[0] > num_joints:
        joint_orient_t = joint_orient_t[:num_joints]

    parent_ids = list(parents_soma)
    if len(parent_ids) > num_joints:
        parent_ids = parent_ids[:num_joints]

    orient_tensor, orient_parent_T = precompute_joint_orient(joint_orient_t, parent_ids)
    abs_rotmats = apply_joint_orient_local(rel_rotmats, orient_tensor, orient_parent_T)

    global_rot_mats = torch.zeros(T, num_joints, 3, 3)
    for j in range(num_joints):
        p = parent_ids[j]
        if p == -1 or p < 0:
            global_rot_mats[:, j] = abs_rotmats[:, j]
        else:
            global_rot_mats[:, j] = torch.einsum(
                "tij,tjk->tik",
                global_rot_mats[:, p],
                abs_rotmats[:, j],
            )

    local_rot_mats_77 = global_rots_to_local_rots(global_rot_mats, skel77)
    local_rot_mats_30 = skel30.from_SOMASkeleton77(local_rot_mats_77)

    local_rot_mats_30 = local_rot_mats_30.unsqueeze(0)
    root_positions = root_transl_t.unsqueeze(0)

    features = motion_rep(
        local_rot_mats_30,
        root_positions,
        to_normalize=True,
        to_canonicalize=True,
    )

    return features.squeeze(0)


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
    cx = min(tx, voxel_resized.shape[0])
    cy = min(ty, voxel_resized.shape[1])
    cz = min(tz, voxel_resized.shape[2])
    result[:cx, :cy, :cz] = voxel_resized[:cx, :cy, :cz]
    return result


def add_text_feat_to_cache(args):
    """Add pre-encoded text features to existing cached npz files."""
    output_dir = Path(args.output_dir)

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
    log.info("LLM2Vec loaded on CPU")

    log.info(f"Encoding {len(unique_texts)} unique texts...")
    text_feat_cache = {}
    encode_batch_size = 128
    for i in tqdm(range(0, len(unique_texts), encode_batch_size), desc="Encoding text batches"):
        batch_texts = unique_texts[i:i + encode_batch_size]
        with torch.no_grad():
            text_feat, _ = text_encoder(batch_texts)
        text_feat_np = text_feat.cpu().numpy()
        for j, text in enumerate(batch_texts):
            text_feat_cache[text] = text_feat_np[j:j+1]

    log.info(f"Writing text features to {len(file_data)} files...")
    success = 0
    failed = 0
    for npz_path, data in tqdm(file_data.items(), desc="Saving"):
        try:
            text = str(data.get("text", "motion"))
            data["text_feat"] = text_feat_cache[text]
            np.savez_compressed(str(npz_path), **data)
            success += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                log.error(f"Failed: {npz_path.name}: {e}")
            continue

    log.info(f"Text encoding done! Success: {success}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soma_lingo_dir", type=str, default=None)
    parser.add_argument("--scene_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--voxel_size", type=str, default="64,64,64")
    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--method", type=str, default="rotvec",
                        choices=["rotvec", "joints_direct"],
                        help="Conversion method: rotvec (FK from rotvec) or joints_direct (from joint positions)")
    parser.add_argument("--encode_text", action="store_true", default=False,
                        help="Encode text with LLM2Vec and cache text features")
    parser.add_argument("--add_text_feat_only", action="store_true", default=False,
                        help="Only add text_feat to existing cached npz files (skip motion/voxel processing)")
    args = parser.parse_args()

    if args.add_text_feat_only:
        add_text_feat_to_cache(args)
        return

    if not args.soma_lingo_dir or not args.scene_dir:
        parser.error("--soma_lingo_dir and --scene_dir are required when not using --add_text_feat_only")

    voxel_size = tuple(map(int, args.voxel_size.split(",")))
    soma_lingo_dir = Path(args.soma_lingo_dir)
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        log.info("LLM2Vec loaded on CPU")

    converter = None
    if args.method == "rotvec":
        log.info("Loading SOMA converter...")
        from soma.geometry.rig_utils import precompute_joint_orient, apply_joint_orient_local
        from soma.geometry.transforms import rotvec_to_matrix
        from kimodo.skeleton import SOMASkeleton77, SOMASkeleton30
        from kimodo.skeleton.transforms import global_rots_to_local_rots

        skel77 = SOMASkeleton77()
        skel30 = SOMASkeleton30()
        converter = {
            "rotvec_to_matrix": rotvec_to_matrix,
            "precompute_joint_orient": precompute_joint_orient,
            "apply_joint_orient_local": apply_joint_orient_local,
            "skel77": skel77,
            "skel30": skel30,
            "global_rots_to_local_rots": global_rots_to_local_rots,
        }

    seg_files = sorted([f for f in soma_lingo_dir.iterdir()
                        if f.name.endswith(".npz") and "_soma" not in f.name])
    log.info(f"Found {len(seg_files)} segment files")

    scene_cache = {}
    text_aug = None
    text_aug_path = soma_lingo_dir.parent.parent / "LINGO" / "dataset" / "dataset" / "text_aug.pkl"
    if text_aug_path.exists():
        with open(text_aug_path, "rb") as f:
            text_aug = pickle.load(f)
        log.info(f"Loaded text_aug: {len(text_aug)} entries")

    success = 0
    skipped = 0
    failed = 0
    first_errors = []

    # Pre-scan: collect scene_name for each non-mirrored segment index.
    # Mirrored segments in the LINGO dataset are ALL labeled "005_mirror"
    # regardless of their actual scene. We fix this by pairing each mirrored
    # segment with its non-mirrored counterpart (seg_idx - N/2) and using
    # the mirrored version of that scene instead.
    non_mirror_scene_by_idx = {}
    for seg_path in tqdm(seg_files, desc="Pre-scanning scenes"):
        seg_idx = int(seg_path.stem.replace("seg_", ""))
        data = np.load(str(seg_path), allow_pickle=True)
        scene_name = str(data.get("scene_name", ""))
        if "_mirror" not in scene_name:
            non_mirror_scene_by_idx[seg_idx] = scene_name
    log.info(f"Pre-scanned {len(non_mirror_scene_by_idx)} non-mirrored scene mappings")

    if non_mirror_scene_by_idx:
        half_offset = max(non_mirror_scene_by_idx.keys()) + 1
        log.info(f"Mirror offset (half dataset size): {half_offset}")

    for seg_path in tqdm(seg_files, desc="Preprocessing"):
        try:
            data = np.load(str(seg_path), allow_pickle=True)
            num_frames = int(data["num_frames"])

            if num_frames < args.min_frames or num_frames > args.max_frames:
                skipped += 1
                continue

            seg_name = seg_path.stem
            seg_idx = int(seg_name.replace("seg_", ""))

            if args.method == "rotvec":
                features = convert_segment_soma_rotvec(seg_path, motion_rep, converter)
            else:
                joints77 = torch.from_numpy(data["soma77_joints"].copy()).float()
                root_transl = torch.from_numpy(data["soma_root_transl"].copy()).float()
                features = joints_to_features_direct(joints77, root_transl, motion_rep)

            if features is None:
                skipped += 1
                continue

            scene_name = str(data["scene_name"])

            # Fix mirrored scene names: use the mirrored version of the
            # partner segment's scene instead of the generic "005_mirror".
            if "_mirror" in scene_name:
                partner_idx = seg_idx - half_offset
                if partner_idx in non_mirror_scene_by_idx:
                    partner_scene = non_mirror_scene_by_idx[partner_idx]
                    corrected_scene = f"{partner_scene}_mirror"
                    if corrected_scene != scene_name:
                        scene_name = corrected_scene

            if scene_name not in scene_cache:
                base_name = scene_name.split("-")[0]
                voxel = None
                for suffix in [scene_name, base_name]:
                    vpath = scene_dir / f"{suffix}.npy"
                    if vpath.exists():
                        voxel = np.load(str(vpath)).astype(np.float32)
                        voxel = downsample_voxel(voxel, voxel_size)
                        break
                if voxel is None:
                    voxel = np.zeros(voxel_size, dtype=np.float32)
                scene_cache[scene_name] = voxel

            if text_aug and seg_idx < len(text_aug):
                texts = text_aug[seg_idx]
                text = texts[0] if isinstance(texts, list) else str(texts)
            else:
                text = str(data.get("dataset", "motion"))

            out_path = output_dir / f"{seg_name}.npz"
            save_dict = dict(
                motion_features=features.numpy() if isinstance(features, torch.Tensor) else features,
                voxel_grid=scene_cache[scene_name],
                text=text,
                length=num_frames,
                scene_name=scene_name,
            )
            if text_encoder is not None:
                with torch.no_grad():
                    text_feat, _ = text_encoder([text])
                save_dict["text_feat"] = text_feat.cpu().numpy()
            np.savez_compressed(out_path, **save_dict)
            success += 1

        except Exception as e:
            failed += 1
            if len(first_errors) < 3:
                first_errors.append(f"{seg_path.name}: {type(e).__name__}: {e}")
            continue

    log.info(f"Done! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    if first_errors:
        log.info("First errors:")
        for err in first_errors:
            log.info(f"  {err}")
    log.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
