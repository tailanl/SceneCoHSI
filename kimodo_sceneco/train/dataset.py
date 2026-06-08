# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LINGO scene-motion dataset for Kimodo-SceneCo training.

Supports two modes:
1. Cached mode (recommended): loads pre-processed .npz files from cache_dir
2. Online mode: converts SOMA data on-the-fly (slow, requires SMPL/Warp)
"""

import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class LINGOSceneMotionDataset(Dataset):
    """LINGO dataset with scene voxel grids, motion features, and text prompts.

    Each sample provides:
    - motion_features: [T, D] normalized motion in KimodoMotionRep format
    - voxel_grid: [C, X, Y, Z] scene voxel occupancy grid (downsampled)
    - text: string prompt
    - length: int, number of valid frames
    - scene_name: string identifier
    """

    def __init__(
        self,
        data_root: str,
        motion_rep=None,
        max_frames: int = 196,
        min_frames: int = 40,
        fps: int = 30,
        voxel_size: Tuple[int, int, int] = (64, 64, 64),
        use_augmented_text: bool = True,
        scene_dropout: float = 0.1,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        soma_data_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        no_soma_conversion: bool = False,
        root_trajectory_data: bool = False,
        # --- External root for Stage2 Root-Guided SceneCo ---
        external_root_enabled: bool = False,
        path_guided_root_dir: Optional[str] = None,
        path_scene_guided_root_dir: Optional[str] = None,
        root_condition_mix: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.motion_rep = motion_rep
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.fps = fps
        self.target_voxel_size = voxel_size
        self.use_augmented_text = use_augmented_text
        self.scene_dropout = scene_dropout
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.no_soma_conversion = no_soma_conversion
        self.root_trajectory_data = root_trajectory_data

        # External root for Stage2 Root-Guided SceneCo training
        self.external_root_enabled = external_root_enabled
        self.path_guided_root_dir = Path(path_guided_root_dir) if path_guided_root_dir else None
        self.path_scene_guided_root_dir = Path(path_scene_guided_root_dir) if path_scene_guided_root_dir else None
        self.root_condition_mix = root_condition_mix or {}

        self.soma_data_root = Path(soma_data_root) if soma_data_root else self.data_root.parent.parent / "SOMA" / "assets"
        self.soma_lingo_dir = self.data_root.parent.parent / "soma_converted_all" / "lingo"

        self.dataset_dir = self.data_root / "dataset"

        self._soma_converter = None

        if self.cache_dir and self.cache_dir.exists():
            log.info(f"Loading from cached data: {self.cache_dir}")
            self._load_cached_index()
        else:
            log.info("Loading LINGO dataset (SOMA-converted)...")
            self._load_metadata()
            self._build_segment_index()
            self._split_data(train_ratio, seed)
            self._preload_scenes()

        log.info(
            f"LINGO dataset: {len(self.segments)} segments ({split}), "
            f"{len(self.scene_cache)} scenes"
        )

    def _load_cached_index(self):
        self.scene_cache = {}
        self.segments = []

        npz_files = sorted(self.cache_dir.glob("seg_*.npz"))
        for f in npz_files:
            data = np.load(str(f), allow_pickle=True)
            length = int(data["length"])
            if length < self.min_frames or length > self.max_frames:
                continue
            scene_name = str(data["scene_name"])
            self.segments.append({
                "cache_path": str(f),
                "length": length,
                "scene_name": scene_name,
            })

        rng = random.Random(42)
        indices = list(range(len(self.segments)))
        rng.shuffle(indices)
        n_train = int(len(indices) * 0.9)

        if self.split == "train":
            selected = indices[:n_train]
        else:
            selected = indices[n_train:]
        self.segments = [self.segments[i] for i in selected]

    def _get_soma_converter(self):
        if self._soma_converter is not None:
            return self._soma_converter
        from soma.geometry.rig_utils import precompute_joint_orient, apply_joint_orient_local
        from soma.geometry.transforms import rotvec_to_matrix
        from kimodo.skeleton import SOMASkeleton77, SOMASkeleton30
        from kimodo.skeleton.transforms import global_rots_to_local_rots

        skel77 = SOMASkeleton77()
        skel30 = SOMASkeleton30()

        self._soma_converter = {
            "rotvec_to_matrix": rotvec_to_matrix,
            "precompute_joint_orient": precompute_joint_orient,
            "apply_joint_orient_local": apply_joint_orient_local,
            "skel77": skel77,
            "skel30": skel30,
            "global_rots_to_local_rots": global_rots_to_local_rots,
        }
        return self._soma_converter

    @staticmethod
    def _manual_soma_fk(abs_local_rots, joint_parents_soma, bind_offsets, root_transl):
        T, J = abs_local_rots.shape[:2]
        device = abs_local_rots.device

        global_rot_mats = torch.zeros(T, J, 3, 3, device=device)
        root_idx = None
        for j in range(J):
            p = joint_parents_soma[j]
            if p == -1 or p < 0:
                root_idx = j
                global_rot_mats[:, j] = abs_local_rots[:, j]
            else:
                global_rot_mats[:, j] = torch.einsum(
                    "tij,tjk->tik",
                    global_rot_mats[:, p],
                    abs_local_rots[:, j],
                )

        return global_rot_mats

    def _load_metadata(self):
        with open(self.dataset_dir / "scene_name.pkl", "rb") as f:
            self.scene_names = pickle.load(f)

        self.start_idx = np.load(str(self.dataset_dir / "start_idx.npy")).flatten()
        self.end_idx = np.load(str(self.dataset_dir / "end_idx.npy")).flatten()
        log.info(f"  segments: {len(self.start_idx)}")

        with open(self.dataset_dir / "text_aug.pkl", "rb") as f:
            self.text_aug = pickle.load(f)

        with open(
            self.dataset_dir / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl",
            "rb",
        ) as f:
            self.lang_dict = pickle.load(f)

        self._scan_soma_files()

    def _scan_soma_files(self):
        self.soma_files = {}
        if self.no_soma_conversion:
            log.info("SOMA conversion disabled, using raw LINGO joints")
            return
        if not self.soma_lingo_dir.exists():
            log.warning(f"SOMA LINGO directory not found: {self.soma_lingo_dir}")
            return

        for f in sorted(self.soma_lingo_dir.iterdir()):
            if f.name.endswith("_soma.npz"):
                seg_prefix = f.name.replace("_soma.npz", "")
                self.soma_files[seg_prefix] = f

        log.info(f"  SOMA files found: {len(self.soma_files)}")

    def _build_segment_index(self):
        self.segments = []
        for i in range(len(self.start_idx)):
            s, e = int(self.start_idx[i]), int(self.end_idx[i])
            length = e - s
            if length < self.min_frames or length > self.max_frames:
                continue

            if i >= len(self.scene_names):
                break
            scene_name = self.scene_names[s]

            if i < len(self.text_aug):
                texts = self.text_aug[i]
            else:
                texts = ["motion"]

            seg_key = f"seg_{i:05d}"
            has_soma = seg_key in self.soma_files

            self.segments.append(
                {
                    "idx": i,
                    "start": s,
                    "end": e,
                    "length": length,
                    "scene_name": scene_name,
                    "texts": texts,
                    "soma_key": seg_key,
                    "has_soma": has_soma,
                }
            )

        total = len(self.segments)
        with_soma = sum(1 for s in self.segments if s["has_soma"])
        log.info(f"  Segments with SOMA data: {with_soma}/{total}")

    def _split_data(self, train_ratio: float, seed: int):
        rng = random.Random(seed)
        indices = list(range(len(self.segments)))
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:]
        else:
            selected = indices

        self.segments = [self.segments[i] for i in selected]

    def _preload_scenes(self):
        self.scene_cache = {}
        scene_dir = self.dataset_dir / "Scene"
        if not scene_dir.exists():
            log.warning(f"Scene directory not found: {scene_dir}")
            return

        unique_scenes = set(seg["scene_name"] for seg in self.segments)
        for scene_name in unique_scenes:
            base_name = scene_name.split("-")[0]
            for suffix in [scene_name, base_name]:
                path = scene_dir / f"{suffix}.npy"
                if path.exists():
                    voxel = np.load(str(path)).astype(np.float32)
                    voxel = self._downsample_voxel(voxel)
                    self.scene_cache[scene_name] = voxel
                    break

        missing = unique_scenes - set(self.scene_cache.keys())
        if missing:
            log.warning(f"Missing scene files for: {list(missing)[:10]}...")

    def _downsample_voxel(self, voxel: np.ndarray) -> np.ndarray:
        tx, ty, tz = self.target_voxel_size
        sx, sy, sz = voxel.shape

        if sx == tx and sy == ty and sz == tz:
            return voxel

        from scipy.ndimage import zoom as scipy_zoom

        zoom_factors = (tx / sx, ty / sy, tz / sz)
        voxel = scipy_zoom(voxel, zoom_factors, order=0) > 0.5
        voxel = voxel.astype(np.float32)

        result = np.zeros((tx, ty, tz), dtype=np.float32)
        cx, cy, cz = min(tx, voxel.shape[0]), min(ty, voxel.shape[1]), min(tz, voxel.shape[2])
        result[:cx, :cy, :cz] = voxel[:cx, :cy, :cz]
        return result

    def _soma_to_kimodo_features(self, seg: dict) -> torch.Tensor:
        soma_key = seg["soma_key"]
        if soma_key not in self.soma_files:
            return self._fallback_joints_to_features(seg)

        soma_path = self.soma_files[soma_key]
        data = np.load(str(soma_path), allow_pickle=True)

        poses_rotvec = data["poses"]
        root_transl = data["transl"]
        joint_orient = data["joint_orient"]

        T = poses_rotvec.shape[0]
        num_joints = poses_rotvec.shape[1]

        converter = self._get_soma_converter()
        rotvec_to_matrix = converter["rotvec_to_matrix"]
        precompute_joint_orient = converter["precompute_joint_orient"]
        apply_joint_orient_local = converter["apply_joint_orient_local"]
        skel77 = converter["skel77"]
        skel30 = converter["skel30"]
        global_rots_to_local_rots = converter["global_rots_to_local_rots"]

        poses_t = torch.from_numpy(poses_rotvec).float()
        root_transl_t = torch.from_numpy(root_transl).float()

        rel_rotmats = rotvec_to_matrix(poses_t)

        joint_orient_t = torch.from_numpy(joint_orient).float()
        if joint_orient_t.shape[0] > num_joints:
            joint_orient_t = joint_orient_t[:num_joints]

        from soma.soma import SOMALayer
        soma_tmp = SOMALayer(
            str(self.soma_data_root),
            identity_model_type="smpl",
            device="cpu",
            mode="warp",
        )
        parent_ids = list(soma_tmp.rig_data["joint_parent_ids"])
        if len(parent_ids) > num_joints:
            parent_ids = parent_ids[:num_joints]

        orient_tensor, orient_parent_T = precompute_joint_orient(joint_orient_t, parent_ids)
        abs_rotmats = apply_joint_orient_local(rel_rotmats, orient_tensor, orient_parent_T)

        global_rot_mats_77 = self._manual_soma_fk(
            abs_rotmats, parent_ids, None, root_transl_t
        )

        local_rot_mats_77 = global_rots_to_local_rots(global_rot_mats_77, skel77)

        local_rot_mats_30 = skel30.from_SOMASkeleton77(local_rot_mats_77)

        local_rot_mats_30 = local_rot_mats_30.unsqueeze(0)
        root_positions = root_transl_t.unsqueeze(0)

        features = self.motion_rep(
            local_rot_mats_30,
            root_positions,
            to_normalize=True,
            to_canonicalize=True,
        )

        return features.squeeze(0)

    def _fallback_joints_to_features(self, seg: dict) -> torch.Tensor:
        s, e = seg["start"], seg["end"]
        joints_path = self.dataset_dir / "human_joints_aligned.npy"
        if not hasattr(self, '_human_joints_lazy'):
            self._human_joints_lazy = np.load(str(joints_path), mmap_mode="r")
        joints = self._human_joints_lazy[s:e]

        T = joints.shape[0]
        joints_t = torch.from_numpy(joints).float()

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
        if joints_t.shape[1] > 10:
            l_ankle = joints_t[:, 7, 1]
            r_ankle = joints_t[:, 8, 1]
            l_foot = joints_t[:, 10, 1]
            r_foot = joints_t[:, 11, 1]
            h_thresh = 0.1
            foot_contacts[:, 0] = (l_ankle < h_thresh).float()
            foot_contacts[:, 1] = (l_foot < h_thresh).float()
            foot_contacts[:, 2] = (r_ankle < h_thresh).float()
            foot_contacts[:, 3] = (r_foot < h_thresh).float()

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

        target_dim = self.motion_rep.motion_rep_dim
        if features.shape[-1] < target_dim:
            pad = torch.zeros(T, target_dim - features.shape[-1])
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > target_dim:
            features = features[:, :target_dim]

        if hasattr(self.motion_rep, 'stats') and self.motion_rep.stats is not None:
            features = self.motion_rep.normalize(features.unsqueeze(0)).squeeze(0)

        return features

    def __len__(self) -> int:
        return len(self.segments)

    def _load_external_root(self, source_id: str, motion_features: np.ndarray) -> Tuple[np.ndarray, str]:
        """Load external root with mixing strategy: GT / path-guided / path+scene guided.

        Args:
            source_id: segment identifier for looking up guided root NPZ files
            motion_features: (T, D) GT motion features (for extracting GT root_slice)

        Returns:
            (root_5d_norm, root_source) — normalized 5D root in Kimodo feature space
        """
        r = np.random.rand()
        p_gt = self.root_condition_mix.get("gt_root", 0.0)
        p_path = self.root_condition_mix.get("path_guided_root", 0.0)
        p_scene = self.root_condition_mix.get("path_scene_guided_root", 0.0)

        if r < p_gt:
            # GT root: slice first 5 dims from motion_features
            root = motion_features[:, :5].copy().astype(np.float32)
            return root, "gt_root"

        elif r < p_gt + p_path:
            if self.path_guided_root_dir is None:
                # fallback to GT
                root = motion_features[:, :5].copy().astype(np.float32)
                return root, "gt_root_fallback"
            npz_path = self.path_guided_root_dir / f"{source_id}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                return data["guided_root_5d_norm"].astype(np.float32), "path_guided_root"
            else:
                root = motion_features[:, :5].copy().astype(np.float32)
                return root, "path_root_missing_gt_fallback"

        else:
            if self.path_scene_guided_root_dir is None:
                # fallback to GT
                root = motion_features[:, :5].copy().astype(np.float32)
                return root, "gt_root_fallback"
            npz_path = self.path_scene_guided_root_dir / f"{source_id}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                return data["guided_root_5d_norm"].astype(np.float32), "path_scene_guided_root"
            else:
                root = motion_features[:, :5].copy().astype(np.float32)
                return root, "path_scene_root_missing_gt_fallback"

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seg = self.segments[index]

        if self.cache_dir and "cache_path" in seg:
            return self._getitem_cached(seg)

        return self._getitem_online(seg)

    def _getitem_cached(self, seg: dict) -> Dict[str, torch.Tensor]:
        data = np.load(seg["cache_path"], allow_pickle=True)

        if "motion_features" in data:
            motion_features = torch.from_numpy(data["motion_features"]).float()
        elif self.root_trajectory_data and "global_root_features" in data and self.motion_rep is not None:
            global_root = torch.from_numpy(data["global_root_features"]).float()
            D = self.motion_rep.motion_rep_dim
            root_slice = self.motion_rep.root_slice
            motion_features = torch.zeros(len(global_root), D)
            motion_features[:, root_slice] = global_root
        else:
            raise KeyError(
                f"Cache file {seg['cache_path']} missing 'motion_features'. "
                f"Available keys: {list(data.keys())}. "
                f"Set root_trajectory_data=True if using root trajectory dataset."
            )

        voxel = data["voxel_grid"].copy()
        length = int(data["length"])
        scene_name = str(data["scene_name"])

        if self.scene_dropout > 0 and self.split == "train" and random.random() < self.scene_dropout:
            voxel = np.zeros_like(voxel)

        voxel_tensor = torch.from_numpy(voxel).unsqueeze(0)
        text = str(data.get("text", "motion"))

        result = {
            "motion_features": motion_features,
            "voxel_grid": voxel_tensor,
            "text": text,
            "length": length,
            "scene_name": scene_name,
        }

        if "text_feat" in data:
            result["text_feat"] = torch.from_numpy(data["text_feat"]).float()

        # External root for Stage2 Root-Guided SceneCo
        if self.external_root_enabled:
            cache_stem = Path(seg["cache_path"]).stem  # e.g. "seg_00001"
            external_root_np, root_source = self._load_external_root(
                source_id=cache_stem,
                motion_features=data["motion_features"] if "motion_features" in data else np.zeros((length, 273)),
            )
            result["external_root"] = torch.from_numpy(external_root_np).float()
            result["external_root_source"] = root_source

        return result

    def _getitem_online(self, seg: dict) -> Dict[str, torch.Tensor]:
        length = seg["length"]

        if seg["has_soma"]:
            motion_features = self._soma_to_kimodo_features(seg)
        else:
            motion_features = self._fallback_joints_to_features(seg)

        scene_name = seg["scene_name"]
        if scene_name in self.scene_cache:
            voxel = self.scene_cache[scene_name].copy()
        else:
            voxel = np.zeros(self.target_voxel_size, dtype=np.float32)

        if self.scene_dropout > 0 and self.split == "train" and random.random() < self.scene_dropout:
            voxel = np.zeros_like(voxel)

        voxel_tensor = torch.from_numpy(voxel).unsqueeze(0)

        texts = seg["texts"]
        text = random.choice(texts) if isinstance(texts, list) else texts

        result = {
            "motion_features": motion_features,
            "voxel_grid": voxel_tensor,
            "text": text,
            "length": length,
            "scene_name": scene_name,
        }

        # External root for Stage2 Root-Guided SceneCo
        if self.external_root_enabled:
            source_id = f"seg_{seg['idx']:05d}"
            motion_np = motion_features.numpy() if isinstance(motion_features, torch.Tensor) else motion_features
            external_root_np, root_source = self._load_external_root(
                source_id=source_id,
                motion_features=motion_np,
            )
            result["external_root"] = torch.from_numpy(external_root_np).float()
            result["external_root_source"] = root_source

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(item["length"] for item in batch)
    B = len(batch)
    D = batch[0]["motion_features"].shape[-1]
    voxel_size = batch[0]["voxel_grid"].shape[1:]

    motion_padded = torch.zeros(B, max_len, D)
    motion_mask = torch.zeros(B, max_len, dtype=torch.bool)
    voxel_grids = torch.zeros(B, 1, *voxel_size)
    texts = []
    lengths = []
    scene_names = []
    text_feats = None

    has_text_feat = "text_feat" in batch[0]
    if has_text_feat:
        feat_dim = batch[0]["text_feat"].shape[-1]
        text_feats = torch.zeros(B, 1, feat_dim)

    for i, item in enumerate(batch):
        L = item["length"]
        motion_padded[i, :L] = item["motion_features"][:L]
        motion_mask[i, :L] = True
        voxel_grids[i] = item["voxel_grid"]
        texts.append(item["text"])
        lengths.append(L)
        scene_names.append(item["scene_name"])
        if has_text_feat and "text_feat" in item:
            text_feats[i] = item["text_feat"]

    result = {
        "motion_features": motion_padded,
        "motion_mask": motion_mask,
        "voxel_grid": voxel_grids,
        "texts": texts,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "scene_names": scene_names,
    }

    if has_text_feat:
        result["text_feat"] = text_feats

    # External root for Stage2 Root-Guided SceneCo
    if "external_root" in batch[0]:
        T_max = max_len  # same as motion max_len
        ext_root_dim = batch[0]["external_root"].shape[-1]  # 5
        external_root = torch.zeros(B, T_max, ext_root_dim)
        external_root_sources = []
        for i, item in enumerate(batch):
            L = item["length"]
            external_root[i, :L] = item["external_root"][:L]
            external_root_sources.append(item.get("external_root_source", "unknown"))
        result["external_root"] = external_root
        result["external_root_source"] = external_root_sources

    return result
