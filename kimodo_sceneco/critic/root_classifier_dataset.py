"""Dataset utilities for training the true RootPathClassifier.

The classifier is trained on meter-space root trajectories decoded from
normalized Kimodo motion features. EnergyGuidance remains separate in
``kimodo_sceneco.guidance.root_guidance``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from kimodo_sceneco.guidance.scene_guidance import build_2d_sdf


MOTION_KEYS = ("motion_features", "motion", "beta_motion", "data")
NEGATIVE_MODES = (
    "shift",
    "wrong_goal",
    "jitter",
    "wrong_heading",
    "reverse_heading",
    "path_shuffle",
    "scene_collision",
)


def find_cache_files(cache_dir: str | Path, split: str | None = None) -> list[Path]:
    cache_dir = Path(cache_dir)
    candidates = [cache_dir]
    if split:
        candidates.insert(0, cache_dir / split)
    candidates.extend([cache_dir / "train", cache_dir / "val"])

    npz_files: list[Path] = []
    pt_files: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            npz_files.extend(sorted(candidate.glob("*.npz")))
            pt_files.extend(sorted(candidate.glob("*.pt")))

    files = sorted(set(npz_files)) + sorted(set(pt_files))
    if not files:
        raise FileNotFoundError(f"No .npz or .pt cache files found in {cache_dir}")

    if split and not (cache_dir / split).exists():
        n = len(files)
        split_idx = max(1, int(n * 0.9))
        if split == "train":
            files = files[:split_idx]
        elif split in {"val", "valid", "validation"}:
            files = files[split_idx:] or files[-1:]

    return files


def load_motion_features(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            for key in MOTION_KEYS:
                if key in data:
                    return np.asarray(data[key], dtype=np.float32)
            raise KeyError(f"No motion feature entry in {path}; keys={list(data.keys())}")

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        for key in MOTION_KEYS:
            if key in data:
                value = data[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                return np.asarray(value, dtype=np.float32)
        keys = list(data.keys()) if isinstance(data, dict) else []
        raise KeyError(f"No motion feature entry in {path}; keys={keys}")

    raise ValueError(f"Unsupported cache file: {path}")


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _available_keys(value) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys())
    return [f"<non-dict output: {type(value).__name__}>"]


def _inverse_motion(motion_rep, unnormalized: torch.Tensor):
    inverse_calls = (
        {
            "is_normalized": False,
            "posed_joints_from": "positions",
            "return_numpy": True,
        },
        {"is_normalized": False, "return_numpy": True},
        {"is_normalized": False},
    )
    last_error: TypeError | None = None
    for kwargs in inverse_calls:
        try:
            return motion_rep.inverse(unnormalized, **kwargs)
        except TypeError as exc:
            last_error = exc
    raise TypeError(
        "motion_rep.inverse could not be called with unnormalized features; "
        f"last TypeError: {last_error}"
    )


def _heading_to_cos_sin(heading: np.ndarray) -> np.ndarray:
    if heading.shape[-1] == 2:
        return heading
    if heading.shape[-1] == 1:
        angle = heading[..., 0]
        return np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    raise ValueError(f"Expected heading with last dim 1 or 2, got shape {heading.shape}")


def extract_root_5d_meter(motion_rep, features_np: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
    """Decode normalized Kimodo features to meter-space root_5d.

    Args:
        motion_rep: Kimodo motion representation with ``unnormalize`` and ``inverse``.
        features_np: ``(T, D)`` normalized Kimodo features.

    Returns:
        ``(T, 5)`` float32 ``[smooth_root_pos(x,y,z), heading_cos, heading_sin]``.
    """
    if motion_rep is None:
        raise ValueError("motion_rep is required; classifier input must be meter-space root")

    feat = torch.from_numpy(np.asarray(features_np, dtype=np.float32)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        unnorm = motion_rep.unnormalize(feat)
        out = _inverse_motion(motion_rep, unnorm)

    if not isinstance(out, dict):
        raise KeyError(
            "motion_rep.inverse output must be a dict containing 'smooth_root_pos' "
            "and heading keys. "
            f"Available keys: {_available_keys(out)}"
        )

    if "smooth_root_pos" in out:
        smooth_root_pos = _to_numpy(out["smooth_root_pos"])[0]
    else:
        raise KeyError(
            "motion_rep.inverse output is missing 'smooth_root_pos'. "
            f"Available keys: {_available_keys(out)}"
        )

    if "global_root_heading" in out:
        heading = _to_numpy(out["global_root_heading"])[0]
    elif "root_heading" in out:
        heading = _to_numpy(out["root_heading"])[0]
    else:
        raise KeyError(
            "motion_rep.inverse output is missing 'global_root_heading' or 'root_heading'. "
            f"Available keys: {_available_keys(out)}"
        )

    heading_cos_sin = _heading_to_cos_sin(heading)
    return np.concatenate([smooth_root_pos, heading_cos_sin], axis=-1).astype(np.float32)


def make_negative_root_numpy(
    root_5d: np.ndarray,
    mode: str,
    scene_sdf: np.ndarray | None = None,
    voxel_size: float = 0.1,
    grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    shift_scale: float = 0.8,
    wrong_goal_scale: float = 1.2,
    jitter_scale: float = 0.15,
) -> np.ndarray:
    root = root_5d.copy()
    T = root.shape[0]

    if mode == "shift":
        root[:, [0, 2]] += np.random.randn(1, 2).astype(np.float32) * shift_scale
    elif mode == "wrong_goal":
        drift = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
        root[:, [0, 2]] += drift * (np.random.randn(1, 2).astype(np.float32) * wrong_goal_scale)
    elif mode == "jitter":
        root[:, [0, 2]] += np.random.randn(T, 2).astype(np.float32) * jitter_scale
    elif mode == "wrong_heading":
        theta = np.random.rand(T).astype(np.float32) * 2.0 * np.pi
        root[:, 3] = np.cos(theta)
        root[:, 4] = np.sin(theta)
    elif mode == "reverse_heading":
        root[:, 3:5] *= -1.0
    elif mode == "path_shuffle":
        order = np.random.permutation(T)
        root[:, [0, 2]] = root[order][:, [0, 2]]
    elif mode == "scene_collision":
        if scene_sdf is None:
            root[:, [0, 2]] += np.random.randn(1, 2).astype(np.float32) * shift_scale
        else:
            sdf = np.asarray(scene_sdf, dtype=np.float32)
            ix, iz = np.unravel_index(np.argmin(sdf), sdf.shape)
            target_x = grid_origin[0] + float(ix) * voxel_size
            target_z = grid_origin[2] + float(iz) * voxel_size
            center = root[:, [0, 2]].mean(axis=0)
            shift = np.asarray([target_x, target_z], dtype=np.float32) - center
            root[:, [0, 2]] += shift[None]
    else:
        raise ValueError(f"Unknown negative mode: {mode}")

    return root


def pad_to_length(array: np.ndarray, max_frames: int, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    width = width or array.shape[-1]
    T = min(array.shape[0], max_frames)
    padded = np.zeros((max_frames, width), dtype=np.float32)
    mask = np.zeros((max_frames,), dtype=bool)
    padded[:T] = array[:T]
    mask[:T] = True
    return padded, mask


class RootClassifierDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        motion_rep,
        split: str = "train",
        positive_ratio: float = 0.5,
        max_frames: int = 196,
        negative_modes: Iterable[str] | None = None,
        use_scene_sdf: bool = False,
        scene_dir: str | Path | None = None,
        sdf_voxel_size: float = 0.1,
        sdf_grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__()
        self.files = find_cache_files(cache_dir, split=split)
        self.motion_rep = motion_rep
        self.split = split
        self.positive_ratio = positive_ratio
        self.max_frames = max_frames
        self.negative_modes = list(
            negative_modes or NEGATIVE_MODES
        )
        unknown_modes = sorted(set(self.negative_modes) - set(NEGATIVE_MODES))
        if unknown_modes:
            raise ValueError(f"Unknown negative mode(s): {unknown_modes}; valid modes={list(NEGATIVE_MODES)}")
        self.use_scene_sdf = use_scene_sdf
        self.scene_dir = Path(scene_dir) if scene_dir else None
        self.sdf_voxel_size = sdf_voxel_size
        self.sdf_grid_origin = sdf_grid_origin

    def __len__(self) -> int:
        return len(self.files)

    def _load_root_5d(self, file_idx: int) -> np.ndarray:
        features = load_motion_features(self.files[file_idx])
        root_5d = extract_root_5d_meter(self.motion_rep, features, device="cpu")
        return root_5d[: self.max_frames]

    def _load_scene_sdf(self, file_idx: int) -> np.ndarray | None:
        if not self.use_scene_sdf:
            return None
        path = self.files[file_idx]
        if path.suffix != ".npz":
            return None
        with np.load(path, allow_pickle=True) as data:
            if "voxel_grid" not in data:
                return None
            voxel = torch.from_numpy(np.asarray(data["voxel_grid"], dtype=np.float32))
        sdf = build_2d_sdf(
            voxel,
            voxel_size=self.sdf_voxel_size,
            grid_origin=self.sdf_grid_origin,
            device=torch.device("cpu"),
        )
        return sdf.cpu().numpy().astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        file_idx = idx % len(self.files)
        root_5d = self._load_root_5d(file_idx)
        scene_sdf = self._load_scene_sdf(file_idx)
        T = min(root_5d.shape[0], self.max_frames)
        root_5d = root_5d[:T]
        target_path_xz = root_5d[:, [0, 2]].copy()

        label = 1.0
        negative_mode = "none"

        if random.random() > self.positive_ratio:
            negative_mode = random.choice(self.negative_modes)
            label = 0.0
            if negative_mode == "path_shuffle" and len(self.files) > 1:
                other_idx = random.randrange(len(self.files) - 1)
                if other_idx >= file_idx:
                    other_idx += 1
                other_root = self._load_root_5d(other_idx)
                T_pair = min(T, other_root.shape[0], self.max_frames)
                root_5d = root_5d[:T_pair]
                target_path_xz = other_root[:T_pair, [0, 2]].copy()
            else:
                root_5d = make_negative_root_numpy(
                    root_5d,
                    negative_mode,
                    scene_sdf=scene_sdf,
                    voxel_size=self.sdf_voxel_size,
                    grid_origin=self.sdf_grid_origin,
                )

        pad_root, pad_mask = pad_to_length(root_5d, self.max_frames, width=5)
        pad_path, _ = pad_to_length(target_path_xz, self.max_frames, width=2)

        item = {
            "root_5d": pad_root.astype(np.float32),
            "target_path_xz": pad_path.astype(np.float32),
            "pad_mask": pad_mask.astype(bool),
            "label": np.float32(label),
            "negative_mode": negative_mode,
            "source_file": str(self.files[file_idx]),
        }
        if self.use_scene_sdf:
            if scene_sdf is None:
                scene_sdf = np.zeros((64, 64), dtype=np.float32)
            item["scene_sdf"] = scene_sdf.astype(np.float32)
        return item


def collate_root_classifier(batch: list[dict]) -> dict:
    result = {
        "root_5d": torch.from_numpy(np.stack([item["root_5d"] for item in batch])).float(),
        "target_path_xz": torch.from_numpy(np.stack([item["target_path_xz"] for item in batch])).float(),
        "pad_mask": torch.from_numpy(np.stack([item["pad_mask"] for item in batch])).bool(),
        "label": torch.from_numpy(np.asarray([item["label"] for item in batch], dtype=np.float32)),
        "negative_mode": [item["negative_mode"] for item in batch],
        "source_file": [item["source_file"] for item in batch],
    }
    if "scene_sdf" in batch[0]:
        result["scene_sdf"] = torch.from_numpy(
            np.stack([item["scene_sdf"] for item in batch])
        ).float()
    return result


root_classifier_collate_fn = collate_root_classifier
