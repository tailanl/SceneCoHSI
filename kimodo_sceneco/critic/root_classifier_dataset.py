"""Dataset for training RootPathSceneClassifier.

Produces (root_5d, target_path_xz, scene_sdf, pad_mask, label) pairs.
Positive = GT root + matching path  (label=1)
Negative = perturbed root + correct path (label=0)
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset


def make_negative_root(root_5d: torch.Tensor, mode: str) -> torch.Tensor:
    root = root_5d.clone()
    B, T, _ = root.shape

    if mode == "shift":
        offset = torch.randn(B, 1, 2, device=root.device) * 0.8
        root[..., [0, 2]] += offset

    elif mode == "wrong_goal":
        drift = torch.linspace(0, 1, T, device=root.device).view(1, T, 1)
        wrong_offset = torch.randn(B, 1, 2, device=root.device) * 1.2
        root[..., [0, 2]] += drift * wrong_offset

    elif mode == "jitter":
        noise = torch.randn(B, T, 2, device=root.device) * 0.15
        root[..., [0, 2]] += noise

    elif mode == "wrong_heading":
        theta = torch.rand(B, T, device=root.device) * 2.0 * torch.pi
        root[..., 3] = torch.cos(theta)
        root[..., 4] = torch.sin(theta)

    elif mode == "reverse_heading":
        root[..., 3:5] = -root[..., 3:5]

    else:
        raise ValueError(f"Unknown negative mode: {mode}")

    return root


def sample_negative_mode():
    return random.choice([
        "shift",
        "wrong_goal",
        "jitter",
        "wrong_heading",
        "reverse_heading",
    ])


class RootClassifierDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        positive_ratio: float = 0.5,
        max_frames: int = 196,
        negative_modes=None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.split = split
        self.positive_ratio = positive_ratio
        self.max_frames = max_frames
        self.negative_modes = negative_modes or [
            "shift",
            "wrong_goal",
            "jitter",
            "wrong_heading",
            "reverse_heading",
            "path_shuffle",
        ]

        # Load cached motion data (same as lingo_smplx_cache)
        self.files = self._load_file_list()

        # Pre-compute number of positives/negatives
        self.num_positives = len(self.files)
        self.num_negatives = self.num_positives  # same count, neg sampled on-the-fly

        self.total_length = self.num_positives + self.num_negatives

    def _load_file_list(self):
        cache_path = os.path.join(self.cache_dir)
        files = sorted([f for f in os.listdir(cache_path) if f.endswith(".npz")])
        # Simple train/val split: last 10% as val
        n = len(files)
        split_idx = int(n * 0.9)
        if self.split == "train":
            return files[:split_idx]
        else:
            return files[split_idx:]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self.num_positives == 0:
            raise IndexError("RootClassifierDataset has no cache files")

        make_positive = random.random() < self.positive_ratio
        if make_positive:
            # Positive sample
            file_idx = idx % self.num_positives
            label = 1
            neg_mode = None
        else:
            # Negative sample
            file_idx = idx % self.num_positives
            label = 0
            neg_mode = random.choice(self.negative_modes)

        root_5d = self._load_root_5d(file_idx)

        # target_path = root xz (GT path), unless path_shuffle replaces it.
        target_path_xz = root_5d[:, [0, 2]].clone()

        # pad_mask
        T = root_5d.shape[0]
        pad_mask = torch.ones(T, dtype=torch.bool)

        # Apply negative perturbation
        if label == 0:
            if neg_mode == "path_shuffle" and self.num_positives > 1:
                other_idx = random.randrange(self.num_positives - 1)
                if other_idx >= file_idx:
                    other_idx += 1
                other_root = self._load_root_5d(other_idx)
                T_pair = min(root_5d.shape[0], other_root.shape[0])
                root_5d = root_5d[:T_pair]
                target_path_xz = other_root[:T_pair, [0, 2]].clone()
                pad_mask = torch.ones(T_pair, dtype=torch.bool)
            else:
                root_5d_neg = make_negative_root(root_5d.unsqueeze(0), neg_mode)
                root_5d = root_5d_neg.squeeze(0)

        return {
            "root_5d": root_5d,              # (T, 5)
            "target_path_xz": target_path_xz, # (T, 2)
            "pad_mask": pad_mask,             # (T,)
            "label": label,                   # 0 or 1
            "negative_mode": neg_mode,        # str or None
            "file": self.files[file_idx],
        }

    def _load_root_5d(self, file_idx: int) -> torch.Tensor:
        filepath = os.path.join(self.cache_dir, self.files[file_idx])
        if filepath.endswith(".npz"):
            data = dict(np.load(filepath, allow_pickle=True))
            # npz stores each key as a separate array; find the motion key
            motion = None
            for key in ["motion", "beta_motion", "data", "motion_features"]:
                if key in data:
                    motion = data[key]
                    break
            if motion is None:
                keys = list(data.keys())
                motion = data[keys[0]]
        else:
            data = torch.load(filepath, weights_only=False)
            motion = data["motion"] if "motion" in data else data.get("beta_motion")
            if motion is None:
                motion = data["data"] if "data" in data else data.get("motion_features")
        if motion is None:
            raise KeyError(
                f"{self.files[file_idx]} has no motion-like tensor"
            )
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)

        T = min(motion.shape[0], self.max_frames)
        return motion[:T, :5].float().clone()


def collate_root_classifier(batch):
    # Pad to max length
    max_len = max(item["root_5d"].shape[0] for item in batch)
    B = len(batch)

    root_5d = torch.zeros(B, max_len, 5)
    target_path_xz = torch.zeros(B, max_len, 2)
    pad_mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    neg_modes = []

    for i, item in enumerate(batch):
        T = item["root_5d"].shape[0]
        root_5d[i, :T] = item["root_5d"]
        target_path_xz[i, :T] = item["target_path_xz"]
        pad_mask[i, :T] = item["pad_mask"]
        labels[i] = item["label"]
        neg_modes.append(item["negative_mode"])

    return {
        "root_5d": root_5d,
        "target_path_xz": target_path_xz,
        "pad_mask": pad_mask,
        "label": labels,
        "negative_mode": neg_modes,
    }
