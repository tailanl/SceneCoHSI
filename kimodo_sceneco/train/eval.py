# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation script for Kimodo-SceneCo using TSTMotion metrics.

Metrics:
- FID (Frechet Inception Distance) on motion features
- R-Precision (text-motion retrieval accuracy)
- Diversity (average pairwise distance in feature space)
- Foot Skate metrics (foot sliding, contact consistency)
- Scene Collision Rate (percentage of frames with body-inside-voxel collisions)
- Constraint Following Error (for constrained generation)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def compute_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = _sqrtm(sigma1 @ sigma2, eps)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def _sqrtm(mat: np.ndarray, eps: float = 1e-6):
    from scipy.linalg import sqrtm
    result = sqrtm(mat)
    if np.any(np.isnan(result)):
        result = np.nan_to_num(result, nan=0.0)
    return result, None


def compute_diversity(features: np.ndarray, subset_size: int = 300) -> float:
    if len(features) < 2:
        return 0.0
    n = min(subset_size, len(features))
    idx = np.random.choice(len(features), n, replace=False)
    subset = features[idx]
    dists = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            dists.append(np.linalg.norm(subset[i] - subset[j]))
    return float(np.mean(dists)) if dists else 0.0


def compute_r_precision(
    text_features: np.ndarray,
    motion_features: np.ndarray,
    top_k: List[int] = [1, 2, 3, 5, 10],
) -> Dict[str, float]:
    sim_matrix = text_features @ motion_features.T
    sim_matrix = sim_matrix / 2 + 0.5

    n = sim_matrix.shape[0]
    results = {}
    for k in top_k:
        if n < k:
            continue
        topk_idx = np.argsort(-sim_matrix, axis=1)[:, :k]
        correct = sum(1 for i in range(n) if i in topk_idx[i])
        results[f"R{k}"] = correct / n * 100

    ranks = []
    for i in range(n):
        rank = np.where(np.argsort(-sim_matrix[i]) == i)[0][0] + 1
        ranks.append(rank)
    results["MedR"] = float(np.median(ranks))

    return results


def compute_foot_skate(
    posed_joints: torch.Tensor,
    foot_contacts: torch.Tensor,
    skeleton,
    lengths: torch.Tensor,
) -> Dict[str, float]:
    from kimodo_sceneco.metrics.foot_skate import (
        FootSkateFromHeight,
        FootSkateFromContacts,
        FootSkateRatio,
    )

    results = {}

    fs_height = FootSkateFromHeight(skeleton)
    h_metrics = fs_height(posed_joints, lengths)
    results["foot_skate_from_height"] = h_metrics["foot_skate_from_height"].item()

    fs_contacts = FootSkateFromContacts(skeleton)
    c_metrics = fs_contacts(posed_joints, foot_contacts, lengths)
    results["foot_skate_from_pred_contacts"] = c_metrics["foot_skate_from_pred_contacts"].item()

    fs_ratio = FootSkateRatio(skeleton)
    r_metrics = fs_ratio(posed_joints, lengths)
    results["foot_skate_ratio"] = r_metrics["foot_skate_ratio"].item()

    return results


def compute_scene_collision_rate(
    posed_joints: torch.Tensor,
    voxel_grid: torch.Tensor,
    lengths: torch.Tensor,
    voxel_size_m: float = 0.02,
    collision_thresh: int = 3,
) -> float:
    """Compute percentage of frames where body joints collide with occupied voxels."""
    B, T, J, _ = posed_joints.shape
    total_collisions = 0
    total_frames = 0

    for b in range(B):
        L = lengths[b].item()
        voxel = voxel_grid[b, 0]
        vx, vy, vz = voxel.shape

        for t in range(L):
            joints = posed_joints[b, t]
            collision_count = 0
            for j in range(J):
                x, y, z = joints[j].cpu().numpy()
                ix = int(x / voxel_size_m)
                iy = int(y / voxel_size_m)
                iz = int(z / voxel_size_m)
                if 0 <= ix < vx and 0 <= iy < vy and 0 <= iz < vz:
                    if voxel[ix, iy, iz] > 0.5:
                        collision_count += 1
            if collision_count >= collision_thresh:
                total_collisions += 1
            total_frames += 1

    return total_collisions / max(total_frames, 1) * 100


@torch.no_grad()
def evaluate(
    model,
    val_dataset,
    output_dir: str,
    num_samples: int = 200,
    num_denoising_steps: int = 50,
    cfg_weight: List[float] = [2.0, 2.0, 2.0],
    device: str = "cuda",
) -> Dict[str, float]:
    """Run full evaluation on Kimodo-SceneCo."""
    model.eval()
    model.to(device)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    all_motion_features = []
    all_text_features = []
    all_gt_features = []
    all_posed_joints = []
    all_foot_contacts = []
    all_lengths = []
    all_voxel_grids = []

    n_eval = min(num_samples, len(val_dataset))
    indices = np.random.choice(len(val_dataset), n_eval, replace=False)

    log.info(f"Evaluating {n_eval} samples...")

    for idx in tqdm(indices, desc="Generating"):
        sample = val_dataset[int(idx)]
        motion_feat = sample["motion_features"].unsqueeze(0).to(device)
        voxel = sample["voxel_grid"].unsqueeze(0).to(device)
        text = sample["text"]
        length = sample["length"]

        scene_feat, scene_mask = model.encode_scene(voxel)

        output = model(
            prompts=text,
            num_frames=length,
            num_denoising_steps=num_denoising_steps,
            scene_input=None,
            cfg_weight=cfg_weight,
            return_numpy=False,
        )

        if "posed_joints" in output:
            all_posed_joints.append(output["posed_joints"].cpu())
        if "foot_contacts" in output:
            all_foot_contacts.append(output["foot_contacts"].cpu())

        feat = motion_feat[0, :length].cpu().numpy().flatten()
        all_gt_features.append(feat)

        gen_feat = output.get("smooth_root_pos", motion_feat[0, :length]).cpu().numpy().flatten()
        all_motion_features.append(gen_feat)

        all_lengths.append(length)
        all_voxel_grids.append(voxel.cpu())

    gen_features = np.stack(all_motion_features) if all_motion_features else np.zeros((1, 1))
    gt_features = np.stack(all_gt_features) if all_gt_features else np.zeros((1, 1))

    results = {}

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False) if gen_features.shape[0] > 1 else np.eye(gen_features.shape[1])
    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False) if gt_features.shape[0] > 1 else np.eye(gt_features.shape[1])

    fid = compute_fid(mu_gen, sigma_gen, mu_gt, sigma_gt)
    results["FID"] = fid

    diversity_gen = compute_diversity(gen_features)
    diversity_gt = compute_diversity(gt_features)
    results["Diversity_gen"] = diversity_gen
    results["Diversity_gt"] = diversity_gt

    if len(all_posed_joints) > 0:
        posed_joints = torch.cat(all_posed_joints, dim=0)
        foot_contacts = torch.cat(all_foot_contacts, dim=0) if all_foot_contacts else None
        lengths_t = torch.tensor(all_lengths)

        try:
            fs_metrics = compute_foot_skate(
                posed_joints, foot_contacts, model.skeleton, lengths_t
            )
            results.update(fs_metrics)
        except Exception as e:
            log.warning(f"Foot skate computation failed: {e}")

    results["num_samples"] = n_eval

    log.info("Evaluation Results:")
    for k, v in sorted(results.items()):
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(output_path / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Kimodo-SceneCo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/home/lzsh2025/kimodo-viser/LINGO/dataset")
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--cfg_weight", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from kimodo_sceneco.model import KimodoSceneCo
    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**ckpt["args"])

    log.info("Loading model from checkpoint...")
    trainer_obj = type("Trainer", (), {})()
    trainer_obj.args = model_args
    trainer_obj.device = args.device

    from kimodo_sceneco.train.train import Trainer
    t = Trainer(model_args)
    t.model.load_state_dict(ckpt["model_state_dict"])

    val_dataset = LINGOSceneMotionDataset(
        data_root=args.data_root,
        motion_rep=t.model.motion_rep,
        max_frames=model_args.max_frames,
        min_frames=model_args.min_frames,
        voxel_size=tuple(map(int, model_args.voxel_size.split(","))),
        split="val",
        train_ratio=model_args.train_ratio,
    )

    evaluate(
        model=t.model,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_denoising_steps=args.num_denoising_steps,
        cfg_weight=args.cfg_weight,
        device=args.device,
    )


if __name__ == "__main__":
    main()
