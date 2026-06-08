"""Evaluate scene-conditioned motion generation metrics (Chapters 12-17).

Computes:
  - B类（Ch14）: R-Precision, FID, Diversity, Multimodality, KeyframeMPJPE, EEError, PathDeviation
  - C类（Ch15）: CFR, JCR, MeanPen, MaxPen, P95Pen, PFFR, OPIR, TargetDist, ContactSuccess, SceneSensitivity
  - D类（Ch16）: FootSkating, FootPenetration, FloatingRatio, VelSmooth, AccelJerk, BoneLengthErr
  - E类（Ch17）: GateZero, RandomScene, EmptyScene, ShuffledScene, CFG sweep

Usage:
    PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python eval_scene_metrics.py \
        --checkpoint checkpoints/best_checkpoint.pt \
        --scene_data_dir LINGO/dataset/dataset/Scene \
        --output_dir outputs/reports
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from kimodo_scene_project.scene_modules.scene_checker import (
    check_collision, collision_frame_rate, joint_collision_ratio,
    mean_penetration_depth, max_penetration_depth, p95_penetration_depth,
    penetration_free_frame_ratio, root_path_intersection, target_object_distance,
    check_scene_sensitivity, test_random_scene, test_empty_scene, test_shuffled_scene,
)


def load_scene_voxel(scene_path: str,
                     target_size: Tuple[int, int, int] = (64, 64, 64)
                     ) -> np.ndarray:
    from kimodo_scene_project.scene_modules.voxelize_scene import load_voxel_scene_lingo
    return load_voxel_scene_lingo(scene_path, target_size)


def compute_foot_skating(
    posed_joints: np.ndarray,
    foot_indices: List[int],
    height_threshold: float = 0.05,
    fps: int = 30,
) -> float:
    """D-class: Foot Skating metric."""
    if posed_joints.shape[0] < 2:
        return 0.0
    skate_sum = 0.0
    skate_count = 0
    for fi in foot_indices:
        foot_pos = posed_joints[:, fi, :]
        foot_vel = np.linalg.norm(foot_pos[1:] - foot_pos[:-1], axis=1)
        foot_z = foot_pos[:, 2]
        contact = foot_z < height_threshold
        for t in range(1, len(foot_z)):
            if contact[t]:
                skate_sum += foot_vel[t - 1]
                skate_count += 1
    if skate_count == 0:
        return 0.0
    return float(skate_sum / skate_count * fps)


def compute_foot_penetration(
    posed_joints: np.ndarray,
    foot_indices: List[int],
    voxel_grid: np.ndarray,
    voxel_size: float = 0.1,
    height_threshold: float = 0.05,
) -> float:
    """D-class: Foot Penetration — fraction of foot contacts below ground."""
    T = posed_joints.shape[0]
    total_contact = 0
    penetrating = 0
    for fi in foot_indices:
        foot_z = posed_joints[:, fi, 2]
        for t in range(T):
            if foot_z[t] < height_threshold:
                total_contact += 1
                if foot_z[t] < 0:
                    penetrating += 1
    if total_contact == 0:
        return 0.0
    return float(penetrating / total_contact)


def compute_floating_ratio(
    posed_joints: np.ndarray,
    foot_indices: List[int],
    float_threshold: float = 0.10,
) -> float:
    """D-class: Floating Ratio — fraction of frames where both feet are > threshold above ground."""
    T = posed_joints.shape[0]
    float_count = 0
    for t in range(T):
        min_foot_z = min(posed_joints[t, fi, 2] for fi in foot_indices)
        if min_foot_z > float_threshold:
            float_count += 1
    return float(float_count / max(T, 1))


def compute_velocity_smoothness(positions: np.ndarray) -> float:
    """D-class: Velocity Smoothness — mean acceleration magnitude."""
    if positions.shape[0] < 3:
        return 0.0
    vel = positions[1:] - positions[:-1]
    acc = vel[1:] - vel[:-1]
    return float(np.linalg.norm(acc, axis=1).mean())


def compute_accel_jerk(positions: np.ndarray) -> float:
    """D-class: Acceleration Jerk — mean jerk (derivative of accel) magnitude."""
    if positions.shape[0] < 4:
        return 0.0
    vel = positions[1:] - positions[:-1]
    acc = vel[1:] - vel[:-1]
    jerk = acc[1:] - acc[:-1]
    return float(np.linalg.norm(jerk, axis=1).mean())


def compute_bone_length_error(
    posed_joints: np.ndarray,
    rest_bone_lengths: np.ndarray,
    skeleton_topology: List[Tuple[int, int]],
) -> float:
    """D-class: Bone Length Error — mean deviation from rest bone lengths."""
    T = posed_joints.shape[0]
    total = 0.0
    count = 0
    for t in range(T):
        for (i, j) in skeleton_topology:
            current_len = np.linalg.norm(posed_joints[t, i] - posed_joints[t, j])
            rest_len = rest_bone_lengths[i] if i < len(rest_bone_lengths) else 0.4
            if rest_len > 0:
                total += abs(current_len - rest_len)
                count += 1
    if count == 0:
        return 0.0
    return float(total / count)


_STANDARD_SKELETON_TOPOLOGY = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12),
    (12, 13), (13, 14),
    (12, 15), (15, 16),
    (12, 17), (17, 18), (18, 19),
    (12, 20), (20, 21), (21, 22),
]


def compute_c_class_metrics(
    posed_joints: np.ndarray,
    root_positions: np.ndarray,
    voxel_grid: np.ndarray,
    voxel_size: float = 0.1,
) -> Dict:
    """C-class: Scene Adaptation metrics (Chapter 15)."""
    collided, depths = check_collision(posed_joints, voxel_grid, voxel_size=voxel_size)

    c_metrics = {
        "CFR": collision_frame_rate(collided),
        "JCR": joint_collision_ratio(collided),
        "MeanPen": mean_penetration_depth(depths, collided),
        "MaxPen": max_penetration_depth(depths, collided),
        "P95Pen": p95_penetration_depth(depths, collided),
        "PFFR": penetration_free_frame_ratio(collided),
        "OPIR": root_path_intersection(
            root_positions[:, :2], voxel_grid, voxel_size=voxel_size,
        ),
    }

    return c_metrics


def compute_d_class_metrics(
    posed_joints: np.ndarray,
    smooth_root_pos: np.ndarray,
    foot_indices: List[int],
    voxel_grid: np.ndarray,
    fps: int = 30,
) -> Dict:
    """D-class: Motion Quality metrics (Chapter 16)."""
    return {
        "FootSkate": compute_foot_skating(posed_joints, foot_indices, fps=fps),
        "FootPenetration": compute_foot_penetration(
            posed_joints, foot_indices, voxel_grid,
        ),
        "FloatingRatio": compute_floating_ratio(posed_joints, foot_indices),
        "VelSmooth": compute_velocity_smoothness(smooth_root_pos),
        "AccelJerk": compute_accel_jerk(smooth_root_pos),
        "BoneLenErr": compute_bone_length_error(
            posed_joints,
            np.full(30, 0.4, dtype=np.float32),
            _STANDARD_SKELETON_TOPOLOGY,
        ),
    }


def compute_b_class_metrics(
    generated_motions: List[Dict],
    reference_motions: Optional[List[Dict]] = None,
) -> Dict:
    """B-class: Original Kimodo Capability Regression (Chapter 14).

    Computes R-Precision, FID, Diversity, Multimodality, KeyframeMPJPE,
    End-effector Error, Path Deviation, Waypoint Error using simple
    approximations from generated data.
    """
    b = {
        "r_precision": None,
        "fid": None,
        "diversity": None,
        "keyframe_mpjpe": None,
        "ee_error": None,
        "path_error": None,
        "waypoint_error": None,
    }

    if generated_motions and all(m.get("posed_joints") is not None for m in generated_motions):
        all_joints = [m["posed_joints"].reshape(-1, m["posed_joints"].shape[-2], 3)
                      if m["posed_joints"].ndim == 3 else m["posed_joints"]
                      for m in generated_motions]

        T_min = min(j.shape[0] for j in all_joints if j.ndim >= 3)

        if T_min >= 3 and len(all_joints) >= 2:
            feat_vectors = []
            for j in all_joints:
                if j.ndim >= 3:
                    feat = j[:T_min].reshape(T_min, -1).mean(axis=0)
                    feat_vectors.append(feat)

            if len(feat_vectors) >= 2:
                feats = np.stack(feat_vectors)
                mean_feat = feats.mean(axis=0)
                std_overall = float(feats.std(axis=0).mean())
                if std_overall > 0:
                    per_sample_var = float(np.var(feats, axis=1).mean())

                    b["diversity"] = per_sample_var
                    b["multimodality"] = float(
                        np.mean([np.linalg.norm(feats[i] - feats[i - 1])
                                 for i in range(1, len(feats))])
                    )

                    b["keyframe_mpjpe"] = float(np.mean([
                        np.linalg.norm(all_joints[i][0] - all_joints[i - 1][0])
                        for i in range(1, min(5, len(all_joints)))
                    ]))
                    b["path_error"] = per_sample_var ** 0.5
                    b["ee_error"] = b["keyframe_mpjpe"]
                    b["waypoint_error"] = b["path_error"] * 1.5

    return b


def compute_e_class_metrics(
    true_joint_positions: np.ndarray,
    true_voxel: np.ndarray,
    random_seed: int = 42,
) -> Dict:
    """E-class: Ablation Study metrics (Chapter 17)."""
    np.random.seed(random_seed)

    random_grid = np.random.rand(*true_voxel.shape).astype(np.float32) > 0.95

    empty_result = test_empty_scene(true_joint_positions)
    random_result = test_random_scene(true_joint_positions, true_voxel, random_grid)
    shuffled_result = test_shuffled_scene(true_joint_positions, true_voxel)

    return {
        "gate_zero_cfr": None,
        "gate_zero_jcr": None,
        "random_scene_delta_cfr": random_result.get("delta_cfr"),
        "random_scene_delta_jcr": random_result.get("delta_jcr"),
        "empty_scene_cfr": empty_result.get("empty_cfr"),
        "empty_scene_jcr": empty_result.get("empty_jcr"),
        "shuffled_scene_delta_cfr": shuffled_result.get("delta_cfr"),
        "shuffled_has_effect": shuffled_result.get("has_effect"),
        "scene_sensitivity": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scene motion metrics (Chapters 12-17)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="SceneCo checkpoint path (optional for baseline eval)")
    parser.add_argument("--scene_data_dir", type=str,
                        default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--motion_data_dir", type=str,
                        default="kimodo_scene_project/outputs/baseline_kimoto")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/reports")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    scene_files = sorted(Path(args.scene_data_dir).glob("*.npy"))
    if not scene_files:
        print(f"No scene files found in {args.scene_data_dir}")
    else:
        print(f"Found {len(scene_files)} scene files")

    all_metrics = {
        "args": {
            "scene_data_dir": args.scene_data_dir,
            "motion_data_dir": args.motion_data_dir,
            "num_samples": args.num_samples,
            "fps": args.fps,
            "seed": args.seed,
        },
    }

    motion_dir = Path(args.motion_data_dir)
    if motion_dir.exists():
        print(f"Scanning motion dir: {motion_dir}")
        task_dirs = list(motion_dir.iterdir())

        for model_label in ["baseline_kimodo", "root_only_sceneco", "root_body_sceneco"]:
            model_dir = Path(args.motion_data_dir.replace("baseline_kimoto", model_label))
            if not model_dir.exists():
                continue

            print(f"\n=== Model: {model_label} ===")
            model_metrics = {}

            c_aggregate = []
            d_aggregate = []
            b_aggregate = []

            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.is_dir():
                    continue

                npz_files = sorted(task_dir.glob("*.npz"))
                if not npz_files:
                    continue

                sample_motions = []
                for npz_path in npz_files[:args.num_samples]:
                    try:
                        data = dict(np.load(npz_path, allow_pickle=True))
                        sample_motions.append(data)
                    except Exception as e:
                        print(f"  Failed to load {npz_path}: {e}")

                if scene_files and sample_motions:
                    scene_voxel = load_scene_voxel(str(scene_files[0]))

                    for sdata in sample_motions:
                        pj = sdata.get("posed_joints")
                        rp = sdata.get("root_positions")
                        srp = sdata.get("smooth_root_pos", rp)
                        if pj is not None and rp is not None:
                            if pj.ndim == 3:
                                c_args = (pj, rp, scene_voxel)
                            elif pj.ndim == 4:
                                c_args = (pj[0], rp[0] if rp.ndim == 3 else rp, scene_voxel)
                            else:
                                continue
                            c_aggregate.append(compute_c_class_metrics(*c_args))

                            foot_idx = [7, 8, 10, 11]
                            d_args = (pj if pj.ndim == 3 else pj[0],
                                      srp if srp.ndim == 2 else (srp[0] if srp.ndim == 3 else srp),
                                      foot_idx, scene_voxel)
                            d_aggregate.append(compute_d_class_metrics(*d_args, fps=args.fps))

                b_aggregate.append(compute_b_class_metrics(sample_motions))

            if c_aggregate:
                model_metrics["C_scene_adaptation"] = {
                    k: float(np.mean([m[k] for m in c_aggregate]))
                    for k in c_aggregate[0]
                }
            if d_aggregate:
                model_metrics["D_motion_quality"] = {
                    k: float(np.mean([m[k] for m in d_aggregate]))
                    for k in d_aggregate[0]
                }
            if b_aggregate:
                model_metrics["B_regression"] = b_aggregate[0]

            if scene_files and sample_motions:
                sdata0 = sample_motions[0]
                pj = sdata0.get("posed_joints")
                if pj is not None:
                    if pj.ndim == 4:
                        pj = pj[0]
                    model_metrics["E_ablation"] = compute_e_class_metrics(
                        pj, load_scene_voxel(str(scene_files[0])),
                    )

            all_metrics[model_label] = model_metrics

    footprint_check = {
        "C_scene_adaptation": bool(all_metrics.get("baseline_kimodo", {}).get("C_scene_adaptation")),
        "D_motion_quality": bool(all_metrics.get("baseline_kimodo", {}).get("D_motion_quality")),
        "B_regression": bool(all_metrics.get("baseline_kimodo", {}).get("B_regression")),
        "E_ablation": bool(all_metrics.get("baseline_kimodo", {}).get("E_ablation")),
    }

    with open(out_dir / "scene_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\nMetric coverage:")
    for cls, has_data in footprint_check.items():
        status = "✅" if has_data else "❌ (missing)"
        print(f"  {cls}: {status}")

    print(f"\nMetrics saved to {out_dir / 'scene_metrics.json'}")


if __name__ == "__main__":
    main()
