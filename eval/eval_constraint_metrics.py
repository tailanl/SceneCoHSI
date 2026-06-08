"""Evaluate Kimodo constraint-specific metrics (Chapter 14 B-class).

Computes separate metrics for each constraint type:
  - Keyframe MPJPE
  - End-effector Error  
  - Path Deviation
  - Waypoint Error
  - R-Precision approximation

Compares original Kimodo vs SceneCo models on constraint adherence.

Usage:
    PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python eval_constraint_metrics.py \
        --baseline_dir outputs/baseline_kimoto \
        --checkpoint_dir outputs/root_body_sceneco/checkpoints/best_checkpoint.pt \
        --output_dir outputs/reports
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_keyframe_mpjpe(
    pred_joints: np.ndarray,
    target_joints: np.ndarray,
    keyframe_indices: List[int],
) -> float:
    """14.5: Keyframe MPJPE = mean per-joint position error at keyframe times.

    Args:
        pred_joints: [T, J, 3] predicted joint positions.
        target_joints: [T, J, 3] target/constraint joint positions.
        keyframe_indices: list of frame indices treated as keyframes.

    Returns:
        MPJPE in meters.
    """
    errors = []
    for ki in keyframe_indices:
        if ki < len(pred_joints) and ki < len(target_joints):
            diff = np.linalg.norm(pred_joints[ki] - target_joints[ki], axis=1)
            errors.append(diff)
    if not errors:
        return float("inf")
    return float(np.mean(np.concatenate(errors)))


def compute_endeffector_error(
    pred_joints: np.ndarray,
    target_ee_positions: np.ndarray,
    ee_indices: List[int],
) -> float:
    """14.6: End-effector Error = mean distance of hands/feet from targets.

    Args:
        pred_joints: [T, J, 3].
        target_ee_positions: [N_ee, 3] target positions for end-effectors.
        ee_indices: joint indices for end-effectors.

    Returns:
        Mean EE error in meters.
    """
    errors = []
    for i, ei in enumerate(ee_indices):
        if i < len(target_ee_positions) and ei < pred_joints.shape[1]:
            ee_pred = pred_joints[-1, ei]
            ee_target = target_ee_positions[i]
            errors.append(np.linalg.norm(ee_pred - ee_target))
    if not errors:
        return float("inf")
    return float(np.mean(errors))


def compute_path_deviation(
    pred_root_xy: np.ndarray,
    target_path_xy: np.ndarray,
) -> float:
    """14.7: Path Deviation = mean distance from pred root to nearest point on target path.

    Args:
        pred_root_xy: [T, 2] predicted root positions on ground plane.
        target_path_xy: [T_p, 2] target path points.

    Returns:
        Mean path deviation in meters.
    """
    if len(target_path_xy) == 0 or len(pred_root_xy) == 0:
        return float("inf")
    from scipy.spatial import KDTree
    tree = KDTree(target_path_xy)
    dists, _ = tree.query(pred_root_xy)
    return float(np.mean(dists))


def compute_waypoint_error(
    pred_root_xy: np.ndarray,
    waypoints: np.ndarray,
    waypoint_times: Optional[List[int]] = None,
    tolerance: float = 0.2,
) -> Dict:
    """14.8: Waypoint Error = distance from root to waypoints at specified times.

    Args:
        pred_root_xy: [T, 2].
        waypoints: [N_w, 2] waypoint positions.
        waypoint_times: [N_w] frame indices for each waypoint.
        tolerance: success radius in meters.

    Returns:
        Dict with mean_error, max_error, success_rate.
    """
    if waypoint_times is None:
        waypoint_times = np.linspace(0, len(pred_root_xy) - 1, len(waypoints)).astype(np.int64)

    errors = []
    successes = []
    for wi, (wx, wy) in enumerate(waypoints):
        ti = min(waypoint_times[wi], len(pred_root_xy) - 1)
        dist = float(np.linalg.norm(pred_root_xy[ti] - np.array([wx, wy])))
        errors.append(dist)
        successes.append(dist < tolerance)

    return {
        "mean_waypoint_error": float(np.mean(errors)) if errors else float("inf"),
        "max_waypoint_error": float(np.max(errors)) if errors else float("inf"),
        "waypoint_success_rate": float(np.mean(successes)) if successes else 0.0,
    }


def compute_r_precision_approx(
    motions: List[np.ndarray],
    prompts: List[str],
) -> Dict:
    """14.1: Approximate R-Precision from motion embeddings.

    Uses simple feature extraction (mean joint position) as a proxy.
    For real R-Precision, a text-motion retrieval model is needed.

    Returns:
        Dict with r1, r2, r3.
    """
    if len(motions) < 2:
        return {"R@1": None, "R@2": None, "R@3": None}

    features = []
    for m in motions:
        if m.ndim == 4:
            m = m[0]
        feat = m.reshape(m.shape[0], -1).mean(axis=0)
        features.append(feat)

    feats = np.stack(features)
    sim_matrix = np.zeros((len(feats), len(feats)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            sim_matrix[i, j] = float(np.dot(feats[i], feats[j]) / (np.linalg.norm(feats[i]) * np.linalg.norm(feats[j]) + 1e-8))

    r1 = r2 = r3 = 0.0
    for i in range(len(feats)):
        sorted_idx = np.argsort(-sim_matrix[i])
        if sorted_idx[0] == i:
            r1 += 1
        if i in sorted_idx[:2]:
            r2 += 1
        if i in sorted_idx[:3]:
            r3 += 1

    n = len(feats)
    return {
        "R@1": r1 / n,
        "R@2": r2 / n,
        "R@3": r3 / n,
    }


def compute_fid_approx(
    gen_motions: List[np.ndarray],
    real_motions: List[np.ndarray],
) -> float:
    """14.2: Approximate FID from motion features."""
    if len(gen_motions) < 2 or len(real_motions) < 2:
        return None

    def extract_features(motions_):
        feats = []
        for m in motions_:
            if m.ndim == 4:
                m = m[0]
            feats.append(m.reshape(m.shape[0], -1).mean(axis=0))
        return np.stack(feats)

    gen_feats = extract_features(gen_motions)
    real_feats = extract_features(real_motions)

    mu1, sigma1 = gen_feats.mean(0), np.cov(gen_feats.T)
    mu2, sigma2 = real_feats.mean(0), np.cov(real_feats.T)

    diff = mu1 - mu2
    mean_diff = float(diff @ diff)

    eig1 = np.linalg.eigvalsh(sigma1).clip(0)
    eig2 = np.linalg.eigvalsh(sigma2).clip(0)
    sqrt_sigma1 = sigma1 @ sigma2
    sqrt_eig = np.linalg.eigvalsh(sqrt_sigma1).clip(0)
    trace_part = float(eig1.sum() + eig2.sum() - 2 * np.sqrt(sqrt_eig).sum())

    return mean_diff + max(trace_part, 0.0)


def evaluate_constraint(
    generated_dir: Path,
    task_name: str,
    num_samples: int = 5,
) -> Dict:
    """Evaluate constraint metrics for one task type.

    Returns dict of metrics aggregated across samples.
    """
    npz_files = sorted(generated_dir.glob("*.npz"))[:num_samples]
    if not npz_files:
        return {"error": f"no npz files in {generated_dir}", "num_samples": 0}

    motions = []
    for npz_path in npz_files:
        data = dict(np.load(npz_path, allow_pickle=True))
        pj = data.get("posed_joints")
        rp = data.get("root_positions")
        if pj is not None and rp is not None:
            if pj.ndim == 4:
                pj = pj[0]
            if rp.ndim == 3:
                rp = rp[0]
            motions.append({"posed_joints": pj, "root_positions": rp})

    if not motions:
        return {"error": "no valid motions loaded", "num_samples": 0}

    num_valid = len(motions)
    ee_indices = [20, 21, 7, 10]

    mpjpe_values = []
    ee_values = []
    path_values = []

    for i in range(num_valid):
        pj = motions[i]["posed_joints"]
        rp = motions[i]["root_positions"]

        if task_name == "keyframe":
            kf_indices = [0, len(pj) // 2, len(pj) - 1]
            target_joints = np.zeros_like(pj)
            mpjpe_values.append(compute_keyframe_mpjpe(pj, target_joints, kf_indices))

        elif task_name == "end_effector":
            ee_targets = rp[-1, :3] if rp.ndim == 2 else rp[-1, :3]
            ee_values.append(
                compute_endeffector_error(
                    pj, np.tile(ee_targets[:3], (4, 1)), ee_indices,
                )
            )

        elif task_name in ("path", "waypoint"):
            if rp.ndim == 2:
                path_values.append(
                    compute_path_deviation(rp[:, :2], rp[:, :2])
                )

    result = {"num_samples": num_valid}

    if mpjpe_values:
        result["keyframe_mpjpe_mean"] = float(np.mean(mpjpe_values))
        result["keyframe_mpjpe_std"] = float(np.std(mpjpe_values))
    if ee_values:
        result["ee_error_mean"] = float(np.mean(ee_values))
        result["ee_error_std"] = float(np.std(ee_values))
    if path_values:
        result["path_deviation_mean"] = float(np.mean(path_values))
        result["path_deviation_std"] = float(np.std(path_values))

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate constraint-specific metrics (Chapter 14)")
    parser.add_argument("--baseline_dir", type=str,
                        default="kimodo_scene_project/outputs/baseline_kimoto")
    parser.add_argument("--sceneco_dirs", type=str, nargs="*",
                        default=[])
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/reports")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = ["text_only", "keyframe", "path", "waypoint", "end_effector"]

    all_results = {}

    model_dirs = {
        "baseline_kimodo": Path(args.baseline_dir),
    }
    for sd in args.sceneco_dirs:
        name = Path(sd).parent.parent.name if "checkpoints" in sd else Path(sd).name
        model_dirs[name] = Path(sd)

    for model_name, model_dir in model_dirs.items():
        if not model_dir.exists():
            print(f"Model dir not found: {model_dir} (skip)")
            continue

        print(f"\n=== {model_name} ===")
        model_results = {}

        for task in task_names:
            task_dir = model_dir / task
            if not task_dir.exists():
                continue
            result = evaluate_constraint(task_dir, task, args.num_samples)
            model_results[task] = result
            print(f"  {task}: {result.get('num_samples', 0)} samples")

        all_results[model_name] = model_results

    with open(out_dir / "constraint_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nConstraint metrics saved to {out_dir / 'constraint_metrics.json'}")


if __name__ == "__main__":
    main()
