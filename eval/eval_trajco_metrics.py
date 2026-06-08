"""Comprehensive metric evaluation for TrajCo experiments (SMPLX format).

Evaluates 7 experiments across:
  - C-class (Scene): CFR, JCR, MeanPen, MaxPen, P95Pen, PFFR, OPIR
  - D-class (Motion Quality): FootSkate, FootPen, Floating, VelSmooth, AccelJerk, BoneLenErr
  - T-class (Trajectory): RootMSE, RootRMSE, HeadingError, PathLengthRatio, PathCurvatureError

Usage:
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH="kimodo:SOMA:$PYTHONPATH" \
    CHECKPOINT_DIR=models HF_HOME=.hf_cache \
    TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu \
    python kimodo_scene_project/eval/eval_trajco_metrics.py \
    --num_samples 10 --output_dir kimodo_scene_project/outputs/eval_trajco --gpu 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

# SMPLX 22-joint foot indices: LeftFoot=10, RightFoot=11
FOOT_INDICES = [10, 11]

EXPERIMENT_CONFIGS = {
    "A_SceneCo": {
        "ckpt": "kimodo_scene_project/outputs/smplx_root_body/checkpoints/best_checkpoint.pt",
        "description": "Plan A: SceneCo root+body (baseline)",
        "use_in_root": True, "use_in_body": True,
        "has_trajco": False,
    },
    "B_TrajCo_add": {
        "ckpt": "kimodo_scene_project/outputs/trajco_smplx/checkpoints/best_checkpoint.pt",
        "description": "Plan B: TrajCo additive root+body",
        "use_in_root": False, "use_in_body": False,
        "has_trajco": True, "trajco_type": "additive",
    },
    "C_SceneCo_TrajCo_add": {
        "ckpt": "kimodo_scene_project/outputs/trajco_sceneco_smplx/checkpoints/best_checkpoint.pt",
        "description": "Plan C: SceneCo + TrajCo additive root+body",
        "use_in_root": True, "use_in_body": True,
        "has_trajco": True, "trajco_type": "additive",
    },
    "D_TrajCo_cross": {
        "ckpt": "kimodo_scene_project/outputs/trajco_cross_smplx/checkpoints/best_checkpoint.pt",
        "description": "Plan D: TrajCo cross-attn root+body",
        "use_in_root": False, "use_in_body": False,
        "has_trajco": True, "trajco_type": "cross_attn",
    },
    "E_SceneCo_TrajCo_cross": {
        "ckpt": "kimodo_scene_project/outputs/trajco_cross_sceneco_smplx/checkpoints/best_checkpoint.pt",
        "description": "Plan E: SceneCo + TrajCo cross-attn root+body",
        "use_in_root": True, "use_in_body": True,
        "has_trajco": True, "trajco_type": "cross_attn",
    },
    "F_S_body_T_root": {
        "ckpt": "kimodo_scene_project/outputs/trajco_cross_root_sceneco_body/checkpoints/best_checkpoint.pt",
        "description": "Plan F: SceneCo body + TrajCo cross-attn root",
        "use_in_root": False, "use_in_body": True,
        "has_trajco": True, "trajco_type": "cross_attn",
    },
    "G_S_all_T_root": {
        "ckpt": "kimodo_scene_project/outputs/trajco_cross_root_sceneco_all/checkpoints/best_checkpoint.pt",
        "description": "Plan G: SceneCo root+body + TrajCo cross-attn root",
        "use_in_root": True, "use_in_body": True,
        "has_trajco": True, "trajco_type": "cross_attn",
    },
}


def build_model(exp_name, device, device_str):
    from kimodo.model.load_model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    class PrecomputedTextEncoder:
        def __init__(self):
            self.llm_dim = 4096
            self.max_len = 77
            self.output_dim = 4096
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Text encoder should not be called — use precomputed text_feat")
        def to(self, device):
            return self
        def train(self, mode=True):
            return self
        def eval(self):
            return self

    cfg = EXPERIMENT_CONFIGS[exp_name]
    print(f"  Building {exp_name} ({cfg['description']})...")

    pretrained = load_model(
        "Kimodo-SMPLX-RP-v1", device="cpu",
        text_encoder=PrecomputedTextEncoder(),
    )
    inner_denoiser = pretrained.denoiser
    if hasattr(inner_denoiser, "model"):
        inner_denoiser = inner_denoiser.model

    use_trajco = cfg.get("has_trajco", False)
    trajco_type = cfg.get("trajco_type", "additive")
    use_trajco_root = use_trajco and cfg["use_in_root"] is False and exp_name != "B_TrajCo_add" and exp_name != "D_TrajCo_cross"

    if exp_name in ("F_S_body_T_root", "G_S_all_T_root"):
        use_trajco_val = False
        use_trajco_root_val = True
        use_trajco_body_val = False
    elif exp_name in ("B_TrajCo_add", "D_TrajCo_cross"):
        use_trajco_val = True
        use_trajco_root_val = False
        use_trajco_body_val = False
    else:
        use_trajco_val = use_trajco
        use_trajco_root_val = False
        use_trajco_body_val = False

    model = KimodoSceneCo(
        denoiser=inner_denoiser,
        text_encoder=pretrained.text_encoder,
        num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64),
            "patch_size": (8, 8, 8),
            "d_model": 256,
            "num_layers": 4,
            "use_dual_vit": False,
            "root_voxel_mode": "full",
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=cfg["use_in_root"],
        use_in_body_model=cfg["use_in_body"],
        use_trajco=use_trajco_val,
        use_trajco_root=use_trajco_root_val,
        use_trajco_body=use_trajco_body_val,
        traj_dim=5,
        trajco_type=trajco_type,
    )
    model = model.to(device)
    model.eval()

    ckpt_path = cfg["ckpt"]
    if not Path(ckpt_path).exists():
        print(f"  WARNING: checkpoint not found: {ckpt_path}")
        return None
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model


def denormalize_root(normalized, mean, std):
    return normalized * std + mean


def load_val_samples(num_samples, seed=42):
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton.definitions import SMPLXSkeleton22

    cache_dir = Path("lingo_smplx_cache")
    traj_cache_dir = Path("lingo_root_trajectory_smplx")

    skeleton = SMPLXSkeleton22()
    stats_path = Path("models") / "Kimodo-SMPLX-RP-v1" / "stats" / "motion"
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=30, stats_path=str(stats_path))

    stats_dir = Path("models") / "Kimodo-SMPLX-RP-v1" / "stats" / "motion" / "global_root"
    mean_root = np.load(stats_dir / "mean.npy")
    std_root = np.load(stats_dir / "std.npy")
    std_root = np.where(std_root < 1e-8, 1.0, std_root)

    cache_files = sorted(cache_dir.glob("seg_*.npz"))
    rng = np.random.RandomState(seed)
    n = min(num_samples, len(cache_files))
    indices = rng.choice(len(cache_files), size=n, replace=False)

    samples = []
    for idx in indices:
        data = np.load(str(cache_files[idx]), allow_pickle=True)
        seg_idx = int(cache_files[idx].stem.split("_")[1])

        voxel = data["voxel_grid"]
        motion_features = data["motion_features"]
        T = int(data["length"])
        motion_features = motion_features[:T]

        traj_file = traj_cache_dir / f"seg_{seg_idx:05d}.npz"
        if traj_file.exists():
            traj_data = np.load(str(traj_file), allow_pickle=True)
            gt_root_norm = traj_data["global_root_features"][:T, :3]
            gt_root = denormalize_root(gt_root_norm, mean_root[:3], std_root[:3])
        else:
            gt_root = None

        samples.append({
            "text": str(data["text"]) if "text" in data else "no-text",
            "num_frames": T,
            "voxel": voxel,
            "scene_name": str(data.get("scene_name", f"scene_{seg_idx}")),
            "motion_features": motion_features,
            "gt_root": gt_root,
            "text_feat": data.get("text_feat"),
        })
    return samples


def generate_motion(model, sample, device, args):
    voxel = sample["voxel"]
    if isinstance(voxel, np.ndarray):
        voxel = torch.from_numpy(voxel).float()
    voxel = voxel.to(device)
    if voxel.ndim == 4:
        voxel = voxel.unsqueeze(1)
    elif voxel.ndim == 3:
        voxel = voxel.unsqueeze(0).unsqueeze(1)

    traj_input = None
    if model.has_trajco:
        root_slice = slice(0, 5)
        traj = sample["motion_features"][:, root_slice]
        if isinstance(traj, np.ndarray):
            traj = torch.from_numpy(traj).float()
        traj = traj.to(device)
        traj_input = traj.unsqueeze(0)

    with torch.no_grad():
        output = model(
            prompts=sample["text"],
            num_frames=sample["num_frames"],
            num_denoising_steps=args.num_denoising_steps,
            cfg_weight=[3.0, 1.5, 2.0],
            cfg_type="scene_separated",
            scene_input=voxel,
            traj_input=traj_input,
            text_feat=(torch.from_numpy(sample["text_feat"]).float().to(device)
                       if sample["text_feat"] is not None else None),
            return_numpy=True,
        )
    return output


SMPLX_22_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]


def compute_scene_metrics(posed_joints, root_positions, voxel_grid, rest_bone_lengths):
    from kimodo_scene_project.eval.eval_scene_metrics import (
        compute_c_class_metrics,
        compute_foot_skating, compute_foot_penetration,
        compute_floating_ratio, compute_velocity_smoothness,
        compute_accel_jerk, compute_bone_length_error,
    )

    pj = np.squeeze(posed_joints)
    rp = np.squeeze(root_positions)

    c = compute_c_class_metrics(pj, rp, voxel_grid)

    d = {
        "FootSkate": compute_foot_skating(pj, FOOT_INDICES, fps=30),
        "FootPenetration": compute_foot_penetration(pj, FOOT_INDICES, voxel_grid),
        "FloatingRatio": compute_floating_ratio(pj, FOOT_INDICES),
        "VelSmooth": compute_velocity_smoothness(rp),
        "AccelJerk": compute_accel_jerk(rp),
        "BoneLenErr": compute_bone_length_error(pj, rest_bone_lengths, SMPLX_22_CONNECTIONS),
    }

    return {**c, **d}


def compute_trajectory_metrics(gen_root, gt_root):
    """T-class: Trajectory quality metrics.

    Args:
        gen_root: (T, 3) generated smooth root positions
        gt_root: (T, 3) ground truth smooth root positions
    Returns:
        dict with RootMSE, RootRMSE, HeadingError, PathLengthRatio
    """
    T = min(gen_root.shape[0], gt_root.shape[0])
    gen = gen_root[:T]
    gt = gt_root[:T]

    per_frame_mse = np.mean((gen - gt) ** 2)
    root_mse = np.mean((gen - gt) ** 2, axis=0)
    root_rmse = np.sqrt(root_mse)

    gen_diff = gen[1:] - gen[:-1]
    gt_diff = gt[1:] - gt[:-1]
    eps = 1e-8

    gen_heading = np.arctan2(gen_diff[:, 2], gen_diff[:, 0])
    gt_heading = np.arctan2(gt_diff[:, 2], gt_diff[:, 0])
    heading_diff = np.abs(gen_heading - gt_heading)
    heading_diff = np.minimum(heading_diff, 2 * np.pi - heading_diff)
    heading_error = np.mean(heading_diff)

    gen_path_len = np.sum(np.linalg.norm(gen_diff, axis=1))
    gt_path_len = np.sum(np.linalg.norm(gt_diff, axis=1))
    path_length_ratio = gen_path_len / (gt_path_len + eps)

    gen_curvature = np.linalg.norm(gen_diff[1:] - gen_diff[:-1], axis=1)
    gt_curvature = np.linalg.norm(gt_diff[1:] - gt_diff[:-1], axis=1)
    curvature_error = np.mean(np.abs(gen_curvature - gt_curvature))

    return {
        "PerFrameMSE": float(per_frame_mse),
        "RootRMSE_X": float(root_rmse[0]),
        "RootRMSE_Y": float(root_rmse[1]),
        "RootRMSE_Z": float(root_rmse[2]),
        "RootRMSE_Mean": float(np.mean(root_rmse)),
        "HeadingError": float(heading_error),
        "PathLengthRatio": float(path_length_ratio),
        "CurvatureError": float(curvature_error),
    }


def aggregate_metrics(per_sample):
    metric_keys = [
        "CFR", "JCR", "MeanPen", "MaxPen", "P95Pen", "PFFR", "OPIR",
        "FootSkate", "FootPenetration", "FloatingRatio",
        "VelSmooth", "AccelJerk", "BoneLenErr",
        "PerFrameMSE", "RootRMSE_X", "RootRMSE_Y", "RootRMSE_Z",
        "RootRMSE_Mean", "HeadingError", "PathLengthRatio", "CurvatureError",
    ]
    agg = {}
    for k in metric_keys:
        vals = [s[k] for s in per_sample if s.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
        agg[f"{k}_std"] = float(np.std(vals)) if vals else None
    agg["avg_gen_time_s"] = float(np.mean([s["gen_time_s"] for s in per_sample]))
    return agg


def print_table(all_results, out_dir):
    exp_names = [k for k in all_results if not k.startswith("_")]

    c_metrics = ["CFR", "JCR", "MeanPen", "MaxPen", "P95Pen", "PFFR", "OPIR"]
    d_metrics = ["FootSkate", "FootPenetration", "FloatingRatio", "VelSmooth", "AccelJerk", "BoneLenErr"]
    t_metrics = ["PerFrameMSE", "RootRMSE_Mean", "HeadingError", "PathLengthRatio", "CurvatureError"]
    all_m = c_metrics + d_metrics + t_metrics

    header = f"{'Experiment':28s}" + "".join(f" | {m:>11s}" for m in all_m)
    sep = "-" * len(header)
    lines = [f"\n{'='*len(header)}", "TrajCo Experiment Metrics Comparison",
             f"{'='*len(header)}", header, sep]

    for exp_name in exp_names:
        agg = all_results[exp_name]["aggregated"]
        row = f"{exp_name:28s}"
        for m in all_m:
            v = agg.get(m)
            row += f" | {v:11.4f}" if v is not None else f" | {'---':>11s}"
        lines.append(row)

    lines.append(sep)
    lines.append("C-class (Scene): CFR↓ JCR↓ MeanPen↓ MaxPen↓ P95Pen↓ PFFR↑ OPIR↓")
    lines.append("D-class (Quality): FootSkate↓ FootPen↓ Float↓ VelSmooth↓ AccelJerk↓ BoneLen↓")
    lines.append("T-class (Trajectory): PerFrameMSE↓ RootRMSE_Mean↓ Heading↓ PathLenRatio(1=best) CurvErr↓")
    lines.append("=" * len(header))

    table = "\n".join(lines)
    print(table)

    with open(out_dir / "metric_table.txt", "w") as f:
        f.write(table)

    csv_lines = ["experiment," + ",".join(all_m)]
    for exp_name in exp_names:
        agg = all_results[exp_name]["aggregated"]
        vals = [f"{agg.get(m, ''):.6f}" if agg.get(m) is not None else "" for m in all_m]
        csv_lines.append(f"{exp_name}," + ",".join(vals))
    with open(out_dir / "metric_table.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/eval_trajco")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", type=str, nargs="*", default=None,
                        help="Specific experiments to evaluate (default: all)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_names = args.experiments or list(EXPERIMENT_CONFIGS.keys())
    print(f"Experiments ({len(exp_names)}):\n  " + "\n  ".join(
        f"{n}: {EXPERIMENT_CONFIGS[n]['description']}" for n in exp_names))

    samples = load_val_samples(args.num_samples, seed=args.seed)
    print(f"\nLoaded {len(samples)} val samples:")
    for i, s in enumerate(samples):
        print(f"  [{i}] '{s['text'][:50]}' ({s['num_frames']}f, scene={s['scene_name']})")

    neutral_joints = torch.load(Path("kimodo/kimodo_sceneco/assets/skeletons/smplx22/joints.p")).numpy()
    rest_bone_lengths = np.zeros(22, dtype=np.float32)
    for (i, j) in SMPLX_22_CONNECTIONS:
        length = np.linalg.norm(neutral_joints[i] - neutral_joints[j])
        rest_bone_lengths[i] = max(rest_bone_lengths[i], length)
        rest_bone_lengths[j] = max(rest_bone_lengths[j], length)
    rest_bone_lengths = np.where(rest_bone_lengths < 1e-6, 0.4, rest_bone_lengths)

    all_results = {}

    for exp_name in exp_names:
        cfg = EXPERIMENT_CONFIGS[exp_name]
        print(f"\n{'='*60}")
        print(f">>> {exp_name}: {cfg['description']}")
        print(f"{'='*60}")

        model = build_model(exp_name, device, device_str)
        if model is None:
            print(f"  SKIP: model loading failed")
            continue

        per_sample = []
        for si, sample in enumerate(tqdm(samples, desc=f"  {exp_name}")):
            t0 = time.time()
            output = generate_motion(model, sample, device, args)
            gen_time = time.time() - t0

            voxel_np = sample["voxel"]
            if hasattr(voxel_np, 'numpy'):
                voxel_np = voxel_np.numpy()

            scene_metrics = compute_scene_metrics(
                output.get("posed_joints"),
                output.get("root_positions"),
                voxel_np,
                rest_bone_lengths,
            )

            gt_root = sample["gt_root"]
            if gt_root is not None:
                gen_root = np.squeeze(output.get("smooth_root_pos", output.get("root_positions")))
                traj_metrics = compute_trajectory_metrics(gen_root, gt_root)
            else:
                traj_metrics = {}

            entry = {
                "sample_idx": si,
                "scene": sample["scene_name"],
                "text": sample["text"],
                "num_frames": sample["num_frames"],
                "gen_time_s": gen_time,
                **scene_metrics,
                **traj_metrics,
            }
            per_sample.append(entry)

        del model
        torch.cuda.empty_cache()

        agg = aggregate_metrics(per_sample)
        all_results[exp_name] = {"per_sample": per_sample, "aggregated": agg}

        print(f"  CFR={agg['CFR']:.4f} MeanPen={agg['MeanPen']:.4f} "
              f"FootSkate={agg['FootSkate']:.4f} RootRMSE={agg['RootRMSE_Mean']:.4f} "
              f"HeadingErr={agg['HeadingError']:.4f} PathLenRatio={agg['PathLengthRatio']:.4f}")

    all_results["_config"] = {
        "num_samples": len(samples),
        "num_denoising_steps": args.num_denoising_steps,
        "experiments": exp_names,
        "seed": args.seed,
    }

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print_table(all_results, out_dir)
    print(f"\nDone. -> {out_dir}/")
    print(f"  all_metrics.json | metric_table.csv | metric_table.txt")


if __name__ == "__main__":
    main()
