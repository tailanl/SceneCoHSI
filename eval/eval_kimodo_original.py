"""Evaluate original Kimodo baseline on text and constraint tasks (Chapter 5).

Generates motions for 5 modes:
  1. text-only
  2. keyframe constraint
  3. path constraint
  4. waypoint constraint
  5. end-effector constraint

Usage:
    PYTHONPATH="kimodo:SOMA:$PYTHONPATH" CHECKPOINT_DIR=models/Kimodo-SOMA-RP-v1.1 \
    HF_HOME=.hf_cache TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local \
    TEXT_ENCODER_DEVICE=cpu python eval_kimodo_original.py \
    --output_dir outputs/baseline_kimoto
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = "0"


def load_kimodo_model():
    from kimodo.model import load_model
    return load_model("Kimodo-SOMA-RP-v1.1", device="cuda")


def build_keyframe_constraint(prompt: str, num_frames: int, skeleton) -> Optional[list]:
    """Build keyframe constraint: set body pose at specific frames.
    
    Uses sampled body poses at t=0, t=T/2, t=T-1.
    """
    from kimodo.constraints import FullBodyConstraintSet
    
    keyframe_times = [0, num_frames // 2, num_frames - 1]
    keyframe_times = sorted(set(max(0, min(num_frames - 1, t)) for t in keyframe_times))
    
    t_tensor = torch.tensor(keyframe_times, dtype=torch.long)
    
    nb_joints = skeleton.nbjoints
    nb_frames = len(keyframe_times)
    
    keyframe_joints = torch.zeros(nb_frames, nb_joints, 3)
    keyframe_rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(nb_frames, nb_joints, 1, 1)
    keyframe_root_2d = torch.zeros(nb_frames, 2)
    
    for i, t in enumerate(keyframe_times):
        progress = t / max(1, num_frames - 1)
        keyframe_root_2d[i, 0] = progress * 1.5
        keyframe_root_2d[i, 1] = 0.0
    
    fc = FullBodyConstraintSet(
        skeleton, t_tensor,
        keyframe_joints, keyframe_rotmats, keyframe_root_2d,
    )
    return [fc]


def build_path_constraint(prompt: str, num_frames: int, skeleton) -> Optional[list]:
    """Build path constraint: guide root trajectory in 2D."""
    from kimodo.constraints import PathConstraintSet
    
    num_waypoints = min(5, num_frames)
    indices = np.linspace(0, num_frames - 1, num_waypoints).astype(np.int64)
    
    root_2d = np.zeros((num_waypoints, 2), dtype=np.float32)
    for i, t in enumerate(indices):
        progress = t / max(1, num_frames - 1)
        root_2d[i, 0] = progress * 1.5
        root_2d[i, 1] = np.sin(progress * np.pi) * 0.3
    
    pc = PathConstraintSet(
        skeleton,
        torch.from_numpy(indices),
        torch.from_numpy(root_2d),
    )
    return [pc]


def build_waypoint_constraint(prompt: str, num_frames: int, skeleton) -> Optional[list]:
    """Build waypoint constraint: root must pass through specific 2D points."""
    from kimodo.constraints import WaypointConstraintSet
    
    waypoints = np.array([
        [-0.5, 0.0],
        [0.0, 0.2],
        [0.5, -0.1],
    ], dtype=np.float32)
    
    wc = WaypointConstraintSet(
        skeleton,
        torch.from_numpy(waypoints),
        torch.tensor([1.0, 1.0, 1.0]),
        device=torch.device("cpu"),
    )
    return [wc]


def build_endeffector_constraint(prompt: str, num_frames: int, skeleton) -> Optional[list]:
    """Build end-effector constraint: guide hands/feet positions."""
    from kimodo.constraints import EndEffectorConstraintSet
    
    keyframe_times = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]
    keyframe_times = sorted(set(max(0, min(num_frames - 1, t)) for t in keyframe_times))
    
    t_tensor = torch.tensor(keyframe_times, dtype=torch.long)
    nb_frames = len(keyframe_times)
    
    pose = torch.zeros(nb_frames, 22, 3)
    rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(nb_frames, 22, 1, 1)
    root_2d = torch.zeros(nb_frames, 2)
    for i in range(nb_frames):
        root_2d[i, 0] = i / max(1, nb_frames - 1) * 1.5
    
    ee = EndEffectorConstraintSet(
        skeleton,
        t_tensor, pose, rot, root_2d,
        joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
    )
    return [ee]


_CONSTRAINT_BUILDERS = {
    "keyframe": build_keyframe_constraint,
    "path": build_path_constraint,
    "waypoint": build_waypoint_constraint,
    "end_effector": build_endeffector_constraint,
}


def run_eval_task(
    model, task_name: str, prompt: str, num_frames: int,
    seed: int, num_samples: int, skeleton,
) -> Tuple[List[Dict], List[Dict]]:
    """Run evaluation for a single prompt in a single task.
    
    Returns (outputs, configs).
    """
    set_seed(seed)
    
    outputs = []
    configs = []
    
    for sample_idx in range(num_samples):
        sample_seed = seed + sample_idx
        set_seed(sample_seed)
        
        try:
            if task_name == "text_only":
                output = model(
                    prompt,
                    num_frames=num_frames,
                    num_denoising_steps=50,
                    num_samples=1,
                    return_numpy=True,
                )
            else:
                builder = _CONSTRAINT_BUILDERS[task_name]
                constraints = builder(prompt, num_frames, model.skeleton)
                
                output = model(
                    prompt,
                    num_frames=num_frames,
                    num_denoising_steps=50,
                    num_samples=1,
                    constraint_lst=constraints,
                    return_numpy=True,
                )
            
            outputs.append({"sample": sample_idx, "seed": sample_seed, "data": output})
            configs.append({"sample": sample_idx, "seed": sample_seed, "prompt": prompt})
            
        except Exception as e:
            import traceback
            print(f"    Sample {sample_idx} FAILED: {e}")
            traceback.print_exc()
            
    return outputs, configs


def save_motion(output: Dict, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{prefix}_motion.npz",
        posed_joints=output.get("posed_joints"),
        root_positions=output.get("root_positions"),
        local_rot_mats=output.get("local_rot_mats"),
        global_rot_mats=output.get("global_rot_mats"),
        foot_contacts=output.get("foot_contacts"),
        global_root_heading=output.get("global_root_heading"),
        smooth_root_pos=output.get("smooth_root_pos", output.get("root_positions")),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate original Kimodo baseline (Chapter 5)")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/baseline_kimoto")
    parser.add_argument("--num_frames", type=int, default=196)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    prompts = [
        "walk forward",
        "sit down",
        "turn around",
        "wave hand",
        "run in a circle",
    ]

    tasks_def = {
        "text_only":       {"prompts": prompts, "needs_constraint": False},
        "keyframe":        {"prompts": prompts[:3], "needs_constraint": True},
        "path":            {"prompts": prompts[:3], "needs_constraint": True},
        "waypoint":        {"prompts": prompts[:3], "needs_constraint": True},
        "end_effector":    {"prompts": prompts[:3], "needs_constraint": True},
    }

    metrics = {
        "seed": args.seed,
        "num_frames": args.num_frames,
        "num_samples": args.num_samples,
        "model": "Kimodo-SOMA-RP-v1.1",
        "tasks": {},
    }

    print("Loading Kimodo model...")
    model = load_kimodo_model()
    model.eval()

    for task_name, task_config in tasks_def.items():
        print(f"\n=== Task: {task_name} ===")
        task_dir = out_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        task_summary = {"prompts": [], "total_saved": 0, "total_failed": 0}

        for prompt_idx, prompt in enumerate(task_config["prompts"]):
            print(f"  Prompt: '{prompt}'")

            outputs_list, configs_list = run_eval_task(
                model, task_name, prompt,
                args.num_frames, args.seed + prompt_idx,
                num_samples=args.num_samples,
                skeleton=model.skeleton,
            )

            saved = 0
            for out_item in outputs_list:
                sample_idx = out_item["sample"]
                prefix = f"{task_name}_{prompt_idx:02d}_{prompt.replace(' ', '_')}_s{sample_idx}"
                save_motion(out_item["data"], task_dir, prefix)
                saved += 1

            failed = args.num_samples - saved
            task_summary["prompts"].append({
                "prompt": prompt,
                "saved": saved,
                "failed": failed,
            })
            task_summary["total_saved"] += saved
            task_summary["total_failed"] += failed

        print(f"  Saved: {task_summary['total_saved']}, Failed: {task_summary['total_failed']}")
        metrics["tasks"][task_name] = task_summary

    with open(out_dir / "baseline_config.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "num_frames": args.num_frames,
            "num_samples": args.num_samples,
            "model": "Kimodo-SOMA-RP-v1.1",
            "prompts": prompts,
        }, f, indent=2)

    with open(out_dir / "metrics_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBaseline evaluation complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
