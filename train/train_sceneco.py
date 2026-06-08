"""Launch training for Kimodo-SceneCo. Maps config to args and delegates to kimodo_sceneco.train.train."""

import argparse
import os
import sys
from pathlib import Path

import yaml


def build_args_from_config(config_path: str) -> list:
    """Build command-line args list from a YAML config file."""
    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)

    args = []

    data = conf.get("data", {})
    args += ["--data_root", str(data.get("data_root", "LINGO/dataset"))]
    if data.get("cache_dir"):
        args += ["--cache_dir", str(data["cache_dir"])]
    args += ["--voxel_size", ",".join(map(str, data.get("voxel_size", [64, 64, 64])))]
    args += ["--max_frames", str(data.get("max_frames", 196))]
    args += ["--min_frames", str(data.get("min_frames", 40))]
    args += ["--fps", str(data.get("fps", 30))]

    training = conf.get("training", {})
    args += ["--pretrained_model", str(conf.get("pretrained_model", "Kimodo-SOMA-RP-v1.1"))]
    args += ["--output_dir", str(conf.get("output_dir", "outputs/sceneco"))]
    args += ["--batch_size", str(training.get("batch_size", 4))]
    args += ["--num_epochs", str(training.get("num_epochs", 100))]
    args += ["--lr", str(training.get("lr", 1e-4))]
    args += ["--weight_decay", str(training.get("weight_decay", 0.01))]
    args += ["--max_grad_norm", str(training.get("max_grad_norm", 1.0))]
    args += ["--prior_weight", str(training.get("prior_weight", 0.5))]
    args += ["--scene_dropout", str(training.get("scene_dropout", 0.1))]
    args += ["--num_base_steps", str(training.get("num_base_steps", 1000))]
    args += ["--accum_steps", str(training.get("accum_steps", 1))]
    args += ["--val_interval", str(training.get("val_interval", 500))]
    args += ["--val_max_batches", str(training.get("val_max_batches", 10))]
    args += ["--log_interval", str(training.get("log_interval", 50))]
    args += ["--num_workers", str(training.get("num_workers", 4))]
    args += ["--seed", str(training.get("seed", 42))]

    train_ratio = data.get("train_ratio", 0.9)
    args += ["--train_ratio", str(train_ratio)]

    if data.get("no_soma_conversion", False):
        args.append("--no_soma_conversion")

    if data.get("root_trajectory_data", False):
        args.append("--root_trajectory_data")

    if data.get("traj_data_dir"):
        args += ["--traj_data_dir", str(data["traj_data_dir"])]

    if training.get("freeze_pretrained", True):
        args.append("--freeze_pretrained")
    else:
        args.append("--no_freeze")

    sceneco = conf.get("sceneco", {})
    args += ["--use_in_root_model", str(sceneco.get("use_in_root_model", True))]
    args += ["--use_in_body_model", str(sceneco.get("use_in_body_model", True))]
    args += ["--sceneco_dropout", str(sceneco.get("dropout", 0.1))]
    args += ["--use_dual_vit", str(sceneco.get("use_dual_vit", True))]
    args += ["--root_voxel_mode", str(sceneco.get("root_voxel_mode", "full"))]

    trajco = conf.get("trajco", {})
    if trajco:
        args += ["--use_trajco", str(trajco.get("use_trajco", True))]
        args += ["--traj_dim", str(trajco.get("traj_dim", 5))]
        args += ["--trajco_dropout", str(trajco.get("trajco_dropout", 0.1))]
        args += ["--traj_loss_weight", str(trajco.get("traj_loss_weight", 1.0))]
        args += ["--traj_dropout", str(trajco.get("traj_dropout", 0.1))]
        args += ["--trajco_type", str(trajco.get("trajco_type", "additive"))]
        if trajco.get("use_trajco_root", False):
            args += ["--use_trajco_root", "true"]
        if trajco.get("use_trajco_body", False):
            args += ["--use_trajco_body", "true"]

    traj_corruption = conf.get("traj_corruption", {})
    if traj_corruption:
        if traj_corruption.get("enabled", False):
            args += ["--traj_corruption_enabled", "true"]
        args += ["--traj_corruption_waypoint_interval", str(traj_corruption.get("waypoint_interval", 30))]
        args += ["--traj_corruption_pos_noise_std", str(traj_corruption.get("pos_noise_std", 0.05))]
        args += ["--traj_corruption_global_shift_std", str(traj_corruption.get("global_shift_std", 0.05))]
        args += ["--traj_corruption_heading_noise_std", str(traj_corruption.get("heading_noise_std", 0.10))]
        args += ["--traj_corruption_recompute_heading", str(traj_corruption.get("recompute_heading_from_path", True))]

    gpu_ids = conf.get("experiment", {}).get("gpu_ids", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    args += ["--device", "cuda:0"]

    if "--resume" in sys.argv:
        args += ["--resume"]

    return args


def main():
    parser = argparse.ArgumentParser(description="Launch Kimodo-SceneCo training")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    train_args = build_args_from_config(str(config_path))

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kimodo"))
    from kimodo_sceneco.train.train import main as train_main

    sys.argv = ["train_sceneco.py"] + train_args
    train_main()


if __name__ == "__main__":
    main()
