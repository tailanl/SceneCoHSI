"""Launch CaKey+SceneCo training. Maps YAML config to args and delegates to training.

Two stages:
  Stage 1 (CaKey): Train CaKey layers only, freeze Kimodo backbone.
                    Randomly samples keyframes, trains inbetweening stability.
  Stage 2 (SceneCo): Load Stage 1 checkpoint, freeze CaKey, train VoxelViT + SceneCo.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def build_stage1_args_from_config(config_path: str) -> list:
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
    args += ["--output_dir", str(conf.get("output_dir", "outputs/cakey"))]
    args += ["--batch_size", str(training.get("batch_size", 4))]
    args += ["--total_steps", str(training.get("total_steps", 200000))]
    args += ["--warmup_steps", str(training.get("warmup_steps", 5000))]
    args += ["--lr", str(training.get("lr", 1e-4))]
    args += ["--weight_decay", str(training.get("weight_decay", 0.01))]
    args += ["--max_grad_norm", str(training.get("max_grad_norm", 1.0))]
    args += ["--num_base_steps", str(training.get("num_base_steps", 1000))]
    args += ["--accum_steps", str(training.get("accum_steps", 1))]
    args += ["--val_interval", str(training.get("val_interval", 500))]
    args += ["--val_max_batches", str(training.get("val_max_batches", 10))]
    args += ["--log_interval", str(training.get("log_interval", 50))]
    args += ["--num_workers", str(training.get("num_workers", 4))]
    args += ["--seed", str(training.get("seed", 42))]
    args += ["--train_ratio", str(data.get("train_ratio", 0.9))]

    cakey = conf.get("cakey", {})
    args += ["--use_cakey_root", str(cakey.get("use_cakey_root", True))]
    args += ["--use_cakey_body", str(cakey.get("use_cakey_body", False))]
    args += ["--cakey_hidden_dim", str(cakey.get("hidden_dim", 2048))]
    args.append("--stage1_cakey")

    keyframe = conf.get("keyframe_sampling", {})
    args += ["--kf_always_first_frame", str(keyframe.get("always_first_frame", True))]
    args += ["--kf_always_last_frame", str(keyframe.get("always_last_frame", True))]
    args += ["--kf_stride_choices", ",".join(map(str, keyframe.get("stride_choices", [20, 30, 40])))]
    args += ["--kf_random_drop", str(keyframe.get("random_drop_middle_keyframes", 0.1))]

    gpu_ids = conf.get("experiment", {}).get("gpu_ids", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    args += ["--device", "cuda:0"]

    return args


def build_stage2_args_from_config(config_path: str) -> list:
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
    args += ["--output_dir", str(conf.get("output_dir", "outputs/cakey_sceneco"))]
    args += ["--batch_size", str(training.get("batch_size", 4))]
    args += ["--total_steps", str(training.get("total_steps", 200000))]
    args += ["--warmup_steps", str(training.get("warmup_steps", 5000))]
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
    args += ["--train_ratio", str(data.get("train_ratio", 0.9))]

    sceneco = conf.get("sceneco", {})
    args += ["--use_in_root_model", str(sceneco.get("use_in_root_model", True))]
    args += ["--use_in_body_model", str(sceneco.get("use_in_body_model", True))]
    args += ["--sceneco_dropout", str(sceneco.get("dropout", 0.1))]
    args += ["--use_dual_vit", str(sceneco.get("use_dual_vit", True))]
    args += ["--root_voxel_mode", str(sceneco.get("root_voxel_mode", "full"))]

    cakey = conf.get("cakey", {})
    args += ["--use_cakey_root", str(cakey.get("use_cakey_root", True))]
    args += ["--use_cakey_body", str(cakey.get("use_cakey_body", False))]
    args += ["--cakey_hidden_dim", str(cakey.get("hidden_dim", 2048))]

    stage1_ckpt = conf.get("stage1_checkpoint", "")
    if stage1_ckpt:
        args += ["--stage1_checkpoint", str(stage1_ckpt)]
    args += ["--freeze_cakey", str(training.get("freeze_cakey", True))]
    args.append("--stage2_sceneco")

    gpu_ids = conf.get("experiment", {}).get("gpu_ids", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    args += ["--device", "cuda:0"]

    return args


def main():
    parser = argparse.ArgumentParser(description="Launch CaKey-SceneCo training")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2"],
                        help="Training stage: stage1=CaKey, stage2=SceneCo")
    args_in = parser.parse_args()

    config_path = Path(args_in.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    if args_in.stage == "stage1":
        train_args = build_stage1_args_from_config(str(config_path))
    else:
        train_args = build_stage2_args_from_config(str(config_path))

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kimodo"))
    from kimodo_scene_project.train.train_cakey import main as train_main

    sys.argv = ["train_cakey_sceneco.py"] + train_args
    train_main()


if __name__ == "__main__":
    main()
