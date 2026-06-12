#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p outputs/e9_hybrid_raw3d_stage2_scenefix60
exec > >(tee -a outputs/e9_hybrid_raw3d_stage2_scenefix60/pipeline.log) 2>&1

date
python scripts/postprocess_root_raw3d.py --input_dir outputs/e6_hybrid_guidance_train/path_only --output_dir outputs/e9_hybrid_raw3d_train_scenefix --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0
python scripts/postprocess_root_raw3d.py --input_dir outputs/e6_hybrid_guidance_val/path_only --output_dir outputs/e9_hybrid_raw3d_val_scenefix --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0
python train/train_stage2_root_guided_sceneco.py configs/stage2_hybrid_root_guided_sceneco.yaml --gpu 0 --output_dir outputs/e9_hybrid_raw3d_stage2_scenefix60 --path_guided_root_dir outputs/e9_hybrid_raw3d_train_scenefix --path_scene_guided_root_dir outputs/e9_hybrid_raw3d_train_scenefix --val_root_dir outputs/e9_hybrid_raw3d_val_scenefix --root_mix_gt 0.3 --root_mix_path 0.0 --root_mix_scene 0.7 --num_epochs 60 --batch_size 4 --num_workers 4
date
