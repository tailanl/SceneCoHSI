#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project

export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p outputs/e10_gt_projected_stage2
exec > >(tee -a outputs/e10_gt_projected_stage2/pipeline.log) 2>&1

date
echo "[E10] Postprocess GT train roots"
python scripts/postprocess_root_raw3d.py \
  --input_dir outputs/e7_gt_root_v3_train \
  --output_dir outputs/e10_gt_projected_train \
  --project_target_path \
  --overwrite_root_keys \
  --update_norm \
  --clearance_m 0.04 \
  --smooth_window 5 \
  --gpu 0

echo "[E10] Postprocess GT val roots"
python scripts/postprocess_root_raw3d.py \
  --input_dir outputs/e7_gt_root_v3_val \
  --output_dir outputs/e10_gt_projected_val \
  --project_target_path \
  --overwrite_root_keys \
  --update_norm \
  --clearance_m 0.04 \
  --smooth_window 5 \
  --gpu 0

echo "[E10] Train Stage2 for 60 epochs"
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_gt_root_sceneco.yaml \
  --gpu 0 \
  --output_dir outputs/e10_gt_projected_stage2 \
  --path_guided_root_dir outputs/e10_gt_projected_train \
  --path_scene_guided_root_dir outputs/e10_gt_projected_train \
  --val_root_dir outputs/e10_gt_projected_val \
  --root_mix_gt 0.0 \
  --root_mix_path 0.0 \
  --root_mix_scene 1.0 \
  --num_epochs 60 \
  --batch_size 4 \
  --num_workers 4

date
