#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project

export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p outputs/e10_gt_projected_stage2_fair80
exec > >(tee -a outputs/e10_gt_projected_stage2_fair80/pipeline.log) 2>&1

date
echo "[E10 fair80] Train Stage2 for 80 epochs on already corrected GT roots"
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_gt_root_sceneco.yaml \
  --gpu 0 \
  --output_dir outputs/e10_gt_projected_stage2_fair80 \
  --path_guided_root_dir outputs/e10_gt_projected_train \
  --path_scene_guided_root_dir outputs/e10_gt_projected_train \
  --val_root_dir outputs/e10_gt_projected_val \
  --root_mix_gt 0.0 \
  --root_mix_path 0.0 \
  --root_mix_scene 1.0 \
  --num_epochs 80 \
  --batch_size 4 \
  --num_workers 4
date
