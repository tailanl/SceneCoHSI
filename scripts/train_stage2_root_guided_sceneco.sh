#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models

python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_root_guided_sceneco.yaml \
  --gpu 1 \
  \
  --external_root_enabled true \
  --use_external_root true \
  --path_guided_root_dir outputs/guided_roots_train/path_only \
  --path_scene_guided_root_dir outputs/guided_roots_train/path_scene \
  --val_root_dir outputs/guided_roots_val/path_scene \
  --root_mix_gt 0.3 \
  --root_mix_path 0.3 \
  --root_mix_scene 0.4 \
  \
  --batch_size 4 \
  --num_epochs 400 \
  --lr 1e-4 \
  --prior_weight 0.0 \
  --scene_dropout 0.1 \
  --num_workers 4 \
  \
  2>&1 | tee outputs/stage2_root_guided_sceneco/train.log
