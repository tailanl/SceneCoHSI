#!/bin/bash
set -euo pipefail

# ============================================================
# Stage2 SceneCo Pipeline: Classifier-Guided Root → Training
# Usage: bash scripts/run_stage2_pipeline.sh
# ============================================================

export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1

GPU=0                    # CUDA_VISIBLE_DEVICES maps GPU 2 → index 0
CFG_ROOT="configs/root_classifier_guidance.yaml"
CFG_STAGE2="configs/stage2_root_guided_sceneco.yaml"
DIR_TRAIN="outputs/guided_roots_train/path_scene"
DIR_VAL="outputs/guided_roots_val/path_scene"

cd "$(dirname "$0")/.."  # → kimodo_scene_project/

echo "============================================================"
echo "Step 1/3: Generate classifier-guided roots (TRAIN split)"
echo "Output dir: $DIR_TRAIN"
echo "============================================================"
python scripts/generate_root_classifier_guidance.py \
  --config "$CFG_ROOT" \
  --output_dir "$DIR_TRAIN" \
  --split train --all --skip_existing \
  --gpu $GPU

echo ""
echo "============================================================"
echo "Step 2/3: Generate classifier-guided roots (VAL split)"
echo "Output dir: $DIR_VAL"
echo "============================================================"
python scripts/generate_root_classifier_guidance.py \
  --config "$CFG_ROOT" \
  --output_dir "$DIR_VAL" \
  --split val --all --skip_existing \
  --gpu $GPU

echo ""
echo "============================================================"
echo "Step 3/3: Stage2 SceneCo Training"
echo "Config: $CFG_STAGE2"
echo "============================================================"
exec python train/train_stage2_root_guided_sceneco.py \
  "$CFG_STAGE2" \
  --gpu $GPU \
  --path_scene_guided_root_dir "$DIR_TRAIN" \
  --val_root_dir "$DIR_VAL"
