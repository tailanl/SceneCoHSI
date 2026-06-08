#!/bin/bash
# train_truman_root_body.sh — Train Root+Body SceneCo on TRUMAN data (NO SOMA)
#
# Uses TRUMAN 24-joint format directly without SOMA conversion.
# SceneCo in BOTH root_model and body_model.
# Dual ViT: separate VoxelViT encoders for root and body.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONHASHSEED=0

export CHECKPOINT_DIR="models/Kimodo-SOMA-RP-v1.1"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

export CUDA_VISIBLE_DEVICES="0,1"

CONFIG="kimodo_scene_project/configs/truman_root_body.yaml"
OUTPUT_DIR="kimodo_scene_project/outputs/truman_root_body"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " Training TRUMAN Root+Body SceneCo (GPU 0,1)"
echo "=============================================="
echo "Config:   $CONFIG"
echo "Output:   $OUTPUT_DIR"
echo "Data:     TRUMAN (24-joint direct, no SOMA)"
echo "SceneCo:  root_model + body_model"
echo "ViT:      dual ViT (separate encoders)"
echo ""

PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python kimodo_scene_project/train/train_sceneco.py "$CONFIG" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=============================================="
echo " TRUMAN Root+Body SceneCo training complete!"
echo " Checkpoints: $OUTPUT_DIR/checkpoints/"
echo " Logs:        $OUTPUT_DIR/train.log"
echo "=============================================="
