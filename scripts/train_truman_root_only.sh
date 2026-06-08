#!/bin/bash
# train_truman_root_only.sh — Train Root-only SceneCo on TRUMAN data (NO SOMA)
#
# Uses TRUMAN 24-joint format directly without SOMA conversion.
# SceneCo cross-attention only in root_model.
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

export CUDA_VISIBLE_DEVICES="0"

CONFIG="kimodo_scene_project/configs/truman_root_only.yaml"
OUTPUT_DIR="kimodo_scene_project/outputs/truman_root_only"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " Training TRUMAN Root-only SceneCo (GPU 0)"
echo "=============================================="
echo "Config:   $CONFIG"
echo "Output:   $OUTPUT_DIR"
echo "Data:     TRUMAN (24-joint direct, no SOMA)"
echo "SceneCo:  root_model only"
echo "ViT:      dual ViT"
echo ""

PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python kimodo_scene_project/train/train_sceneco.py "$CONFIG" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=============================================="
echo " TRUMAN Root-only SceneCo training complete!"
echo " Checkpoints: $OUTPUT_DIR/checkpoints/"
echo " Logs:        $OUTPUT_DIR/train.log"
echo "=============================================="
