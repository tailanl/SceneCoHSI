#!/bin/bash
# train_root_only.sh — Train Root-only SceneCo (GPU 0,1,2,3)
#
# Freezes all original Kimodo parameters and only trains:
#   - Voxel ViT scene encoder
#   - SceneCo cross-attention layers in root_model
#   - SceneCo gate parameters

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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

CONFIG="kimodo_scene_project/configs/sceneco_root_only.yaml"
OUTPUT_DIR="kimodo_scene_project/outputs/root_only_sceneco"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " Training Root-only SceneCo (GPU 0,1,2,3)"
echo "=============================================="
echo "Config:   $CONFIG"
echo "Output:   $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python kimodo_scene_project/train/train_sceneco.py "$CONFIG" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=============================================="
echo " Root-only SceneCo training complete!"
echo " Checkpoints: $OUTPUT_DIR/checkpoints/"
echo " Logs:        $OUTPUT_DIR/train.log"
