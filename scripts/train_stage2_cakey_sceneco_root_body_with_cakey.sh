#!/bin/bash
# train_stage2_cakey_sceneco_root_body_with_cakey.sh
# Stage 2: SceneCo + CaKey joint training (root+body, dual ViT) from Stage 1 CaKey checkpoint.
# Freezes Kimodo backbone only. CaKey + SceneCo + VoxelViT all trainable in both branches.
# Uses GPU 4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models/Kimodo-SOMA-RP-v1.1"
export HF_HOME=".hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

STAGE2_CONFIG="kimodo_scene_project/configs/stage2_cakey_sceneco_root_body_with_cakey.yaml"
STAGE2_OUTPUT="kimodo_scene_project/outputs/cakey_sceneco_root_body_with_cakey"
STAGE1_CKPT="kimodo_scene_project/outputs/cakey_root_body/checkpoints/best_checkpoint.pt"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    exit 1
fi

echo "=============================================="
echo " Stage 2: SceneCo + CaKey Root+Body (joint train, dual ViT)"
echo "=============================================="
echo "Config:    $STAGE2_CONFIG"
echo "Output:    $STAGE2_OUTPUT"
echo "Stage1 CK: $STAGE1_CKPT"
echo "GPU:       4"
echo ""

mkdir -p "$STAGE2_OUTPUT"

export CUDA_VISIBLE_DEVICES="1"
PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}" python kimodo_scene_project/train/train_cakey_sceneco.py "$STAGE2_CONFIG" --stage stage2 \
    2>&1 | tee "$STAGE2_OUTPUT/train.log"

echo ""
echo "=============================================="
echo " DONE: SceneCo + CaKey Root+Body joint training complete."
echo " Checkpoint: $STAGE2_OUTPUT/checkpoints/best_checkpoint.pt"
echo "=============================================="