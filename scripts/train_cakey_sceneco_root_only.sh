#!/bin/bash
# train_cakey_sceneco_root_only.sh
# Two-stage training: CaKey -> SceneCo (root_model only).
# Stage 1: Train CaKey in root_model for stable keyframe inbetweening.
# Stage 2: Freeze CaKey, train VoxelViT + SceneCo in root_model.

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

# ---- Stage 1: CaKey Root-Only ----
STAGE1_CONFIG="kimodo_scene_project/configs/stage1_cakey_root_only.yaml"
STAGE1_OUTPUT="kimodo_scene_project/outputs/cakey_root_only"
STAGE1_CKPT="$STAGE1_OUTPUT/checkpoints/best_checkpoint.pt"

echo "=============================================="
echo " STAGE 1: CaKey Root-Only Training"
echo "=============================================="
echo "Config:   $STAGE1_CONFIG"
echo "Output:   $STAGE1_OUTPUT"
echo "GPU:      2"
echo ""

mkdir -p "$STAGE1_OUTPUT"

export CUDA_VISIBLE_DEVICES="2"
PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}" python kimodo_scene_project/train/train_cakey_sceneco.py "$STAGE1_CONFIG" --stage stage1 \
    2>&1 | tee "$STAGE1_OUTPUT/train.log"

echo ""
echo ">>> Stage 1 complete."
echo ""

# ---- Stage 2: SceneCo Root-Only (CaKey frozen) ----
STAGE2_CONFIG="kimodo_scene_project/configs/stage2_cakey_sceneco_root_only.yaml"
STAGE2_OUTPUT="kimodo_scene_project/outputs/cakey_sceneco_root_only"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    echo "Stage 1 may have failed. Aborting Stage 2."
    exit 1
fi

echo "=============================================="
echo " STAGE 2: SceneCo Root-Only (CaKey frozen)"
echo "=============================================="
echo "Config:   $STAGE2_CONFIG"
echo "Output:   $STAGE2_OUTPUT"
echo "GPU:      2"
echo ""

mkdir -p "$STAGE2_OUTPUT"

export CUDA_VISIBLE_DEVICES="2"
PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}" python kimodo_scene_project/train/train_cakey_sceneco.py "$STAGE2_CONFIG" --stage stage2 \
    2>&1 | tee "$STAGE2_OUTPUT/train.log"

echo ""
echo "=============================================="
echo " ALL DONE! CaKey + SceneCo Root-Only training complete."
echo ""
echo " Stage 1 checkpoint: $STAGE1_CKPT"
echo " Stage 2 checkpoint: $STAGE2_OUTPUT/checkpoints/best_checkpoint.pt"
echo "=============================================="
