#!/bin/bash
# ============================================================================
# Dual-GPU training: root_body + root_only using lingo_root_trajectory_smplx
# GPU 0: sceneco_smplx_root_body_rtdata.yaml  (root+body, dual ViT)
# GPU 1: sceneco_smplx_root_only_rtdata.yaml  (root only, dual ViT)
#
# Estimated runtime: ~16-20 hours (50 epochs × ~20 min/epoch)
# Usage: bash kimodo_scene_project/scripts/train_rtdata_dual.sh
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

SESSION="rtdata-train"
PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python
TRAIN_SCRIPT="kimodo_scene_project/train/train_sceneco.py"

CONFIG_GPU0="kimodo_scene_project/configs/sceneco_smplx_root_body_rtdata.yaml"
CONFIG_GPU1="kimodo_scene_project/configs/sceneco_smplx_root_only_rtdata.yaml"

export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"
export PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}"

echo "=============================================="
echo " Dual-GPU Root Trajectory Training"
echo "=============================================="
echo " GPU 0: $CONFIG_GPU0"
echo " GPU 1: $CONFIG_GPU1"
echo " Session:  $SESSION"
echo " Python:   $($PYTHON --version)"
echo " Time:     $(date -Iseconds)"
echo "=============================================="

# Kill existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

# --- GPU 0: root_body ---
tmux new-session -d -s "$SESSION" -n "gpu0-root_body"

GPU0_CMD="cd $PROJECT_DIR && \
export PYTHONHASHSEED=0 && \
export CHECKPOINT_DIR='models' && \
export HF_HOME='.hf_cache' && \
export TEXT_ENCODERS_DIR='text_encoders' && \
export TEXT_ENCODER_MODE='local' && \
export TEXT_ENCODER_DEVICE='cpu' && \
export PYTHONPATH='kimodo:SOMA:\${PYTHONPATH:-}' && \
export CUDA_VISIBLE_DEVICES=0 && \
$PYTHON -u $TRAIN_SCRIPT $CONFIG_GPU0 2>&1 | tee kimodo_scene_project/outputs/smplx_root_body_rtdata/train.log"

tmux send-keys -t "$SESSION:0" "$GPU0_CMD" C-m

echo "  [GPU 0] → tmux:$SESSION:0 (gpu0-root_body)"
echo "       Config:  $CONFIG_GPU0"
echo "       Output:  kimodo_scene_project/outputs/smplx_root_body_rtdata/"

# --- GPU 1: root_only ---
tmux new-window -t "$SESSION" -n "gpu1-root_only"

GPU1_CMD="cd $PROJECT_DIR && \
export PYTHONHASHSEED=0 && \
export CHECKPOINT_DIR='models' && \
export HF_HOME='.hf_cache' && \
export TEXT_ENCODERS_DIR='text_encoders' && \
export TEXT_ENCODER_MODE='local' && \
export TEXT_ENCODER_DEVICE='cpu' && \
export PYTHONPATH='kimodo:SOMA:\${PYTHONPATH:-}' && \
export CUDA_VISIBLE_DEVICES=1 && \
$PYTHON -u $TRAIN_SCRIPT $CONFIG_GPU1 2>&1 | tee kimodo_scene_project/outputs/smplx_root_only_rtdata/train.log"

tmux send-keys -t "$SESSION:1" "$GPU1_CMD" C-m

echo "  [GPU 1] → tmux:$SESSION:1 (gpu1-root_only)"
echo "       Config:  $CONFIG_GPU1"
echo "       Output:  kimodo_scene_project/outputs/smplx_root_only_rtdata/"

tmux select-window -t "$SESSION:0"

echo ""
echo "=============================================="
echo " TRAINING LAUNCHED"
echo "=============================================="
echo " Attach:   tmux attach -t $SESSION"
echo " Kill:     tmux kill-session -t $SESSION"
echo ""
echo " Navigate: Ctrl-b 0 (GPU 0), Ctrl-b 1 (GPU 1)"
echo " Detach:   Ctrl-b d"
echo ""
echo " Monitor logs:"
echo "   tail -f kimodo_scene_project/outputs/smplx_root_body_rtdata/train.log"
echo "   tail -f kimodo_scene_project/outputs/smplx_root_only_rtdata/train.log"
echo "=============================================="
