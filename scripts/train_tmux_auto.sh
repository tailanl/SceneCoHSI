#!/bin/bash
# Multi-GPU training auto-launcher via tmux
# Starts training on GPUs 3, 6, 7 automatically
# Usage: bash scripts/train_tmux_auto.sh [steps] [--gpus "3,6,7"]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

SESSION="kimodo-train"
PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python
STEPS="${1:-200000}"
GPU_LIST="${2:-3,6,7}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"

CONFIG="kimodo_scene_project/configs/sceneco_root_only.yaml"
LOG_INTERVAL=100
FROZEN_CHECK=500
CKPT_VERIFY=5000

export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

echo "=============================================="
echo " Multi-GPU Training Auto-Launcher"
echo "=============================================="
echo " GPUs:           ${GPUS[*]}"
echo " Steps per GPU:  $STEPS"
echo " Config:         $CONFIG"
echo " Python:         $($PYTHON --version)"
echo " Time:           $(date -Iseconds)"
echo "=============================================="

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

first_window=true
win_idx=0

for real_gpu in "${GPUS[@]}"; do
    OUTPUT="kimodo_scene_project/outputs/root_only_sceneco_gpu${real_gpu}"
    mkdir -p "$OUTPUT/checkpoints"
    WNAME="gpu${real_gpu}"
    LOGFILE="$OUTPUT/train_gpu${real_gpu}_$(date +%Y%m%d_%H%M%S).log"

    if $first_window; then
        tmux new-session -d -s "$SESSION" -n "$WNAME"
        first_window=false
    else
        tmux new-window -t "$SESSION" -n "$WNAME"
    fi

    CMD="cd $PROJECT_DIR && \
export PYTHONHASHSEED=0 && \
export CHECKPOINT_DIR='models' && \
export HF_HOME='.hf_cache' && \
export TEXT_ENCODERS_DIR='text_encoders' && \
export TEXT_ENCODER_MODE='local' && \
export TEXT_ENCODER_DEVICE='cpu' && \
export CUDA_VISIBLE_DEVICES=$real_gpu && \
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \
    '$CONFIG' \
    --gpu 0 \
    --steps $STEPS \
    --log_interval $LOG_INTERVAL \
    --check_frozen_every $FROZEN_CHECK \
    --ckpt_verify_every $CKPT_VERIFY \
    --output_dir '$OUTPUT' \
    2>&1 | tee '$LOGFILE'"

    tmux send-keys -t "$SESSION:$win_idx" "$CMD" C-m

    echo "  [GPU $real_gpu] → tmux:$SESSION:$win_idx ($WNAME)"
    echo "       Output: $OUTPUT"
    echo "       Log:    $LOGFILE"

    ((win_idx++))
done

tmux select-window -t "$SESSION:0"

echo ""
echo "=============================================="
echo " TRAINING LAUNCHED on GPUs: ${GPUS[*]}"
echo "=============================================="
echo " Session:  $SESSION"
echo " Attach:   tmux attach -t $SESSION"
echo " Kill:     tmux kill-session -t $SESSION"
echo ""
echo " Navigate windows: ctrl-b 0, ctrl-b 1, ctrl-b 2"
echo " Detach:           ctrl-b d"
echo ""
echo " Monitor logs via:"
for real_gpu in "${GPUS[@]}"; do
    echo "   tail -f kimodo_scene_project/outputs/root_only_sceneco_gpu${real_gpu}/train_gpu${real_gpu}_*.log"
done
echo "=============================================="
