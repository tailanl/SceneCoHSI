#!/bin/bash
# Multi-GPU training launcher via tmux
# Usage: bash scripts/train_tmux_multi_gpu.sh [steps]
# Runs training on GPUs 3, 6, 7 simultaneously in separate tmux windows
# Default: 200000 steps (full training, ~1 day)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

SESSION="kimodo-train"
PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python
STEPS="${1:-200000}"
CONFIG="kimodo_scene_project/configs/sceneco_root_only.yaml"
LOG_INTERVAL=100
FROZEN_CHECK=500
CKPT_VERIFY=5000

# Export common env vars for all training processes
export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

echo "=============================================="
echo " Multi-GPU Training Launcher (tmux)"
echo "=============================================="
echo " GPUs:           3, 6, 7"
echo " Steps per GPU:  $STEPS"
echo " Config:         $CONFIG"
echo " Log interval:   $LOG_INTERVAL"
echo " Python:         $($PYTHON --version)"
echo " Time:           $(date -Iseconds)"
echo "=============================================="

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create new session with first window (GPU 3)
tmux new-session -d -s "$SESSION" -n "gpu3"

# ---- Window 0: GPU 3 ----
OUTPUT3="kimodo_scene_project/outputs/root_only_sceneco_gpu3"
mkdir -p "$OUTPUT3/checkpoints"
tmux send-keys -t "$SESSION:0" "cd $PROJECT_DIR && \\
export CUDA_VISIBLE_DEVICES=3 && \\
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \\
    '$CONFIG' \\
    --gpu 3 \\
    --steps $STEPS \\
    --log_interval $LOG_INTERVAL \\
    --check_frozen_every $FROZEN_CHECK \\
    --ckpt_verify_every $CKPT_VERIFY \\
    --output_dir '$OUTPUT3' \\
    2>&1 | tee '$OUTPUT3/train_gpu3_\$(date +%Y%m%d_%H%M%S).log'" C-m

echo "  [GPU 3] Output: $OUTPUT3"

# ---- Window 1: GPU 6 ----
tmux new-window -t "$SESSION" -n "gpu6"
OUTPUT6="kimodo_scene_project/outputs/root_only_sceneco_gpu6"
mkdir -p "$OUTPUT6/checkpoints"
tmux send-keys -t "$SESSION:1" "cd $PROJECT_DIR && \\
export CUDA_VISIBLE_DEVICES=6 && \\
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \\
    '$CONFIG' \\
    --gpu 6 \\
    --steps $STEPS \\
    --log_interval $LOG_INTERVAL \\
    --check_frozen_every $FROZEN_CHECK \\
    --ckpt_verify_every $CKPT_VERIFY \\
    --output_dir '$OUTPUT6' \\
    2>&1 | tee '$OUTPUT6/train_gpu6_\$(date +%Y%m%d_%H%M%S).log'" C-m

echo "  [GPU 6] Output: $OUTPUT6"

# ---- Window 2: GPU 7 ----
tmux new-window -t "$SESSION" -n "gpu7"
OUTPUT7="kimodo_scene_project/outputs/root_only_sceneco_gpu7"
mkdir -p "$OUTPUT7/checkpoints"
tmux send-keys -t "$SESSION:2" "cd $PROJECT_DIR && \\
export CUDA_VISIBLE_DEVICES=7 && \\
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \\
    '$CONFIG' \\
    --gpu 7 \\
    --steps $STEPS \\
    --log_interval $LOG_INTERVAL \\
    --check_frozen_every $FROZEN_CHECK \\
    --ckpt_verify_every $CKPT_VERIFY \\
    --output_dir '$OUTPUT7' \\
    2>&1 | tee '$OUTPUT7/train_gpu7_\$(date +%Y%m%d_%H%M%S).log'" C-m

echo "  [GPU 7] Output: $OUTPUT7"

# ---- Switch to first window and attach ----
tmux select-window -t "$SESSION:0"

echo ""
echo "=============================================="
echo " LAUNCHED! Commands are loaded in tmux windows."
echo " Press ENTER in each window to start training."
echo " Session: $SESSION"
echo " Attach:  tmux attach -t $SESSION"
echo " Kill:    tmux kill-session -t $SESSION"
echo " Windows: gpu3 (ctrl-b 0), gpu6 (ctrl-b 1), gpu7 (ctrl-b 2)"
echo "=============================================="
echo ""
echo "--- To attach now ---"
echo "    tmux attach -t $SESSION"
echo ""
