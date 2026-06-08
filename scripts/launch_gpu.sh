#!/bin/bash
# Launch training on a specific GPU side-by-side (does NOT kill existing sessions)
# Usage: bash scripts/launch_gpu.sh <GPU_ID> [steps]

set -uo pipefail

GPU="$1"
STEPS="${2:-200000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

SESSION="kimodo-train"
PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python
CONFIG="kimodo_scene_project/configs/sceneco_root_only.yaml"
OUTPUT="kimodo_scene_project/outputs/root_only_sceneco_gpu${GPU}"
LOGFILE="$OUTPUT/train_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
WNAME="gpu${GPU}"
LOG_INTERVAL=100
FROZEN_CHECK=500
CKPT_VERIFY=5000

mkdir -p "$OUTPUT/checkpoints"

# Check if session exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n "$WNAME"
else
    tmux new-session -d -s "$SESSION" -n "$WNAME"
fi

CMD="cd $PROJECT_DIR && \
export PYTHONHASHSEED=0 && \
export CHECKPOINT_DIR='models' && \
export HF_HOME='.hf_cache' && \
export TEXT_ENCODERS_DIR='text_encoders' && \
export TEXT_ENCODER_MODE='local' && \
export TEXT_ENCODER_DEVICE='cpu' && \
export CUDA_VISIBLE_DEVICES=$GPU && \
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \
    '$CONFIG' \
    --gpu 0 \
    --steps $STEPS \
    --log_interval $LOG_INTERVAL \
    --check_frozen_every $FROZEN_CHECK \
    --ckpt_verify_every $CKPT_VERIFY \
    --output_dir '$OUTPUT' \
    2>&1 | tee '$LOGFILE'"

tmux send-keys -t "$SESSION:$WNAME" "$CMD" C-m

echo "✅ GPU $GPU launched in tmux:$SESSION:$WNAME"
echo "   Log: $LOGFILE"
echo "   Output: $OUTPUT"
