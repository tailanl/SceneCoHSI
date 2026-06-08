#!/bin/bash
# Launch 3 experiments on GPUs 1, 2, 3
# Exp1 (GPU1): 原版 - 单 scene encoder, 两 stage 共用
# Exp2 (GPU2): 双 ViT 标准 - root+body 各独立 ViT, 都看全 voxel
# Exp3 (GPU3): 双 ViT floor - root ViT 只看底部 floor, body ViT 看全 voxel
#
# Usage: bash scripts/launch_3experiments.sh [steps]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

STEPS="${1:-200000}"
SESSION="kimodo-3exp"
PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python
CONFIG="kimodo_scene_project/configs/sceneco_root_only.yaml"
LOG_INTERVAL=100
FROZEN_CHECK=500
CKPT_VERIFY=5000

echo "=============================================="
echo " 3-Experiment Training Launcher"
echo "=============================================="
echo " GPU 1: 原版 (single encoder, shared)"
echo " GPU 2: 双 ViT 标准 (root+body, full voxel)"
echo " GPU 3: 双 ViT floor (root floor-only, body full)"
echo " Steps:  $STEPS"
echo "=============================================="

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

# Common env vars (embedded in each tmux command)
ENV="export CUDA_VISIBLE_DEVICES=\$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CHECKPOINT_DIR=models HF_HOME=.hf_cache TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu PYTHONHASHSEED=0 &&"

# ---------- Experiment 0: 原版 (single encoder) ----------
GPU=1
OUTPUT="kimodo_scene_project/outputs/single_vit_gpu${GPU}"
mkdir -p "$OUTPUT/checkpoints"
LOGFILE="$OUTPUT/train_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"

CMD0="cd $PROJECT_DIR && $ENV \
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \
    '$CONFIG' \
    --gpu 0 \
    --steps $STEPS \
    --log_interval $LOG_INTERVAL \
    --check_frozen_every $FROZEN_CHECK \
    --ckpt_verify_every $CKPT_VERIFY \
    --output_dir '$OUTPUT' \
    --dual_vit false \
    2>&1 | tee '$LOGFILE'"

# ---------- Experiment 1: 双 ViT 标准 ----------
GPU=2
OUTPUT="kimodo_scene_project/outputs/dual_vit_gpu${GPU}"
mkdir -p "$OUTPUT/checkpoints"
LOGFILE="$OUTPUT/train_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"

CMD1="cd $PROJECT_DIR && $ENV \
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \
    '$CONFIG' \
    --gpu 0 \
    --steps $STEPS \
    --log_interval $LOG_INTERVAL \
    --check_frozen_every $FROZEN_CHECK \
    --ckpt_verify_every $CKPT_VERIFY \
    --output_dir '$OUTPUT' \
    --dual_vit true \
    --root_voxel_mode full \
    2>&1 | tee '$LOGFILE'"

# ---------- Experiment 2: 双 ViT floor ----------
GPU=3
OUTPUT="kimodo_scene_project/outputs/dual_vit_floor_gpu${GPU}"
mkdir -p "$OUTPUT/checkpoints"
LOGFILE="$OUTPUT/train_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"

CMD2="cd $PROJECT_DIR && $ENV \
PYTHONPATH=\"kimodo:SOMA:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \
    '$CONFIG' \
    --gpu 0 \
    --steps $STEPS \
    --log_interval $LOG_INTERVAL \
    --check_frozen_every $FROZEN_CHECK \
    --ckpt_verify_every $CKPT_VERIFY \
    --output_dir '$OUTPUT' \
    --dual_vit true \
    --root_voxel_mode floor \
    2>&1 | tee '$LOGFILE'"

# Launch in tmux windows
tmux new-session -d -s "$SESSION" -n "gpu1-orig"
tmux send-keys -t "$SESSION:0" "$CMD0" C-m
echo "  [GPU 1] 原版 → tmux:$SESSION:0"

tmux new-window -t "$SESSION" -n "gpu2-dual"
tmux send-keys -t "$SESSION:1" "$CMD1" C-m
echo "  [GPU 2] 双ViT → tmux:$SESSION:1"

tmux new-window -t "$SESSION" -n "gpu3-floor"
tmux send-keys -t "$SESSION:2" "$CMD2" C-m
echo "  [GPU 3] floor → tmux:$SESSION:2"

tmux select-window -t "$SESSION:0"

echo ""
echo "=============================================="
echo " ALL 3 EXPERIMENTS LAUNCHED"
echo "=============================================="
echo " Session:  $SESSION"
echo " Attach:   tmux attach -t $SESSION"
echo " Kill:     tmux kill-session -t $SESSION"
echo ""
echo " Navigate: ctrl-b 0/1/2"
echo " Detach:   ctrl-b d"
echo ""
echo " Monitor:"
echo "   tail -f kimodo_scene_project/outputs/single_vit_gpu1/train_gpu1_*.log"
echo "   tail -f kimodo_scene_project/outputs/dual_vit_gpu2/train_gpu2_*.log"
echo "   tail -f kimodo_scene_project/outputs/dual_vit_floor_gpu3/train_gpu3_*.log"
echo "=============================================="
