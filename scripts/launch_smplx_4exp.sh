#!/bin/bash
# Launch 4 SMPLX SceneCo experiments on GPUs 3, 4, 5, 6
# GPU3: dual_vit_floor — root-only + dual ViT + floor voxel
# GPU4: body_only    — SceneCo in body_model only
# GPU5: root_only    — SceneCo in root_model only (full voxel)
# GPU6: root_body    — SceneCo in both root_model and body_model
#
# Usage: bash scripts/launch_smplx_4exp.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

SESSION="smplx-4exp"
PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo " SMPLX 4-Experiment Training Launcher"
echo "=============================================="
echo " Model:  Kimodo-SMPLX-RP-v1 (no SOMA conversion)"
echo " GPU 3: dual_vit_floor"
echo " GPU 4: body_only"
echo " GPU 5: root_only"
echo " GPU 6: root_body"
echo " Python: $($PYTHON --version)"
echo " Time:   $(date -Iseconds)"
echo "=============================================="

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

ENV="export CUDA_VISIBLE_DEVICES=\$GPU && \
export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
export CHECKPOINT_DIR=models && \
export HF_HOME=.hf_cache && \
export HF_ENDPOINT=https://hf-mirror.com && \
export TEXT_ENCODERS_DIR=text_encoders && \
export TEXT_ENCODER_MODE=local && \
export TEXT_ENCODER_DEVICE=cpu && \
export PYTHONHASHSEED=0 &&"

# ---------- Exp 1: Dual ViT Floor (GPU 3) ----------
GPU=3
CONFIG="kimodo_scene_project/configs/sceneco_smplx_dual_vit_floor.yaml"
OUTPUT="kimodo_scene_project/outputs/smplx_dual_vit_floor"
mkdir -p "$OUTPUT/checkpoints"
LOGFILE1="$OUTPUT/train_${TIMESTAMP}.log"

CMD1="cd $PROJECT_DIR && $ENV \
PYTHONPATH=\"kimodo:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_sceneco.py \
    '$CONFIG' \
    2>&1 | tee '$LOGFILE1'"

# ---------- Exp 2: Body Only (GPU 4) ----------
GPU=4
CONFIG="kimodo_scene_project/configs/sceneco_smplx_body_only.yaml"
OUTPUT="kimodo_scene_project/outputs/smplx_body_only"
mkdir -p "$OUTPUT/checkpoints"
LOGFILE2="$OUTPUT/train_${TIMESTAMP}.log"

CMD2="cd $PROJECT_DIR && $ENV \
PYTHONPATH=\"kimodo:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_sceneco.py \
    '$CONFIG' \
    2>&1 | tee '$LOGFILE2'"

# ---------- Exp 3: Root Only (GPU 5) ----------
GPU=5
CONFIG="kimodo_scene_project/configs/sceneco_smplx_root_only.yaml"
OUTPUT="kimodo_scene_project/outputs/smplx_root_only"
mkdir -p "$OUTPUT/checkpoints"
LOGFILE3="$OUTPUT/train_${TIMESTAMP}.log"

CMD3="cd $PROJECT_DIR && $ENV \
PYTHONPATH=\"kimodo:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_sceneco.py \
    '$CONFIG' \
    2>&1 | tee '$LOGFILE3'"

# ---------- Exp 4: Root + Body (GPU 6) ----------
GPU=6
CONFIG="kimodo_scene_project/configs/sceneco_smplx_root_body.yaml"
OUTPUT="kimodo_scene_project/outputs/smplx_root_body"
mkdir -p "$OUTPUT/checkpoints"
LOGFILE4="$OUTPUT/train_${TIMESTAMP}.log"

CMD4="cd $PROJECT_DIR && $ENV \
PYTHONPATH=\"kimodo:\${PYTHONPATH:-}\" $PYTHON -u kimodo_scene_project/train/train_sceneco.py \
    '$CONFIG' \
    2>&1 | tee '$LOGFILE4'"

tmux new-session -d -s "$SESSION" -n "gpu3-floor"
tmux send-keys -t "$SESSION:0" "$CMD1" C-m
echo "  [GPU 3] dual_vit_floor → tmux:$SESSION:0"

tmux new-window -t "$SESSION" -n "gpu4-body"
tmux send-keys -t "$SESSION:1" "$CMD2" C-m
echo "  [GPU 4] body_only     → tmux:$SESSION:1"

tmux new-window -t "$SESSION" -n "gpu5-root"
tmux send-keys -t "$SESSION:2" "$CMD3" C-m
echo "  [GPU 5] root_only     → tmux:$SESSION:2"

tmux new-window -t "$SESSION" -n "gpu6-rootbody"
tmux send-keys -t "$SESSION:3" "$CMD4" C-m
echo "  [GPU 6] root_body     → tmux:$SESSION:3"

tmux select-window -t "$SESSION:0"

echo ""
echo "=============================================="
echo " ALL 4 EXPERIMENTS LAUNCHED"
echo "=============================================="
echo " Session:  $SESSION"
echo " Attach:   tmux attach -t $SESSION"
echo " Kill:     tmux kill-session -t $SESSION"
echo ""
echo " Navigate: ctrl-b 0/1/2/3"
echo " Detach:   ctrl-b d"
echo ""
echo " Monitor logs:"
echo "   tail -f kimodo_scene_project/outputs/smplx_dual_vit_floor/train_*.log"
echo "   tail -f kimodo_scene_project/outputs/smplx_body_only/train_*.log"
echo "   tail -f kimodo_scene_project/outputs/smplx_root_only/train_*.log"
echo "   tail -f kimodo_scene_project/outputs/smplx_root_body/train_*.log"
echo "=============================================="