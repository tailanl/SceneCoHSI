#!/bin/bash
# ============================================================
#  2-Stage SceneCo Training on GPU 1
#  Stage 1: Root-only (loss on first 5 dims), 150k steps
#  Stage 2: Full body, freeze Stage1 SceneCo, 100k steps
# ============================================================
set -e

# ---- env ----
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=models
export HF_HOME=.hf_cache
export TEXT_ENCODERS_DIR=text_encoders
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DEVICE=cpu
export PYTHONHASHSEED=0

cd /home/lzsh2025/kimodo-viser
PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python

OUT1=kimodo_scene_project/outputs/stage1_root_only
OUT2=kimodo_scene_project/outputs/stage2_full_body

mkdir -p "$OUT1/checkpoints" "$OUT2/checkpoints"

# ============================================================
#  STAGE 1 : Root-Only
# ============================================================
echo "=================================="
echo "  STAGE 1: Root-Only Training"
echo "  GPU 1, batch_size=8, 150k steps"
echo "  Loss: first 5 dims only (root+hdr)"
echo "  Dual ViT, floor mode"
echo "  $(date)"
echo "=================================="

$PYTHON kimodo_scene_project/train/train_gpu3_monitor.py \
    kimodo_scene_project/configs/stage1_root_only.yaml \
    --steps 150000 \
    --dual_vit true \
    --root_voxel_mode floor \
    --root_only \
    --batch_size_override 8 \
    --skip_preflight \
    2>&1 | tee "$OUT1/train_log.txt"

# ---- find best checkpoint ----
STAGE1_CKPT=$(ls -t "$OUT1/checkpoints/checkpoint_step"*_final.pt 2>/dev/null | head -1)
if [ -z "$STAGE1_CKPT" ]; then
    STAGE1_CKPT=$(ls -t "$OUT1/checkpoints/checkpoint_step"*.pt 2>/dev/null | head -1)
fi

echo ""
echo "=================================="
echo "  STAGE 1 DONE"
echo "  Checkpoint: $STAGE1_CKPT"
echo "  $(date)"
echo "=================================="

# ============================================================
#  STAGE 2 : Full Body (freeze SceneCo)
# ============================================================
echo ""
echo "=================================="
echo "  STAGE 2: Full Body Training"
echo "  GPU 1, batch_size=8, 100k steps"
echo "  Freeze Stage1 SceneCo layers"
echo "  Load: $STAGE1_CKPT"
echo "  $(date)"
echo "=================================="

$PYTHON kimodo_scene_project/train/train_gpu3_monitor.py \
    kimodo_scene_project/configs/stage2_full_body.yaml \
    --steps 100000 \
    --dual_vit true \
    --root_voxel_mode floor \
    --freeze_sceneco \
    --scene_co_ckpt "$STAGE1_CKPT" \
    --batch_size_override 8 \
    --skip_preflight \
    2>&1 | tee "$OUT2/train_log.txt"

echo ""
echo "=================================="
echo "  STAGE 2 DONE"
echo "  $(date)"
echo "=================================="
