#!/bin/bash
# =============================================================================
# 方案1: TRUMAN Root-Only SceneCo 训练 (GPU 5)
#
# TRUMAN 24-joint 直接转换 (不经过 SOMA)
# SceneCo 仅在 root_model, dual ViT
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# --- 环境变量 ----------------------------------------------------------------
export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models/Kimodo-SOMA-RP-v1.1"
export HF_HOME=".hf_cache"
export HF_HUB_OFFLINE=1
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

TRUMAN_DIR="TRUMAN"
CACHE_DIR="kimodo/kimodo_sceneco/cached_data_truman"
OUTPUT_DIR="kimodo_scene_project/outputs/truman_root_only"

# =============================================================================
# Step 1: 预处理 TRUMAN 数据（只需执行一次）
# =============================================================================
if [ ! -f "$CACHE_DIR/done_preprocess" ]; then
    echo "=============================================="
    echo " Step 1a: 预处理 TRUMAN 数据 (motion+voxel) -> $CACHE_DIR"
    echo "=============================================="

    mkdir -p "$CACHE_DIR"

    PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}" \
    python kimodo_scene_project/preprocess/preprocess_truman.py \
        --truman_dir "$TRUMAN_DIR" \
        --output_dir "$CACHE_DIR" \
        --voxel_size 64,64,64 \
        --max_frames 196 \
        --min_frames 40

    echo ""
    echo "=============================================="
    echo " Step 1b: 批量文本编码 (LLM2Vec)"
    echo "=============================================="

    PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}" \
    python kimodo_scene_project/preprocess/preprocess_truman.py \
        --output_dir "$CACHE_DIR" \
        --add_text_feat_only

    touch "$CACHE_DIR/done_preprocess"
    echo "预处理完成"
else
    echo "Skip Step 1: 预处理已完成 ($CACHE_DIR/done_preprocess)"
fi

# =============================================================================
# Step 2: 训练 root_only SceneCo (GPU 5)
# =============================================================================
echo ""
echo "=============================================="
echo " Step 2: TRUMAN Root-Only SceneCo 训练"
echo " GPU: 5"
echo " SceneCo: root_model only | ViT: dual | Steps: 200k"
echo "=============================================="

export CUDA_VISIBLE_DEVICES="5"
mkdir -p "$OUTPUT_DIR"

PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}" python -c "
import os, sys
sys.path.insert(0, 'kimodo')

args = [
    '--data_root', 'TRUMAN',
    '--cache_dir', 'kimodo/kimodo_sceneco/cached_data_truman',
    '--voxel_size', '64,64,64',
    '--max_frames', '196',
    '--min_frames', '40',
    '--fps', '30',
    '--pretrained_model', 'Kimodo-SOMA-RP-v1.1',
    '--output_dir', '$OUTPUT_DIR',
    '--batch_size', '4',
    '--num_epochs', '200',
    '--lr', '1e-4',
    '--weight_decay', '0.01',
    '--max_grad_norm', '1.0',
    '--prior_weight', '0.5',
    '--scene_dropout', '0.1',
    '--num_base_steps', '1000',
    '--accum_steps', '1',
    '--val_interval', '500',
    '--val_max_batches', '10',
    '--log_interval', '50',
    '--num_workers', '0',
    '--seed', '42',
    '--train_ratio', '0.9',
    '--use_in_root_model', 'true',
    '--use_in_body_model', 'false',
    '--sceneco_dropout', '0.1',
    '--use_dual_vit', 'true',
    '--root_voxel_mode', 'full',
    '--freeze_pretrained',
    '--device', 'cuda:0',
]

sys.argv = ['train_truman_root_only.py'] + args
from kimodo_sceneco.train.train import main
main()
" 2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=============================================="
echo " 方案1 训练完成!"
echo " Checkpoint: $OUTPUT_DIR/checkpoints/best_checkpoint.pt"
echo " Log:        $OUTPUT_DIR/train.log"
echo "=============================================="
