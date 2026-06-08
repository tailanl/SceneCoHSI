#!/bin/bash
# ============================================================
# Evaluate trained SceneCo models (Exp1-4)
# Metrics: Collision Rate | Degradation Ratio | Foot Skate
#
# Usage: bash eval_exp.sh <exp_type> <checkpoint_path> [gpu_id]
# ============================================================
set -e

EXP_TYPE="${1:-exp2}"
CHECKPOINT="${2:-./exp2_rewrite_layer_output/checkpoints/best_checkpoint.pt}"
GPU="${3:-0}"

export CUDA_VISIBLE_DEVICES=$GPU
export CHECKPOINT_DIR=/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models
export TEXT_ENCODER_MODE=local
export HF_HUB_OFFLINE=1

cd /home/lzsh2025/kimodo-viser/kimodo

echo "=============================================="
echo " Evaluating ${EXP_TYPE}"
echo " Checkpoint: ${CHECKPOINT}"
echo " GPU: ${GPU}"
echo "=============================================="

python -m kimodo_sceneco.exp.eval_exp \
    --exp_type $EXP_TYPE \
    --checkpoint "$CHECKPOINT" \
    --data_root /home/lzsh2025/kimodo-viser/LINGO/dataset \
    --cache_dir /home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/cached_data \
    --output_dir "./eval_${EXP_TYPE}" \
    --batch_size 4 \
    --num_samples 200 \
    --num_denoising_steps 50 \
    --cfg_weight 2.0 2.0 2.0 \
    --val_max_batches 10 \
    --num_workers 4 \
    --device cuda
