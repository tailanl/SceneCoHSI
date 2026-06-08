#!/bin/bash
# Train Kimodo-SceneCo on LINGO dataset
# Usage: bash kimodo/kimodo_sceneco/scripts/train.sh [GPU_ID]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIMODO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate kimodo conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate kimodo

export PYTHONPATH="$KIMODO_ROOT:${PYTHONPATH:-}"
cd "$KIMODO_ROOT"

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "KIMODO_ROOT=$KIMODO_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m kimodo_sceneco.train.train \
    --data_root /home/lzsh2025/kimodo-viser/LINGO/dataset \
    --pretrained_model Kimodo-SOMA-RP-v1.1 \
    --baseline_model Kimodo-SOMA-RP-v1.1 \
    --output_dir ./sceneco_output \
    --voxel_size 64,64,64 \
    --patch_size 8,8,8 \
    --scene_dim 256 \
    --scene_num_heads 4 \
    --scene_num_layers 4 \
    --scene_ff_dim 512 \
    --num_base_steps 1000 \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --prior_weight 0.5 \
    --scene_dropout 0.1 \
    --freeze_pretrained \
    --max_frames 196 \
    --min_frames 40 \
    --val_interval 500 \
    --val_max_batches 10 \
    --log_interval 50 \
    --num_workers 4 \
    --seed 42 \
    --device cuda
