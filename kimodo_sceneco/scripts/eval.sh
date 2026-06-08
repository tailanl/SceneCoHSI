#!/bin/bash
# Evaluate Kimodo-SceneCo
# Usage: bash kimodo/kimodo_sceneco/scripts/eval.sh [CHECKPOINT_PATH] [GPU_ID]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIMODO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate kimodo conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate kimodo

export PYTHONPATH="$KIMODO_ROOT:${PYTHONPATH:-}"
cd "$KIMODO_ROOT"

CKPT=${1:-"./sceneco_output/checkpoints/best_checkpoint.pt"}
GPU_ID=${2:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Python: $(which python)"
echo "Checkpoint=$CKPT"

python -m kimodo_sceneco.train.eval \
    --checkpoint $CKPT \
    --data_root /home/lzsh2025/kimodo-viser/LINGO/dataset \
    --output_dir ./eval_output \
    --num_samples 200 \
    --num_denoising_steps 50 \
    --cfg_weight 2.0 2.0 2.0 \
    --device cuda
