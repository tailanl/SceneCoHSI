#!/bin/bash
# Batch two-stage inference on LINGO val set
# Usage: bash kimodo_scene_project/eval/batch_two_stage_inference.sh [start_idx] [end_idx] [gpu_id] [stage2_model]

set -eo pipefail

START_IDX=${1:-0}
END_IDX=${2:-73}
GPU=${3:-0}
STAGE2_MODEL=${4:-"Kimodo-SOMA-RP-v1.1"}
CKPT="kimodo_scene_project/outputs/root_only_sceneco/checkpoints/best_checkpoint.pt"
OUTPUT_DIR="kimodo_scene_project/outputs/two_stage_inference"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

export PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}"
export CHECKPOINT_DIR="models"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"
export PYTHONHASHSEED="0"

echo "============================================================"
echo "Batch Two-Stage Inference on LINGO Val Set"
echo "  Samples: ${START_IDX} -> ${END_IDX}"
echo "  GPU: ${GPU}"
echo "  Stage2 Model: ${STAGE2_MODEL}"
echo "  Checkpoint: ${CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

FAILED=()
for i in $(seq "$START_IDX" "$END_IDX"); do
    echo ""
    echo "--- [$(date '+%H:%M:%S')] Sample $i / $END_IDX ---"
    if python kimodo_scene_project/eval/two_stage_inference.py \
        --ckpt_path "$CKPT" \
        --dataset_sample "$i" \
        --num_samples 1 \
        --gpu "$GPU" \
        --output_dir "$OUTPUT_DIR" \
        --stage2_model "$STAGE2_MODEL" \
        --seed 42; then
        echo "  OK: sample $i"
    else
        echo "  FAILED: sample $i"
        FAILED+=("$i")
    fi
done

echo ""
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL DONE. All ${START_IDX}-${END_IDX} samples completed."
else
    echo "FAILED samples: ${FAILED[*]}"
fi
echo "Output: ${OUTPUT_DIR}"
