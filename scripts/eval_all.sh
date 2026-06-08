#!/bin/bash
# eval_all.sh — Run full evaluation pipeline for all checkpoints
#
# Steps:
#   1. Check frozen parameter integrity
#   2. Gate=0 equivalence test
#   3. Scene condition sensitivity test
#   4. Collision and penetration metrics (scene-aware)
#   5. Text alignment and FID metrics
#   6. Generate report tables

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONHASHSEED=0

export CHECKPOINT_DIR="models/Kimodo-SOMA-RP-v1.1"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

OUTPUT_DIR="kimodo_scene_project/outputs/reports"
SCENE_DIR="LINGO/dataset/dataset/Scene"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " SceneCo Full Evaluation Pipeline"
echo "=============================================="

CKPT_ROOT_ONLY="kimodo_scene_project/outputs/root_only_sceneco/checkpoints/best_checkpoint.pt"
CKPT_ROOT_BODY="kimodo_scene_project/outputs/root_body_sceneco/checkpoints/best_checkpoint.pt"

echo ""
echo "[Step 1/5] Checking frozen parameters..."
if [ -f "$CKPT_ROOT_BODY" ]; then
    python kimodo_scene_project/train/check_frozen_params.py \
        --before "$CKPT_ROOT_BODY" \
        --after "$CKPT_ROOT_BODY" \
        --exclude "sceneco,scene_encoder,scene_null_embed" \
        --tolerance 1e-8
else
    echo "Checkpoint not found: $CKPT_ROOT_BODY (skip)"
fi

echo ""
echo "[Step 2/5] Compare checkpoints (alpha/gate stats)..."
python kimodo_scene_project/train/compare_checkpoints.py \
    --checkpoints "$CKPT_ROOT_ONLY" "$CKPT_ROOT_BODY" \
    --topk 5

echo ""
echo "[Step 3/5] Scene collision & penetration metrics..."
for CKPT in "$CKPT_ROOT_ONLY" "$CKPT_ROOT_BODY"; do
    if [ -f "$CKPT" ]; then
        CKPT_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
        echo "  Evaluating: $CKPT_NAME"
        PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python kimodo_scene_project/eval/eval_scene_metrics.py \
            --checkpoint "$CKPT" \
            --scene_data_dir "$SCENE_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --num_samples 5 \
            --seed 1234
    fi
done

echo ""
echo "[Step 4/5] Original Kimodo baseline regression..."
PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python kimodo_scene_project/eval/eval_kimodo_original.py \
    --output_dir "kimodo_scene_project/outputs/baseline_kimoto" \
    --num_frames 196 \
    --num_samples 3 \
    --seed 1234

echo ""
echo "[Step 5/5] Generate report tables..."
python kimodo_scene_project/eval/make_report_tables.py \
    --metrics_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --format csv

python kimodo_scene_project/eval/make_report_tables.py \
    --metrics_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --format latex

echo ""
echo "=============================================="
echo " Evaluation complete!"
echo " Reports: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/
