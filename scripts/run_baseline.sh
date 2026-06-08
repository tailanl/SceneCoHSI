#!/bin/bash
# run_baseline.sh — Run original Kimodo baseline evaluation
#
# This script generates the baseline results that will be used for comparison
# when evaluating SceneCo models. It saves:
#   1. Generated motion files for text-only, keyframe, path, waypoint, end-effector
#   2. Metrics JSON
#   3. Config and seed information

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

OUTPUT_DIR="kimodo_scene_project/outputs/baseline_kimoto"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " Running Kimodo Baseline Evaluation"
echo "=============================================="
echo "Output dir: $OUTPUT_DIR"
echo ""

PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python kimodo_scene_project/eval/eval_kimodo_original.py \
    --output_dir "$OUTPUT_DIR" \
    --num_frames 196 \
    --num_samples 5 \
    --seed 1234

echo ""
echo "=============================================="
echo " Baseline complete!"
echo " Results: $OUTPUT_DIR"
echo " ls $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/
