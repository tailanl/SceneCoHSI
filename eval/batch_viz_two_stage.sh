#!/bin/bash
# Batch visualize two-stage inference results
# Usage: bash kimodo_scene_project/eval/batch_viz_two_stage.sh

set -euo pipefail

INPUT_DIR="kimodo_scene_project/outputs/two_stage_inference"
OUTPUT_DIR="kimodo_scene_project/outputs/two_stage_viz"
FPS=20

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "============================================================"
echo "Batch Visualization of Two-Stage Inference Results"
echo "  Input:  ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  FPS:    ${FPS}"
echo "============================================================"

python kimodo_scene_project/eval/viz_two_stage.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fps "$FPS"

echo ""
echo "Done. Videos and trajectory plots saved to ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"/*.mp4 2>/dev/null | head -5
ls -la "${OUTPUT_DIR}"/*.png 2>/dev/null | head -5
