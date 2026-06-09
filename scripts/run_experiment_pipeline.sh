#!/bin/bash
# ============================================================
# Master Experiment Pipeline
# Run prediction and evaluation for any experiment
#
# Usage:
#   bash scripts/run_experiment_pipeline.sh E1
#   bash scripts/run_experiment_pipeline.sh E2
#   bash scripts/run_experiment_pipeline.sh E7 --checkpoint outputs/e7_gt_root_stage2_sceneco/checkpoints/best_checkpoint.pt
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CHECKPOINT_DIR=$PROJECT_DIR/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EXP=${1:-}
CHECKPOINT=${2:-}
GPU=${GPU:-0}

usage() {
    echo "Usage: bash scripts/run_experiment_pipeline.sh <EXP_ID> [--checkpoint PATH] [--gpu N]"
    echo ""
    echo "Experiments:"
    echo "  E1  EnergyGuidance + Original Body"
    echo "  E2  ClassifierGuidance + Original Body"
    echo "  E3  HybridGuidance + Original Body"
    echo "  E4  EnergyGuidance + Stage2 SceneCo"
    echo "  E5  ClassifierGuidance + Stage2 SceneCo"
    echo "  E6  HybridGuidance + Stage2 SceneCo"
    echo "  E7  GTRoot + Stage2 SceneCo"
    exit 1
}

[ -z "$EXP" ] && usage

# Parse extra args
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        *) shift ;;
    esac
done

declare -A ROOT_DIR BODY_DIR METRIC_PREFIX

ROOT_DIR["E1"]="outputs/e1_energy_guidance_root"
ROOT_DIR["E2"]="outputs/e2_classifier_guidance_root"
ROOT_DIR["E3"]="outputs/e3_hybrid_guidance_root"
ROOT_DIR["E4"]="outputs/e4_energy_guidance_val/path_only"
ROOT_DIR["E5"]="outputs/e5_classifier_guidance_val/path_only"
ROOT_DIR["E6"]="outputs/e6_hybrid_guidance_val/path_only"
ROOT_DIR["E7"]="outputs/e7_gt_root_val"

BODY_DIR["E1"]="outputs/e1_energy_guidance_body"
BODY_DIR["E2"]="outputs/e2_classifier_guidance_body"
BODY_DIR["E3"]="outputs/e3_hybrid_guidance_body"
BODY_DIR["E4"]="outputs/e4_energy_stage2_sceneco/val_gen"
BODY_DIR["E5"]="outputs/e5_classifier_stage2_sceneco/val_gen"
BODY_DIR["E6"]="outputs/e6_hybrid_stage2_sceneco/val_gen"
BODY_DIR["E7"]="outputs/e7_gt_root_stage2_sceneco/val_gen"

METRIC_PREFIX["E1"]="e1_energy_guidance_original_body"
METRIC_PREFIX["E2"]="e2_classifier_guidance_original_body"
METRIC_PREFIX["E3"]="e3_hybrid_guidance_original_body"
METRIC_PREFIX["E4"]="e4_energy_guidance_stage2_sceneco"
METRIC_PREFIX["E5"]="e5_classifier_guidance_stage2_sceneco"
METRIC_PREFIX["E6"]="e6_hybrid_guidance_stage2_sceneco"
METRIC_PREFIX["E7"]="e7_gt_root_stage2_sceneco"

ROOT="${ROOT_DIR[$EXP]}"
BODY="${BODY_DIR[$EXP]}"
NAME="${METRIC_PREFIX[$EXP]}"

echo "========================================"
echo "Pipeline: $EXP"
echo "  Root dir:  $ROOT"
echo "  Body dir:  $BODY"
echo "  GPU:       $GPU"
echo "  Checkpoint: ${CHECKPOINT:-N/A (original body)}"
echo "========================================"

# Check if root files exist
ROOT_COUNT=$(ls "$ROOT"/*.npz 2>/dev/null | wc -l)
if [ "$ROOT_COUNT" -eq 0 ]; then
    echo "ERROR: No root files in $ROOT"
    exit 1
fi
echo "Root files: $ROOT_COUNT"

# ---- Step A: Body Generation ----
if [ -n "$CHECKPOINT" ]; then
    echo ""
    echo "--- Step A: Stage2 Body Generation ---"
    mkdir -p "$BODY"
    python scripts/generate_body_from_root.py \
        --root_dir "$ROOT" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$BODY" \
        --num_denoising_steps 50 \
        --cfg_weight 2.0 2.0 \
        --gpu "$GPU" \
        2>&1 | tee "$BODY/generate_body.log"
else
    echo ""
    echo "--- Step A: Original Body (already generated or skip) ---"
    BODY_COUNT=$(ls "$BODY"/*.npz 2>/dev/null | wc -l)
    if [ "$BODY_COUNT" -eq 0 ]; then
        mkdir -p "$BODY"
        python scripts/generate_body_from_root.py \
            --root_dir "$ROOT" \
            --output_dir "$BODY" \
            --num_denoising_steps 50 \
            --cfg_weight 2.0 2.0 \
            --gpu "$GPU" \
            2>&1 | tee "$BODY/generate_body.log"
    else
        echo "Body files already exist: $BODY_COUNT"
    fi
fi

# Root fix check
grep -i "root fix" "$BODY/generate_body.log" || true

# ---- Step B: Path Metrics ----
echo ""
echo "--- Step B: Path Metrics ---"
python eval/eval_path_metrics.py \
    --pred_dir "$BODY" \
    --output_csv "$BODY/path_metrics.csv" \
    --method "$NAME" \
    2>&1 | tee "$BODY/eval_path.log"

# ---- Step C: Scene Metrics ----
echo ""
echo "--- Step C: Scene Metrics ---"
python eval/eval_sceneadapt_metrics.py \
    --pred_dir "$BODY" \
    --scene_dir LINGO/dataset/dataset/Scene \
    --output_csv "$BODY/scene_metrics.csv" \
    --method "$NAME" \
    2>&1 | tee "$BODY/eval_scene.log"

echo ""
echo "========================================"
echo "Pipeline $EXP COMPLETE"
echo "  Path metrics:   $BODY/path_metrics.csv"
echo "  Scene metrics:  $BODY/scene_metrics.csv"
echo "========================================"
