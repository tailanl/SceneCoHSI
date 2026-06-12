#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project

RUN_ROOT="${RUN_ROOT:-outputs/retrain_mirrorfix50}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/latest_ckpt_eval}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
BODY_STEPS="${BODY_STEPS:-50}"
CFG_WEIGHT="${CFG_WEIGHT:-2.0 2.0}"

export CHECKPOINT_DIR="$PWD/models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "${EVAL_ROOT}"

latest_ckpt() {
  local stage_dir="$1"
  local epoch_ckpt="${stage_dir}/checkpoints/epoch_0050.pt"
  local best_ckpt="${stage_dir}/checkpoints/best_checkpoint.pt"
  if [[ -f "${epoch_ckpt}" ]]; then
    echo "${epoch_ckpt}"
  else
    echo "${best_ckpt}"
  fi
}

run_exp() {
  local exp="$1"
  local stage_dir="$2"
  local root_val="$3"
  local method="$4"
  local ckpt
  ckpt="$(latest_ckpt "${stage_dir}")"
  local pred_dir="${EVAL_ROOT}/${exp}/pred"
  mkdir -p "${pred_dir}"

  echo "=== ${exp} ==="
  echo "checkpoint: ${ckpt}"
  echo "root_val: ${root_val}"
  echo "pred_dir: ${pred_dir}"

  python scripts/generate_body_from_root.py \
    --root_dir "${root_val}" \
    --output_dir "${pred_dir}" \
    --checkpoint "${ckpt}" \
    --num_denoising_steps "${BODY_STEPS}" \
    --cfg_weight ${CFG_WEIGHT} \
    --max_samples "${MAX_SAMPLES}" \
    --skip_existing \
    --gpu 0 \
    2>&1 | tee "${pred_dir}/generate_body.log"

  python eval/eval_path_metrics.py \
    --pred_dir "${pred_dir}" \
    --output_csv "${pred_dir}/path_metrics.csv" \
    --method "${method}" \
    2>&1 | tee "${pred_dir}/eval_path.log"

  python eval/eval_sceneadapt_metrics.py \
    --pred_dir "${pred_dir}" \
    --output_csv "${pred_dir}/scene_metrics.csv" \
    --method "${method}" \
    2>&1 | tee "${pred_dir}/eval_scene.log"
}

run_exp E4 "${RUN_ROOT}/e4_energy_stage2" outputs/e4_energy_guidance_val/path_only e4_latest_energy
run_exp E5 "${RUN_ROOT}/e5_classifier_stage2" outputs/e5_classifier_guidance_val/path_only e5_latest_classifier
run_exp E6 "${RUN_ROOT}/e6_hybrid_stage2" outputs/e6_hybrid_guidance_val/path_only e6_latest_hybrid
run_exp E7 "${RUN_ROOT}/e7_gt_stage2" outputs/e7_gt_root_v3_val e7_latest_gt
run_exp E8 "${RUN_ROOT}/e8_classifier_raw3d_stage2" outputs/e8_classifier_raw3d_val e8_latest_classifier_raw3d
run_exp E9 "${RUN_ROOT}/e9_hybrid_raw3d_stage2" outputs/e9_hybrid_raw3d_val e9_latest_hybrid_raw3d
run_exp E10 "${RUN_ROOT}/e10_gt_projected_stage2" outputs/e10_gt_projected_val e10_latest_gt_projected

python scripts/summarize_latest_eval_metrics.py --eval_root "${EVAL_ROOT}"

python scripts/render_retrain_mirrorfix50_scene_videos.py \
  --run_root "${RUN_ROOT}" \
  --registry "${RUN_ROOT}/eval_viz/experiment_registry.json" \
  --eval_root "${EVAL_ROOT}" \
  --include E4 E5 E6 E7 E8 E9 E10 \
  --sample_idx 0 \
  --videos_per_exp 1 \
  --output_dir "${EVAL_ROOT}/videos/scene_actions"

echo "latest checkpoint eval/viz complete: $(date)"
