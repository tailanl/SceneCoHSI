#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_postprocess() {
  local input_dir="$1"
  local output_dir="$2"
  local gpu="$3"
  python scripts/postprocess_root_raw3d.py \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --project_target_path \
    --overwrite_root_keys \
    --update_norm \
    --clearance_m 0.04 \
    --smooth_window 5 \
    --gpu "$gpu"
}

run_postprocess outputs/e5_classifier_guidance_train/path_only outputs/e8_classifier_raw3d_train 0
run_postprocess outputs/e5_classifier_guidance_val/path_only outputs/e8_classifier_raw3d_val 0
run_postprocess outputs/e6_hybrid_guidance_train/path_only outputs/e9_hybrid_raw3d_train 0
run_postprocess outputs/e6_hybrid_guidance_val/path_only outputs/e9_hybrid_raw3d_val 0
run_postprocess outputs/e7_gt_root_v3_train outputs/e10_gt_projected_train 0
run_postprocess outputs/e7_gt_root_v3_val outputs/e10_gt_projected_val 0

run_postprocess outputs/e5_classifier_guidance_train/path_only outputs/e8_classifier_raw3d_train_scenefix 0
run_postprocess outputs/e5_classifier_guidance_val/path_only outputs/e8_classifier_raw3d_val_scenefix 0
run_postprocess outputs/e6_hybrid_guidance_train/path_only outputs/e9_hybrid_raw3d_train_scenefix 0
run_postprocess outputs/e6_hybrid_guidance_val/path_only outputs/e9_hybrid_raw3d_val_scenefix 0
run_postprocess outputs/e7_gt_root_v3_train outputs/e10_gt_projected_train_scenefix 0
run_postprocess outputs/e7_gt_root_v3_val outputs/e10_gt_projected_val_scenefix 0

run_postprocess outputs/e5_classifier_guidance_train/path_only outputs/e8_classifier_raw3d_train_scenefix80 0
run_postprocess outputs/e5_classifier_guidance_val/path_only outputs/e8_classifier_raw3d_val_scenefix80 0
run_postprocess outputs/e6_hybrid_guidance_train/path_only outputs/e9_hybrid_raw3d_train_scenefix80 0
run_postprocess outputs/e6_hybrid_guidance_val/path_only outputs/e9_hybrid_raw3d_val_scenefix80 0
run_postprocess outputs/e7_gt_root_v3_train outputs/e10_gt_projected_train_scenefix80 0
run_postprocess outputs/e7_gt_root_v3_val outputs/e10_gt_projected_val_scenefix80 0
