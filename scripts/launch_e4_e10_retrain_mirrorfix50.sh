#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project

SESSION="${SESSION:-e4_e10_retrain50}"
RUN_ROOT="${RUN_ROOT:-outputs/retrain_mirrorfix50}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

common_env='export CHECKPOINT_DIR=$PWD/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID'

launch_train() {
  local win="$1"
  local gpu="$2"
  local config="$3"
  local output_dir="$4"
  local train_root="$5"
  local val_root="$6"
  local mix_gt="$7"
  local mix_path="$8"
  local mix_scene="$9"

  mkdir -p "$output_dir"
  local cmd
  cmd="cd /home/lzsh2025/kimodo-viser/kimodo_scene_project && \
${common_env} && export CUDA_VISIBLE_DEVICES=${gpu} && \
python train/train_stage2_root_guided_sceneco.py ${config} \
  --gpu 0 \
  --output_dir ${output_dir} \
  --external_root_enabled true \
  --use_external_root true \
  --path_guided_root_dir ${train_root} \
  --path_scene_guided_root_dir ${train_root} \
  --val_root_dir ${val_root} \
  --root_mix_gt ${mix_gt} \
  --root_mix_path ${mix_path} \
  --root_mix_scene ${mix_scene} \
  --num_epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --save_every_epochs 50 \
  2>&1 | tee -a ${output_dir}/pipeline.log"

  tmux new-window -t "${SESSION}" -n "${win}" "${cmd}"
}

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n monitor "cd /home/lzsh2025/kimodo-viser/kimodo_scene_project && watch -n 30 'date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo; find ${RUN_ROOT} -path \"*/checkpoints/epoch_0050.pt\" -o -path \"*/checkpoints/best_checkpoint.pt\" | sort'"

launch_train E4 0 configs/stage2_energy_root_guided_sceneco.yaml \
  "${RUN_ROOT}/e4_energy_stage2" \
  outputs/e4_energy_guidance_train/path_only \
  outputs/e4_energy_guidance_val/path_only \
  0.3 0.7 0.0

launch_train E5 1 configs/stage2_classifier_root_guided_sceneco.yaml \
  "${RUN_ROOT}/e5_classifier_stage2" \
  outputs/e5_classifier_guidance_train/path_only \
  outputs/e5_classifier_guidance_val/path_only \
  0.3 0.7 0.0

launch_train E6 2 configs/stage2_hybrid_root_guided_sceneco.yaml \
  "${RUN_ROOT}/e6_hybrid_stage2" \
  outputs/e6_hybrid_guidance_train/path_only \
  outputs/e6_hybrid_guidance_val/path_only \
  0.3 0.7 0.0

launch_train E7 3 configs/stage2_gt_root_sceneco.yaml \
  "${RUN_ROOT}/e7_gt_stage2" \
  outputs/e7_gt_root_v3_train \
  outputs/e7_gt_root_v3_val \
  0.0 1.0 0.0

launch_train E8 4 configs/stage2_classifier_root_guided_sceneco.yaml \
  "${RUN_ROOT}/e8_classifier_raw3d_stage2" \
  outputs/e8_classifier_raw3d_train \
  outputs/e8_classifier_raw3d_val \
  0.3 0.0 0.7

launch_train E9 5 configs/stage2_hybrid_root_guided_sceneco.yaml \
  "${RUN_ROOT}/e9_hybrid_raw3d_stage2" \
  outputs/e9_hybrid_raw3d_train \
  outputs/e9_hybrid_raw3d_val \
  0.3 0.0 0.7

launch_train E10 6 configs/stage2_gt_root_sceneco.yaml \
  "${RUN_ROOT}/e10_gt_projected_stage2" \
  outputs/e10_gt_projected_train \
  outputs/e10_gt_projected_val \
  0.0 0.0 1.0

tmux select-window -t "${SESSION}:monitor"
echo "launched ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
echo "outputs: ${RUN_ROOT}"
