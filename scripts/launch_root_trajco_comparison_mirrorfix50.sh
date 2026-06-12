#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project

SESSION="${SESSION:-root_trajco_compare50}"
RUN_ROOT="${RUN_ROOT:-outputs/retrain_mirrorfix50}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-80}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAGE1_RESUME="${STAGE1_RESUME:-}"

STAGE1_OUT="${RUN_ROOT}/root_trajco_stage1"
STAGE2_OUT="${RUN_ROOT}/root_trajco_stage2_sceneco_body"
STAGE1_CKPT="${STAGE1_OUT}/checkpoints/best_checkpoint.pt"

mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}"

cmd="set -euo pipefail
cd /home/lzsh2025/kimodo-viser/kimodo_scene_project
export CHECKPOINT_DIR=\$PWD/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU}

echo '[root_trajco] Stage1 root TrajCo start:' \$(date)
stage1_resume_args=()
if test -n \"${STAGE1_RESUME}\"; then
  stage1_resume_args=(--resume ${STAGE1_RESUME})
elif test -f ${STAGE1_CKPT}; then
  stage1_resume_args=(--resume ${STAGE1_CKPT})
fi
python train/train_twostep_stage1.py configs/twostep_stage1_trajco_root.yaml \\
  --gpu 0 \\
  --output_dir ${STAGE1_OUT} \\
  --num_epochs ${EPOCHS} \\
  --batch_size ${STAGE1_BATCH_SIZE} \\
  --num_workers ${NUM_WORKERS} \\
  --save_every_epochs 50 \\
  \"\${stage1_resume_args[@]}\" \\
  2>&1 | tee -a ${STAGE1_OUT}/pipeline.log

test -f ${STAGE1_CKPT}

echo '[root_trajco] Stage2 body SceneCo start:' \$(date)
python train/train_twostep_stage2.py configs/twostep_stage2_sceneco_body.yaml \\
  --gpu 0 \\
  --stage1_ckpt ${STAGE1_CKPT} \\
  --output_dir ${STAGE2_OUT} \\
  --num_epochs ${EPOCHS} \\
  --batch_size ${STAGE2_BATCH_SIZE} \\
  --num_workers ${NUM_WORKERS} \\
  --save_every_epochs 50 \\
  2>&1 | tee -a ${STAGE2_OUT}/pipeline.log

echo '[root_trajco] complete:' \$(date)"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n train "${cmd}"
tmux new-window -t "${SESSION}" -n monitor "cd /home/lzsh2025/kimodo-viser/kimodo_scene_project && watch -n 30 'date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo; find ${STAGE1_OUT} ${STAGE2_OUT} -path \"*/checkpoints/epoch_0050.pt\" -o -path \"*/checkpoints/best_checkpoint.pt\" | sort'"
tmux select-window -t "${SESSION}:train"

echo "launched ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
echo "stage1: ${STAGE1_OUT}"
echo "stage2: ${STAGE2_OUT}"
