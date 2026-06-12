#!/usr/bin/env bash
set -euo pipefail

cd /home/lzsh2025/kimodo-viser/kimodo_scene_project

SESSION="${SESSION:-latest_ckpt_eval_viz}"
RUN_ROOT="${RUN_ROOT:-outputs/retrain_mirrorfix50}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/latest_ckpt_eval}"
GPU="${GPU:-7}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
BODY_STEPS="${BODY_STEPS:-50}"
CFG_WEIGHT="${CFG_WEIGHT:-2.0 2.0}"

mkdir -p "${EVAL_ROOT}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n eval "cd /home/lzsh2025/kimodo-viser/kimodo_scene_project && CUDA_VISIBLE_DEVICES=${GPU} RUN_ROOT=${RUN_ROOT} EVAL_ROOT=${EVAL_ROOT} MAX_SAMPLES=${MAX_SAMPLES} BODY_STEPS=${BODY_STEPS} CFG_WEIGHT='${CFG_WEIGHT}' bash scripts/run_latest_ckpt_eval_viz_worker.sh"
tmux new-window -t "${SESSION}" -n monitor "cd /home/lzsh2025/kimodo-viser/kimodo_scene_project && watch -n 30 'date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo; find ${EVAL_ROOT} -maxdepth 3 -type f \\( -name \"*.csv\" -o -name \"*.mp4\" -o -name \"ANALYSIS.md\" \\) | sort'"

echo "launched ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
echo "outputs: ${EVAL_ROOT}"
