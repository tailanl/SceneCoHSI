#!/bin/bash
# Run 2D top-down video visualization for all 4 experiments + original KiMoDo
# Each experiment on its own GPU, in parallel
# Exp1→GPU1, Exp2→GPU2, Exp3→GPU3, Exp4→GPU4, Original→GPU5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJ_DIR/../topdown_videos"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

cd "$PROJ_DIR"

mkdir -p "$OUTPUT_DIR"

export TEXT_ENCODER_DEVICE=cpu
export TEXT_ENCODER_MODE=local
export HF_HUB_OFFLINE=1
export CHECKPOINT_DIR=/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models
export MPLCONFIGDIR=/tmp

run_exp() {
    local gpu_id=$1
    local exp=$2
    local logfile="$OUTPUT_DIR/${exp}.log"

    echo "[$(date '+%H:%M:%S')] Starting $exp on GPU $gpu_id" | tee "$logfile"
    CUDA_VISIBLE_DEVICES=$gpu_id python -m kimodo_sceneco.exp.vis_topdown_video \
        --gpu 0 \
        --output_dir "$OUTPUT_DIR" \
        --num_samples 3 \
        --exp "$exp" \
        >> "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] $exp DONE" | tee -a "$logfile"
}

echo "============================================"
echo "Launching 5 parallel visualization tasks"
echo "Output: $OUTPUT_DIR"
echo "============================================"

run_exp 1 exp1 &
PID1=$!
sleep 5

run_exp 2 exp2 &
PID2=$!
sleep 5

run_exp 3 exp3 &
PID3=$!
sleep 5

run_exp 4 exp4 &
PID4=$!
sleep 5

run_exp 5 original &
PID5=$!

echo ""
echo "Waiting for all tasks..."
echo "  exp1 PID=$PID1 (GPU 1)"
echo "  exp2 PID=$PID2 (GPU 2)"
echo "  exp3 PID=$PID3 (GPU 3)"
echo "  exp4 PID=$PID4 (GPU 4)"
echo "  orig PID=$PID5 (GPU 5)"
echo ""

wait $PID1 $PID2 $PID3 $PID4 $PID5

echo ""
echo "============================================"
echo "ALL DONE!"
echo "Videos in: $OUTPUT_DIR"
echo "============================================"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | head -5
echo "..."
ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l
echo "total videos"
