#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python

echo "=============================================="
echo " Kimodo-SceneCo Training on GPU 3"
echo " Training with EXHAUSTIVE CHECKS"
echo "=============================================="
echo "Using Python: $($PYTHON --version)"
echo "Project dir: $PROJECT_DIR"
echo ""

# ============================================================
# SYSTEM-LEVEL PRELIMINARY CHECKS (non-fatal)
# ============================================================
echo ">>> SYSTEM PRELIMINARY CHECKS <<<"

if command -v nvidia-smi &>/dev/null; then
    set +e
    echo ""
    echo "--- nvidia-smi GPU summary ---"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | head -8 || echo "  (nvidia-smi query failed — driver may not be loaded, but PyTorch may still access GPU)"
    echo ""

    GPU3_STATUS=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i 3 2>/dev/null || echo "")
    if [ -n "$GPU3_STATUS" ]; then
        GPU3_USED=$(echo "$GPU3_STATUS" | awk -F',' '{print $2}' | xargs)
        GPU3_TOTAL=$(echo "$GPU3_STATUS" | awk -F',' '{print $3}' | xargs)
        GPU3_FREE=$((GPU3_TOTAL - GPU3_USED))
        echo "[CHECK] GPU 3: ${GPU3_USED}MB used / ${GPU3_TOTAL}MB total (${GPU3_FREE}MB free)"
        if [ "$GPU3_USED" -gt 1000 ]; then
            echo "[CHECK] ⚠️  GPU 3 has >1GB memory in use — check if another process is running!"
        fi

        GPU3_PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 3 2>/dev/null || echo "")
        if [ -n "$GPU3_PROCS" ]; then
            echo "[CHECK] ⚠️  GPU 3 running processes:"
            echo "$GPU3_PROCS"
        else
            echo "[CHECK] GPU 3: No compute processes running"
        fi
    else
        echo "[CHECK] GPU 3 not found via nvidia-smi (will try PyTorch detection)"
    fi
    set -e
else
    echo "[CHECK] nvidia-smi not available"
fi

echo ""
echo "--- System Resources ---"
echo "[CHECK] CPU cores: $(nproc)"
echo "[CHECK] RAM: $(free -h | awk '/^Mem:/{print $3 " used / " $2 " total"}')"
echo "[CHECK] Disk: $(df -h "$PROJECT_DIR" | awk 'NR==2{print $4 " free / " $2 " total"}')"
echo ""

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

echo "[CHECK] Environment variables:"
echo "  PYTHONHASHSEED=$PYTHONHASHSEED"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  TEXT_ENCODERS_DIR=$TEXT_ENCODERS_DIR"
echo "  TEXT_ENCODER_MODE=$TEXT_ENCODER_MODE"
echo "  TEXT_ENCODER_DEVICE=$TEXT_ENCODER_DEVICE"

export CUDA_VISIBLE_DEVICES="3"

CONFIG="kimodo_scene_project/configs/sceneco_root_only.yaml"
OUTPUT_DIR="kimodo_scene_project/outputs/root_only_sceneco"
LOG_FILE="$OUTPUT_DIR/train_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

# Capture all training parameters for logging
echo ""
echo "=============================================="
echo " TRAINING CONFIGURATION"
echo "=============================================="
echo "Config:       $CONFIG"
echo "Output:       $OUTPUT_DIR"
echo "Log:          $LOG_FILE"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Time:         $(date -Iseconds)"
echo ""

# ============================================================
# PYTHON ENVIRONMENT CHECK
# ============================================================
echo "[CHECK] Verifying Python environment..."
$PYTHON -c "
import sys
import torch
import numpy
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  NumPy: {numpy.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(min(torch.cuda.device_count(), 8)):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1024**3:.1f} GB)')
print('  ✅ Python environment OK')
"
echo ""

# ============================================================
# LAUNCH TRAINING
# ============================================================
echo "=============================================="
echo " LAUNCHING TRAINING"
echo "=============================================="
echo ""

PYTHONPATH="kimodo:SOMA:${PYTHONPATH:-}" $PYTHON -u kimodo_scene_project/train/train_gpu3_monitor.py \
    "$CONFIG" \
    --gpu 3 \
    --steps "${1:-200}" \
    --log_interval 10 \
    --check_frozen_every 50 \
    --ckpt_verify_every 200 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================
# POST-TRAINING CHECKS
# ============================================================
echo ""
echo "=============================================="
echo " POST-TRAINING CHECKS"
echo "=============================================="
echo " Training finished with exit code: $EXIT_CODE"
echo " Log:    $LOG_FILE"
echo " Output: $OUTPUT_DIR"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== CHECK: FROZEN PARAMS ==="
    BEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints/"*_final.pt 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints/"*.pt 2>/dev/null | head -1)
    fi
    if [ -n "$BEST_CKPT" ]; then
        echo "  Using checkpoint: $BEST_CKPT"
        $PYTHON kimodo_scene_project/train/check_frozen_params.py \
            --checkpoint "$BEST_CKPT" \
            --exclude "sceneco,scene_encoder,scene_null_embed,voxel_vit" \
            --tolerance 1e-8
    else
        echo "  ⚠️  No checkpoint found to verify frozen params"
    fi
    echo ""

    echo "=== CHECK: MONITOR LOGS ==="
    if [ -f "$OUTPUT_DIR/step_monitor.jsonl" ]; then
        LOG_COUNT=$(wc -l < "$OUTPUT_DIR/step_monitor.jsonl")
        echo "  Step monitor entries: $LOG_COUNT"
        echo "  Last 3 entries:"
        tail -3 "$OUTPUT_DIR/step_monitor.jsonl" | $PYTHON -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        loss = d.get('losses', {}).get('loss', 'N/A')
        step = d.get('step', 'N/A')
        print(f'    step={step}, loss={loss}')
    except:
        pass
" 2>/dev/null || echo "  (could not parse)"
    else
        echo "  ⚠️  No step_monitor.jsonl found"
    fi
    echo ""

    # Final GPU status
    set +e
    if command -v nvidia-smi &>/dev/null; then
        echo "=== CHECK: FINAL GPU STATUS ==="
        GPU3_FINAL=$(nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits -i 3 2>/dev/null || echo "")
        if [ -n "$GPU3_FINAL" ]; then
            echo "  GPU 3: $GPU3_FINAL"
        fi
    fi
    set -e
else
    echo "❌ Training FAILED with exit code: $EXIT_CODE"
    echo ""

    # Dump last GPU status for debugging
    set +e
    if command -v nvidia-smi &>/dev/null; then
        echo "=== DEBUG: GPU STATUS ==="
        nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader -i 3 2>/dev/null || echo "  No GPU info"
    fi
    set -e

    echo ""
    echo "=== DEBUG: LAST LOG LINES ==="
    tail -30 "$LOG_FILE" 2>/dev/null || echo "  No log file"
fi

echo ""
echo "=============================================="
echo " TRAINING RUN COMPLETE"
echo "=============================================="

exit $EXIT_CODE
