#!/bin/bash
# ============================================================================
# TrajCo 五方案对比训练 — tmux 启动脚本
#
# Plan D: TrajCo-Cross only          (GPU 0, tmux: trajco_D)
# Plan E: TrajCo-Cross + SceneCo     (GPU 1, tmux: trajco_E)
# Plan A: SceneCo baseline           (GPU 2, tmux: trajco_A)
# Plan B: TrajCo additive only       (GPU 3, tmux: trajco_B)
# Plan C: TrajCo additive + SceneCo  (GPU 4, tmux: trajco_C)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
KIMODO_DIR="$(dirname "$PROJECT_DIR")"
CHECKPOINT_DIR="$KIMODO_DIR/models"
TEXT_ENCODERS_DIR="$KIMODO_DIR/text_encoders"
CONFIG_DIR="$PROJECT_DIR/configs"
TRAIN_SCRIPT="$PROJECT_DIR/train/train_sceneco.py"

CONDA_ENV="kimodo"

GPU_D=0
GPU_E=1
GPU_A=2
GPU_B=3
GPU_C=4

ENV_PREFIX="LOCAL_CACHE=true TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu TEXT_ENCODERS_DIR=$TEXT_ENCODERS_DIR CHECKPOINT_DIR=$CHECKPOINT_DIR"

_launch() {
    local session="$1" gpu="$2" config="$3" logfile="$4"
    mkdir -p "$PROJECT_DIR/logs"
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" -n train
    tmux send-keys -t "$session" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$session" "cd $KIMODO_DIR" Enter
    tmux send-keys -t "$session" \
        "$ENV_PREFIX CUDA_VISIBLE_DEVICES=$gpu python $TRAIN_SCRIPT $config 2>&1 | tee $logfile" Enter
}

# ======================== Cross-Attention Variants ========================
start_plan_d() {
    _launch "trajco_D" "$GPU_D" \
        "$CONFIG_DIR/trajco_cross_smplx.yaml" \
        "$PROJECT_DIR/logs/plan_D_trajco_cross_only.log"
    echo "  Plan D (TrajCo-Cross only)        → tmux: trajco_D, GPU: $GPU_D"
}

start_plan_e() {
    _launch "trajco_E" "$GPU_E" \
        "$CONFIG_DIR/trajco_cross_sceneco_smplx.yaml" \
        "$PROJECT_DIR/logs/plan_E_trajco_cross_sceneco.log"
    echo "  Plan E (TrajCo-Cross + SceneCo)   → tmux: trajco_E, GPU: $GPU_E"
}

# ======================== Additive Variants ========================
start_plan_a() {
    _launch "trajco_A" "$GPU_A" \
        "$CONFIG_DIR/sceneco_smplx_root_body.yaml" \
        "$PROJECT_DIR/logs/plan_A_sceneco_baseline.log"
    echo "  Plan A (SceneCo baseline)         → tmux: trajco_A, GPU: $GPU_A"
}

start_plan_b() {
    _launch "trajco_B" "$GPU_B" \
        "$CONFIG_DIR/trajco_smplx.yaml" \
        "$PROJECT_DIR/logs/plan_B_trajco_only.log"
    echo "  Plan B (TrajCo additive only)     → tmux: trajco_B, GPU: $GPU_B"
}

start_plan_c() {
    _launch "trajco_C" "$GPU_C" \
        "$CONFIG_DIR/trajco_sceneco_smplx.yaml" \
        "$PROJECT_DIR/logs/plan_C_sceneco_trajco.log"
    echo "  Plan C (TrajCo add. + SceneCo)    → tmux: trajco_C, GPU: $GPU_C"
}

# ========================= Main =========================
echo "=============================================="
echo "  TrajCo 五方案训练启动"
echo "  D/E: cross-attn  |  A/B/C: additive"
echo "=============================================="

start_plan_d
start_plan_e
start_plan_a
start_plan_b
start_plan_c

echo ""
echo "=============================================="
echo "  全部启动完成！"
echo "=============================================="
echo ""
echo "  查看会话列表:  tmux ls"
echo "  进入 Plan D:   tmux attach -t trajco_D"
echo "  进入 Plan E:   tmux attach -t trajco_E"
echo "  进入 Plan A:   tmux attach -t trajco_A"
echo "  进入 Plan B:   tmux attach -t trajco_B"
echo "  进入 Plan C:   tmux attach -t trajco_C"
echo ""
echo "  监控日志:"
echo "    tail -f logs/plan_D_trajco_cross_only.log"
echo "    tail -f logs/plan_E_trajco_cross_sceneco.log"
echo "    tail -f logs/plan_A_sceneco_baseline.log"
echo "    tail -f logs/plan_B_trajco_only.log"
echo "    tail -f logs/plan_C_sceneco_trajco.log"
echo ""
echo "  停止全部:"
echo "    tmux kill-session -t trajco_D"
echo "    tmux kill-session -t trajco_E"
echo "    tmux kill-session -t trajco_A"
echo "    tmux kill-session -t trajco_B"
echo "    tmux kill-session -t trajco_C"
echo ""
