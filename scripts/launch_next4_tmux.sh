#!/bin/bash
# ============================================================================
# SceneCoHSI Next 4 Experiments — tmux Launch Script
#
# Plan H-clean:     SceneCo body-only + TrajCo cross root+body + clean GT  (GPU 2)
# Plan H-coarse:    SceneCo body-only + TrajCo cross root+body + coarse    (GPU 3)
# Plan D-coarse:    no SceneCo + TrajCo cross root+body + coarse           (GPU 4)
# Plan F-coarse:    SceneCo body-only + TrajCo cross root-only + coarse    (GPU 5)
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

GPU_H_CLEAN=2
GPU_H_COARSE=3
GPU_D_COARSE=4
GPU_F_COARSE=5

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

# ======================== Plan H-clean (GPU 2) ========================
start_h_clean() {
    _launch "trajco_H_clean" "$GPU_H_CLEAN" \
        "$CONFIG_DIR/trajco_cross_root_body_sceneco_body_clean.yaml" \
        "$PROJECT_DIR/logs/plan_H_clean.log"
    echo "  Plan H-clean (SceneCo body + TrajCo root+body, clean GT) → tmux: trajco_H_clean, GPU: $GPU_H_CLEAN"
}

# ======================== Plan H-coarse (GPU 3) ========================
start_h_coarse() {
    _launch "trajco_H_coarse" "$GPU_H_COARSE" \
        "$CONFIG_DIR/trajco_cross_root_body_sceneco_body_coarse.yaml" \
        "$PROJECT_DIR/logs/plan_H_coarse.log"
    echo "  Plan H-coarse (SceneCo body + TrajCo root+body, coarse) → tmux: trajco_H_coarse, GPU: $GPU_H_COARSE"
}

# ======================== Plan D-coarse (GPU 4) ========================
start_d_coarse() {
    _launch "trajco_D_coarse" "$GPU_D_COARSE" \
        "$CONFIG_DIR/trajco_cross_root_body_coarse.yaml" \
        "$PROJECT_DIR/logs/plan_D_coarse.log"
    echo "  Plan D-coarse (no SceneCo + TrajCo root+body, coarse) → tmux: trajco_D_coarse, GPU: $GPU_D_COARSE"
}

# ======================== Plan F-coarse (GPU 5) ========================
start_f_coarse() {
    _launch "trajco_F_coarse" "$GPU_F_COARSE" \
        "$CONFIG_DIR/trajco_cross_root_sceneco_body_coarse.yaml" \
        "$PROJECT_DIR/logs/plan_F_coarse.log"
    echo "  Plan F-coarse (SceneCo body + TrajCo root-only, coarse) → tmux: trajco_F_coarse, GPU: $GPU_F_COARSE"
}

echo "============================================================================"
echo " SceneCoHSI Next 4 Experiments"
echo "============================================================================"

start_h_clean
start_h_coarse
start_d_coarse
start_f_coarse

echo ""
echo "All 4 experiments launched."
echo ""
echo "Check status:"
echo "  tmux attach -t trajco_H_clean"
echo "  tmux attach -t trajco_H_coarse"
echo "  tmux attach -t trajco_D_coarse"
echo "  tmux attach -t trajco_F_coarse"
echo ""
echo "Logs:"
echo "  tail -f kimodo_scene_project/logs/plan_H_clean.log"
echo "  tail -f kimodo_scene_project/logs/plan_H_coarse.log"
echo "  tail -f kimodo_scene_project/logs/plan_D_coarse.log"
echo "  tail -f kimodo_scene_project/logs/plan_F_coarse.log"
