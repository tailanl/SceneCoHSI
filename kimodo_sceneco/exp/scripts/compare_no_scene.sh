#!/bin/bash
# ============================================================
# Compare original KiMoDo vs trained SceneCo (NO SCENE input)
# Verifies trained models retain Kimodo capabilities
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES="${1:-0}"
export CHECKPOINT_DIR=/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models
export TEXT_ENCODER_MODE=local
export HF_HUB_OFFLINE=1

cd /home/lzsh2025/kimodo-viser/kimodo

python -m kimodo_sceneco.exp.compare_no_scene
