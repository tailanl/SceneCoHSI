#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR=models
export HF_HOME=.hf_cache
export TEXT_ENCODERS_DIR=text_encoders
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DEVICE=cpu
export PYTHONHASHSEED=0
export PATH="/home/lzsh2025/bin:$PATH"

cd /home/lzsh2025/kimodo-viser

PYTHON=/home/lzsh2025/miniconda3/envs/kimodo/bin/python

echo "=== START EVALUATION $(date) ==="
echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "ffmpeg: $(which ffmpeg)"

rm -f kimodo_scene_project/outputs/eval_3exp/single_vit/*.mp4
rm -f kimodo_scene_project/outputs/eval_3exp/dual_vit/*.mp4
rm -f kimodo_scene_project/outputs/eval_3exp/dual_vit_floor/*.mp4

$PYTHON kimodo_scene_project/eval/batch_eval_3exp.py --gpu 0 --fps 20 --denoising-steps 50 2>&1 | tee kimodo_scene_project/outputs/eval_3exp/eval_log.txt

echo "=== EVAL DONE $(date) ==="
