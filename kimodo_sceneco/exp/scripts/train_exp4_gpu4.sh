#!/bin/bash
# ============================================================
# Experiment 4: Rewrite Layer — SceneCo on BODY MODEL only
# root: SA -> FFN (original) | body: SA -> SceneCo -> FFN
# Uses GPU 5 | batch=12 | ~1 day
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES=5
export CHECKPOINT_DIR=/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models
export TEXT_ENCODER_MODE=local
export HF_HUB_OFFLINE=1

cd /home/lzsh2025/kimodo-viser/kimodo

echo "=============================================="
echo " Experiment 4: Body-Only SceneCo (GPU 5)"
echo " root=SA->FFN | body=SA->SceneCo->FFN"
echo " batch_size=12 | epochs=100"
echo "=============================================="

python -m kimodo_sceneco.exp.train_exp \
    --exp_type exp4 \
    --data_root /home/lzsh2025/kimodo-viser/LINGO/dataset \
    --cache_dir /home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/cached_data \
    --pretrained_model Kimodo-SOMA-RP-v1.1 \
    --freeze_pretrained \
    --output_dir ./exp4_body_only_output \
    --batch_size 12 \
    --accum_steps 1 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --num_epochs 100 \
    --prior_weight 0.5 \
    --scene_dropout 0.1 \
    --scene_dim 256 \
    --scene_num_heads 4 \
    --scene_num_layers 4 \
    --scene_ff_dim 512 \
    --sceneco_dropout 0.1 \
    --max_frames 196 \
    --min_frames 40 \
    --val_interval 500 \
    --val_max_batches 10 \
    --log_interval 50 \
    --num_workers 4 \
    --seed 42
