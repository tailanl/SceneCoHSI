#!/bin/bash
# ============================================================
# Experiment 1: Monkey-Patch (GPU 2)
# SceneCo injected AFTER each TransformerEncoderLayer (SA+FFN)
# Batch=12 on 4090 24GB, ~1 day for 100 epochs
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES=2
export CHECKPOINT_DIR=/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models
export TEXT_ENCODER_MODE=local
export HF_HUB_OFFLINE=1

cd /home/lzsh2025/kimodo-viser/kimodo

echo "=============================================="
echo " Experiment 1: Monkey-Patch (GPU 2)"
echo " batch_size=12 | epochs=100"
echo " ~$(python3 -c "print(int(13955/12*100*0.35/3600))" 2>/dev/null || echo "12-16")h estimated"
echo "=============================================="

python -m kimodo_sceneco.exp.train_exp \
    --exp_type exp1 \
    --data_root /home/lzsh2025/kimodo-viser/LINGO/dataset \
    --cache_dir /home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/cached_data \
    --pretrained_model Kimodo-SOMA-RP-v1.1 \
    --freeze_pretrained \
    --output_dir ./exp1_monkey_patch_output \
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
