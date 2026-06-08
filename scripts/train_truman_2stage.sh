#!/bin/bash
# train_truman_2stage.sh — Two-Stage SceneCo Training on TRUMAN
#
# Uses TRUMAN 24-joint format directly without SOMA conversion.
# Dual ViT throughout both stages.
#
# Stage 1: Train SceneCo in root_model only (root trajectory)
#   - freeze_pretrained, large batch_size=64, 150k steps
#   - root_voxel_mode=floor
#
# Stage 2: Load Stage 1 checkpoint, add SceneCo in body_model
#   - freeze_pretrained, batch_size=32, 100k steps
#   - full voxel for both encoders

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models/Kimodo-SOMA-RP-v1.1"
export HF_HOME=".hf_cache"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"

# ==========================================
# Stage 1: Root-Only Training
# ==========================================
STAGE1_OUTPUT="kimodo_scene_project/outputs/truman_stage1_root_only"

echo "=============================================="
echo " STAGE 1: TRUMAN Root-Only SceneCo (GPU 0,1)"
echo "=============================================="
echo "Output:   $STAGE1_OUTPUT"
echo "Data:     TRUMAN (24-joint direct, no SOMA)"
echo "SceneCo:  root_model only"
echo "ViT:      dual ViT (root: floor voxel)"
echo "Batch:    64, Steps: 150000"
echo ""

export CUDA_VISIBLE_DEVICES="0,1"

mkdir -p "$STAGE1_OUTPUT"

PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python -c "
import os, sys, yaml
sys.path.insert(0, 'kimodo')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

args = [
    '--data_root', 'TRUMAN',
    '--cache_dir', 'kimodo/kimodo_sceneco/cached_data_truman',
    '--voxel_size', '64,64,64',
    '--max_frames', '196',
    '--min_frames', '40',
    '--fps', '30',
    '--pretrained_model', 'Kimodo-SOMA-RP-v1.1',
    '--output_dir', '$STAGE1_OUTPUT',
    '--batch_size', '64',
    '--total_steps', '150000',
    '--warmup_steps', '5000',
    '--lr', '1e-4',
    '--weight_decay', '0.01',
    '--max_grad_norm', '1.0',
    '--prior_weight', '0.5',
    '--scene_dropout', '0.1',
    '--num_base_steps', '1000',
    '--accum_steps', '1',
    '--val_interval', '500',
    '--val_max_batches', '10',
    '--log_interval', '10',
    '--num_workers', '0',
    '--seed', '42',
    '--train_ratio', '0.9',
    '--use_in_root_model', 'true',
    '--use_in_body_model', 'false',
    '--sceneco_dropout', '0.1',
    '--use_dual_vit', 'true',
    '--root_voxel_mode', 'floor',
    '--freeze_pretrained',
    '--device', 'cuda:0',
]

sys.argv = ['train_stage1.py'] + args
from kimodo_sceneco.train.train import main
main()
" 2>&1 | tee "$STAGE1_OUTPUT/train_stage1.log"

echo ""
echo "Stage 1 complete!"
echo "=============================================="

# ==========================================
# Stage 2: Full Motion Training
# ==========================================
STAGE2_OUTPUT="kimodo_scene_project/outputs/truman_stage2_full_body"
STAGE1_CKPT="$STAGE1_OUTPUT/checkpoints/best_checkpoint.pt"

echo ""
echo "=============================================="
echo " STAGE 2: TRUMAN Root+Body SceneCo (GPU 0,1)"
echo "=============================================="
echo "Stage1 CKPT: $STAGE1_CKPT"
echo "Output:      $STAGE2_OUTPUT"
echo "SceneCo:     root_model + body_model"
echo "ViT:         dual ViT (both: full voxel)"
echo "Batch:       32, Steps: 100000"
echo ""

export CUDA_VISIBLE_DEVICES="0,1"

mkdir -p "$STAGE2_OUTPUT"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    echo "Please ensure Stage 1 training completed successfully."
    exit 1
fi

PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python -c "
import os, sys, torch
sys.path.insert(0, 'kimodo')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

args = [
    '--data_root', 'TRUMAN',
    '--cache_dir', 'kimodo/kimodo_sceneco/cached_data_truman',
    '--voxel_size', '64,64,64',
    '--max_frames', '196',
    '--min_frames', '40',
    '--fps', '30',
    '--pretrained_model', 'Kimodo-SOMA-RP-v1.1',
    '--output_dir', '$STAGE2_OUTPUT',
    '--batch_size', '32',
    '--total_steps', '100000',
    '--warmup_steps', '5000',
    '--lr', '1e-4',
    '--weight_decay', '0.01',
    '--max_grad_norm', '1.0',
    '--prior_weight', '0.5',
    '--scene_dropout', '0.1',
    '--num_base_steps', '1000',
    '--accum_steps', '1',
    '--val_interval', '500',
    '--val_max_batches', '10',
    '--log_interval', '10',
    '--num_workers', '0',
    '--seed', '42',
    '--train_ratio', '0.9',
    '--use_in_root_model', 'true',
    '--use_in_body_model', 'true',
    '--sceneco_dropout', '0.1',
    '--use_dual_vit', 'true',
    '--root_voxel_mode', 'full',
    '--freeze_pretrained',
    '--device', 'cuda:0',
]

sys.argv = ['train_stage2.py'] + args

from kimodo_sceneco.train.train import Trainer
from kimodo_sceneco.train.train import parse_args

train_args = parse_args()

stage1_ckpt_path = '$STAGE1_CKPT'
print(f'Loading Stage 1 checkpoint from: {stage1_ckpt_path}')
ckpt = torch.load(stage1_ckpt_path, map_location='cpu', weights_only=False)
stage1_state = ckpt['model_state_dict']

trainer = Trainer(train_args)

trainer.model.to('cpu')
missing, unexpected = trainer.model.load_state_dict(stage1_state, strict=False)
print(f'Loaded Stage1 checkpoint: {len(missing)} missing keys, {len(unexpected)} unexpected keys')
if missing:
    scene_keys = [k for k in missing if 'scene_encoder' in k or 'sceneco' in k]
    other_keys = [k for k in missing if k not in scene_keys]
    print(f'  Missing scene-related: {len(scene_keys)}, other: {len(other_keys)}')

trainer.model.to('cuda:0')
trainer.train()
" 2>&1 | tee "$STAGE2_OUTPUT/train_stage2.log"

echo ""
echo "=============================================="
echo " TRUMAN Two-Stage SceneCo training complete!"
echo " Stage 1 checkpoint: $STAGE1_CKPT"
echo " Stage 2 checkpoint: $STAGE2_OUTPUT/checkpoints/best_checkpoint.pt"
echo "=============================================="
