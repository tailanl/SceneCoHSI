#!/bin/bash
# Quick sanity check: verify freeze logic for CaKey two-stage training.
# Tests Stage 1 (CaKey only trainable) and Stage 2 (SceneCo only trainable, CaKey frozen).
# Uses GPU 2, single GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONHASHSEED=0
export CHECKPOINT_DIR="models/Kimodo-SOMA-RP-v1.1"
export HF_HOME=".hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
export TEXT_ENCODERS_DIR="text_encoders"
export TEXT_ENCODER_MODE="local"
export TEXT_ENCODER_DEVICE="cpu"
export CUDA_VISIBLE_DEVICES="2"

echo "=============================================="
echo " Freeze Logic Sanity Check"
echo "=============================================="
echo ""

python -c "
import sys, os
sys.path.insert(0, 'kimodo')

import torch
import torch.nn as nn

# === Stage 1: CaKey Root+Body ===
print('>>> Stage 1: CaKey Root+Body')

from kimodo.model import load_model as load_kimodo_model
from kimodo_sceneco.model import KimodoSceneCo

# Load pretrained
class DummyTE:
    def __call__(self, t): raise RuntimeError
    def to(self, d): return self
    def eval(self): pass

print('  Loading Kimodo pretrained...')
kimodo_pt = load_kimodo_model('Kimodo-SOMA-RP-v1.1', device='cpu', text_encoder=DummyTE())
denoiser = kimodo_pt.denoiser.model
te = kimodo_pt.text_encoder
del kimodo_pt

scene_enc_cfg = {
    'voxel_size': (64,64,64), 'patch_size': (8,8,8), 'in_channels': 1,
    'd_model': 256, 'num_heads': 4, 'num_layers': 4, 'ff_dim': 512,
    'sceneco_dropout': 0.1, 'use_dual_vit': True, 'root_voxel_mode': 'full',
}

print('  Building model with CaKey root+body...')
model = KimodoSceneCo(
    denoiser=denoiser, text_encoder=te, num_base_steps=1000,
    scene_encoder_type='voxel_vit', scene_encoder_config=scene_enc_cfg,
    device='cpu', cfg_type='nocfg',
    use_in_root_model=False, use_in_body_model=False,
    use_cakey_root=True, use_cakey_body=True, cakey_hidden_dim=2048,
)

model.freeze_for_cakey()

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
frozen_cakey = [n for n, p in model.named_parameters() if 'cakey' in n and not p.requires_grad]
frozen_sceneco = [n for n, p in model.named_parameters() if 'sceneco' in n and p.requires_grad]
frozen_voxel = [n for n, p in model.named_parameters() if 'voxel_vit' in n and p.requires_grad]
frozen_scene_enc = [n for n, p in model.named_parameters() if 'scene_encoder' in n and p.requires_grad]

print(f'  Trainable params: {len(trainable)}')
print(f'  Cakey params: trainable={len([n for n in trainable if \"cakey\" in n])}, frozen={len(frozen_cakey)}')
print(f'  SceneCo params (should be frozen): trainable={len(frozen_sceneco)}')
print(f'  VoxelViT params (should be frozen): trainable={len(frozen_voxel)}')
print(f'  SceneEncoder params (should be frozen): trainable={len(frozen_scene_enc)}')

errors = []
if frozen_cakey:
    errors.append(f'ERROR: {len(frozen_cakey)} CaKey params are FROZEN (should be trainable)')
if frozen_sceneco:
    errors.append(f'ERROR: {len(frozen_sceneco)} SceneCo params are TRAINABLE (should be frozen)')
if frozen_voxel:
    errors.append(f'ERROR: {len(frozen_voxel)} VoxelViT params are TRAINABLE (should be frozen)')
if frozen_scene_enc:
    errors.append(f'ERROR: {len(frozen_scene_enc)} SceneEncoder params are TRAINABLE (should be frozen)')

if errors:
    print('')
    print('  STAGE 1 FAILED:')
    for e in errors: print(f'    {e}')
else:
    print('')
    print('  STAGE 1 PASSED ✓')
    for n in trainable[:3]:
        print(f'    trainable: {n}')
    print(f'    ... ({len(trainable)} total)')

# Save a dummy checkpoint
print('')
print('  Saving dummy Stage 1 checkpoint...')
import tempfile
tmpdir = tempfile.mkdtemp()
ckpt_path = os.path.join(tmpdir, 'test_ckpt.pt')
torch.save({'model_state_dict': model.state_dict(), 'global_step': 1}, ckpt_path)

# Clear everything
del model, denoiser, te

# === Stage 2: CaKey+SceneCo Root+Body with Dual ViT ===
print('')
print('>>> Stage 2: CaKey+SceneCo Root+Body (Dual ViT)')

print('  Loading Kimodo pretrained again...')
kimodo_pt = load_kimodo_model('Kimodo-SOMA-RP-v1.1', device='cpu', text_encoder=DummyTE())
denoiser = kimodo_pt.denoiser.model
te = kimodo_pt.text_encoder
del kimodo_pt

print('  Building model with CaKey + SceneCo + Dual ViT...')
model2 = KimodoSceneCo(
    denoiser=denoiser, text_encoder=te, num_base_steps=1000,
    scene_encoder_type='voxel_vit', scene_encoder_config=scene_enc_cfg,
    device='cpu', cfg_type='scene_separated',
    use_in_root_model=True, use_in_body_model=True,
    use_cakey_root=True, use_cakey_body=True, cakey_hidden_dim=2048,
)

print('  Loading Stage 1 CaKey checkpoint...')
ckpt = torch.load(ckpt_path, map_location='cpu')
missing, unexpected = model2.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f'  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')

model2.freeze_for_sceneco()

trainable2 = [n for n, p in model2.named_parameters() if p.requires_grad]
frozen_cakey2 = [n for n, p in model2.named_parameters() if 'cakey' in n and p.requires_grad]
sceneco_trainable = [n for n in trainable2 if 'sceneco' in n]
scene_enc_trainable = [n for n in trainable2 if 'scene_encoder' in n]
cakey_in_trainable = [n for n in trainable2 if 'cakey' in n]

print(f'  Trainable params: {len(trainable2)}')
print(f'  Cakey params: trainable={len(cakey_in_trainable)} (should be 0), frozen={len(frozen_cakey2)}')
print(f'  SceneCo trainable: {len(sceneco_trainable)}')
print(f'  SceneEncoder+VoxelViT trainable: {len(scene_enc_trainable)}')

errors2 = []
if cakey_in_trainable:
    errors2.append(f'ERROR: {len(cakey_in_trainable)} CaKey params TRAINABLE (should be frozen)')
if not sceneco_trainable:
    errors2.append(f'ERROR: 0 SceneCo params trainable (should be >0)')
if not scene_enc_trainable:
    errors2.append(f'ERROR: 0 SceneEncoder/VoxelViT params trainable (should be >0)')

if errors2:
    print('')
    print('  STAGE 2 FAILED:')
    for e in errors2: print(f'    {e}')
else:
    print('')
    print('  STAGE 2 PASSED ✓')
    for n in trainable2[:5]:
        print(f'    trainable: {n}')
    print(f'    ... ({len(trainable2)} total)')

# Summary
print('')
print('==============================================')
if errors or errors2:
    print(' RESULT: FAILED')
else:
    print(' RESULT: ALL PASSED ✓')
    print('         Stage1: only CaKey trainable')
    print('         Stage2: only SceneCo+VoxelViT trainable, CaKey frozen')
print('==============================================')
"
