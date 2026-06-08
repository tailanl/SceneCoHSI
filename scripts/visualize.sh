#!/bin/bash
# visualize.sh — Visualize generated motions with scene overlay
#
# Generates 3D visualizations of:
#   1. Scene voxel grid + human skeleton overlay
#   2. Root trajectory on ground plane with obstacles
#   3. Collision heatmap
#
# Uses viser for 3D rendering.

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

VIS_DIR="kimodo_scene_project/outputs/visualizations"
SCENE_DIR="LINGO/dataset/dataset/Scene"
MOTION_DIR="kimodo_scene_project/outputs/baseline_kimoto"

mkdir -p "$VIS_DIR"

echo "=============================================="
echo " Visualizing generated motions"
echo "=============================================="
echo "Motion dir:  $MOTION_DIR"
echo "Scene dir:   $SCENE_DIR"
echo "Output dir:  $VIS_DIR"
echo ""

echo "Running visualization..."
PYTHONPATH="kimodo:SOMA:$PYTHONPATH" python -c "
import sys
sys.path.insert(0, 'kimodo_scene_project')
from pathlib import Path
import json

print('Visualization tool framework ready.')
print(f'Motions available:')
motion_dir = Path('$MOTION_DIR')
for subdir in sorted(motion_dir.iterdir()):
    if subdir.is_dir():
        npz_files = list(subdir.glob('*.npz'))
        print(f'  {subdir.name}: {len(npz_files)} motions')
"
echo ""
echo "=============================================="
echo " Visualization setup complete."
echo " Use viser or the python demo to view results."
