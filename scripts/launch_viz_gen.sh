#!/bin/bash
set -euo pipefail

SESSION="viz_gen"
GPU=7
CONDA_ENV="kimodo"
KIMODO_DIR="/home/lzsh2025/kimodo-viser"
OUT_DIR="$KIMODO_DIR/kimodo_scene_project/outputs/viz_generated"

mkdir -p "$OUT_DIR"

tmux kill-session -t $SESSION 2>/dev/null || true
sleep 0.5

tmux new-session -d -s $SESSION -n viz
tmux send-keys -t $SESSION "conda activate $CONDA_ENV" Enter
tmux send-keys -t $SESSION "cd $KIMODO_DIR" Enter
tmux send-keys -t $SESSION "export CUDA_VISIBLE_DEVICES=$GPU" Enter
tmux send-keys -t $SESSION "export CHECKPOINT_DIR=$KIMODO_DIR/models" Enter
tmux send-keys -t $SESSION "export LOCAL_CACHE=true" Enter
tmux send-keys -t $SESSION "python -u kimodo_scene_project/scripts/visualize_generated_motion.py --experiments D E F --num_samples 3 --gpu 0 --output_dir kimodo_scene_project/outputs/viz_generated 2>&1 | tee $OUT_DIR/viz_gen.log" Enter

echo "Started viz_gen tmux session on GPU $GPU"
echo "Check: tmux attach -t $SESSION"
echo "Log: tail -f $OUT_DIR/viz_gen.log"
