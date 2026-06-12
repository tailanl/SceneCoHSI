#!/usr/bin/env bash
# ============================================================
# E1-E10 全实验重跑 — tmux 一键启动
#
# Usage:
#   bash scripts/rerun_all_experiments.sh
#     → 启动 prep 窗口，修复镜像数据 + 重生成 raw3d 根轨迹
#     → 创建 E1-E10 各实验窗口
#
#   tmux attach -t experiments   # 查看进度
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="experiments"

export CHECKPOINT_DIR="$PROJECT_DIR/models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_DIR"

BODY_STEPS=50
CFG_WEIGHT="2.0 2.0"
STAGE2_EPOCHS=80
BATCH_SIZE=4
NUM_WORKERS=4

# ── 辅助：在每个窗口执行命令前设置 GPU ─────────────────
cmd_gpu() {
    local gpu=$1; shift
    echo "export CUDA_VISIBLE_DEVICES=$gpu && $*"
}

# ── 创建 tmux 会话 ──────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n "prep" -x 200 -y 60

# ============================================================
# Window 0: Prep — 修复镜像数据 + 重新生成 raw3d 根轨迹
# ============================================================
tmux send-keys -t "$SESSION:prep" "
echo '=== PREP: Fix mirror data + regenerate raw3d roots ==='
date

# 修复所有现有 root 目录中的镜像场景名
python scripts/fix_lingo_mirror_data.py \
  --cache_dir lingo_smplx_cache \
  --scene_dir LINGO/dataset/dataset/Scene \
  --dirs \
    outputs/e1_energy_guidance_root \
    outputs/e2_classifier_guidance_root \
    outputs/e3_hybrid_guidance_root \
    outputs/e5_classifier_guidance_train/path_only \
    outputs/e5_classifier_guidance_val/path_only \
    outputs/e6_hybrid_guidance_train/path_only \
    outputs/e6_hybrid_guidance_val/path_only \
    outputs/e7_gt_root_v3_train \
    outputs/e7_gt_root_v3_val \
  --report_csv outputs/mirror_fix_rerun.csv \
  2>&1 | tee outputs/mirror_fix_rerun.log

# 重新生成 E8/E9/E10 raw3d 根轨迹
bash scripts/rerun_raw3d_root_data_after_mirror_fix.sh \
  2>&1 | tee outputs/rerun_raw3d_after_mirror_fix.log

echo '=== PREP DONE ==='
date
touch /tmp/experiments_prep_done
" C-m

# ============================================================
# Window 1: E1 — EnergyGuidance + Original Body (GPU 7)
# ============================================================
tmux new-window -t "$SESSION" -n "E1"
tmux send-keys -t "$SESSION:E1" "
export CUDA_VISIBLE_DEVICES=7
echo '=== E1: EnergyGuidance + Original Body ==='
ROOT=outputs/e1_energy_guidance_root
BODY=outputs/e1_energy_guidance_body
mkdir -p \$ROOT \$BODY

# 生成根轨迹 (val split, scene guidance)
python scripts/generate_root_guidance.py --output_dir \$ROOT --split val --scene_guidance --gpu 0 2>&1 | tee \$ROOT/generate.log

# 身体生成 (原始 Kimodo)
python scripts/generate_body_from_root.py --root_dir \$ROOT --output_dir \$BODY --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log

# 评估
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e1_energy_guidance_original_body
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e1_energy_guidance_original_body
echo '=== E1 DONE ==='
date
" C-m

# ============================================================
# Window 2: E2 — ClassifierGuidance + Original Body (GPU 7)
# ============================================================
tmux new-window -t "$SESSION" -n "E2"
tmux send-keys -t "$SESSION:E2" "
export CUDA_VISIBLE_DEVICES=7
echo '=== E2: ClassifierGuidance + Original Body ==='
ROOT=outputs/e2_classifier_guidance_root
BODY=outputs/e2_classifier_guidance_body
mkdir -p \$ROOT \$BODY

python scripts/generate_root_classifier_guidance.py --classifier_ckpt outputs/root_path_classifier/best.pt --output_dir \$ROOT --split val --gpu 0 2>&1 | tee \$ROOT/generate.log
python scripts/generate_body_from_root.py --root_dir \$ROOT --output_dir \$BODY --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e2_classifier_guidance_original_body
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e2_classifier_guidance_original_body
echo '=== E2 DONE ==='
date
" C-m

# ============================================================
# Window 3: E3 — HybridGuidance + Original Body (GPU 7)
# ============================================================
tmux new-window -t "$SESSION" -n "E3"
tmux send-keys -t "$SESSION:E3" "
export CUDA_VISIBLE_DEVICES=7
echo '=== E3: HybridGuidance + Original Body ==='
ROOT=outputs/e3_hybrid_guidance_root
BODY=outputs/e3_hybrid_guidance_body
mkdir -p \$ROOT \$BODY

python scripts/generate_root_classifier_guidance.py --classifier_ckpt outputs/root_path_classifier/best.pt --output_dir \$ROOT --split val --hybrid --gpu 0 2>&1 | tee \$ROOT/generate.log
python scripts/generate_body_from_root.py --root_dir \$ROOT --output_dir \$BODY --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e3_hybrid_guidance_original_body
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e3_hybrid_guidance_original_body
echo '=== E3 DONE ==='
date
" C-m

# ============================================================
# Window 4: E4 — EnergyGuidance + Stage2 SceneCo (GPU 0)
# ============================================================
tmux new-window -t "$SESSION" -n "E4"
tmux send-keys -t "$SESSION:E4" "
export CUDA_VISIBLE_DEVICES=0
echo '=== E4: EnergyGuidance + Stage2 SceneCo ==='
ROOT_TRAIN=outputs/e4_energy_guidance_train/path_only
ROOT_VAL=outputs/e4_energy_guidance_val/path_only
STAGE2=outputs/e4_v3_stage2
BODY=\$STAGE2/val_gen
mkdir -p \$ROOT_TRAIN \$ROOT_VAL \$STAGE2 \$BODY

python scripts/generate_root_guidance.py --output_dir \$ROOT_TRAIN --split train --scene_guidance --gpu 0 2>&1 | tee \$ROOT_TRAIN/generate.log
python scripts/generate_root_guidance.py --output_dir \$ROOT_VAL --split val --scene_guidance --gpu 0 2>&1 | tee \$ROOT_VAL/generate.log

python train/train_stage2_root_guided_sceneco.py configs/stage2_energy_root_guided_sceneco.yaml \
  --gpu 0 --output_dir \$STAGE2 \
  --path_guided_root_dir \$ROOT_TRAIN --path_scene_guided_root_dir \$ROOT_TRAIN \
  --val_root_dir \$ROOT_VAL \
  --root_mix_gt 0.3 --root_mix_path 0.0 --root_mix_scene 0.7 \
  --num_epochs $STAGE2_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
  2>&1 | tee \$STAGE2/train.log

CKPT=\$(ls \$STAGE2/checkpoints/best_checkpoint.pt 2>/dev/null || echo '')
if [ -n \"\$CKPT\" ]; then
  python scripts/generate_body_from_root.py --root_dir \$ROOT_VAL --output_dir \$BODY --checkpoint \$CKPT --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
fi
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e4_energy_guidance_stage2_sceneco
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e4_energy_guidance_stage2_sceneco
echo '=== E4 DONE ==='
date
" C-m

# ============================================================
# Window 5: E5 — ClassifierGuidance + Stage2 SceneCo (GPU 1)
# ============================================================
tmux new-window -t "$SESSION" -n "E5"
tmux send-keys -t "$SESSION:E5" "
export CUDA_VISIBLE_DEVICES=1
echo '=== E5: ClassifierGuidance + Stage2 SceneCo ==='
ROOT_TRAIN=outputs/e5_classifier_guidance_train/path_only
ROOT_VAL=outputs/e5_classifier_guidance_val/path_only
STAGE2=outputs/e5_v3_stage2
BODY=\$STAGE2/val_gen
mkdir -p \$STAGE2 \$BODY

python train/train_stage2_root_guided_sceneco.py configs/stage2_classifier_root_guided_sceneco.yaml \
  --gpu 0 --output_dir \$STAGE2 \
  --path_guided_root_dir \$ROOT_TRAIN --path_scene_guided_root_dir \$ROOT_TRAIN \
  --val_root_dir \$ROOT_VAL \
  --root_mix_gt 0.3 --root_mix_path 0.0 --root_mix_scene 0.7 \
  --num_epochs $STAGE2_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
  2>&1 | tee \$STAGE2/train.log

CKPT=\$(ls \$STAGE2/checkpoints/best_checkpoint.pt 2>/dev/null || echo '')
if [ -n \"\$CKPT\" ]; then
  python scripts/generate_body_from_root.py --root_dir \$ROOT_VAL --output_dir \$BODY --checkpoint \$CKPT --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
fi
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e5_classifier_guidance_stage2_sceneco
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e5_classifier_guidance_stage2_sceneco
echo '=== E5 DONE ==='
date
" C-m

# ============================================================
# Window 6: E6 — HybridGuidance + Stage2 SceneCo (GPU 2)
# ============================================================
tmux new-window -t "$SESSION" -n "E6"
tmux send-keys -t "$SESSION:E6" "
export CUDA_VISIBLE_DEVICES=2
echo '=== E6: HybridGuidance + Stage2 SceneCo ==='
ROOT_TRAIN=outputs/e6_hybrid_guidance_train/path_only
ROOT_VAL=outputs/e6_hybrid_guidance_val/path_only
STAGE2=outputs/e6_v3_stage2
BODY=\$STAGE2/val_gen
mkdir -p \$STAGE2 \$BODY

python train/train_stage2_root_guided_sceneco.py configs/stage2_hybrid_root_guided_sceneco.yaml \
  --gpu 0 --output_dir \$STAGE2 \
  --path_guided_root_dir \$ROOT_TRAIN --path_scene_guided_root_dir \$ROOT_TRAIN \
  --val_root_dir \$ROOT_VAL \
  --root_mix_gt 0.3 --root_mix_path 0.0 --root_mix_scene 0.7 \
  --num_epochs $STAGE2_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
  2>&1 | tee \$STAGE2/train.log

CKPT=\$(ls \$STAGE2/checkpoints/best_checkpoint.pt 2>/dev/null || echo '')
if [ -n \"\$CKPT\" ]; then
  python scripts/generate_body_from_root.py --root_dir \$ROOT_VAL --output_dir \$BODY --checkpoint \$CKPT --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
fi
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e6_hybrid_guidance_stage2_sceneco
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e6_hybrid_guidance_stage2_sceneco
echo '=== E6 DONE ==='
date
" C-m

# ============================================================
# Window 7: E7 — GTRoot + Stage2 SceneCo (GPU 3)
# ============================================================
tmux new-window -t "$SESSION" -n "E7"
tmux send-keys -t "$SESSION:E7" "
export CUDA_VISIBLE_DEVICES=3
echo '=== E7: GTRoot + Stage2 SceneCo ==='
ROOT_TRAIN=outputs/e7_gt_root_v3_train
ROOT_VAL=outputs/e7_gt_root_v3_val
STAGE2=outputs/e7_v3_stage2
BODY=\$STAGE2/val_gen
mkdir -p \$STAGE2 \$BODY

python train/train_stage2_root_guided_sceneco.py configs/stage2_gt_root_sceneco.yaml \
  --gpu 0 --output_dir \$STAGE2 \
  --path_guided_root_dir \$ROOT_TRAIN --path_scene_guided_root_dir \$ROOT_TRAIN \
  --val_root_dir \$ROOT_VAL \
  --root_mix_gt 1.0 --root_mix_path 0.0 --root_mix_scene 0.0 \
  --num_epochs $STAGE2_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
  2>&1 | tee \$STAGE2/train.log

CKPT=\$(ls \$STAGE2/checkpoints/best_checkpoint.pt 2>/dev/null || echo '')
if [ -n \"\$CKPT\" ]; then
  python scripts/generate_body_from_root.py --root_dir \$ROOT_VAL --output_dir \$BODY --checkpoint \$CKPT --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
fi
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e7_gt_root_stage2_sceneco
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e7_gt_root_stage2_sceneco
echo '=== E7 DONE ==='
date
" C-m

# ============================================================
# Window 8: E8 — Classifier+Raw3d + Stage2 SceneCo (GPU 4)
# ============================================================
tmux new-window -t "$SESSION" -n "E8"
tmux send-keys -t "$SESSION:E8" "
export CUDA_VISIBLE_DEVICES=4
echo '=== E8: Classifier+Raw3d + Stage2 SceneCo ==='
ROOT_TRAIN=outputs/e8_classifier_raw3d_train
ROOT_VAL=outputs/e8_classifier_raw3d_val
STAGE2=outputs/e8_classifier_raw3d_stage2
BODY=\$STAGE2/val_gen
mkdir -p \$ROOT_TRAIN \$ROOT_VAL \$STAGE2 \$BODY

echo '[E8] Postprocessing roots...'
python scripts/postprocess_root_raw3d.py --input_dir outputs/e5_classifier_guidance_train/path_only --output_dir \$ROOT_TRAIN --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0
python scripts/postprocess_root_raw3d.py --input_dir outputs/e5_classifier_guidance_val/path_only --output_dir \$ROOT_VAL --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0

echo '[E8] Fixing mirror data...'
python scripts/fix_lingo_mirror_data.py --cache_dir lingo_smplx_cache --scene_dir LINGO/dataset/dataset/Scene --dirs \$ROOT_TRAIN \$ROOT_VAL --report_csv \$STAGE2/mirror_fix.csv

echo '[E8] Training Stage2 (80 epochs)...'
python train/train_stage2_root_guided_sceneco.py configs/stage2_classifier_root_guided_sceneco.yaml \
  --gpu 0 --output_dir \$STAGE2 --path_guided_root_dir \$ROOT_TRAIN --path_scene_guided_root_dir \$ROOT_TRAIN \
  --val_root_dir \$ROOT_VAL --root_mix_gt 0.3 --root_mix_path 0.0 --root_mix_scene 0.7 \
  --num_epochs $STAGE2_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
  2>&1 | tee \$STAGE2/train.log

CKPT=\$(ls \$STAGE2/checkpoints/best_checkpoint.pt 2>/dev/null || echo '')
if [ -n \"\$CKPT\" ]; then
  python scripts/generate_body_from_root.py --root_dir \$ROOT_VAL --output_dir \$BODY --checkpoint \$CKPT --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
fi
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e8_classifier_raw3d_stage2_sceneco
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e8_classifier_raw3d_stage2_sceneco
echo '=== E8 DONE ==='
date
" C-m

# ============================================================
# Window 9: E9 — Hybrid+Raw3d + Stage2 SceneCo (GPU 5)
# ============================================================
tmux new-window -t "$SESSION" -n "E9"
tmux send-keys -t "$SESSION:E9" "
export CUDA_VISIBLE_DEVICES=5
echo '=== E9: Hybrid+Raw3d + Stage2 SceneCo ==='
ROOT_TRAIN=outputs/e9_hybrid_raw3d_train
ROOT_VAL=outputs/e9_hybrid_raw3d_val
STAGE2=outputs/e9_hybrid_raw3d_stage2
BODY=\$STAGE2/val_gen
mkdir -p \$ROOT_TRAIN \$ROOT_VAL \$STAGE2 \$BODY

echo '[E9] Postprocessing roots...'
python scripts/postprocess_root_raw3d.py --input_dir outputs/e6_hybrid_guidance_train/path_only --output_dir \$ROOT_TRAIN --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0
python scripts/postprocess_root_raw3d.py --input_dir outputs/e6_hybrid_guidance_val/path_only --output_dir \$ROOT_VAL --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0

echo '[E9] Fixing mirror data...'
python scripts/fix_lingo_mirror_data.py --cache_dir lingo_smplx_cache --scene_dir LINGO/dataset/dataset/Scene --dirs \$ROOT_TRAIN \$ROOT_VAL --report_csv \$STAGE2/mirror_fix.csv

echo '[E9] Training Stage2 (80 epochs)...'
python train/train_stage2_root_guided_sceneco.py configs/stage2_hybrid_root_guided_sceneco.yaml \
  --gpu 0 --output_dir \$STAGE2 --path_guided_root_dir \$ROOT_TRAIN --path_scene_guided_root_dir \$ROOT_TRAIN \
  --val_root_dir \$ROOT_VAL --root_mix_gt 0.3 --root_mix_path 0.0 --root_mix_scene 0.7 \
  --num_epochs $STAGE2_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
  2>&1 | tee \$STAGE2/train.log

CKPT=\$(ls \$STAGE2/checkpoints/best_checkpoint.pt 2>/dev/null || echo '')
if [ -n \"\$CKPT\" ]; then
  python scripts/generate_body_from_root.py --root_dir \$ROOT_VAL --output_dir \$BODY --checkpoint \$CKPT --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
fi
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e9_hybrid_raw3d_stage2_sceneco
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e9_hybrid_raw3d_stage2_sceneco
echo '=== E9 DONE ==='
date
" C-m

# ============================================================
# Window 10: E10 — GT Projected + Stage2 SceneCo (GPU 6)
# ============================================================
tmux new-window -t "$SESSION" -n "E10"
tmux send-keys -t "$SESSION:E10" "
export CUDA_VISIBLE_DEVICES=6
echo '=== E10: GT Projected + Stage2 SceneCo ==='
ROOT_TRAIN=outputs/e10_gt_projected_train
ROOT_VAL=outputs/e10_gt_projected_val
STAGE2=outputs/e10_gt_projected_stage2
BODY=\$STAGE2/val_gen
mkdir -p \$ROOT_TRAIN \$ROOT_VAL \$STAGE2 \$BODY

echo '[E10] Postprocessing roots...'
python scripts/postprocess_root_raw3d.py --input_dir outputs/e7_gt_root_v3_train --output_dir \$ROOT_TRAIN --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0
python scripts/postprocess_root_raw3d.py --input_dir outputs/e7_gt_root_v3_val --output_dir \$ROOT_VAL --project_target_path --overwrite_root_keys --update_norm --clearance_m 0.04 --smooth_window 5 --gpu 0

echo '[E10] Fixing mirror data...'
python scripts/fix_lingo_mirror_data.py --cache_dir lingo_smplx_cache --scene_dir LINGO/dataset/dataset/Scene --dirs \$ROOT_TRAIN \$ROOT_VAL --report_csv \$STAGE2/mirror_fix.csv

echo '[E10] Training Stage2 (80 epochs)...'
python train/train_stage2_root_guided_sceneco.py configs/stage2_gt_root_sceneco.yaml \
  --gpu 0 --output_dir \$STAGE2 --path_guided_root_dir \$ROOT_TRAIN --path_scene_guided_root_dir \$ROOT_TRAIN \
  --val_root_dir \$ROOT_VAL --root_mix_gt 1.0 --root_mix_path 0.0 --root_mix_scene 0.0 \
  --num_epochs $STAGE2_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
  2>&1 | tee \$STAGE2/train.log

CKPT=\$(ls \$STAGE2/checkpoints/best_checkpoint.pt 2>/dev/null || echo '')
if [ -n \"\$CKPT\" ]; then
  python scripts/generate_body_from_root.py --root_dir \$ROOT_VAL --output_dir \$BODY --checkpoint \$CKPT --num_denoising_steps $BODY_STEPS --cfg_weight $CFG_WEIGHT --gpu 0 2>&1 | tee \$BODY/generate_body.log
fi
python eval/eval_path_metrics.py --pred_dir \$BODY --output_csv \$BODY/path_metrics.csv --method e10_gt_projected_stage2_sceneco
python eval/eval_sceneadapt_metrics.py --pred_dir \$BODY --output_csv \$BODY/scene_metrics.csv --method e10_gt_projected_stage2_sceneco
echo '=== E10 DONE ==='
date
" C-m

# ── 完成 ─────────────────────────────────────────────────
echo ""
echo "=================================="
echo " tmux session '$SESSION' created"
echo "   Prep  → fixing mirror + raw3d roots"
echo "   E1-E10 → one window per experiment"
echo ""
echo " Attach:  tmux attach -t $SESSION"
echo " List:    tmux list-windows -t $SESSION"
echo " Kill:    tmux kill-session -t $SESSION"
echo "=================================="
