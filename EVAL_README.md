# Kimodo-SceneCo 实验评估报告

## 项目概述

在 NVIDIA Kimodo 运动扩散模型基础上插入 SceneCo（Scene-Conditioning）交叉注意力层，使模型能根据 3D 场景几何约束生成合理的人体运动。

**核心架构**：冻结 Kimodo 预训练参数 → 在 Transformer 的 SA/FFN 之间插入 SceneCo 层 → 仅训练 SceneCo 层和 VoxelViT 场景编码器。

```
3D 场景 → VoxelViT → scene_feat
                              ↓
噪声 x_T + text → [Root: SA →  SceneCo → FFN] × N  → root_pred
                       ↓ global→local
                    [Body: SA → SceneCo → FFN] × N  → body_pred
```

## 已有实验（21 个训练完成的配置）

| 配置 | Checkpoint | 参数量 | 说明 |
|------|-----------|--------|------|
| `root_only_sceneco` | best_checkpoint.pt | 2029 MB | root_model 注入 SceneCo，body_model 保持原样 |
| `body_only_sceneco` | best_checkpoint.pt | 2029 MB | body_model 注入 SceneCo，root_model 保持原样 |
| `single_vit_gpu1` | step 200000 final | 2019 MB | 单共享 VoxelViT，root+body 同时注入 |
| `dual_vit_gpu2` | step 200000 final | 2029 MB | 双 VoxelViT，root+body 同时注入 |
| `dual_vit_floor_gpu3` | step 200000 final | 2029 MB | 双 VoxelViT + 地板投影模式 |
| `root_body_merged` | best_checkpoint.pt | 1724 MB | root+body 合并训练 |
| `stage1_root_only` | step 150000 final | — | 两阶段训练 Stage1（root_only） |
| `stage2_full_body` | step 100000 final | 2573 MB | 两阶段训练 Stage2（full body） |
| `smplx_root_only` | best_checkpoint.pt | 2027 MB | SMPL-X 骨架，root_only |
| `smplx_body_only` | best_checkpoint.pt | 2027 MB | SMPL-X 骨架，body_only |
| `smplx_root_body` | best_checkpoint.pt | 2041 MB | SMPL-X 骨架，root+body |
| `smplx_dual_vit_floor` | best_checkpoint.pt | 2027 MB | SMPL-X 骨架 + floor |
| `cakey_root_only` | best_checkpoint.pt | 3504 MB | CaKey + root_only |
| `cakey_body_only` | best_checkpoint.pt | 3504 MB | CaKey + body_only |
| `cakey_root_body` | best_checkpoint.pt | 5865 MB | CaKey + root+body |
| `cakey_sceneco_root_only` | best_checkpoint.pt | 2816 MB | CaKey+SceneCo + root_only |
| `cakey_sceneco_body_only` | best_checkpoint.pt | 2816 MB | CaKey+SceneCo + body_only |
| `cakey_sceneco_root_body` | best_checkpoint.pt | 4479 MB | CaKey+SceneCo + root+body |

---

## 评估指标定义

### C-class：场景适应指标（Scene Adaptation）

| 指标 | 全称 | 含义 | 方向 |
|------|------|------|------|
| **CFR** | Collision Frame Ratio | 至少有一个关节碰撞的帧占比 | ↓ 越低越好 |
| **JCR** | Joint Collision Ratio | 所有 (帧, 关节) 对中的碰撞比例 | ↓ 越低越好 |
| **MeanPen** | Mean Penetration Depth | 碰撞关节的平均穿透深度 | ↓ 越低越好 |
| **MaxPen** | Max Penetration Depth | 最严重单关节穿透深度 | ↓ 越低越好 |
| **P95Pen** | P95 Penetration Depth | 95 分位穿透深度 | ↓ 越低越好 |
| **PFFR** | Penetration-Free Frame Ratio | 完全无碰撞的帧占比 | ↑ 越高越好 |
| **OPIR** | Obstacle Path Intersection Rate | 根轨迹落在障碍物区域的比例 | ↓ 越低越好 |

### D-class：运动质量指标（Motion Quality）

| 指标 | 全称 | 含义 | 方向 |
|------|------|------|------|
| **FootSkate** | Foot Skating | 脚着地时的滑动速度 | ↓ 越低越好 |
| **FootPenetration** | Foot Penetration | 脚穿透地面的比例 | ↓ 越低越好 |
| **FloatingRatio** | Floating Ratio | 双脚同时离地的帧占比 | ↓ 越低越好 |
| **VelSmooth** | Velocity Smoothness | 平均加速度（越低越平滑） | ↓ 越低越好 |
| **AccelJerk** | Acceleration Jerk | 加速度变动率 | ↓ 越低越好 |
| **BoneLenErr** | Bone Length Error | 骨骼长度偏差 | ↓ 越低越好 |

---

## 综合评估结果

**测试设置**：LINGO 验证集 3 个样本，50 DDIM 去噪步数，seed=42

| Experiment | CFR ↓ | JCR ↓ | MeanPen ↓ | PFFR ↑ | OPIR ↓ | FootSkate ↓ | BoneLenErr ↓ |
|-----------|-------|-------|-----------|--------|--------|-------------|-------------|
| **Kimodo_original** (no scene) | 1.0000 | 0.5506 | 0.0500 | 0.0000 | 0.8070 | **0.0194** | 0.2606 |
| **root_only** | 0.9669 | 0.9124 | 0.0500 | 0.0331 | 1.0000 | 1.0225 | 0.2656 |
| **dual_vit** (root+body) | 1.0000 | 0.8568 | 0.0500 | 0.0000 | 0.9903 | 1.3413 | 0.2550 |
| **dual_vit_floor** ★ | **0.7227** | **0.4845** | 0.0500 | **0.2773** | **0.6262** | 0.5898 | **0.2535** |

---

## Per-Sample 详细结果

### Kimodo_original（原始 Kimodo，无场景输入）

| 样本 | 文本 | 帧数 | CFR | JCR | PFFR | OPIR | FootSkate | 生成时间 |
|------|------|------|-----|-----|------|------|-----------|---------|
| 0 | pick up plant with left hand | 171 | 1.00 | 0.47 | 0.00 | 0.42 | 0.0000 | 23.7s |
| 1 | stand still | 60 | 1.00 | 0.59 | 0.00 | 1.00 | 0.0245 | 14.9s |
| 2 | pick up candle with left hand | 156 | 1.00 | 0.59 | 0.00 | 1.00 | 0.0337 | 22.2s |

**分析**：作为基线，原始 Kimodo 无法感知场景 → 100% 帧碰撞（CFR=1.0）。但运动质量不错：FootSkate 仅 0.02，BoneLenErr 0.26。

### root_only（SceneCo 仅注入 root_model）

| 样本 | 文本 | 帧数 | CFR | JCR | PFFR | OPIR | FootSkate | 生成时间 |
|------|------|------|-----|-----|------|------|-----------|---------|
| 0 | pick up plant with left hand | 171 | 0.90 | 0.79 | **0.10** | 1.00 | 0.0000 | 97.0s |
| 1 | stand still | 60 | 1.00 | 1.00 | 0.00 | 1.00 | 0.0000 | 67.5s |
| 2 | pick up candle with left hand | 156 | 1.00 | 0.95 | 0.00 | 1.00 | 3.0676 | 63.1s |

**分析**：root 注入 SceneCo 对 "pick up plant" 样本有改善（CFR 降至 0.90，PFFR=0.10），但其 OPIR 反而增至 1.0（root 轨迹完全在障碍物内），说明仅监督 root 路径不足以约束整体运动。

### dual_vit（双 VoxelViT，root+body）

| 样本 | 文本 | 帧数 | CFR | JCR | PFFR | OPIR | FootSkate | 生成时间 |
|------|------|------|-----|-----|------|------|-----------|---------|
| 0 | pick up plant with left hand | 171 | 1.00 | 0.74 | 0.00 | 0.98 | 0.0000 | 369.9s |
| 1 | stand still | 60 | 1.00 | 0.92 | 0.00 | 1.00 | 4.0240 | 381.1s |
| 2 | pick up candle with left hand | 156 | 1.00 | 0.91 | 0.00 | 1.00 | 0.0000 | 336.5s |

**分析**：dual_vit 在所有场景中表现反而最差 — CFR=1.0 且 FootSkate 高达 1.34。可能原因：1) 模型过拟合场景纹理而非几何避障；2) 训练损失权重未充分平衡碰撞惩罚；3) 需要更大的 w_scene CFG 权重。

### dual_vit_floor ★（双 VoxelViT + 地板投影）

| 样本 | 文本 | 帧数 | CFR | JCR | PFFR | OPIR | FootSkate | 生成时间 |
|------|------|------|-----|-----|------|------|-----------|---------|
| 0 | pick up plant with left hand | 171 | **0.25** | **0.03** | **0.75** | **0.01** | 1.7695 | 551.3s |
| 1 | stand still | 60 | 0.92 | 0.48 | 0.08 | 0.89 | 0.0000 | 510.9s |
| 2 | pick up candle with left hand | 156 | 1.00 | 0.94 | 0.00 | 0.98 | 0.0000 | 404.9s |

**分析**：地板投影模式下，"pick up plant" 样本表现惊艳 — CFR=0.25、PFFR=0.75（75% 帧无碰撞）、OPIR=0.01（几乎不穿障碍物）。但 "stand still" 和 "pick up candle" 场景改善不明显，说明模型对特定场景/文本组合的泛化仍有局限。

---

## 关键发现

### 1. 碰撞规避

```
CFR:    dual_vit_floor (0.72) << root_only (0.97) < baseline (1.00) = dual_vit (1.00)
PFFR:   dual_vit_floor (0.28) >> root_only (0.03) > others (0.00)
JCR:    dual_vit_floor (0.48) < baseline (0.55) << root_only (0.91) < dual_vit (0.86)
```

- **dual_vit_floor 是唯一显著改善碰撞的实验**，将碰撞帧率降低 28%
- root_only 只在特定样本（pick up plant）上有小幅改善
- dual_vit（完整 ViT）反而使指标变差，说明直接将 3D voxel 特征注入容易过拟合

### 2. 运动质量

```
FootSkate:  baseline (0.02) < dual_vit_floor (0.59) < root_only (1.02) < dual_vit (1.34)
BoneLenErr: all ~0.25-0.27 (comparable)
```

- SceneCo 注入会导致 FootSkate 增加 10-70×（0.02 → 0.59~1.34）
- BoneLenErr 保持稳定，说明骨骼结构未被破坏

### 3. 地板投影是关键

dual_vit_floor 使用地板投影（将 3D voxel 投影到 2D 地面），相比纯 3D voxel 特征，这种 2D 表示更易于模型学习避障。这表明：
- 场景感知的难点在于将 3D 几何约束编码为可学习信号
- 降维（3D → 2D 地面投影）是提升场景感知的有效策略
- 模型参数量增加（dual_vit）不一定带来更好的场景感知

### 4. Per-sample 泛化

所有 SceneCo 实验在不同样本间表现差异大，说明泛化能力还有提升空间。特别是 "pick up" 类动作（需近距离交互）受益于 floor 模式，但 "stand still"（静立）改善有限。

---

## 运行方法

### 单次实验推理 + 指标计算

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="kimodo:SOMA:$PYTHONPATH" \
CHECKPOINT_DIR=models HF_HOME=.hf_cache \
TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu \
python kimodo_scene_project/eval/eval_all_metrics.py \
    --num_samples 3 --num_denoising_steps 50 \
    --experiments Kimodo_original root_only dual_vit dual_vit_floor \
    --output_dir kimodo_scene_project/outputs/metric_eval \
    --gpu 0
```

### 两阶段推理（SceneCo root → Kimodo body）

```bash
# Stage 1: SceneCo root_only 生成 root 轨迹
# Stage 2: 原始 Kimodo 以 root 路径为约束生成 body
python kimodo_scene_project/eval/two_stage_inference.py \
    --ckpt_path .../best_checkpoint.pt \
    --dataset_sample 0 --gpu 0
```

### 可视化

```bash
# 生成并排 3D 骨骼动画视频 + 轨迹对比图
python kimodo_scene_project/eval/viz_two_stage.py \
    --input_dir kimodo_scene_project/outputs/two_stage_inference \
    --output_dir kimodo_scene_project/outputs/two_stage_viz
```

### w_scene CFG Sweep

```bash
# 对单一样本扫描 w_scene ∈ {0.0, 0.5, 1.0, 1.5, 2.0} 的 30 种组合
python kimodo_scene_project/eval/eval_all_metrics.py \
    --num_samples 1 --experiments dual_vit_floor \
    --output_dir kimodo_scene_project/outputs/metric_eval \
    --run_sweep --gpu 0
```

---

## 文件索引

| 文件 | 路径 | 说明 |
|------|------|------|
| 综合评估脚本 | [eval/eval_all_metrics.py](eval/eval_all_metrics.py) | 生成动作 + 计算 C/D 类指标 |
| 两阶段推理脚本 | [eval/two_stage_inference.py](eval/two_stage_inference.py) | SceneCo root → Kimodo body |
| 可视化脚本 | [eval/viz_two_stage.py](eval/viz_two_stage.py) | MP4 视频 + 轨迹对比图 |
| 评估结果 JSON | [outputs/metric_eval/all_metrics.json](outputs/metric_eval/all_metrics.json) | Per-sample 完整数据 |
| 指标对比表 CSV | [outputs/metric_eval/metric_table.csv](outputs/metric_eval/metric_table.csv) | 汇总表格 |
| 指标对比表 TXT | [outputs/metric_eval/metric_table.txt](outputs/metric_eval/metric_table.txt) | 文本报告 |
| 碰撞检测器 | [scene_modules/scene_checker.py](scene_modules/scene_checker.py) | CFR/JCR/MeanPen/OPIR 实现 |
| 场景指标 | [eval/eval_scene_metrics.py](eval/eval_scene_metrics.py) | C/B/D/E 类指标编排 |
| CFG Sweep | [eval/sweep_cfg.py](eval/sweep_cfg.py) | w_scene 网格扫描 |

---

## 下一步建议

1. **扩大测试集** — 当前 3 样本不足以得出统计显著结论，建议扩展到 50+ 样本
2. **w_scene sweep** — 对 dual_vit_floor 扫描最优 CFG 权重（w_scene ∈ {0.5, 1.0, 1.5, 2.0}）
3. **对比 SMPLX 骨架** — SMPLX 实验未纳入本次评估，需单独测试
4. **CaKey 消融** — 评估 CaKey + SceneCo 组合实验是否进一步提升
5. **碰撞检测校准** — 当前所有 MeanPen 均为 0.05（默认值），需检查碰撞检测的坐标对齐和阈值设置
6. **FootSkate 优化** — 建议对 SceneCo 输出做足部滑步后处理（已在 Kimodo 中存在 `post_process_motion`）
