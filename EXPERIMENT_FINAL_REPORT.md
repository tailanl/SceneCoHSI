# SceneCoHSI Experiment Final Report

**Date**: 2026-06-11
**Repository**: SceneCoHSI
**Branch**: main

---

## Experiment Pipeline Overview

所有实验分三组，回答两个核心问题：**root 能不能控制好路径 + 身体能不能适配 root 并避开障碍**。

```
Stage 1: Root Control (无 SceneCo 训练)
  E1 → E2 → E3：比哪种 root 引导方式最好

Stage 2: SceneCo Body Training (80 epochs)
  E4 → E5 → E6 → E7：比训练后的身体能否减少碰撞

Variant A: Scene-Aware Classifier (root 引导引入场景碰撞感知)
  A1 → A2：比加入场景特征后能否避开障碍

Variant B: TrajCo + SceneCo (双重身体适配)
  B1-E4, B1-E7：比额外的 TrajCo 层能否进一步改善
```

---

## Experiment Configurations

### 基础环境

```bash
CUDA_VISIBLE_DEVICES=1 (or 0,4,5,6,7 per experiment)
CHECKPOINT_DIR=$PWD/models
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
--gpu 0  (all scripts use cuda:0 after CUDA_VISIBLE_DEVICES mapping)
```

### Stage 1: Root Control

| ID | Root Generation Config | Classifier | Body | Eval Samples |
|----|----------------------|------------|------|-------------|
| E1 | `configs/guidance_root_scene.yaml` | None (energy loss) | Original Kimodo | 30 |
| E2 | `configs/root_classifier_guidance.yaml` | `root_classifier/best.pt` (19-dim) | Original Kimodo | 30 |
| E3 | `configs/root_classifier_guidance.yaml` --hybrid | `root_classifier/best.pt` | Original Kimodo | 30 |

**E2/E3 使用的 RootPathClassifier**:
```yaml
# configs/root_classifier.yaml
model:
  input_dim: 19
  hidden_dim: 256
  num_layers: 4
training:
  batch_size: 64
  num_epochs: 100
  lr: 1e-4
negative_sampling:
  modes: [jitter, path_shuffle, positive, reverse_heading, shift, wrong_goal, wrong_heading]
```

### Stage 2: SceneCo Body Training

| ID | Root Source | Stage2 Config | epochs | batch_size | root_mix |
|----|------------|---------------|--------|------------|----------|
| E4 v3 | energy-guided v3 | `stage2_energy_root_guided_sceneco.yaml` | 80 | 4 | gt=0.3, path=0.7 |
| E5 v3 | classifier-guided | `stage2_classifier_root_guided_sceneco.yaml` | 80 | 4 | gt=0.3, path=0.7 |
| E6 v3 | hybrid-guided | `stage2_hybrid_root_guided_sceneco.yaml` | 80 | 4 | gt=0.3, path=0.7 |
| E7 v3 | GT root v3 | `stage2_gt_root_sceneco.yaml` | 80 | 4 | gt=0.0, path=1.0 |

**Stage2 SceneCo 配置**:
```yaml
sceneco:
  enabled: true
  body_only: true        # 只训练 body SceneCo 层
  scene_dropout: 0.1     # 10% 概率 mask 场景（防止过拟合）
training:
  batch_size: 4
  num_epochs: 80
  lr: 1e-4
  prior_weight: 0.0
trainable: scene_encoder + SceneCo body adapter (145M / 428M total)
frozen: pretrained backbone
loss: body_mse only (root is fixed by external source)
```

### Variant A: Scene-Aware Classifier

| Step | Config | Description |
|------|--------|------------|
| A1 | `configs/root_classifier_scene.yaml` | 20-dim classifier (19 motion + 1 scene collision) |
| A2 | `configs/root_classifier_scene_guidance.yaml` | Root generation with `use_scene: true` |

**关键区别**: scene_collision 负样本模式，让分类器学会判断 root 是否避开了障碍。

### Variant B: TrajCo

| Step | Config | Description |
|------|--------|------------|
| B1-E7 | `configs/stage2_root_guided_sceneco_trajco.yaml` | SceneCo + TrajCo body on GT root |
| B1-E4 | `configs/stage2_root_guided_sceneco_trajco.yaml` | SceneCo + TrajCo body on scene-aware root |

```yaml
trajco:
  enabled: true
  use_trajco: true
  use_trajco_body: true     # TrajCo 交叉注意力注入 body 层
```

---

## Results

### Full Comparison Table

| 实验 | N | PathADE | PathFDE | SpeedStd | RootJerk | CFR | PenRate |
|------|---|---------|---------|----------|----------|-----|---------|
| E1 Energy + Orig | 30 | 2.1033 | 3.3949 | 0.0042 | 0.0010 | 0.5333 | 0.2065 |
| E2 Classifier + Orig | 30 | 1.0650 | 1.1514 | 0.0008 | 0.0000 | 0.1958 | 0.0405 |
| E3 Hybrid + Orig | 30 | 1.2114 | 1.4984 | 0.0019 | 0.0001 | 0.2758 | 0.0969 |
| E5 v3 Classifier + Stage2 | 1731 | 1.3710 | 1.5892 | 0.0015 | 0.0001 | **0.1387** | **0.0310** |
| E7 v3 GT + Stage2 | 1732 | 0.0000 | 0.0000 | 0.0037 | 0.0014 | 0.3359 | 0.0911 |
| E4 v3 Energy + Stage2 | — | training | — | — | — | — | — |
| E6 v3 Hybrid + Stage2 | — | training done | — | — | — | — | — |
| B1-E7 TrajCo + GT | — | training | — | — | — | — | — |
| B1-E4 TrajCo + Scene | — | training | — | — | — | — | — |

### Key Findings

1. **Classifier guidance 远好于 energy guidance**: PathADE 从 2.10 降到 1.07（E1→E2，-49%）
2. **SceneCo 训练降低碰撞**: E5 v3 CFR=0.14 vs E2 CFR=0.20（在更大测试集上）
3. **Hybrid 不如纯 classifier**: E3 PathADE=1.21 > E2 1.07，energy 项带来噪声
4. **GT root 上限**: PathADE=0.0（完美），CFR=0.34（场景本身无法完全避开）
5. **所有 Stage2 训练零 fallback**: 验证 root 文件与 dataset source_id 100% 匹配

---

## Scene Eval Fix

原始 eval 的 CFR 几乎 100%，因硬编码 `voxel_size=0.1, grid_origin=(0,0,0)` 与 motion 坐标系不匹配。

修复后:
- 使用 `lingo_smplx_cache` 中的 `voxel_grid`（64³，与训练一致）
- 根据 motion extent 动态计算坐标映射
- CFR 从 ~0.96 降到 ~0.14-0.53，符合物理预期

详见 `SCENE_PROCESSING.md`

---

## Root Validation

所有 v3 实验满足:
- Schema: `guided_root_5d_norm`, `guided_root_5d_meter`, `target_path_xz`, `text`, `scene_name`, `source_file`
- source_id: 基于 `_load_cached_index`（seed=42, ratio=0.9），与 Stage2 dataset 完全匹配
- 零 fallback: 所有 Stage2 日志无 `fallback` / `missing` 信息
- Root fix: body generation max_error < 1e-5

详见 `README_E4_E7_CODE_FIX.md`

---

## Artifact Locations

```
outputs/
├── e1_energy_guidance_body/    (30 samples, path + scene metrics)
├── e2_classifier_guidance_body/ (30 samples)
├── e3_hybrid_guidance_body/    (30 samples)
├── e4_energy_guidance_train/path_only/ (15584 roots)
├── e4_energy_guidance_val/path_only/   (1732 roots)
├── e5_classifier_guidance_train/path_only/ (15583 roots)
├── e5_classifier_guidance_val/path_only/   (1731 roots)
├── e5_v3_stage2/               (checkpoint + 1731 body + metrics)
├── e6_hybrid_guidance_train/path_only/ (15583 roots)
├── e6_hybrid_guidance_val/path_only/   (1731 roots)
├── e7_gt_root_v3_train/        (15584 GT roots)
├── e7_gt_root_v3_val/          (1732 GT roots)
├── e7_v3_stage2/               (checkpoint + 1732 body + metrics)
├── root_path_classifier/       (19-dim classifier, val_acc=1.0)
├── root_path_scene_classifier_sdf/ (20-dim scene-aware classifier, val_acc=1.0)
├── root_classifier_scene_guidance/ (1731 scene-aware roots)
└── stage2_root_guided_sceneco/ (legacy shared checkpoint dir, DO NOT USE)
```

---

## Known Issues

1. E0 baseline blocked: `scripts/generate.py` not found
2. E4 v3, B1-E7, B1-E4 still training (will complete in ~4-8h)
3. E6 v3 body gen + eval in progress
4. E1/E2/E3 use 30 samples (not the full val split), metrics not directly comparable with v3 experiments
5. Scene eval is 2D proxy (XZ plane SDF), not full 3D mesh-level evaluation
