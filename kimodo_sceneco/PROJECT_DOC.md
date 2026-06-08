# Kimodo-SceneCo: 场景感知的人体运动生成

## 项目概述

Kimodo-SceneCo 是基于 NVIDIA Kimodo 运动扩散模型扩展的场景感知运动生成系统。通过在 Kimodo 的两阶段 Transformer 去噪器中插入 SceneCo（Scene-conditioning）交叉注意力层，并使用 Voxel ViT 编码 3D 场景信息，使模型能够根据场景约束生成合理的运动。

核心思想来源于 SceneAdapt（arXiv 2510.13044），但跳过了 CaKey 层（因为 Kimodo 已有约束系统），直接在 Self-Attention 和 FFN 之间插入场景 Cross-Attention。

---

## 目录结构

```
kimodo/kimodo_sceneco/
├── model/                          # 模型代码
│   ├── __init__.py                 # 导出 KimodoSceneCo, VoxelViT, BBoxEncoder
│   ├── backbone.py                 # ★ 核心改动：SceneCo Transformer 层
│   ├── cfg.py                      # ★ 扩展 CFG：scene_separated 四分支
│   ├── diffusion.py                # 扩散过程（与原版相同）
│   ├── kimodo_model.py             # ★ KimodoSceneCo 主模型类
│   ├── scene_encoder.py            # ★ 新增：VoxelViT + BBoxEncoder
│   ├── twostage_denoiser.py        # ★ 传递 scene_feat 到两阶段去噪器
│   ├── llm2vec/                    # 文本编码器（与原版相同）
│   ├── loading.py                  # 模型加载（与原版相同）
│   ├── load_model.py               # 模型加载入口（与原版相同）
│   ├── metrics/                    # 评估指标
│   │   ├── tmr.py                  # TMR: R-Precision, FID
│   │   ├── foot_skate.py           # 脚滑指标
│   │   └── constraints.py          # 约束跟随指标
│   └── ...
├── train/                          # ★ 新增：训练代码
│   ├── __init__.py
│   ├── dataset.py                  # LINGO 场景-动作数据集
│   ├── train.py                    # 训练脚本（含损失函数、验证逻辑）
│   └── eval.py                     # 评估脚本（TSTMotion 指标）
└── scripts/                        # ★ 新增：启动脚本
    ├── train.sh                    # 训练启动脚本
    └── eval.sh                     # 评估启动脚本
```

---

## 架构设计

### 1. 原始 Kimodo 架构

```
文本 → LLM2Vec → text_feat
噪声 x_T → [两阶段 Transformer 去噪器] ← timestep
              ├── Stage 1: Root Model → 全局根运动 (5d)
              │         └── TransformerEncoderLayer: SA → FFN
              ├── 坐标转换: global_root → local_root
              └── Stage 2: Body Model → 局部身体运动 (364d)
                        └── TransformerEncoderLayer: SA → FFN
```

### 2. Kimodo-SceneCo 改造后

```
文本 → LLM2Vec → text_feat
场景 → VoxelViT → scene_feat [B, P, 256]
噪声 x_T → [两阶段 SceneCo Transformer 去噪器] ← timestep
              ├── Stage 1: Root Model → 全局根运动 (5d)
              │         └── SceneCoTransformerEncoderLayer:
              │               SA → SceneCo(scene_feat) → FFN    ← 新增
              ├── 坐标转换: global_root → local_root
              └── Stage 2: Body Model → 局部身体运动 (364d)
                        └── SceneCoTransformerEncoderLayer:
                              SA → SceneCo(scene_feat) → FFN    ← 新增
```

### 3. SceneCo 层细节

```python
# 每个 SceneCoTransformerEncoderLayer 内部:
x = x + SelfAttention(LayerNorm(x))       # 冻结
x = x + SceneCo(LayerNorm(x), scene_feat) # 新增，只训练此层
x = x + FFN(LayerNorm(x))                 # 冻结

# SceneCo 内部:
scene_kv = scene_proj(scene_feat)          # [B, P, D_s] → [B, P, D]
h_norm = LayerNorm(h)
attn_out = CrossAttention(h_norm, scene_kv, scene_kv)
output = h + Dropout(attn_out)
```

### 4. Scene-Separated CFG

```
out = out_uncond
    + w_text · (out_text - out_uncond)         # 文本引导
    + w_constraint · (out_constraint - out_uncond)  # 约束引导
    + w_scene · (out_scene - out_uncond)        # 场景引导（新增）
```

四分支：
| 分支 | 文本 | 约束 | 场景 |
|------|------|------|------|
| out_text | ✅ | ❌ | ✅ |
| out_constraint | ❌ | ✅ | ✅ |
| out_scene | ❌ | ❌ | ✅ |
| out_uncond | ❌ | ❌ | ❌ |

---

## 场景编码器

### VoxelViT

```
3D 场景体素网格 [B, 1, 64, 64, 64]
    ↓ 切分为 3D patch (8×8×8)
    ↓ 512 个 patch, 每个体积 512
    ↓ 线性投影: 512 → 256
    ↓ + 可学习 3D 位置编码
    ↓ ViT Encoder (4层, 4头)
    ↓ LayerNorm
场景特征 [B, 512, 256] + mask [B, 512]
```

### BBoxEncoder（备选）

```
场景包围框 [B, N_obj, 6] (center+size)
    ↓ 线性投影: 6 → 256
    ↓ + 标签嵌入 + 位置编码
    ↓ TransformerEncoder (2层, 4头)
    ↓ LayerNorm
场景特征 [B, N_obj, 256] + mask [B, N_obj]
```

---

## 数据集：LINGO

### 数据格式

| 数据项 | 文件 | 格式 |
|--------|------|------|
| 人体关节 | `human_joints_aligned.npy` | `(2915752, 28, 3)` float64 |
| 场景名称 | `scene_name.pkl` | list, len=2915752, 111个唯一场景 |
| 动作段起始 | `start_idx.npy` | `(19450,)` |
| 动作段结束 | `end_idx.npy` | `(19450,)` |
| 文本描述 | `text_aug.pkl` | list of list, 如 `['walk forward']` |
| 场景体素 | `Scene/{name}.npy` | `(300, 100, 400)` bool, 0.02m/voxel |
| 语言-运动字典 | `language_motion_dict__inter_and_loco__16.pkl` | dict, 2275973条 |

### 数据预处理

1. **体素下采样**: `(300, 100, 400)` → `(64, 64, 64)`，使用 scipy.ndimage.zoom (order=0)
2. **关节→运动特征**: SMPL-X 28关节 → KimodoMotionRep 格式 (369d)
   - smooth_root_pos[3] + global_root_heading[2] + local_joints[84] + velocities[84] + foot_contacts[4] + padding[192]
3. **场景名匹配**: 从 `scene_name.pkl` 查找对应体素文件，支持 `{name}.npy` 和 `{base}.npy` 回退
4. **Scene Dropout**: 训练时 10% 概率将场景体素置零（用于 CFG 训练）

### 数据集统计

- 总动作段: 19,450
- 过滤后 (40-196帧): ~17,316
- 训练集 (90%): ~15,584
- 验证集 (10%): ~1,732
- 唯一场景: 111
- 场景体素文件: 254 (含 mirror)

---

## 训练

### 损失函数

参考 SceneAdapt 和 Kimodo 的设计：

#### L_diff: 扩散 MSE 损失（主损失）

```python
# Kimodo 预测 x_0 参数化
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1-ᾱ_t) · ε
pred_x0 = model(x_t, t, text_feat, scene_feat, ...)
loss_mse = MSE(pred_x0 · mask, x_0 · mask)
```

#### L_prior: 先验保持损失（防止文本生成退化）

```python
# 场景输入置零，确保无场景时模型仍能正常生成
pred_x0_null = model(x_t, t, text_feat, scene_feat=null, ...)
loss_prior = MSE(pred_x0_null · mask, x_0 · mask)
```

#### 总损失

```
L_total = L_diff + λ_prior · L_prior
```

默认 `λ_prior = 0.5`。

### 冻结策略

```python
model.freeze_pretrained()
# 冻结所有参数，只解冻:
# - sceneco: SceneCoLayer 中的 cross_attn + scene_proj + norm + dropout
# - scene_encoder: VoxelViT 全部参数
# - scene_null_embed: CFG 的 null 场景嵌入
```

### 验证策略

每 `val_interval`（默认500步）步验证一次：

1. **SceneCo 验证**: 计算带场景的 val_loss 和 val_mse
2. **Baseline 验证**: 计算原始 Kimodo（无场景）的 val_mse
3. **退化比率**: `degradation_ratio = val_mse_sceneco / baseline_mse`
   - 如果退化比率 > 1.5，说明场景注入严重损害了原有生成能力
   - 如果退化比率 ≈ 1.0，说明场景注入没有负面影响

### 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 16 | 批大小 |
| `lr` | 1e-4 | 学习率 |
| `num_epochs` | 100 | 训练轮数 |
| `prior_weight` | 0.5 | 先验保持损失权重 |
| `scene_dropout` | 0.1 | 场景 dropout 概率 |
| `max_frames` | 196 | 最大帧数 |
| `min_frames` | 40 | 最小帧数 |
| `val_interval` | 500 | 验证间隔步数 |
| `voxel_size` | 64,64,64 | 体素网格分辨率 |
| `patch_size` | 8,8,8 | VoxelViT patch 大小 |
| `scene_dim` | 256 | 场景特征维度 |
| `scene_num_layers` | 4 | VoxelViT 层数 |
| `scene_num_heads` | 4 | VoxelViT 注意力头数 |

### 启动训练

```bash
cd /home/lzsh2025/kimodo-viser/kimodo
bash kimodo_sceneco/scripts/train.sh 0  # GPU 0
```

或手动：

```bash
python -m kimodo_sceneco.train.train \
    --data_root /home/lzsh2025/kimodo-viser/LINGO/dataset \
    --pretrained_model Kimodo-SOMA-RP-v1.1 \
    --baseline_model Kimodo-SOMA-RP-v1.1 \
    --freeze_pretrained \
    --output_dir ./sceneco_output \
    --device cuda
```

---

## 评估

### TSTMotion 评估指标

| 指标 | 说明 | 代码位置 |
|------|------|----------|
| **FID** | 生成运动与真实运动之间的 Frechet 距离 | eval.py: compute_fid() |
| **Diversity** | 生成运动的多样性（平均成对距离） | eval.py: compute_diversity() |
| **R-Precision** | 文本-运动检索准确率 (R1/R2/R5/R10/MedR) | eval.py: compute_r_precision() |
| **Foot Skate** | 脚滑行速度/比例 | metrics/foot_skate.py |
| **Scene Collision Rate** | 身体与场景体素的碰撞率 | eval.py: compute_scene_collision_rate() |
| **Constraint Error** | 约束跟随误差 | metrics/constraints.py |

### 启动评估

```bash
bash kimodo_sceneco/scripts/eval.sh ./sceneco_output/checkpoints/best_checkpoint.pt 0
```

---

## 推理使用

```python
from kimodo_sceneco.model import KimodoSceneCo
from kimodo.model import load_model

# 1. 加载预训练 Kimodo
pretrained = load_model("Kimodo-SOMA-RP-v1.1", device="cuda")

# 2. 创建 KimodoSceneCo
model = KimodoSceneCo(
    denoiser=pretrained.denoiser.model,
    text_encoder=pretrained.text_encoder,
    num_base_steps=1000,
    scene_encoder_type="voxel_vit",
    scene_encoder_config={
        "voxel_size": (64, 64, 64),
        "patch_size": (8, 8, 8),
        "d_model": 256,
    },
    cfg_type="scene_separated",
    device="cuda",
)

# 3. 加载训练好的 SceneCo 权重
ckpt = torch.load("best_checkpoint.pt")
model.load_state_dict(ckpt["model_state_dict"])

# 4. 推理
import numpy as np
voxel_grid = np.load("scene.npy")  # (300, 100, 400) bool
voxel_tensor = preprocess_voxel(voxel_grid)  # → (1, 1, 64, 64, 64)

output = model(
    prompts="a person walks to the sofa and sits down",
    num_frames=120,
    num_denoising_steps=50,
    scene_input=voxel_tensor,
    cfg_weight=[2.0, 2.0, 2.0],  # [text, constraint, scene]
)
```

---

## 与 SceneAdapt 的对比

| | SceneAdapt (MDM) | Kimodo-SceneCo |
|---|---|---|
| **基础模型** | MDM (单阶段 TransformerDecoder) | Kimodo (两阶段 TransformerEncoder) |
| **CaKey 层** | ✅ 需要（MDM 无约束学习能力） | ❌ 不需要（Kimodo 已有约束系统） |
| **SceneCo 层** | Cross-Attention (SA↔FFN 之间) | Cross-Attention (SA↔FFN 之间) |
| **场景编码器** | Voxel ViT | Voxel ViT + BBoxEncoder |
| **CFG** | 文本+场景二分支 | 文本+约束+场景三分支 |
| **先验保持** | 10% 场景 dropout + null embedding | L_prior 损失 + scene_dropout |
| **约束系统** | 无 | Root2D / FullBody / EndEffector |
| **两阶段** | 无 | Root Model + Body Model |

---

## 关键设计决策

1. **跳过 CaKey**: Kimodo 的 `motion_mask` concat 机制已经让模型学会了约束处理，不需要额外的仿射调制层
2. **SceneCo 在两个阶段都插入**: Root Model 的 SceneCo 引导根轨迹避障，Body Model 的 SceneCo 引导身体与物体交互
3. **先验保持损失**: 防止场景注入损害文本生成能力，训练时同时计算无场景分支的 MSE
4. **Scene Dropout**: 10% 概率将场景输入置零，为推理时的场景 CFG 提供无条件分支
5. **VoxelViT 保留空间结构**: 输出 patch-wise 特征而非全局池化，允许不同时间帧关注不同场景区域
6. **退化监控**: 验证时同时评估 SceneCo 和原始 Kimodo，计算退化比率
