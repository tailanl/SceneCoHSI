# 轨迹注入 Kimodo 方案设计

## 背景

参考 `cmc_release` 项目的轨迹控制人体动作生成方法，将轨迹（trajectory）条件注入到 Kimodo 扩散模型中。Kimodo 已有的场景注入机制（SceneCo）提供了良好的设计模板。

---

## 一、现有场景注入机制回顾（SceneCo）

### 1.1 SceneCoLayer 结构

位置：[kimodo_sceneco/exp/shared/sceneco_layers.py](file:///home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/exp/shared/sceneco_layers.py)

```
                    ┌──────────────────────────┐
                    │     SceneCoLayer         │
                    │                          │
  motion tokens ───►│  h (B,T,d_model)        │
                    │       │                  │
                    │       ▼                  │
                    │  LayerNorm → Q           │
                    │       │                  │
                    │       ▼                  │
  scene_feat ──────►│  scene_proj → K, V       │
                    │       │                  │
                    │       ▼                  │
                    │  Cross-Attention(Q,K,V)  │
                    │       │                  │
                    │       ▼                  │
                    │  out_proj + Dropout      │
                    │       │                  │
                    │       ▼                  │
                    │  gate = sigmoid(alpha)   │
                    │       │                  │
                    │       ▼                  │
                    │  result = h + gate * attn_out
                    └──────────────────────────┘
```

**关键设计点：**
1. **Q 来自运动 token，K/V 来自场景特征** — 跨模态交叉注意力
2. **可学习的门控 α** — 初始化为 -5（sigmoid≈0.007），渐进式激活场景信号
3. **残差连接** — `result = h + gate * attn_out`，确保不破坏原始运动分布
4. **零初始化最后一层** — 保证训练初期场景信号趋于零

### 1.2 SceneCo 插入位置

在 [kimodo_model.py](file:///home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/model/kimodo_model.py) 中，每个 Transformer 层之后插入 SceneCoLayer：

```python
for i, layer in enumerate(block.seqTransEncoder.layers):
    # 1. 标准 Self-Attention + FFN
    xseq = layer(xseq, src_key_padding_mask=src_key_padding_mask)
    
    # 2. SceneCo Cross-Attention（仅在 root_model 或 body_model 启用）
    if scene_feat is not None and i < len(block.sceneco_layers):
        xseq = block.sceneco_layers[i](xseq, scene_feat, scene_mask)
```

### 1.3 场景编码器 VoxelViT

- 输入：(B, 1, 64, 64, 64) 体素网格
- 输出：(B, 512, 256) patch-wise 特征
- 架构：3D patch embedding → Positional Encoding → Transformer Encoder (4 layers)

### 1.4 训练策略

- **冻结预训练模型**：只训练 SceneCoLayer + VoxelViT
- **Prior Preservation Loss**：50% 权重加 text-only 数据，防止文本生成能力退化
- **Scene Dropout**：10% 概率用零场景替代，增强鲁棒性

---

## 二、轨迹注入方案设计

基于 SceneCo 的模式和 CMC 的思路，设计 `TrajCo` 方案。分为 **三个层次**，从简单到复杂：

### 方案 A：轨迹作为约束（最简，已部分支持）✅ 推荐快速验证

复用 Kimodo 已有的 `Root2DConstraintSet` 约束机制。

#### 已有实现
[two_stage_inference.py](file:///home/lzsh2025/kimodo-viser/kimodo_scene_project/eval/two_stage_inference.py) 中 Stage 2 使用：

```python
from kimodo.constraints import Root2DConstraintSet

constraint = Root2DConstraintSet(
    skeleton=model.skeleton,
    frame_indices=torch.arange(num_frames),
    smooth_root_2d=torch.from_numpy(root_2d).float(),
)

output = model(
    prompts=prompt,
    num_frames=num_frames,
    constraint_lst=[constraint],  # 传入二维根轨迹
    cfg_weight=[2.0, 2.0],
)
```

#### 优点
- 零代码改动
- 通过 `motion_mask` + `observed_motion` 机制实现条件生成

#### 局限
- 仅支持二维根轨迹（xz 平面）
- 约束在 motion representation 空间施加，而非显式条件注入
- 不支持多关节轨迹

---

### 方案 B：TrajCo — TrajEncoder + TrajCoLayer 条件注入 ⭐ 推荐

为根轨迹建立独立的 `TrajEncoder` + `TrajCoLayer`，参照 CMC 的 HintBlock 设计。分为两个子方案对比：

| | B1：TrajCo (w/ SceneCo) | B2：TrajCo (w/o SceneCo) |
|---|---|---|
| **条件组合** | Text + Scene + Traj 三条件 | Text + Traj 双条件 |
| **场景感知** | ✅ VoxelViT 体素 | ❌ 无场景输入 |
| **定位** | 类 CMC 多条件扩展（已有场景基础上加轨迹） | 类 CMC 纯轨迹控制（仅用 Kimodo 骨架） |
| **对比实验意义** | 验证场景+轨迹联合的有效性 | 与 CMC 原论文方案对齐，纯轨迹消融 |

---

### B1：TrajCo (w/ SceneCo) — 场景+轨迹联合注入

#### 整体架构

```
┌──────────┐    ┌─────────────┐    ┌──────────┐
│   Text   │    │   Scene      │    │   Traj   │
│ Encoder  │    │  (VoxelViT)  │    │  Encoder │
└────┬─────┘    └──────┬──────┘    └─────┬─────┘
     │                 │                  │
     ▼                 ▼                  ▼
┌────────────────────────────────────────────────┐
│              Kimodo Denoiser                    │
│  ┌─────────────────────────────────────────┐   │
│  │  For each Transformer layer:            │   │
│  │    1. Self-Attention + FFN              │   │
│  │    2. SceneCoLayer (cross-attn scene)   │   │
│  │    3. TrajCoLayer (additive trajectory)  │   │
│  └─────────────────────────────────────────┘   │
└────────────────────────────────────────────────┘
```

**插入位置**（SceneCo layer 之后）：

```python
# SceneCo Cross-Attention
if scene_feat is not None and i < len(_self.sceneco_layers):
    xseq = _self.sceneco_layers[i](xseq, scene_feat, scene_mask)
# TrajCo 残差注入
if traj_feats is not None and i < len(_self.trajco_layers):
    xseq = _self.trajco_layers[i](xseq, traj_feats, traj_mask)
```

**训练策略**：

```
冻结: Kimodo backbone + SceneCo + VoxelViT
训练: TrajEncoder + TrajCoLayer

损失: L_diff + L_prior_scene + L_traj
  - L_diff: 标准扩散 MSE（全运动）
  - L_prior_scene: text+scene 数据 (traj_mask 全零)，保持场景能力
  - L_traj: root 轨迹 MSE（lambda_traj=1.0）
```

---

### B2：TrajCo (w/o SceneCo) — 纯轨迹控制（类 CMC 消融）

#### 整体架构

```
┌──────────┐    ┌──────────┐
│   Text   │    │   Traj   │
│ Encoder  │    │  Encoder │
└────┬─────┘    └─────┬─────┘
     │                 │
     ▼                 ▼
┌────────────────────────────────────────┐
│          Kimodo Denoiser                │
│  ┌─────────────────────────────────┐   │
│  │  For each Transformer layer:    │   │
│  │    1. Self-Attention + FFN      │   │
│  │    2. TrajCoLayer (additive)    │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

与 B1 的区别：**不使用原始 Kimodo_sceneco 作为基座，直接基于原始 Kimodo 模型**（不含 SceneCo 和 VoxelViT），仅加入 TrajCo 模块。这与 CMC 的设定对齐：没有场景条件，只有文本 + 轨迹。

**插入位置**（原始 Kimodo backbone 的每层 Transformer 之后）：

```python
# 标准 Self-Attention + FFN
xseq = layer(xseq, src_key_padding_mask=src_key_padding_mask)
# TrajCo 残差注入（替代 SceneCo 的位置）
if traj_feats is not None and i < len(_self.trajco_layers):
    xseq = _self.trajco_layers[i](xseq, traj_feats, traj_mask)
```

**训练策略**：

```
冻结: Kimodo backbone（原始，不含 SceneCo）
训练: TrajEncoder + TrajCoLayer

损失: L_diff + L_prior + L_traj
  - L_diff: 标准扩散 MSE（全运动）
  - L_prior: text-only 数据 (traj_mask 全零)，维持文本生成能力
  - L_traj: root 轨迹 MSE（lambda_traj=1.0）
```

---

### 公共模块：TrajEncoder + TrajCoLayer（B1/B2 共用）

两个子方案使用**相同的**编码器和注入层，仅基座模型和训练数据不同。

#### TrajEncoder：轨迹编码器

```python
class TrajEncoder(nn.Module):
    """将根轨迹编码为 per-frame 特征。
    
    参考 CMC 的 HintBlock 设计：MLP 编码 + 零初始化 + 稀疏激活。
    
    输入: (B, T, 5) = smooth_root_pos(3) + global_root_heading(2)
    输出: (B, T, d_model)  per-frame 轨迹特征
    """
    def __init__(self, input_dim=5, d_model=512):
        super().__init__()
        self.traj_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            zero_module(nn.Linear(d_model, d_model)),  # 零初始化
        )
    
    def forward(self, traj, traj_mask):
        feats = self.traj_proj(traj)                     # (B, T, d_model)
        feats = feats * traj_mask.unsqueeze(-1)          # 仅控制帧注入
        return feats
```

#### TrajCoLayer：轨迹注入层

```python
class TrajCoLayer(nn.Module):
    """将轨迹特征以残差方式注入运动 token。
    
    相比于 SceneCo 的 cross-attention（Q→运动，K/V→场景），
    这里使用残差相加，因为轨迹是 per-frame 时序信号，
    天然与运动 token 在时间维度上对齐。
    """
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.traj_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor([-5.0]))  # 门控, sigmoid≈0.007
    
    def forward(self, motion_tokens, traj_feats, traj_mask=None):
        T_motion = traj_feats.shape[1]
        motion_part = motion_tokens[:, -T_motion:]
        prefix_part = motion_tokens[:, :-T_motion]
        
        traj_signal = self.traj_proj(traj_feats)
        gate = torch.sigmoid(self.alpha)
        
        if traj_mask is not None:
            traj_signal = traj_signal * traj_mask.unsqueeze(-1)
        
        motion_part = self.norm(motion_part + gate * self.dropout(traj_signal))
        return torch.cat([prefix_part, motion_part], dim=1)
```

#### 共享损失函数

```python
# 轨迹约束损失（仅计算 root 部分）
root_slice = model.motion_rep.root_slice
pred_root = pred_x0[..., root_slice]
gt_root = x_start[..., root_slice]

loss_root = F.mse_loss(
    pred_root * padding_mask * traj_mask_3d,
    gt_root * padding_mask * traj_mask_3d,
)
loss = L_diff + lambda_prior * L_prior + lambda_traj * loss_root
```

#### 共同训练细节

```
- TrajEncoder: MLP (5→512→512→512→512), zero_init 最后一层
- TrajCoLayer: 每层 Transformer 后插入，与 SceneCo 设计一致
- 门控 α: 初始 -5 (sigmoid≈0.007)，渐进学习
- Traj Dropout: 10% 概率轨迹全零，增强无轨迹场景的鲁棒性
- 冻结策略: backbone 完全冻结（B1 含 SceneCo，B2 不含），仅训练注入模块
```

---

### 方案 C：两阶段轨迹控制（类比 CMC 完整方案）🚀 最强大

完全复现 CMC 的两阶段设计，但用 Kimodo 的架构替换：

```
┌──────────────────────────────────────────────────────────┐
│ Stage 1: 轨迹 → 根运动（类 CMC Stage 1）                    │
│                                                          │
│   Text + Scene ──► SceneCo root_model                    │
│                         +                                │
│   Traj ──────────► TrajCoLayer (HintBlock + LBFGS)       │
│                         │                                │
│                         ▼                                │
│                    pred_root (global_root_features)        │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2: 根运动 + 身体（类 CMC Stage 2）                    │
│                                                          │
│   pred_root ──► replacement 强制覆盖被控维度               │
│   Text ──────► Kimodo body_model                         │
│                         │                                │
│                         ▼                                │
│                    full body motion                       │
└──────────────────────────────────────────────────────────┘
```

#### Stage 1 实现要点

1. **TrajEncoder** — 将根轨迹 (B,T,5) 编码为 (B,T,512)
2. **前向注入** — TrajCoLayer 残差相加（同方案 B）
3. **梯度引导** — LBFGS 在每个扩散步微调预测均值

```python
def lbfgs_guide_root(self, mean_pred, t, traj_target, traj_mask):
    """LBFGS 引导：微调预测均值使其 root 轨迹逼近目标"""
    n_guide_steps = 100 if t[0] == 0 else 10 if t[0] < 10 else 1
    
    hint = traj_target * self.raw_std + self.raw_mean
    hint = hint * traj_mask
    
    x = mean_pred.clone().detach().requires_grad_(True)
    
    def closure():
        lbfgs.zero_grad()
        pred_root = self.motion_rep.extract_global_root(x, normalized=True)
        loss = torch.norm((pred_root - hint) * traj_mask)
        loss.backward()
        return loss
    
    lbfgs = torch.optim.LBFGS([x], lr=lr, ...)
    lbfgs.step(closure)
    return x
```

#### Stage 2 实现要点

复用 Kimodo 的 `motion_mask` + `observed_motion`（Replacement 机制）：

```python
# 用 Stage 1 输出覆盖被控维度
observed_motion[:, :, root_slice] = pred_root  
motion_mask[:, :, root_slice] = traj_mask.float()

# 传入 Kimodo denoiser
output = model.denoiser(
    cfg_weight, x_t, pad_mask, text_feat, text_pad_mask, t,
    motion_mask=motion_mask,
    observed_motion=observed_motion,
    scene_feat_root=scene_feat_root,  # 场景条件
    scene_mask_root=scene_mask_root,
)
```

---

## 三、方案对比与推荐路径

| | 方案 A（约束） | 方案 B（TrajCo） | 方案 C（两阶段） |
|---|---|---|---|
| **实现复杂度** | ⭐ 零改动 | ⭐⭐ 中等 | ⭐⭐⭐ 较高 |
| **数据类型** | 2D 根轨迹 (xz) | 3D 根轨迹 (5维) | 3D 根轨迹 (5维) |
| **控制精度** | 中（软约束） | 中（条件注入） | 高（硬替换） |
| **与场景兼容** | ✅ 完全兼容 | ✅ 并行注入 | ✅ Stage1 共享条件 |
| **多关节支持** | ❌ 不支持 | 🔧 可扩展 | 🔧 可扩展 |
| **代码改动量** | 0 | ~200 行 | ~400 行 |

### 推荐路径

```
第一步：方案 A 快速验证
  → 使用两阶段推理：SceneCo root_only → Root2DConstraint → Kimodo body
  → 验证轨迹条件对生成质量的影响

第二步：方案 B 联合训练
  → 实现 TrajEncoder + TrajCoLayer
  → 在 SceneCo 基础上微调，冻结 backbone
  → 实现文本 + 场景 + 轨迹三条件联合生成

第三步：方案 C 精确控制（可选）
  → 添加 LBFGS 梯度引导
  → 两阶段训练：Stage1 轨迹→根，Stage2 根+文本→身体
```

---

## 四、数据集准备

### 已有工具
[extract_root_trajectory.py](file:///home/lzsh2025/kimodo-viser/kimodo_scene_project/scripts/extract_root_trajectory.py) 已支持从多种数据源提取根轨迹：

```bash
# 从 SMPLX 缓存提取
python kimodo_scene_project/scripts/extract_root_trajectory.py \
    --source smplx_cache \
    --output_dir lingo_root_trajectory_smplx \
    --split both

# 从 SOMA 缓存提取
python kimodo_scene_project/scripts/extract_root_trajectory.py \
    --source soma_cache \
    --output_dir lingo_root_trajectory_soma \
    --split both
```

生成的数据格式：
```python
{
    "global_root_features": np.float32,  # (T, 5) smooth_root_pos(3) + heading(2)
    "local_root_features": np.float32,   # (T, 4) local_root_rot_vel(1) + vel(2) + y(1)
    "voxel_grid": np.float32,           # (64, 64, 64) 场景体素
    "length": np.int64,                 # 帧数
    "scene_name": str,                  # 场景标识
    "text": str,                        # 文本提示
}
```

### 训练配置

在现有 config 基础上添加轨迹参数：

```yaml
# configs/sceneco_smplx_root_body_traj.yaml
data:
  root_trajectory_data: true           # 加载轨迹数据集
  traj_data_dir: "lingo_root_trajectory_smplx"
  traj_dim: 5                          # 根轨迹维度

trajco:
  use_in_root_model: true              # 轨迹注入到 root_model
  use_in_body_model: false             # 可选：也注入到 body_model
  d_model: 512
  dropout: 0.1
  traj_dropout: 0.1                    # 训练时随机丢弃轨迹

training:
  freeze_pretrained: true
  traj_loss_weight: 1.0                # 轨迹损失权重
```

---

## 五、关键代码改动清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `kimodo_sceneco/model/traj_encoder.py` | TrajEncoder 轨迹编码器 |
| `kimodo_sceneco/exp/shared/trajco_layers.py` | TrajCoLayer 轨迹注入层 |

### 修改文件

| 文件 | 改动内容 |
|------|----------|
| `kimodo_sceneco/model/kimodo_model.py` | KimodoSceneCo 添加 traj 参数和 TrajCoLayer 创建 |
| `kimodo_sceneco/train/train.py` | 训练脚本加载轨迹数据、传递 traj_feats |
| `kimodo_sceneco/train/dataset.py` | 已支持 `root_trajectory_data` 标志 |

### 训练命令

```bash
# 方案 B: 联合训练
python train/train_sceneco.py configs/sceneco_smplx_root_body_traj.yaml

# 方案 C: 两阶段训练
# Stage 1: root_only + TrajCo
python train/train_sceneco.py configs/stage1_trajco_root.yaml
# Stage 2: body + root replacement  
python train/train_sceneco.py configs/stage2_body_conditioned_on_root.yaml
```

---

## 六、与 CMC 方案的对比总结

| 特性 | CMC | Kimodo + TrajCo |
|------|-----|-----------------|
| 轨迹注入方式 | HintBlock + LBFGS | TrajCoLayer (残差) + 可选 LBFGS |
| 场景感知 | ❌ 不支持 | ✅ VoxelViT + SceneCo |
| 文本编码 | CLIP | LLM2Vec (Llama-3-8B) |
| 骨架 | HumanML3D (22 joints) | SOMA30 / SMPLX22 |
| 运动表征 | 263维 HML vector | KimodoMotionRep |
| 两阶段 | 扩散 S1+S2 | 扩散（复用现有 DDIM） |
| 代码基底 | 独立项目 | Kimodo 插件式扩展 |
