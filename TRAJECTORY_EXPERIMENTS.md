# TrajCo 轨迹注入实验报告

## 概述

在 KimodoSceneCo 框架基础上，设计并实现了两种轨迹注入方式（additive residual 和 cross-attention），探索根轨迹对运动生成的影响。使用 LINGO 数据集（SMPLX 格式）训练了 7 组对照实验。

---

## 两种注入方式

### Additive Residual (TrajCoLayer)
```
hidden = hidden + sigmoid(α) × MLP(traj_feat)
```
- 参考 CMC 的 HintBlock 设计
- 轨迹经 MLP 编码后作为残差逐层加到 motion hidden states
- 门控 α 初始化为 -5 (sigmoid≈0.007)，近似零初始化

### Cross-Attention (TrajCoCrossLayer)
```
hidden = hidden + sigmoid(α) × CrossAttn(Q=motion, K/V=traj)
```
- 镜像 SceneCo 的跨注意力机制
- Q 来自 motion hidden states，K/V 来自轨迹特征
- 让模型主动"查询"轨迹信息来调整运动

### 轨迹数据
- **5 维**: smooth_root_pos(x,y,z) + heading(cosθ, sinθ)
- 归一化后通过 TrajEncoder (5→512→1024→512→512, zero_init last) 编码

---

## 共享配置

| 参数 | 值 |
|---|---|
| 预训练模型 | `Kimodo-SMPLX-RP-v1` |
| 骨架 | SMPLX 22 关节 |
| 动作数据 | `lingo_smplx_cache`（归一化完整运动特征） |
| 轨迹数据 | `lingo_root_trajectory_smplx` |
| 场景数据 | LINGO 体素 64³ |
| 冻结 Backbone | ✅（仅训练 SceneCo/TrajCo 参数） |
| Epochs | 400 |
| Batch Size | 4 |
| LR | 1e-4 |
| 精度 | bf16 |

---

## 七个实验详解

### Plan A — 纯场景 Baseline
| 项 | 内容 |
|---|---|
| GPU | 2 |
| SceneCo | ✅ root_model + body_model |
| TrajCo | ❌ 无 |
| 目的 | 场景条件对动作生成的基准效果 |

**结果**：

| 指标 | 值 | 说明 |
|---|---|---|
| CFR | 0.00 | 完美避碰 |
| PFFR | 1.00 | 无穿透帧 |
| FootSkate | 10.61 | 脚滑严重，方差大 |
| VelSmooth | 0.60 | 运动抖动剧烈 |
| RootRMSE | 4.59m | 轨迹几乎不可控 |
| PathLenRatio | 574x | 路径极度偏短 |
| SceneCo gate(α) | -1.87 (↑) | 门控在正确打开 |

**结论**：SceneCo 能完美避碰，但无轨迹约束导致运动失控。

---

### Plan B — 纯轨迹 Additive
| 项 | 内容 |
|---|---|
| GPU | 3 |
| SceneCo | ❌ 无 |
| TrajCo | ✅ additive residual, root + body |

**结果**：

| 指标 | 值 |
|---|---|
| CFR | 0.71 |
| RootRMSE | 0.94m |
| FootSkate | 0.93 |
| VelSmooth | 0.04 |

**结论**：轨迹约束比 A 好 5 倍，但 additive 精度有限，无场景导致高碰撞。

---

### Plan C — 场景 + 轨迹 Additive 联合
| 项 | 内容 |
|---|---|
| GPU | 4 |
| SceneCo | ✅ root + body |
| TrajCo | ✅ additive residual, root + body |
| 注入顺序 | Transformer → SceneCo → TrajCo |

**结果**：

| 指标 | 值 |
|---|---|
| CFR | 0.00 |
| RootRMSE | 5.62m（最差） |
| PerFrameMSE | 56.68 |
| FootSkate | 4.95 |
| SceneCo gate(α) | -3.76（接近关闭） |

**结论**：SceneCo 和 TrajCo additive 互相干扰，轨迹精度甚至不如纯 SceneCo。**additive 模式在联合条件下不可用。**

---

### Plan D — 纯轨迹 Cross-Attention
| 项 | 内容 |
|---|---|
| GPU | 0 |
| SceneCo | ❌ 无 |
| TrajCo | ✅ cross-attention, root + body |

**结果**：

| 指标 | 值 |
|---|---|
| RootRMSE | 0.046m（🏆 最佳） |
| PerFrameMSE | 0.0045（🏆 最佳） |
| RMSE_Y | 0.012m（近乎完美） |
| CurvatureError | 0.017 |
| VelSmooth | 0.017 |
| CFR | 0.44（无场景故有碰撞） |

**结论**：轨迹精度遥遥领先（4.6cm），cross-attention 的机制能精准映射轨迹→运动。唯一缺陷是无场景导致碰撞。

---

### Plan E — 场景 + 轨迹 Cross-Attention 全量联合
| 项 | 内容 |
|---|---|
| GPU | 1 |
| SceneCo | ✅ root + body |
| TrajCo | ✅ cross-attention, root + body |

**结果**：

| 指标 | 值 |
|---|---|
| CFR | 0.00 |
| RootRMSE | 0.24m |
| PerFrameMSE | 0.14 |
| FootSkate | 0.75 |
| VelSmooth | 0.066 |

**结论**：最均衡方案——零碰撞 + 轨迹第二 (24cm)。cross-attn 模式下 TrajCo 能和 SceneCo 共存。

---

### Plan F — SceneCo(Body) + TrajCo(Root) 分离注入
| 项 | 内容 |
|---|---|
| GPU | 3 |
| SceneCo | ✅ body_model only |
| TrajCo | ✅ cross-attention, root_model only |
| 设计意图 | 场景→身体，轨迹→根，各司其职 |

**结果**：

| 指标 | 值 |
|---|---|
| CFR | 0.00 |
| RootRMSE | 0.67m |
| FootSkate | 0.14（🏆 最佳） |
| VelSmooth | 0.003（🏆 最佳） |
| AccelJerk | 0.004（🏆 最佳） |
| PathLenRatio | 7.7x（🏆 最接近真值） |
| CurvatureError | 0.004（🏆 最佳） |

**结论**：动作质量全面最优。零碰撞 + 极度平滑，但轨迹比 D 差 15 倍。

---

### Plan G — SceneCo 全量 + TrajCo(Root)
| 项 | 内容 |
|---|---|
| GPU | 5 |
| SceneCo | ✅ root + body |
| TrajCo | ✅ cross-attention, root_model only |
| 设计意图 | 与 F 对比，验证 root 加 SceneCo 是否干扰 TrajCo |

**结果**：

| 指标 | 值 |
|---|---|
| CFR | 0.00 |
| RootRMSE | 1.24m（比 F 差 2 倍） |
| RMSE_Y | 2.09m（高度漂移严重） |
| FootSkate | 0.13 |
| FloatingRatio | 0.80 |

**结论**：root 的 SceneCo 直接干扰 TrajCo — Y 方向 2m 漂移。**SceneCo 不应加在 root_model。**

---

## 完整对比表

| Plan | CFR↓ | PFFR↑ | FootSkate↓ | VelSmooth↓ | RootRMSE↓ | PathLenRatio |
|---|---|---|---|---|---|---|
| **A** (SceneCo) | 0.00 | 1.00 | 10.61 | 0.60 | 4.59 | 574x |
| **B** (T add) | 0.71 | 0.29 | 0.93 | 0.04 | 0.94 | 58x |
| **C** (S+T add) | 0.00 | 1.00 | 4.95 | 0.24 | 5.62 | 206x |
| **D** (T cross) | 0.44 | 0.56 | 0.98 | 0.017 | **0.046** | 18.5x |
| **E** (S+T cross) | 0.00 | 1.00 | 0.75 | 0.066 | 0.24 | 54x |
| **F** (S body+T root) | 0.00 | 1.00 | **0.14** | **0.003** | 0.67 | **7.7x** |
| **G** (S all+T root) | 0.00 | 1.00 | 0.13 | 0.020 | 1.24 | 21x |

### 指标说明

| 类别 | 指标 | 说明 |
|---|---|---|
| C 类 (场景) | CFR | 碰撞帧比例，越低越好 |
| | PFFR | 无穿透帧比例，越高越好 |
| D 类 (质量) | FootSkate | 足部滑动速度，越低越好 |
| | VelSmooth | 平均加速度幅值，越低越平滑 |
| T 类 (轨迹) | RootRMSE | 根轨迹均方根误差 (m)，越低越好 |
| | PathLenRatio | 路径长度比 (1=完美)，越低越接近真值 |

---

## 核心发现

### 1. Cross-Attention 碾压 Additive（架构级差异）

```
Cross-Attention RootRMSE: 0.046m (D) / 0.24m (E)
Additive RootRMSE:        0.94m  (B) / 5.62m (C)
                          差了 20~23 倍
```

Cross-attention 的 Q/K/V 匹配让轨迹信息精准投射到 motion 序列。Additive 残差无法实现这种信号路由。

### 2. SceneCo 放 body 即可，root 不需要

```
F (body only):  RootRMSE=0.67m, FootSkate=0.14
G (root+body):  RootRMSE=1.24m, Y漂移=2.09m
```

root_model 里的 SceneCo 和 TrajCo 竞争同一个 hidden space，导致轨迹失控。

### 3. SceneCo gate 行为

- **Plan A** (SceneCo only): gate 从 -5 升至 -1.87 (↑)，正确学习利用场景
- **Plans C/E/F/G** (SceneCo + TrajCo): gate 困在 -4 附近，TrajCo 抢占了信号

### 4. 所有注入 TrajCo 的实验动作都比纯 SceneCo 平滑

TrajCo 提供的轨迹先验大幅减少了运动抖动 (FootSkate: 10.6→0.14~0.98)。

---

## 最优方案

**当前理论最佳**: Plan D 的轨迹精度 + Plan F 的 SceneCo-body 避碰

即：**TrajCo cross-attention root+body + SceneCo body-only**，此组合未在本次实验中直接训练。

次优已训练方案：

| 优先级 | Plan | 理由 |
|---|---|---|
| 1 | **E** | 零碰撞 + 轨迹 24cm，最均衡 |
| 2 | **F** | 零碰撞 + 动作最平滑，轨迹 67cm |
| 3 | **D** | 轨迹 4.6cm 但无场景避碰 |

---

## 评估日志

评估在 GPU 7 上运行，使用 10 个 LINGO 验证集样本。结果保存在 `outputs/eval_trajco/`。

```
kimodo_scene_project/outputs/eval_trajco/
├── all_metrics.json       # 完整指标数据
├── metric_table.csv       # CSV 格式对比表
└── metric_table.txt       # 文本格式对比表
```
