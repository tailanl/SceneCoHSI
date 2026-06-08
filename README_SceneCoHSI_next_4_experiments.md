# SceneCoHSI 下一轮实验 README：从 GT-root Oracle 走向 Planner-like Trajectory

> 本 README 用于指导 SceneCoHSI / KimodoSceneCo 下一轮 4 个实验。  
> 当前重点不是继续验证 additive/cross-attention 的差异，而是从 **GT root oracle 条件** 推进到 **coarse/noisy planner-like root 条件**。

---

## 0. 当前最重要的前提

已有 D/E/F/G 实验都传入了 **GT-like 5D root trajectory**，因此它们属于 **oracle trajectory 条件实验**。

当前 5D trajectory 是：

```text
smooth_root_pos(x, y, z) + global_root_heading(cosθ, sinθ)
```

也就是：

```text
traj[t] = [
    root_x,
    root_y,
    root_z,
    heading_cos,
    heading_sin
]
```

所以 D/E/F/G 可以证明：

```text
1. TrajCo cross-attention 能非常有效地利用 GT root。
2. SceneCo 可以降低碰撞。
3. SceneCo 放进 root_model 会干扰 TrajCo。
4. SceneCo 更适合放进 body_model。
5. TrajCo 是否进入 body_model 会影响轨迹精度和身体质量。
```

但是 D/E/F/G 不能证明：

```text
外部 planner 给一个粗路径时，模型也能正常生成自然动作。
```

下一轮实验的目标就是补上这个证据。

---

## 1. 已有 D/E/F/G 能说明什么

### 1.1 Plan D：TrajCo Cross Root+Body

```text
SceneCo: 无
TrajCo: cross-attention, root + body
Root 输入: GT 5D root
```

结果：

```text
RootRMSE = 0.046m
PerFrameMSE = 0.0045
RMSE_Y = 0.012m
CFR = 0.44
```

结论：

```text
Plan D 证明 TrajCo cross-attention 在 GT root 条件下轨迹控制非常强。
但是它没有场景条件，所以碰撞严重。
```

---

### 1.2 Plan E：SceneCo 全量 + TrajCo Cross 全量

```text
SceneCo: root + body
TrajCo: cross-attention, root + body
Root 输入: GT 5D root
```

结果：

```text
CFR = 0.00
RootRMSE = 0.24m
FootSkate = 0.75
VelSmooth = 0.066
```

结论：

```text
Plan E 说明 SceneCo 和 TrajCo cross-attention 可以共存。
它实现了零碰撞和较好的轨迹精度。
但是相比 Plan D，轨迹精度下降，说明 SceneCo 进入 root/body 后会影响轨迹跟随。
```

---

### 1.3 Plan F：SceneCo Body-only + TrajCo Root-only

```text
SceneCo: body only
TrajCo: cross-attention, root only
Root 输入: GT 5D root
```

结果：

```text
CFR = 0.00
RootRMSE = 0.67m
FootSkate = 0.14
VelSmooth = 0.003
AccelJerk = 0.004
PathLenRatio = 7.7x
CurvatureError = 0.004
```

结论：

```text
Plan F 证明 root/body 分工是有效的。
SceneCo 放到 body_model 可以改善碰撞、脚滑和平滑性。
TrajCo 放到 root_model 可以提供基本轨迹控制。
```

但是：

```text
TrajCo 只进 root 时，轨迹精度不如 D/E。
```

---

### 1.4 Plan G：SceneCo Root+Body + TrajCo Root-only

```text
SceneCo: root + body
TrajCo: cross-attention, root only
Root 输入: GT 5D root
```

结果：

```text
CFR = 0.00
RootRMSE = 1.24m
RMSE_Y = 2.09m
FloatingRatio = 0.80
```

结论：

```text
SceneCo 不应该放进 root_model。
root_model 里的 SceneCo 会干扰 TrajCo，导致 root 漂移，尤其是 Y 方向高度漂移。
```

---

## 2. 当前核心结论

已有实验支持以下判断：

```text
1. Additive TrajCo 不再优先。
2. TrajCo cross-attention 是当前最强轨迹注入方式。
3. SceneCo 能解决碰撞，但不能放进 root_model。
4. SceneCo 更适合放进 body_model。
5. 当前理论最佳结构还没有直接训练：
   SceneCo body-only + TrajCo cross root+body。
```

所以，下一轮实验应该围绕两条主线展开：

```text
主线 1：
    SceneCo body-only + TrajCo cross root+body
    目标：合并 D 的轨迹精度和 F 的身体质量。

主线 2：
    SceneCo body-only + TrajCo cross root-only
    目标：测试在 coarse/noisy trajectory 下，F 的稳定分工是否更鲁棒。
```

---

## 3. 下一轮排 4 个实验

推荐顺序：

```text
1. Plan H-clean
2. Plan H-coarse/noisy
3. Plan D-coarse/noisy
4. Plan F-coarse/noisy
```

这 4 个实验构成完整证据链：

```text
H-clean:
    验证当前理论最佳结构在 GT-root oracle 下的上限。

H-coarse/noisy:
    验证同一个结构在 planner-like root 条件下能不能用。

D-coarse/noisy:
    去掉 SceneCo，只测 TrajCo root+body 本身对 coarse/noisy root 的鲁棒性。

F-coarse/noisy:
    测试更稳定的 root/body 分工方案在 planner-like root 条件下是否更鲁棒。
```

---

# 4. 实验 1：Plan H-clean

## 4.1 实验名称

```text
Plan H-clean
```

建议配置文件名：

```text
configs/trajco_cross_root_body_sceneco_body_clean.yaml
```

建议输出目录：

```text
outputs/trajco_cross_root_body_sceneco_body_clean
```

---

## 4.2 结构

```text
SceneCo:
    root_model = false
    body_model = true

TrajCo:
    root_model = true
    body_model = true
    type = cross_attn

Root 输入:
    clean GT 5D root trajectory
```

---

## 4.3 目的

H-clean 的目的是验证：

```text
Plan D 的轨迹精度
+
Plan F 的 SceneCo body-only 避碰和平滑
能否合并到一个模型中。
```

它仍然是 oracle 实验，因为输入是 GT 5D root。

---

## 4.4 配置片段

```yaml
experiment:
  name: "trajco_cross_root_body_sceneco_body_clean"
  description: "Plan H-clean: SceneCo body-only + TrajCo cross root+body with clean GT 5D root"

sceneco:
  use_in_root_model: false
  use_in_body_model: true

  d_model: 1024
  scene_feat_dim: 256
  nhead: 8
  dropout: 0.1

trajco:
  use_trajco_root: true
  use_trajco_body: true
  trajco_type: cross_attn
  traj_dim: 5
  trajco_dropout: 0.1
  traj_dropout: 0.1
  traj_loss_weight: 1.0

traj_corruption:
  enabled: false
```

---

## 4.5 预期结果

合理预期：

```text
CFR:        0.00
PFFR:       1.00
RootRMSE:   0.05m ~ 0.30m
FootSkate:  0.14 ~ 0.75
VelSmooth:  0.003 ~ 0.066
```

如果达到：

```text
CFR = 0.00
RootRMSE < 0.30m
FootSkate < 0.40
```

就可以作为当前最强 oracle 模型。

---

## 4.6 失败判断

如果 H-clean 的 RootRMSE 明显大于 E 或 F，例如：

```text
RootRMSE > 0.70m
```

说明：

```text
TrajCo root+body 和 SceneCo body-only 仍然存在干扰。
```

如果 CFR 不是 0：

```text
CFR > 0.05
```

说明：

```text
SceneCo body-only 没有成功发挥避碰作用。
需要检查 body_model 里的 SceneCo 是否真的打开。
```

---

# 5. 实验 2：Plan H-coarse/noisy

## 5.1 实验名称

```text
Plan H-coarse/noisy
```

建议配置文件名：

```text
configs/trajco_cross_root_body_sceneco_body_coarse.yaml
```

建议输出目录：

```text
outputs/trajco_cross_root_body_sceneco_body_coarse
```

---

## 5.2 结构

结构和 H-clean 完全一样：

```text
SceneCo:
    root_model = false
    body_model = true

TrajCo:
    root_model = true
    body_model = true
    type = cross_attn
```

唯一变化是：

```text
输入 TrajCo 的 root 不再是 clean GT。
而是 coarse/noisy 5D root。
```

---

## 5.3 目的

这个实验是最重要的主实验。

它回答：

```text
如果外部 planner 只给一个粗路径，
模型是否还能生成合理、平滑、少碰撞的动作？
```

H-clean 是上限。  
H-coarse/noisy 是接近真实使用场景的实验。

---

## 5.4 Coarse/noisy root 构造

不要只加随机噪声。

推荐模拟外部 planner：

```text
GT 5D root trajectory
    ↓
每隔 K 帧取一个 waypoint
    ↓
用线性插值或 spline 插值回 T 帧
    ↓
根据插值后的路径重新计算 heading
    ↓
加入小扰动
    ↓
作为 TrajCo 输入
```

原因：

```text
真实 planner 通常输出 sparse waypoints，
不是每一帧精确 root。
```

---

## 5.5 配置片段

```yaml
experiment:
  name: "trajco_cross_root_body_sceneco_body_coarse"
  description: "Plan H-coarse/noisy: SceneCo body-only + TrajCo cross root+body with planner-like 5D root"

sceneco:
  use_in_root_model: false
  use_in_body_model: true

trajco:
  use_trajco_root: true
  use_trajco_body: true
  trajco_type: cross_attn
  traj_dim: 5
  trajco_dropout: 0.1
  traj_dropout: 0.1
  traj_loss_weight: 1.0

traj_corruption:
  enabled: true

  # 每 30 帧保留一个 waypoint。30fps 下约等于 1 秒一个点。
  waypoint_interval: 30

  # 每帧位置扰动
  pos_noise_std: 0.05

  # 整条轨迹整体偏移
  global_shift_std: 0.05

  # heading 扰动，0.10 rad 约等于 5.7 度
  heading_noise_std: 0.10

  # 插值后重新根据路径切线计算 heading
  recompute_heading_from_path: true
```

---

## 5.6 预期结果

H-coarse/noisy 一定会比 H-clean 差，这是正常的。

可以接受：

```text
CFR:        0.00
RootRMSE:   0.30m ~ 0.70m
FootSkate:  < 0.50
VelSmooth:  < 0.05
```

如果达到：

```text
CFR = 0.00
RootRMSE < 0.70m
FootSkate < 0.50
```

就可以说：

```text
在 planner-like root 条件下，
SceneCo body-only + TrajCo cross root+body 仍然有效。
```

---

# 6. 实验 3：Plan D-coarse/noisy

## 6.1 实验名称

```text
Plan D-coarse/noisy
```

建议配置文件名：

```text
configs/trajco_cross_root_body_coarse.yaml
```

建议输出目录：

```text
outputs/trajco_cross_root_body_coarse
```

---

## 6.2 结构

```text
SceneCo:
    root_model = false
    body_model = false

TrajCo:
    root_model = true
    body_model = true
    type = cross_attn

Root 输入:
    coarse/noisy 5D root
```

---

## 6.3 目的

这个实验是 H-coarse/noisy 的对照。

它回答：

```text
在没有 SceneCo 的情况下，
TrajCo cross-attention 对 coarse/noisy root 本身是否鲁棒？
```

为什么要跑它：

```text
如果 H-coarse/noisy 结果差，
需要知道是 coarse/noisy trajectory 本身太难，
还是 SceneCo body-only 和 TrajCo root+body 有冲突。
```

---

## 6.4 配置片段

```yaml
experiment:
  name: "trajco_cross_root_body_coarse"
  description: "Plan D-coarse/noisy: TrajCo cross root+body with planner-like 5D root, no SceneCo"

sceneco:
  use_in_root_model: false
  use_in_body_model: false

trajco:
  use_trajco_root: true
  use_trajco_body: true
  trajco_type: cross_attn
  traj_dim: 5
  trajco_dropout: 0.1
  traj_dropout: 0.1
  traj_loss_weight: 1.0

traj_corruption:
  enabled: true
  waypoint_interval: 30
  pos_noise_std: 0.05
  global_shift_std: 0.05
  heading_noise_std: 0.10
  recompute_heading_from_path: true
```

---

## 6.5 预期结果

D-coarse/noisy 没有 SceneCo，所以可能仍然有碰撞。

合理预期：

```text
RootRMSE:   0.20m ~ 0.60m
CFR:        可能 > 0
FootSkate:  可能接近 D 或稍差
VelSmooth:  接近 D 或稍差
```

它最重要的指标是：

```text
RootRMSE
heading_angle_error
```

不是 CFR。

因为它的任务是：

```text
验证 TrajCo 本身对 coarse/noisy root 是否鲁棒。
```

---

# 7. 实验 4：Plan F-coarse/noisy

## 7.1 实验名称

```text
Plan F-coarse/noisy
```

建议配置文件名：

```text
configs/trajco_cross_root_sceneco_body_coarse.yaml
```

建议输出目录：

```text
outputs/trajco_cross_root_sceneco_body_coarse
```

---

## 7.2 结构

```text
SceneCo:
    root_model = false
    body_model = true

TrajCo:
    root_model = true
    body_model = false
    type = cross_attn

Root 输入:
    coarse/noisy 5D root
```

---

## 7.3 目的

这个实验是 Plan F 的 planner-like 版本。

它回答：

```text
当 root trajectory 从 clean GT 变成 coarse/noisy planner-like 输入时，
“SceneCo body-only + TrajCo root-only” 这种更明确的 root/body 分工是否更稳定？
```

为什么要加 F-coarse：

```text
H-coarse/noisy 里 TrajCo 同时进入 root 和 body。
这可能带来更好的轨迹控制，但也可能让 noisy trajectory 影响 body，导致身体动作变差。

F-coarse/noisy 只把 TrajCo 放进 root。
Body 只看 SceneCo。
因此它可能牺牲一点轨迹精度，但换来更平滑、更稳的身体动作。
```

它是 H-coarse/noisy 的关键对照。

---

## 7.4 配置片段

```yaml
experiment:
  name: "trajco_cross_root_sceneco_body_coarse"
  description: "Plan F-coarse/noisy: SceneCo body-only + TrajCo cross root-only with planner-like 5D root"

sceneco:
  use_in_root_model: false
  use_in_body_model: true

trajco:
  use_trajco_root: true
  use_trajco_body: false
  trajco_type: cross_attn
  traj_dim: 5
  trajco_dropout: 0.1
  traj_dropout: 0.1
  traj_loss_weight: 1.0

traj_corruption:
  enabled: true
  waypoint_interval: 30
  pos_noise_std: 0.05
  global_shift_std: 0.05
  heading_noise_std: 0.10
  recompute_heading_from_path: true
```

---

## 7.5 预期结果

F-coarse/noisy 可能不是轨迹最准的，但应该更稳。

合理预期：

```text
CFR:        0.00
RootRMSE:   0.50m ~ 1.00m
FootSkate:  < 0.30
VelSmooth:  < 0.02
AccelJerk:  低于 H-coarse/noisy
FloatingRatio: 接近 0
```

它最重要的不是打败 H-coarse/noisy 的 RootRMSE，而是看：

```text
1. 是否仍然保持零碰撞。
2. 是否保持 Plan F 的平滑优势。
3. noisy root 是否不会破坏 body motion。
```

---

## 7.6 F-coarse 和 H-coarse 怎么比较

| 对比项 | H-coarse/noisy | F-coarse/noisy |
|---|---|---|
| SceneCo | body-only | body-only |
| TrajCo root | 开 | 开 |
| TrajCo body | 开 | 关 |
| 轨迹精度 | 预期更好 | 可能稍差 |
| 身体平滑 | 可能受 noisy traj 影响 | 预期更稳 |
| 脚滑 | 可能较低或中等 | 预期更低 |
| 碰撞 | 应为 0 | 应为 0 |

判断规则：

```text
如果 H-coarse RootRMSE 明显优于 F-coarse，且 FootSkate/VelSmooth 没有变差：
    选择 H 结构。

如果 H-coarse RootRMSE 只比 F-coarse 好一点，但 FootSkate/VelSmooth 明显更差：
    选择 F 结构。

如果 F-coarse 更平滑、更稳定，但轨迹误差略大：
    可以把 F 作为主稳定方案，把 H 作为高精度方案。
```

---

# 8. 四个实验如何比较

## 8.1 主比较表

| 实验 | SceneCo | TrajCo | Root 输入 | 主要问题 |
|---|---|---|---|---|
| H-clean | body-only | cross root+body | clean GT 5D root | 最优结构上限 |
| H-coarse/noisy | body-only | cross root+body | coarse/noisy 5D root | planner-like 输入是否可用 |
| D-coarse/noisy | none | cross root+body | coarse/noisy 5D root | TrajCo 本身是否鲁棒 |
| F-coarse/noisy | body-only | cross root-only | coarse/noisy 5D root | 分工式结构是否更稳 |

---

## 8.2 预期排序

理想情况下：

```text
轨迹精度：
H-clean <= D-coarse/noisy <= H-coarse/noisy <= F-coarse/noisy

碰撞：
H-clean ≈ H-coarse/noisy ≈ F-coarse/noisy < D-coarse/noisy

动作质量：
F-coarse/noisy <= H-coarse/noisy <= D-coarse/noisy
```

注意：

```text
这里的 <= 表示指标更低更好。
```

---

# 9. 轨迹 corruption 参考实现

下面是参考函数。实际接入时，可以放到：

```text
kimodo_sceneco/data/traj_corruption.py
```

或训练脚本的数据预处理部分。

```python
import torch
import torch.nn.functional as F


def linear_resample_waypoints(pos, waypoint_interval):
    """
    pos:
        (B, T, 3)

    返回：
        coarse_pos: (B, T, 3)

    做法：
        每隔 waypoint_interval 帧取一个点，再插值回 T 帧。
    """
    B, T, D = pos.shape
    device = pos.device

    waypoint_idx = torch.arange(0, T, waypoint_interval, device=device)
    if waypoint_idx[-1] != T - 1:
        waypoint_idx = torch.cat([waypoint_idx, torch.tensor([T - 1], device=device)])

    waypoint_pos = pos[:, waypoint_idx]  # (B, K, 3)

    # 用 1D interpolate 对时间维插值
    # 输入需要是 (B, C, K)
    waypoint_pos_ch = waypoint_pos.transpose(1, 2)  # (B, 3, K)
    coarse_pos_ch = F.interpolate(
        waypoint_pos_ch,
        size=T,
        mode="linear",
        align_corners=True,
    )
    coarse_pos = coarse_pos_ch.transpose(1, 2)  # (B, T, 3)

    return coarse_pos


def recompute_heading_from_pos(pos):
    """
    pos:
        (B, T, 3)

    返回：
        heading: (B, T, 2)
    """
    delta = pos[:, 1:] - pos[:, :-1]  # (B, T-1, 3)

    # 注意：这里的 atan2 顺序必须和你的 Kimodo heading 定义一致。
    # 如果可视化后发现朝向转了 90 度，就需要交换 x/z 或改符号。
    theta = torch.atan2(delta[..., 2], delta[..., 0])  # z, x

    # 复制最后一帧 heading
    theta = torch.cat([theta, theta[:, -1:]], dim=1)

    heading = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    return heading


def corrupt_5d_traj(
    traj_gt,
    traj_mask,
    waypoint_interval=30,
    pos_noise_std=0.05,
    global_shift_std=0.05,
    heading_noise_std=0.10,
    recompute_heading=True,
):
    """
    traj_gt:
        (B, T, 5)
        [x, y, z, cos, sin]

    traj_mask:
        (B, T)
    """
    traj = traj_gt.clone()
    mask = traj_mask[..., None].to(traj.dtype)

    pos = traj[..., 0:3]

    # 1. coarse waypoint interpolation
    pos = linear_resample_waypoints(pos, waypoint_interval)

    # 2. per-frame x/z noise
    pos_noise = torch.randn_like(pos) * pos_noise_std
    pos_noise[..., 1] = 0.0
    pos = pos + pos_noise * mask

    # 3. global x/z shift
    shift = torch.randn(
        pos.shape[0],
        1,
        3,
        device=pos.device,
        dtype=pos.dtype,
    ) * global_shift_std
    shift[..., 1] = 0.0
    pos = pos + shift * mask

    # 4. recompute or perturb heading
    if recompute_heading:
        heading = recompute_heading_from_pos(pos)
    else:
        heading = traj[..., 3:5]

    theta = torch.atan2(heading[..., 1], heading[..., 0])
    theta = theta + torch.randn_like(theta) * heading_noise_std

    heading = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    traj_out = torch.cat([pos, heading], dim=-1)
    traj_out = traj_out * mask

    return traj_out
```

---

# 10. 训练前必须做的可视化检查

在跑 H-coarse/noisy、D-coarse/noisy、F-coarse/noisy 前，必须可视化：

```text
1. clean GT root path
2. coarse/noisy root path
3. heading arrows
4. floor voxel / obstacle map
5. 起点和终点
```

如果 coarse/noisy path 明显穿墙或离 GT 太远，不要训练。

建议输出：

```text
debug_vis/
├── sample_000_root_path.png
├── sample_001_root_path.png
├── sample_002_root_path.png
└── ...
```

图里至少画：

```text
蓝色：GT root path
橙色：coarse/noisy root path
箭头：heading
灰色：可行走区域 / 障碍区域
```

---

# 11. 评估指标

## 11.1 场景指标

```text
CFR:
    碰撞帧比例，越低越好

PFFR:
    无穿透帧比例，越高越好
```

---

## 11.2 轨迹指标

```text
RootRMSE:
    root 轨迹误差，越低越好

RMSE_Y:
    root 高度误差，重点检查是否漂浮

PathLenRatio:
    路径长度比，越接近 1 越好

CurvatureError:
    轨迹曲率误差，越低越好

heading_angle_error:
    朝向角误差，建议新增
```

---

## 11.3 动作质量指标

```text
FootSkate:
    脚滑，越低越好

VelSmooth:
    速度/加速度平滑性，越低越好

AccelJerk:
    加速度 jerk，越低越好

FloatingRatio:
    漂浮比例，越低越好
```

---

# 12. 结果如何解释

## 12.1 如果 H-clean 最好

可以说明：

```text
SceneCo body-only + TrajCo cross root+body 是当前最优 oracle 结构。
```

---

## 12.2 如果 H-coarse/noisy 也好

可以说明：

```text
该结构不只是能利用 GT root，
也可以适配 planner-like root。
```

这是最有价值的结论。

---

## 12.3 如果 D-coarse/noisy 轨迹好，但 H-coarse/noisy 轨迹差

说明：

```text
SceneCo body-only 仍然会影响 TrajCo root+body。
```

下一步应该优先看 F-coarse/noisy 的结果。

---

## 12.4 如果 F-coarse/noisy 比 H-coarse/noisy 更稳

说明：

```text
在粗轨迹条件下，TrajCo 不一定应该进入 body。
Body 只看 SceneCo，Root 只看 TrajCo，可能更稳定。
```

这时推荐主方案改为：

```text
SceneCo body-only + TrajCo root-only
```

即 F 结构。

---

## 12.5 如果 H-coarse/noisy 比 F-coarse/noisy 明显更好

说明：

```text
TrajCo 进入 body 对 coarse/noisy root 仍然有帮助。
root+body TrajCo 是更好的主结构。
```

这时推荐主方案使用 H 结构。

---

## 12.6 如果 D-coarse/noisy、H-coarse/noisy、F-coarse/noisy 都差

说明问题不在 SceneCo，而在 trajectory corruption：

```text
1. waypoint 太稀疏
2. heading 计算错
3. 噪声太大
4. 轨迹没有和 motion/voxel 对齐
```

优先修：

```text
1. waypoint_interval: 30 → 15
2. heading 可视化
3. pos_noise_std: 0.05 → 0.03
4. global_shift_std: 0.05 → 0.03
```

---

# 13. 不建议现在排的实验

## 13.1 不再优先 additive

已有结果说明：

```text
Additive:
    B RootRMSE = 0.94m
    C RootRMSE = 5.62m

Cross-attention:
    D RootRMSE = 0.046m
    E RootRMSE = 0.24m
```

所以 additive 暂时不值得继续排。

---

## 13.2 不再让 SceneCo 进入 root

Plan G 已经显示：

```text
RootRMSE = 1.24m
RMSE_Y = 2.09m
FloatingRatio = 0.80
```

所以当前阶段不要再尝试：

```text
SceneCo root + TrajCo root
```

除非后面专门设计解耦门控、分支投影或分层融合。

---

# 14. 推荐实验顺序

推荐：

```text
Step 1:
    跑 H-clean

Step 2:
    如果 H-clean 成功，跑 H-coarse/noisy

Step 3:
    跑 D-coarse/noisy，作为 no-scene 对照

Step 4:
    跑 F-coarse/noisy，作为稳定分工对照
```

如果 H-clean 失败：

```text
先不要跑 H-coarse/noisy。
先检查 H 结构是否有 bug。
```

如果 H-clean 成功，H-coarse/noisy 失败：

```text
对比 D-coarse/noisy 和 F-coarse/noisy。
判断是 coarse trajectory 本身问题，还是 H 结构中 TrajCo 进入 body 带来的问题。
```

---

# 15. 最终实验列表

| 优先级 | 实验 | SceneCo | TrajCo | Root 输入 | 目的 |
|---|---|---|---|---|---|
| 1 | H-clean | body-only | cross root+body | clean GT 5D root | 最优结构 oracle 上限 |
| 2 | H-coarse/noisy | body-only | cross root+body | coarse/noisy 5D root | planner-like 输入是否可用 |
| 3 | D-coarse/noisy | none | cross root+body | coarse/noisy 5D root | TrajCo 本身对粗轨迹是否鲁棒 |
| 4 | F-coarse/noisy | body-only | cross root-only | coarse/noisy 5D root | 分工式结构是否更稳定 |

---

# 16. 最短结论

下一轮实验固定为：

```text
1. H-clean:
   SceneCo body-only + TrajCo cross root+body + clean GT root

2. H-coarse/noisy:
   SceneCo body-only + TrajCo cross root+body + coarse/noisy root

3. D-coarse/noisy:
   no SceneCo + TrajCo cross root+body + coarse/noisy root

4. F-coarse/noisy:
   SceneCo body-only + TrajCo cross root-only + coarse/noisy root
```

这四个实验分别回答：

```text
1. 最优结构上限是多少？
2. 这个结构能不能从 GT-root oracle 走向 planner-like root？
3. coarse/noisy root 退化到底是 TrajCo 自身问题，还是 SceneCo 组合问题？
4. 在粗轨迹条件下，是否更应该采用 root/body 明确分工的 F 结构？
```

当前阶段最重要的主线是：

```text
从 GT root oracle 条件
推进到
外部 planner-like trajectory 条件。
```
