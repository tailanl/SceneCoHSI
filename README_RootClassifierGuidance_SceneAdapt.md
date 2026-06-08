# SceneCoHSI：Root Classifier Guidance + SceneAdapt 风格评估修改方案

> 目标：不再使用 TrajCo cross-attention 直接注入轨迹。  
> 新方案改为在采样阶段使用 **Classifier Guidance** 引导 Kimodo 的 Root Stage 生成目标 root 轨迹，同时加入路径平滑、速度均匀、朝向一致和场景避障项。  
> 然后使用 **SceneAdapt-style 指标** 评估：路径一致性、场景碰撞、穿模、脚滑、动作稳定性。

---

## 0. 为什么要改

已有 D / E / F / Hclean 四个实验说明：

```text
D / TrajCo only:
    TrajCo root+body，效果最好，但 root 仍有抖动。

F / T-root + SC-body:
    TrajCo root-only + SceneCo body-only，人物动作出现扭曲、root/body 对不上。

Hclean / T-all + SC-body:
    TrajCo root+body + SceneCo body-only，和 F 接近，没有明显改善。

E / SceneCo only:
    SceneCo root+body，效果最差，root 轨迹基本崩掉。
```

因此当前结论是：

```text
1. TrajCo cross-attention 在 clean / GT-like trajectory 下有效。
2. 但是直接把轨迹作为 cross-attention 条件塞进网络，容易带来抖动和 body 扭曲。
3. SceneCo 直接进 Root Stage 很危险。
4. Body Stage 中 TrajCo 和 SceneCo 同层 cross-attention 可能发生竞争。
```

所以后续主线改为：

```text
不改 denoiser 网络结构；
不再新增 TrajCo cross-attention；
在采样阶段用 Classifier Guidance 引导 root 轨迹；
再用场景 SDF / voxel guidance 减少碰撞和穿模。
```

---

## 1. 新方案要证明的两件事情

### 1.1 证明轨迹能控制 root

要证明：

```text
给定外部路径 / waypoint / planner path
    ↓
Kimodo 生成 root
    ↓
root 是否贴近路径
root 是否到达目标
root 是否转向平滑
root 速度是否均匀
root 是否能给 Stage2 使用
```

核心指标：

```text
PathADE
PathFDE
WaypointError
HeadingError
SpeedStd
RootAccel
RootJerk
```

---

### 1.2 证明加入场景后减少碰撞和穿模

要证明：

```text
同一条路径、同一个场景：

Path-only Kimodo:
    只跟路径，不看场景
    可能穿墙、撞桌子、进入不可行走区域

Path + Scene Guidance:
    跟路径，同时避开 occupied / non-walkable 区域
    collision / penetration 应下降
```

核心指标：

```text
CollisionFrameRate
PenetrationRate
PenetrationMean
PenetrationMax
NonWalkableRootRate
SceneSDFPenalty
```

---

## 2. 项目当前相关代码位置

当前仓库已经有以下基础结构：

```text
kimodo_sceneco/model/kimodo_model.py
    KimodoSceneCo 主模型包装
    已经 patch denoiser forward
    已经包含 SceneCo / TrajCo 插入逻辑
    已经包含 DDIMSampler / Diffusion 相关采样对象

kimodo_sceneco/model/twostage_denoiser.py
    Root Model → local root → Body Model
    root_model 预测 global root
    body_model 使用 local root + noisy body 预测 body

kimodo_sceneco/train/train.py
    当前训练 loss 和 batch 逻辑
    当前会把 traj_feats / scene_feats 放入 model_kwargs

scripts/_exp_root_stage2.py
    已经有“给定 root，生成 body”的 Stage2 测试思路
    可作为 body-only / fixed-root 生成脚本的参考

eval/
    后续新增路径一致性、场景碰撞、穿模指标
```

当前 motion feature 布局是：

```text
[ smooth_root_pos(3) | heading(2) | local_joints(66) | global_rot_data(132) | velocities(66) | foot_contacts(4) ]
```

因此 root 控制直接针对前 5 维：

```text
root_slice = [0:5]
root_5d = smooth_root_pos(3) + heading(cos, sin)
```

---

## 3. 推荐新增文件

建议新增这些文件：

```text
kimodo_sceneco/
├── guidance/
│   ├── __init__.py
│   ├── root_guidance.py          # 路径、速度、平滑、heading guidance
│   ├── scene_guidance.py         # voxel / SDF / walkable guidance
│   └── path_utils.py             # path resample、heading、smooth 工具
│
├── eval/
│   ├── eval_path_metrics.py      # PathADE / PathFDE / HeadingError / SpeedStd
│   ├── eval_scene_metrics.py     # Collision / Penetration / NonWalkable
│   └── eval_sceneadapt_metrics.py# 汇总成 SceneAdapt-style CSV
│
scripts/
├── generate_root_guidance.py     # 使用 classifier guidance 生成 root
├── generate_body_from_root.py    # 固定 root，用 Stage2 生成 body
└── run_path_scene_eval.py        # 一键生成 + 评估
│
configs/
└── guidance_root_scene.yaml      # 新实验配置
```

---

## 4. 整体流程

新方案分两步。

### Step 1：Root Guidance 生成平滑 root

```text
text prompt
target path / waypoints
optional scene SDF
        ↓
Kimodo Root Stage
        ↓
Classifier Guidance 修改采样
        ↓
guided_root_5d
```

输出：

```text
guided_root_5d: (B, T, 5)
    [x, y, z, heading_cos, heading_sin]
```

这个 root 必须满足：

```text
1. 跟目标路径一致
2. 速度尽量匀称
3. 转向平滑
4. heading 与路径切线一致
5. 不进入 occupied / non-walkable 区域
6. root_y 不漂
```

---

### Step 2：固定 root，Stage2 生成 body

```text
guided_root_5d
        ↓
global_root_to_local_root
        ↓
Body Denoiser
        ↓
predicted_body

final_motion = [guided_root_5d | predicted_body]
```

这一阶段可以先不传 initial pose。

如果不传 initial pose：

```text
Body 的第 0 帧从 noisy body 生成。
优点：简单。
缺点：第一帧不一定等于外部数据集初始姿态。
```

---

## 5. Root Classifier Guidance 设计

### 5.1 Guidance loss 总公式

每个采样 step 预测出 `pred_x0` 后，取 root 5D：

```python
root = pred_x0[..., root_slice]
pos = root[..., 0:3]
heading = root[..., 3:5]
xz = pos[..., [0, 2]]
```

总 loss：

```text
L_guidance =
    λ_path      * L_path
  + λ_goal      * L_goal
  + λ_speed     * L_speed
  + λ_smooth    * L_smooth
  + λ_heading   * L_heading
  + λ_height    * L_height
  + λ_scene     * L_scene
```

---

### 5.2 路径一致性

目标：

```text
root 轨迹靠近给定 path。
```

loss：

```python
L_path = mean(distance(root_xz[t], target_path)^2)
```

第一版可以用 dense target path：

```python
target_path: (B, T, 2)
L_path = ((xz - target_path) ** 2).sum(dim=-1).mean()
```

如果输入是 sparse waypoints，需要先插值成 T 帧。

---

### 5.3 终点误差

目标：

```text
最后一帧到达目标点。
```

loss：

```python
L_goal = ((xz[:, -1] - target_path[:, -1]) ** 2).sum(dim=-1).mean()
```

---

### 5.4 速度均匀

目标：

```text
避免忽快忽慢。
```

loss：

```python
vel = xz[:, 1:] - xz[:, :-1]
speed = vel.norm(dim=-1)
L_speed = speed.var(dim=-1).mean()
```

---

### 5.5 轨迹平滑

目标：

```text
减少 root 抖动。
```

loss：

```python
acc = xz[:, 2:] - 2 * xz[:, 1:-1] + xz[:, :-2]
L_smooth = (acc ** 2).sum(dim=-1).mean()
```

可以再加 jerk：

```python
jerk = xz[:, 3:] - 3*xz[:, 2:-1] + 3*xz[:, 1:-2] - xz[:, :-3]
L_jerk = (jerk ** 2).sum(dim=-1).mean()
```

---

### 5.6 heading 和路径切线一致

目标：

```text
人朝向应该大致沿着路径方向。
```

loss：

```python
vel = xz[:, 1:] - xz[:, :-1]
path_theta = torch.atan2(vel[..., 1], vel[..., 0])

heading_theta = torch.atan2(
    heading[:, :-1, 1],
    heading[:, :-1, 0],
)

diff = heading_theta - path_theta
diff = torch.atan2(torch.sin(diff), torch.cos(diff))

L_heading = (diff ** 2).mean()
```

---

### 5.7 root 高度稳定

目标：

```text
防止 root_y 上下乱跳。
```

loss：

```python
root_y = pos[..., 1]
L_height = ((root_y - root_y[:, :1]) ** 2).mean()
```

或者：

```python
L_height_smooth = ((root_y[:, 1:] - root_y[:, :-1]) ** 2).mean()
```

---

## 6. Scene Guidance 设计

### 6.1 用 voxel / SDF 做 root 避障

从场景 voxel 构建 2D walkable map 或 3D SDF。

推荐第一版先做 2D root-level guidance：

```text
scene voxel
    ↓
occupancy map on XZ
    ↓
distance transform / SDF
    ↓
root_xz 距离 obstacle 越近，loss 越大
```

loss：

```python
sdf_value = sample_sdf(scene_sdf, pos)
L_scene = relu(margin - sdf_value)^2.mean()
```

其中：

```text
sdf_value > 0:
    离障碍物有距离

sdf_value < 0:
    在障碍物内部

margin:
    安全距离，比如 0.10m 或 0.15m
```

---

### 6.2 NonWalkableRootRate

评估时计算：

```text
root 落入 occupied / non-walkable 区域的帧比例
```

这个指标可以直接说明：

```text
加入 scene guidance 后，root 是否更少走进障碍区域。
```

---

### 6.3 body collision / penetration

Body 生成后再评估：

```text
body vertices / joints / feet 与 scene voxel 或 SDF 的穿透情况。
```

第一版可以用 joints / pelvis / feet 近似：

```text
joint sphere collision
foot penetration
pelvis penetration
```

正式版本用 SMPL-X vertices：

```text
每帧 vertices 采样 scene SDF
sdf < 0 的顶点为 penetration
```

输出：

```text
PenetrationRate
PenetrationMean
PenetrationMax
CollisionFrameRate
```

---

## 7. 具体代码：root_guidance.py

新增：

```text
kimodo_sceneco/guidance/root_guidance.py
```

参考实现：

```python
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn.functional as F


@dataclass
class RootGuidanceConfig:
    enabled: bool = True

    # loss weights
    w_path: float = 10.0
    w_goal: float = 20.0
    w_speed: float = 1.0
    w_smooth: float = 2.0
    w_jerk: float = 0.5
    w_heading: float = 2.0
    w_height: float = 1.0
    w_scene: float = 5.0

    # scene
    scene_margin: float = 0.10

    # guidance scale
    scale: float = 0.05

    # when to apply guidance
    start_step: int = 0
    end_step: int = 50


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def compute_root_guidance_loss(
    pred_x0: torch.Tensor,
    target_path_xz: torch.Tensor,
    root_slice,
    cfg: RootGuidanceConfig,
    scene_sdf: Optional[torch.Tensor] = None,
    sample_sdf_fn=None,
) -> Dict[str, torch.Tensor]:
    """
    pred_x0:
        (B, T, D), normalized or motion feature space.

    target_path_xz:
        (B, T, 2), same coordinate system as root xz.

    root_slice:
        usually slice(0, 5).

    Returns:
        dict with total loss and each component.
    """
    root = pred_x0[..., root_slice]

    pos = root[..., 0:3]
    heading = root[..., 3:5]
    xz = pos[..., [0, 2]]

    # 1. path dense loss
    loss_path = ((xz - target_path_xz) ** 2).sum(dim=-1).mean()

    # 2. final goal loss
    loss_goal = ((xz[:, -1] - target_path_xz[:, -1]) ** 2).sum(dim=-1).mean()

    # 3. speed uniformity
    vel = xz[:, 1:] - xz[:, :-1]
    speed = vel.norm(dim=-1)
    loss_speed = speed.var(dim=-1).mean()

    # 4. acceleration smoothness
    acc = xz[:, 2:] - 2 * xz[:, 1:-1] + xz[:, :-2]
    loss_smooth = (acc ** 2).sum(dim=-1).mean()

    # 5. jerk
    if xz.shape[1] >= 4:
        jerk = xz[:, 3:] - 3 * xz[:, 2:-1] + 3 * xz[:, 1:-2] - xz[:, :-3]
        loss_jerk = (jerk ** 2).sum(dim=-1).mean()
    else:
        loss_jerk = pred_x0.new_tensor(0.0)

    # 6. heading-path consistency
    path_theta = torch.atan2(vel[..., 1], vel[..., 0])
    heading_theta = torch.atan2(heading[:, :-1, 1], heading[:, :-1, 0])
    loss_heading = (angle_diff(heading_theta, path_theta) ** 2).mean()

    # 7. root height stability
    root_y = pos[..., 1]
    loss_height = ((root_y - root_y[:, :1]) ** 2).mean()

    # 8. scene sdf
    if scene_sdf is not None and sample_sdf_fn is not None:
        sdf_value = sample_sdf_fn(scene_sdf, pos)
        loss_scene = F.relu(cfg.scene_margin - sdf_value).pow(2).mean()
    else:
        loss_scene = pred_x0.new_tensor(0.0)

    total = (
        cfg.w_path * loss_path
        + cfg.w_goal * loss_goal
        + cfg.w_speed * loss_speed
        + cfg.w_smooth * loss_smooth
        + cfg.w_jerk * loss_jerk
        + cfg.w_heading * loss_heading
        + cfg.w_height * loss_height
        + cfg.w_scene * loss_scene
    )

    return {
        "total": total,
        "path": loss_path,
        "goal": loss_goal,
        "speed": loss_speed,
        "smooth": loss_smooth,
        "jerk": loss_jerk,
        "heading": loss_heading,
        "height": loss_height,
        "scene": loss_scene,
    }
```

---

## 8. 具体代码：path_utils.py

新增：

```text
kimodo_sceneco/guidance/path_utils.py
```

作用：

```text
1. sparse waypoint → dense path
2. root path smoothing
3. heading recomputation
4. velocity / acceleration statistics
```

参考：

```python
import torch
import torch.nn.functional as F


def resample_path_to_length(path_xz: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    path_xz:
        (B, K, 2)

    return:
        (B, target_len, 2)
    """
    path_ch = path_xz.transpose(1, 2)  # (B, 2, K)
    dense = F.interpolate(path_ch, size=target_len, mode="linear", align_corners=True)
    return dense.transpose(1, 2)


def smooth_path_xz(path_xz: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Simple moving average smoothing.
    """
    if kernel_size <= 1:
        return path_xz

    B, T, C = path_xz.shape
    pad = kernel_size // 2
    x = path_xz.transpose(1, 2)  # (B, 2, T)
    x = F.pad(x, (pad, pad), mode="replicate")

    weight = torch.ones(C, 1, kernel_size, device=path_xz.device, dtype=path_xz.dtype)
    weight = weight / kernel_size

    y = F.conv1d(x, weight, groups=C)
    return y.transpose(1, 2)


def heading_from_path_xz(path_xz: torch.Tensor) -> torch.Tensor:
    """
    path_xz:
        (B, T, 2)

    return:
        heading cos/sin, (B, T, 2)
    """
    vel = path_xz[:, 1:] - path_xz[:, :-1]
    theta = torch.atan2(vel[..., 1], vel[..., 0])
    theta = torch.cat([theta, theta[:, -1:]], dim=1)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
```

---

## 9. 修改 kimodo_model.py：加入 guidance 参数

当前 `KimodoSceneCo` 已经包装了 denoiser、text encoder、Diffusion、DDIMSampler，并且 patch 了 denoiser forward。新增 guidance 时不建议先动 Root / Body 网络结构，而是在采样流程里加。

需要给 generation / sampling 函数增加参数：

```python
target_path_xz=None
root_guidance_cfg=None
scene_sdf=None
enable_root_guidance=False
```

伪代码：

```python
def generate(
    self,
    prompts,
    num_frames,
    target_path_xz=None,
    root_guidance_cfg=None,
    scene_sdf=None,
    enable_root_guidance=False,
    **kwargs,
):
    ...
```

在 denoising loop 中：

```python
for step, t in enumerate(timesteps):
    x_t.requires_grad_(enable_root_guidance)

    pred_x0 = self.denoiser_or_denoising_step(
        x_t,
        ...
    )

    if enable_root_guidance:
        losses = compute_root_guidance_loss(
            pred_x0=pred_x0,
            target_path_xz=target_path_xz,
            root_slice=self.motion_rep.root_slice,
            cfg=root_guidance_cfg,
            scene_sdf=scene_sdf,
            sample_sdf_fn=sample_sdf,
        )

        grad = torch.autograd.grad(losses["total"], x_t)[0]

        x_t = x_t - root_guidance_cfg.scale * grad
        x_t = x_t.detach()

        # optional: recompute pred_x0 after guided x_t
        pred_x0 = self.denoiser_or_denoising_step(
            x_t,
            ...
        )

    x_t = ddim_update(x_t, pred_x0, t)
```

注意：

```text
1. guidance 不要一开始全 step 都开很强。
2. 建议只在中早期 denoising step 开，最后几步减弱。
3. guidance scale 从 0.01 / 0.03 / 0.05 小范围开始。
```

---

## 10. 如果当前 sampler 不方便改

如果 `DDIMSampler` 封装太深，不方便直接插入 gradient update，可以新增一个脚本级 sampling loop：

```text
scripts/generate_root_guidance.py
```

在脚本里手写 DDIM loop：

```python
x_t = torch.randn(B, T, D, device=device)

for i, t in enumerate(timesteps):
    x_t.requires_grad_(True)

    pred_x0 = model(
        cfg_weight,
        x_t,
        x_pad_mask,
        text_feat,
        text_pad_mask,
        t,
        scene_feat_root=None,
        scene_feat_body=None,
        traj_feats=None,
        cfg_type="text",
    )

    losses = compute_root_guidance_loss(...)
    grad = torch.autograd.grad(losses["total"], x_t)[0]

    x_t = x_t - scale * grad
    x_t = x_t.detach()

    x_t = ddim_step(x_t, pred_x0, t)
```

这会比改原来的 sampler 更安全，适合先 debug。

---

## 11. Stage2 fixed-root body generation

路径 guidance 生成 `guided_root_5d` 后，可以使用 Stage2 生成 body。

建议新增：

```text
scripts/generate_body_from_root.py
```

核心逻辑：

```python
guided_root_5d = load_guided_root(...)

# 1. 每个 denoising step 固定 root
x_t[..., root_slice] = guided_root_5d

# 2. forward 中跳过 Root Model 或强制 root_motion_pred = guided_root_5d
pred_x0 = model(
    ...,
    external_root=guided_root_5d,
    use_external_root=True,
)

# 3. 输出固定 root + generated body
pred_x0[..., root_slice] = guided_root_5d
```

需要在 `kimodo_model.py` 或 `twostage_denoiser.py` 中增加：

```python
external_root=None
use_external_root=False
```

并把 root stage 改成：

```python
if use_external_root:
    root_motion_pred = external_root
else:
    root_motion_pred = self.root_model(...)
```

然后继续：

```python
root_motion_local = motion_rep.global_root_to_local_root(
    root_motion_pred,
    normalized=True,
    lengths=lengths,
)

body_x = x[..., motion_rep.body_slice]
x_new = torch.cat([root_motion_local, body_x], dim=-1)
predicted_body = body_model(...)
output = torch.cat([root_motion_pred, predicted_body], dim=-1)
```

注意：

```text
如果不传 initial pose：
    不需要 observed_motion / motion_mask。
    Body 从 noisy body 开始生成。
```

---

## 12. 评估脚本：eval_path_metrics.py

新增：

```text
eval/eval_path_metrics.py
```

输出 CSV 字段：

```text
sample_id
method
PathADE
PathFDE
WaypointError
HeadingError
SpeedMean
SpeedStd
RootAccel
RootJerk
RootYSmooth
```

参考实现：

```python
def compute_path_metrics(root_5d, target_path_xz):
    pos = root_5d[..., 0:3]
    heading = root_5d[..., 3:5]
    xz = pos[..., [0, 2]]

    path_ade = ((xz - target_path_xz) ** 2).sum(-1).sqrt().mean()
    path_fde = ((xz[:, -1] - target_path_xz[:, -1]) ** 2).sum(-1).sqrt().mean()

    vel = xz[:, 1:] - xz[:, :-1]
    speed = vel.norm(dim=-1)
    speed_mean = speed.mean()
    speed_std = speed.std()

    acc = xz[:, 2:] - 2*xz[:, 1:-1] + xz[:, :-2]
    root_accel = acc.norm(dim=-1).mean()

    jerk = xz[:, 3:] - 3*xz[:, 2:-1] + 3*xz[:, 1:-2] - xz[:, :-3]
    root_jerk = jerk.norm(dim=-1).mean()

    path_theta = torch.atan2(vel[..., 1], vel[..., 0])
    heading_theta = torch.atan2(heading[:, :-1, 1], heading[:, :-1, 0])
    diff = torch.atan2(
        torch.sin(heading_theta - path_theta),
        torch.cos(heading_theta - path_theta),
    )
    heading_error = diff.abs().mean()

    return {
        "PathADE": path_ade.item(),
        "PathFDE": path_fde.item(),
        "SpeedMean": speed_mean.item(),
        "SpeedStd": speed_std.item(),
        "RootAccel": root_accel.item(),
        "RootJerk": root_jerk.item(),
        "HeadingError": heading_error.item(),
    }
```

---

## 13. 评估脚本：eval_sceneadapt_metrics.py

新增：

```text
eval/eval_sceneadapt_metrics.py
```

输出字段：

```text
sample_id
method
CollisionFrameRate
PenetrationRate
PenetrationMean
PenetrationMax
NonWalkableRootRate
SceneSDFPenalty
FootSlide
FloatingRatio
```

第一版可以用 voxel occupancy：

```python
def compute_nonwalkable_root_rate(root_pos, occ_map):
    """
    root_pos:
        (T, 3)

    occ_map:
        2D occupied / non-walkable map.
    """
    # world/canonical xyz -> voxel index
    # occupied index = collision
    ...
```

正式版用 SDF：

```python
sdf_values = sample_sdf(scene_sdf, vertices)
penetrating = sdf_values < 0

penetration_rate = penetrating.float().mean()
penetration_mean = (-sdf_values[penetrating]).mean()
penetration_max = (-sdf_values[penetrating]).max()

collision_frame_rate = penetrating.any(dim=-1).float().mean()
```

如果暂时没有 SMPL-X vertices，可以先用 joints 近似，但报告里必须标注：

```text
joint-based collision proxy，不等同于 mesh-level penetration。
```

---

## 14. 新实验配置 guidance_root_scene.yaml

新增：

```yaml
experiment:
  name: "guidance_root_scene"
  description: "Root classifier guidance for path control and scene collision reduction"

model:
  checkpoint: "models/Kimodo-SMPLX-RP-v1"
  use_trajco: false
  use_sceneco: false

generation:
  num_frames: 196
  num_denoising_steps: 50
  cfg_type: "text"
  cfg_weight: [2.0, 2.0, 2.0]

path_guidance:
  enabled: true
  scale: 0.03

  w_path: 10.0
  w_goal: 20.0
  w_speed: 1.0
  w_smooth: 2.0
  w_jerk: 0.5
  w_heading: 2.0
  w_height: 1.0

scene_guidance:
  enabled: false
  w_scene: 0.0
  scene_margin: 0.10

stage2:
  use_external_root: true
  use_initial_pose: false
  fix_root_each_step: true

eval:
  path_metrics: true
  sceneadapt_metrics: true
  use_mesh_vertices: false
```

第二组实验改成：

```yaml
scene_guidance:
  enabled: true
  w_scene: 5.0
  scene_margin: 0.10
```

---

## 15. 新实验矩阵

建议先跑 3 组：

| 实验 | 路径控制 | 场景控制 | 目的 |
|---|---|---|---|
| **Kimodo-Text** | 否 | 否 | 原始 baseline |
| **Path-Guidance** | 是 | 否 | 证明 root 可被路径控制 |
| **Path+Scene-Guidance** | 是 | 是 | 证明加入场景后减少碰撞和穿模 |

如果还想比较你现在已有方案：

| 实验 | 目的 |
|---|---|
| **D / TrajCo-only** | 旧 cross-attention 轨迹控制 baseline |
| **F / T-root SC-body** | 旧 root/body 分工 baseline |
| **Path-Guidance** | 新 classifier guidance 轨迹控制 |
| **Path+Scene-Guidance** | 新场景适配方案 |

---

## 16. 运行命令建议

### 16.1 只做路径 guidance

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --path_mode dense \
  --scene_guidance false \
  --output_dir outputs/guidance_path_only
```

### 16.2 路径 + 场景 guidance

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --path_mode dense \
  --scene_guidance true \
  --output_dir outputs/guidance_path_scene
```

### 16.3 固定 guided root 生成 body

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/guidance_path_scene/root_npz \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/guidance_path_scene_body
```

### 16.4 评估路径一致性

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/guidance_path_scene_body \
  --target_path_dir outputs/target_paths \
  --output_csv outputs/guidance_path_scene/path_metrics.csv
```

### 16.5 评估 SceneAdapt-style 指标

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/guidance_path_scene_body \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/guidance_path_scene/scene_metrics.csv
```

---

## 17. 预期结果

### 17.1 Path-Guidance 应该做到

相比 Kimodo-Text：

```text
PathADE 下降
PathFDE 下降
HeadingError 下降
```

同时：

```text
SpeedStd 不应明显变大
RootJerk 不应明显变大
FootSlide 不应明显变大
```

如果 PathADE 降了，但 RootJerk 很大，说明 guidance 太硬，需要提高 smooth / jerk 权重或降低 scale。

---

### 17.2 Path+Scene-Guidance 应该做到

相比 Path-Guidance：

```text
NonWalkableRootRate 下降
CollisionFrameRate 下降
PenetrationRate 下降
PenetrationMean 下降
PenetrationMax 下降
```

同时保持：

```text
PathADE 不应大幅变差
PathFDE 不应大幅变差
```

如果场景碰撞下降但 PathADE 大幅变差，说明 scene guidance 太强。

---

## 18. 调参顺序

不要一开始所有 loss 都开。

### Round 1：只开 path + goal

```yaml
w_path: 10.0
w_goal: 20.0
w_speed: 0.0
w_smooth: 0.0
w_heading: 0.0
w_scene: 0.0
scale: 0.03
```

目标：

```text
先确认能拉到路径。
```

---

### Round 2：加入 smooth + speed

```yaml
w_speed: 1.0
w_smooth: 2.0
w_jerk: 0.5
```

目标：

```text
减少 root 抖动，速度更均匀。
```

---

### Round 3：加入 heading

```yaml
w_heading: 2.0
```

目标：

```text
避免横着走、倒着走、突然转身。
```

---

### Round 4：加入 scene

```yaml
w_scene: 5.0
scene_margin: 0.10
```

目标：

```text
减少障碍物碰撞和穿模。
```

---

## 19. 常见错误排查

### 19.1 root 没跟路径

可能原因：

```text
guidance scale 太小
path / root 坐标不一致
target_path_xz 没归一化到 root feature 空间
loss 没有对 x_t 求梯度
```

检查：

```text
打印 grad.norm()
可视化 pred root 和 target path
```

---

### 19.2 root 抖动更严重

可能原因：

```text
path guidance 太硬
smooth / jerk 权重太低
guidance 在最后几步仍然太强
```

解决：

```text
降低 scale
提高 w_smooth / w_jerk
最后 10 个 denoising step 关闭 guidance
```

---

### 19.3 heading 错，人物横着走

可能原因：

```text
atan2(x,z) 顺序错
cos/sin 顺序错
路径坐标轴和 Kimodo 坐标轴不一致
```

解决：

```text
画 heading arrow
对比 GT heading 与重新计算 heading
```

---

### 19.4 场景避障导致不跟路径

可能原因：

```text
target path 本身穿过障碍
scene guidance 权重太大
SDF 坐标转换错
```

解决：

```text
先可视化 path + voxel
先用低 w_scene
对 path 做 scene-aware 修正
```

---

## 20. 最终报告应该怎么写

新实验报告建议结构：

```text
1. 背景：
   TrajCo cross-attention 在 clean root 下有效，但对外部路径/场景融合不稳定。

2. 方法：
   改为 root classifier guidance。
   Guidance loss = path + goal + speed + smooth + heading + scene.

3. 实验 1：
   Kimodo-Text vs Path-Guidance。
   证明轨迹控制有效。

4. 实验 2：
   Path-Guidance vs Path+Scene-Guidance。
   证明加入场景后碰撞和穿模减少。

5. 指标：
   PathADE / PathFDE / HeadingError / SpeedStd / RootJerk
   CollisionFrameRate / PenetrationRate / PenetrationMean / PenetrationMax

6. 结论：
   轨迹控制不再通过 cross-attention 注入，而是在采样阶段用 guidance 引导 root。
   场景不再直接注入 Root Transformer，而通过 SDF/voxel guidance 影响 root 采样。
```

---

## 21. 最短总结

你要改成：

```text
原来：
    5D root trajectory → TrajEncoder → TrajCo cross-attention → denoiser hidden state

现在：
    target path / scene SDF → guidance loss
    guidance loss → gradient
    gradient → denoising sampling step
    最终得到 guided root

然后：
    guided root → Stage2 Body Denoiser
```

核心改动：

```text
1. 新增 root_guidance.py
2. 新增 scene_guidance.py
3. 在采样 loop 加 guidance gradient update
4. 新增 fixed-root Stage2 body generation
5. 新增 Path + SceneAdapt-style evaluation
```

最终要证明：

```text
1. Path guidance 能控制 root。
2. Scene guidance 能减少碰撞和穿模。
3. 生成 root 更平滑、速度更均匀，能作为 Stage2 的稳定条件。
```
