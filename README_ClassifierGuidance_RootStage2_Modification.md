# README：SceneCoHSI 从 TrajCo 改为 Root Classifier Guidance 的完整修改方案

> 目标：  
> 不再使用 **TrajCo cross-attention** 直接注入轨迹。  
> 改为在 diffusion 采样阶段使用 **Classifier Guidance / Energy Guidance** 引导 Kimodo Root Stage 生成可控、平滑、速度均匀、朝向合理的 `guided_root_5d`。  
> 然后固定该 root，交给 Stage2 / Body Denoiser 生成身体动作。  
> 最后用路径一致性指标和 SceneAdapt-style 指标证明：
>
> 1. 加入轨迹后，root 能被控制。
> 2. 加入场景后，碰撞和穿模减少。

---

## 0. 当前问题与修改动机

已有 DEFH 实验说明：

```text
D / TrajCo only:
    TrajCo root+body 效果最好，但仍有 root 抖动。

F / TrajCo-root + SceneCo-body:
    加入 SceneCo 后，人物动作有扭曲，root/body 对不上。

Hclean / TrajCo-root+body + SceneCo-body:
    和 F 接近，没有明显改善。

E / SceneCo only:
    SceneCo root+body 效果最差，root 轨迹严重失控。
```

当前结论：

```text
1. TrajCo cross-attention 对 clean / GT-like root trajectory 有效。
2. 但直接把轨迹特征塞进 Transformer hidden state，容易造成 root 抖动和 body 扭曲。
3. SceneCo 直接进入 Root Stage 很危险。
4. Body Stage 中 TrajCo 和 SceneCo 同时 cross-attention 可能发生条件竞争。
```

因此改成：

```text
不训练新的 TrajEncoder / TrajCo adapter。
不把 path 编码后塞进网络。
不让 SceneCo 直接进入 Root hidden state。
在采样阶段，用路径 loss 和场景 loss 的梯度引导 root。
```

---

## 1. 总体方法

新流程：

```text
target path / waypoint / planner path
        ↓
Classifier Guidance loss
        ↓
在 diffusion 采样阶段引导 Root Stage
        ↓
guided_root_5d
        ↓
固定 guided_root_5d
        ↓
Stage2 / Body Denoiser 生成 body
        ↓
final_motion = [guided_root_5d | generated_body]
```

其中：

```text
guided_root_5d = [
    smooth_root_pos_x,
    smooth_root_pos_y,
    smooth_root_pos_z,
    heading_cos,
    heading_sin
]
```

---

## 2. Classifier Guidance 是什么

这里的 Classifier Guidance 不是新增网络层，也不是训练一个分类器。

它是采样时的梯度引导：

```text
当前 noisy motion x_t
        ↓
Kimodo denoiser 预测 pred_x0
        ↓
从 pred_x0 中取 root_5d
        ↓
计算 root 和目标 path / scene 的 loss
        ↓
对 x_t 求梯度
        ↓
用梯度修改 x_t
        ↓
继续 DDIM / DDPM 采样
```

核心公式：

```text
x_t ← x_t - guidance_scale · ∂L_guidance / ∂x_t
```

训练 loss 是更新模型参数：

```text
θ ← θ - lr · ∂L / ∂θ
```

Classifier Guidance 是更新当前采样状态：

```text
x_t ← x_t - scale · ∂L / ∂x_t
```

所以：

```text
不需要训练 Kimodo 参数。
不需要训练 TrajEncoder。
不需要训练新的 cross-attention。
只需要在采样时计算 loss 和梯度。
```

---

## 3. 与 TrajCo 的区别

| 项目 | TrajCo cross-attention | Classifier Guidance |
|---|---|---|
| 放在哪里 | Transformer hidden state 内部 | diffusion 采样循环中 |
| 是否训练新模块 | 需要训练 TrajEncoder / TrajCo | 通常不需要 |
| 条件形式 | trajectory token / K,V | path / scene 可微 loss |
| 对外部 path | 容易分布偏移 | 更灵活 |
| 是否污染 Body hidden state | 可能会 | 不直接改 Body hidden state |
| 场景避障 | 需要 SceneCo 或新模块 | 直接加 SDF / voxel loss |
| 控制强度 | 由网络学习 | 由 guidance scale 控制 |

---

## 4. 需要证明的两件事

### 4.1 证明轨迹能控制 root

对比：

```text
Kimodo-Text
vs
Path-Guidance
```

指标：

```text
PathADE
PathFDE
WaypointError
HeadingError
SpeedStd
RootAccel
RootJerk
```

预期：

```text
Path-Guidance 的 PathADE / PathFDE 明显低于 Kimodo-Text。
RootJerk 不应明显升高。
SpeedStd 不应明显升高。
HeadingError 应下降或保持。
```

---

### 4.2 证明加入场景后减少碰撞和穿模

对比：

```text
Path-Guidance
vs
Path+Scene-Guidance
```

指标：

```text
CollisionFrameRate
NonWalkableRootRate
PenetrationRate
PenetrationMean
PenetrationMax
SceneSDFPenalty
```

预期：

```text
Path+Scene-Guidance 的碰撞 / 穿模 / non-walkable root 指标低于 Path-Guidance。
PathADE / PathFDE 不应大幅变差。
```

---

## 5. 需要修改的文件总表

| 优先级 | 文件 | 修改目的 |
|---|---|---|
| P0 | `kimodo_sceneco/model/kimodo_model.py` | 接入 root guidance 采样循环；支持 external_root |
| P0 | `kimodo_sceneco/model/twostage_denoiser.py` | 增加 `external_root/use_external_root`，让 Body condition 真来自外部 root |
| P0 | `kimodo_sceneco/guidance/root_guidance.py` | 完善 path / smooth / heading / scene guidance loss |
| P0 | `scripts/generate_root_guidance.py` | 支持外部 path / waypoint / planner path；保存 guided root |
| P0 | `scripts/generate_body_from_root.py` | fixed-root Stage2，确保 Body 使用 guided root |
| P1 | `kimodo_sceneco/guidance/scene_guidance.py` | 修 voxel size、axis order、origin、SDF 采样 |
| P1 | `configs/guidance_root_scene.yaml` | 增加完整 guidance 配置 |
| P1 | `scripts/visualize_guided_root.py` | 可视化 target path、root、heading、scene |
| P2 | `eval/eval_path_metrics.py` | 路径一致性评估 |
| P2 | `eval/eval_sceneadapt_metrics.py` | SceneAdapt-style proxy，后续扩展 mesh-level penetration |
| P2 | `scripts/compare_guidance_results.py` | 汇总不同实验结果 |

---

# 6. 修改 1：完善 `root_guidance.py`

文件：

```text
kimodo_sceneco/guidance/root_guidance.py
```

## 6.1 Guidance loss 总公式

```text
L_guidance =
    λ_path         · L_path
  + λ_goal         · L_goal
  + λ_speed        · L_speed
  + λ_smooth       · L_smooth
  + λ_jerk         · L_jerk
  + λ_heading      · L_heading
  + λ_heading_norm · L_heading_norm
  + λ_height       · L_height
  + λ_scene        · L_scene
```

---

## 6.2 每个 loss 的作用

| Loss | 作用 |
|---|---|
| `L_path` | root 贴近目标路径 |
| `L_goal` | 最后一帧到达目标 |
| `L_speed` | 速度均匀，避免忽快忽慢 |
| `L_smooth` | 减少 root 加速度抖动 |
| `L_jerk` | 减少高频抖动 |
| `L_heading` | heading 和路径切线一致 |
| `L_heading_norm` | 保证 `[cos, sin]` 是单位向量 |
| `L_height` | root_y 不乱跳 |
| `L_scene` | root 避开 occupied / obstacle 区域 |

---

## 6.3 推荐实现

```python
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn.functional as F


@dataclass
class RootGuidanceConfig:
    enabled: bool = True

    # guidance scale
    scale: float = 0.03
    max_grad_norm: float = 1.0

    # step range
    start_step: int = 0
    end_step: int = 50

    # loss weights
    w_path: float = 10.0
    w_goal: float = 20.0
    w_speed: float = 1.0
    w_smooth: float = 2.0
    w_jerk: float = 0.5
    w_heading: float = 2.0
    w_heading_norm: float = 0.5
    w_height: float = 1.0
    w_scene: float = 5.0

    # scene safety margin
    scene_margin: float = 0.10


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def denormalize_root_5d(root_norm, motion_rep, root_slice):
    """
    root_norm:
        (B, T, 5), normalized Kimodo feature.
    return:
        (B, T, 5), meter/canonical coordinate.
    """
    # 注意：这里的 mean/std 字段名需要按照项目 motion_rep 真实字段确认。
    # 如果 motion_rep 有 unnormalize()，优先使用 motion_rep.unnormalize()。
    mean = motion_rep.mean[..., root_slice].to(root_norm.device)
    std = motion_rep.std[..., root_slice].to(root_norm.device)
    return root_norm * std + mean


def compute_root_guidance_loss(
    pred_x0: torch.Tensor,
    target_path_xz: torch.Tensor,
    root_slice,
    cfg: RootGuidanceConfig,
    scene_sdf: Optional[torch.Tensor] = None,
    sample_sdf_fn=None,
    motion_rep=None,
    root_is_normalized: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    pred_x0:
        (B, T, D), Kimodo motion feature.

    target_path_xz:
        (B, T, 2), meter/canonical coordinate.

    root_slice:
        root feature slice, usually first 5 dims.

    return:
        dict of total loss and components.
    """
    root = pred_x0[..., root_slice]

    if root_is_normalized:
        if motion_rep is None:
            raise ValueError("motion_rep is required when root_is_normalized=True")
        root = denormalize_root_5d(root, motion_rep, root_slice)

    pos = root[..., 0:3]
    heading = root[..., 3:5]
    xz = pos[..., [0, 2]]

    # path loss
    loss_path = ((xz - target_path_xz) ** 2).sum(dim=-1).mean()

    # goal loss
    loss_goal = ((xz[:, -1] - target_path_xz[:, -1]) ** 2).sum(dim=-1).mean()

    # velocity / speed
    vel = xz[:, 1:] - xz[:, :-1]
    speed = vel.norm(dim=-1)
    loss_speed = speed.var(dim=-1).mean()

    # smooth acceleration
    acc = xz[:, 2:] - 2 * xz[:, 1:-1] + xz[:, :-2]
    loss_smooth = (acc ** 2).sum(dim=-1).mean()

    # jerk
    if xz.shape[1] >= 4:
        jerk = xz[:, 3:] - 3 * xz[:, 2:-1] + 3 * xz[:, 1:-2] - xz[:, :-3]
        loss_jerk = (jerk ** 2).sum(dim=-1).mean()
    else:
        loss_jerk = pred_x0.new_tensor(0.0)

    # heading-path consistency
    path_theta = torch.atan2(vel[..., 1], vel[..., 0])
    heading_theta = torch.atan2(heading[:, :-1, 1], heading[:, :-1, 0])
    loss_heading = (angle_diff(heading_theta, path_theta) ** 2).mean()

    # heading unit norm
    loss_heading_norm = ((heading.norm(dim=-1) - 1.0) ** 2).mean()

    # root height stability
    root_y = pos[..., 1]
    loss_height = ((root_y - root_y[:, :1]) ** 2).mean()

    # scene sdf
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
        + cfg.w_heading_norm * loss_heading_norm
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
        "heading_norm": loss_heading_norm,
        "height": loss_height,
        "scene": loss_scene,
    }
```

---

# 7. 修改 2：在 `kimodo_model.py` 中新增 `predict_x0()`

文件：

```text
kimodo_sceneco/model/kimodo_model.py
```

目的：

```text
把 denoiser 的 clean prediction 单独拿出来。
Classifier Guidance 要对 pred_x0[root_slice] 算 loss。
```

新增：

```python
def predict_x0(
    self,
    motion,
    pad_mask,
    text_feat,
    text_pad_mask,
    t_map,
    first_heading_angle,
    motion_mask,
    observed_motion,
    cfg_weight,
    scene_feat_root=None,
    scene_mask_root=None,
    scene_feat_body=None,
    scene_mask_body=None,
    traj_feats=None,
    traj_mask=None,
    cfg_type=None,
    external_root=None,
    use_external_root=False,
):
    pred_x0 = self.denoiser(
        cfg_weight,
        motion,
        pad_mask,
        text_feat,
        text_pad_mask,
        t_map,
        first_heading_angle,
        motion_mask,
        observed_motion,
        scene_feat_root=scene_feat_root,
        scene_mask_root=scene_mask_root,
        scene_feat_body=scene_feat_body,
        scene_mask_body=scene_mask_body,
        traj_feats=traj_feats,
        traj_mask=traj_mask,
        external_root=external_root,
        use_external_root=use_external_root,
        cfg_type=cfg_type,
    )
    return pred_x0
```

---

# 8. 修改 3：新增 `denoising_step_with_root_guidance()`

文件：

```text
kimodo_sceneco/model/kimodo_model.py
```

新增：

```python
def denoising_step_with_root_guidance(
    self,
    motion,
    pad_mask,
    text_feat,
    text_pad_mask,
    t,
    first_heading_angle,
    motion_mask,
    observed_motion,
    num_denoising_steps,
    cfg_weight,
    root_guidance_cfg,
    target_path_xz,
    scene_sdf=None,
    scene_feat_root=None,
    scene_mask_root=None,
    scene_feat_body=None,
    scene_mask_body=None,
    traj_feats=None,
    traj_mask=None,
    cfg_type=None,
    sdf_voxel_size=0.1,
    sdf_grid_origin=(0.0, 0.0, 0.0),
    step_id=0,
):
    from kimodo_sceneco.guidance.root_guidance import compute_root_guidance_loss
    from kimodo_sceneco.guidance.scene_guidance import sample_sdf_2d

    # 如果不在 guidance step 范围，走普通 denoising
    if not (root_guidance_cfg.start_step <= step_id <= root_guidance_cfg.end_step):
        with torch.inference_mode():
            return self.denoising_step(
                motion,
                pad_mask,
                text_feat,
                text_pad_mask,
                t,
                first_heading_angle,
                motion_mask,
                observed_motion,
                num_denoising_steps,
                cfg_weight,
                scene_feat_root=scene_feat_root,
                scene_mask_root=scene_mask_root,
                scene_feat_body=scene_feat_body,
                scene_mask_body=scene_mask_body,
                traj_feats=traj_feats,
                traj_mask=traj_mask,
                cfg_type=cfg_type,
            ), {}

    # 1. diffusion timestep mapping
    use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps[0])
    self.diffusion.calc_diffusion_vars(use_timesteps)
    t_map = map_tensor[t]

    # 2. 当前 noisy motion 需要梯度
    x = motion.detach().requires_grad_(True)

    # 3. 预测 pred_x0
    pred_x0 = self.predict_x0(
        motion=x,
        pad_mask=pad_mask,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        t_map=t_map,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        cfg_weight=cfg_weight,
        scene_feat_root=scene_feat_root,
        scene_mask_root=scene_mask_root,
        scene_feat_body=scene_feat_body,
        scene_mask_body=scene_mask_body,
        traj_feats=traj_feats,
        traj_mask=traj_mask,
        cfg_type=cfg_type,
    )

    # 4. guidance loss
    losses = compute_root_guidance_loss(
        pred_x0=pred_x0,
        target_path_xz=target_path_xz,
        root_slice=self.motion_rep.root_slice,
        cfg=root_guidance_cfg,
        scene_sdf=scene_sdf,
        sample_sdf_fn=lambda sdf, pos: sample_sdf_2d(
            sdf,
            pos,
            voxel_size=sdf_voxel_size,
            grid_origin=sdf_grid_origin,
        ) if sdf is not None else None,
        motion_rep=self.motion_rep,
        root_is_normalized=True,
    )

    # 5. grad wrt x_t
    grad = torch.autograd.grad(losses["total"], x)[0]

    # 6. 只保留 root 部分梯度
    root_grad = torch.zeros_like(grad)
    root_grad[..., self.motion_rep.root_slice] = grad[..., self.motion_rep.root_slice]
    grad = root_grad

    # 7. 梯度裁剪，避免 root 抖动
    grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    max_norm = getattr(root_guidance_cfg, "max_grad_norm", 1.0)
    grad = grad * (max_norm / grad_norm).clamp(max=1.0)

    # 8. 更新 x_t
    x_guided = x - root_guidance_cfg.scale * grad
    x_guided = x_guided.detach()

    # 9. 用 guided x_t 正常采样
    with torch.inference_mode():
        x_tm1 = self.denoising_step(
            x_guided,
            pad_mask,
            text_feat,
            text_pad_mask,
            t,
            first_heading_angle,
            motion_mask,
            observed_motion,
            num_denoising_steps,
            cfg_weight,
            scene_feat_root=scene_feat_root,
            scene_mask_root=scene_mask_root,
            scene_feat_body=scene_feat_body,
            scene_mask_body=scene_mask_body,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            cfg_type=cfg_type,
        )

    return x_tm1, {k: v.detach() for k, v in losses.items()}
```

---

# 9. 修改 4：改 `_generate()` 的采样循环

文件：

```text
kimodo_sceneco/model/kimodo_model.py
```

在 `_generate()` 参数中增加：

```python
enable_root_guidance=False
root_guidance_cfg=None
target_path_xz=None
scene_sdf=None
sdf_voxel_size=0.1
sdf_grid_origin=(0.0, 0.0, 0.0)
external_root=None
use_external_root=False
fix_root_each_step=False
```

采样循环改成：

```python
guidance_logs = []

for step_id, i in enumerate(progress_bar(indices)):
    t = torch.tensor([i] * cur_mot.size(0), device=self.device)

    # fixed-root Stage2 时，每步先固定 root
    if use_external_root and fix_root_each_step:
        cur_mot[..., self.motion_rep.root_slice] = external_root

    if enable_root_guidance and root_guidance_cfg is not None:
        cur_mot, losses = self.denoising_step_with_root_guidance(
            motion=cur_mot,
            pad_mask=pad_mask,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            t=t,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            num_denoising_steps=num_denoising_steps,
            cfg_weight=cfg_weight,
            root_guidance_cfg=root_guidance_cfg,
            target_path_xz=target_path_xz,
            scene_sdf=scene_sdf,
            scene_feat_root=scene_feat_root,
            scene_mask_root=scene_mask_root,
            scene_feat_body=scene_feat_body,
            scene_mask_body=scene_mask_body,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            cfg_type=cfg_type,
            sdf_voxel_size=sdf_voxel_size,
            sdf_grid_origin=sdf_grid_origin,
            step_id=step_id,
        )

        if losses:
            guidance_logs.append({
                "step": step_id,
                **{k: float(v.detach().cpu()) for k, v in losses.items()},
            })
    else:
        with torch.inference_mode():
            cur_mot = self.denoising_step(
                cur_mot,
                pad_mask,
                text_feat,
                text_pad_mask,
                t,
                first_heading_angle,
                motion_mask,
                observed_motion,
                num_denoising_steps,
                cfg_weight,
                scene_feat_root=scene_feat_root,
                scene_mask_root=scene_mask_root,
                scene_feat_body=scene_feat_body,
                scene_mask_body=scene_mask_body,
                traj_feats=traj_feats,
                traj_mask=traj_mask,
                cfg_type=cfg_type,
                external_root=external_root,
                use_external_root=use_external_root,
            )

    # fixed-root Stage2 时，每步后再次固定 root
    if use_external_root and fix_root_each_step:
        cur_mot[..., self.motion_rep.root_slice] = external_root
```

---

# 10. 修改 5：给 denoiser forward 增加 `external_root`

需要改两个地方。

---

## 10.1 `twostage_denoiser.py`

文件：

```text
kimodo_sceneco/model/twostage_denoiser.py
```

在 `forward()` 参数中加：

```python
external_root=None
use_external_root=False
```

root stage 改为：

```python
if use_external_root and external_root is not None:
    root_motion_pred = external_root
else:
    root_motion_pred = self.root_model(...)
```

然后继续：

```python
root_motion_local = self.motion_rep.global_root_to_local_root(
    root_motion_pred,
    normalized=True,
    lengths=lengths,
)
body_x = x[..., self.motion_rep.body_slice]
x_new = torch.cat([root_motion_local, body_x], axis=-1)
predicted_body = self.body_model(...)
output = torch.cat([root_motion_pred, predicted_body], axis=-1)
```

---

## 10.2 `kimodo_model.py` 的 patched forward

文件：

```text
kimodo_sceneco/model/kimodo_model.py
```

在 `_sceneco_denoiser_forward()` 中加：

```python
external_root=None
use_external_root=False
```

root stage 改为：

```python
if use_external_root and external_root is not None:
    root_motion_pred = external_root
else:
    if root_has_mods:
        root_motion_pred = _self.root_model(
            x_extended,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle=first_heading_angle,
            scene_feat=_feat_root,
            scene_mask=_mask_root,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            cakey_kwargs=cakey_kwargs_root,
        )
    else:
        root_motion_pred = _self.root_model(
            x_extended,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle=first_heading_angle,
        )
```

---

# 11. 修改 6：Stage2 fixed-root 生成

脚本：

```text
scripts/generate_body_from_root.py
```

正确流程：

```text
guided_root_5d
    ↓
external_root
    ↓
use_external_root=True
    ↓
global_root_to_local_root(external_root)
    ↓
Body Denoiser
    ↓
generated_body
```

采样中必须每步固定：

```python
cur_mot[..., root_slice] = external_root
```

调用 denoising step 时必须传：

```python
external_root=external_root
use_external_root=True
```

采样后检查：

```python
root_error = (final_motion[..., root_slice] - external_root).abs().max().item()
print("max root fixed error:", root_error)
assert root_error < 1e-5
```

如果只做 `cur_mot[..., root_slice] = external_root`，但不传 `external_root/use_external_root`，Body Denoiser 可能仍然使用 Root Model 预测出来的 root condition。那样会出现：

```text
最终 root 是 external_root，
但 body 不是基于 external_root 生成的。
```

这会导致：

```text
root/body 对不上
人物扭曲
脚滑
```

---

# 12. 修改 7：`generate_root_guidance.py`

当前脚本已经有脚本级 gradient guidance。保留这个脚本，但改成更标准。

## 12.1 支持外部路径

增加参数：

```bash
--target_path_file
--waypoint_file
--planner_path_dir
```

输入格式：

```text
dense path:
    .npy / .npz
    shape = (T, 2)

waypoint:
    .npy / .npz
    shape = (K, 2)
    需要插值到 (T, 2)
```

---

## 12.2 保存 root

保存：

```text
outputs/guidance_path_only/root_npz/sample_000_guided_root.npz
```

内容：

```python
{
    "guided_root_5d_norm": guided_root_norm,
    "guided_root_5d_meter": guided_root_meter,
    "target_path_xz": target_path_xz,
    "text": text,
    "scene_name": scene_name,
}
```

---

## 12.3 保存 guidance loss log

保存：

```text
outputs/guidance_path_only/guidance_loss_log.csv
```

字段：

```text
sample_id
step
total
path
goal
speed
smooth
jerk
heading
heading_norm
height
scene
grad_norm
```

---

# 13. 修改 8：`scene_guidance.py`

文件：

```text
kimodo_sceneco/guidance/scene_guidance.py
```

## 13.1 voxel size

项目场景体素下采样后是 `64×64×64`，物理尺度约 6.4m，因此默认应使用：

```text
0.1m / voxel
```

配置：

```yaml
scene_guidance:
  voxel_size: 0.1
  grid_origin: [0.0, 0.0, 0.0]
  axis_order: "ZYX_to_XYZ"
```

---

## 13.2 scene loader

支持三种路径：

```text
Scene/{scene_name}.npy
Scene/{scene_name}/semantic_voxel_grid.npy
Scene/{scene_name}/voxel_grid.npy
```

如果 `scene_guidance.enabled=true` 但 scene 加载失败，直接报错。

不要静默变成 no-scene。

---

## 13.3 2D SDF

第一版可以做 root-level 2D SDF：

```text
scene voxel
    ↓
XZ occupancy map
    ↓
distance transform / SDF
    ↓
root_xz 采样 SDF
```

loss：

```python
loss_scene = ReLU(scene_margin - sdf_value)^2
```

---

# 14. 配置文件

新增或修改：

```text
configs/guidance_root_scene.yaml
```

推荐：

```yaml
experiment:
  name: "root_classifier_guidance_sceneadapt"
  description: "Classifier-guidance root control + SceneAdapt-style evaluation"

model:
  use_trajco: false
  use_sceneco: false

generation:
  num_frames: 196
  num_denoising_steps: 50
  cfg_type: "text"
  cfg_weight: [2.0, 2.0, 2.0]

root_guidance:
  enabled: true
  scale: 0.03
  max_grad_norm: 1.0
  start_step: 0
  end_step: 40

  w_path: 10.0
  w_goal: 20.0
  w_speed: 1.0
  w_smooth: 2.0
  w_jerk: 0.5
  w_heading: 2.0
  w_heading_norm: 0.5
  w_height: 1.0
  w_scene: 0.0

scene_guidance:
  enabled: false
  w_scene: 0.0
  scene_margin: 0.10
  voxel_size: 0.1
  grid_origin: [0.0, 0.0, 0.0]
  axis_order: "ZYX_to_XYZ"

stage2:
  use_external_root: true
  use_initial_pose: false
  fix_root_each_step: true

eval:
  path_metrics: true
  sceneadapt_metrics: true
  mesh_level_penetration: false
```

---

# 15. 新增可视化脚本

新增：

```text
scripts/visualize_guided_root.py
```

每个样本输出：

```text
target path
generated root
heading arrows
scene occupancy
SDF contour
non-walkable points
```

这一步必须做，因为 scene guidance 最容易错在：

```text
voxel_size
axis order
grid origin
path/root coordinate
heading 方向
```

---

# 16. 评估脚本

## 16.1 路径一致性

文件：

```text
eval/eval_path_metrics.py
```

指标：

```text
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

证明：

```text
加入轨迹之后，root 是否控制好了。
```

---

## 16.2 SceneAdapt-style proxy

文件：

```text
eval/eval_sceneadapt_metrics.py
```

指标：

```text
CollisionFrameRate
NonWalkableRootRate
PenetrationRate
PenetrationMean
PenetrationMax
SceneSDFPenalty
```

注意：

```text
第一版是 joint-level / 2D SDF proxy。
正式论文级穿模需要 SMPL-X vertices + 3D SDF。
```

---

# 17. 实验矩阵

## 17.1 第一组：证明路径控制

| 实验 | 路径 guidance | 场景 guidance | 目的 |
|---|---:|---:|---|
| Kimodo-Text | 否 | 否 | 原始 baseline |
| Path-Guidance | 是 | 否 | 证明 root 被路径控制 |
| Path+Smooth | 是 | 否 | 证明 root 更平滑、速度更匀称 |

预期：

```text
PathADE / PathFDE:
    Kimodo-Text > Path-Guidance

RootJerk / SpeedStd:
    Path+Smooth <= Path-Guidance
```

---

## 17.2 第二组：证明场景效果

| 实验 | 路径 guidance | 场景 guidance | 目的 |
|---|---:|---:|---|
| Path-Guidance | 是 | 否 | 不看场景 baseline |
| Path+Scene-Guidance | 是 | 是 | 验证场景减少碰撞和穿模 |

预期：

```text
NonWalkableRootRate:
    Path+Scene < Path-only

CollisionFrameRate:
    Path+Scene < Path-only

PenetrationRate / Mean / Max:
    Path+Scene < Path-only
```

---

## 17.3 第三组：证明 Stage2 可用

| 实验 | Root 来源 | Stage2 | 目的 |
|---|---|---|---|
| PathOnly+Stage2 | Path-Guidance root | fixed-root body generation | 不看场景 baseline |
| PathScene+Stage2 | Path+Scene root | fixed-root body generation | 场景 root + body 生成 |

验收：

```text
max_abs(final_root - guided_root) < 1e-5
body 跟 root 走
FootSlide 不爆炸
root/body 不明显分离
```

---

# 18. 开发顺序

不要一次全部改。建议顺序如下。

## Step 1：Path guidance 最小版

只开：

```yaml
w_path: 10
w_goal: 20
w_speed: 0
w_smooth: 0
w_jerk: 0
w_heading: 0
w_scene: 0
```

目标：

```text
确认 PathADE / PathFDE 能下降。
确认 grad_norm > 0。
```

---

## Step 2：加入 smooth / speed / heading

打开：

```yaml
w_speed: 1
w_smooth: 2
w_jerk: 0.5
w_heading: 2
w_heading_norm: 0.5
```

目标：

```text
减少 root 抖动。
速度更均匀。
heading 更贴近路径切线。
```

---

## Step 3：加入 scene guidance

打开：

```yaml
w_scene: 5
scene_margin: 0.1
```

目标：

```text
NonWalkableRootRate 下降。
CollisionFrameRate 下降。
PenetrationRate 下降。
```

---

## Step 4：fixed-root Stage2

打开：

```yaml
stage2:
  use_external_root: true
  fix_root_each_step: true
  use_initial_pose: false
```

目标：

```text
guided_root 能作为 Stage2 的 root condition。
Body 可以根据 fixed root 生成动作。
```

---

# 19. 对应你的要求

| 你的要求 | 本 README 的对应方案 |
|---|---|
| “加入轨迹之后是不是能够控制好了” | Path-Guidance；PathADE / PathFDE / HeadingError |
| “加入场景之后能不能实现类似 SceneAdapt 的效果” | Path+Scene-Guidance；CFR / Penetration / NonWalkableRootRate |
| “直接传入路径，不看场景，Kimodo 生成动作，把结果放入场景 SceneAdapt 指标对比” | Path-Guidance 作为不看场景 baseline，再用 SceneAdapt-style proxy 指标评估 |
| “证明加入场景之后减少碰撞和穿模” | Path-Guidance vs Path+Scene-Guidance |
| “轨迹加入换一个方式，不再 TrajCo” | 不再使用 TrajEncoder / TrajCo cross-attention，改为采样阶段 gradient guidance |
| “Classifier Guidance 引导 root 轨迹生成” | `denoising_step_with_root_guidance()` |
| “轨迹转向平滑” | `L_smooth + L_jerk + L_heading` |
| “速度匀称” | `L_speed = Var(speed)` |
| “能够给第二阶段使用” | `guided_root_5d → external_root → global_root_to_local_root → Body Denoiser` |
| “不传 initial pose” | `use_initial_pose: false`，Body 从 noisy body 自己生成 |

---

# 20. 最短总结

最终要实现的是：

```text
第一阶段：
    Kimodo 正常从噪声采样。
    每个 denoising step 中：
        预测 pred_x0
        取 root
        计算 path / smooth / heading / scene loss
        对 x_t 求梯度
        只修改 root 部分
    得到 guided_root_5d。

第二阶段：
    fixed guided_root_5d。
    跳过 Root Denoiser 或强制 root_motion_pred = guided_root_5d。
    Body Denoiser 根据 guided_root_5d 生成 body。

评估：
    PathADE / PathFDE 证明轨迹控制。
    CFR / Penetration / NonWalkableRootRate 证明场景碰撞和穿模减少。
```

一句话：

```text
Classifier Guidance 不加在模型前面，不训练新模块，而是在每个 diffusion 采样 step 里用路径/场景 loss 的梯度修改 x_t，把 root 一步步拉向目标路径，并保持平滑、速度均匀和场景可行。
```
