# SceneCoHSI：Root Classifier Guidance 方案剩余修改 README

> 当前仓库已经搭出了 `guidance/`、路径指标、SceneAdapt-style proxy 指标、`generate_root_guidance.py`、`generate_body_from_root.py` 等基础文件。  
> 但是现在还没有完全形成“可运行闭环”。最关键的缺口是：**guidance 还没有真正进入 denoising loop；external_root 还没有接入 KimodoSceneCo 的 patched forward；路径/场景坐标和归一化还需要修正。**

---

## 0. 当前目标

现在要实现的目标不是继续用 TrajCo cross-attention 注入轨迹，而是：

```text
目标路径 / waypoints / planner path
        ↓
Classifier Guidance
        ↓
引导 Kimodo Root Stage 生成 root
        ↓
guided_root_5d
        ↓
固定 root，Stage2 Body Denoiser 生成 body
        ↓
final_motion = [guided_root_5d | generated_body]
```

需要证明两件事：

```text
1. 加入路径 guidance 后，root 是否能被路径控制。
2. 加入场景 guidance 后，collision / penetration / non-walkable root 是否下降。
```

---

## 1. 当前已经完成的内容

### 1.1 已有 guidance 模块

当前已经有：

```text
kimodo_sceneco/guidance/
├── root_guidance.py
├── scene_guidance.py
└── path_utils.py
```

其中 `root_guidance.py` 已经包含：

```text
L_path
L_goal
L_speed
L_smooth
L_jerk
L_heading
L_height
L_scene
```

也就是路径一致性、终点误差、速度均匀、轨迹平滑、jerk、heading-path 一致性、root 高度稳定、scene SDF 避障。

---

### 1.2 已有场景 SDF 初版

当前 `scene_guidance.py` 已经有：

```text
build_2d_sdf()
sample_sdf_2d()
```

它可以从 voxel 构造 2D SDF，然后按 root 的 XZ 位置采样。

---

### 1.3 已有评估脚本

当前已经有：

```text
eval/eval_path_metrics.py
eval/eval_sceneadapt_metrics.py
```

`eval_path_metrics.py` 已经包含：

```text
PathADE
PathFDE
SpeedMean
SpeedStd
RootAccel
RootJerk
HeadingError
RootYSmooth
```

`eval_sceneadapt_metrics.py` 已经包含：

```text
CollisionFrameRate
NonWalkableRootRate
PenetrationRate
PenetrationMean
PenetrationMax
SceneSDFPenalty
```

注意：目前这个 SceneAdapt-style 评估还是 **2D / joint-level proxy**，不是 SMPL-X mesh vertices 的正式 3D penetration。

---

### 1.4 `TwostageDenoiser` 已经有 external_root 分支

当前 `kimodo_sceneco/model/twostage_denoiser.py` 里已经有：

```python
external_root: Optional[torch.Tensor] = None
use_external_root: bool = False
```

并且已经有：

```python
if use_external_root and external_root is not None:
    root_motion_pred = external_root
else:
    root_motion_pred = self.root_model(...)
```

这说明 **底层 TwostageDenoiser 已经支持 external_root**。

但是这还不够，因为当前主要使用的 `KimodoSceneCo` 会 monkey-patch denoiser forward，patched forward 里还没有 external_root 分支。

---

## 2. 现在还没有完全符合要求的地方

### 2.1 `kimodo_model.py` 还没有接入 root guidance

当前 `kimodo_sceneco/model/kimodo_model.py` 中没有真正出现：

```python
root_guidance_cfg
target_path_xz
scene_sdf
compute_root_guidance_loss
sample_sdf_2d
```

因此现在的状态是：

```text
guidance loss 文件存在，
但 KimodoSceneCo 的 denoising loop 没有真正调用它。
```

---

### 2.2 `_generate()` 里使用了 `torch.inference_mode()`

当前 `_generate()` denoising loop 里是：

```python
with torch.inference_mode():
    cur_mot = self.denoising_step(...)
```

但是 classifier guidance 必须要梯度：

```python
cur_mot.requires_grad_(True)
loss = guidance_loss(pred_x0)
grad = torch.autograd.grad(loss, cur_mot)
cur_mot = cur_mot - scale * grad
```

所以 guidance loop 不能放在 `torch.inference_mode()` 或 `torch.no_grad()` 内。

---

### 2.3 `generate_root_guidance.py` 当前不是真正 classifier guidance

当前脚本有两个问题：

```text
1. 它直接调用原始 kimodo.model.load_model，而不是 KimodoSceneCo wrapper。
2. 它在 with torch.no_grad() 里面调用 model(... root_guidance_cfg=...)。
```

如果模型内部没有自己处理这些参数，这个脚本就不会真正做 classifier guidance。

正确做法是：

```text
脚本内手写 denoising loop
或者
把 guidance 正式接入 KimodoSceneCo._generate()
```

建议优先接入 `KimodoSceneCo._generate()`，这样以后所有生成脚本都能复用。

---

### 2.4 `KimodoSceneCo` patched forward 还没有 `external_root`

虽然 `twostage_denoiser.py` 已经支持 external_root，但 `kimodo_model.py` 里 `_sceneco_denoiser_forward()` 还没有：

```python
external_root=None
use_external_root=False
```

也没有：

```python
if use_external_root:
    root_motion_pred = external_root
else:
    root_motion_pred = _self.root_model(...)
```

所以如果运行的是 `KimodoSceneCo` wrapper，`external_root` 目前不会生效。

---

### 2.5 路径 guidance 坐标 / 归一化可能不一致

当前 `generate_root_guidance.py` 提取 target path 的方式是：

```python
gt_root_xz = extract_gt_root_path(model.motion_rep, feat_t)
target_path_xz = gt_root_xz
```

这是 **世界坐标 / meter 坐标**。

但是 denoising 中的 `pred_x0` 通常是 **normalized motion feature**。

如果直接在 `compute_root_guidance_loss(pred_x0, target_path_xz)` 里比较：

```python
pred_x0[..., root_slice]  vs  target_path_xz
```

那会出现尺度不一致：

```text
pred_x0 root: normalized feature space
target_path_xz: meter / world coordinate
```

这会导致 guidance 方向错误。

必须二选一：

```text
方案 A：把 target path 转成 normalized root feature，再在 normalized feature space 里算 path loss。
方案 B：把 pred root 反归一化成 meter root，再在 world coordinate 里算 path / scene loss。
```

推荐第一版用方案 B：

```text
pred_root_norm
    ↓ denormalize root 5D
pred_root_meters
    ↓
path loss + scene loss
```

这样 PathADE、Scene SDF、可视化都在同一坐标系里。

---

### 2.6 Scene voxel size 目前默认 0.02，不符合项目 README

当前 `scene_guidance.py` 和脚本中默认：

```python
voxel_size = 0.02
```

但是项目 README 里场景体素说明是：

```text
原始场景体素 300×100×400，下采样到 64×64×64。
物理范围约 6.4m。
体素大小约 0.1m/voxel。
```

所以第一版应该把配置改成：

```yaml
scene_guidance:
  voxel_size: 0.1
```

或者从 scene metadata 自动计算。

否则 SDF 距离、碰撞、non-walkable root 都可能错。

---

### 2.7 SceneAdapt-style 指标目前还是 proxy

当前 `eval_sceneadapt_metrics.py` 用：

```text
root
22 个 joints
2D SDF
```

来估计碰撞 / 穿模。

这可以作为快速 proxy，但如果报告里要正式说“穿模减少”，需要后续加：

```text
SMPL-X mesh vertices
3D scene SDF
PenetrationRate_vertices
PenetrationMean_vertices
PenetrationMax_vertices
CollisionFrameRate_vertices
```

---

## 3. 必须修改的文件清单

按优先级：

```text
1. kimodo_sceneco/model/kimodo_model.py
   - 加 root_guidance 参数
   - 修改 _generate()，实现 classifier guidance loop
   - 给 patched denoiser forward 加 external_root/use_external_root

2. scripts/generate_root_guidance.py
   - 不要 with torch.no_grad()
   - 不要假设原始 Kimodo model 支持 root_guidance_cfg
   - 改成调用 KimodoSceneCo guidance generation 或脚本内手写 sampling loop

3. scripts/generate_body_from_root.py
   - 确认加载的是支持 external_root 的模型
   - 每个 denoising step 固定 root_slice
   - 保存 final root 是否等于 external_root 的校验结果

4. kimodo_sceneco/guidance/root_guidance.py
   - 增加 root denormalization 支持
   - 区分 normalized-space loss 和 meter-space loss

5. kimodo_sceneco/guidance/scene_guidance.py
   - 修 voxel_size 默认值
   - 修 grid_origin / axis order
   - 增加可视化检查函数

6. eval/eval_sceneadapt_metrics.py
   - 明确标注当前是 joint-level 2D proxy
   - 后续增加 mesh-level 3D penetration
```

---

## 4. 修改 1：在 `KimodoSceneCo.__call__` / `forward` 增加 guidance 参数

在 `kimodo_sceneco/model/kimodo_model.py` 中，生成函数需要增加：

```python
root_guidance_cfg=None
target_path_xz=None
scene_sdf=None
sdf_voxel_size=0.1
sdf_grid_origin=(0.0, 0.0, 0.0)
enable_root_guidance=False
```

同时传给 `_generate()`：

```python
motion = self._generate(
    texts,
    max_frames,
    num_denoising_steps=num_denoising_steps,
    pad_mask=motion_pad_mask,
    first_heading_angle=first_heading_angle,
    motion_mask=motion_mask,
    observed_motion=observed_motion,
    cfg_weight=cfg_weight,
    cfg_type=cfg_type,
    scene_feat_root=scene_feat_root,
    scene_mask_root=scene_mask_root,
    scene_feat_body=scene_feat_body,
    scene_mask_body=scene_mask_body,
    traj_feats=traj_feats,
    traj_mask=traj_mask,
    root_guidance_cfg=root_guidance_cfg,
    target_path_xz=target_path_xz,
    scene_sdf=scene_sdf,
    sdf_voxel_size=sdf_voxel_size,
    sdf_grid_origin=sdf_grid_origin,
    enable_root_guidance=enable_root_guidance,
    progress_bar=progress_bar,
)
```

---

## 5. 修改 2：在 `_generate()` 中实现 guidance loop

当前 `_generate()` 中是：

```python
for i in progress_bar(indices):
    t = torch.tensor([i] * cur_mot.size(0), device=self.device)
    with torch.inference_mode():
        cur_mot = self.denoising_step(...)
```

需要改成：

```python
for step_id, i in enumerate(progress_bar(indices)):
    t = torch.tensor([i] * cur_mot.size(0), device=self.device)

    if enable_root_guidance and root_guidance_cfg is not None:
        cur_mot = self.denoising_step_with_root_guidance(
            cur_mot=cur_mot,
            pad_mask=pad_mask,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            t=t,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            num_denoising_steps=num_denoising_steps,
            cfg_weight=cfg_weight,
            scene_feat_root=scene_feat_root,
            scene_mask_root=scene_mask_root,
            scene_feat_body=scene_feat_body,
            scene_mask_body=scene_mask_body,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            cfg_type=cfg_type,
            root_guidance_cfg=root_guidance_cfg,
            target_path_xz=target_path_xz,
            scene_sdf=scene_sdf,
            sdf_voxel_size=sdf_voxel_size,
            sdf_grid_origin=sdf_grid_origin,
            step_id=step_id,
        )
    else:
        with torch.inference_mode():
            cur_mot = self.denoising_step(...)
```

注意：

```text
只有 guidance 分支需要梯度。
普通分支继续 inference_mode。
```

---

## 6. 修改 3：新增 `denoising_step_with_root_guidance()`

在 `KimodoSceneCo` 中新增方法：

```python
def denoising_step_with_root_guidance(
    self,
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
    scene_feat_root,
    scene_mask_root,
    scene_feat_body,
    scene_mask_body,
    traj_feats,
    traj_mask,
    cfg_type,
    root_guidance_cfg,
    target_path_xz,
    scene_sdf=None,
    sdf_voxel_size=0.1,
    sdf_grid_origin=(0.0, 0.0, 0.0),
    step_id=0,
):
    ...
```

核心逻辑：

```python
from kimodo_sceneco.guidance.root_guidance import compute_root_guidance_loss
from kimodo_sceneco.guidance.scene_guidance import sample_sdf_2d

# 1. 是否在指定 step 范围内启用
if not (root_guidance_cfg.start_step <= step_id <= root_guidance_cfg.end_step):
    with torch.inference_mode():
        return self.denoising_step(...)

# 2. 允许对当前 noisy motion 求梯度
x = cur_mot.detach().requires_grad_(True)

# 3. 只预测 pred_x0，不做 DDIM update
pred_x0 = self.predict_x0_from_denoiser(
    x=x,
    pad_mask=pad_mask,
    text_feat=text_feat,
    text_pad_mask=text_pad_mask,
    t=t,
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

# 4. 计算 guidance loss
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
    ),
    motion_rep=self.motion_rep,
    root_is_normalized=True,
)

# 5. 反传到 x
grad = torch.autograd.grad(losses["total"], x)[0]

# 6. 更新 x
x_guided = x - root_guidance_cfg.scale * grad
x_guided = x_guided.detach()

# 7. 用 guided x 做原来的 denoising step
with torch.inference_mode():
    out = self.denoising_step(
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

return out
```

---

## 7. 关键问题：需要拿到 `pred_x0`

Classifier guidance 正常应该对 `pred_x0` 算 loss，而不是对最终 `x_{t-1}` 直接算。

因此需要新增一个函数：

```python
predict_x0_from_denoiser(...)
```

这个函数只做：

```text
cur_mot + condition + timestep
    ↓
denoiser / CFG denoiser
    ↓
pred_x0
```

不要做 DDIM update。

如果当前 `denoising_step()` 内部已经混合了：

```text
denoiser predict x0
+
sampler update x_{t-1}
```

就必须拆开。

建议结构：

```python
def predict_x0_from_denoiser(...):
    pred_x0 = self.denoiser(
        cur_mot,
        pad_mask,
        text_feat,
        text_pad_mask,
        t,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        scene_feat_root=scene_feat_root,
        scene_mask_root=scene_mask_root,
        scene_feat_body=scene_feat_body,
        scene_mask_body=scene_mask_body,
        traj_feats=traj_feats,
        traj_mask=traj_mask,
    )
    return pred_x0
```

如果用了 classifier-free guidance，则要复用现有 `ClassifierFreeGuidedModel` 逻辑。

---

## 8. 修改 4：root guidance 必须处理归一化

当前 `compute_root_guidance_loss()` 直接从：

```python
root = pred_x0[..., root_slice]
```

取 root。

这需要补参数：

```python
motion_rep=None
root_is_normalized=True
```

推荐修改为：

```python
def compute_root_guidance_loss(
    pred_x0,
    target_path_xz,
    root_slice,
    cfg,
    scene_sdf=None,
    sample_sdf_fn=None,
    motion_rep=None,
    root_is_normalized=True,
):
    root = pred_x0[..., root_slice]

    if root_is_normalized:
        root = denormalize_root_5d(root, motion_rep, root_slice)

    ...
```

新增函数：

```python
def denormalize_root_5d(root_norm, motion_rep, root_slice):
    """
    root_norm:
        (B, T, 5)

    return:
        root_5d in meter / heading cos-sin coordinate.
    """
    mean = motion_rep.mean[..., root_slice].to(root_norm.device)
    std = motion_rep.std[..., root_slice].to(root_norm.device)
    return root_norm * std + mean
```

实际 `mean/std` 字段名要按 `motion_rep` 真实实现检查。  
如果没有直接字段，就用项目已有 `motion_rep.unnormalize()`，但要保持 torch tensor 和梯度。

---

## 9. 修改 5：patched denoiser forward 加 external_root

在 `kimodo_model.py` 的 `_sceneco_denoiser_forward()` 中加参数：

```python
external_root=None
use_external_root=False
```

把：

```python
root_motion_pred = _self.root_model(...)
```

改成：

```python
if use_external_root and external_root is not None:
    root_motion_pred = external_root
else:
    root_motion_pred = _self.root_model(...)
```

完整位置：

```python
def _sceneco_denoiser_forward(
    _self,
    x,
    x_pad_mask,
    text_feat,
    text_feat_pad_mask,
    timesteps,
    first_heading_angle=None,
    motion_mask=None,
    observed_motion=None,
    scene_feat=None,
    scene_mask=None,
    scene_feat_root=None,
    scene_mask_root=None,
    scene_feat_body=None,
    scene_mask_body=None,
    traj_feats=None,
    traj_mask=None,
    cakey_kwargs_root=None,
    cakey_kwargs_body=None,
    external_root=None,
    use_external_root=False,
):
    ...
```

然后 root 部分：

```python
if use_external_root and external_root is not None:
    root_motion_pred = external_root
else:
    root_motion_pred = _self.root_model(...)
```

这样 `generate_body_from_root.py` 才能真正固定 root。

---

## 10. 修改 6：每个 step 固定 root_slice

在 fixed-root Stage2 body generation 里，必须每一步都做：

```python
cur_mot[..., root_slice] = external_root
```

并且 denoiser 输出后也做：

```python
pred_x0[..., root_slice] = external_root
```

否则 diffusion sampler 可能把 root 改掉。

建议新增配置：

```yaml
stage2:
  use_external_root: true
  fix_root_each_step: true
```

采样 loop 伪代码：

```python
for i in timesteps:
    if use_external_root:
        cur_mot[..., root_slice] = external_root

    pred = denoising_step(..., external_root=external_root, use_external_root=True)

    if use_external_root:
        pred[..., root_slice] = external_root

    cur_mot = pred
```

---

## 11. 修改 7：`generate_root_guidance.py`

当前脚本要改成二选一。

### 方案 A：调用修改后的 `KimodoSceneCo`

```python
output = model(
    prompts=[sample["text"]],
    num_frames=T,
    num_denoising_steps=...,
    root_guidance_cfg=root_guidance_cfg,
    target_path_xz=target_path_xz,
    scene_sdf=scene_sdf,
    enable_root_guidance=True,
)
```

前提是 `KimodoSceneCo` 已经支持这些参数。

同时删除外层：

```python
with torch.no_grad():
```

因为 guidance 需要梯度。

---

### 方案 B：脚本里手写采样 loop

如果暂时不想改 `KimodoSceneCo`，就直接在脚本里手写：

```python
cur_mot = torch.randn(...)
for i in timesteps:
    cur_mot.requires_grad_(True)
    pred_x0 = model.denoiser(...)
    loss = compute_root_guidance_loss(...)
    grad = torch.autograd.grad(loss["total"], cur_mot)[0]
    cur_mot = cur_mot - scale * grad
    cur_mot = ddim_step(...)
```

优点：

```text
调试快。
```

缺点：

```text
后续复用差。
```

推荐最终采用方案 A。

---

## 12. 修改 8：`generate_body_from_root.py`

现在要确认三件事：

```text
1. 加载的模型是否真的使用了支持 external_root 的 denoiser。
2. external_root 是 normalized root_5d，而不是 meter root。
3. 每个 denoising step 是否固定 root_slice。
```

建议保存调试检查：

```python
root_error = (final_motion[..., root_slice] - external_root).abs().max().item()
print("max root fixed error:", root_error)
```

要求：

```text
root_error < 1e-5
```

如果不是，说明 root 没有被固定。

---

## 13. 修改 9：scene_guidance 坐标修正

当前 `voxel_size=0.02` 需要改。

配置中加入：

```yaml
scene_guidance:
  enabled: false
  w_scene: 0.0
  scene_margin: 0.10
  voxel_size: 0.1
  grid_origin: [0.0, 0.0, 0.0]
  axis_order: "XYZ"
```

如果原始 LINGO scene 是：

```text
300×100×400，轴序 Z,Y,X
```

则加载后要转成：

```python
# input: (Z, Y, X)
voxel_grid = np.transpose(voxel_grid, (2, 1, 0))  # -> (X, Y, Z)
```

并把这个逻辑写进：

```python
load_scene_voxel_aligned()
```

---

## 14. 修改 10：增加可视化检查

必须新增或扩展脚本：

```text
scripts/visualize_guided_root.py
```

每个样本输出：

```text
1. target path
2. generated root
3. heading arrows
4. scene occupancy / SDF contour
5. non-walkable frames marked red
```

如果没有这一步，很难判断错误来自：

```text
path guidance
heading
scene 坐标
voxel_size
root normalization
```

---

## 15. 修改 11：评估结果汇总表

新增：

```text
scripts/compare_guidance_results.py
```

输入：

```text
outputs/kimodo_text/
outputs/path_guidance/
outputs/path_scene_guidance/
```

输出：

```text
comparison.csv
comparison.md
```

字段：

```text
Method
PathADE
PathFDE
HeadingError
SpeedStd
RootJerk
CollisionFrameRate
NonWalkableRootRate
PenetrationRate
PenetrationMean
PenetrationMax
FootSlide
```

目标是能直接写成结论：

```text
Path-Guidance vs Kimodo-Text:
    PathADE / PathFDE 下降。

Path+Scene-Guidance vs Path-Guidance:
    CFR / Penetration / NonWalkableRootRate 下降。
```

---

## 16. 修改优先级

### P0：必须先改，否则跑不通

```text
[ ] kimodo_model.py: guidance 参数接入
[ ] kimodo_model.py: _generate() 去掉 guidance 分支的 inference_mode
[ ] kimodo_model.py: 新增 denoising_step_with_root_guidance()
[ ] kimodo_model.py: patched denoiser forward 加 external_root
[ ] root_guidance.py: 处理 normalized root 和 meter target path 的尺度统一
```

---

### P1：跑得通但结果可能错

```text
[ ] scene_guidance.py: voxel_size 0.02 → 0.1 或自动读取
[ ] scene_guidance.py: 处理 LINGO scene axis order
[ ] generate_body_from_root.py: 每 step 固定 root_slice
[ ] generate_root_guidance.py: 删除 no_grad，改用真实 guidance loop
[ ] 可视化 target path / root / heading / scene
```

---

### P2：论文级评估需要

```text
[ ] eval_sceneadapt_metrics.py: 增加 SMPL-X mesh vertices penetration
[ ] eval_sceneadapt_metrics.py: 3D SDF 而不是 2D SDF proxy
[ ] compare_guidance_results.py: 多方法汇总
[ ] 保存 per-frame collision / penetration 可视化
```

---

## 17. 最小可运行闭环

如果时间紧，只做这个最小闭环：

```text
1. 只开 path + goal guidance，不开 scene。
2. 在 KimodoSceneCo._generate() 中加 gradient update。
3. 输出 guided_root。
4. eval_path_metrics.py 验证 PathADE / PathFDE。
5. 再把 guided_root 作为 external_root 给 Stage2。
6. 检查 final root 是否完全等于 guided_root。
```

最小实验表：

| 实验 | 目标 |
|---|---|
| Kimodo-Text | baseline |
| Path-Guidance | 证明 root 可控 |
| Path+Smooth-Guidance | 证明抖动降低 |
| Path+Scene-Guidance | 证明碰撞减少 |

---

## 18. 判断是否符合要求的验收标准

### 18.1 Guidance 是否真的生效

必须满足：

```text
grad.norm() > 0
PathADE(Path-Guidance) < PathADE(Kimodo-Text)
PathFDE(Path-Guidance) < PathFDE(Kimodo-Text)
```

---

### 18.2 Root 是否足够平滑

必须满足：

```text
RootJerk 不明显升高
SpeedStd 不明显升高
HeadingError 下降或保持
```

如果 PathADE 下降但 RootJerk 激增，说明 guidance 太硬。

---

### 18.3 Scene 是否真的减少碰撞

必须满足：

```text
CollisionFrameRate(Path+Scene) < CollisionFrameRate(Path-only)
NonWalkableRootRate(Path+Scene) < NonWalkableRootRate(Path-only)
PenetrationRate(Path+Scene) < PenetrationRate(Path-only)
```

---

### 18.4 Stage2 是否正确使用 external root

必须满足：

```text
max_abs(final_output[root_slice] - external_root) < 1e-5
```

如果不满足，说明 fixed-root 没有成功。

---

## 19. 最终结论

当前仓库已经完成了：

```text
guidance loss
scene SDF 初版
path metrics
sceneadapt proxy metrics
生成脚本初版
external_root 的底层 twostage_denoiser 分支
```

但还必须修改：

```text
1. 把 guidance 真正接入 KimodoSceneCo 的 denoising loop。
2. 去掉 guidance 分支的 inference_mode / no_grad。
3. 解决 normalized pred_x0 与 meter path 的尺度统一。
4. 把 external_root 接进 KimodoSceneCo patched forward。
5. 每个 sampling step 固定 root_slice。
6. 修 scene voxel_size、origin、axis order。
7. 增加可视化和汇总评估。
```

一句话：

```text
现在是“文件和 loss 都有了”，但还不是“算法闭环已经跑通”。
下一步最重要的是打通：
target path → root guidance gradient → guided root → fixed-root Stage2 → Path/SceneAdapt 评估。
```
