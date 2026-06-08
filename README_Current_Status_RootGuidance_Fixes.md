# README：SceneCoHSI Root Classifier Guidance 当前修改状态与剩余修复方案

> 本 README 基于当前 GitHub 最新状态整理。  
> 当前仓库已经实现了 **Root Classifier Guidance 主流程的大部分核心代码**，但还没有完全闭环。  
> 最大剩余问题是：  
>
> 1. `generate_body_from_root.py` 还没有真正把 `external_root/use_external_root` 传给 denoiser，所以 fixed-root Stage2 仍不可靠。  
> 2. `eval_sceneadapt_metrics.py` 仍使用 `voxel_size=0.02`，而 guidance 配置与 scene guidance 已经使用 `0.1`，评估尺度不一致。  
> 3. `eval_sceneadapt_metrics.py` 还不支持 `Scene/{scene_name}.npy` 这种场景文件路径。  
> 4. `fix_root_each_step` 已经写进 config，但主生成流程中没有显式搜索到对应实现，需要补齐。  

---

## 1. 当前目标

最终目标是把原来的 TrajCo cross-attention 轨迹注入方式改成：

```text
target path / waypoint / planner path
        ↓
Classifier Guidance loss
        ↓
采样阶段引导 Root Stage
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
guided_root_5d = smooth_root_pos(3) + heading(cos, sin)
```

需要证明两件事：

```text
1. 加入路径之后，root 是否能够被控制好。
2. 加入场景之后，碰撞和穿模是否减少，是否接近 SceneAdapt-style 效果。
```

---

## 2. 当前已经完成的内容

### 2.1 `kimodo_model.py` 已经接入主流程 guidance

当前 `kimodo_sceneco/model/kimodo_model.py` 已经包含：

```text
predict_x0()
denoising_step_with_root_guidance()
root_guidance_cfg
target_path_xz
scene_sdf
external_root
use_external_root
```

这说明主模型已经具备：

```text
x_t
    ↓
predict_x0
    ↓
compute_root_guidance_loss
    ↓
torch.autograd.grad
    ↓
只保留 root_slice 梯度
    ↓
gradient clipping
    ↓
guided x_t
    ↓
DDIM step
```

因此，**Classifier Guidance 主流程基本完成**。

---

### 2.2 `kimodo_model.py` patched forward 已经支持 external root

当前 patched denoiser forward 已经有：

```python
external_root=None
use_external_root=False
```

并且 root stage 中已经有：

```python
if use_external_root and external_root is not None:
    root_motion_pred = external_root
else:
    root_motion_pred = _self.root_model(...)
```

这说明如果 `external_root/use_external_root` 真正传入 denoiser，则 Body Stage 的 root condition 可以来自外部 root。

---

### 2.3 `twostage_denoiser.py` 已经支持 external root

当前底层 `TwostageDenoiser.forward()` 也已经有：

```python
external_root: Optional[torch.Tensor] = None
use_external_root: bool = False
```

并且已经按如下逻辑运行：

```python
if use_external_root and external_root is not None:
    root_motion_pred = external_root
else:
    root_motion_pred = self.root_model(...)
```

这说明底层 two-stage denoiser 已具备 fixed-root Stage2 的基础能力。

---

### 2.4 `root_guidance.py` 基本完整

当前 `root_guidance.py` 已经包含：

```text
L_path
L_goal
L_speed
L_smooth
L_jerk
L_heading
L_heading_norm
L_height
L_scene
```

并且已经支持：

```text
normalized root → meter/canonical root
```

也就是：

```text
pred_x0[root_slice] 是 normalized feature
target_path_xz 是 meter/canonical path
```

这部分已经比较接近可用状态。

---

### 2.5 `scene_guidance.py` 初版可用

当前 scene guidance 已经支持：

```text
2D SDF
sample_sdf_2d
voxel_size = 0.1
```

这和 `configs/guidance_root_scene.yaml` 中的 scene guidance 参数是一致的。

---

### 2.6 配置文件已经有关键参数

当前 `configs/guidance_root_scene.yaml` 已经包含：

```yaml
path_guidance:
  enabled: true
  scale: 0.03
  max_grad_norm: 1.0
  w_path: 10.0
  w_goal: 20.0
  w_speed: 1.0
  w_smooth: 2.0
  w_jerk: 0.5
  w_heading: 2.0
  w_heading_norm: 0.5
  w_height: 1.0
  start_step: 0
  end_step: 40

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
```

这部分方向正确。

---

## 3. 当前还没有完成的关键问题

### 3.1 `generate_body_from_root.py` 没有真正使用 external root condition

当前最大问题：

```text
generate_body_from_root.py 没有搜索到 use_external_root。
```

也就是说，脚本现在大概率只是做了：

```python
cur_mot[..., root_slice] = external_root
cur_mot = model.denoising_step(...)
cur_mot[..., root_slice] = external_root
```

这只能保证：

```text
最终输出 root 被手动替换成 external_root
```

但不能保证：

```text
Body Denoiser 的 root condition 来自 external_root
```

正确逻辑必须是：

```python
cur_mot = model.denoising_step(
    ...,
    external_root=external_root,
    use_external_root=True,
)
```

否则可能出现：

```text
final root = guided_root
body condition = root_model 自己预测的 root
```

这会导致：

```text
root/body 对不上
人物扭曲
脚滑
动作不跟 root
```

---

### 3.2 `generate_body_from_root.py` 可能加载的是原始 Kimodo，而不是 `KimodoSceneCo`

如果脚本仍然使用：

```python
from kimodo.model.load_model import load_model
model = load_model(...)
```

那么它加载的是原始 Kimodo 模型，而不是 `KimodoSceneCo` wrapper。

这会造成：

```text
kimodo_sceneco/model/kimodo_model.py 里新增的 predict_x0 / guidance / external_root 不一定生效。
```

建议统一使用 `KimodoSceneCo` wrapper 或者把 external_root 支持也同步进原始 Kimodo model。

---

### 3.3 `eval_sceneadapt_metrics.py` 的 voxel size 仍是 0.02

当前 `eval_sceneadapt_metrics.py` 中仍然可见：

```python
voxel_size=0.02
```

但是：

```text
scene_guidance.py 默认 voxel_size = 0.1
configs/guidance_root_scene.yaml 中 voxel_size = 0.1
```

这会造成：

```text
生成时用 0.1m/voxel 判断场景
评估时用 0.02m/voxel 判断场景
```

结果会非常不可信。

必须统一成：

```python
voxel_size=0.1
```

---

### 3.4 `eval_sceneadapt_metrics.py` 不支持 `Scene/{scene_name}.npy`

当前 loader 只查：

```text
Scene/{scene_name}/semantic_voxel_grid.npy
Scene/{scene_name}/voxel_grid.npy
```

但项目数据很可能是：

```text
Scene/{scene_name}.npy
```

所以要同时支持：

```text
Scene/{scene_name}.npy
Scene/{scene_name}/semantic_voxel_grid.npy
Scene/{scene_name}/voxel_grid.npy
```

否则会出现：

```text
scene voxel 读取失败
scene metrics 变成 NaN
```

---

### 3.5 `fix_root_each_step` 配置存在，但主流程未明确实现

配置文件里已经有：

```yaml
fix_root_each_step: true
```

但是 `kimodo_model.py` 当前没有搜索到：

```text
fix_root_each_step
```

建议补进 `_generate()` 和 `generate_body_from_root.py`：

```python
if use_external_root and fix_root_each_step:
    cur_mot[..., self.motion_rep.root_slice] = external_root
```

并且：

```text
每个 step 前固定一次
每个 step 后再固定一次
```

---

### 3.6 SceneAdapt 评估目前仍是 proxy

当前 `eval_sceneadapt_metrics.py` 是：

```text
joint-level / 2D SDF proxy
```

不是正式：

```text
SMPL-X mesh vertices + 3D SDF penetration
```

所以报告中必须写：

```text
SceneAdapt-style proxy metrics
```

不能直接写：

```text
完整 SceneAdapt 指标
```

后续论文级版本需要加：

```text
SMPL-X vertices
3D scene SDF
vertex-level PenetrationRate
vertex-level PenetrationMean
vertex-level PenetrationMax
```

---

## 4. 必须继续修改的文件

### 4.1 修改 `scripts/generate_body_from_root.py`

目标：

```text
让 Body Denoiser 真正使用 external_root 作为 root condition。
```

需要改成：

```python
cur_mot = model.denoising_step(
    cur_mot,
    motion_pad_mask,
    text_feat,
    text_pad_mask,
    t,
    first_heading_angle,
    motion_mask,
    observed_motion,
    torch.tensor([num_denoising_steps], device=device),
    cfg_weight,
    external_root=external_root,
    use_external_root=True,
)
```

并且每个 step 前后：

```python
cur_mot[..., root_slice] = external_root
```

最后保存：

```python
root_error = (cur_mot[..., root_slice] - external_root).abs().max().item()
print("max root fixed error:", root_error)
assert root_error < 1e-5
```

---

### 4.2 修改 `generate_body_from_root.py` 的模型加载

推荐使用 `KimodoSceneCo`：

```python
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo
```

如果你当前是先 `load_model()` 得到原始 Kimodo，再包装成 `KimodoSceneCo`，逻辑应该类似：

```python
base_model = load_model(...)
model = KimodoSceneCo(
    denoiser=base_model.denoiser,
    text_encoder=base_model.text_encoder,
    num_base_steps=base_model.num_base_steps,
    scene_encoder_type="voxel_vit",
    scene_encoder_config=...,
    device=device,
)
```

如果项目已有专用 loader，优先复用已有 loader。

---

### 4.3 修改 `eval/eval_sceneadapt_metrics.py`

#### 4.3.1 voxel size 改成 0.1

把：

```python
def build_2d_occupancy(voxel_grid, voxel_size=0.02, ...)
```

改成：

```python
def build_2d_occupancy(voxel_grid, voxel_size=0.1, ...)
```

把：

```python
def compute_scene_metrics(..., voxel_size=0.02, ...)
```

改成：

```python
def compute_scene_metrics(..., voxel_size=0.1, ...)
```

---

#### 4.3.2 scene loader 支持三种路径

改成：

```python
def load_scene_voxel(scene_name, scene_dir):
    scene_dir = Path(scene_dir)

    candidates = [
        scene_dir / f"{scene_name}.npy",
        scene_dir / scene_name / "semantic_voxel_grid.npy",
        scene_dir / scene_name / "voxel_grid.npy",
    ]

    for path in candidates:
        if path.exists():
            voxel = np.load(str(path))
            # 如需要，统一 ZYX → XYZ
            if voxel.ndim == 3 and voxel.shape[0] != voxel.shape[-1]:
                voxel = np.transpose(voxel, (2, 1, 0))
            return voxel

    return None
```

---

### 4.4 修改 `kimodo_model.py`：加入 `fix_root_each_step`

在 `_generate()` 参数中增加：

```python
fix_root_each_step: bool = False
```

在 denoising loop 中：

```python
if use_external_root and fix_root_each_step and external_root is not None:
    cur_mot[..., self.motion_rep.root_slice] = external_root
```

step 后再做一次：

```python
if use_external_root and fix_root_each_step and external_root is not None:
    cur_mot[..., self.motion_rep.root_slice] = external_root
```

这样可以保证：

```text
采样中的 root 不被 sampler 意外改掉。
```

---

### 4.5 确认 `denoising_step()` 透传 external root

当前 `denoising_step()` 已经有：

```python
external_root=None
use_external_root=False
```

也传给 denoiser。保留这个逻辑。

需要确保所有调用方都能传进来：

```text
_generate()
generate_body_from_root.py
后续 fixed-root body generation
```

---

## 5. 修改后的验收标准

### 5.1 Root Guidance 验收

运行 Path-Guidance 后必须满足：

```text
grad_norm > 0
PathADE(Path-Guidance) < PathADE(Kimodo-Text)
PathFDE(Path-Guidance) < PathFDE(Kimodo-Text)
```

同时检查：

```text
RootJerk 不应显著升高
SpeedStd 不应显著升高
HeadingError 不应显著升高
```

---

### 5.2 Fixed-root Stage2 验收

运行 `generate_body_from_root.py` 后必须满足：

```text
max_abs(final_root - external_root) < 1e-5
```

如果大于 `1e-5`，说明 fixed-root 没有成功。

还要可视化：

```text
guided root path
generated body pelvis/root
feet motion
root/body alignment
```

---

### 5.3 SceneAdapt-style 评估验收

运行 scene metrics 前必须确认：

```text
scene voxel 不为 None
voxel_size = 0.1
scene_name 能正确加载
```

Path+Scene-Guidance 应该相对 Path-Guidance 降低：

```text
CollisionFrameRate
NonWalkableRootRate
PenetrationRate
PenetrationMean
PenetrationMax
```

---

## 6. 推荐运行命令

注意：使用物理 1 号 GPU 时，推荐：

```bash
export CUDA_VISIBLE_DEVICES=1
```

此时程序内部看到的是第 0 张卡，所以参数写：

```bash
--gpu 0
```

---

### 6.1 Path-only root guidance

```bash
cd /path/to/SceneCoHSI
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models

python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/guidance_path_only \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --gpu 0
```

---

### 6.2 Path+Scene root guidance

先把 `configs/guidance_root_scene.yaml` 改成：

```yaml
scene_guidance:
  enabled: true
  w_scene: 5.0
```

然后运行：

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/guidance_path_scene \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --gpu 0
```

---

### 6.3 Fixed-root Stage2 body generation

修完 `generate_body_from_root.py` 后运行：

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/guidance_path_scene/root_npz \
  --output_dir outputs/guidance_path_scene_body \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0
```

---

### 6.4 Path metrics

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/guidance_path_scene_body \
  --target_path_dir outputs/guidance_path_scene/root_npz \
  --output_csv outputs/guidance_path_scene_body/path_metrics.csv \
  --method path_scene_guidance
```

---

### 6.5 SceneAdapt-style proxy metrics

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/guidance_path_scene_body \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/guidance_path_scene_body/scene_metrics.csv \
  --method path_scene_guidance
```

---

## 7. 最终实验表

| 实验 | 路径 guidance | 场景 guidance | Stage2 fixed root | 用途 |
|---|---:|---:|---:|---|
| Kimodo-Text | 否 | 否 | 否 | 原始 baseline |
| Path-Guidance | 是 | 否 | 否 | 证明 root 可被路径控制 |
| Path+Smooth | 是 | 否 | 否 | 证明 root 更平滑、速度更均匀 |
| Path+Scene | 是 | 是 | 否 | 证明 root 避障有效 |
| PathOnly+Stage2 | 是 | 否 | 是 | 不看场景的 Stage2 baseline |
| PathScene+Stage2 | 是 | 是 | 是 | 证明场景减少 body/scene 碰撞 |

---

## 8. 当前完成度判断

| 模块 | 状态 |
|---|---|
| Root guidance loss | 基本完成 |
| Root guidance 主流程 | 基本完成 |
| external_root patched forward | 基本完成 |
| TwostageDenoiser external root | 基本完成 |
| Config | 基本完成 |
| generate_root_guidance.py | 可用于 root guidance 测试 |
| generate_body_from_root.py | 仍需修，当前 fixed-root Stage2 不可靠 |
| eval_sceneadapt_metrics.py | 仍需修 voxel_size 和 scene loader |
| mesh-level penetration | 未完成，当前仅 proxy |

当前整体完成度：

```text
约 80%
```

最关键剩余：

```text
1. 修 generate_body_from_root.py，让 external_root 真正进入 Body condition。
2. 修 eval_sceneadapt_metrics.py 的 voxel_size 和 scene loader。
3. 在 kimodo_model.py 中显式实现 fix_root_each_step。
```

---

## 9. 最短结论

当前 GitHub 已经实现了大部分 root classifier guidance 主流程，但还不能说“全部功能完成”。

现在必须补齐：

```text
fixed-root Stage2 真正使用 external_root
scene metrics 的 voxel_size 与 scene loader
fix_root_each_step 主流程
```

补齐后，才能正式运行：

```text
Path-Guidance
Path+Scene-Guidance
PathOnly+Stage2
PathScene+Stage2
```

并用：

```text
PathADE / PathFDE
CollisionFrameRate / PenetrationRate / NonWalkableRootRate
```

证明你的两个核心目标：

```text
1. 轨迹能够控制 root。
2. 场景 guidance 能减少碰撞和穿模。
```
