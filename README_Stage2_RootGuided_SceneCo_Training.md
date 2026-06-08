# README：Stage2 Root-Guided SceneCo 训练方案

> 目的：训练一个 **输入 guided root / external root** 的 Stage2 Body SceneCo 模型。  
> Root 不再由 Stage1 生成，也不再使用 TrajCo。Root 先由 **Classifier Guidance** 生成，训练时固定该 root，Stage2 使用 **SceneCo body-only** 生成身体动作。  
>
> 最终目标：
>
> ```text
> target path / waypoint / planner path
>     ↓
> Root Classifier Guidance
>     ↓
> guided_root_5d
>     ↓
> Stage2 fixed-root Body Denoiser
>     ↓
> SceneCo body-only
>     ↓
> generated body motion
> ```

---

## 0. 当前项目状态

当前仓库已经具备以下基础：

```text
1. KimodoSceneCo / TwostageDenoiser 已支持 Root → Body 两阶段结构。
2. motion feature 是 273 维：
   [smooth_root_pos(3) | heading(2) | local_joints(66) | global_rot_data(132) | velocities(66) | foot_contacts(4)]
3. root_slice 是前 5 维：
   smooth_root_pos(3) + heading(cos, sin)
4. SceneCo / TrajCo 训练代码已有。
5. Root Classifier Guidance 生成 guided_root_5d 的脚本已有。
6. fixed-root Stage2 生成脚本已有。
```

但是，当前训练代码还不是“Stage2 Root-Guided SceneCo 训练”：

```text
1. 当前 train.py 中 training_losses() 没有传 external_root / use_external_root。
2. 当前 _prepare_batch() 没有从 batch 读 guided_root_5d。
3. 当前 loss_mask 主要支持 root_trajectory_data 或全维 MSE，不是专门 body-only loss。
4. 当前训练还是按普通 SceneCo / TrajCo adapter 训练流程走。
```

所以本 README 说明如何新增一条专门训练线：

```text
Stage2RootGuidedSceneCo Training
```

---

## 1. 训练目标

训练一个模型：

```text
输入：
    external_root / guided_root_5d
    scene voxel
    text
    noisy body
    timestep

输出：
    body motion
```

最终输出：

```text
final_motion = [external_root | predicted_body]
```

其中：

```text
external_root = guided_root_5d
              = smooth_root_pos(3) + heading(cos, sin)
```

---

## 2. 和原先 F / H 实验的区别

原来的 F / H 是：

```text
F:
    Root Stage: TrajCo
    Body Stage: SceneCo

H:
    Root Stage: TrajCo
    Body Stage: TrajCo + SceneCo
```

现在新的训练是：

```text
Stage2RootGuidedSceneCo:
    Root Stage: 不预测 root，直接使用 external_root
    Body Stage: SceneCo body-only
    TrajCo: 关闭
```

也就是：

```text
不再训练 TrajEncoder。
不再训练 TrajCo。
不让 SceneCo 进入 Root Stage。
只训练 Stage2 的 SceneCo body adapter，让 body 适配 guided root 和场景。
```

---

## 3. 训练数据

每个样本需要以下字段：

```text
motion_features:    (T, 273)  原始 GT motion feature
motion_mask:        (T,)      有效帧 mask
voxel_grid:         (64,64,64)
text / text_feat:   文本或预计算文本特征
scene_name:         场景名
external_root:      (T, 5)    guided_root_5d，必须是 normalized Kimodo feature space
```

### 3.1 external_root 来源

建议准备三类 root：

| 类型 | 说明 | 用途 |
|---|---|---|
| `gt_root` | 从 GT motion 的 root_slice 直接切出 | 训练初期稳定 body completion |
| `path_guided_root` | 只用 path guidance 生成 | 模拟只看路径的 root |
| `path_scene_guided_root` | path + scene guidance 生成 | 最终推理时使用 |

推荐混合比例：

```yaml
root_condition_mix:
  gt_root: 0.3
  path_guided_root: 0.3
  path_scene_guided_root: 0.4
```

如果刚开始训练不稳定，先用：

```yaml
root_condition_mix:
  gt_root: 0.5
  path_scene_guided_root: 0.5
```

---

## 4. 预生成 guided root

### 4.1 Path-only guided root

使用 1 号物理 GPU：

```bash
cd /path/to/SceneCoHSI

export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models

python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/guided_roots_train/path_only \
  --split train \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --gpu 0
```

### 4.2 Path+Scene guided root

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/guided_roots_train/path_scene \
  --split train \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --scene_guidance \
  --gpu 0
```

### 4.3 验证集 guided root

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/guided_roots_val/path_scene \
  --split val \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --scene_guidance \
  --gpu 0
```

---

## 5. guided root 文件格式

建议每个样本保存为：

```text
outputs/guided_roots_train/path_scene/
├── seg_000001.npz
├── seg_000002.npz
└── ...
```

每个 `.npz` 包含：

```python
{
    "guided_root_5d_norm":  (T, 5),  # normalized Kimodo root feature
    "guided_root_5d_meter": (T, 5),  # meter/canonical root for visualization
    "target_path_xz":      (T, 2),
    "scene_name":          str,
    "text":                str,
    "source_id":           str,
}
```

训练时必须使用：

```text
guided_root_5d_norm
```

不要把 meter-space root 直接喂给模型。

---

## 6. 需要新增配置文件

新增：

```text
configs/stage2_root_guided_sceneco.yaml
```

建议内容：

```yaml
experiment:
  name: stage2_root_guided_sceneco
  description: Train SceneCo body-only with external guided root for Stage2.

model:
  pretrained_model: models/Kimodo-SMPLX-RP-v1
  baseline_model: null

data:
  data_root: LINGO/dataset
  cache_dir: lingo_smplx_cache
  scene_dir: LINGO/dataset/dataset/Scene
  max_frames: 196
  train_ratio: 0.9

external_root:
  enabled: true
  use_external_root: true
  fix_root_each_step: true
  use_initial_pose: false
  root_format: normalized_5d
  path_guided_root_dir: outputs/guided_roots_train/path_only
  path_scene_guided_root_dir: outputs/guided_roots_train/path_scene
  val_root_dir: outputs/guided_roots_val/path_scene

root_condition_mix:
  gt_root: 0.3
  path_guided_root: 0.3
  path_scene_guided_root: 0.4

sceneco:
  enabled: true
  use_in_root_model: false
  use_in_body_model: true
  sceneco_dropout: 0.1
  use_dual_vit: true
  root_voxel_mode: full

trajco:
  enabled: false
  use_trajco: false
  use_trajco_root: false
  use_trajco_body: false

training:
  batch_size: 4
  accum_steps: 1
  num_epochs: 400
  lr: 1.0e-4
  weight_decay: 0.01
  max_grad_norm: 1.0
  prior_weight: 0.0
  scene_dropout: 0.1
  num_workers: 4
  device: cuda
  output_dir: outputs/stage2_root_guided_sceneco

freeze:
  freeze_pretrained: true
  train_scene_encoder: true
  train_body_sceneco: true
  train_root_model: false
  train_body_backbone: false
  train_text_encoder: false

loss:
  body_mse: 1.0
  body_velocity: 0.1
  foot_slide: 0.05
  scene_collision: 0.0
  root_loss: 0.0
  body_only_loss: true
```

说明：

```text
第一版先把 scene_collision 设为 0.0。
等 body 能稳定生成后，再加 scene_collision。
```

---

## 7. 需要修改 Dataset

文件：

```text
kimodo_sceneco/train/dataset.py
```

目标：让 dataset 返回 `external_root`。

### 7.1 新增 Dataset 参数

在 `LINGOSceneMotionDataset.__init__()` 中增加：

```python
external_root_enabled: bool = False
path_guided_root_dir: Optional[str] = None
path_scene_guided_root_dir: Optional[str] = None
root_condition_mix: Optional[dict] = None
```

### 7.2 读取 external root

在 `__getitem__()` 中：

```python
sample = {
    "motion_features": motion_features,
    "motion_mask": motion_mask,
    "voxel_grid": voxel_grid,
    "text": text,
    "scene_name": scene_name,
    "source_id": source_id,
}
```

新增：

```python
if self.external_root_enabled:
    external_root, root_source = self._load_external_root(
        source_id=source_id,
        motion_features=motion_features,
    )
    sample["external_root"] = external_root.astype(np.float32)
    sample["external_root_source"] = root_source
```

### 7.3 root 混合逻辑

```python
def _load_external_root(self, source_id, motion_features):
    r = np.random.rand()

    p_gt = self.root_condition_mix.get("gt_root", 0.0)
    p_path = self.root_condition_mix.get("path_guided_root", 0.0)
    p_scene = self.root_condition_mix.get("path_scene_guided_root", 0.0)

    if r < p_gt:
        root_slice = slice(0, 5)
        return motion_features[:, root_slice], "gt_root"

    elif r < p_gt + p_path:
        path = Path(self.path_guided_root_dir) / f"{source_id}.npz"
        data = np.load(path)
        return data["guided_root_5d_norm"], "path_guided_root"

    else:
        path = Path(self.path_scene_guided_root_dir) / f"{source_id}.npz"
        data = np.load(path)
        return data["guided_root_5d_norm"], "path_scene_guided_root"
```

如果你的缓存文件不是 `source_id` 命名，需要建立 mapping：

```text
cache sample id → guided root npz file
```

---

## 8. 需要修改 collate_fn

文件：

```text
kimodo_sceneco/train/dataset.py
```

在 `collate_fn()` 中增加：

```python
if "external_root" in batch[0]:
    external_root = torch.stack([
        torch.from_numpy(item["external_root"]) for item in batch
    ], dim=0)
    collated["external_root"] = external_root

if "external_root_source" in batch[0]:
    collated["external_root_source"] = [
        item["external_root_source"] for item in batch
    ]
```

输出：

```text
external_root: (B, T, 5)
```

---

## 9. 需要修改 train.py 参数

文件：

```text
kimodo_sceneco/train/train.py
```

### 9.1 新增 CLI 参数

在 `parse_args()` 中增加：

```python
parser.add_argument("--use_external_root_training", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--path_guided_root_dir", type=str, default=None)
parser.add_argument("--path_scene_guided_root_dir", type=str, default=None)
parser.add_argument("--val_root_dir", type=str, default=None)

parser.add_argument("--root_mix_gt", type=float, default=0.3)
parser.add_argument("--root_mix_path", type=float, default=0.3)
parser.add_argument("--root_mix_scene", type=float, default=0.4)

parser.add_argument("--body_only_loss", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--use_external_root", type=lambda x: x.lower() == "true", default=False)
```

---

## 10. 修改 `_build_dataset()`

文件：

```text
kimodo_sceneco/train/train.py
```

当前 trainer 已经使用：

```python
LINGOSceneMotionDataset(...)
```

需要把 external root 参数传进去：

```python
root_condition_mix = {
    "gt_root": self.args.root_mix_gt,
    "path_guided_root": self.args.root_mix_path,
    "path_scene_guided_root": self.args.root_mix_scene,
}

ds_kwargs.update({
    "external_root_enabled": self.args.use_external_root_training,
    "path_guided_root_dir": self.args.path_guided_root_dir,
    "path_scene_guided_root_dir": self.args.path_scene_guided_root_dir,
    "root_condition_mix": root_condition_mix,
})
```

验证集建议固定使用 `path_scene_guided_root`：

```python
val_ds_kwargs = dict(ds_kwargs)
val_ds_kwargs["root_condition_mix"] = {
    "gt_root": 0.0,
    "path_guided_root": 0.0,
    "path_scene_guided_root": 1.0,
}
val_ds_kwargs["path_scene_guided_root_dir"] = self.args.val_root_dir
```

---

## 11. 修改 `_prepare_batch()`

文件：

```text
kimodo_sceneco/train/train.py
```

当前 `_prepare_batch()` 会返回：

```python
{
    "x_start": motion,
    "x_pad_mask": mask,
    "scene_feat_root": scene_feat_root,
    "scene_mask_root": scene_mask_root,
    "scene_feat_body": scene_feat_body,
    "scene_mask_body": scene_mask_body,
    ...
}
```

新增：

```python
external_root = None

if self.args.use_external_root_training:
    external_root = batch["external_root"].to(self.device)
```

然后 model_kwargs 里加入：

```python
"use_external_root": self.args.use_external_root,
"external_root": external_root,
```

---

## 12. 只开 Body SceneCo，关闭 Root SceneCo 和 TrajCo

训练命令中必须设置：

```bash
--use_in_root_model false
--use_in_body_model true
--use_trajco false
--use_trajco_root false
--use_trajco_body false
```

意义：

```text
Root Stage:
    不使用 SceneCo
    不使用 TrajCo
    不预测 root

Body Stage:
    使用 SceneCo
    使用 external_root 的 local root condition
```

---

## 13. 修改 training loss：传 external_root

文件：

```text
kimodo_sceneco/train/train.py
```

当前 `SceneCoDiffusionLoss.training_losses()` 调用 model 时没有传：

```text
external_root
use_external_root
```

需要改为：

```python
pred_x0 = model(
    cfg_weight,
    x_t,
    model_kwargs["x_pad_mask"],
    model_kwargs["text_feat"],
    model_kwargs["text_pad_mask"],
    t,
    first_heading_angle=model_kwargs.get("first_heading_angle"),
    motion_mask=model_kwargs.get("motion_mask"),
    observed_motion=model_kwargs.get("observed_motion"),
    scene_feat_root=model_kwargs.get("scene_feat_root"),
    scene_mask_root=model_kwargs.get("scene_mask_root"),
    scene_feat_body=model_kwargs.get("scene_feat_body"),
    scene_mask_body=model_kwargs.get("scene_mask_body"),
    traj_feats=model_kwargs.get("traj_feats"),
    traj_mask=model_kwargs.get("traj_mask"),
    external_root=model_kwargs.get("external_root"),
    use_external_root=model_kwargs.get("use_external_root", False),
    cfg_type=model_kwargs.get("cfg_type", "nocfg"),
)
```

prior branch 也要传：

```python
pred_x0_null = model(
    cfg_weight,
    x_t,
    model_kwargs["x_pad_mask"],
    model_kwargs["text_feat"],
    model_kwargs["text_pad_mask"],
    t,
    first_heading_angle=model_kwargs.get("first_heading_angle"),
    motion_mask=model_kwargs.get("motion_mask"),
    observed_motion=model_kwargs.get("observed_motion"),
    scene_feat_root=None,
    scene_mask_root=None,
    scene_feat_body=None,
    scene_mask_body=None,
    traj_feats=None,
    traj_mask=None,
    external_root=model_kwargs.get("external_root"),
    use_external_root=model_kwargs.get("use_external_root", False),
    cfg_type="nocfg",
)
```

如果第一版不做 prior preservation，建议：

```bash
--prior_weight 0.0
```

---

## 14. 修改 loss_mask：只监督 body

目标：

```text
root 是 external_root，不是模型要学习预测的东西。
训练时只应该监督 body_slice。
```

当前 `training_losses()` 已支持 `loss_mask`。只需要在 `_prepare_batch()` 中构造 body-only loss mask。

```python
loss_mask = None

if self.args.body_only_loss:
    D = motion.shape[-1]
    body_slice = self.model.motion_rep.body_slice

    loss_mask = torch.zeros(D, device=self.device)
    loss_mask[body_slice] = 1.0
```

返回：

```python
"loss_mask": loss_mask
```

不要再用 root-only loss：

```python
loss_mask[root_slice] = 1.0
```

这次应该是：

```python
loss_mask[body_slice] = 1.0
```

---

## 15. 可选：训练时固定 x_t 的 root

为了让训练和推理一致，可以在 loss 中加：

```python
if model_kwargs.get("use_external_root", False):
    root_slice = model.motion_rep.root_slice
    x_t[..., root_slice] = model_kwargs["external_root"]
```

位置：

```text
SceneCoDiffusionLoss.training_losses()
q_sample 之后
model forward 之前
```

推荐加上：

```python
if model_kwargs.get("use_external_root", False) and model_kwargs.get("external_root") is not None:
    root_slice = model.motion_rep.root_slice if hasattr(model, "motion_rep") else None
    if root_slice is not None:
        x_t[..., root_slice] = model_kwargs["external_root"]
```

但注意：如果 `model` 是 `ClassifierFreeGuidedModel` 包装，要用：

```python
root_slice = model.model.motion_rep.root_slice
```

或在 trainer 里直接把 `root_slice` 传进 `model_kwargs`。

---

## 16. 冻结策略

第一版建议冻结 Kimodo backbone，只训练：

```text
1. scene_encoder / VoxelViT
2. body SceneCo cross-attention adapter
3. body SceneCo projection
4. body SceneCo gate / alpha
```

冻结：

```text
1. root_model
2. body_model backbone
3. text_encoder
4. TrajCo
```

当前 train.py 已有 `freeze_pretrained`，会调用：

```text
model.freeze_pretrained()
```

但你要确认它不会训练 root SceneCo，因为这次需要：

```text
use_in_root_model = false
use_in_body_model = true
```

如果需要更严格，新增函数：

```python
def freeze_for_stage2_root_guided_sceneco(self):
    for p in self.parameters():
        p.requires_grad = False

    # scene encoder
    for p in self.scene_encoder.parameters():
        p.requires_grad = True

    # only body SceneCo adapters
    for name, p in self.named_parameters():
        if "body" in name and ("sceneco" in name.lower() or "scene" in name.lower() or "alpha" in name.lower()):
            p.requires_grad = True
```

---

## 17. 新增训练配置脚本

建议新增脚本：

```text
scripts/train_stage2_root_guided_sceneco.sh
```

内容：

```bash
#!/usr/bin/env bash
set -e

cd /path/to/SceneCoHSI

export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models

python -m kimodo_sceneco.train.train \
  --data_root LINGO/dataset \
  --cache_dir lingo_smplx_cache \
  --pretrained_model models/Kimodo-SMPLX-RP-v1 \
  --output_dir outputs/stage2_root_guided_sceneco \
  \
  --use_external_root_training true \
  --use_external_root true \
  --path_guided_root_dir outputs/guided_roots_train/path_only \
  --path_scene_guided_root_dir outputs/guided_roots_train/path_scene \
  --val_root_dir outputs/guided_roots_val/path_scene \
  --root_mix_gt 0.3 \
  --root_mix_path 0.3 \
  --root_mix_scene 0.4 \
  --body_only_loss true \
  \
  --use_in_root_model false \
  --use_in_body_model true \
  --use_trajco false \
  --use_trajco_root false \
  --use_trajco_body false \
  \
  --batch_size 4 \
  --accum_steps 1 \
  --num_epochs 400 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --prior_weight 0.0 \
  --scene_dropout 0.1 \
  --num_workers 4 \
  --device cuda \
  2>&1 | tee outputs/stage2_root_guided_sceneco/train.log
```

注意：

```text
CUDA_VISIBLE_DEVICES=1 后，代码内部使用 cuda:0。
```

---

## 18. 执行训练

```bash
chmod +x scripts/train_stage2_root_guided_sceneco.sh
bash scripts/train_stage2_root_guided_sceneco.sh
```

监控：

```bash
tail -f outputs/stage2_root_guided_sceneco/train.log
```

TensorBoard：

```bash
tensorboard --logdir outputs/stage2_root_guided_sceneco/logs --port 6006
```

---

## 19. 验证生成

训练完成后，使用训练好的 checkpoint：

```bash
export CUDA_VISIBLE_DEVICES=1

python scripts/generate_body_from_root.py \
  --root_dir outputs/guided_roots_val/path_scene \
  --checkpoint outputs/stage2_root_guided_sceneco/checkpoints/best_checkpoint.pt \
  --output_dir outputs/stage2_root_guided_sceneco/val_gen \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0
```

必须检查日志：

```text
Root fix max_error < 1e-5
```

如果大于 `1e-5`，说明 fixed-root 逻辑没有生效。

---

## 20. 评估

### 20.1 Path metrics

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/stage2_root_guided_sceneco/val_gen \
  --output_csv outputs/stage2_root_guided_sceneco/path_metrics.csv \
  --method stage2_root_guided_sceneco
```

### 20.2 SceneAdapt-style proxy metrics

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/stage2_root_guided_sceneco/val_gen \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/stage2_root_guided_sceneco/scene_metrics.csv \
  --method stage2_root_guided_sceneco
```

### 20.3 对比表

至少对比以下方法：

| 方法 | Root 来源 | Stage2 | SceneCo |
|---|---|---|---|
| PathOnly+Stage2-NoScene | Path-guided root | 原始 body | 无 |
| PathScene+Stage2-NoScene | Path+Scene root | 原始 body | 无 |
| PathOnly+Stage2-SceneCo | Path-guided root | 训练后 body | 有 |
| PathScene+Stage2-SceneCo | Path+Scene root | 训练后 body | 有 |

要证明：

```text
1. PathScene root 相比 PathOnly root，NonWalkableRootRate 更低。
2. Stage2 SceneCo 相比 NoScene body，PenetrationRate / CollisionFrameRate 更低。
3. FootSlide / RootJerk 不爆炸。
```

---

## 21. 训练成功标准

### 21.1 训练 loss

训练 loss 应该稳定下降。

重点观察：

```text
train/loss
val/loss
grad_norm
alpha / gate
```

如果 loss 上千，不一定异常，因为你的 motion feature 是 273 维，且 MSE 是按所有 body feature 汇总。建议同时看：

```text
per-element loss = loss / body_dim
```

---

### 21.2 root 固定

必须满足：

```text
max_abs(final_root - external_root) < 1e-5
```

---

### 21.3 场景指标

训练后应当相对 no-scene Stage2 改善：

```text
CollisionFrameRate ↓
PenetrationRate ↓
PenetrationMean ↓
PenetrationMax ↓
NonWalkableRootRate 不应恶化
```

---

### 21.4 动作质量

不能明显恶化：

```text
FootSlide 不应爆炸
VelErr 不应爆炸
Root/body 不应明显分离
人物不应扭曲
```

---

## 22. 常见问题

### 问题 1：训练时 body 和 root 对不上

检查：

```text
external_root 是否是 normalized 5D
use_external_root 是否真的传入 denoiser
loss 是否只算 body_slice
x_t root 是否在训练时被 external_root 替换
```

---

### 问题 2：SceneCo 训练后人物扭曲

降低 SceneCo 强度：

```yaml
sceneco_dropout: 0.2
gate_init: -7
```

或者只在 body 后几层插入 SceneCo：

```yaml
body_sceneco_layers: [8, 10, 12, 14]
```

---

### 问题 3：场景指标没有提升

检查：

```text
scene voxel 是否加载正确
voxel_size 是否是 0.1
scene axis 是否 ZYX → XYZ
root 是否本身穿过障碍
```

注意：

```text
如果 root 本身穿墙，Stage2 SceneCo 无法修 root-level collision。
```

---

### 问题 4：训练太慢

先减少：

```text
num_epochs: 50
num_samples: 小规模
batch_size: 2
```

确认能跑通后再扩大。

---

## 23. 最短总结

本训练方案是：

```text
1. 先用 Root Classifier Guidance 生成 guided_root_5d。
2. 训练时把 guided_root_5d 当作 external_root。
3. 跳过 Root Stage 的 root prediction。
4. Body Denoiser 使用 global_root_to_local_root(external_root) 作为条件。
5. 只开 SceneCo body-only。
6. loss 只监督 body_slice。
7. 冻结 Kimodo backbone，只训练 SceneCo body adapter / scene encoder。
```

最终目标：

```text
让 Stage2 在已知 guided root 的情况下，
利用场景信息生成更少碰撞、更少穿模、更自然的 body motion。
```
