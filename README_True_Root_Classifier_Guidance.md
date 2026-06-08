# README：真正的 Root Classifier Guidance 注入 Root 的完整方案

> 目标：把当前“手写 path / smooth / heading / scene loss”的 root guidance，升级为**真正的 trained classifier guidance**。  
> 真正的 classifier guidance 需要训练一个 `RootPathSceneClassifier / RootSceneCritic`，让它判断候选 root 是否满足目标路径、速度平滑、heading 合理、场景可行；采样时用 classifier 的梯度引导 Kimodo Root Stage 生成 `guided_root_5d`。  
>
> 当前已有的手写 loss guidance 不删除，但必须改名为：
>
> ```text
> Root Energy Guidance / Analytical Root Guidance
> ```
>
> 新增的严格版本叫：
>
> ```text
> Root Classifier Guidance
> ```

---

## 0. 为什么要改

之前的实现流程是：

```text
target path
    ↓
L_path + L_goal + L_speed + L_smooth + L_heading + L_scene
    ↓
对 diffusion 当前状态 x_t 求梯度
    ↓
修改 root_slice
```

这不是严格意义上的 classifier guidance，因为它没有训练 classifier。  
它更准确叫：

```text
Energy Guidance / Loss Guidance / Analytical Guidance
```

真正的 classifier guidance 应该是：

```text
target path / waypoint / planner path
        ↓
trained RootPathSceneClassifier
        ↓
classifier 输出 valid score
        ↓
L_cls = -log p(valid | root, path, scene)
        ↓
对 x_t 求梯度
        ↓
只更新 root_slice
        ↓
guided_root_5d
```

---

## 1. 最终系统结构

完整 pipeline：

```text
target path / waypoint / planner path
        ↓
RootPathSceneClassifier
        ↑
Kimodo denoiser 预测的 pred_x0[root_slice]
        ↓
classifier score: p(valid | root, path, scene)
        ↓
loss_cls = BCEWithLogits(logit, valid=1)
        ↓
∂loss_cls / ∂x_t
        ↓
只保留 root_slice 梯度
        ↓
x_t = x_t - classifier_guidance_scale * grad
        ↓
guided_root_5d
        ↓
fixed-root Stage2
        ↓
Body SceneCo
        ↓
final_motion = [guided_root_5d | generated_body]
```

---

## 2. 方法命名

以后实验和代码建议使用：

| 名称 | 含义 |
|---|---|
| `NoGuidance` | 原始 Kimodo，不加 root 引导 |
| `EnergyGuidance` | 当前手写 loss guidance |
| `ClassifierGuidance` | 新增训练 classifier 后的 guidance |
| `HybridGuidance` | 手写 loss + classifier score |
| `ClassifierGuidance + Stage2SceneCo` | 最终完整方法之一 |
| `HybridGuidance + Stage2SceneCo` | 稳定增强版 |

不要再把 `L_path + L_smooth` 版本叫 classifier guidance。

---

## 3. 需要新增的文件

```text
kimodo_sceneco/
├── critic/
│   ├── __init__.py
│   ├── root_path_scene_classifier.py
│   ├── root_classifier_features.py
│   ├── root_classifier_dataset.py
│   └── train_root_classifier.py
│
scripts/
├── build_root_classifier_dataset.py
├── generate_root_classifier_guidance.py
└── compare_guidance_methods.py
│
configs/
├── root_classifier.yaml
├── root_classifier_guidance.yaml
└── root_hybrid_guidance.yaml
```

已有的：

```text
kimodo_sceneco/guidance/root_guidance.py
```

保留，但在报告里称为 `EnergyGuidance baseline`。

---

## 4. RootPathSceneClassifier 的输入输出

### 4.1 输入

```text
root_5d:        (B, T, 5)
target_path_xz: (B, T, 2)
scene_sdf:      optional，2D SDF / occupancy
pad_mask:       (B, T)
```

其中：

```text
root_5d = [x, y, z, heading_cos, heading_sin]
```

注意：

```text
classifier 输入建议使用 meter / canonical coordinate。
不要直接用 normalized Kimodo feature 训练 classifier。
```

采样时如果 `pred_x0[root_slice]` 是 normalized，需要：

```text
pred_x0[root_slice]
    ↓
denormalize_root_5d / motion_rep.unnormalize
    ↓
root_5d_meter
    ↓
classifier
```

### 4.2 输出

```text
logit_valid: (B, 1)
```

含义：

```text
该 root 是否 valid：
    路径一致
    终点正确
    速度平滑
    heading 合理
    场景可行
```

训练标签：

```text
valid root   → label = 1
invalid root → label = 0
```

---

## 5. 每帧特征构造

新增文件：

```text
kimodo_sceneco/critic/root_classifier_features.py
```

建议构造这些特征：

```text
root_xz
target_path_xz
root_xz - target_path_xz
distance_to_target_path
root_velocity
target_velocity
root_speed
target_speed
heading_cos_sin
path_direction_cos_sin
heading_path_angle_error
scene_sdf_value
```

示例代码：

```python
import torch


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def build_root_classifier_features(
    root_5d: torch.Tensor,
    target_path_xz: torch.Tensor,
    scene_sdf=None,
    sample_sdf_fn=None,
) -> torch.Tensor:
    """
    root_5d:
        (B, T, 5), meter/canonical:
        [x, y, z, heading_cos, heading_sin]

    target_path_xz:
        (B, T, 2)

    return:
        frame_feat: (B, T, C)
    """
    pos = root_5d[..., 0:3]
    heading = root_5d[..., 3:5]

    root_xz = pos[..., [0, 2]]
    target_xz = target_path_xz

    root_vel = root_xz[:, 1:] - root_xz[:, :-1]
    root_vel = torch.cat([root_vel, root_vel[:, -1:]], dim=1)

    target_vel = target_xz[:, 1:] - target_xz[:, :-1]
    target_vel = torch.cat([target_vel, target_vel[:, -1:]], dim=1)

    root_speed = root_vel.norm(dim=-1, keepdim=True)
    target_speed = target_vel.norm(dim=-1, keepdim=True)

    path_theta = torch.atan2(target_vel[..., 1], target_vel[..., 0])
    path_dir = torch.stack([torch.cos(path_theta), torch.sin(path_theta)], dim=-1)

    heading_theta = torch.atan2(heading[..., 1], heading[..., 0])
    heading_path_error = angle_diff(heading_theta, path_theta).unsqueeze(-1)

    root_minus_target = root_xz - target_xz
    dist_to_target = root_minus_target.norm(dim=-1, keepdim=True)

    if scene_sdf is not None and sample_sdf_fn is not None:
        sdf_value = sample_sdf_fn(scene_sdf, pos).unsqueeze(-1)
    else:
        sdf_value = torch.zeros_like(dist_to_target)

    frame_feat = torch.cat(
        [
            root_xz,
            target_xz,
            root_minus_target,
            dist_to_target,
            root_vel,
            target_vel,
            root_speed,
            target_speed,
            heading,
            path_dir,
            heading_path_error,
            sdf_value,
        ],
        dim=-1,
    )

    return frame_feat
```

---

## 6. Classifier 网络结构

新增文件：

```text
kimodo_sceneco/critic/root_path_scene_classifier.py
```

第一版用 Transformer Encoder：

```python
import torch
import torch.nn as nn


class RootPathSceneClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frame_feat, pad_mask=None):
        """
        frame_feat:
            (B, T, C)

        pad_mask:
            (B, T), True = valid frame
        """
        h = self.input_proj(frame_feat)

        if pad_mask is not None:
            src_key_padding_mask = ~pad_mask.bool()
        else:
            src_key_padding_mask = None

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        if pad_mask is not None:
            mask = pad_mask.float().unsqueeze(-1)
            h = h * mask
            pooled = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = h.mean(dim=1)

        logit = self.head(pooled)
        return logit
```

---

## 7. Classifier 训练数据

### 7.1 正样本 label=1

第一版：

```text
GT root + GT path + correct scene
```

后续可以加入：

```text
Path-guided root 中指标好的样本
Path+Scene-guided root 中指标好的样本
```

好样本可用指标筛选：

```text
PathADE < threshold
RootJerk < threshold
NonWalkableRootRate == 0
```

### 7.2 负样本 label=0

必须构造负样本。建议：

| 负样本类型 | 构造方式 | 目的 |
|---|---|---|
| `shift` | root 整体平移 0.5m～1.5m | 学 path mismatch |
| `wrong_goal` | 后半段漂到错误终点 | 学 FDE |
| `jitter` | root_xz 加高频噪声 | 学 root smoothness |
| `wrong_heading` | heading 随机旋转 | 学 heading-path consistency |
| `reverse_heading` | heading 反向 | 学倒走错误 |
| `path_shuffle` | root 和别的样本 path 配对 | 学 root/path 不匹配 |
| `scene_shift` | root 平移到障碍物 | 学 scene feasibility |
| `scene_shuffle` | root/path 正确，scene 换掉 | 学 scene 条件 |

示例：

```python
import torch
import random


def make_negative_root(root_5d: torch.Tensor, mode: str) -> torch.Tensor:
    root = root_5d.clone()
    B, T, _ = root.shape

    if mode == "shift":
        offset = torch.randn(B, 1, 2, device=root.device) * 0.8
        root[..., [0, 2]] += offset

    elif mode == "wrong_goal":
        drift = torch.linspace(0, 1, T, device=root.device).view(1, T, 1)
        wrong_offset = torch.randn(B, 1, 2, device=root.device) * 1.2
        root[..., [0, 2]] += drift * wrong_offset

    elif mode == "jitter":
        noise = torch.randn(B, T, 2, device=root.device) * 0.15
        root[..., [0, 2]] += noise

    elif mode == "wrong_heading":
        theta = torch.rand(B, T, device=root.device) * 2.0 * torch.pi
        root[..., 3] = torch.cos(theta)
        root[..., 4] = torch.sin(theta)

    elif mode == "reverse_heading":
        root[..., 3:5] = -root[..., 3:5]

    else:
        raise ValueError(f"Unknown negative mode: {mode}")

    return root


def sample_negative_mode():
    return random.choice([
        "shift",
        "wrong_goal",
        "jitter",
        "wrong_heading",
        "reverse_heading",
    ])
```

---

## 8. Dataset

新增：

```text
kimodo_sceneco/critic/root_classifier_dataset.py
```

每个 item 返回：

```python
{
    "root_5d": root_5d_meter,             # (T, 5)
    "target_path_xz": target_path_xz,     # (T, 2)
    "scene_sdf": scene_sdf,               # optional
    "pad_mask": pad_mask,                 # (T,)
    "label": label,                       # 0 / 1
    "negative_mode": mode_or_none,
}
```

### 正样本

```python
root_5d = gt_root_5d_meter
target_path_xz = gt_root_5d_meter[:, [0, 2]]
label = 1
```

### 负样本

```python
root_5d = make_negative_root(gt_root_5d_meter, mode)
target_path_xz = gt_root_5d_meter[:, [0, 2]]
label = 0
```

### path_shuffle 负样本

```python
root_5d = gt_root_5d_meter_of_sample_i
target_path_xz = gt_path_of_sample_j
label = 0
```

---

## 9. 训练 classifier

新增：

```text
kimodo_sceneco/critic/train_root_classifier.py
```

核心逻辑：

```python
import torch
import torch.nn.functional as F

from kimodo_sceneco.critic.root_path_scene_classifier import RootPathSceneClassifier
from kimodo_sceneco.critic.root_classifier_features import build_root_classifier_features


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        root_5d = batch["root_5d"].to(device)
        target_path_xz = batch["target_path_xz"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        label = batch["label"].float().to(device)

        scene_sdf = batch.get("scene_sdf", None)
        if scene_sdf is not None:
            scene_sdf = scene_sdf.to(device)

        frame_feat = build_root_classifier_features(
            root_5d=root_5d,
            target_path_xz=target_path_xz,
            scene_sdf=scene_sdf,
            sample_sdf_fn=None,
        )

        logit = model(frame_feat, pad_mask=pad_mask).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            prob = torch.sigmoid(logit)
            pred = (prob > 0.5).float()
            correct = (pred == label).float().sum().item()

        total_loss += loss.item() * label.numel()
        total_correct += correct
        total_count += label.numel()

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }
```

---

## 10. classifier 配置

新增：

```text
configs/root_classifier.yaml
```

```yaml
experiment:
  name: root_path_scene_classifier
  output_dir: outputs/root_path_scene_classifier

data:
  cache_dir: lingo_smplx_cache
  scene_dir: LINGO/dataset/dataset/Scene
  max_frames: 196
  use_scene_sdf: false

negative_sampling:
  positive_ratio: 0.5
  modes:
    - shift
    - wrong_goal
    - jitter
    - wrong_heading
    - reverse_heading
    - path_shuffle
  shift_std: 0.8
  jitter_std: 0.15

model:
  input_dim: 20
  hidden_dim: 256
  num_layers: 4
  num_heads: 4
  dropout: 0.1

training:
  batch_size: 64
  num_epochs: 100
  lr: 1.0e-4
  weight_decay: 1.0e-4
  max_grad_norm: 1.0
  num_workers: 4
  gpu: 0
```

---

## 11. 训练 classifier 命令

使用物理 1 号 GPU：

```bash
cd /path/to/SceneCoHSI

export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models

python kimodo_sceneco/critic/train_root_classifier.py \
  --config configs/root_classifier.yaml \
  --output_dir outputs/root_path_scene_classifier \
  --batch_size 64 \
  --num_epochs 100 \
  --lr 1e-4 \
  --gpu 0 \
  2>&1 | tee outputs/root_path_scene_classifier/train.log
```

输出：

```text
outputs/root_path_scene_classifier/best.pt
outputs/root_path_scene_classifier/latest.pt
```

成功标准：

```text
val_acc > 0.85
positive_score_mean > negative_score_mean
AUC > 0.85
```

并按 negative mode 分开统计 accuracy。

---

## 12. 采样时使用 classifier guidance

在 `kimodo_sceneco/model/kimodo_model.py` 中新增：

```text
denoising_step_with_root_classifier_guidance()
```

核心伪代码：

```python
def denoising_step_with_root_classifier_guidance(
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
    root_classifier,
    target_path_xz,
    scene_sdf=None,
    classifier_guidance_scale=0.05,
    max_grad_norm=1.0,
    scene_feat_root=None,
    scene_mask_root=None,
    scene_feat_body=None,
    scene_mask_body=None,
    cfg_type=None,
):
    from kimodo_sceneco.critic.root_classifier_features import build_root_classifier_features
    from kimodo_sceneco.guidance.root_guidance import denormalize_root_5d

    use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps[0])
    self.diffusion.calc_diffusion_vars(use_timesteps)
    t_map = map_tensor[t]

    x = motion.detach().requires_grad_(True)

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
        cfg_type=cfg_type,
    )

    root_norm = pred_x0[..., self.motion_rep.root_slice]

    root_meter = denormalize_root_5d(
        root_norm,
        motion_rep=self.motion_rep,
        root_slice=self.motion_rep.root_slice,
    )

    frame_feat = build_root_classifier_features(
        root_5d=root_meter,
        target_path_xz=target_path_xz,
        scene_sdf=scene_sdf,
        sample_sdf_fn=None,
    )

    logit = root_classifier(frame_feat, pad_mask=pad_mask)

    label_valid = torch.ones_like(logit)
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        logit,
        label_valid,
    )

    grad = torch.autograd.grad(loss_cls, x)[0]

    # 只引导 root
    root_grad = torch.zeros_like(grad)
    root_grad[..., self.motion_rep.root_slice] = grad[..., self.motion_rep.root_slice]
    grad = root_grad

    grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    grad = grad * (max_grad_norm / grad_norm).clamp(max=1.0)

    x_guided = x - classifier_guidance_scale * grad
    x_guided = x_guided.detach()

    with torch.inference_mode():
        pred_clean = self.predict_x0(
            motion=x_guided,
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
            cfg_type=cfg_type,
        )

        x_tm1 = self.sampler(use_timesteps, x_guided, pred_clean, t)

    return x_tm1, {
        "loss_cls": loss_cls.detach(),
        "score_valid": torch.sigmoid(logit).mean().detach(),
    }
```

---

## 13. Hybrid Guidance

为了稳定，可以使用：

```text
HybridGuidance = ClassifierGuidance + EnergyGuidance
```

总 loss：

```text
L_total = λ_cls · L_classifier + λ_energy · L_energy
```

实现：

```python
energy_losses = compute_root_guidance_loss(...)
loss_energy = energy_losses["total"]

logit = root_classifier(...)
loss_cls = BCEWithLogits(logit, valid_label)

loss_total = w_classifier * loss_cls + w_energy * loss_energy

grad = torch.autograd.grad(loss_total, x)[0]
```

---

## 14. classifier guidance 配置

新增：

```text
configs/root_classifier_guidance.yaml
```

```yaml
experiment:
  name: root_classifier_guidance
  output_dir: outputs/root_classifier_guidance

model:
  checkpoint: models/Kimodo-SMPLX-RP-v1

root_classifier:
  checkpoint: outputs/root_path_scene_classifier/best.pt
  input_dim: 20
  hidden_dim: 256
  num_layers: 4
  num_heads: 4

classifier_guidance:
  enabled: true
  scale: 0.05
  max_grad_norm: 1.0
  start_step: 0
  end_step: 40
  use_scene: false

hybrid:
  enabled: false
  w_classifier: 1.0
  w_energy: 0.3

generation:
  num_frames: 196
  num_denoising_steps: 50
  cfg_weight: [2.0, 2.0]
  gpu: 0
```

---

## 15. 生成 classifier-guided root

新增：

```text
scripts/generate_root_classifier_guidance.py
```

运行：

```bash
cd /path/to/SceneCoHSI

export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models

python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_scene_classifier/best.pt \
  --output_dir outputs/root_classifier_guidance/path_only \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --gpu 0
```

Hybrid 版本：

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_scene_classifier/best.pt \
  --output_dir outputs/root_hybrid_guidance/path_only \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --hybrid \
  --gpu 0
```

---

## 16. Stage2 使用 classifier-guided root

原始 body：

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/root_classifier_guidance/path_only \
  --output_dir outputs/root_classifier_guidance/path_only_body \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0
```

使用训练过的 Stage2 SceneCo：

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/root_classifier_guidance/path_only \
  --checkpoint outputs/stage2_root_guided_sceneco/checkpoints/best_checkpoint.pt \
  --output_dir outputs/root_classifier_guidance/path_only_body_sceneco \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0
```

---

## 17. 评估

Path metrics：

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/root_classifier_guidance/path_only_body \
  --output_csv outputs/root_classifier_guidance/path_metrics.csv \
  --method root_classifier_guidance
```

SceneAdapt-style proxy metrics：

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/root_classifier_guidance/path_only_body \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/root_classifier_guidance/scene_metrics.csv \
  --method root_classifier_guidance
```

---

## 18. 实验矩阵

必须比较：

| 方法 | Root 控制 | 是否训练 classifier | Body |
|---|---|---:|---|
| `NoGuidance` | 无 | 否 | 原始 Body |
| `EnergyGuidance` | 手写 loss | 否 | 原始 Body |
| `ClassifierGuidance` | classifier score | 是 | 原始 Body |
| `HybridGuidance` | 手写 loss + classifier score | 是 | 原始 Body |
| `EnergyGuidance + Stage2SceneCo` | 手写 loss | 否 | SceneCo |
| `ClassifierGuidance + Stage2SceneCo` | classifier score | 是 | SceneCo |
| `HybridGuidance + Stage2SceneCo` | 手写 loss + classifier score | 是 | SceneCo |

指标：

```text
PathADE
PathFDE
HeadingError
SpeedStd
RootJerk
NonWalkableRootRate
CollisionFrameRate
PenetrationRate
PenetrationMean
PenetrationMax
FootSlide
```

---

## 19. 报告写法

推荐表述：

```text
我们将 root 轨迹控制从 hand-crafted energy guidance 扩展为真正的 trained classifier guidance。具体地，我们训练 RootPathSceneClassifier 来判断候选 root 是否满足目标路径、运动平滑性、heading 合理性和场景可行性。采样时，Kimodo denoiser 先预测 pred_x0，然后 classifier 对 pred_x0[root] 打分。我们将 -log p(valid) 作为 classifier loss，并将该 loss 的梯度反传到当前 diffusion state x_t，仅更新 root_slice，从而实现 root 的 classifier-guided control。
```

同时说明：

```text
EnergyGuidance 是 baseline。
ClassifierGuidance 是主方法。
HybridGuidance 是稳定增强版本。
```

---

## 20. 最终结论

真正的 Root Classifier Guidance 需要：

```text
1. 训练 RootPathSceneClassifier。
2. classifier 输入 root、target path、可选 scene。
3. classifier 输出 valid score。
4. 采样时使用 L_cls = BCEWithLogits(logit, 1)。
5. 对 x_t 求梯度。
6. 只更新 root_slice。
7. 得到 classifier-guided root。
8. 将该 root 作为 external_root 给 Stage2。
9. 与 EnergyGuidance / HybridGuidance 比较。
```

当前已有的手写 loss guidance 不废弃，但必须定位为：

```text
EnergyGuidance baseline
```

新增严格方法定位为：

```text
RootClassifierGuidance
```
