# README：True Root Classifier Guidance 继续修改方案（基于当前 GitHub 状态）

> 目标：把当前仓库里的 True Root Classifier Guidance 改到**可训练、可生成、可评估**的状态。  
> 当前 GitHub 已经添加了 `critic/`、`root_classifier_guidance.yaml`、`generate_root_classifier_guidance.py` 等框架，但还不能直接跑。  
>
> 主要问题：
>
> ```text
> 1. 新增 Python / YAML 文件疑似被压成单行，可能直接 SyntaxError / YAML parse error。
> 2. generate_root_classifier_guidance.py 已经传 root_classifier 等参数，但 KimodoSceneCo 还不一定完整接收。
> 3. classifier dataset / generation 里可能直接使用 normalized root，而不是 meter-space root。
> 4. generate_root_classifier_guidance.py 可能只搜索 .pt，但项目缓存主要是 .npz。
> 5. 当前 classifier 还没有真正使用 scene，只能叫 RootPathClassifier，不能叫 RootPathSceneClassifier。
> ```
>
> 这份 README 是给本地 AI / Codex 的直接修改说明。

---

## 0. 目标方法定义

### 0.1 当前已有方法：Energy Guidance

当前已有的手写 loss guidance 应命名为：

```text
RootEnergyGuidance
AnalyticalRootGuidance
LossGuidance
```

它的形式是：

```text
target path / scene
    ↓
L_path + L_goal + L_speed + L_smooth + L_heading + L_scene
    ↓
∂L / ∂x_t
    ↓
update root_slice
```

这个方法不训练 classifier。

---

### 0.2 本次要实现的方法：True Root Classifier Guidance

真正 classifier guidance 是：

```text
target path / waypoint / planner path
        ↓
RootPathSceneClassifier
        ↑
pred_x0[root_slice]
        ↓
classifier score: p(valid | root, path, scene)
        ↓
L_cls = BCEWithLogits(logit, valid=1)
        ↓
∂L_cls / ∂x_t
        ↓
只更新 root_slice
        ↓
classifier-guided root_5d
```

也就是：

```text
训练 classifier → classifier 给 score → score loss 反传 → 修改 diffusion 采样状态 x_t
```

---

## 1. 当前必须先做的硬检查

在任何训练 / 生成前，先执行：

```bash
python -m py_compile \
  kimodo_sceneco/model/kimodo_model.py \
  kimodo_sceneco/critic/root_path_scene_classifier.py \
  kimodo_sceneco/critic/root_classifier_features.py \
  kimodo_sceneco/critic/root_classifier_dataset.py \
  kimodo_sceneco/critic/train_root_classifier.py \
  scripts/generate_root_classifier_guidance.py
```

如果有 `SyntaxError`，先修格式。  
当前很多新增文件可能被压缩成超长单行，例如：

```python
"""docstring""" import torch import torch.nn as nn ...
```

这是非法 Python，必须改成正常多行代码。

---

## 2. YAML 配置必须修成合法多行格式

检查：

```bash
python - <<'PY'
import yaml

for p in [
    "configs/root_classifier.yaml",
    "configs/root_classifier_guidance.yaml",
]:
    print("checking", p)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(type(cfg), cfg.keys() if isinstance(cfg, dict) else cfg)
PY
```

如果 YAML 是这样：

```yaml
experiment: name: root_path_scene_classifier output_dir: ...
```

就是错误的。

应该改成：

```yaml
experiment:
  name: root_path_scene_classifier
  output_dir: outputs/root_path_scene_classifier
```

---

## 3. 正确文件结构

最终应有：

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
├── generate_root_classifier_guidance.py
└── compare_guidance_methods.py
│
configs/
├── root_classifier.yaml
└── root_classifier_guidance.yaml
```

已有的：

```text
kimodo_sceneco/guidance/root_guidance.py
```

继续保留，作为：

```text
EnergyGuidance baseline
```

---

## 4. P0：修 Python 文件格式

### 4.1 `root_path_scene_classifier.py`

文件：

```text
kimodo_sceneco/critic/root_path_scene_classifier.py
```

应包含正常 Transformer classifier：

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

    def forward(self, frame_feat: torch.Tensor, pad_mask: torch.Tensor | None = None):
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

        return self.head(pooled)
```

---

### 4.2 `root_classifier_features.py`

文件：

```text
kimodo_sceneco/critic/root_classifier_features.py
```

应包含：

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

    return torch.cat(
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
```

特征维度：

```text
root_xz                  2
target_xz                2
root_minus_target         2
dist_to_target            1
root_vel                  2
target_vel                2
root_speed                1
target_speed              1
heading                   2
path_dir                  2
heading_path_error        1
sdf_value                 1
--------------------------------
total                    19
```

所以 config 中：

```yaml
model:
  input_dim: 19
```

如果额外加入 root_y、height_delta 等特征，再相应修改 input_dim。

---

## 5. P0：修配置文件

### 5.1 `configs/root_classifier.yaml`

应改成：

```yaml
experiment:
  name: root_path_classifier
  output_dir: outputs/root_path_classifier

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
  input_dim: 19
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

注意：

```text
use_scene_sdf: false 时，这个模型只能叫 RootPathClassifier。
```

不要报告成 RootPathSceneClassifier。

---

### 5.2 `configs/root_classifier_guidance.yaml`

应改成：

```yaml
experiment:
  name: root_classifier_guidance
  output_dir: outputs/root_classifier_guidance

model:
  checkpoint: models/Kimodo-SMPLX-RP-v1

root_classifier:
  checkpoint: outputs/root_path_classifier/best.pt
  input_dim: 19
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

## 6. P1：修 classifier 输入空间

### 问题

当前代码如果直接用：

```python
root_5d = motion[:T, :5]
target_path_xz = motion[:T, [0, 2]]
```

这是错误的。

原因：

```text
motion_features 是 z-score normalized Kimodo feature。
classifier 应该学习实际几何关系，输入应是 meter/canonical root。
```

---

### 正确做法

必须使用 motion representation 反归一化：

```text
motion_features
    ↓
motion_rep.unnormalize()
    ↓
motion_rep.inverse()
    ↓
smooth_root_pos
    ↓
global_root_heading
    ↓
root_5d_meter = [smooth_root_pos, global_root_heading]
```

### 建议新增工具函数

新增文件或放在 dataset 中：

```python
def extract_root_5d_meter(motion_rep, features_np, device="cpu"):
    """
    features_np:
        (T, D), normalized Kimodo feature.

    return:
        root_5d_meter: (T, 5)
    """
    import torch
    import numpy as np

    feat = torch.from_numpy(features_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        unnorm = motion_rep.unnormalize(feat)
        out = motion_rep.inverse(
            unnorm,
            is_normalized=False,
            return_numpy=True,
        )

    smooth_root_pos = out["smooth_root_pos"][0]

    if "global_root_heading" in out:
        heading = out["global_root_heading"][0]
    elif "root_heading" in out:
        heading = out["root_heading"][0]
    else:
        raise KeyError("Cannot find global root heading in motion_rep.inverse output")

    root_5d_meter = np.concatenate([smooth_root_pos, heading], axis=-1)
    return root_5d_meter.astype(np.float32)
```

如果 `motion_rep.inverse()` 的 key 名不同，按项目真实输出修改。

---

## 7. P1：修 `.npz` cache 加载

### 问题

如果 `generate_root_classifier_guidance.py` 只找：

```python
glob("*.pt")
```

这是错的。项目缓存通常是：

```text
lingo_smplx_cache/seg_XXXXX.npz
```

### 修改

```python
def find_cache_files(cache_dir):
    from pathlib import Path

    cache_dir = Path(cache_dir)
    candidates = [
        cache_dir,
        cache_dir / "train",
        cache_dir / "val",
    ]

    files = []
    for c in candidates:
        if c.exists():
            files.extend(sorted(c.glob("*.npz")))
            files.extend(sorted(c.glob("*.pt")))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No .npz or .pt cache files found in {cache_dir}")

    return files
```

加载时：

```python
def load_motion_features(path):
    import torch
    import numpy as np

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "motion_features" in data:
            return data["motion_features"]
        if "motion" in data:
            return data["motion"]
        raise KeyError(f"No motion_features in {path}")

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
        if "motion_features" in data:
            return data["motion_features"].numpy()
        if "motion" in data:
            return data["motion"].numpy()
        raise KeyError(f"No motion_features in {path}")

    raise ValueError(path)
```

---

## 8. P1：修 `root_classifier_dataset.py`

Dataset 必须返回 meter-space root。

核心逻辑：

```python
class RootClassifierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir,
        motion_rep,
        max_frames=196,
        positive_ratio=0.5,
        negative_modes=None,
        use_scene_sdf=False,
        scene_dir=None,
    ):
        self.files = find_cache_files(cache_dir)
        self.motion_rep = motion_rep
        self.max_frames = max_frames
        self.positive_ratio = positive_ratio
        self.negative_modes = negative_modes or [
            "shift",
            "wrong_goal",
            "jitter",
            "wrong_heading",
            "reverse_heading",
            "path_shuffle",
        ]

    def __getitem__(self, idx):
        file = self.files[idx]
        features = load_motion_features(file)

        root_5d = extract_root_5d_meter(
            self.motion_rep,
            features,
            device="cpu",
        )

        T = min(root_5d.shape[0], self.max_frames)
        root_5d = root_5d[:T]
        target_path_xz = root_5d[:, [0, 2]].copy()

        label = 1.0
        negative_mode = "none"

        if np.random.rand() > self.positive_ratio:
            negative_mode = sample_negative_mode()
            root_5d = make_negative_root_numpy(root_5d, negative_mode)
            label = 0.0

        pad_root, pad_mask = pad_to_length(root_5d, self.max_frames)
        pad_path, _ = pad_to_length(target_path_xz, self.max_frames)

        return {
            "root_5d": pad_root.astype(np.float32),
            "target_path_xz": pad_path.astype(np.float32),
            "pad_mask": pad_mask.astype(bool),
            "label": np.float32(label),
            "negative_mode": negative_mode,
            "source_file": str(file),
        }
```

---

## 9. P2：接入 KimodoSceneCo 的 classifier guidance 参数

### 当前问题

`generate_root_classifier_guidance.py` 已经传：

```python
root_classifier=root_classifier
classifier_guidance_scale=...
classifier_max_grad_norm=...
root_classifier_start_step=...
root_classifier_end_step=...
hybrid=...
w_classifier=...
w_energy=...
```

但 `KimodoSceneCo.__call__()` / `_generate()` 如果没有这些参数，会报：

```text
TypeError: unexpected keyword argument 'root_classifier'
```

---

### 必须修改 `kimodo_model.py`

在以下函数中增加参数并透传：

```text
KimodoSceneCo.__call__()
KimodoSceneCo._multiprompt()
KimodoSceneCo._generate()
```

参数：

```python
root_classifier=None
classifier_guidance_scale: float = 0.05
classifier_max_grad_norm: float = 1.0
root_classifier_start_step: int = 0
root_classifier_end_step: int = 40
hybrid: bool = False
w_classifier: float = 1.0
w_energy: float = 0.3
```

---

## 10. P2：新增 `denoising_step_with_root_classifier_guidance()`

文件：

```text
kimodo_sceneco/model/kimodo_model.py
```

新增方法：

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
    classifier_max_grad_norm=1.0,
    scene_feat_root=None,
    scene_mask_root=None,
    scene_feat_body=None,
    scene_mask_body=None,
    traj_feats=None,
    traj_mask=None,
    cfg_type=None,
    hybrid=False,
    w_classifier=1.0,
    w_energy=0.3,
):
    import torch
    import torch.nn.functional as F

    from kimodo_sceneco.critic.root_classifier_features import build_root_classifier_features
    from kimodo_sceneco.guidance.root_guidance import (
        denormalize_root_5d,
        compute_root_guidance_loss,
    )

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
        traj_feats=traj_feats,
        traj_mask=traj_mask,
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

    loss_cls = F.binary_cross_entropy_with_logits(logit, label_valid)

    if hybrid:
        energy_losses = compute_root_guidance_loss(
            pred_x0=pred_x0,
            target_path_xz=target_path_xz,
            root_slice=self.motion_rep.root_slice,
            cfg=self.root_guidance_cfg,
            scene_sdf=scene_sdf,
            sample_sdf_fn=None,
            motion_rep=self.motion_rep,
            root_is_normalized=True,
        )
        loss_total = w_classifier * loss_cls + w_energy * energy_losses["total"]
    else:
        energy_losses = {}
        loss_total = loss_cls

    grad = torch.autograd.grad(loss_total, x)[0]

    root_grad = torch.zeros_like(grad)
    root_grad[..., self.motion_rep.root_slice] = grad[..., self.motion_rep.root_slice]
    grad = root_grad

    grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    grad = grad * (classifier_max_grad_norm / grad_norm).clamp(max=1.0)

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
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            cfg_type=cfg_type,
        )

        x_tm1 = self.sampler(use_timesteps, x_guided, pred_clean, t)

    logs = {
        "loss_cls": loss_cls.detach(),
        "score_valid": torch.sigmoid(logit).mean().detach(),
        "loss_total": loss_total.detach(),
    }

    for k, v in energy_losses.items():
        logs[f"energy_{k}"] = v.detach()

    return x_tm1, logs
```

注意：

```text
上面用了 self.root_guidance_cfg，仅在 hybrid=True 时需要。
如果项目中没有 self.root_guidance_cfg，应把 energy cfg 作为参数传入。
```

---

## 11. P2：修改 `_generate()` 分支逻辑

在 denoising loop 中加入：

```python
apply_classifier_guidance = (
    root_classifier is not None
    and root_classifier_start_step <= step_id <= root_classifier_end_step
)

if apply_classifier_guidance:
    cur_mot, cls_logs = self.denoising_step_with_root_classifier_guidance(
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
        root_classifier=root_classifier,
        target_path_xz=target_path_xz,
        scene_sdf=scene_sdf,
        classifier_guidance_scale=classifier_guidance_scale,
        classifier_max_grad_norm=classifier_max_grad_norm,
        scene_feat_root=scene_feat_root,
        scene_mask_root=scene_mask_root,
        scene_feat_body=scene_feat_body,
        scene_mask_body=scene_mask_body,
        traj_feats=traj_feats,
        traj_mask=traj_mask,
        cfg_type=cfg_type,
        hybrid=hybrid,
        w_classifier=w_classifier,
        w_energy=w_energy,
    )
elif enable_root_guidance:
    # existing EnergyGuidance branch
    ...
else:
    # normal denoising
    ...
```

---

## 12. P2：修 `generate_root_classifier_guidance.py`

### 12.1 文件查找

必须支持：

```text
.npz
.pt
```

优先 `.npz`。

### 12.2 target_path_xz

不能再用：

```python
target_path_xz = motion[:T, [0, 2]]
```

必须改成：

```python
root_5d_meter = extract_root_5d_meter(model.motion_rep, motion_features)
target_path_xz = root_5d_meter[:, [0, 2]]
```

### 12.3 model call 参数

调用前确保 `KimodoSceneCo.__call__()` 已经接收：

```python
root_classifier=root_classifier
classifier_guidance_scale=...
classifier_max_grad_norm=...
root_classifier_start_step=...
root_classifier_end_step=...
hybrid=...
w_classifier=...
w_energy=...
```

---

## 13. P3：Scene classifier 后续扩展

当前如果：

```yaml
use_scene_sdf: false
```

则方法只能叫：

```text
RootPathClassifier
```

要升级为 `RootPathSceneClassifier`，需要：

```text
1. dataset 加载 scene voxel。
2. scene voxel 转 2D SDF。
3. feature builder 中采样 sdf_value。
4. 加 scene_shift / scene_shuffle 负样本。
5. 配置 use_scene_sdf: true。
```

这不是第一版必须，但最终要证明 scene classifier 时必须补。

---

## 14. 执行命令

### 14.1 先检查语法

```bash
python -m py_compile \
  kimodo_sceneco/model/kimodo_model.py \
  kimodo_sceneco/critic/root_path_scene_classifier.py \
  kimodo_sceneco/critic/root_classifier_features.py \
  kimodo_sceneco/critic/root_classifier_dataset.py \
  kimodo_sceneco/critic/train_root_classifier.py \
  scripts/generate_root_classifier_guidance.py
```

### 14.2 检查 YAML

```bash
python - <<'PY'
import yaml
for p in ["configs/root_classifier.yaml", "configs/root_classifier_guidance.yaml"]:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(p, cfg.keys())
PY
```

### 14.3 训练 classifier

使用物理 1 号 GPU：

```bash
cd /path/to/SceneCoHSI

export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models

python kimodo_sceneco/critic/train_root_classifier.py \
  --config configs/root_classifier.yaml \
  --output_dir outputs/root_path_classifier \
  --batch_size 64 \
  --num_epochs 100 \
  --lr 1e-4 \
  --gpu 0 \
  2>&1 | tee outputs/root_path_classifier/train.log
```

### 14.4 生成 classifier-guided root

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/root_classifier_guidance/path_only \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --gpu 0
```

### 14.5 Stage2 body

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/root_classifier_guidance/path_only \
  --output_dir outputs/root_classifier_guidance/path_only_body \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0
```

---

## 15. 实验矩阵

必须比较：

| 方法 | Root 控制 | 是否训练 classifier | Body |
|---|---|---:|---|
| `NoGuidance` | 无 | 否 | 原始 Body |
| `EnergyGuidance` | 手写 loss | 否 | 原始 Body |
| `ClassifierGuidance` | classifier score | 是 | 原始 Body |
| `HybridGuidance` | 手写 loss + classifier score | 是 | 原始 Body |
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

## 16. 验收标准

### 16.1 classifier 训练

```text
val_acc > 0.85
AUC > 0.85
positive_score_mean > negative_score_mean
```

并且按 negative mode 分开看：

```text
shift
wrong_goal
jitter
wrong_heading
reverse_heading
path_shuffle
```

不能只看总体 acc。

---

### 16.2 classifier guidance 生成

日志中应看到：

```text
score_valid 上升
loss_cls 下降或稳定
grad_norm > 0
```

评估上应看到：

```text
ClassifierGuidance 的 PathADE / PathFDE 优于 NoGuidance
ClassifierGuidance 的 RootJerk / SpeedStd 不爆炸
```

---

### 16.3 和 EnergyGuidance 的比较

如果 ClassifierGuidance 不如 EnergyGuidance，不能硬说成功。应报告：

```text
EnergyGuidance 更稳定；
ClassifierGuidance 需要更好的负样本或 HybridGuidance。
```

HybridGuidance 通常应该更稳：

```text
手写 loss 保证几何约束；
classifier score 提供 learned validity prior。
```

---

## 17. 最短结论

当前 GitHub 框架已经有了，但还不能直接跑 true classifier guidance。必须先修：

```text
1. Python / YAML 格式。
2. .npz cache 加载。
3. meter-space root 提取。
4. KimodoSceneCo 接收 root_classifier 参数。
5. 新增 denoising_step_with_root_classifier_guidance。
6. generate_root_classifier_guidance.py 的 target path 提取。
```

修完后，才是真正的：

```text
训练 RootPathClassifier → classifier score → ∂L_cls/∂x_t → root_slice 注入
```

当前手写 loss guidance 保留为：

```text
EnergyGuidance baseline
```
