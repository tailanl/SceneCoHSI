# README：SceneCoHSI 实验执行手册（详细版，带检查点）

> 适用对象：执行能力较弱的 AI / Codex。  
> 核心原则：**不要一次做完所有实验。每个阶段必须先 smoke test，再正式训练，再评估。检查不过，不许进入下一阶段。**

---

## 0. 总规则

每执行一个实验，都必须维护这些文件：

```text
AUTORESEARCH_STATE.md
TASK_CHECKLIST.md
COMMAND_LOG.md
NEXT_ACTIONS.md
FINAL_REPORT.md
```

每一步命令都要记录到 `COMMAND_LOG.md`：

```text
Command:
Result: PASS / FAIL
Key output:
Generated files:
Next action:
```

如果失败，必须停止当前实验，写明：

```text
1. 失败命令
2. 报错摘要
3. 已生成文件
4. 下一步修复命令
```

禁止事项：

```text
1. 不要跳过 smoke test。
2. 不要在 root 文件不存在时训练 Stage2。
3. 不要在大量 fallback 时训练 Stage2。
4. 不要在 py_compile / YAML 检查失败时训练。
5. 不要把 EnergyGuidance 叫成 ClassifierGuidance。
6. 不要用 normalized motion[:T, :5] 训练 classifier。
```

---

## 1. 统一环境设置

进入仓库：

```bash
cd /path/to/SceneCoHSI
```

使用物理 1 号 GPU：

```bash
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models
```

注意：设置 `CUDA_VISIBLE_DEVICES=1` 后，程序内部这张卡是 `cuda:0`，所以命令里写：

```bash
--gpu 0
```

---

## 2. 全局预检查：所有实验前必须做

### 2.1 Python 语法检查

```bash
python -m py_compile \
  kimodo_sceneco/model/kimodo_model.py \
  kimodo_sceneco/critic/root_path_scene_classifier.py \
  kimodo_sceneco/critic/root_classifier_features.py \
  kimodo_sceneco/critic/root_classifier_dataset.py \
  kimodo_sceneco/critic/train_root_classifier.py \
  scripts/generate_root_classifier_guidance.py \
  scripts/generate_root_guidance.py \
  scripts/generate_body_from_root.py \
  train/train_stage2_root_guided_sceneco.py
```

检查点：

```text
必须没有 SyntaxError / IndentationError。
失败则停止所有训练，先修代码格式。
```

### 2.2 YAML 检查

```bash
python - <<'PY'
import yaml
files = [
    "configs/root_classifier.yaml",
    "configs/root_classifier_guidance.yaml",
    "configs/guidance_root_scene.yaml",
    "configs/stage2_root_guided_sceneco.yaml",
]
for p in files:
    print("Checking:", p)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"{p} did not parse as dict"
    print("OK:", cfg.keys())
PY
```

检查点：

```text
所有 YAML 必须 safe_load 成 dict。
失败则先修 YAML，不许训练。
```

### 2.3 缓存数据检查

```bash
python - <<'PY'
from pathlib import Path
cache_dir = Path("lingo_smplx_cache")
files = sorted(cache_dir.glob("*.npz")) + sorted(cache_dir.glob("*.pt"))
print("num files:", len(files))
print("first files:", files[:5])
assert len(files) > 0, "No cache files found"
PY
```

检查点：

```text
必须找到 .npz 或 .pt。
项目通常应有 lingo_smplx_cache/seg_XXXXX.npz。
```

---

## 3. 实验总表

| 编号 | 实验 | Root 来源 | Body 来源 | 是否训练 | 主要目的 |
|---:|---|---|---|---:|---|
| E0 | NoGuidance + Original Body | Kimodo 原始 root | 原始 Body | 否 | 原始 baseline |
| E1 | EnergyGuidance + Original Body | 手写 loss guided root | 原始 Body | 否 | 验证 EnergyGuidance 控制 root |
| E2 | ClassifierGuidance + Original Body | classifier-guided root | 原始 Body | 否 | 验证真正 classifier guidance |
| E3 | HybridGuidance + Original Body | classifier + energy root | 原始 Body | 否 | 验证 hybrid 是否更稳 |
| E4 | EnergyGuidance + Stage2 SceneCo | Energy root | 训练后 SceneCo Body | 是 | 验证 Stage2 SceneCo 是否改善 body |
| E5 | ClassifierGuidance + Stage2 SceneCo | Classifier root | 训练后 SceneCo Body | 是 | 主方法 |
| E6 | HybridGuidance + Stage2 SceneCo | Hybrid root | 训练后 SceneCo Body | 可选 | 增强版 |
| E7 | GTRoot + Stage2 SceneCo | GT root | 训练后 SceneCo Body | 是 / 上限 | Debug / 上限 |

推荐顺序：

```text
E1 → E4 → E2 → E5 → E3 → E6 → E7
```

资源少时先做：

```text
E1, E2, E4, E5
```

---

# 4. E0：NoGuidance + Original Body

## 目的

建立原始 baseline。

## 步骤

```bash
mkdir -p outputs/e0_noguidance_original_body
```

如果项目有原始生成脚本：

```bash
python scripts/generate.py \
  --output_dir outputs/e0_noguidance_original_body \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --gpu 0 \
  2>&1 | tee outputs/e0_noguidance_original_body/generate.log
```

如果没有 `scripts/generate.py`，停止并记录：

```text
BLOCKED: 原始 baseline 生成脚本不存在，需要新增 scripts/generate_noguidance_baseline.py。
```

## 检查点

```bash
ls outputs/e0_noguidance_original_body
```

必须有 motion 输出和 `generate.log`。

## 评估

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e0_noguidance_original_body \
  --output_csv outputs/e0_noguidance_original_body/path_metrics.csv \
  --method e0_noguidance_original_body
```

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e0_noguidance_original_body \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/e0_noguidance_original_body/scene_metrics.csv \
  --method e0_noguidance_original_body
```

---

# 5. E1：EnergyGuidance + Original Body

## 目的

验证手写 loss guidance 是否能控制 root。

## Step 1：生成 EnergyGuidance root

```bash
mkdir -p outputs/e1_energy_guidance_root
```

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/e1_energy_guidance_root \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --gpu 0 \
  2>&1 | tee outputs/e1_energy_guidance_root/generate_root.log
```

### 检查点 1：root 文件存在

```bash
ls outputs/e1_energy_guidance_root | head
```

检查 `.npz` 字段：

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
files = sorted(Path("outputs/e1_energy_guidance_root").glob("*.npz"))
print("num files:", len(files))
assert len(files) > 0
x = np.load(files[0], allow_pickle=True)
print("keys:", x.files)
assert "guided_root_5d_norm" in x.files
assert "target_path_xz" in x.files
PY
```

失败则停止，修 `generate_root_guidance.py` 保存格式。

## Step 2：Original Body 生成

```bash
mkdir -p outputs/e1_energy_guidance_body
```

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/e1_energy_guidance_root \
  --output_dir outputs/e1_energy_guidance_body \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0 \
  2>&1 | tee outputs/e1_energy_guidance_body/generate_body.log
```

### 检查点 2：root 固定

```bash
grep -i "root fix" outputs/e1_energy_guidance_body/generate_body.log || true
```

要求：

```text
root fix max error < 1e-5
```

如果没有 root fix 日志，停止，修 `generate_body_from_root.py`。

## Step 3：评估

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e1_energy_guidance_body \
  --output_csv outputs/e1_energy_guidance_body/path_metrics.csv \
  --method e1_energy_guidance_original_body
```

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e1_energy_guidance_body \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/e1_energy_guidance_body/scene_metrics.csv \
  --method e1_energy_guidance_original_body
```

---

# 6. E2：ClassifierGuidance + Original Body

## 目的

验证真正 trained classifier guidance 是否有效。

---

## E2-A：训练 RootPathClassifier

### Step 1：组件 shape 测试

```bash
python - <<'PY'
import torch
from kimodo_sceneco.critic.root_classifier_features import build_root_classifier_features
from kimodo_sceneco.critic.root_path_scene_classifier import RootPathSceneClassifier
B, T = 2, 196
root = torch.randn(B, T, 5)
path = torch.randn(B, T, 2)
mask = torch.ones(B, T).bool()
feat = build_root_classifier_features(root, path)
print("feature shape:", feat.shape)
assert feat.shape[-1] in [19, 20]
model = RootPathSceneClassifier(input_dim=feat.shape[-1])
logit = model(feat, mask)
print("logit shape:", logit.shape)
assert logit.shape == (B, 1)
PY
```

如果 feature 维度是 20，则 `configs/root_classifier.yaml` 和 `configs/root_classifier_guidance.yaml` 里的 `input_dim` 必须是 20。  
如果是 19，则全部统一为 19。不要混用。

### Step 2：训练 smoke test

```bash
mkdir -p outputs/root_path_classifier_smoke
```

```bash
python kimodo_sceneco/critic/train_root_classifier.py \
  --config configs/root_classifier.yaml \
  --output_dir outputs/root_path_classifier_smoke \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 1e-4 \
  --gpu 0 \
  2>&1 | tee outputs/root_path_classifier_smoke/train.log
```

检查：

```bash
ls outputs/root_path_classifier_smoke
```

必须有：

```text
latest.pt
train_log.csv
```

最好有：

```text
best.pt
```

如果没有 `best.pt`，必须修保存逻辑，不能继续正式训练。

### Step 3：正式训练 classifier

使用 tmux：

```bash
tmux new -s root_classifier_train
```

在 tmux 中：

```bash
cd /path/to/SceneCoHSI
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models
mkdir -p outputs/root_path_classifier

python kimodo_sceneco/critic/train_root_classifier.py \
  --config configs/root_classifier.yaml \
  --output_dir outputs/root_path_classifier \
  --batch_size 64 \
  --num_epochs 100 \
  --lr 1e-4 \
  --gpu 0 \
  2>&1 | tee outputs/root_path_classifier/train.log
```

退出 tmux：`Ctrl+B` 然后 `D`。  
恢复：

```bash
tmux attach -t root_classifier_train
```

### 检查点：classifier 是否合格

```bash
tail -n 100 outputs/root_path_classifier/train.log
```

最低要求：

```text
val_acc > 0.85
positive_score_mean > negative_score_mean
loss 下降
```

如果达不到，停止 E2，检查：

```text
1. dataset 是否用了 meter-space root；
2. target_path_xz 是否来自 smooth_root_pos[:, [0,2]]；
3. negative samples 是否真的改变 root；
4. input_dim 是否一致。
```

---

## E2-B：生成 classifier-guided root

### Step 1：smoke generation

```bash
mkdir -p outputs/e2_classifier_guidance_root_smoke
```

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/e2_classifier_guidance_root_smoke \
  --num_samples 2 \
  --num_denoising_steps 5 \
  --gpu 0 \
  2>&1 | tee outputs/e2_classifier_guidance_root_smoke/generate.log
```

检查：

```bash
ls outputs/e2_classifier_guidance_root_smoke
```

必须有：

```text
*.npz
guidance_log.csv
generate.log
```

日志必须包含：

```text
loss_cls
score_valid
grad_norm
```

否则说明 classifier guidance 没真正生效。

### Step 2：正式生成 root

```bash
mkdir -p outputs/e2_classifier_guidance_root
```

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/e2_classifier_guidance_root \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --gpu 0 \
  2>&1 | tee outputs/e2_classifier_guidance_root/generate.log
```

---

## E2-C：Original Body

```bash
mkdir -p outputs/e2_classifier_guidance_body
```

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/e2_classifier_guidance_root \
  --output_dir outputs/e2_classifier_guidance_body \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0 \
  2>&1 | tee outputs/e2_classifier_guidance_body/generate_body.log
```

检查 root：

```bash
grep -i "root fix" outputs/e2_classifier_guidance_body/generate_body.log || true
```

要求：`max error < 1e-5`。

## E2-D：评估

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e2_classifier_guidance_body \
  --output_csv outputs/e2_classifier_guidance_body/path_metrics.csv \
  --method e2_classifier_guidance_original_body
```

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e2_classifier_guidance_body \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/e2_classifier_guidance_body/scene_metrics.csv \
  --method e2_classifier_guidance_original_body
```

---

# 7. E3：HybridGuidance + Original Body

## 目的

验证 classifier loss + energy loss 是否更稳。

```text
L_total = w_classifier * L_cls + w_energy * L_energy
```

## Step 1：生成 hybrid root

```bash
mkdir -p outputs/e3_hybrid_guidance_root
```

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/e3_hybrid_guidance_root \
  --num_samples 30 \
  --num_denoising_steps 50 \
  --hybrid \
  --gpu 0 \
  2>&1 | tee outputs/e3_hybrid_guidance_root/generate.log
```

检查：

```bash
ls outputs/e3_hybrid_guidance_root | head
```

必须有 `.npz` 和 `guidance_log.csv`。

## Step 2：Original Body

```bash
mkdir -p outputs/e3_hybrid_guidance_body
```

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/e3_hybrid_guidance_root \
  --output_dir outputs/e3_hybrid_guidance_body \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0 \
  2>&1 | tee outputs/e3_hybrid_guidance_body/generate_body.log
```

## Step 3：评估

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e3_hybrid_guidance_body \
  --output_csv outputs/e3_hybrid_guidance_body/path_metrics.csv \
  --method e3_hybrid_guidance_original_body
```

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e3_hybrid_guidance_body \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/e3_hybrid_guidance_body/scene_metrics.csv \
  --method e3_hybrid_guidance_original_body
```

---

# 8. E4：EnergyGuidance + Stage2 SceneCo

## 目的

训练 Stage2 SceneCo，让 body 适配 EnergyGuidance root。

---

## Step 1：生成 train / val Energy root

Train：

```bash
mkdir -p outputs/e4_energy_guidance_train/path_only
```

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/e4_energy_guidance_train/path_only \
  --split train \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --gpu 0 \
  2>&1 | tee outputs/e4_energy_guidance_train/path_only/generate.log
```

Val：

```bash
mkdir -p outputs/e4_energy_guidance_val/path_only
```

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --output_dir outputs/e4_energy_guidance_val/path_only \
  --split val \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --gpu 0 \
  2>&1 | tee outputs/e4_energy_guidance_val/path_only/generate.log
```

检查：

```bash
ls outputs/e4_energy_guidance_train/path_only | head
ls outputs/e4_energy_guidance_val/path_only | head
```

必须看到 `.npz`。

---

## Step 2：配置 Stage2

```bash
cp configs/stage2_root_guided_sceneco.yaml configs/stage2_energy_root_guided_sceneco.yaml
```

确认配置或命令行中使用：

```text
path_guided_root_dir = outputs/e4_energy_guidance_train/path_only
path_scene_guided_root_dir = outputs/e4_energy_guidance_train/path_only
val_root_dir = outputs/e4_energy_guidance_val/path_only
root_mix_gt = 0.3
root_mix_path = 0.7
root_mix_scene = 0.0
```

---

## Step 3：smoke training

```bash
mkdir -p outputs/e4_energy_stage2_sceneco_smoke
```

```bash
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_energy_root_guided_sceneco.yaml \
  --gpu 0 \
  --external_root_enabled true \
  --use_external_root true \
  --path_guided_root_dir outputs/e4_energy_guidance_train/path_only \
  --path_scene_guided_root_dir outputs/e4_energy_guidance_train/path_only \
  --val_root_dir outputs/e4_energy_guidance_val/path_only \
  --root_mix_gt 0.3 \
  --root_mix_path 0.7 \
  --root_mix_scene 0.0 \
  --batch_size 2 \
  --num_epochs 1 \
  --lr 1e-4 \
  --prior_weight 0.0 \
  --scene_dropout 0.1 \
  --num_workers 0 \
  2>&1 | tee outputs/e4_energy_stage2_sceneco_smoke/train.log
```

检查：

```bash
grep -i "fallback" outputs/e4_energy_stage2_sceneco_smoke/train.log || true
grep -i "external_root" outputs/e4_energy_stage2_sceneco_smoke/train.log || true
```

如果大量 fallback，停止。说明 root 文件名没有匹配 dataset sample id。

---

## Step 4：正式训练

```bash
tmux new -s e4_energy_stage2_sceneco
```

在 tmux 中：

```bash
cd /path/to/SceneCoHSI
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models
mkdir -p outputs/e4_energy_stage2_sceneco

python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_energy_root_guided_sceneco.yaml \
  --gpu 0 \
  --external_root_enabled true \
  --use_external_root true \
  --path_guided_root_dir outputs/e4_energy_guidance_train/path_only \
  --path_scene_guided_root_dir outputs/e4_energy_guidance_train/path_only \
  --val_root_dir outputs/e4_energy_guidance_val/path_only \
  --root_mix_gt 0.3 \
  --root_mix_path 0.7 \
  --root_mix_scene 0.0 \
  --batch_size 4 \
  --num_epochs 400 \
  --lr 1e-4 \
  --prior_weight 0.0 \
  --scene_dropout 0.1 \
  --num_workers 4 \
  2>&1 | tee outputs/e4_energy_stage2_sceneco/train.log
```

---

## Step 5：生成和评估

```bash
mkdir -p outputs/e4_energy_stage2_sceneco/val_gen
```

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/e4_energy_guidance_val/path_only \
  --checkpoint outputs/e4_energy_stage2_sceneco/checkpoints/best_checkpoint.pt \
  --output_dir outputs/e4_energy_stage2_sceneco/val_gen \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0 \
  2>&1 | tee outputs/e4_energy_stage2_sceneco/val_gen/generate_body.log
```

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e4_energy_stage2_sceneco/val_gen \
  --output_csv outputs/e4_energy_stage2_sceneco/path_metrics.csv \
  --method e4_energy_guidance_stage2_sceneco
```

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e4_energy_stage2_sceneco/val_gen \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/e4_energy_stage2_sceneco/scene_metrics.csv \
  --method e4_energy_guidance_stage2_sceneco
```

---

# 9. E5：ClassifierGuidance + Stage2 SceneCo

## 目的

主方法：classifier-guided root + Stage2 SceneCo body。

---

## Step 1：生成 train / val classifier root

Train：

```bash
mkdir -p outputs/e5_classifier_guidance_train/path_only
```

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/e5_classifier_guidance_train/path_only \
  --split train \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --gpu 0 \
  2>&1 | tee outputs/e5_classifier_guidance_train/path_only/generate.log
```

Val：

```bash
mkdir -p outputs/e5_classifier_guidance_val/path_only
```

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/e5_classifier_guidance_val/path_only \
  --split val \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --gpu 0 \
  2>&1 | tee outputs/e5_classifier_guidance_val/path_only/generate.log
```

检查：

```bash
ls outputs/e5_classifier_guidance_train/path_only | head
ls outputs/e5_classifier_guidance_val/path_only | head
```

---

## Step 2：配置 Stage2

```bash
cp configs/stage2_root_guided_sceneco.yaml configs/stage2_classifier_root_guided_sceneco.yaml
```

确认配置或命令行中使用：

```text
path_guided_root_dir = outputs/e5_classifier_guidance_train/path_only
path_scene_guided_root_dir = outputs/e5_classifier_guidance_train/path_only
val_root_dir = outputs/e5_classifier_guidance_val/path_only
root_mix_gt = 0.3
root_mix_path = 0.7
root_mix_scene = 0.0
```

---

## Step 3：smoke training

```bash
mkdir -p outputs/e5_classifier_stage2_sceneco_smoke
```

```bash
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_classifier_root_guided_sceneco.yaml \
  --gpu 0 \
  --external_root_enabled true \
  --use_external_root true \
  --path_guided_root_dir outputs/e5_classifier_guidance_train/path_only \
  --path_scene_guided_root_dir outputs/e5_classifier_guidance_train/path_only \
  --val_root_dir outputs/e5_classifier_guidance_val/path_only \
  --root_mix_gt 0.3 \
  --root_mix_path 0.7 \
  --root_mix_scene 0.0 \
  --batch_size 2 \
  --num_epochs 1 \
  --lr 1e-4 \
  --prior_weight 0.0 \
  --scene_dropout 0.1 \
  --num_workers 0 \
  2>&1 | tee outputs/e5_classifier_stage2_sceneco_smoke/train.log
```

检查：

```bash
grep -i "fallback" outputs/e5_classifier_stage2_sceneco_smoke/train.log || true
grep -i "external_root" outputs/e5_classifier_stage2_sceneco_smoke/train.log || true
```

大量 fallback 则停止。

---

## Step 4：正式训练

```bash
tmux new -s e5_classifier_stage2_sceneco
```

在 tmux 中：

```bash
cd /path/to/SceneCoHSI
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models
mkdir -p outputs/e5_classifier_stage2_sceneco

python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_classifier_root_guided_sceneco.yaml \
  --gpu 0 \
  --external_root_enabled true \
  --use_external_root true \
  --path_guided_root_dir outputs/e5_classifier_guidance_train/path_only \
  --path_scene_guided_root_dir outputs/e5_classifier_guidance_train/path_only \
  --val_root_dir outputs/e5_classifier_guidance_val/path_only \
  --root_mix_gt 0.3 \
  --root_mix_path 0.7 \
  --root_mix_scene 0.0 \
  --batch_size 4 \
  --num_epochs 400 \
  --lr 1e-4 \
  --prior_weight 0.0 \
  --scene_dropout 0.1 \
  --num_workers 4 \
  2>&1 | tee outputs/e5_classifier_stage2_sceneco/train.log
```

---

## Step 5：生成与评估

```bash
mkdir -p outputs/e5_classifier_stage2_sceneco/val_gen
```

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/e5_classifier_guidance_val/path_only \
  --checkpoint outputs/e5_classifier_stage2_sceneco/checkpoints/best_checkpoint.pt \
  --output_dir outputs/e5_classifier_stage2_sceneco/val_gen \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0 \
  2>&1 | tee outputs/e5_classifier_stage2_sceneco/val_gen/generate_body.log
```

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e5_classifier_stage2_sceneco/val_gen \
  --output_csv outputs/e5_classifier_stage2_sceneco/path_metrics.csv \
  --method e5_classifier_guidance_stage2_sceneco
```

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e5_classifier_stage2_sceneco/val_gen \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/e5_classifier_stage2_sceneco/scene_metrics.csv \
  --method e5_classifier_guidance_stage2_sceneco
```

---

# 10. E6：HybridGuidance + Stage2 SceneCo

E6 和 E5 完全相同，只是 root 生成时加入 `--hybrid`。

## 生成 hybrid train / val root

```bash
mkdir -p outputs/e6_hybrid_guidance_train/path_only
mkdir -p outputs/e6_hybrid_guidance_val/path_only
```

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/e6_hybrid_guidance_train/path_only \
  --split train \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --hybrid \
  --gpu 0 \
  2>&1 | tee outputs/e6_hybrid_guidance_train/path_only/generate.log
```

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_guidance.yaml \
  --classifier_ckpt outputs/root_path_classifier/best.pt \
  --output_dir outputs/e6_hybrid_guidance_val/path_only \
  --split val \
  --num_samples -1 \
  --num_denoising_steps 50 \
  --hybrid \
  --gpu 0 \
  2>&1 | tee outputs/e6_hybrid_guidance_val/path_only/generate.log
```

然后照 E5 的 Stage2 训练流程，把 root 路径改为：

```text
outputs/e6_hybrid_guidance_train/path_only
outputs/e6_hybrid_guidance_val/path_only
```

---

# 11. E7：GTRoot + Stage2 SceneCo

## 目的

上限实验。判断 Stage2 SceneCo 的能力上限。

如果 E7 好但 E4/E5 差：

```text
root guidance 质量不够
```

如果 E7 也差：

```text
Stage2 SceneCo 本身有问题
```

## Step 1：导出 GT root

需要脚本：

```text
scripts/export_gt_root_for_stage2.py
```

要求每个 `.npz` 包含：

```text
guided_root_5d_norm = motion_features[:, :5]
guided_root_5d_meter
target_path_xz
text
scene_name
source_file
```

命令：

```bash
mkdir -p outputs/e7_gt_root_train
mkdir -p outputs/e7_gt_root_val
```

```bash
python scripts/export_gt_root_for_stage2.py \
  --split train \
  --output_dir outputs/e7_gt_root_train \
  --gpu 0
```

```bash
python scripts/export_gt_root_for_stage2.py \
  --split val \
  --output_dir outputs/e7_gt_root_val \
  --gpu 0
```

检查：

```bash
ls outputs/e7_gt_root_train | head
ls outputs/e7_gt_root_val | head
```

## Step 2：训练 Stage2

```bash
cp configs/stage2_root_guided_sceneco.yaml configs/stage2_gt_root_sceneco.yaml
```

训练：

```bash
tmux new -s e7_gt_root_stage2_sceneco
```

在 tmux 中：

```bash
cd /path/to/SceneCoHSI
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR=$PWD/models
mkdir -p outputs/e7_gt_root_stage2_sceneco

python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_gt_root_sceneco.yaml \
  --gpu 0 \
  --external_root_enabled true \
  --use_external_root true \
  --path_guided_root_dir outputs/e7_gt_root_train \
  --path_scene_guided_root_dir outputs/e7_gt_root_train \
  --val_root_dir outputs/e7_gt_root_val \
  --root_mix_gt 0.0 \
  --root_mix_path 1.0 \
  --root_mix_scene 0.0 \
  --batch_size 4 \
  --num_epochs 400 \
  --lr 1e-4 \
  --prior_weight 0.0 \
  --scene_dropout 0.1 \
  --num_workers 4 \
  2>&1 | tee outputs/e7_gt_root_stage2_sceneco/train.log
```

## Step 3：生成和评估

```bash
mkdir -p outputs/e7_gt_root_stage2_sceneco/val_gen
```

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/e7_gt_root_val \
  --checkpoint outputs/e7_gt_root_stage2_sceneco/checkpoints/best_checkpoint.pt \
  --output_dir outputs/e7_gt_root_stage2_sceneco/val_gen \
  --num_denoising_steps 50 \
  --cfg_weight 2.0 2.0 \
  --gpu 0 \
  2>&1 | tee outputs/e7_gt_root_stage2_sceneco/val_gen/generate_body.log
```

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e7_gt_root_stage2_sceneco/val_gen \
  --output_csv outputs/e7_gt_root_stage2_sceneco/path_metrics.csv \
  --method e7_gt_root_stage2_sceneco
```

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e7_gt_root_stage2_sceneco/val_gen \
  --scene_dir LINGO/dataset/dataset/Scene \
  --output_csv outputs/e7_gt_root_stage2_sceneco/scene_metrics.csv \
  --method e7_gt_root_stage2_sceneco
```

---

# 12. 最终汇总检查

执行：

```bash
find outputs -name "path_metrics.csv" -o -name "scene_metrics.csv"
```

至少应看到：

```text
E1 path/scene metrics
E2 path/scene metrics
E4 path/scene metrics
E5 path/scene metrics
```

---

# 13. 最终对比表模板

把每个 CSV 的结果填入：

| 实验 | PathADE | PathFDE | RootJerk | SpeedStd | CFR | PenetrationRate | FootSlide |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 NoGuidance + Original Body |  |  |  |  |  |  |  |
| E1 EnergyGuidance + Original Body |  |  |  |  |  |  |  |
| E2 ClassifierGuidance + Original Body |  |  |  |  |  |  |  |
| E3 HybridGuidance + Original Body |  |  |  |  |  |  |  |
| E4 EnergyGuidance + Stage2 SceneCo |  |  |  |  |  |  |  |
| E5 ClassifierGuidance + Stage2 SceneCo |  |  |  |  |  |  |  |
| E6 HybridGuidance + Stage2 SceneCo |  |  |  |  |  |  |  |
| E7 GTRoot + Stage2 SceneCo |  |  |  |  |  |  |  |

---

# 14. 成功判断

## RootPathClassifier 成功

```text
val_acc > 0.85
positive_score_mean > negative_score_mean
loss 下降
```

## ClassifierGuidance 成功

```text
score_valid 有意义
loss_cls 不爆炸
grad_norm > 0
PathADE / PathFDE 优于 E0
RootJerk / SpeedStd 不爆炸
```

## Stage2 SceneCo 成功

```text
root fix max error < 1e-5
CollisionFrameRate 下降
PenetrationRate 下降
FootSlide 不爆炸
人物不明显扭曲
```

---

# 15. 给弱 AI 的最终提醒

```text
1. 每个实验先 smoke test。
2. smoke test 失败，不许正式训练。
3. 每个 body 生成后检查 root fix max error。
4. 每个训练前检查 root 文件是否存在。
5. 每个 Stage2 训练前检查 fallback。
6. 每个实验结束立刻跑 metrics。
7. 每一步都更新 COMMAND_LOG.md。
8. 不要跳过检查点。
```