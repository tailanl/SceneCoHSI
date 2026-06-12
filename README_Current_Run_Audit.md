# README：当前 E1–E7 实验运行状态审计手册

> 目标：  
> 本 README 用于让 Codex / 本地 AI **只检查当前实验运行状态**。  
> 本阶段不修改代码、不改配置、不启动新训练、不停止已有任务。  
> 只允许读取仓库、读取日志、检查输出文件，并生成审计报告。

---

## 0. 当前任务定义

你正在 SceneCoHSI 仓库中工作。

当前任务是：

```text
AUDIT ONLY
```

用户说 E1–E7 实验已经跑起来了。你的任务是检查：

```text
1. 哪些实验正在运行；
2. 哪些实验已经产生输出；
3. 哪些实验是有效的；
4. 哪些实验只是“进程在跑”，但实验逻辑不成立；
5. 哪些输出缺失；
6. 哪些 checkpoint / log / metrics 已存在；
7. classifier guidance 是否真的生效；
8. Stage2 是否真的使用 external_root；
9. body generation 是否真的固定 root；
10. 下一步应该做什么。
```

---

## 1. 严格禁止事项

本阶段只做审计。

禁止：

```text
1. 不要修改代码；
2. 不要修改 YAML 配置；
3. 不要启动新的训练；
4. 不要停止已有训练；
5. 不要 kill 任何进程；
6. 不要启动长任务；
7. 不要运行会改变模型权重的命令；
8. 不要运行会重新生成大量实验输出的命令；
9. 不要假设“进程在跑”就等于“实验有效”。
```

只允许创建或更新这三个审计报告文件：

```text
CURRENT_RUN_AUDIT.md
CURRENT_EXPERIMENT_STATUS.md
CURRENT_BLOCKERS.md
```

---

## 2. 核心判断原则

必须使用以下判断原则：

```text
Running is not equal to valid experiment.
```

也就是说：

```text
一个任务正在运行 ≠ 实验成立
一个 .pt 文件存在 ≠ classifier 训练成功
一个 .npz 文件存在 ≠ root 输出格式正确
一个 Stage2 进程存在 ≠ external_root 真的被使用
一个 body 生成完成 ≠ root 真的被固定
```

每个实验必须根据对应检查点判断。

---

# 3. Step 0：基础仓库状态检查

运行：

```bash
pwd
git status --short
git branch --show-current
git log -1 --oneline
```

记录到：

```text
CURRENT_RUN_AUDIT.md
```

必须记录：

```text
1. 当前路径；
2. 当前分支；
3. 最近一次 commit；
4. 当前是否有未提交修改。
```

---

# 4. Step 1：检查当前运行中的进程

运行：

```bash
ps -ef | grep -E "train_root_classifier|generate_root_classifier|generate_root_guidance|train_stage2|generate_body_from_root|python" | grep -v grep
```

运行：

```bash
nvidia-smi
```

记录：

```text
1. 进程命令；
2. PID；
3. 正在运行的脚本；
4. 可能对应的实验编号：E1 / E2 / E3 / E4 / E5 / E6 / E7；
5. GPU 显存占用；
6. 任务看起来是否 active / idle。
```

如果没有相关进程，写：

```text
No active experiment process found.
```

---

# 5. Step 2：查找输出目录和产物

运行：

```bash
find outputs -maxdepth 3 -type d | sort
```

然后运行：

```bash
find outputs -maxdepth 4 -type f \( -name "*.log" -o -name "*.csv" -o -name "*.pt" -o -name "*.npz" \) | sort | head -n 300
```

---

## 5.1 目录到实验的映射规则

按下面规则映射：

| 实验 | 目录 |
|---|---|
| E1 | `outputs/e1_energy_guidance_root` |
| E1 | `outputs/e1_energy_guidance_body` |
| E2 | `outputs/root_path_classifier` |
| E2 | `outputs/root_path_classifier_smoke` |
| E2 | `outputs/e2_classifier_guidance_root` |
| E2 | `outputs/e2_classifier_guidance_body` |
| E3 | `outputs/e3_hybrid_guidance_root` |
| E3 | `outputs/e3_hybrid_guidance_body` |
| E4 | `outputs/e4_energy_guidance_train/path_only` |
| E4 | `outputs/e4_energy_guidance_val/path_only` |
| E4 | `outputs/e4_energy_stage2_sceneco` |
| E4 | `outputs/e4_energy_stage2_sceneco_smoke` |
| E5 | `outputs/e5_classifier_guidance_train/path_only` |
| E5 | `outputs/e5_classifier_guidance_val/path_only` |
| E5 | `outputs/e5_classifier_stage2_sceneco` |
| E5 | `outputs/e5_classifier_stage2_sceneco_smoke` |
| E6 | `outputs/e6_hybrid_guidance_train/path_only` |
| E6 | `outputs/e6_hybrid_guidance_val/path_only` |
| E6 | `outputs/e6_hybrid_stage2_sceneco` |
| E7 | `outputs/e7_gt_root_train` |
| E7 | `outputs/e7_gt_root_val` |
| E7 | `outputs/e7_gt_root_stage2_sceneco` |

如果项目实际使用了不同目录名，可以根据日志和文件内容推断，但必须在报告里写清楚真实路径。

---

# 6. Step 3：只检查 Python / YAML 基础有效性，不修复

运行：

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

然后运行：

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
    print("OK:", p, list(cfg.keys()))
PY
```

重要：

```text
如果检查失败，但已有任务正在运行，不要停止任务。
只记录失败。
不要在审计阶段修复文件。
```

---

# 7. Step 4：检查 RootPathClassifier 训练状态

查找日志：

```bash
find outputs -type f \( -path "*root_path_classifier*/train.log" -o -path "*root_path_classifier*/train_log.csv" \) | sort
```

对每个找到的 log 运行：

```bash
tail -n 120 <LOG_PATH>
```

检查 checkpoint：

```bash
find outputs -path "*root_path_classifier*" -type f \( -name "latest.pt" -o -name "best.pt" -o -name "*.pt" -o -name "train_log.csv" \) -ls
```

---

## 7.1 RootPathClassifier 有效标准

一个有效的 RootPathClassifier 训练必须有：

```text
1. latest.pt；
2. best.pt；
3. train_log.csv；
4. train_loss 被打印；
5. val_loss 被打印；
6. train_acc 或 val_acc 被打印；
7. positive_score_mean 被打印；
8. negative_score_mean 被打印；
9. positive_score_mean > negative_score_mean。
```

如果这些缺失，标记：

```text
E2 classifier training: INVALID 或 INCOMPLETE
```

注意：

```text
不要因为 .pt 文件存在就说 classifier 训练成功。
```

---

# 8. Step 5：检查 EnergyGuidance root generation

查找 EnergyGuidance 输出：

```bash
find outputs -type f -path "*energy*guidance*/*.npz" | head -n 20
```

检查 `.npz` 文件内容：

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

candidates = []
for pattern in [
    "outputs/e1_energy_guidance_root/*.npz",
    "outputs/e4_energy_guidance_train/path_only/*.npz",
    "outputs/e4_energy_guidance_val/path_only/*.npz",
]:
    candidates += sorted(Path(".").glob(pattern))

print("num candidate npz:", len(candidates))

for p in candidates[:5]:
    data = np.load(p, allow_pickle=True)
    print("FILE:", p)
    print("KEYS:", data.files)
    for k in data.files:
        try:
            v = data[k]
            print(" ", k, getattr(v, "shape", None), getattr(v, "dtype", None))
        except Exception as e:
            print(" ", k, "ERROR", e)
PY
```

---

## 8.1 EnergyGuidance root 文件有效标准

一个适合 Stage2 使用的 root 文件最好包含：

```text
guided_root_5d_norm
guided_root_5d_meter
target_path_xz
text
scene_name
source_file
```

如果只包含旧字段，例如：

```text
gen_root
gt_root_xz
gen_joints
```

记录：

```text
EnergyGuidance generated outputs, but output format may not match Stage2 root loader.
```

---

# 9. Step 6：检查 ClassifierGuidance root generation

查找 classifier guidance 日志：

```bash
find outputs -type f -path "*classifier*guidance*/generate*.log" | sort
```

对每个日志运行：

```bash
tail -n 120 <LOG_PATH>
```

搜索必要 marker：

```bash
grep -R "loss_cls\|score_valid\|grad_norm" outputs/*classifier* 2>/dev/null | head -n 100
```

---

## 9.1 ClassifierGuidance 有效标准

有效的 ClassifierGuidance generation 必须在日志中出现：

```text
loss_cls
score_valid
grad_norm
```

如果缺失，标记：

```text
ClassifierGuidance: NOT VERIFIED
```

---

## 9.2 检查 classifier guidance 输出 `.npz`

运行：

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

candidates = []
for pattern in [
    "outputs/e2_classifier_guidance_root/*.npz",
    "outputs/e5_classifier_guidance_train/path_only/*.npz",
    "outputs/e5_classifier_guidance_val/path_only/*.npz",
    "outputs/e3_hybrid_guidance_root/*.npz",
    "outputs/e6_hybrid_guidance_train/path_only/*.npz",
    "outputs/e6_hybrid_guidance_val/path_only/*.npz",
]:
    candidates += sorted(Path(".").glob(pattern))

print("num candidate npz:", len(candidates))

for p in candidates[:10]:
    data = np.load(p, allow_pickle=True)
    print("FILE:", p)
    print("KEYS:", data.files)
    for k in data.files:
        try:
            v = data[k]
            print(" ", k, getattr(v, "shape", None), getattr(v, "dtype", None))
        except Exception as e:
            print(" ", k, "ERROR", e)
PY
```

检查是否有：

```text
guided_root_5d_norm
guided_root_5d_meter
target_path_xz
```

如果缺失，记录为：

```text
output-format blocker
```

---

# 10. Step 7：检查 Stage2 training

查找 Stage2 日志：

```bash
find outputs -type f \( -path "*stage2*/train.log" -o -path "*stage2*/logs/*.log" \) | sort
```

对每个 Stage2 log 运行：

```bash
tail -n 150 <LOG_PATH>
```

搜索 external root 和 fallback：

```bash
grep -Ri "external_root" outputs/*stage2* 2>/dev/null | head -n 100
grep -Ri "fallback" outputs/*stage2* 2>/dev/null | head -n 100
grep -Ri "missing" outputs/*stage2* 2>/dev/null | head -n 100
```

查找 checkpoint：

```bash
find outputs -path "*stage2*" -type f \( -name "*.pt" -o -name "*.ckpt" -o -name "best_checkpoint.pt" -o -name "latest_checkpoint.pt" \) -ls
```

---

## 10.1 Stage2 Root-Guided SceneCo 有效标准

有效 Stage2 训练必须满足：

```text
1. external_root enabled；
2. use_external_root true；
3. root source 从预期目录读取；
4. 没有大量 fallback；
5. loss 是 finite；
6. checkpoint 被保存。
```

如果 Stage2 正在跑，但有大量 fallback，写：

```text
Stage2 process is running, but experiment is not valid because external root files are not matching dataset sample ids.
```

---

# 11. Step 8：检查 body generation

查找 body generation 日志：

```bash
find outputs -type f -path "*body*/generate*.log" -o -path "*val_gen*/generate*.log" | sort
```

对每个 body generation log 运行：

```bash
tail -n 120 <LOG_PATH>
```

搜索 root fix：

```bash
grep -Ri "root fix\|max error\|max_error" outputs 2>/dev/null | head -n 100
```

---

## 11.1 Fixed-root body generation 有效标准

有效的 fixed-root body generation 必须报告：

```text
root fix max error
```

并且数值应满足：

```text
root fix max error < 1e-5
```

如果 root fix error 没有打印，写：

```text
Body generation ran, but fixed-root correctness is not verified because root fix max error is missing from log.
```

---

# 12. Step 9：检查 metrics

查找 metrics：

```bash
find outputs -type f \( -name "path_metrics.csv" -o -name "scene_metrics.csv" \) | sort
```

打印前几行：

```bash
for f in $(find outputs -type f \( -name "path_metrics.csv" -o -name "scene_metrics.csv" \) | sort); do
  echo "===== $f ====="
  head -n 5 "$f"
done
```

记录：

```text
1. 哪些实验有 path_metrics.csv；
2. 哪些实验有 scene_metrics.csv；
3. 哪些生成完成但没有评估；
4. 哪些 metrics 文件为空或格式异常。
```

如果某个实验有生成输出但没有 metrics，标记：

```text
INCOMPLETE
```

---

# 13. Step 10：生成实验状态总表

创建或更新：

```text
CURRENT_EXPERIMENT_STATUS.md
```

必须包含下表：

```markdown
| Experiment | Process Running | Root Generated | Body Generated | Classifier Active | Stage2 external_root Active | Root Fix Verified | Metrics Exists | Status | Blocker |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| E0 NoGuidance + Original Body |  |  |  | N/A | N/A | N/A |  |  |  |
| E1 EnergyGuidance + Original Body |  |  |  | N/A | N/A |  |  |  |  |
| E2 ClassifierGuidance + Original Body |  |  |  |  | N/A |  |  |  |  |
| E3 HybridGuidance + Original Body |  |  |  |  | N/A |  |  |  |  |
| E4 EnergyGuidance + Stage2 SceneCo |  |  |  | N/A |  |  |  |  |  |
| E5 ClassifierGuidance + Stage2 SceneCo |  |  |  |  |  |  |  |  |  |
| E6 HybridGuidance + Stage2 SceneCo |  |  |  |  |  |  |  |  |  |
| E7 GTRoot + Stage2 SceneCo |  |  |  | N/A |  |  |  |  |  |
```

---

## 13.1 状态标签

只能使用以下状态标签：

```text
VALID_RUNNING
VALID_DONE
RUNNING_NOT_VERIFIED
INCOMPLETE
BLOCKED
INVALID
NOT_STARTED
```

含义：

| 状态 | 含义 |
|---|---|
| `VALID_RUNNING` | 正在运行，且当前日志显示关键逻辑有效 |
| `VALID_DONE` | 已完成，输出、日志、metrics 均有效 |
| `RUNNING_NOT_VERIFIED` | 在运行，但缺少关键验证 marker |
| `INCOMPLETE` | 有部分输出，但缺少后续必要产物 |
| `BLOCKED` | 被缺失脚本、缺失 checkpoint、格式不匹配等阻塞 |
| `INVALID` | 运行过，但关键逻辑错误，例如 classifier 未生效或 Stage2 大量 fallback |
| `NOT_STARTED` | 没有发现进程或输出 |

---

# 14. Step 11：生成最终审计总结

创建或更新：

```text
CURRENT_RUN_AUDIT.md
```

必须包含以下章节：

```text
1. Active jobs
2. Git status
3. Python/YAML check results
4. Output directories found
5. E1–E7 experiment status
6. Classifier training status
7. Classifier guidance status
8. EnergyGuidance status
9. Stage2 external_root status
10. Body root-fix status
11. Metrics status
12. Critical blockers
13. Safe next action
```

---

# 15. Step 12：生成阻塞项文件

创建或更新：

```text
CURRENT_BLOCKERS.md
```

格式：

```markdown
# CURRENT_BLOCKERS

## Blocker 1

- Experiment:
- Severity: HIGH / MEDIUM / LOW
- Evidence:
- Log path:
- Required fix:
- Can continue other experiments: YES / NO

## Blocker 2

...
```

---

# 16. 最终判断规则

不能写：

```text
training succeeded
```

除非：

```text
1. 正确输出文件存在；
2. 日志包含要求的 marker；
3. 如果应该评估，则 metrics 存在。
```

不能写：

```text
classifier guidance works
```

除非：

```text
1. loss_cls 存在；
2. score_valid 存在；
3. grad_norm 存在；
4. classifier checkpoint 存在；
5. generated root .npz 存在。
```

不能写：

```text
Stage2 root-guided works
```

除非：

```text
1. external_root 在日志中被确认；
2. fallback 缺失或很少；
3. checkpoint 存在；
4. body generation root fix 被验证。
```

---

# 17. 最终交付物

本审计阶段最终只交付三个文件：

```text
CURRENT_RUN_AUDIT.md
CURRENT_EXPERIMENT_STATUS.md
CURRENT_BLOCKERS.md
```

不要交付修改后的代码。  
不要交付新的训练输出。  
不要交付新的 checkpoint。

---

# 18. 开始执行

现在开始执行审计。

第一条命令：

```bash
pwd
git status --short
git branch --show-current
git log -1 --oneline
```
