# README_E4_E7_CODE_FIX

本文档记录本次对 E4/E7 修复方案的代码补齐结果。

## 结论

本次检查发现原修改仍未通过逻辑检查，因此已补齐脚本层面的关键问题。

已修复的是代码和默认路径，不包括重新生成全量 root、不包括重新训练 Stage2、不包括 body generation。

已有旧实验输出仍不能直接视为有效。E4/E7 仍需要用修复后的脚本重新生成 v3 root，并重新跑 Stage2/body/metrics 后，才能判定实验有效。

## 本次发现的问题

### E4

`scripts/generate_root_guidance.py` 存在以下问题：

1. `target_path_xz` 在 external path 分支下可能未初始化。
2. `gt_root_xz` 只在 fallback 分支中生成，导致 external path 模式保存结果时可能未定义。
3. `guided_root_5d_meter` 的 heading 保存逻辑不稳，可能退回到全 0 heading。
4. 脚本使用上一级目录的 cache/LINGO 路径，而 Stage2 和 E7 导出使用项目内 `lingo_smplx_cache`，这会造成 split/source_id 不一致。

### E7

`scripts/export_gt_root_for_stage2_v2.py` 存在以下问题：

1. cache 中的 `motion_features` 已经是 normalized。
2. 脚本又执行了一次 `mr.normalize(feat_t)`，导致 meter-space root 语义错误。
3. `guided_root_5d_meter` 的 heading 使用了 normalized feature 中的 `feat[:, 3:5]`，不是 inverse 后的真实 meter-space heading。

### Metrics/Pipeline

`eval/eval_sceneadapt_metrics.py` 只扫描前 500 个 cache 文件来找 scene voxel，可能导致 E4/E7 后续 scene metrics 大量 NaN。

`scripts/run_experiment_pipeline.sh` 的 E4/E7 默认 root/body 目录仍指向旧输出，不适合修复后的 v3 实验。

## 已修改文件

### `scripts/generate_root_guidance.py`

已修改内容：

1. 统一路径为项目目录：

```text
PROJECT_DIR = kimodo_scene_project
cache = PROJECT_DIR / "lingo_smplx_cache"
LINGO = PROJECT_DIR / "LINGO"
CHECKPOINT_DIR = PROJECT_DIR / "models"
```

2. 每个 sample 开始时固定初始化：

```python
gt_root_xz = extract_gt_root_path(...)
target_path_xz = None
```

3. 新增 `root_5d_meter_from_output(output)`，从 inverse 输出中保存真实：

```text
smooth_root_pos
global_root_heading
```

4. 保存完整 schema：

```text
guided_root_5d_norm
guided_root_5d_meter
target_path_xz
text
scene_name
source_file
```

5. 删除全 0 heading fallback。若 inverse 输出缺少或 shape 不对，会直接报错，不会静默生成错误 root。

### `scripts/export_gt_root_for_stage2_v2.py`

已修改内容：

1. 去掉二次 normalize：

```python
# 修复前
norm_feat = mr.normalize(feat_t)
output = mr.inverse(norm_feat, is_normalized=True, return_numpy=True)

# 修复后
output = mr.inverse(feat_t, is_normalized=True, return_numpy=True)
```

2. `guided_root_5d_meter` 使用 inverse 输出：

```python
root_meter = output["smooth_root_pos"][0]
heading = output["global_root_heading"][0]
guided_root_5d_meter = np.concatenate([root_meter[:, :3], heading], axis=-1)
```

3. 增加 `--start_idx` 和 `--max_samples`，便于小样本检查或分段导出。

4. 保持 cache-based split，与 Stage2 dataset 的 source_id 匹配。

### `eval/eval_sceneadapt_metrics.py`

已修改内容：

1. 不再只扫描前 500 个 cache 文件。
2. 新增完整 scene cache index：

```python
build_scene_cache_index(cache_dir)
```

3. 计算 metrics 时通过 scene index 找对应 cached voxel grid。
4. 自动创建 `output_csv` 的父目录。

### `scripts/run_experiment_pipeline.sh`

已修改内容：

1. E4 root 默认改为：

```text
outputs/e4_energy_guidance_val_v3
```

2. E7 root 默认改为：

```text
outputs/e7_gt_root_v3_val
```

3. E4/E7 body 输出默认改为 v3 目录，避免覆盖旧无效输出：

```text
outputs/e4_energy_stage2_sceneco_v3/val_gen
outputs/e7_gt_root_stage2_sceneco_v3/val_gen
```

4. scene metrics 参数改为：

```text
--cache_dir lingo_smplx_cache
```

## 已执行的轻量检查

### Python 语法检查

通过：

```text
scripts/generate_root_guidance.py
scripts/export_gt_root_for_stage2_v2.py
scripts/patch_e4_roots.py
eval/eval_sceneadapt_metrics.py
```

### Bash 语法检查

通过：

```text
scripts/run_experiment_pipeline.sh
```

### 参数加载检查

通过：

```text
python scripts/export_gt_root_for_stage2_v2.py --help
python scripts/generate_root_guidance.py --help
```

### 关键逻辑检查

通过：

```text
E4 uses project cache
E4 initializes target_path_xz
E4 saves real global_root_heading
E4 has no zero-heading fallback
E7 has no second normalize
E7 inverse decodes normalized cache directly
E7 exposes --max_samples
Metrics builds full scene cache index
Pipeline E4/E7 default root dirs use v3
```

### Split 检查

项目内 `lingo_smplx_cache` 的 Stage2 split 数量为：

```text
train = 15584
val   = 1732
total = 17316
```

这与之前审计要求一致。

## 修复后应如何重新生成 E7

E7 是 GT-root，优先级最高，因为导出成本低于 E4 energy generation。

建议输出到干净 v3 目录：

```bash
python scripts/export_gt_root_for_stage2_v2.py \
  --split train \
  --output_dir outputs/e7_gt_root_v3_train \
  --gpu 0

python scripts/export_gt_root_for_stage2_v2.py \
  --split val \
  --output_dir outputs/e7_gt_root_v3_val \
  --gpu 0
```

生成后必须检查：

```text
outputs/e7_gt_root_v3_train: 15584 files
outputs/e7_gt_root_v3_val: 1732 files
missing_stems = 0
extra_stems = 0
schema_missing_files = 0
bad_shape_files = 0
```

## 修复后应如何重新生成 E4

E4 是 energy-guided root，需要全量生成，成本较高。

建议输出到干净 v3 目录：

```bash
python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --split train \
  --num_samples -1 \
  --output_dir outputs/e4_energy_guidance_train_v3 \
  --gpu 0

python scripts/generate_root_guidance.py \
  --config configs/guidance_root_scene.yaml \
  --split val \
  --num_samples -1 \
  --output_dir outputs/e4_energy_guidance_val_v3 \
  --gpu 0
```

生成后必须检查：

```text
outputs/e4_energy_guidance_train_v3: 15584 files
outputs/e4_energy_guidance_val_v3: 1732 files
missing_stems = 0
extra_stems = 0
schema_missing_files = 0
bad_shape_files = 0
```

## Stage2 重跑要求

E4 Stage2 必须使用：

```text
--path_guided_root_dir outputs/e4_energy_guidance_train_v3
--path_scene_guided_root_dir outputs/e4_energy_guidance_train_v3
--val_root_dir outputs/e4_energy_guidance_val_v3
```

E7 Stage2 必须使用：

```text
--path_guided_root_dir outputs/e7_gt_root_v3_train
--path_scene_guided_root_dir outputs/e7_gt_root_v3_train
--val_root_dir outputs/e7_gt_root_v3_val
--root_mix_gt 0.0
--root_mix_path 1.0
--root_mix_scene 0.0
```

Stage2 日志必须满足：

```text
external_root_enabled=True
use_external_root=True
external_root_sources 使用 path_guided_root 或 path_scene_guided_root
无 fallback
无 missing
loss/val_loss finite
checkpoint 存在
```

如果日志中出现：

```text
path_root_missing_gt_fallback
path_scene_root_missing_gt_fallback
fallback
missing
```

则该 Stage2 实验仍然无效。

## 尚未完成

本次没有执行以下操作：

1. 没有重新生成全量 E4 v3 root。
2. 没有重新生成全量 E7 v3 root。
3. 没有启动新的 Stage2 训练。
4. 没有启动 body generation。
5. 没有生成新的 metrics。

因此当前状态是：

```text
代码补齐已完成并通过轻量检查。
实验 artifact 仍需重新生成和重新审计。
```
