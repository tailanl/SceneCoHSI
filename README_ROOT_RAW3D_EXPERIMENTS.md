# Root Raw3D 修正实验代码说明

本次新增代码用于解决 root 侧的三个问题：

1. 目标点或 target path 出界。
2. root guidance / classifier 生成的 root 没有和最新 `raw3d_floor_filtered` 评估口径对齐。
3. root 送入 Stage2 前缺少可行区域投影和修正。

新增文件：

| 文件 | 作用 |
|---|---|
| `kimodo_sceneco/guidance/raw_scene_root.py` | raw LINGO 场景加载、XZ 栅格映射、可行域构建、root 可行性统计、最近 free-space 投影 |
| `scripts/postprocess_root_raw3d.py` | 批量修正 root `.npz`，可选同步更新 `guided_root_5d_norm`，输出 Stage2 可直接读取的 root 文件 |

## 设计原则

这套代码使用和当前可信评估一致的场景定义：

```text
Raw scene: LINGO/dataset/dataset/Scene/{scene}.npy
Voxel size: 0.02 m
Scene shape: (Z, Y, X)
X/Z: centered at 0
Floor ignore: Y < 0.08 m
```

也就是说，后处理看到的障碍/可行区域和 `eval/eval_sceneadapt_metrics.py --metric_mode raw3d` 使用的是同一套坐标和 floor-filtered 逻辑。

## E8：Classifier root + raw3d root correction

先用现有 classifier guidance 生成 root：

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_scene_guidance.yaml \
  --classifier_ckpt outputs/root_path_scene_classifier_sdf/best.pt \
  --output_dir outputs/e8_classifier_raw_root_pre \
  --cache_dir lingo_smplx_cache \
  --split val \
  --all \
  --skip_existing \
  --gpu 0
```

然后用 raw3d 可行域修正 root，并覆盖为 Stage2 可读 schema：

```bash
python scripts/postprocess_root_raw3d.py \
  --input_dir outputs/e8_classifier_raw_root_pre \
  --output_dir outputs/e8_classifier_raw_root \
  --project_target_path \
  --overwrite_root_keys \
  --update_norm \
  --clearance_m 0.04 \
  --smooth_window 5 \
  --gpu 0
```

关键输出：

```text
outputs/e8_classifier_raw_root/*.npz
outputs/e8_classifier_raw_root/raw3d_root_postprocess_summary.csv
```

`*.npz` 中会包含：

```text
guided_root_5d_norm
guided_root_5d_meter
target_path_xz
corrected_root_5d_meter
corrected_guided_root_5d_norm
corrected_target_path_xz
raw3d_root_changed_mask
raw3d_target_changed_mask
```

## E9：Hybrid root + raw3d root correction

先生成 hybrid root：

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_hybrid_guidance.yaml \
  --classifier_ckpt outputs/root_path_scene_classifier_sdf/best.pt \
  --output_dir outputs/e9_hybrid_raw_root_pre \
  --cache_dir lingo_smplx_cache \
  --split val \
  --all \
  --hybrid \
  --skip_existing \
  --gpu 0
```

再修正：

```bash
python scripts/postprocess_root_raw3d.py \
  --input_dir outputs/e9_hybrid_raw_root_pre \
  --output_dir outputs/e9_hybrid_raw_root \
  --project_target_path \
  --overwrite_root_keys \
  --update_norm \
  --clearance_m 0.04 \
  --smooth_window 5 \
  --gpu 0
```

## E10：GT root projected-to-walkable

先导出 GT root：

```bash
python scripts/export_gt_root_for_stage2_v2.py \
  --split val \
  --output_dir outputs/e10_gt_root_raw_pre \
  --gpu 0
```

再把 GT target/root 投影到 raw3d 可行区域：

```bash
python scripts/postprocess_root_raw3d.py \
  --input_dir outputs/e10_gt_root_raw_pre \
  --output_dir outputs/e10_gt_root_projected \
  --project_target_path \
  --overwrite_root_keys \
  --update_norm \
  --clearance_m 0.04 \
  --smooth_window 5 \
  --gpu 0
```

E10 的目的不是证明方法好，而是量化：

```text
E7 高 CFR 中，有多少来自 GT root / target 本身不在 raw3d 可行区域。
```

## Stage2 使用方式

Stage2 训练或生成时，把对应配置里的 external root 目录改成：

```text
E8: outputs/e8_classifier_raw_root
E9: outputs/e9_hybrid_raw_root
E10: outputs/e10_gt_root_projected
```

必须确认 Stage2 日志中出现：

```text
external_root_enabled=True, use_external_root=True
external_root: shape=...
external_root_sources: ['path_guided_root', ...] 或 ['gt_root', ...]
```

同时不能有大量：

```text
fallback
missing
```

## 评估和验收标准

后处理阶段先看：

```text
raw3d_root_postprocess_summary.csv
```

重点列：

```text
root_invalid_before
root_invalid_after
root_occupied_before
root_occupied_after
root_out_of_bounds_before
root_out_of_bounds_after
root_changed_frames
root_max_shift_m
target_invalid_before
target_invalid_after
```

预期：

```text
root_invalid_after 接近 0
target_invalid_after 接近 0
root_max_shift_m 不应普遍过大
```

Stage2 body 生成后仍需重新跑：

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir <stage2_val_gen_dir> \
  --cache_dir lingo_smplx_cache \
  --output_csv <stage2_output_dir>/scene_metrics.csv \
  --metric_mode raw3d
```

有效结果必须同时满足：

1. root 目录 schema 完整。
2. Stage2 日志确认 external_root 生效。
3. body 生成日志确认 `Root fix max_error < 1e-5`。
4. `scene_metrics.csv` 中 `MetricMode=raw3d_floor_filtered`。
5. CFR / NonWalkableRootRate / PenetrationRate 低于对应未修正版本。

## 注意

`--overwrite_root_keys` 会把修正后的 root 写回 Stage2 使用的标准 key。为了避免 normalized root 和 meter root 不一致，输入文件包含 `guided_root_5d_norm` 时必须同时使用：

```text
--update_norm
```

如果只是检查修正效果，不要加 `--overwrite_root_keys`，脚本会保留原 root，只新增 `corrected_*` 字段。
