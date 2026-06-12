# 最新实验结果梳理

检查时间：2026-06-11  
可信评估口径：`raw3d_floor_filtered`，即使用 raw 3D 场景占据并过滤地板层后的 CFR / Penetration 指标。  
机器可读汇总：`outputs/latest_results_raw3d_summary.csv`

## 结论

1. E4、E5、E6、E7 的 v3 Stage2 结果现在都有完整 body 输出、`path_metrics.csv`、`scene_metrics.csv`。
2. E4、E6、B1_E7 的 body 生成日志均报告 `Root fix max_error: 0.00e+00 | all_passed: True`，固定 root 正确性通过。
3. E4、E6、B1_E7 的 Stage2 训练日志均确认 `external_root_enabled=True, use_external_root=True`，首个 batch 有 `external_root` 和 `external_root_sources`，且未发现 `fallback` 或 `missing`。
4. E6 是目前 full-v3 非 GT-root 结果中综合最好的：PathADE 最低，PenetrationRate 也最低；E5 和 E6 的 CFR 基本持平。
5. E4 明显偏差更大：PathADE、PathFDE、CFR、PenetrationRate 都高于 E5/E6。
6. E7 是 GT root，因此 PathADE/PathFDE 为 0；但 CFR 和 NonWalkableRootRate 很高，说明数据/场景本身存在不少 GT root 落在非可行区域或占据区域的样本，不能把 E7 的高 CFR 解释为 body 生成单独失败。
7. B1_E7_TrajCo 相比 E7_v3 没有改善场景指标，CFR 反而从 0.335901 升到 0.338230，PenetrationRate 从 0.091078 升到 0.091325。
8. `outputs/current_model_test/e5_v3` 和 `outputs/current_model_test/e7_v3` 的 `scene_metrics.csv` 没有 `MetricMode=raw3d_floor_filtered`，CFR 明显偏高，不应作为当前主报告指标。

## 主结果表

| 实验 | 输出目录 | 样本数 | Body NPZ | PathADE | PathFDE | RootJerk | CFR | NonWalkRoot | PenRate | OutIgnored | 评估口径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 Energy + Original Body | `outputs/e1_energy_guidance_body` | 30 | - | 2.103299 | 3.394884 | 0.000989 | 0.134841 | 0.069764 | 0.063641 | 0.265687 | raw3d_floor_filtered |
| E2 Classifier + Original Body | `outputs/e2_classifier_guidance_body` | 30 | - | 1.064963 | 1.151368 | 0.000039 | 0.002257 | 0.000709 | 0.000135 | 0.149478 | raw3d_floor_filtered |
| E3 Hybrid + Original Body | `outputs/e3_hybrid_guidance_body` | 30 | - | 1.211427 | 1.498404 | 0.000102 | 0.035898 | 0.016680 | 0.015599 | 0.145855 | raw3d_floor_filtered |
| E4 Energy + Stage2 SceneCo | `outputs/e4_v3_stage2` | 1753 | 1753 | 1.870138 | 2.803696 | 0.000910 | 0.211850 | 0.078644 | 0.073230 | 0.226584 | raw3d_floor_filtered |
| E5 Classifier + Stage2 SceneCo | `outputs/e5_v3_stage2` | 1731 | 1731 | 1.370983 | 1.589155 | 0.000121 | 0.138655 | 0.028761 | 0.030975 | 0.155022 | raw3d_floor_filtered |
| E6 Hybrid + Stage2 SceneCo | `outputs/e6_v3_stage2` | 1731 | 1731 | 1.355138 | 1.551671 | 0.000108 | 0.138782 | 0.027296 | 0.029898 | 0.156442 | raw3d_floor_filtered |
| E7 GT Root + Stage2 SceneCo | `outputs/e7_v3_stage2` | 1732 | 1732 | 0.000000 | 0.000000 | 0.001409 | 0.335901 | 0.106412 | 0.091078 | 0.160557 | raw3d_floor_filtered |
| B1_E7 GT Root + Stage2 + TrajCo | `outputs/B1_E7_sceneco_trajco` | 1732 | 1732 | 0.000000 | 0.000000 | 0.001409 | 0.338230 | 0.106412 | 0.091325 | 0.161190 | raw3d_floor_filtered |

## Stage2 有效性证据

E4:

* 训练日志：`outputs/e4_v3_stage2/train.log`
* `external_root_enabled=True, use_external_root=True`
* 首个 batch 有 `external_root: shape=...`
* `external_root_sources`: `path_guided_root`
* `fallback` 计数：0
* `missing` 计数：0
* checkpoint：`outputs/e4_v3_stage2/checkpoints/best_checkpoint.pt`
* 生成日志：`outputs/e4_v3_stage2/val_gen/gen.log`
* root fix：`max_error: 0.00e+00`

E6:

* 训练日志：`outputs/e6_v3_stage2/train.log`
* `external_root_enabled=True, use_external_root=True`
* 首个 batch 有 `external_root: shape=...`
* `external_root_sources`: `path_guided_root`
* `fallback` 计数：0
* `missing` 计数：0
* checkpoint：`outputs/e6_v3_stage2/checkpoints/best_checkpoint.pt`
* 生成日志：`outputs/e6_v3_stage2/val_gen/gen.log`
* root fix：`max_error: 0.00e+00`

B1_E7_TrajCo:

* 训练日志：`outputs/B1_E7_sceneco_trajco/train.log`
* `external_root_enabled=True, use_external_root=True`
* 首个 batch 有 `external_root: shape=...`
* `external_root_sources`: `gt_root`
* `TrajCo: enabled=True root=False body=True`
* `fallback` 计数：0
* `missing` 计数：0
* checkpoint：`outputs/B1_E7_sceneco_trajco/checkpoints/best_checkpoint.pt`
* 生成日志：`outputs/B1_E7_sceneco_trajco/val_gen/gen.log`
* root fix：`max_error: 0.00e+00`

## 指标分析

E4 vs E5/E6:

* E4 的 PathADE 为 1.870138，明显高于 E5 的 1.370983 和 E6 的 1.355138。
* E4 的 CFR 为 0.211850，高于 E5/E6 的约 0.139。
* E4 的 PenRate 为 0.073230，也明显高于 E5/E6 的约 0.030。
* 结论：E4 虽然生成链路有效，但 root 和场景质量明显弱于 classifier/hybrid guidance。

E5 vs E6:

* E6 PathADE 更低：1.355138 vs 1.370983。
* E6 PathFDE 更低：1.551671 vs 1.589155。
* E5 CFR 略低：0.138655 vs 0.138782，差距极小。
* E6 NonWalkRoot 和 PenRate 更低：0.027296 / 0.029898，优于 E5 的 0.028761 / 0.030975。
* 结论：E6 当前可作为 full-v3 非 GT-root 的最佳主结果；E5 是非常接近的对照。

E7 与 B1_E7_TrajCo:

* E7_v3 使用 GT root，因此 PathADE 和 PathFDE 为 0。
* E7_v3 CFR=0.335901，NonWalkRoot=0.106412，说明 GT root 本身在相当一部分样本中与 raw 场景非可行区域冲突。
* B1_E7_TrajCo CFR=0.338230，PenRate=0.091325，均略差于 E7_v3。
* 结论：当前 TrajCo body 注入没有改善 GT-root 场景碰撞表现；E7 的高 CFR 应优先归因于 GT root/场景标注或场景可行区域定义，而不是简单归因于 Stage2 body。

## 不建议继续引用的旧结果

| 目录 | 问题 |
|---|---|
| `outputs/current_model_test/e5_v3` | `scene_metrics.csv` 无 `MetricMode=raw3d_floor_filtered`，CFR=0.279888，疑似旧评估口径 |
| `outputs/current_model_test/e7_v3` | `scene_metrics.csv` 无 `MetricMode=raw3d_floor_filtered`，CFR=0.528705，疑似旧评估口径 |
| `outputs/e7_gt_root_stage2_sceneco` | 样本数 1548，不是当前完整 v3 的 1732 样本 |

## 当前安全下一步

1. 主报告和 PPT 应改用 `outputs/latest_results_raw3d_summary.csv` 与本文件的主结果表。
2. E4/E5/E6/E7/B1_E7 的视频示例需要按最新目录重新挑选，尤其不能再用 GT root 本身出界或撞场景的样本来说明 body 方法失败。
3. 对 E7 高 CFR 的分析应增加 GT root 可行性统计：按 scene/sample 找出 NonWalkableRootRate 高的样本，单独标注为数据或场景可行区域问题。
4. 如果要继续优化方法，优先方向不是 B1_E7_TrajCo，而是让 root guidance 或 classifier guidance 更显式地使用场景可行区域约束。
