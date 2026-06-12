# CURRENT_REEVALUATION_REPORT

## 评估设置

本次重新评估使用修正后的 `eval/eval_sceneadapt_metrics.py` 默认模式：

```bash
--metric_mode raw3d
--floor_ignore_height 0.08
```

也就是说，CFR 使用原始 LINGO `Scene/{scene}.npy` 的 3D 坐标和 `0.02m/voxel` 尺度计算，并忽略 `Y < 0.08m` 的地板/脚接触层。旧的 64^3 动态 2D 投影结果不再作为主指标。

## 重新评估结果

| Experiment | Samples | PathADE | PathFDE | CFR | NonWalkableRootRate | PenetrationRate | OutOfScene/FloorIgnored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 EnergyGuidance + Original Body | 30 | 2.103299 | 3.394884 | 0.134841 | 0.069764 | 0.063641 | 0.265687 |
| E2 ClassifierGuidance + Original Body | 30 | 1.064963 | 1.151368 | 0.002257 | 0.000709 | 0.000135 | 0.149478 |
| E3 HybridGuidance + Original Body | 30 | 1.211427 | 1.498404 | 0.035898 | 0.016680 | 0.015599 | 0.145855 |
| E5 ClassifierGuidance + Stage2 SceneCo | 1731 | 1.370983 | 1.589155 | 0.138655 | 0.028761 | 0.030975 | 0.155022 |
| E7 GTRoot + Stage2 SceneCo | 1732 | 0.000000 | 0.000000 | 0.335901 | 0.106412 | 0.091078 | 0.160557 |
| E7 old legacy dir | 1548 | 0.392742 | 0.392059 | 0.330381 | 0.109157 | 0.093047 | 0.158820 |

## 初步结论

1. 旧 CFR 虚高问题已修正：所有新 `scene_metrics.csv` 都带有 `MetricMode=raw3d_floor_filtered`。
2. E2 在 30 个小样本上碰撞最低，CFR=0.002257，但它不是完整 val 规模。
3. E5 在完整 val 上 CFR=0.138655，明显低于 E7 的 CFR=0.335901，但 PathADE=1.370983，路径跟随不如 E7。
4. E7 使用 GT root，PathADE/FDE 为 0 是预期结果；但 body 仍有较高碰撞，说明固定 root 不等于 body-scene 合理。
5. E4/E6 没有可评估的 body generation 输出，本次未纳入正式指标表。

## 输出文件

```text
outputs/reevaluation_raw3d_summary.csv
outputs/topdown_scene_video_report_raw3d/all_sample_metrics.csv
outputs/topdown_scene_video_report_raw3d/model_metrics_summary.csv
outputs/topdown_scene_video_report_raw3d/anomaly_samples.csv
outputs/topdown_scene_video_report_raw3d/PRELIMINARY_ANALYSIS.md
```

已覆盖更新的官方 metrics：

```text
outputs/e1_energy_guidance_body/path_metrics.csv
outputs/e1_energy_guidance_body/scene_metrics.csv
outputs/e2_classifier_guidance_body/path_metrics.csv
outputs/e2_classifier_guidance_body/scene_metrics.csv
outputs/e3_hybrid_guidance_body/path_metrics.csv
outputs/e3_hybrid_guidance_body/scene_metrics.csv
outputs/e5_v3_stage2/path_metrics.csv
outputs/e5_v3_stage2/scene_metrics.csv
outputs/e7_v3_stage2/path_metrics.csv
outputs/e7_v3_stage2/scene_metrics.csv
outputs/e7_gt_root_stage2_sceneco/path_metrics.csv
outputs/e7_gt_root_stage2_sceneco/scene_metrics.csv
```

## 注意

`raw3d` 指标仍然是 22 个关节点对体素占用的采样检查，不是完整 SMPL-X mesh SDF 穿透评估。`SceneSDFPenalty` 在新版 raw3d 下不适用，因此为 `NaN`。后续比较应优先看 `CollisionFrameRate`、`PenetrationRate` 和 GT-normalized excess CFR。

