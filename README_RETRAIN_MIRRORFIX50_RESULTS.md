# retrain_mirrorfix50 实验配置与指标汇总

更新时间：2026-06-12

本文档汇总 `outputs/retrain_mirrorfix50` 这一批 LINGO / Kimodo / SceneCo 实验的配置、指标和可视化位置。

## 数据与评估口径

- 数据版本：修复 mirror 数据问题，并修复 raw-scene 轴向处理问题后的数据。
- 主要输出目录：`outputs/retrain_mirrorfix50`
- 可视化整理目录：`outputs/retrain_mirrorfix50/visualization_results`
- `Pene` 定义：报告中 `Pene = PenetrationMean`。
- 30-sample：之前用于快速检查的 30 个测试样本。
- 300-sample 1h：按 1 小时预算生成的 300 个测试样本预测。

## 实验配置

| 实验 | Root 来源 | Body 模型 | SceneCo | raw3d 投影 | 说明 |
|---|---|---|---|---|---|
| E1 | energy guidance | 原版 Kimodo body | 否 | 否 | no-train baseline |
| E2 | classifier guidance | 原版 Kimodo body | 否 | 否 | no-train baseline |
| E3 | energy + classifier hybrid | 原版 Kimodo body | 否 | 否 | no-train baseline |
| E4 | energy guidance | retrain Stage2 SceneCo body | 是 | 否 | root guidance + SceneCo |
| E5 | classifier guidance | retrain Stage2 SceneCo body | 是 | 否 | root guidance + SceneCo |
| E6 | energy + classifier hybrid | retrain Stage2 SceneCo body | 是 | 否 | root guidance + SceneCo |
| E7 | GT root | retrain Stage2 SceneCo body | 是 | 否 | oracle root 上限 |
| E8 | E5 classifier root | retrain Stage2 SceneCo body | 是 | 是 | classifier root 投影到 raw3d walkable 区域 |
| E9 | E6 hybrid root | retrain Stage2 SceneCo body | 是 | 是 | hybrid root 投影到 raw3d walkable 区域 |
| E10 | GT root | retrain Stage2 SceneCo body | 是 | 是 | GT root 投影版本，用于检查投影影响 |

## 最新 checkpoint / epoch50 状态

`latest_ckpt_eval_1h` 的 checkpoint 选择规则是：若存在 `epoch_0050.pt` 则使用它，否则使用 `best_checkpoint.pt`。

| 实验 | 300-sample 预测完成度 | checkpoint |
|---|---:|---|
| E4 | 300/300 | `best_checkpoint.pt`，当前没有 `epoch_0050.pt` |
| E5 | 300/300 | `epoch_0050.pt` |
| E6 | 300/300 | `epoch_0050.pt` |
| E7 | 300/300 | `epoch_0050.pt` |
| E8 | 300/300 | `epoch_0050.pt` |
| E9 | 300/300 | `epoch_0050.pt` |
| E10 | 300/300 | `epoch_0050.pt` |

## 30-sample 指标

| 实验 | PathADE | PathFDE | CFR | NonWalkableRootRate | PenetrationRate | Pene |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 2.1033 | 3.3949 | 0.3295 | 0.2087 | 0.1168 | 0.5333 |
| E2 | 1.0650 | 1.1514 | 0.0317 | 0.0080 | 0.0096 | 0.0667 |
| E3 | 1.2114 | 1.4984 | 0.0613 | 0.0158 | 0.0203 | 0.2000 |
| E4 | 1.9948 | 3.1784 | 0.3476 | 0.2027 | 0.1436 | 0.6000 |
| E5 | 1.1777 | 1.5414 | 0.2090 | 0.0253 | 0.0330 | 0.4000 |
| E6 | 1.1682 | 1.4510 | 0.1504 | 0.0288 | 0.0195 | 0.3000 |
| E7 | 0.0000 | 0.0000 | 0.0866 | 0.0000 | 0.0131 | 0.3333 |
| E8 | 0.8875 | 1.0691 | 0.3587 | 0.2667 | 0.1394 | 0.5333 |
| E9 | 0.8580 | 0.9500 | 0.3515 | 0.2667 | 0.1384 | 0.4333 |
| E10 | 0.0008 | 0.0000 | 0.3835 | 0.2667 | 0.1367 | 0.6000 |

30-sample 指标文件：

- E1-E3：`outputs/retrain_mirrorfix50/visualization_results/metrics/no_train/summary_metrics.csv`
- E4-E10：`outputs/retrain_mirrorfix50/visualization_results/metrics/latest_30sample/latest_metrics_summary.csv`

## 300-sample 1h 指标

| 实验 | PathADE | PathFDE | CFR | NonWalkableRootRate | PenetrationRate | Pene |
|---|---:|---:|---:|---:|---:|---:|
| E4 | 1.9956 | 3.1516 | 0.5577 | 0.2444 | 0.1492 | 0.7933 |
| E5 | 1.3465 | 1.6128 | 0.3840 | 0.0435 | 0.0497 | 0.5500 |
| E6 | 1.3168 | 1.5283 | 0.3613 | 0.0292 | 0.0457 | 0.5267 |
| E7 | 0.0000 | 0.0000 | 0.1360 | 0.0101 | 0.0156 | 0.3700 |
| E8 | 1.2707 | 1.4517 | 0.2697 | 0.0567 | 0.0434 | 0.5233 |
| E9 | 1.2345 | 1.3784 | 0.2566 | 0.0567 | 0.0413 | 0.4533 |
| E10 | 0.0005 | 0.0000 | 0.1540 | 0.0567 | 0.0325 | 0.3767 |

300-sample 指标文件：

- `outputs/retrain_mirrorfix50/visualization_results/metrics/latest_300sample_1h/latest_metrics_summary.csv`

## 可视化结果

统一入口：

- `outputs/retrain_mirrorfix50/visualization_results`

主要视频目录：

- `videos/no_train_E1_E3`: E1-E3 原版 Kimodo baseline，每个实验 1 个视频。
- `videos/latest_30sample_E4_E10`: 30-sample E4-E10，每个实验 1 个视频。
- `videos/latest_300sample_1h_E4_E10`: 300-sample E4-E10，每个实验 1 个视频。
- `videos/latest_epoch50_300sample_extra`: 从 300-sample 预测中额外渲染的视频，每个实验 3 个，共 21 个。

报告文件：

- `outputs/retrain_mirrorfix50/visualization_results/reports/lingo_root_guidance_30sample_report.pptx`
- `outputs/retrain_mirrorfix50/visualization_results/reports/lingo_root_guidance_30sample_report.pdf`

## 简要结论

- 原版 Kimodo baseline 中，E2 在 30-sample 上碰撞和穿透指标最好。
- Root guidance + SceneCo 中，E7 作为 GT root oracle 上限表现最好。
- Learned-root 的 SceneCo 设置里，E6 在 30-sample 上优于 E4/E5；300-sample 下 E8/E9/E10 的 raw3d 投影版本降低了 NonWalkableRootRate，但仍需结合视频判断轨迹是否贴边或绕行。
- E4 当前没有 epoch50 checkpoint，因此最新 300-sample 预测仍使用 best checkpoint；E5-E10 已按 epoch50 预测。

