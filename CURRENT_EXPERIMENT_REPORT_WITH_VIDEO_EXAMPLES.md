# CURRENT_EXPERIMENT_REPORT_WITH_VIDEO_EXAMPLES

> 2026-06-11 update: this report is partially superseded. E4 and E6 now have full v3 Stage2 body outputs and raw3d metrics. Use `LATEST_RESULTS_SUMMARY.md` and `outputs/latest_results_raw3d_summary.csv` as the current authoritative metric summary. The video examples in this file were generated before the latest E4/E6/B1_E7 outputs were added.

## 1. 实验配置

| ID | Root source | Body generator | Stage2 SceneCo | Evaluation samples | Output dir | Status |
| --- | --- | --- | --- | ---: | --- | --- |
| E1 | EnergyGuidance root | Original Kimodo body | No | 30 | `outputs/e1_energy_guidance_body` | Evaluated |
| E2 | ClassifierGuidance root | Original Kimodo body | No | 30 | `outputs/e2_classifier_guidance_body` | Evaluated |
| E3 | HybridGuidance root | Original Kimodo body | No | 30 | `outputs/e3_hybrid_guidance_body` | Evaluated |
| E4 | EnergyGuidance root | Stage2 SceneCo body | Yes | 0 body outputs | `outputs/e4_v3_stage2` | Not evaluated: missing body generation |
| E5 | ClassifierGuidance root | Stage2 SceneCo body | Yes | 1731 | `outputs/e5_v3_stage2` | Evaluated |
| E6 | HybridGuidance root | Stage2 SceneCo body | Yes | 0 body outputs | `outputs/e6_v3_stage2` | Not evaluated: missing body generation |
| E7 | Ground-truth root | Stage2 SceneCo body | Yes | 1732 | `outputs/e7_v3_stage2` | Evaluated |

Evaluation protocol:

```text
Scene metric mode: raw3d_floor_filtered
Raw scene source: LINGO/dataset/dataset/Scene/{scene}.npy
Scene voxel size: 0.02 m
Floor/contact ignore threshold: Y < 0.08 m
Main scene metrics: CollisionFrameRate, NonWalkableRootRate, PenetrationRate
```

The old 64^3 dynamic 2D projection CFR is not used as the main metric because GT motions also produced high CFR under that proxy.

## 2. 指标汇总

| Experiment | Samples | PathADE | PathFDE | CFR | NonWalkableRootRate | PenetrationRate | OutOfScene/FloorIgnored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 EnergyGuidance + Original Body | 30 | 2.103299 | 3.394884 | 0.134841 | 0.069764 | 0.063641 | 0.265687 |
| E2 ClassifierGuidance + Original Body | 30 | 1.064963 | 1.151368 | 0.002257 | 0.000709 | 0.000135 | 0.149478 |
| E3 HybridGuidance + Original Body | 30 | 1.211427 | 1.498404 | 0.035898 | 0.016680 | 0.015599 | 0.145855 |
| E5 ClassifierGuidance + Stage2 SceneCo | 1731 | 1.370983 | 1.589155 | 0.138655 | 0.028761 | 0.030975 | 0.155022 |
| E7 GTRoot + Stage2 SceneCo | 1732 | 0.000000 | 0.000000 | 0.335901 | 0.106412 | 0.091078 | 0.160557 |

## 3. 指标分析

Path following:

* E7 is the upper-bound root control case, so PathADE and PathFDE are 0.
* Among non-GT roots, E2 has the best path following in the small 30-sample set: PathADE=1.064963.
* E1 has the worst path following: PathADE=2.103299 and PathFDE=3.394884.
* E5 is full validation scale and uses classifier-guided roots with Stage2 body generation. Its PathADE=1.370983, worse than E2's small-sample result but evaluated on many more samples.

Scene collision:

* E2 has the lowest CFR in the 30-sample group: CFR=0.002257.
* E3 is also low on the 30-sample group: CFR=0.035898.
* E5 has CFR=0.138655 on 1731 validation samples, lower than E7's CFR=0.335901.
* E7 proves that perfect root tracking does not guarantee body-scene validity. The body can still collide even when the root is fixed to GT.

Important caveat:

* E1/E2/E3 are small 30-sample experiments.
* E5/E7 are full validation-scale experiments.
* E2/E3 should not be treated as final winners until they are run at full validation scale.

## 4. 视频案例

E1-E5 examples are selected from high-anomaly cases. E7 is selected as a clean GT-root success case because the original anomaly-selected E7 sample (`seg_02081`) had GT/root itself on top-down occupied space and should not be used as a normal E7 example.

Visualization legend:

* Black dashed line: target path.
* Black circle/star: target start/end.
* Blue line: generated root path.
* Green circle/red X: generated start/end.
* If the red X is outside the room, the generated endpoint is out of bounds; it is not the target point.

| ID | Sample | Scene | Text | PathADE | CFR | PenRate | Video |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| E1 | sample_003 | 5 | walk back left | 4.561216 | 0.285714 | 0.163836 | `outputs/experiment_report_raw3d_with_videos/videos/E1_sample_003_topdown.mp4` |
| E2 | seg_00006 | 5 | turn right | 1.561380 | 0.060606 | 0.002755 | `outputs/experiment_report_raw3d_with_videos/videos/E2_seg_00006_topdown.mp4` |
| E3 | seg_00012 | 5 | walk forward | 2.348529 | 0.274194 | 0.129032 | `outputs/experiment_report_raw3d_with_videos/videos/E3_seg_00012_topdown.mp4` |
| E5 | seg_03542 | 42 | turn left | 2.471167 | 1.000000 | 0.409091 | `outputs/experiment_report_raw3d_with_videos/videos/E5_seg_03542_topdown.mp4` |
| E7 | seg_02317 | 30 | walk forward | 0.000000 | 0.000000 | 0.000000 | `outputs/experiment_report_raw3d_with_videos/videos/E7_seg_02317_topdown.mp4` |

Recommended interpretation:

* E1 sample_003: path failure and moderate collision, useful as a bad energy-guidance example.
* E2 seg_00006: classifier guidance has low penetration even in an anomaly-selected sample.
* E3 seg_00012: hybrid improves over E1 on average but can still produce collision-heavy cases.
* E5 seg_03542: full-val Stage2 example where classifier-root + SceneCo still fails badly.
* E7 seg_02317: clean GT-root success case with PathADE=0 and CFR=0.
* E7 seg_02081 is excluded from the main examples because the GT/generated root path itself lies on top-down occupied space; it is a dataset/scene-root validity warning, not a clean body-generation example.

Rejected E7 anomaly example:

```text
sample: seg_02081
reason: GT/generated root lies on occupied top-down scene area
top-down root occupied rate: 0.903226
raw3d NonWalkableRootRate: 0.870968
raw3d CFR: 1.000000
```

## 5. Generated Report Assets

```text
outputs/experiment_report_raw3d_with_videos/
  all_sample_metrics.csv
  model_metrics_summary.csv
  anomaly_samples.csv
  figures/model_metrics_summary.png
  videos/*.mp4
  thumbnails/*.png

outputs/experiment_report_ppt/SceneCoHSI_Experiment_Report_With_Videos.pptx
```

## 6. Suggested Next Steps

1. Run E2/E3-style classifier/hybrid guidance on the full validation split for fair comparison against E5/E7.
2. Generate body outputs and metrics for E4 and E6 before making final E1-E7 conclusions.
3. Add GT-normalized excess CFR to the official report table, because raw3d GT still has non-zero collision baseline on some samples.
4. For final paper-grade collision metrics, replace joint-point occupancy checks with SMPL-X mesh-to-scene SDF evaluation.
