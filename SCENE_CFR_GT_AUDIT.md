# SCENE_CFR_GT_AUDIT

## Conclusion

The old CFR should not be used as an absolute collision rate.

Using ground-truth LINGO motions, the old 64^3 dynamic 2D projection gives
GT CFR around 0.59 on E5/E7 validation samples. That is too high for a
metric intended to measure generated-motion collision.

The likely cause is not only "the floor" in a narrow sense. The raw LINGO
scene has the floor layer encoded as occupied (`y=0` is all true), and the
old evaluator also projected 3D occupancy into a 2D obstacle map and ignored
joint height. This makes floor/low geometry/walls contaminate CFR.

## GT Checks

Spot checks:

| sample | scene | old GT CFR | raw 3D floor-filtered GT CFR |
| --- | --- | ---: | ---: |
| seg_00013 | 005 | 1.0000 | 0.0000 |
| seg_02317 | 030 | 1.0000 | 0.0000 |
| seg_04074 | 046 | 0.6066 | 0.5082 |

Full E5/E7 validation comparison:

| model | samples | old GT CFR | old generated CFR | raw3D GT CFR | raw3D generated CFR | raw3D excess CFR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E5 | 1731 | 0.590810 | 0.279888 | 0.326910 | 0.138655 | 0.073949 |
| E7 | 1732 | 0.590720 | 0.528705 | 0.327298 | 0.335901 | 0.050693 |

The corrected raw3D metric still has a non-zero GT baseline, so final analysis
should report both absolute raw3D CFR and GT-normalized excess CFR.

## Code Change

`eval/eval_sceneadapt_metrics.py` now defaults to:

```bash
--metric_mode raw3d
--floor_ignore_height 0.08
--scene_dir LINGO/dataset/dataset/Scene
```

This mode uses the original LINGO `Scene/{scene}.npy` grid with the documented
0.02m voxel size and ignores the floor/contact layer below 0.08m.

The old behavior is still available:

```bash
python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e7_v3_stage2/val_gen \
  --output_csv /tmp/e7_legacy.csv \
  --method e7_legacy \
  --metric_mode legacy2d
```

## New Audit Outputs

```text
outputs/scene_cfr_gt_audit/e5_e7_gt_raw3d_cfr_comparison.csv
outputs/scene_cfr_gt_audit/summary.csv
outputs/scene_cfr_gt_audit/e5_v3_scene_metrics_raw3d.csv
outputs/scene_cfr_gt_audit/e7_v3_scene_metrics_raw3d.csv
```

## Safe Next Action

Recompute `scene_metrics.csv` for every completed experiment with the corrected
default evaluator. Treat all older `scene_metrics.csv` CFR values as legacy
proxy values unless the CSV contains `MetricMode=raw3d_floor_filtered`.

