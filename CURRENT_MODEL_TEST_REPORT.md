# CURRENT_MODEL_TEST_REPORT

Generated: 2026-06-11 09:31 Asia/Shanghai

## Scope

Read `EXPERIMENT_PROGRESS.md`, then tested the models with completed usable outputs:

- E5 v3: `outputs/e5_v3_stage2/checkpoints/best_checkpoint.pt` with completed `val_gen`.
- E7 v3: `outputs/e7_v3_stage2/checkpoints/best_checkpoint.pt`; body generation completed during this check.
- Scene-aware RootPath classifier: `outputs/root_path_scene_classifier_sdf/best.pt`.

Still running and not evaluated as final:

- E4 v3 Stage2 training.
- E6 v3 Stage2 training.
- B1-E7 TrajCo training.
- A2 scene-aware root generation.

## Commands Run

```bash
python eval/eval_path_metrics.py \
  --pred_dir outputs/e5_v3_stage2/val_gen \
  --output_csv outputs/current_model_test/e5_v3/path_metrics.csv \
  --method e5_v3_classifier_stage2

python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e5_v3_stage2/val_gen \
  --cache_dir lingo_smplx_cache \
  --output_csv outputs/current_model_test/e5_v3/scene_metrics.csv \
  --method e5_v3_classifier_stage2

python eval/eval_path_metrics.py \
  --pred_dir outputs/e7_v3_stage2/val_gen \
  --output_csv outputs/current_model_test/e7_v3/path_metrics.csv \
  --method e7_v3_gt_stage2

python eval/eval_sceneadapt_metrics.py \
  --pred_dir outputs/e7_v3_stage2/val_gen \
  --cache_dir lingo_smplx_cache \
  --output_csv outputs/current_model_test/e7_v3/scene_metrics.csv \
  --method e7_v3_gt_stage2
```

An additional independent E7 root-fix check compared:

- input root: `outputs/e7_gt_root_v3_val/*.npz`, key `guided_root_5d_meter[:, :3]`
- generated body root: `outputs/e7_v3_stage2/val_gen/*.npz`, key `gen_root`

## Results

| Model | Samples | PathADE | PathFDE | CollisionFrameRate | NonWalkableRootRate | PenetrationRate | SceneSDFPenalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E5 v3 Classifier + Stage2 | 1731 | 1.370983 | 1.589155 | 0.279888 | 0.049589 | 0.064710 | 0.013867 |
| E7 v3 GT + Stage2 | 1732 | 0.000000 | 0.000000 | 0.528705 | 0.032253 | 0.096336 | 0.001577 |

E7 root-fix check:

```json
{
  "num_body_files": 1732,
  "num_checked": 1732,
  "num_missing_root": 0,
  "max_root_meter_error": 0.0,
  "mean_root_meter_error": 0.0,
  "all_less_than_1e-5": true
}
```

Scene-aware classifier status:

- `train_log.csv` rows: 100
- last `train_loss`: 0.0004276327
- last `val_loss`: 0.0001085726
- last `val_acc`: 1.0
- last `positive_score_mean`: 0.9999356270
- last `negative_score_mean`: 0.0001505557
- A2 generation is still running; `guidance_log.csv` and `scene_collision_log.csv` are not available until that process exits.

## Outputs

Metrics:

- `outputs/current_model_test/e5_v3/path_metrics.csv`
- `outputs/current_model_test/e5_v3/scene_metrics.csv`
- `outputs/current_model_test/e7_v3/path_metrics.csv`
- `outputs/current_model_test/e7_v3/scene_metrics.csv`
- `outputs/current_model_test/e7_v3/root_fix_meter_check.json`
- `outputs/current_model_test/summary.json`

Visualizations:

- `outputs/current_model_test/figures/e5_e7_mean_metrics_comparison.png`
- `outputs/current_model_test/figures/e5_e7_root_trajectory_comparison.png`
- `outputs/current_model_test/figures/e5_mean_metrics.png`
- `outputs/current_model_test/figures/e5_path_metrics_distribution.png`
- `outputs/current_model_test/figures/e5_scene_metrics_distribution.png`
- `outputs/current_model_test/figures/e5_root_trajectory_samples.png`
- `outputs/current_model_test/figures/scene_classifier_root_samples_partial.png`

## Notes

E7 has perfect path metrics because it uses GT root. Its body collision metrics are not automatically better than E5; scene collision is a body-generation issue, not a root-path issue.

A2 currently has partial `.npz` outputs only. Do not report scene-aware classifier guidance as fully verified until `guidance_log.csv`, `scene_collision_log.csv`, and full root output count are present.
