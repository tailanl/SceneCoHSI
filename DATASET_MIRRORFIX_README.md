# LINGO Dataset Repair Notes

This note documents the dataset-side repairs applied before rerunning the E-series experiments.

## Scope

The fixes affect local LINGO-derived caches and generated root-conditioning NPZ files under this project. They do not modify the original upstream LINGO repository.

Primary repaired data locations:

- `lingo_smplx_cache`
- `lingo_root_trajectory_smplx`
- `outputs/e4_energy_guidance_train/path_only`
- `outputs/e4_energy_guidance_val/path_only`
- `outputs/e5_classifier_guidance_train/path_only`
- `outputs/e5_classifier_guidance_val/path_only`
- `outputs/e6_hybrid_guidance_train/path_only`
- `outputs/e6_hybrid_guidance_val/path_only`
- `outputs/e7_gt_root_v3_train`
- `outputs/e7_gt_root_v3_val`
- `outputs/e8_classifier_raw3d_train`
- `outputs/e8_classifier_raw3d_val`
- `outputs/e9_hybrid_raw3d_train`
- `outputs/e9_hybrid_raw3d_val`
- `outputs/e10_gt_projected_train`
- `outputs/e10_gt_projected_val`

## Fix 1: Raw Scene Axis Convention

LINGO raw scene occupancy grids are stored as `(X, Y, Z)`, matching the official `lingo-release` dataset code:

- grid shape `(300, 100, 400)` maps to `X, Y, Z`
- occupancy lookup is `occ[ix, iy, iz]`
- train grid bounds are `[-3, 0, -4, 3, 2, 4, 300, 100, 400]`

The local raw-scene helper previously treated raw scene arrays as `(Z, Y, X)`, which transposed top-down occupancy. This affected root projection and raw3d walkability checks.

Corrected local behavior:

- `kimodo_sceneco/guidance/raw_scene_root.py` now treats raw scenes as `(X, Y, Z)`.
- top-down occupancy is computed as `raw_scene[:, y_start:, :].any(axis=1)`.
- scene metadata uses `width=shape[0]`, `height=shape[1]`, `depth=shape[2]`.

Validation performed:

- scene `005` raw shape: `(300, 100, 400)`
- top-down occupancy shape: `(300, 400)`
- sampled cells matched direct checks from `raw[ix, y_start:, iz].any()`

## Fix 2: Mirror Scene Name Repair

Many mirrored cache entries were incorrectly labeled as `005_mirror`. This happened because mirrored samples were not paired back to their source non-mirrored cache segment when writing `scene_name`.

Correct behavior:

- each mirrored segment is paired with the corresponding non-mirrored segment in the valid cache ordering
- the mirrored scene name becomes `<original_scene>_mirror`
- example: a mirrored segment from scene `043` becomes `043_mirror`, not `005_mirror`

Repair script:

- `scripts/fix_lingo_mirror_data.py`

Reports:

- `outputs/mirror_fix_cache_report.csv`
- `outputs/mirror_fix_root_report.csv`
- `outputs/mirror_fix_e4_report.csv`

Repair counts observed:

- `lingo_smplx_cache`: 8542 files repaired
- `lingo_root_trajectory_smplx`: 8542 files repaired
- E4 energy root train/val: 9239 files repaired total
- E5/E6/E7/E8/E9/E10 derived root dirs: repaired through the root report pass

After repair, remaining `005_mirror` entries are legitimate mirrors of actual scene `005`.

## Fix 3: Raw3D Root Reprojection

After fixing mirror names and raw scene axes, raw3d root-conditioning data was regenerated through:

- `scripts/postprocess_root_raw3d.py`

Relevant regenerated outputs:

- `outputs/e8_classifier_raw3d_train`
- `outputs/e8_classifier_raw3d_val`
- `outputs/e9_hybrid_raw3d_train`
- `outputs/e9_hybrid_raw3d_val`
- `outputs/e10_gt_projected_train`
- `outputs/e10_gt_projected_val`

The postprocess uses:

- `--project_target_path`
- `--overwrite_root_keys`
- `--update_norm`
- `--clearance_m 0.04`
- `--smooth_window 5`

Observed invalid-root reduction examples:

- E8 train: `0.284160 -> 0.020471`
- E8 val: `0.302766 -> 0.021375`
- E9 val: `0.299242 -> 0.021375`
- E10 train: `0.203072 -> 0.020470`
- E10 val: `0.196129 -> 0.021363`

## GT Scene Metrics

GT validation metrics were recomputed after the scene-name and axis fixes with:

```bash
python scripts/eval_gt_lingo_scene_metrics.py --split val --output_dir outputs/gt_scene_metrics
```

Output files:

- `outputs/gt_scene_metrics/gt_val_scene_metrics_corrected.csv`
- `outputs/gt_scene_metrics/gt_val_scene_metrics_summary.csv`

Observed corrected GT validation summary:

- samples: `1732`
- `CollisionFrameRate`: `0.13005815997891088`
- `PenetrationRate`: `0.0126687079946753`
- `PenetrationMean`: `0.004803695150115473`
- `PenetrationMax`: `0.004803695150115473`
- `OutOfSceneOrFloorIgnoredJointRate`: `0.1390243643823216`
- `IgnoredFloorJointRate`: `0.13881431474597675`
- `OutOfBoundsJointRate`: `0.00021004963634483473`
- `PooledPenetrationRate`: `0.015203927097094158`

Note: these are local joint-occupancy penetration metrics, not strict mesh-vertex penetration metrics from the paper.

## Training After Repair

Stage2 retraining was launched after the dataset repair with:

```bash
bash scripts/launch_e4_e10_retrain_mirrorfix50.sh
```

The launcher starts E4-E10 in tmux session:

- `e4_e10_retrain50`

Output root:

- `outputs/retrain_mirrorfix50`

Checkpoint behavior:

- `best_checkpoint.pt` is still saved on best validation loss
- fixed epoch checkpoints are saved every 50 epochs via `--save_every_epochs 50`
- expected periodic checkpoint at epoch 50: `checkpoints/epoch_0050.pt`

## Root TrajCo Comparison

An additional root-stage TrajCo comparison is launched with:

```bash
bash scripts/launch_root_trajco_comparison_mirrorfix50.sh
```

This uses the repaired root trajectory cache for Stage1, then loads the Stage1 root checkpoint for Stage2 body SceneCo training:

- Stage1 root TrajCo data: `lingo_root_trajectory_smplx`
- Stage2 body SceneCo data: `lingo_smplx_cache`
- tmux session: `root_trajco_compare50`
- Stage1 output: `outputs/retrain_mirrorfix50/root_trajco_stage1`
- Stage2 output: `outputs/retrain_mirrorfix50/root_trajco_stage2_sceneco_body`

## Unified Test and Visualization Root

All current retraining, testing, and visualization artifacts for this repaired run are organized under:

- `outputs/retrain_mirrorfix50`

No-train baselines E1-E3 are linked into the run root with:

```bash
python scripts/prepare_retrain_mirrorfix50_test_viz.py --run_root outputs/retrain_mirrorfix50
```

Generated registry and structure artifacts:

- `outputs/retrain_mirrorfix50/eval_viz/experiment_registry.json`
- `outputs/retrain_mirrorfix50/eval_viz/experiment_registry.csv`
- `outputs/retrain_mirrorfix50/eval_viz/figures/experiment_structure.png`
- `outputs/retrain_mirrorfix50/eval_viz/figures/no_train_root_overlay.png`

The registry-driven smoke evaluator can run immediately on the no-train baselines:

```bash
python scripts/eval_retrain_mirrorfix50_registry.py \
  --run_root outputs/retrain_mirrorfix50 \
  --families no_train_baseline \
  --max_samples 30
```

Smoke outputs:

- `outputs/retrain_mirrorfix50/eval_viz/test_smoke/all_sample_metrics.csv`
- `outputs/retrain_mirrorfix50/eval_viz/test_smoke/summary_metrics.csv`
- `outputs/retrain_mirrorfix50/eval_viz/test_smoke/summary_metrics.md`

Scene-action videos are rendered from the same registry with:

```bash
python scripts/render_retrain_mirrorfix50_scene_videos.py \
  --run_root outputs/retrain_mirrorfix50 \
  --include E1 E2 E3 \
  --sample_idx 0 \
  --videos_per_exp 1
```

Video outputs:

- `outputs/retrain_mirrorfix50/eval_viz/videos/scene_actions/E1/sample_000_scene_action.mp4`
- `outputs/retrain_mirrorfix50/eval_viz/videos/scene_actions/E2/seg_00000_scene_action.mp4`
- `outputs/retrain_mirrorfix50/eval_viz/videos/scene_actions/E3/seg_00000_scene_action.mp4`
