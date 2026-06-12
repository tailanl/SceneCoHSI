# CURRENT_BLOCKERS

Re-audit timestamp: 2026-06-10 16:50 Asia/Shanghai. No listed blocker was cleared by the latest check.

## Blocker 1

* Experiment: E4 EnergyGuidance + Stage2 SceneCo
* Severity: HIGH
* Evidence: `outputs/e4_fast150/train.log` reports `external_root_sources (first 3): ['path_guided_root', 'path_root_missing_gt_fallback', 'path_root_missing_gt_fallback']`. E4 energy root NPZs also miss `guided_root_5d_meter`, `target_path_xz`, and `source_file`.
* Log path: `outputs/e4_fast150/train.log`; root dirs `outputs/e4_energy_guidance_train/path_only`, `outputs/e4_energy_guidance_val/path_only`
* Required fix: Regenerate or convert E4 energy roots with the full required schema and sample IDs that match the Stage2 dataset before trusting/restarting E4 Stage2.
* Can continue other experiments: YES

## Blocker 2

* Experiment: E7 GTRoot + Stage2 SceneCo
* Severity: HIGH
* Evidence: `outputs/e7_gt_root_stage2_sceneco/train.log` reports `external_root_sources (first 3): ['path_root_missing_gt_fallback', 'path_root_missing_gt_fallback', 'path_root_missing_gt_fallback']`.
* Log path: `outputs/e7_gt_root_stage2_sceneco/train.log`
* Required fix: Fix GT-root external-root file naming/sample-id matching, verify `external_root_sources` uses matched root files, then rerun E7 Stage2.
* Can continue other experiments: YES

## Blocker 3

* Experiment: E7 GTRoot + Stage2 SceneCo
* Severity: MEDIUM
* Evidence: `outputs/e7_gt_root_train` and `outputs/e7_gt_root_val` NPZ files contain `guided_root_5d_norm`, `target_path_xz`, `text`, `scene_name`, and `source_file`, but miss `guided_root_5d_meter`.
* Log path: NPZ audit of `outputs/e7_gt_root_train/seg_00000.npz` and `outputs/e7_gt_root_val/seg_00014.npz`
* Required fix: Add or regenerate `guided_root_5d_meter` for GT-root artifacts if downstream tools require the full root schema.
* Can continue other experiments: YES

## Blocker 4

* Experiment: E1 EnergyGuidance + Original Body
* Severity: MEDIUM
* Evidence: `outputs/e1_energy_guidance_root/sample_000.npz` has only `gen_root`, `gt_root_xz`, `gen_joints`, `gt_joints`, `text`, and `scene_name`; it misses `guided_root_5d_norm`, `guided_root_5d_meter`, `target_path_xz`, and `source_file`.
* Log path: `outputs/e1_energy_guidance_root/generate_root.log`
* Required fix: Regenerate or convert E1 root outputs to the required root NPZ schema.
* Can continue other experiments: YES

## Blocker 5

* Experiment: E2 ClassifierGuidance + Original Body
* Severity: HIGH
* Evidence: Body NPZ files and metrics exist, but no body generation log with `root fix`, `max error`, or `max_error` was found for E2. Body generation ran, but fixed-root correctness is not verified because root fix max error is missing from log.
* Log path: No E2 `generate_body.log` found under `outputs/e2_classifier_guidance_body`
* Required fix: Produce or rerun body generation with root-fix max error logging and verify max error is less than `1e-5`.
* Can continue other experiments: YES

## Blocker 6

* Experiment: E3 HybridGuidance + Original Body
* Severity: HIGH
* Evidence: Body NPZ files and metrics exist, but no body generation log with `root fix`, `max error`, or `max_error` was found for E3. Body generation ran, but fixed-root correctness is not verified because root fix max error is missing from log.
* Log path: No E3 `generate_body.log` found under `outputs/e3_hybrid_guidance_body`
* Required fix: Produce or rerun body generation with root-fix max error logging and verify max error is less than `1e-5`.
* Can continue other experiments: YES

## Blocker 7

* Experiment: E4/E5/E6 Stage2 SceneCo
* Severity: MEDIUM
* Evidence: Stage2 training jobs are running and checkpoints/logs exist, but no Stage2 body-generation outputs or `path_metrics.csv`/`scene_metrics.csv` were found for E4, E5, or E6.
* Log path: `outputs/e4_energy_stage2_sceneco/train.log`, `outputs/e5_classifier_stage2_sceneco/train.log`, `outputs/e6_hybrid_stage2_sceneco/train.log`, plus fast150 logs
* Required fix: After a valid Stage2 checkpoint is selected, run body generation with root-fix max error logging and then compute path/scene metrics.
* Can continue other experiments: YES
