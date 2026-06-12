# CURRENT_RUN_AUDIT

Re-audit timestamp: 2026-06-10 16:50 Asia/Shanghai.

User note checked: the user reported that another AI fixed this about 12 hours earlier. This re-audit does not find evidence that the critical blockers were cleared. Git HEAD is unchanged, root NPZ schemas are unchanged, E4/E7 still have Stage2 fallback evidence, and E2/E3/E4/E5/E6 still lack the required completed body/root-fix/metrics chain.

## 1. Active jobs

Audit timestamp for the final process snapshot: 2026-06-10 16:48 Asia/Shanghai.

| PID | GPU | GPU Mem | Script | Likely experiment | Evidence |
| ---: | --: | ------: | ------ | ----------------- | -------- |
| 2563754 | 0 | 5716 MiB | `train/train_stage2_root_guided_sceneco.py` | E4 EnergyGuidance Stage2, 400 epochs | Running 17:50:07; config `stage2_energy_root_guided_sceneco.yaml`; root dirs `outputs/e4_energy_guidance_train/path_only` and `outputs/e4_energy_guidance_val/path_only`. |
| 3129300 | 1 | 5640 MiB | `train/train_stage2_root_guided_sceneco.py` | E4 EnergyGuidance Stage2 fast150 | Running 47:19; output dir `outputs/e4_fast150`; first sources include fallback. |
| 1921445 | 2 | 5706 MiB | `train/train_stage2_root_guided_sceneco.py` | Generic/legacy Stage2, not mapped to E1-E7 | Running 2-00:42:50; uses `configs/stage2_root_guided_sceneco.yaml` with `outputs/guided_roots_train/path_scene`. |
| 3172176 | 3 | 13814 MiB | `scripts/extract_datas_masks_v5.py` | Not an E1-E7 experiment | Running 24:48; data mask extraction under `../datasets/-1lab`. |
| 3129433 | 4 | 5630 MiB | `train/train_stage2_root_guided_sceneco.py` | E5 ClassifierGuidance Stage2 fast150 | Running 47:15; output dir `outputs/e5_fast150`; first sources all `path_guided_root`. |
| 2554957 | 5 | 5632 MiB | `train/train_stage2_root_guided_sceneco.py` | E5 ClassifierGuidance Stage2, 400 epochs | Running 17:56:02; root dirs `outputs/e5_classifier_guidance_train/path_only` and `outputs/e5_classifier_guidance_val/path_only`. |
| 2555190 | 6 | 5632 MiB | `train/train_stage2_root_guided_sceneco.py` | E6 HybridGuidance Stage2, 400 epochs | Running 17:55:58; root dirs `outputs/e6_hybrid_guidance_train/path_only` and `outputs/e6_hybrid_guidance_val/path_only`. |
| 3129565 | 7 | 5676 MiB | `train/train_stage2_root_guided_sceneco.py` | E6 HybridGuidance Stage2 fast150 | Running 47:12; output dir `outputs/e6_fast150`; first sources all `path_guided_root`. |

No root generation or body generation experiment process was active in the final process snapshot.

## 2. Git status

Required first commands were run in order.

```text
pwd: /home/lzsh2025/kimodo-viser/kimodo_scene_project
git status --short: ?? README_Current_Run_Audit.md
git branch --show-current: master
git log -1 --oneline: 270abcc feat: Stage2 classifier-guided root pipeline + experiment configs
```

Current `git status --short` also includes the three audit report files. No code or YAML fix is visible in git status.

## 3. Python/YAML check results

Python compile check passed for:

```text
kimodo_sceneco/model/kimodo_model.py
kimodo_sceneco/critic/root_path_scene_classifier.py
kimodo_sceneco/critic/root_classifier_features.py
kimodo_sceneco/critic/root_classifier_dataset.py
kimodo_sceneco/critic/train_root_classifier.py
scripts/generate_root_classifier_guidance.py
scripts/generate_root_guidance.py
scripts/generate_body_from_root.py
train/train_stage2_root_guided_sceneco.py
```

YAML parse check passed for:

```text
configs/root_classifier.yaml
configs/root_classifier_guidance.yaml
configs/guidance_root_scene.yaml
configs/stage2_root_guided_sceneco.yaml
```

## 4. Output directories found

Relevant artifacts found:

| Directory | Files observed |
| --------- | -------------- |
| `outputs/root_path_classifier` | `latest.pt`, `best.pt`, `train_log.csv`, `train.log` |
| `outputs/e1_energy_guidance_root` | 30 root NPZ files and `generate_root.log` |
| `outputs/e1_energy_guidance_body` | 30 body NPZ files, `generate_body.log`, `path_metrics.csv`, `scene_metrics.csv` |
| `outputs/e2_classifier_guidance_root` | 30 root NPZ files, `generate.log`, `guidance_log.csv` |
| `outputs/e2_classifier_guidance_body` | 30 body NPZ files, `path_metrics.csv`, `scene_metrics.csv` |
| `outputs/e3_hybrid_guidance_root` | 30 root NPZ files, `guidance_log.csv` |
| `outputs/e3_hybrid_guidance_body` | 30 body NPZ files, `path_metrics.csv`, `scene_metrics.csv` |
| `outputs/e4_energy_guidance_train/path_only` | 13911 root NPZ files and `generate.log` |
| `outputs/e4_energy_guidance_val/path_only` | 1548 root NPZ files and `generate.log` |
| `outputs/e4_energy_stage2_sceneco` | Growing `train.log`; no per-directory checkpoint file found |
| `outputs/e4_fast150` | Growing `train.log`, `checkpoints/best_checkpoint.pt` |
| `outputs/e5_classifier_guidance_train/path_only` | 15583 root NPZ files and `generate.log` |
| `outputs/e5_classifier_guidance_val/path_only` | 1731 root NPZ files and `generate.log` |
| `outputs/e5_classifier_stage2_sceneco` | Growing `train.log`; no per-directory checkpoint file found |
| `outputs/e5_fast150` | Growing `train.log`, `checkpoints/best_checkpoint.pt` |
| `outputs/e6_hybrid_guidance_train/path_only` | 15583 root NPZ files and `generate.log` |
| `outputs/e6_hybrid_guidance_val/path_only` | 1731 root NPZ files and `generate.log` |
| `outputs/e6_hybrid_stage2_sceneco` | Growing `train.log`; no per-directory checkpoint file found |
| `outputs/e6_fast150` | Growing `train.log`, `checkpoints/best_checkpoint.pt` |
| `outputs/e7_gt_root_train` | 8730 GT-root NPZ files |
| `outputs/e7_gt_root_val` | 1548 GT-root NPZ files |
| `outputs/e7_gt_root_stage2_sceneco` | `train.log`, `val_gen`, `path_metrics.csv`, `scene_metrics.csv` |
| `outputs/stage2_root_guided_sceneco/checkpoints` | Shared `epoch_*.pt` and `best_checkpoint.pt` checkpoint files |

The latest output discovery found no new E1-E7 body/metrics directories beyond the previous audit. Fresh changes are growing Stage2 logs for E4/E5/E6 and fast150 logs/checkpoints.

## 5. E1-E7 experiment status

E1 produced root, body, and metrics. Body root fix is verified with `Root fix max_error: 0.00e+00`, but root NPZ files use the old key format and are missing required audit keys, so E1 is not marked valid done.

E2 produced classifier-guided roots, original-body outputs, and metrics. Classifier guidance markers are present, but body root-fix max error is missing from logs. Status: INCOMPLETE.

E3 produced hybrid roots, original-body outputs, and metrics. `guidance_log.csv` contains `loss_cls`, `score_valid`, and `grad_norm`, but body root-fix max error is missing from logs. Status: INCOMPLETE.

E4 has active Stage2 jobs and energy root outputs. The E4 400-epoch runtime log starts with all first sources as `path_guided_root`, but the E4 fast150 runtime log still shows external-root fallback and E4 root NPZ files still miss required fields. Status: INVALID.

E5 has active Stage2 jobs, verified classifier-guided roots, classifier markers, external-root runtime evidence, finite losses, and fast150 checkpoint output. Body generation and metrics are pending. Status: VALID_RUNNING.

E6 has active Stage2 jobs, verified hybrid roots, classifier markers, external-root runtime evidence, finite losses, and fast150 checkpoint output. Body generation and metrics are pending. Status: VALID_RUNNING.

E7 has GT-root files, Stage2 training log, generated body files, root-fix verification, and metrics. However, Stage2 external-root sources are logged as `path_root_missing_gt_fallback`, so the Stage2 root-guided experiment is invalid despite body and metric outputs. Status: INVALID.

## 6. Classifier training status

RootPathClassifier full training is valid by the requested checks:

```text
outputs/root_path_classifier/latest.pt exists
outputs/root_path_classifier/best.pt exists
outputs/root_path_classifier/train_log.csv exists
outputs/root_path_classifier/train.log exists
```

Required metrics appear in logs and CSV. Final epoch evidence:

```text
train_loss=0.0016
train_acc=0.9998
val_loss=0.0002
val_acc=1.0000
positive_score_mean=0.9996
negative_score_mean=0.0000
```

Final validation also reports positive score mean 0.9999 and negative score mean 0.0001, so positive_score_mean > negative_score_mean.

The smoke classifier exists, but its one-epoch CSV row has positive_score_mean below negative_score_mean. The full classifier at `outputs/root_path_classifier/best.pt` is the valid classifier checkpoint for experiment evidence.

## 7. Classifier guidance status

Classifier guidance is verified for E2, E3, E5, and E6 root generation because logs/CSV contain:

```text
loss_cls
score_valid
grad_norm
```

Output-format checks:

| Experiment | Root directory | Count | Required root keys |
| ---------- | -------------- | ----: | ------------------ |
| E2 | `outputs/e2_classifier_guidance_root` | 30 | present |
| E3 | `outputs/e3_hybrid_guidance_root` | 30 | present |
| E5 train | `outputs/e5_classifier_guidance_train/path_only` | 15583 | present |
| E5 val | `outputs/e5_classifier_guidance_val/path_only` | 1731 | present |
| E6 train | `outputs/e6_hybrid_guidance_train/path_only` | 15583 | present |
| E6 val | `outputs/e6_hybrid_guidance_val/path_only` | 1731 | present |

E2 log ends with `Done. Saved 30 guided roots`. E5 train/val logs end with 15583 and 1731 saved roots. E6 val log ends with 1731 saved roots; E6 train also contains the required markers.

The old `outputs/root_classifier_guidance_smoke_body/generate_body.log` failed during body load with OmegaConf `UnsupportedValueType: Value 'device' is not a supported primitive type`; it is not counted as a valid body-generation experiment.

## 8. EnergyGuidance status

E1 energy guidance generated 30 root NPZ files, but first-file keys are:

```text
['gen_root', 'gt_root_xz', 'gen_joints', 'gt_joints', 'text', 'scene_name']
```

Missing required root keys:

```text
guided_root_5d_norm
guided_root_5d_meter
target_path_xz
source_file
```

E4 energy guidance generated train/val roots, but first-file keys are:

```text
['gen_root', 'gt_root_xz', 'gen_joints', 'gt_joints', 'text', 'scene_name', 'guided_root_5d_norm']
```

Missing required root keys:

```text
guided_root_5d_meter
target_path_xz
source_file
```

This is an output-format blocker for energy roots.

## 9. Stage2 external_root status

Runtime evidence:

| Experiment | Log path | external_root enabled | Runtime root source evidence | Checkpoint evidence | Status |
| ---------- | -------- | --------------------- | ---------------------------- | ------------------- | ------ |
| E4 400 | `outputs/e4_energy_stage2_sceneco/train.log` | yes | first 3 sources all `path_guided_root` | log says checkpoints saved, but no per-directory `.pt` found | RUNNING_NOT_VERIFIED |
| E4 fast150 | `outputs/e4_fast150/train.log` | yes | first 3 sources include two `path_root_missing_gt_fallback` entries | `outputs/e4_fast150/checkpoints/best_checkpoint.pt` exists | INVALID |
| E5 400 | `outputs/e5_classifier_stage2_sceneco/train.log` | yes | first 3 sources all `path_guided_root` | log says checkpoints saved, but no per-directory `.pt` found | VALID_RUNNING |
| E5 fast150 | `outputs/e5_fast150/train.log` | yes | first 3 sources all `path_guided_root` | `outputs/e5_fast150/checkpoints/best_checkpoint.pt` exists | VALID_RUNNING |
| E6 400 | `outputs/e6_hybrid_stage2_sceneco/train.log` | yes | first 3 sources include `path_guided_root`, `path_guided_root`, `gt_root`; no fallback in this startup line | log says checkpoints saved, but no per-directory `.pt` found | VALID_RUNNING |
| E6 fast150 | `outputs/e6_fast150/train.log` | yes | first 3 sources all `path_guided_root` | `outputs/e6_fast150/checkpoints/best_checkpoint.pt` exists | VALID_RUNNING |
| E7 | `outputs/e7_gt_root_stage2_sceneco/train.log` | yes | first 3 sources all `path_root_missing_gt_fallback` | shared checkpoint files exist | INVALID |

Stage2 process is running, but experiment is not valid because external root files are not matching dataset sample ids. This applies to the E4 fast150 evidence and to the completed E7 Stage2 log.

Current fallback counts from the re-audit:

```text
outputs/e4_energy_stage2_sceneco/train.log:0
outputs/e4_fast150/train.log:1
outputs/e5_classifier_stage2_sceneco/train.log:0
outputs/e5_fast150/train.log:0
outputs/e6_hybrid_stage2_sceneco/train.log:0
outputs/e6_fast150/train.log:0
outputs/e7_gt_root_stage2_sceneco/train.log:1
outputs/e7_gt_root_stage2_fast150/train.log:1
outputs/stage2_root_guided_sceneco/train.log:7
```

## 10. Body root-fix status

Verified:

```text
outputs/e1_energy_guidance_body/generate_body.log:
Root fix max_error: 0.00e+00 | mean: 0.00e+00 | all_passed: True

outputs/e7_gt_root_stage2_sceneco/val_gen/generate_body.log:
Root fix max_error: 0.00e+00 | mean: 0.00e+00 | all_passed: True
```

Not verified:

E2 and E3 body NPZ files exist, but no body generation log with root-fix max error was found. Body generation ran, but fixed-root correctness is not verified because root fix max error is missing from log.

E4, E5, and E6 Stage2 body generation outputs were not found.

## 11. Metrics status

Metrics found:

```text
outputs/e1_energy_guidance_body/path_metrics.csv
outputs/e1_energy_guidance_body/scene_metrics.csv
outputs/e2_classifier_guidance_body/path_metrics.csv
outputs/e2_classifier_guidance_body/scene_metrics.csv
outputs/e3_hybrid_guidance_body/path_metrics.csv
outputs/e3_hybrid_guidance_body/scene_metrics.csv
outputs/e7_gt_root_stage2_sceneco/path_metrics.csv
outputs/e7_gt_root_stage2_sceneco/scene_metrics.csv
```

Metrics missing:

```text
E0: no root/body/metrics artifacts found
E4: no Stage2 body metrics found
E5: no Stage2 body metrics found
E6: no Stage2 body metrics found
```

E7 metrics exist, but they do not make E7 valid because Stage2 external_root fell back instead of using matched GT root files.

## 12. Critical blockers

1. E4 energy roots are missing required root NPZ keys, and E4 fast150 Stage2 logs show fallback.
2. E7 Stage2 external_root is invalid because runtime sources are `path_root_missing_gt_fallback`.
3. E7 GT root NPZ files are missing `guided_root_5d_meter`.
4. E1 energy root NPZ files are old-format and missing required root keys.
5. E2 and E3 body generation cannot be called valid because root-fix max error is missing from logs.
6. E4, E5, and E6 have no Stage2 body generation or metrics yet.

## 13. Safe next action

Do not stop the running jobs during audit. Let the verified E5/E6 Stage2 runs continue if compute policy allows.

Do not trust E4 fast150 or E7 as valid Stage2 root-guided experiments. The safe next experimental action, after this audit, is to fix or regenerate the root artifact mapping/schema so Stage2 logs show matched external roots with no fallback, then rerun only the affected Stage2 experiments. After valid Stage2 checkpoints exist, run body generation with root-fix max error logging and then compute path/scene metrics.
