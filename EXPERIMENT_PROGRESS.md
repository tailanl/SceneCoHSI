# Experiment Progress Report

**Generated**: 2026-06-11 09:15
**Repository**: SceneCoHSI

---

## Experiment Pipeline

### Stage 1: Root Control (no SceneCo training)

| ID | Method | Root Source | Body | PathADE | CFR | PenRate | Status |
|----|--------|------------|------|---------|-----|---------|--------|
| E1 | Energy Guidance | hand-crafted energy loss | Original Kimodo | 2.10 | 0.533 | 0.207 | ✅ DONE |
| E2 | Classifier Guidance | trained RootPathClassifier | Original Kimodo | 1.07 | 0.196 | 0.041 | ✅ DONE |
| E3 | Hybrid Guidance | classifier + energy | Original Kimodo | 1.21 | 0.276 | 0.097 | ✅ DONE |

**Key finding**: Classifier guidance (E2) outperforms hand-crafted energy (E1) by 49% on PathADE.

### Stage 2: SceneCo Body Training (80 epochs, valid roots)

| ID | Method | Root Source | Body | PathADE | CFR | Fallback | Status |
|----|--------|------------|------|---------|-----|----------|--------|
| E4 v3 | Energy + Stage2 | energy-guided | SceneCo trained | — | — | 0 ✅ | 🔄 training 1/80 |
| E5 v3 | Classifier + Stage2 | classifier-guided | SceneCo trained | 1.15 | 0.27 | 0 ✅ | ✅ DONE |
| E6 v3 | Hybrid + Stage2 | hybrid-guided | SceneCo trained | — | — | 0 ✅ | 🔄 training 78/80 |
| E7 v3 | GT + Stage2 | GT root (upper bound) | SceneCo trained | — | — | 0 ✅ | 🔄 body gen |

### Variant A: Scene-Aware Root Classifier

| Step | Description | Result | Status |
|------|------------|--------|--------|
| A1 | Train scene-aware classifier (20-dim, with scene_collision mode) | val_acc=1.0000 | ✅ DONE |
| A2 | Generate scene-aware classifier-guided roots (val split, 1732 samples) | 304/1732 roots | 🔄 running |

### Variant B: Stage2 + SceneCo + TrajCo

| Step | Description | Config | Status |
|------|-----------|--------|--------|
| B1-E7 | Stage2 with TrajCo on GT roots | `stage2_root_guided_sceneco_trajco.yaml` | 🔄 epoch 39/80 |
| B1-E4 | Stage2 with TrajCo on scene-guided roots | same config | ⏳ queued (needs A2 roots) |

### E0: NoGuidance Baseline

| Status | Blocker |
|--------|---------|
| BLOCKED | `scripts/generate.py` missing |

---

## Running Processes (tmux)

```
Session              GPU   Progress
e4_v3_train          0     E4 Stage2 80ep, epoch 1/80
e5_body_eval         7     E5 body gen + eval (DONE, tmux dead)
e7_body_eval         7     E7 body gen + eval (just started)
A2_scene_roots       1     Scene-aware root generation, 304/1732
e5_stage2_train      (dead) E5 v3 training (completed)
e6_v3_train          4     E6 v3, epoch 78/80
B1_E7_trajco         4     B1-E7 TrajCo, epoch 39/80
e5_stage2_train      5     E5 slow 400ep (legacy, keep)
e6_stage2_train      6     E6 slow 400ep (legacy, keep)
```

---

## Output Files

```
outputs/e5_v3_stage2/
  checkpoints/best_checkpoint.pt
  path_metrics.csv
  scene_metrics.csv
  val_gen/*.npz (1731 body outputs)

outputs/e7_v3_stage2/
  checkpoints/best_checkpoint.pt

outputs/root_path_scene_classifier_sdf/
  best.pt, latest.pt, train_log.csv, final_metrics.json

outputs/e4_energy_guidance_train/path_only/  15584 roots ✅
outputs/e4_energy_guidance_val/path_only/    1732 roots ✅
outputs/e5_classifier_guidance_train/path_only/  15583 roots ✅
outputs/e5_classifier_guidance_val/path_only/    1731 roots ✅
outputs/e6_hybrid_guidance_train/path_only/  15583 roots ✅
outputs/e6_hybrid_guidance_val/path_only/    1731 roots ✅
outputs/e7_gt_root_v3_train/  15584 roots ✅
outputs/e7_gt_root_v3_val/    1732 roots ✅
```

---

## Validation Checks

| Check | E4 v3 | E5 v3 | E6 v3 | E7 v3 | B1-E7 |
|-------|-------|-------|-------|-------|-------|
| external_root_enabled=True | ✅ | ✅ | ✅ | ✅ | ✅ |
| use_external_root=True | ✅ | ✅ | ✅ | ✅ | ✅ |
| Zero fallback | ✅ | ✅ | ✅ | ✅ | ✅ |
| Root schema complete (6 keys) | ✅ | — | — | ✅ | — |
| source_id 100% match | ✅ | ✅ | ✅ | ✅ | ✅ |
| Root fix max_error < 1e-5 | — | ✅ | — | — | — |
| path_metrics.csv | — | ✅ | — | — | — |
| scene_metrics.csv | — | ✅ | — | — | — |

---

## Fixed Issues

1. **Scene eval**: Fixed coordinate mapping (use cache voxel_grid + motion extent, not hard-coded voxel_size/grid_origin)
2. **Root schema**: Added `guided_root_5d_norm`, `guided_root_5d_meter`, `target_path_xz`, `source_file` to all root files
3. **source_id matching**: Root filenames now match dataset `_load_cached_index` source_id (cache-based, seed 42)
4. **Stage2 output_dir**: All trainings use explicit `--output_dir` to avoid mutual overwrites
5. **E7 double-normalize fix**: `export_gt_root_for_stage2_v2.py` no longer applies second `mr.normalize()`

---

## Pending (after current trainings complete)

- E4 v3 body gen + eval
- E6 v3 body gen + eval
- E7 v3 eval (body gen running)
- B1-E7 body gen + eval
- A2 complete → B1-E4 training → body gen + eval
