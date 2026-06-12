# Experiments E1–E10 Configuration Reference

> Auto-generated 2026-06-12. Covers all E-series experiments with root generation method, body model, config file, output directory, and status.

---

## Overview

| Label | Root Method | Body Method | Config YAML | Output Dir | Status | SceneCo |
|-------|------------|-------------|-------------|-----------|--------|---------|
| **E0** | No guidance | Original Kimodo | N/A (script missing) | `outputs/baseline_kimoto/` | BLOCKED | No |
| **E1** | Energy loss | Original Kimodo | `guidance_root_scene.yaml` | `outputs/e1_energy_guidance_body/` | DONE (30 val) | No |
| **E2** | 19-dim Classifier | Original Kimodo | `root_classifier_guidance.yaml` | `outputs/e2_classifier_guidance_body/` | DONE (30 val) | No |
| **E3** | Hybrid (cls+energy) | Original Kimodo | `root_classifier_guidance.yaml --hybrid` | `outputs/e3_hybrid_guidance_body/` | DONE (30 val) | No |
| **E4** | Energy-guided v3 | Stage2 SceneCo | `stage2_energy_root_guided_sceneco.yaml` | `outputs/e4_v3_stage2/` | DONE (full val) | Yes |
| **E5** | Classifier-guided v3 | Stage2 SceneCo | `stage2_classifier_root_guided_sceneco.yaml` | `outputs/e5_v3_stage2/` | DONE (full val) | Yes |
| **E6** | Hybrid-guided v3 | Stage2 SceneCo | `stage2_hybrid_root_guided_sceneco.yaml` | `outputs/e6_v3_stage2/` | DONE (full val) | Yes |
| **E7** | GT root v3 | Stage2 SceneCo | `stage2_gt_root_sceneco.yaml` | `outputs/e7_v3_stage2/` | DONE (full val) | Yes |
| **E8** | Classifier + raw3d | Stage2 SceneCo | `stage2_classifier_root_guided_sceneco.yaml` | `outputs/e8_classifier_raw3d_stage2/` | TRAINING (8/60) | Yes |
| **E9** | Hybrid + raw3d | Stage2 SceneCo | `stage2_hybrid_root_guided_sceneco.yaml` | `outputs/e9_hybrid_raw3d_stage2/` | TRAINING (9/60) | Yes |
| **E10** | GT projected | Stage2 SceneCo | `stage2_gt_root_sceneco.yaml` | `outputs/e10_gt_projected_stage2/` | TRAINING (8/60) | Yes |

---

## E0 — NoGuidance + Original Body (BLOCKED)

- **Purpose**: Baseline Kimodo without any path guidance or scene conditioning.
- **Root**: Unmodified Kimodo diffusion output (no guidance).
- **Body**: Original Kimodo body model.
- **Config**: N/A (requires `scripts/generate.py` — file missing).
- **Status**: Cannot run until sampling script is restored.

---

## E1 — EnergyGuidance + Original Body (DONE)

- **Purpose**: Path-following via hand-crafted energy loss.
- **Root**: Energy loss on target path waypoints + scene collision penalty.
- **Body**: Original Kimodo body (no SceneCo, no TrajCo).
- **Config**: `configs/guidance_root_scene.yaml`
- **Output**: `outputs/e1_energy_guidance_body/` (30 val samples)
- **Results**: PathADE=2.10, CFR=0.1348
- **Known issues**: Old root NPZ format (missing `guided_root_5d_norm`, `guided_root_5d_meter`).

---

## E2 — ClassifierGuidance + Original Body (DONE)

- **Purpose**: Path-following via trained classifier guidance.
- **Root**: 19-dim RootPathClassifier guidance (val_acc=1.0).
- **Classifier**: `outputs/root_path_classifier/best.pt`
- **Body**: Original Kimodo body (no SceneCo).
- **Config**: `configs/root_classifier_guidance.yaml`
- **Output**: `outputs/e2_classifier_guidance_body/` (30 val samples)
- **Results**: PathADE=1.07, CFR=0.0023

---

## E3 — HybridGuidance + Original Body (DONE)

- **Purpose**: Combined classifier + energy loss guidance.
- **Root**: Hybrid guidance (classifier logits + scene energy loss).
- **Body**: Original Kimodo body (no SceneCo).
- **Config**: `configs/root_classifier_guidance.yaml` with `--hybrid`
- **Output**: `outputs/e3_hybrid_guidance_body/` (30 val samples)
- **Results**: PathADE=1.21, CFR=0.0359

---

## E4 — EnergyGuidance + Stage2 SceneCo (DONE)

- **Purpose**: Energy-guided root + SceneCo body adapter.
- **Root**: Energy loss + raw3d correction (postprojection to walkable region).
- **Body**: Stage2 SceneCo body adapter (80 epochs, body_only, body_mse loss, scene_dropout=0.1).
- **Config**: `configs/stage2_energy_root_guided_sceneco.yaml`
- **Output**: `outputs/e4_v3_stage2/` (1753 body NPZ + metrics)
- **Results**: PathADE=1.87, CFR=0.2119, PenRate=0.0732

---

## E5 — ClassifierGuidance + Stage2 SceneCo (DONE)

- **Purpose**: Classifier-guided root + SceneCo body adapter.
- **Root**: 19-dim classifier guidance (from E2 classifier).
- **Body**: Stage2 SceneCo body adapter (80 epochs, body_only, body_mse loss, scene_dropout=0.1).
- **Config**: `configs/stage2_classifier_root_guided_sceneco.yaml`
- **Output**: `outputs/e5_v3_stage2/` (1731 body NPZ + metrics)
- **Results**: PathADE=1.37, CFR=0.1387, PenRate=0.0310

---

## E6 — HybridGuidance + Stage2 SceneCo (DONE)

- **Purpose**: Hybrid-guided root + SceneCo body adapter.
- **Root**: Hybrid classifier + energy guidance.
- **Body**: Stage2 SceneCo body adapter (80 epochs, body_only, body_mse loss, scene_dropout=0.1).
- **Config**: `configs/stage2_hybrid_root_guided_sceneco.yaml`
- **Output**: `outputs/e6_v3_stage2/` (1731 body NPZ + metrics)
- **Results**: PathADE=1.36, CFR=0.1388, PenRate=0.0299

---

## E7 — GTRoot + Stage2 SceneCo (DONE)

- **Purpose**: Upper-bound: GT root with SceneCo body.
- **Root**: Ground-truth root trajectory exported from dataset.
- **Body**: Stage2 SceneCo body adapter (80 epochs, body_only, body_mse loss, scene_dropout=0.1).
- **Config**: `configs/stage2_gt_root_sceneco.yaml`
- **Output**: `outputs/e7_v3_stage2/` (1732 body NPZ + metrics)
- **Results**: PathADE=0.0 (GT root), CFR=0.3359, PenRate=0.0911
- **Note**: Proves perfect root tracking does not guarantee low body collision.

---

## E8 — Classifier + Raw3d + Stage2 SceneCo (TRAINING)

- **Purpose**: E5 roots postprocessed through raw3d walkable projection + SceneCo.
- **Root**: E5 classifier roots, projected to walkable XZ region via `postprocess_root_raw3d.py` (clearance=0.04m, smooth_window=5).
- **Body**: Stage2 SceneCo body adapter (60/80 epochs).
- **Config**: `configs/stage2_classifier_root_guided_sceneco.yaml`
- **Output**: `outputs/e8_classifier_raw3d_stage2/` (epoch ~8/60)
- **Variants**:
  - `e8_classifier_raw3d_stage2_fair80/` (80 epochs)
  - `e8_classifier_raw3d_stage2_scenefix60/` (60 epochs, re-postprocessed)
  - `e8_classifier_raw3d_stage2_scenefix80/` (80 epochs, re-postprocessed)

---

## E9 — Hybrid + Raw3d + Stage2 SceneCo (TRAINING)

- **Purpose**: E6 roots postprocessed through raw3d walkable projection + SceneCo.
- **Root**: E6 hybrid roots, projected to walkable XZ region.
- **Body**: Stage2 SceneCo body adapter (60/80 epochs).
- **Config**: `configs/stage2_hybrid_root_guided_sceneco.yaml`
- **Output**: `outputs/e9_hybrid_raw3d_stage2/` (epoch ~9/60)
- **Variants**:
  - `e9_hybrid_raw3d_stage2_fair80/` (80 epochs)
  - `e9_hybrid_raw3d_stage2_scenefix60/` (60 epochs, re-postprocessed)
  - `e9_hybrid_raw3d_stage2_scenefix80/` (80 epochs, re-postprocessed)

---

## E10 — GT Projected + Stage2 SceneCo (TRAINING)

- **Purpose**: Quantify how much of E7's CFR comes from GT root being outside walkable region.
- **Root**: GT roots postprocessed with `postprocess_root_raw3d.py` (projected into walkable XZ).
- **Body**: Stage2 SceneCo body adapter (60/80 epochs).
- **Config**: `configs/stage2_gt_root_sceneco.yaml`
- **Output**: `outputs/e10_gt_projected_stage2/` (epoch ~8/60)
- **Variants**:
  - `e10_gt_projected_stage2_fair80/` (80 epochs)
  - `e10_gt_projected_stage2_scenefix60/` (60 epochs, re-postprocessed)
  - `e10_gt_projected_stage2_scenefix80/` (80 epochs, re-postprocessed)

---

## Experiment Design Logic

```
E0-E3:  Root guidance only, NO SceneCo body
        ↳ E0: no guidance      → baseline
        ↳ E1: energy guidance  → hand-crafted
        ↳ E2: classifier       → learned
        ↳ E3: hybrid           → combined

E4-E7:  Root guidance + Stage2 SceneCo body (v3)
        ↳ E4: energy root + SceneCo
        ↳ E5: classifier root + SceneCo
        ↳ E6: hybrid root + SceneCo
        ↳ E7: GT root + SceneCo        → upper bound

E8-E10: Raw3d-corrected roots + Stage2 SceneCo body
        ↳ E8: classifier + raw3d + SceneCo
        ↳ E9: hybrid + raw3d + SceneCo
        ↳ E10: GT projected + SceneCo   → oracular upper walkable

Key comparisons:
  E1 vs E4:  does SceneCo body improve over original body given same root method?
  E2 vs E5:  same question for classifier roots
  E3 vs E6:  same question for hybrid roots
  E7:       oracle — what are the best possible metrics with perfect root?
  E4/E5/E6: which root guidance works best with SceneCo?
  E8/E9:    does raw3d walkable projection improve metrics?
  E10:      does project GT root to walkable explain E7's CFR?
```

---

## Output Directory Structure (per experiment)

```
outputs/<exp>/
├── stage2/
│   ├── checkpoints/
│   │   ├── best_checkpoint.pt
│   │   ├── epoch_020.pt
│   │   └── ...
│   ├── train.log
│   ├── pipeline.log
│   └── body_generation/
│       ├── seg_XXXXX.npz       # body joint + root
│       ├── path_metrics.csv
│       └── scene_metrics.csv
├── <root_data>/
│   ├── train/
│   │   └── seg_XXXXX.npz       # guided roots
│   └── val/
│       └── seg_XXXXX.npz
└── ...
```

### CSVs Field Reference

| CSV | Key Columns |
|-----|-------------|
| `path_metrics.csv` | PathADE, PathFDE, RootFixMaxError |
| `scene_metrics.csv` | CollisionFrameRate, NonWalkableRootRate, PenetrationRate, PenetrationMean, PenetrationMax, SceneSDFPenalty |

---

## Related Experiments (A/B Series)

| Label | Description | Status |
|-------|-------------|--------|
| **A1** | Scene-aware root classifier training (20-dim: 19 motion + 1 scene_collision) | DONE (val_acc=1.0) |
| **A2** | Scene-aware classifier root generation (1731 val roots) | DONE |
| **B1-E4** | A2 scene-aware roots + Stage2 SceneCo + TrajCo body | TRAINED (no metrics) |
| **B1-E7** | GT roots + Stage2 SceneCo + TrajCo body (CFR=0.3382) | DONE |

See `README_ROOT_GUIDANCE_SCENE_AND_TRAJCO.md` for A/B series details.
