# Full Experiment Comparison: All Variants

**Date**: 2026-06-11

---

## Experiment Matrix

### Group 1: Root Control (no body training)

| ID | Root Method | Body | PathADE | CFR | Key Config |
|----|------------|------|---------|-----|-----------|
| E1 | Energy loss | Kimodo original | 2.1033 | 0.5333 | `guidance_root_scene.yaml`, hand-crafted loss |
| E2 | Classifier | Kimodo original | 1.0650 | 0.1958 | `root_classifier_guidance.yaml`, 19-dim classifier, val_acc=1.0 |
| E3 | Hybrid (cls+energy) | Kimodo original | 1.2114 | 0.2758 | `root_classifier_guidance.yaml --hybrid` |

### Group 2: SceneCo Body Training (Stage2, 80 epochs)

| ID | Root Source | Body Training | PathADE | CFR | Key Config |
|----|------------|--------------|---------|-----|-----------|
| E4 v3 | Energy-guided | SceneCo body adapt | training | — | `stage2_energy_root_guided_sceneco.yaml` |
| E5 v3 | Classifier-guided | SceneCo body adapt | 1.3710 | 0.1387 | `stage2_classifier_root_guided_sceneco.yaml` |
| E6 v3 | Hybrid-guided | SceneCo body adapt | running | — | `stage2_hybrid_root_guided_sceneco.yaml` |
| E7 v3 | GT root | SceneCo body adapt | 0.0000 | 0.3359 | `stage2_gt_root_sceneco.yaml` |

### Group 3: Older Experiments (different training framework, no body-gen eval)

| ID | Architecture | epochs | best_val_loss | traj_loss | SceneCo | TrajCo |
|----|-------------|--------|--------------|-----------|---------|--------|
| `trajco_cross_root_sceneco_body` | TrajCo(cross) root + SceneCo body | 400 | 9.0 (est) | 1.12 | ✅ body | ✅ root |
| `trajco_body_sceneco_gt_root` | GT root + TrajCo body + SceneCo body | 400 | 1.92 | 0 | ✅ body | ✅ body |
| `body_only_sceneco` | SceneCo body only | 100 | 1039 | N/A | ✅ body | ❌ |
| `root_only_sceneco` | SceneCo root only | 100 | 2071 | N/A | ✅ root | ❌ |
| `smplx_root_body` | SMPLX root+body (baseline) | 400 | 297 | 0 | ✅ both | ❌ |

### Group 4: New Variants (Scene-aware + TrajCo)

| ID | Description | Status | Config |
|----|------------|--------|--------|
| A1 | Scene-aware root classifier (20-dim) | ✅ val_acc=1.0 | `root_classifier_scene.yaml` |
| A2 | Scene-aware classifier root generation | ✅ 1731 roots | `root_classifier_scene_guidance.yaml use_scene:true` |
| B1-E4 | Stage2 SceneCo+TrajCo on scene roots | 🔄 training | `stage2_root_guided_sceneco_trajco.yaml` |
| B1-E7 | Stage2 SceneCo+TrajCo on GT roots | 🔄 training | `stage2_root_guided_sceneco_trajco.yaml` |

---

## Configuration Reference

### Stage2 SceneCo Training (E4-E7 v3)

```yaml
sceneco:
  enabled: true
  body_only: true
training:
  batch_size: 4
  num_epochs: 80
  lr: 1e-4
  prior_weight: 0.0
  scene_dropout: 0.1
loss: body_mse only
trainable: scene_encoder + SceneCo body adapter (145M/428M)
frozen: pretrained Kimodo backbone
root_source: external (fixed, not trained)
```

### Stage2 TrajCo Training (B1-E4, B1-E7)

```yaml
sceneco:
  enabled: true
  body_only: true
trajco:
  enabled: true
  use_trajco: true
  use_trajco_body: true
  use_trajco_root: false
```

### Older TrajCo + SceneCo (trajco_cross_root_sceneco_body)

```yaml
# Different training script (kimodo_sceneco.train.train)
SceneCo: root_model=False, body_model=True
TrajCo: enabled, type=cross_attn, traj_dim=5, root=True, body=False
loss: mse + prior_loss + traj_loss
```

---

## Key Findings

1. **Classifier guidance beats energy guidance**: E2 PathADE=1.07 vs E1 2.10 (49% better)
2. **SceneCo reduces collision**: E5 v3 CFR=0.14 vs E2 CFR=0.20 (30% relative improvement)
3. **GT root is upper bound**: E7 PathADE=0.0 (perfect), but CFR=0.34 shows scene density limits
4. **Older TrajCo experiments**: `trajco_body_sceneco_gt_root` achieved best_val=1.92, suggesting body prediction is near-perfect with GT root
5. **SceneCo alone on body**: `body_only_sceneco` had high val_loss (~1039), suggesting body-only SceneCo without proper root conditioning is difficult
6. **All v3 experiments have zero fallback**: 100% source_id matching validated

---

## Scene Evaluation Fix

CFR in E1-E3 originally reported ~0.96-1.0 (wrong). Fixed by:
- Using cached `voxel_grid` (64³, same as training) instead of raw scene
- Dynamic coordinate mapping from motion extent instead of hard-coded `voxel_size=0.1`

See `SCENE_PROCESSING.md` for details.

---

## Video Demonstrations

See `outputs/viz_videos/` for skeleton animation comparisons:
- `E1_comparison.mp4`: Energy root, jittery motion
- `E2_comparison.mp4`: Classifier root, clean trajectory
- `E3_comparison.mp4`: Hybrid root, intermediate quality
- `E5_v3_comparison.mp4`: Classifier + SceneCo body, reduced collision
- `E7_v3_comparison.mp4`: GT root + SceneCo body, upper bound
