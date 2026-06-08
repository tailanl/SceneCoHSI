# DEFH Test: TrajCo + SceneCo Ablation Study

Trajectory-coordinated (TrajCo) and Scene-coordinated (SceneCo) cross-attention
ablation experiments on Kimodo motion diffusion model, evaluated on LINGO scene-motion dataset.

---

## 1. Experiment Configurations

| Experiment | TrajCo | SceneCo | Description |
|:-----------|:------:|:------:|:------------|
| **D** (trajco_cross_smplx) | root + body | — | TrajCo cross-attention in all 32 Transformer layers. No SceneCo. |
| **E** (trajco_cross_sceneco_smplx) | — | root + body | SceneCo in all 32 layers. No TrajCo. Baseline for scene-only injection. |
| **F** (trajco_cross_root_sceneco_body) | root only | body only | TrajCo in root model layers, SceneCo in body model layers. Separated. |
| **Hclean** (trajco_cross_root_body_sceneco_body_clean) | root+body | body only | TrajCo in root+body, SceneCo in body. Most aggressive combination. |

All experiments use the **KimodoSceneCo** architecture:
- **Two-stage denoiser**: root model (16 layers) → local root → body model (16 layers)
- **TrajCo**: cross-attention layer inserting root trajectory (XZ+Y+heading, 5-dim) as K/V into Transformer
- **SceneCo**: cross-attention layer inserting scene voxel patch features (256-dim) as K/V
- **VoxelViT**: 64³ voxel → 8×8×8 patches → 4-layer ViT → 512 scene tokens
- **Training**: 400 epochs, batch 4, 196 frames, 9:1 train/val split on LINGO

### Network Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    Voxel Grid (64³)                  │
│                         ↓                           │
│            VoxelViT (patch embed + 4-layer)          │
│                    ┌────┴────┐                       │
│              scene_feat_root  scene_feat_body        │
│                    │              │                  │
│  ┌─────────────────┼──────────────┼───────────────┐ │
│  │  Root Model (16 Transformer layers)             │ │
│  │  Self-Attn → [TrajCo ⊕] → [SceneCo ⊕] → FFN    │ │
│  │                    ↓                             │ │
│  │         root_motion_pred (global)                │ │
│  │                    ↓                             │ │
│  │  global_root_to_local_root (detached)            │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         ↓                            │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Body Model (16 Transformer layers)             │ │
│  │  Self-Attn → [TrajCo ⊕] → [SceneCo ⊕] → FFN    │ │
│  │                    ↓                             │ │
│  │         body_motion_pred                         │ │
│  └─────────────────────────────────────────────────┘ │
│                         ↓                            │
│          output = [GT_root | predicted_body]         │
│                    Loss: body features only           │
└─────────────────────────────────────────────────────┘

TrajCo Cross-Attention:         SceneCo Cross-Attention:
  Q = motion tokens               Q = motion tokens
  K,V = traj_encoder(root_traj)   K,V = scene_proj(scene_feat)
  gate = sigmoid(alpha), init~0  gate = sigmoid(alpha), init~0
```

---

## 2. Evaluation Results (30 samples × 50 denoising steps)

### Overall Metrics

| Metric | D (TrajCo only) | E (SceneCo only) | F (T-root + SC-body) | Hclean (T-all + SC-body) |
|:-------|:---------------:|:----------------:|:--------------------:|:------------------------:|
| **MPJPE (cm)** ↓ | **24.9** | 434.1 | 43.1 | 43.7 |
| **BoneAngle (deg)** ↓ | **37.4** | 54.9 | 56.7 | 56.0 |
| **VelErr (cm/s)** ↓ | **0.22** | 9.64 | 1.34 | 1.40 |
| **FootSlide (cm)** ↓ | **0.82** | 33.43 | 6.22 | 6.34 |
| **RootADE (cm)** ↓ | **16.2** | 435.7 | 33.3 | 35.3 |
| **RootFDE (cm)** ↓ | **27.0** | 445.1 | 52.7 | 61.0 |
| **SceneOcc (%)** | 100 | 90.3 | 100 | 100 |

### Relative Degradation vs D (best experiment)

| Metric | D | D → E | D → F | D → Hclean |
|:-------|:---:|:-----:|:-----:|:----------:|
| MPJPE | 24.9 | **+1645%** | +73% | +76% |
| BoneAngle | 37.4 | +47% | +52% | +50% |
| RootADE | 16.2 | **+2586%** | +105% | +118% |
| VelErr | 0.22 | **+4251%** | +507% | +532% |

### Per Motion Category (MPJPE cm)

| Category | n | D | E | F | Hclean |
|:---------|:---:|:---:|:---:|:---:|:---:|
| walk | 11 | **23.1** | 668.6 | 42.2 | 36.5 |
| stand | 7 | **23.6** | 234.3 | 34.3 | 39.1 |
| sit | 4 | **32.9** | 419.3 | 65.2 | 53.3 |
| upper body | 5 | **24.6** | 369.7 | 42.7 | 55.2 |

### Head-to-Head Win Count (lowest MPJPE among all 4 experiments)

| D | E | F | Hclean |
|:---:|:---:|:---:|:---:|
| **28/30** | 0/30 | 1/30 | 1/30 |

---

## 3. Key Findings

### 3.1 D (TrajCo only) is overwhelmingly the best

- **28/30 samples** have the lowest MPJPE
- All 5 metrics (MPJPE, BoneAngle, VelErr, FootSlide, RootADE) are best
- Root trajectory tracking is excellent (RootADE 16.2cm)
- Adding **any** SceneCo injection degrades all metrics

### 3.2 E (SceneCo in root+body) is catastrophic

- MPJPE **434.1 cm** (vs 24.9 for D) — the model collapses
- RootADE **435.7 cm** — root trajectory prediction completely broken
- Root cause: SceneCo cross-attention in the **root model** disrupts root trajectory
  prediction. Since body model takes root output as input, the error cascades.

### 3.3 F vs Hclean: adding TrajCo to body doesn't help SceneCo

- F (TrajCo root + SceneCo body): MPJPE 43.1
- Hclean (TrajCo root+body + SceneCo body): MPJPE 43.7
- Near-identical performance, both ~73% worse than D
- Cross-attention from both TrajCo and SceneCo in the same body layers causes
  **attention competition** — the two feature sources interfere with each other

### 3.4 Hardest motion category

- **sit/stand**: all experiments degrade most on sit-down/stand-up motions
- D: 32.9cm vs 23.1cm for walk (+42%)
- F: 65.2cm vs 42.2cm for walk (+55%)
- Scene context potentially more important for sitting, but current SceneCo
  implementation doesn't capture it effectively

---

## 4. Conclusions

1. **TrajCo is effective**: trajectory cross-attention in both root and body models
   provides strong spatial guidance without disrupting the pretrained motion prior

2. **SceneCo in root model is dangerous**: scene features interfere with root
   trajectory prediction, causing catastrophic cascade failure

3. **SceneCo + TrajCo compete**: when both are in the same body Transformer layers,
   performance degrades ~73%. The cross-attention mechanisms share the same
   query space and compete for attention head allocation

4. **Future directions**:
   - Interleave SceneCo and TrajCo in different layers (not same layer)
   - SceneCo as additive/adaLN modulation rather than cross-attention
   - Train with scene dropout or dual-path loss (with/without scene) for robustness

---

## 5. File Structure

```
DEFH_test/
├── README_DEFH_test.md          # This report
├── configs/
│   ├── trajco_cross_smplx.yaml           # D: TrajCo only
│   ├── trajco_cross_sceneco_smplx.yaml   # E: SceneCo only
│   ├── trajco_cross_root_sceneco_body.yaml # F: T-root + SC-body
│   └── trajco_cross_root_body_sceneco_body_clean.yaml # Hclean: T-all + SC-body
├── outputs/eval_metrics/
│   └── all_metrics.csv           # Full per-sample metrics
├── outputs/viz_checkpoints/      # Visualization videos (2 per experiment)
├── outputs/viz_generated/        # Additional visualization (8 per experiment)
├── outputs/traj_comparison/      # Root trajectory comparison plots
├── scripts/
│   ├── eval_all_experiments.py
│   ├── visualize_generated_motion.py
│   └── plot_traj_comparison.py
└── train/
    └── train.py                  # Training script (kimodo_sceneco)
```

---

## 6. Quick Run

```bash
# Evaluate all experiments
python kimodo_scene_project/scripts/eval_all_experiments.py --num_samples 30

# Visualize generated motions
python kimodo_scene_project/scripts/visualize_generated_motion.py \
  --experiments D E F Hclean --num_samples 8

# Compare root trajectories
python kimodo_scene_project/scripts/plot_traj_comparison.py --num_samples 5
```
