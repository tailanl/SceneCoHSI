# FINAL_REPORT.md

## Status: 3 COMPLETE, 1 IN PROGRESS, 4 PENDING/BLOCKED

### Completed Experiments
| Exp | Method | Status | PathADE | PathFDE | SpeedStd | RootJerk | CFR | PenetrationRate |
|-----|--------|--------|---------|---------|----------|----------|-----|-----------------|
| E1 | EnergyGuidance + Original Body | **DONE** | 2.1033 | 3.3949 | 0.0042 | 0.0010 | 0.9617 | 0.0679 |
| E2 | ClassifierGuidance + Original Body | **DONE** | 1.0650 | 1.1514 | 0.0008 | 0.0000 | 1.0000 | 0.0846 |
| E3 | HybridGuidance + Original Body | **DONE** | 1.2114 | 1.4984 | 0.0019 | 0.0001 | 1.0000 | 0.0846 |

### In Progress
| Exp | Method | Status | Details |
|-----|--------|--------|---------|
| E4 | EnergyGuidance + Stage2 SceneCo | **IN_PROGRESS** | Root generation in tmux (e4_full_root), ~11h remaining for train set |
| E7 | GTRoot + Stage2 SceneCo | **ROOT_EXPORT_DONE** | 8731 train + 1548 val GT roots exported, Stage2 training pending |

### Pending/Blocked
| Exp | Method | Status | Blocker |
|-----|--------|--------|---------|
| E0 | NoGuidance + Original Body | BLOCKED | scripts/generate.py missing |
| E5 | ClassifierGuidance + Stage2 SceneCo | PENDING | GPU needed for root generation (blocked by E4) |
| E6 | HybridGuidance + Stage2 SceneCo | PENDING | GPU needed for root generation (blocked by E4) |

### Experiment Comparison Table
| 实验 | PathADE | PathFDE | SpeedMean | SpeedStd | RootJerk | CFR | PenetrationRate |
|---|---|---|---|---|---|---|---|
| E0 NoGuidance + Original Body | N/A (BLOCKED) | | | | | | |
| E1 EnergyGuidance + Original Body | 2.1033 | 3.3949 | 0.0268 | 0.0042 | 0.0010 | 0.9617 | 0.0679 |
| E2 ClassifierGuidance + Original Body | 1.0650 | 1.1514 | 0.0033 | 0.0008 | 0.0000 | 1.0000 | 0.0846 |
| E3 HybridGuidance + Original Body | 1.2114 | 1.4984 | 0.0087 | 0.0019 | 0.0001 | 1.0000 | 0.0846 |
| E4 EnergyGuidance + Stage2 SceneCo | IN PROGRESS | | | | | | |
| E5 ClassifierGuidance + Stage2 SceneCo | PENDING | | | | | | |
| E6 HybridGuidance + Stage2 SceneCo | PENDING | | | | | | |
| E7 GTRoot + Stage2 SceneCo | PENDING (roots ready) | | | | | | |

### Key Findings
1. **Classifier guidance (E2) significantly outperforms energy-only (E1)**: PathADE reduced from 2.10 to 1.07 (-49%).
2. **Hybrid (E3) is between E1 and E2**: PathADE=1.21, suggesting energy component adds noise.
3. **Scene metrics are similar across methods**: CollisionFrameRate ~1.0 (very high) for all, indicating original body model has inherent scene collision issues that Stage2 training should address.
4. **Classifier trained to perfection**: val_acc=1.0000, all per-mode accuracies=1.0.
5. **Root fix works**: max error = 0.0 for all body generations.

### Produced Files
**Checkpoints**:
- outputs/root_path_classifier/best.pt (val_acc=1.0)
- outputs/root_path_classifier/latest.pt
- outputs/root_path_classifier/train_log.csv

**Metrics CSV** (6 files):
- outputs/e1_energy_guidance_body/path_metrics.csv
- outputs/e1_energy_guidance_body/scene_metrics.csv
- outputs/e2_classifier_guidance_body/path_metrics.csv
- outputs/e2_classifier_guidance_body/scene_metrics.csv
- outputs/e3_hybrid_guidance_body/path_metrics.csv
- outputs/e3_hybrid_guidance_body/scene_metrics.csv

**Generated body NPZ**: 30 per experiment (E1, E2, E3)

**E7 GT root export**: 8731 train + 1548 val NPZ files

**E4 root generation** (in progress): Currently in tmux session `e4_full_root`

### Active Processes
- `tmux: e4_full_root` - Generating E4 EnergyGuidance roots for full train+val set
- Output: outputs/e4_energy_guidance_train/path_only/ and outputs/e4_energy_guidance_val/path_only/

### Environment
- CUDA_VISIBLE_DEVICES=1
- GPU arg: --gpu 0
- HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
