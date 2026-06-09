# AUTORESEARCH_STATE.md

## Status: E4_IN_PROGRESS (3 done, 1 running, 3 pending, 1 blocked)

### Experiments
| ID | Experiment | Status | Key Result |
|----|-----------|--------|------------|
| E0 | NoGuidance + Original Body | BLOCKED | scripts/generate.py missing |
| E1 | EnergyGuidance + Original Body | **DONE** | PathADE=2.10, CFR=0.96 |
| E2 | ClassifierGuidance + Original Body | **DONE** | PathADE=1.07, val_acc=1.0 |
| E3 | HybridGuidance + Original Body | **DONE** | PathADE=1.21, CFR=1.0 |
| E4 | EnergyGuidance + Stage2 SceneCo | **IN_PROGRESS** | Root gen in tmux, ~11h remaining |
| E5 | ClassifierGuidance + Stage2 SceneCo | PENDING | Blocked by E4 GPU usage |
| E6 | HybridGuidance + Stage2 SceneCo | PENDING | Blocked by E4 GPU usage |
| E7 | GTRoot + Stage2 SceneCo | ROOTS_READY | 8731 train + 1548 val GT roots exported |

### Execution Order
E1 ✓ → E4 (in progress) → E2 ✓ → E3 ✓ → (E4 full) → E5 → E6 → E7 → E0(blocked)

### Active Processes
- tmux e4_full_root: Generating E4 energy-guided train roots (~77/13911, ~3s/sample)
- E7 GT root export: COMPLETED (8731 train, 1548 val)

### Environment
- CUDA_VISIBLE_DEVICES=1, GPU arg: --gpu 0
- HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
