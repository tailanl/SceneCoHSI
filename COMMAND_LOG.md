# COMMAND_LOG.md

## Global Pre-checks
| # | Command | Result | Key Output | Next Action |
|---|---------|--------|------------|-------------|
| 2.1 | py_compile on 9 files | PASS | All compiled | Proceed |
| 2.2 | YAML safe_load on 4 configs | PASS | All valid dicts | Proceed |
| 2.3 | Cache check | PASS | 17316 .npz files | Start E1 |

## E1: EnergyGuidance + Original Body
| # | Command | Result | Key Output | Next Action |
|---|---------|--------|------------|-------------|
| 1 | generate_root_guidance.py 30 samples | PASS | 30 npz in e1_energy_guidance_root | Checkpoint 1 |
| C1 | Check root keys | PASS | gen_root, gt_root_xz (actual keys) | Step 2 |
| 2 | generate_body_from_root.py (fixed normalize) | PASS | 30 body npz, root fix max=0.0 | Step 3 |
| 3a | eval_path_metrics.py | PASS | PathADE=2.10, PathFDE=3.39 | Step 3b |
| 3b | eval_sceneadapt_metrics.py | PASS | CFR=0.962, PenetrationRate=0.068 | Complete |

**E1 COMPLETED** ✓

## E4: EnergyGuidance + Stage2 SceneCo
| # | Command | Result | Key Output | Next Action |
|---|---------|--------|------------|-------------|
| 1a | generate train root 100 samples | PASS | 100 npz with guided_root_5d_norm | Step 1b |
| 1b | generate val root 100 samples | PASS | 100 npz with guided_root_5d_norm | Step 3 |
| 3 | Smoke training | PASS (partial, timed out at step 5700) | val_loss decreased 41→17 | Full train in tmux |
| C | Fallback check | WARN | 100/15584 root files (most batches fallback) | Need full root set |

**E4 SMOKE PASS, FULL TRAINING NEEDS ALL ROOT FILES IN TMUX (~23h)**

## E2: ClassifierGuidance + Original Body (IN PROGRESS)
