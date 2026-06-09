# TASK_CHECKLIST.md

## Phase 1: Global Pre-checks
- [x] 2.1 Python syntax check (py_compile all 9 files)
- [x] 2.2 YAML check (safe_load all 4 configs)
- [x] 2.3 Cache data check (lingo_smplx_cache .npz files)

## Phase 2: Experiments

### E1: EnergyGuidance + Original Body ✓
- [x] Step 1: Generate EnergyGuidance root
- [x] Checkpoint 1: Root files exist, keys correct
- [x] Step 2: Generate Original Body from root
- [x] Checkpoint 2: Root fix max error < 1e-5
- [x] Step 3: Evaluate path metrics
- [x] Step 3: Evaluate scene metrics

### E4: EnergyGuidance + Stage2 SceneCo (SMOKE DONE)
- [x] Step 1: Generate train energy root (100 smoke)
- [x] Step 1: Generate val energy root (100 smoke)
- [x] Step 2: Configure Stage2 config
- [x] Step 3: Smoke training (PASS, timed out at step 5700)
- [ ] Step 4: Formal training (needs 15584 root files, ~23h)
- [ ] Step 5: Generate & evaluate

### E2: ClassifierGuidance + Original Body ✓
- [x] E2-A Step 1: Component shape test (dim=19)
- [x] E2-A Step 2: Classifier smoke train (val_acc=0.625)
- [x] E2-A Step 3: Formal classifier train (val_acc=1.0, 100 epochs)
- [x] E2-A Checkpoint: val_acc=1.0, positive > negative
- [x] E2-B Step 1: Smoke generate classifier-guided root
- [x] E2-B Step 2: Formal generate classifier root (30 samples)
- [x] E2-C: Generate Original Body
- [x] E2-D: Evaluate

### E5: ClassifierGuidance + Stage2 SceneCo
- [ ] Step 1: Generate train classifier root
- [ ] Step 1: Generate val classifier root
- [ ] Step 2: Configure Stage2 config
- [ ] Step 3: Smoke training
- [ ] Step 4: Formal training
- [ ] Step 5: Generate & evaluate

### E3: HybridGuidance + Original Body ✓
- [x] Step 1: Generate hybrid root
- [x] Step 2: Generate Original Body
- [x] Step 3: Evaluate

### E6: HybridGuidance + Stage2 SceneCo
- [ ] Step 1: Generate hybrid train/val root
- [ ] Step 2-5: Stage2 training & eval

### E7: GTRoot + Stage2 SceneCo
- [ ] Step 1: Export GT root train/val
- [ ] Step 2-3: Stage2 training & eval

### E0: NoGuidance + Original Body
- [x] BLOCKED: scripts/generate.py missing
