# TASK_CHECKLIST.md

## Phase 1: Global Pre-checks
- [x] 2.1 Python syntax check (py_compile all 9 files)
- [x] 2.2 YAML check (safe_load all 4 configs)
- [x] 2.3 Cache data check (lingo_smplx_cache .npz files)

## Phase 2: Experiments

### E1: EnergyGuidance + Original Body ✓
- [x] Step 1: Generate EnergyGuidance root (30 samples, ~3 min)
- [x] Checkpoint 1: Root files exist, keys correct
- [x] Step 2: Generate Original Body from root
- [x] Checkpoint 2: Root fix max error = 0.0 (< 1e-5)
- [x] Step 3: Evaluate path metrics (PathADE=2.10)
- [x] Step 3: Evaluate scene metrics (CFR=0.96)

### E4: EnergyGuidance + Stage2 SceneCo (IN PROGRESS)
- [x] Step 1: Generate train energy root (100 smoke samples)
- [x] Step 1: Generate val energy root (100 smoke samples)
- [x] Step 2: Configure Stage2 config (copied)
- [x] Step 3: Smoke training (PASS: loss decreases, val_loss 41→17)
- [/] Step 4: Formal training - root generation in tmux (~11h remaining)
- [ ] Step 5: Generate & evaluate

### E2: ClassifierGuidance + Original Body ✓
- [x] E2-A Step 1: Component shape test (dim=19, fixed config)
- [x] E2-A Step 2: Classifier smoke train (val_acc=0.625)
- [x] E2-A Step 3: Formal classifier train (val_acc=1.000, 100 epochs)
- [x] E2-A Checkpoint: val_acc=1.0, positive > negative ✓
- [x] E2-B Step 1: Smoke generate classifier-guided root (2 samples)
- [x] E2-B Step 2: Formal generate classifier root (30 samples)
- [x] E2-C: Generate Original Body (root fix=0.0)
- [x] E2-D: Evaluate (PathADE=1.07, CFR=1.0)

### E5: ClassifierGuidance + Stage2 SceneCo
- [ ] Step 1: Generate train classifier root (blocked by E4 GPU)
- [ ] Step 2-5: Stage2 training & eval

### E3: HybridGuidance + Original Body ✓
- [x] Step 1: Generate hybrid root (30 samples)
- [x] Step 2: Generate Original Body (root fix=0.0)
- [x] Step 3: Evaluate (PathADE=1.21, CFR=1.0)

### E6: HybridGuidance + Stage2 SceneCo
- [ ] Step 1: Generate hybrid train/val root (blocked by E4 GPU)
- [ ] Step 2-5: Stage2 training & eval

### E7: GTRoot + Stage2 SceneCo
- [x] Step 1: Export GT root train (8731 files, CPU-only)
- [x] Step 1: Export GT root val (1548 files, CPU-only)
- [ ] Step 2-3: Stage2 training & eval (blocked by E4 GPU)

### E0: NoGuidance + Original Body
- [x] BLOCKED: scripts/generate.py missing
