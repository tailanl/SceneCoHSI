# NEXT_ACTIONS.md

## Currently Running (4 tmux sessions)

### E4: EnergyGuidance + Stage2 SceneCo
- **tmux**: `e4_full_root` on GPU 1
- **Status**: Generating energy-guided train roots (777/13911, ~3s/sample, ~11h remaining)
- **After completion**: Train+val roots done → start Stage2 full training

### E5: ClassifierGuidance + Stage2 SceneCo  
- **tmux**: `e5_roots_train` on GPU 4
- **Status**: Generating classifier-guided train roots (279/15584, ~4s/sample, ~17h remaining)
- After train roots: val roots → Stage2 training

### E6: HybridGuidance + Stage2 SceneCo
- **tmux**: `e6_roots_train` on GPU 5
- **Status**: Generating hybrid-guided train roots (281/15584, ~4s/sample, ~17h remaining)
- After: val roots → Stage2 training

### E7: GTRoot + Stage2 SceneCo
- **tmux**: `e7_stage2_full` on GPU 3
- **Status**: Full Stage2 training, epoch 1/400 (~6 min/epoch, ~40h total)
- GT roots: 8731 train + 1548 val already exported

## Free GPUs: 6, 7
- GPU 6: Ready for E5 Stage2 training (once roots done)
- GPU 7: Ready for E6 Stage2 training (once roots done)

## When each finishes:
1. E5 roots done → start `e5_stage2_full` on GPU 6
2. E6 roots done → start `e6_stage2_full` on GPU 7  
3. E4 roots done → start `e4_stage2_full` on whichever GPU frees up first
4. E7 training done → generate body → evaluate
5. All Stage2 trainings done → final eval for each

## Completed (3/8):
- E1: PathADE=2.10 ✓
- E2: PathADE=1.07 ✓
- E3: PathADE=1.21 ✓
- E0: BLOCKED
