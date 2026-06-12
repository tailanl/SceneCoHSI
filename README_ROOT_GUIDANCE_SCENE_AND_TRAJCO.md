# Root Guidance Scene Classifier and TrajCo Variants

This README records the two code variants added after the E4/E7 audit.

## Variant A: scene-aware RootPath classifier guidance

Goal: keep the current classifier-based root guidance path, but give the classifier scene collision context and write collision-check logs.

Main files:

- `configs/root_classifier_scene.yaml`
- `configs/root_classifier_scene_guidance.yaml`
- `kimodo_sceneco/critic/root_classifier_dataset.py`
- `kimodo_sceneco/critic/train_root_classifier.py`
- `kimodo_sceneco/model/kimodo_model.py`
- `scripts/generate_root_classifier_guidance.py`

What changed:

- Classifier training can enable `data.use_scene_sdf: true`.
- The classifier feature dimension becomes 20 when scene SDF is present.
- A new `scene_collision` negative mode shifts root paths into occupied/low-SDF scene cells.
- Classifier guidance now passes `scene_sdf` into `build_root_classifier_features()`, so a 20-d scene-aware checkpoint actually receives scene input.
- Generation writes `scene_collision_log.csv` with `root_collision_rate`, `root_min_sdf`, `root_mean_sdf`, and `root_sdf_penalty`.

Expected commands:

```bash
python kimodo_sceneco/critic/train_root_classifier.py \
  --config configs/root_classifier_scene.yaml \
  --output_dir outputs/root_path_scene_classifier_sdf

python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_scene_guidance.yaml \
  --output_dir outputs/root_classifier_scene_guidance \
  --classifier_ckpt outputs/root_path_scene_classifier_sdf/best.pt
```

Validity checks:

- `outputs/root_path_scene_classifier_sdf/best.pt` exists.
- `outputs/root_path_scene_classifier_sdf/latest.pt` exists.
- `outputs/root_path_scene_classifier_sdf/train_log.csv` has `train_loss`, `val_loss`, `train_acc` or `val_acc`, `positive_score_mean`, `negative_score_mean`.
- `positive_score_mean > negative_score_mean`.
- Guidance logs contain `loss_cls`, `score_valid`, and `grad_norm`.
- Generated root `.npz` files contain `guided_root_5d_norm`, `guided_root_5d_meter`, `target_path_xz`, `text`, `scene_name`, and `source_file`.
- `scene_collision_log.csv` exists and has finite collision/SDF values.

Important: old 19-d classifier checkpoints are not scene-aware. A scene-aware run requires a new 20-d classifier checkpoint trained with `configs/root_classifier_scene.yaml`.

## Variant B: Stage2 external_root + original TrajCo

Goal: keep Stage2 root-guided SceneCo, but also inject the fixed external root trajectory through the original TrajCo mechanism.

Main files:

- `configs/stage2_root_guided_sceneco_trajco.yaml`
- `kimodo_sceneco/model/kimodo_model.py`
- `kimodo_sceneco/model/cfg.py`
- `kimodo_sceneco/model/trajco_layers.py`
- `train/train_stage2_root_guided_sceneco.py`
- `scripts/generate_body_from_root.py`

What changed:

- `KimodoSceneCo` now supports `use_trajco`, `use_trajco_root`, `use_trajco_body`, `encode_traj()`, and TrajCo layer injection in the canonical model.
- CFG now duplicates `traj_feats/traj_mask` when it expands the batch for classifier-free guidance.
- Stage2 training reads the `trajco:` block and encodes `external_root` as TrajCo input.
- Body generation can instantiate the TrajCo architecture and pass encoded `external_root` into every denoising step.
- Stage2 checkpoint loading in body generation now reads `model_state_dict` when present.

Stage2 training command shape:

```bash
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_root_guided_sceneco_trajco.yaml \
  --output_dir outputs/stage2_E4_sceneco_trajco \
  --path_scene_guided_root_dir <E4_root_dir> \
  --val_root_dir <E4_val_root_dir>
```

For an E7/GT-root variant, use the same config but force GT roots in the mix:

```bash
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_root_guided_sceneco_trajco.yaml \
  --output_dir outputs/stage2_E7_gtroot_sceneco_trajco \
  --root_mix_gt 1.0 \
  --root_mix_path 0.0 \
  --root_mix_scene 0.0
```

Body generation command shape:

```bash
python scripts/generate_body_from_root.py \
  --root_dir <root_npz_dir> \
  --output_dir <body_output_dir> \
  --checkpoint outputs/stage2_E4_sceneco_trajco/checkpoints/best_checkpoint.pt \
  --use_trajco \
  --trajco_body
```

Validity checks:

- Stage2 log must show `external_root_enabled=True` and `use_external_root=True`.
- Stage2 log must show `TrajCo: enabled=True ... body=True ... layers(..., body>0)`.
- Stage2 dataset logs must show external root sources, not many fallback messages.
- Loss must be finite and checkpoints must be saved.
- Body generation log must contain `Root fix max_error`, and the max error must be `< 1e-5`.

## Checks already run

No training or generation was started. Only light checks were run:

```bash
python -c "... compile modified Python files ..."
python -c "... parse new YAML configs ..."
python kimodo_sceneco/critic/train_root_classifier.py --help
python scripts/generate_root_classifier_guidance.py --help
python train/train_stage2_root_guided_sceneco.py --help
python scripts/generate_body_from_root.py --help
```

The feature check confirmed classifier features are 19-d without scene SDF and 20-d with scene SDF.
