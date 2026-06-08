State updated: 2026-06-08

Objective: make True Root Classifier Guidance runnable without replacing it with hand-crafted EnergyGuidance.

Current status:
- Python/YAML formatting checks pass.
- RootPathClassifier dataset now decodes normalized Kimodo cache features through `motion_rep.unnormalize()` and `motion_rep.inverse()` to meter-space `root_5d`.
- `.npz` and `.pt` cache files are supported.
- Classifier features follow the requested 20-dimensional contract from P1-A.
- Training smoke test passes with a one-batch limited run.
- Classifier-guided generation smoke test passes with a one-step local run.
- Full classifier training completed in tmux on 2026-06-08 with final validation accuracy 1.0 and AUC 1.0.
- Classifier-guided root generation completed for 30 samples under `outputs/root_classifier_guidance/path_only`.
- Rerun requested runnable-check sequence completed without full training:
  - `py_compile` passed for all critic modules, `kimodo_model.py`, and `generate_root_classifier_guidance.py`.
  - Target Python files are valid multi-line code, not compressed one-line files.
  - `configs/root_classifier.yaml` and `configs/root_classifier_guidance.yaml` pass `yaml.safe_load`.
  - Dataset smoke passes with meter-space decoded root `(B,T,5)` and 20-dim classifier features.
  - Classifier training smoke passes and now always saves an initial `best.pt`.
  - Classifier-guided generation smoke passes using the smoke checkpoint.

Notes:
- `EnergyGuidance` remains in `kimodo_sceneco/guidance/root_guidance.py` as the analytical baseline.
- Scene SDF is not enabled in the provided configs (`use_scene_sdf: false`), so the trained method should be reported as RootPathClassifier, not a full scene classifier.
- `scripts/generate_root_classifier_guidance.py` defaults to a zero text encoder fallback so local smoke tests do not require `gradio_client` or a text encoder service. Use `--real_text_encoder` to use the model-configured text encoder.
- Training checkpoints and generated roots are under ignored `outputs/` paths and are intentionally not part of the git commit unless force-added.
- Full training was not restarted during the rerun; only smoke tests were executed.

P0 syntax/YAML repair DONE:
- Phase scope: syntax and YAML repair only.
- Checked files:
  - `kimodo_sceneco/model/kimodo_model.py`
  - `kimodo_sceneco/critic/root_path_scene_classifier.py`
  - `kimodo_sceneco/critic/root_classifier_features.py`
  - `kimodo_sceneco/critic/root_classifier_dataset.py`
  - `kimodo_sceneco/critic/train_root_classifier.py`
  - `scripts/generate_root_classifier_guidance.py`
  - `configs/root_classifier.yaml`
  - `configs/root_classifier_guidance.yaml`
- `py_compile` passed for all requested Python files.
- `yaml.safe_load` passed for both requested root classifier YAML files.
- No dataset, training, classifier-guided generation, or Kimodo integration work was performed in this P0-only phase.

P1-A feature builder and classifier model DONE:
- Phase scope: feature builder and classifier model only.
- Dataset, KimodoSceneCo, and generation script were not modified in this phase.
- No training was run in this phase.
- `build_root_classifier_features(...)` now returns `(B, T, 20)` and keeps `root_y`.
- Feature order is `root_xz`, `root_y`, `target_xz`, `root_minus_target`, `dist_to_target`, `root_vel`, `target_vel`, `root_speed`, `target_speed`, `heading`, `path_dir`, `heading_path_error`, `sdf_value`.
- `RootPathSceneClassifier(input_dim=20)` returns logits with shape `(B, 1)` and does not apply sigmoid in `forward()`.
- Shape test passed with feature shape `torch.Size([2, 196, 20])` and logit shape `torch.Size([2, 1])`.
- `py_compile` passed for `root_classifier_features.py` and `root_path_scene_classifier.py`.

P1-B RootClassifierDataset DONE:
- Phase scope: `kimodo_sceneco/critic/root_classifier_dataset.py` only, plus required tracking-doc updates.
- `find_cache_files(...)` supports `.npz` and `.pt` cache files and returns `.npz` files first for `lingo_smplx_cache/seg_XXXXX.npz`.
- `load_motion_features(...)` supports `.npz`, dict-style `.pt`, and tensor `.pt` cache payloads.
- `extract_root_5d_meter(...)` always decodes normalized cache features through `motion_rep.unnormalize()` and `motion_rep.inverse(..., is_normalized=False)` before classifier use.
- `root_5d_meter` is `[smooth_root_pos_x, smooth_root_pos_y, smooth_root_pos_z, heading_cos, heading_sin]`.
- Missing inverse-output keys now raise helpful errors that include available keys.
- `target_path_xz` remains `smooth_root_pos[:, [0, 2]]`.
- Positive samples and all requested negative modes are supported: `shift`, `wrong_goal`, `jitter`, `wrong_heading`, `reverse_heading`, and `path_shuffle`.
- Added the required `root_classifier_collate_fn` name as an alias of the existing collate function.
- Did not modify `KimodoSceneCo`, the generation script, or run full model training in this phase.

P1-C classifier training smoke test DONE:
- Phase scope: `kimodo_sceneco/critic/train_root_classifier.py` only, plus required tracking-doc updates.
- `--config`, `--output_dir`, `--batch_size`, `--num_epochs`, `--lr`, and `--gpu` are supported.
- `configs/root_classifier.yaml` is loaded, with CLI flags taking precedence.
- The trainer loads the SMPL-X Kimodo `motion_rep` from `models/Kimodo-SMPLX-RP-v1`, builds `RootClassifierDataset`, and trains `RootPathSceneClassifier(input_dim=20)` with `BCEWithLogitsLoss`.
- Per-epoch logs include `train_loss`, `train_acc`, `val_loss`, `val_acc`, `positive_score_mean`, `negative_score_mean`, optional sklearn AUC, and train/val accuracy by negative mode.
- `latest.pt`, `best.pt`, `train_log.csv`, `train.log`, and `final_metrics.json` were saved under `outputs/root_path_classifier_smoke`.
- The exact smoke command was run outside the sandbox so `CUDA_VISIBLE_DEVICES=1` exposed physical GPU 1 as `cuda:0`.
- Smoke output directories automatically cap train/val batches unless explicit batch-limit flags are supplied, preventing accidental full training during smoke phases.
- Did not modify `KimodoSceneCo`, the root generation script, or run full training in this phase.

P2-A true classifier guidance integration DONE:
- Phase scope: `kimodo_sceneco/model/kimodo_model.py` plus a minimal `kimodo_sceneco/model/__init__.py` import-path bootstrap required by the exact import test.
- Dataset and classifier training script were not modified in this phase.
- Full generation was not run.
- `KimodoSceneCo.denoising_step_with_root_classifier_guidance(...)` supports the true RootPathSceneClassifier guidance path.
- The classifier step uses `x = motion.detach().requires_grad_(True)`, predicts `pred_x0`, denormalizes `pred_x0[..., self.motion_rep.root_slice]` to meter-space root, builds 20D root classifier features, applies `BCEWithLogitsLoss` against valid labels, and backpropagates through the current motion.
- Gradients are restricted to `self.motion_rep.root_slice`, non-root gradients are zeroed, clipped by `classifier_max_grad_norm`, and applied with `classifier_guidance_scale`.
- Logs include `loss_cls`, `score_valid`, `loss_total`, `grad_norm`, and `energy_*` fields when `hybrid=True`.
- Hybrid mode combines classifier loss with existing EnergyGuidance using `w_classifier` and `w_energy`; existing EnergyGuidance remains available.
- `root_classifier`, `classifier_guidance_scale`, `classifier_max_grad_norm`, `root_classifier_start_step`, `root_classifier_end_step`, `hybrid`, `w_classifier`, and `w_energy` are accepted by `__call__`, `_multiprompt`, and `_generate` with requested defaults.
- `_generate` branch priority is classifier guidance first, then EnergyGuidance, then normal denoising.
- `py_compile` passed for `kimodo_sceneco/model/kimodo_model.py`, and `KimodoSceneCo` imports successfully.

P2-B classifier-guided root generation smoke DONE:
- Phase scope: `scripts/generate_root_classifier_guidance.py` plus required tracking-doc updates.
- The script loads `configs/root_classifier_guidance.yaml`, `KimodoSceneCo`, and a `RootPathSceneClassifier` checkpoint.
- Cache discovery supports `.npz` and `.pt`, prefers `.npz`, prefers project-local `lingo_smplx_cache`, and ignores temporary cache files.
- `target_path_xz` is extracted from decoded meter-space root via `extract_root_5d_meter(model.motion_rep, motion_features, device=device)[:, [0, 2]]`; normalized motion slices are not used as the target path.
- The script passes `root_classifier`, `classifier_guidance_scale`, `classifier_max_grad_norm`, classifier step range, `hybrid`, `w_classifier`, `w_energy`, and `target_path_xz` into `KimodoSceneCo`.
- Outputs are saved as `.npz` files containing `guided_root_5d_norm`, `guided_root_5d_meter`, `target_path_xz`, `text`, `scene_name`, and `source_file`.
- `guidance_log.csv` is saved with `sample_id`, `step`, `loss_cls`, `score_valid`, `loss_total`, and `grad_norm`.
- Smoke generation on physical GPU 1 completed for 2 samples with 5 denoising steps and logged `loss_cls` plus `score_valid`.
