Task checklist for True Root Classifier Guidance

- [x] Read `README_True_Classifier_Guidance_Fixes.md`.
- [x] Fix Python/YAML formatting.
- [x] Implement RootPathClassifier dataset, features, model, and training path.
- [x] Ensure classifier input uses meter-space root decoded from motion representation, not normalized `motion[:T, :5]`.
- [x] Support `.npz` cache files and `.pt` cache files.
- [x] Integrate classifier guidance parameters and denoising branch into `KimodoSceneCo`.
- [x] Implement/fix `scripts/generate_root_classifier_guidance.py`.
- [x] Run `py_compile`.
- [x] Run YAML parse check.
- [x] Run dataset smoke test.
- [x] Run classifier training smoke test.
- [x] Run classifier-guided generation smoke test.
- [x] Maintain `AUTORESEARCH_STATE.md`, `TASK_CHECKLIST.md`, `COMMAND_LOG.md`, and `FINAL_REPORT.md`.
- [x] Complete full classifier training in `train_classifier` tmux session.
- [x] Verify `outputs/root_path_scene_classifier/best.pt` exists.
- [x] Generate 30 classifier-guided root samples.
- [x] Update `root_guidance.py` docstring to label it as the EnergyGuidance baseline.

P0 syntax/YAML repair only

- [x] Rewrite/check compressed one-line target Python files as valid multi-line Python.
- [x] Rewrite/check `configs/root_classifier.yaml` as valid multi-line YAML.
- [x] Rewrite/check `configs/root_classifier_guidance.yaml` as valid multi-line YAML.
- [x] Run requested `py_compile` command.
- [x] Run requested `yaml.safe_load` command.
- [x] Mark P0 as DONE.
- [x] Avoid dataset, classifier training, Kimodo integration, and method redesign work in this P0-only phase.

P1-A feature builder and classifier model only

- [x] Keep `root_y` in root classifier features.
- [x] Make `build_root_classifier_features(...)` output `(B, T, 20)`.
- [x] Preserve required feature order.
- [x] Verify `RootPathSceneClassifier(input_dim=20)` outputs `(B, 1)`.
- [x] Confirm classifier `forward()` returns logits without sigmoid.
- [x] Confirm masked mean pooling is used.
- [x] Run requested shape test.
- [x] Run requested `py_compile` for feature builder and classifier model.
- [x] Avoid dataset changes, KimodoSceneCo changes, generation-script changes, and training in this P1-A phase.

P1-B RootClassifierDataset only

- [x] Modify only `kimodo_sceneco/critic/root_classifier_dataset.py` for code changes.
- [x] Keep `KimodoSceneCo` unchanged.
- [x] Keep the generation script unchanged.
- [x] Do not train the full model.
- [x] Ensure `find_cache_files(...)` supports `.npz` and `.pt` and prefers `.npz`.
- [x] Ensure `load_motion_features(...)` supports `.npz` and `.pt`.
- [x] Ensure classifier `root_5d` is decoded through `motion_rep.unnormalize()` and `motion_rep.inverse()`, not normalized `motion[:T, :5]`.
- [x] Ensure `root_5d_meter` is `[smooth_root_pos_x, smooth_root_pos_y, smooth_root_pos_z, heading_cos, heading_sin]`.
- [x] Ensure `target_path_xz` is `smooth_root_pos[:, [0, 2]]`.
- [x] Support positive samples and negative modes: `shift`, `wrong_goal`, `jitter`, `wrong_heading`, `reverse_heading`, `path_shuffle`.
- [x] Add helpful inverse-output key errors.
- [x] Provide `pad_to_length(...)`, `RootClassifierDataset`, and `root_classifier_collate_fn`.
- [x] Run requested cache discovery test.
- [x] Run requested `py_compile`.
- [x] Update `COMMAND_LOG.md`, `AUTORESEARCH_STATE.md`, and `TASK_CHECKLIST.md`.

P1-C classifier training smoke test

- [x] Modify only `kimodo_sceneco/critic/train_root_classifier.py` for code changes.
- [x] Keep `KimodoSceneCo` unchanged.
- [x] Keep the root generation script unchanged.
- [x] Do not run full training.
- [x] Support `--config`.
- [x] Support `--output_dir`.
- [x] Support `--batch_size`.
- [x] Support `--num_epochs`.
- [x] Support `--lr`.
- [x] Support `--gpu`.
- [x] Load `configs/root_classifier.yaml`.
- [x] Load Kimodo SMPL-X `motion_rep` correctly.
- [x] Build `RootClassifierDataset`.
- [x] Train `RootPathSceneClassifier(input_dim=20)` with `BCEWithLogitsLoss`.
- [x] Save `latest.pt`.
- [x] Save `best.pt`.
- [x] Save `train_log.csv`.
- [x] Print per-epoch `train_loss`, `train_acc`, `val_loss`, and `val_acc`.
- [x] Print `positive_score_mean`, `negative_score_mean`, optional AUC, and accuracy by negative mode.
- [x] Skip AUC without failing if sklearn is unavailable.
- [x] Run requested `py_compile`.
- [x] Run requested GPU smoke command.
- [x] Update `COMMAND_LOG.md`, `AUTORESEARCH_STATE.md`, and `TASK_CHECKLIST.md`.

P2-A integrate true classifier guidance into KimodoSceneCo

- [x] Do not modify dataset.
- [x] Do not modify classifier training script.
- [x] Do not run full generation.
- [x] Add/update `denoising_step_with_root_classifier_guidance(...)`.
- [x] Use `x = motion.detach().requires_grad_(True)`.
- [x] Call `self.predict_x0(...)`.
- [x] Extract `root_norm = pred_x0[..., self.motion_rep.root_slice]`.
- [x] Denormalize root with `denormalize_root_5d(...)`.
- [x] Build root classifier features with `build_root_classifier_features(...)`.
- [x] Compute classifier BCE-with-logits loss against valid labels.
- [x] Support hybrid EnergyGuidance combination.
- [x] Backpropagate `loss_total` with `autograd.grad`.
- [x] Keep only root-slice gradients and zero non-root gradients.
- [x] Clip classifier-guidance gradients.
- [x] Apply `classifier_guidance_scale`.
- [x] Continue the normal diffusion sampling step.
- [x] Return logs containing `loss_cls`, `score_valid`, `loss_total`, and `grad_norm`.
- [x] Include `energy_*` logs when `hybrid=True`.
- [x] Pass root classifier parameters through `__call__`, `_multiprompt`, and `_generate`.
- [x] Preserve existing EnergyGuidance branch.
- [x] Use classifier-guidance branch before EnergyGuidance branch.
- [x] Run requested `py_compile`.
- [x] Run requested `KimodoSceneCo` import test.
- [x] Update `COMMAND_LOG.md`, `AUTORESEARCH_STATE.md`, and `TASK_CHECKLIST.md`.

P2-B classifier-guided root generation smoke test

- [x] Load `configs/root_classifier_guidance.yaml`.
- [x] Load `KimodoSceneCo`.
- [x] Load `RootPathSceneClassifier` checkpoint.
- [x] Support `.npz` and `.pt` cache files.
- [x] Prefer `.npz`.
- [x] Prefer project-local `lingo_smplx_cache`.
- [x] Build `target_path_xz` from meter-space decoded root.
- [x] Do not use normalized `motion[:T, [0,2]]` as target path.
- [x] Call model with `root_classifier`.
- [x] Call model with `classifier_guidance_scale`.
- [x] Call model with `classifier_max_grad_norm`.
- [x] Call model with classifier start/end steps.
- [x] Call model with `hybrid`, `w_classifier`, and `w_energy`.
- [x] Call model with `target_path_xz`.
- [x] Save each sample as `.npz` with required fields.
- [x] Save `guidance_log.csv` with required fields.
- [x] Run requested `py_compile`.
- [x] Run requested 2-sample smoke generation on physical GPU 1.
- [x] Verify at least 2 `.npz` outputs.
- [x] Verify log contains `loss_cls` and `score_valid`.
- [x] Update `COMMAND_LOG.md`, `AUTORESEARCH_STATE.md`, and `TASK_CHECKLIST.md`.
