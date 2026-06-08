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
