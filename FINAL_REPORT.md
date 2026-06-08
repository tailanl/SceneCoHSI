Final report: True Root Classifier Guidance

Implemented exactly as true classifier guidance, not as a replacement analytical EnergyGuidance method.

Changed areas:
- `kimodo_sceneco/critic/root_classifier_features.py`
  - Restored the requested 19-dimensional RootPathClassifier feature contract.
  - Handles one-frame clips safely.
- `kimodo_sceneco/critic/root_classifier_dataset.py`
  - Supports `.npz` and `.pt` cache files.
  - Extracts meter-space `root_5d` using `motion_rep.unnormalize()` and `motion_rep.inverse()`.
  - Produces positive and negative samples with padded roots, target paths, masks, labels, and source file metadata.
- `kimodo_sceneco/critic/train_root_classifier.py`
  - Builds SMPL-X Kimodo motion representation from checkpoint stats.
  - Trains RootPathClassifier on decoded meter-space roots.
  - Adds optional smoke-test batch limits while preserving full-training defaults.
- `kimodo_sceneco/model/kimodo_model.py`
  - Classifier guidance branch is wired through KimodoSceneCo generation.
  - Denoising guidance computes BCE classifier loss on predicted x0 root, backpropagates through sampling state, and applies gradient only to `root_slice`.
  - Analytical EnergyGuidance remains available only as baseline/hybrid component.
- `scripts/generate_root_classifier_guidance.py`
  - Supports `.npz` and `.pt` caches.
  - Builds `target_path_xz` from meter-space decoded root, not normalized feature columns.
  - Loads RootPathClassifier checkpoints and runs classifier-guided generation.
  - Includes a zero text encoder fallback for local smoke tests; `--real_text_encoder` uses the configured real encoder.
- `configs/root_classifier.yaml` and `configs/root_classifier_guidance.yaml`
  - Fixed to valid YAML and 19 classifier input dimensions.

Verification:
- Python compile check: passed.
- YAML parse check: passed.
- Dataset smoke test: passed on real `lingo_smplx_cache`.
- Classifier training smoke test: passed.
- Classifier-guided generation smoke test: passed.
- Full classifier training: completed with final validation accuracy 1.0 and AUC 1.0.
- 30-sample classifier-guided root generation: completed under `outputs/root_classifier_guidance/path_only`.

Remaining experimental work:
- Since `use_scene_sdf: false`, this is currently RootPathClassifier guidance, not a proven RootPathSceneClassifier.
