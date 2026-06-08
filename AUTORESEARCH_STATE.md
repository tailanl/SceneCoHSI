State updated: 2026-06-08

Objective: make True Root Classifier Guidance runnable without replacing it with hand-crafted EnergyGuidance.

Current status:
- Python/YAML formatting checks pass.
- RootPathClassifier dataset now decodes normalized Kimodo cache features through `motion_rep.unnormalize()` and `motion_rep.inverse()` to meter-space `root_5d`.
- `.npz` and `.pt` cache files are supported.
- Classifier features follow the requested 19-dimensional contract.
- Training smoke test passes with a one-batch limited run.
- Classifier-guided generation smoke test passes with a one-step local run.
- Full classifier training completed in tmux on 2026-06-08 with final validation accuracy 1.0 and AUC 1.0.
- Classifier-guided root generation completed for 30 samples under `outputs/root_classifier_guidance/path_only`.
- Rerun requested runnable-check sequence completed without full training:
  - `py_compile` passed for all critic modules, `kimodo_model.py`, and `generate_root_classifier_guidance.py`.
  - Target Python files are valid multi-line code, not compressed one-line files.
  - `configs/root_classifier.yaml` and `configs/root_classifier_guidance.yaml` pass `yaml.safe_load`.
  - Dataset smoke passes with meter-space decoded root `(B,T,5)` and 19-dim classifier features.
  - Classifier training smoke passes and now always saves an initial `best.pt`.
  - Classifier-guided generation smoke passes using the smoke checkpoint.

Notes:
- `EnergyGuidance` remains in `kimodo_sceneco/guidance/root_guidance.py` as the analytical baseline.
- Scene SDF is not enabled in the provided configs (`use_scene_sdf: false`), so the trained method should be reported as RootPathClassifier, not a full scene classifier.
- `scripts/generate_root_classifier_guidance.py` defaults to a zero text encoder fallback so local smoke tests do not require `gradio_client` or a text encoder service. Use `--real_text_encoder` to use the model-configured text encoder.
- Training checkpoints and generated roots are under ignored `outputs/` paths and are intentionally not part of the git commit unless force-added.
- Full training was not restarted during the rerun; only smoke tests were executed.
