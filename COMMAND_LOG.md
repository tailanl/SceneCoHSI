Command log

1. `python -m py_compile kimodo_sceneco/model/kimodo_model.py kimodo_sceneco/critic/root_path_scene_classifier.py kimodo_sceneco/critic/root_classifier_features.py kimodo_sceneco/critic/root_classifier_dataset.py kimodo_sceneco/critic/train_root_classifier.py scripts/generate_root_classifier_guidance.py`
   - Result: passed.

2. YAML parse check for `configs/root_classifier.yaml` and `configs/root_classifier_guidance.yaml`
   - Result: passed. Both parsed as dictionaries with expected top-level keys.

3. Dataset smoke test using `lingo_smplx_cache`, `models/Kimodo-SMPLX-RP-v1`, `max_frames=32`
   - Result: passed.
   - Dataset length: 15584 train samples.
   - Batch root shape: `[2, 32, 5]`.
   - Classifier feature shape: `[2, 32, 19]`.

4. Classifier training smoke test
   - Command: `python kimodo_sceneco/critic/train_root_classifier.py --config configs/root_classifier.yaml --output_dir /tmp/root_path_classifier_smoke_fast --cache_dir lingo_smplx_cache --batch_size 2 --num_epochs 1 --num_workers 0 --max_frames 16 --hidden_dim 32 --num_layers 1 --num_heads 4 --max_train_batches 1 --max_val_batches 1 --gpu 0`
   - Result: passed.
   - Saved checkpoint: `/tmp/root_path_classifier_smoke_fast/best.pt`.

5. Classifier-guided generation smoke test
   - Command: `CHECKPOINT_DIR=$PWD/models python scripts/generate_root_classifier_guidance.py --config configs/root_classifier_guidance.yaml --classifier_ckpt /tmp/root_path_classifier_smoke_fast/best.pt --output_dir /tmp/root_classifier_guidance_smoke --cache_dir lingo_smplx_cache --num_samples 1 --num_denoising_steps 1 --gpu 0`
   - Result: passed.
   - Saved one guided root and metadata under `/tmp/root_classifier_guidance_smoke`.

6. Final `py_compile` rerun
   - Result: passed.

7. Final YAML parse rerun
   - Result: passed.

8. Full classifier training in tmux
   - Command: `tmux new-session -d -s train_classifier "cd ~/kimodo-viser/kimodo_scene_project && CUDA_VISIBLE_DEVICES=1 python kimodo_sceneco/critic/train_root_classifier.py --cache_dir lingo_smplx_cache --output_dir outputs/root_path_scene_classifier --batch_size 64 --num_epochs 100 --lr 1e-4 --gpu 0 2>&1 | tee outputs/root_path_scene_classifier/train.log"`
   - Result: completed.
   - Final metrics: `acc=1.0`, `auc=1.0`, `pos_score=0.9999348521232605`, `neg_score=5.87872855248861e-05`.
   - Checkpoint verified: `outputs/root_path_scene_classifier/best.pt`.

9. Classifier-guided root generation
   - Command: `python scripts/generate_root_classifier_guidance.py --classifier_ckpt outputs/root_path_scene_classifier/best.pt --output_dir outputs/root_classifier_guidance/path_only --num_samples 30 --gpu 0`
   - Result: passed.
   - Output verification: 30 `.npy` guided roots and 30 metadata entries under `outputs/root_classifier_guidance/path_only`.

10. Rerun runnable-check `py_compile` for requested files
   - Command: `python -m py_compile kimodo_sceneco/critic/*.py kimodo_sceneco/model/kimodo_model.py scripts/generate_root_classifier_guidance.py`
   - Result: passed.
   - Scope: all critic modules, `kimodo_model.py`, and `generate_root_classifier_guidance.py`.

11. Check for compressed one-line Python files
   - Command: counted lines and max line length for `kimodo_sceneco/critic/*.py`, `kimodo_sceneco/model/kimodo_model.py`, and `scripts/generate_root_classifier_guidance.py`.
   - Result: passed.
   - No target Python file is compressed into an illegal single line; all target files are valid multi-line Python.

12. YAML parse check
   - Command: `yaml.safe_load()` for `configs/root_classifier.yaml` and `configs/root_classifier_guidance.yaml`.
   - Result: passed.
   - Both configs parse to dictionaries with expected top-level keys.

13. Meter-space root and feature-dimension audit
   - Command: `rg` audit for `motion[:T,:5]`, `root_y`, `input_dim`, `extract_root_5d_meter`, and `target_path_xz`.
   - Result: passed.
   - `root_classifier_dataset.py` decodes cache features through `motion_rep.unnormalize()` plus `motion_rep.inverse()` before returning `root_5d`.
   - `generate_root_classifier_guidance.py` supports `.npz` and builds `target_path_xz` from meter-space `root_5d_meter[:, [0, 2]]`.
   - `root_y` is not included in classifier features; all classifier `input_dim` values are unified at 19.

14. Dataset smoke test rerun
   - Command: loaded `models/Kimodo-SMPLX-RP-v1` motion rep, built `RootClassifierDataset('lingo_smplx_cache', max_frames=32)`, collated a batch, and built classifier features.
   - Result: passed.
   - Dataset length: 15584.
   - Batch root shape: `(2, 32, 5)`.
   - Feature shape: `(2, 32, 19)`.
   - Meter-space root magnitude check: max abs root position about `1.547`.

15. Classifier training smoke test rerun
   - Initial result: process passed but exposed a smoke edge case where `best.pt` may not be saved when first `val_acc == 0.0`.
   - Fix: changed `best_val_acc` initialization from `0.0` to `-1.0` in `kimodo_sceneco/critic/train_root_classifier.py`, so the first validation always saves a checkpoint.
   - Rerun command: `python kimodo_sceneco/critic/train_root_classifier.py --config configs/root_classifier.yaml --output_dir /tmp/root_path_classifier_smoke_rerun2 --cache_dir lingo_smplx_cache --batch_size 2 --num_epochs 1 --num_workers 0 --max_frames 16 --hidden_dim 32 --num_layers 1 --num_heads 4 --max_train_batches 1 --max_val_batches 1 --gpu 0`
   - Result: passed and saved `/tmp/root_path_classifier_smoke_rerun2/best.pt`.

16. Classifier-guided generation smoke test rerun
   - Command: `CHECKPOINT_DIR=$PWD/models python scripts/generate_root_classifier_guidance.py --config configs/root_classifier_guidance.yaml --classifier_ckpt /tmp/root_path_classifier_smoke_rerun2/best.pt --output_dir /tmp/root_classifier_guidance_smoke_rerun --cache_dir lingo_smplx_cache --num_samples 1 --num_denoising_steps 1 --gpu 0`
   - Result: passed.
   - Saved 1 guided root to `/tmp/root_classifier_guidance_smoke_rerun`.

Notes:
- An earlier smoke attempt used the full YAML classifier size on CPU before CLI precedence was fixed. It was superseded by the passing limited smoke test above.

17. P0 syntax/YAML repair only: requested `py_compile`
   - Command:
     ```bash
     python -m py_compile \
       kimodo_sceneco/model/kimodo_model.py \
       kimodo_sceneco/critic/root_path_scene_classifier.py \
       kimodo_sceneco/critic/root_classifier_features.py \
       kimodo_sceneco/critic/root_classifier_dataset.py \
       kimodo_sceneco/critic/train_root_classifier.py \
       scripts/generate_root_classifier_guidance.py
     ```
   - Output:
     ```text
     <no output>
     ```
   - Result: passed with exit code 0.

18. P0 syntax/YAML repair only: requested YAML parse check
   - Command:
     ```bash
     python - <<'PY'
     import yaml

     for p in [
         "configs/root_classifier.yaml",
         "configs/root_classifier_guidance.yaml",
     ]:
         print("Checking:", p)
         with open(p, "r", encoding="utf-8") as f:
             cfg = yaml.safe_load(f)
         assert isinstance(cfg, dict), f"{p} did not parse as a dict"
         print("OK:", cfg.keys())
     PY
     ```
   - Output:
     ```text
     Checking: configs/root_classifier.yaml
     OK: dict_keys(['experiment', 'data', 'negative_sampling', 'model', 'training'])
     Checking: configs/root_classifier_guidance.yaml
     OK: dict_keys(['experiment', 'model', 'root_classifier', 'classifier_guidance', 'hybrid', 'generation'])
     ```
   - Result: passed with exit code 0.
   - Phase note: no dataset smoke, training smoke, generation smoke, method redesign, or integration changes were performed in this P0-only phase.

19. P1-A feature builder and classifier model shape test
   - Phase scope: feature builder and classifier model only; no dataset changes, no KimodoSceneCo changes, no generation-script changes, and no training.
   - Change summary:
     - `kimodo_sceneco/critic/root_classifier_features.py` now keeps `root_y` in the feature vector.
     - Feature order is `root_xz`, `root_y`, `target_xz`, `root_minus_target`, `dist_to_target`, `root_vel`, `target_vel`, `root_speed`, `target_speed`, `heading`, `path_dir`, `heading_path_error`, `sdf_value`.
     - Classifier feature dimension is 20.
     - `configs/root_classifier.yaml`, `configs/root_classifier_guidance.yaml`, and the `train_root_classifier.py` CLI default use `input_dim=20`.
   - Command:
     ```bash
     python - <<'PY'
     import torch
     from kimodo_sceneco.critic.root_classifier_features import build_root_classifier_features
     from kimodo_sceneco.critic.root_path_scene_classifier import RootPathSceneClassifier

     B, T = 2, 196
     root = torch.randn(B, T, 5)
     path = torch.randn(B, T, 2)
     mask = torch.ones(B, T).bool()

     feat = build_root_classifier_features(root, path)
     print("feature shape:", feat.shape)
     assert feat.shape == (B, T, 20)

     model = RootPathSceneClassifier(input_dim=20)
     logit = model(feat, mask)
     print("logit shape:", logit.shape)
     assert logit.shape == (B, 1)
     PY
     ```
   - Output:
     ```text
     feature shape: torch.Size([2, 196, 20])
     logit shape: torch.Size([2, 1])
     ```
   - Result: passed with exit code 0.

20. P1-A feature builder and classifier model `py_compile`
   - Command:
     ```bash
     python -m py_compile \
       kimodo_sceneco/critic/root_classifier_features.py \
       kimodo_sceneco/critic/root_path_scene_classifier.py
     ```
   - Output:
     ```text
     <no output>
     ```
   - Result: passed with exit code 0.
   - Additional syntax check: `python -m py_compile kimodo_sceneco/critic/train_root_classifier.py` passed after synchronizing the CLI default `input_dim` to 20.

21. P1-B RootClassifierDataset cache discovery test, run before edits
   - Phase scope: RootClassifierDataset only; no `KimodoSceneCo` changes, no generation-script changes, and no full training.
   - Command:
     ```bash
     python - <<'PY'
     from kimodo_sceneco.critic.root_classifier_dataset import find_cache_files

     files = find_cache_files("lingo_smplx_cache")
     print("Number of cache files:", len(files))
     print("First files:", files[:5])
     assert len(files) > 0
     assert any(str(f).endswith(".npz") for f in files) or any(str(f).endswith(".pt") for f in files)
     PY
     ```
   - Output:
     ```text
     Number of cache files: 17316
     First files: [PosixPath('lingo_smplx_cache/seg_00000.npz'), PosixPath('lingo_smplx_cache/seg_00001.npz'), PosixPath('lingo_smplx_cache/seg_00002.npz'), PosixPath('lingo_smplx_cache/seg_00003.npz'), PosixPath('lingo_smplx_cache/seg_00004.npz')]
     ```
   - Result: passed with exit code 0.

22. P1-B RootClassifierDataset implementation update
   - Changed only `kimodo_sceneco/critic/root_classifier_dataset.py` for code.
   - `find_cache_files(...)` now supports `.npz` and `.pt` while ordering `.npz` files first.
   - `load_motion_features(...)` supports `.npz`, dict-style `.pt`, and tensor `.pt` cache payloads.
   - `extract_root_5d_meter(...)` always uses `motion_rep.unnormalize()` followed by `motion_rep.inverse(..., is_normalized=False)` and reports available keys if expected inverse keys are missing.
   - `root_5d_meter` is decoded from `smooth_root_pos` plus heading cosine/sine, and `target_path_xz` is built from `root_5d[:, [0, 2]]`.
   - `make_negative_root_numpy(...)` supports all requested modes, including `path_shuffle`.
   - Added `root_classifier_collate_fn` as the required collate-function name.

23. P1-B RootClassifierDataset `py_compile`
   - Command:
     ```bash
     python -m py_compile kimodo_sceneco/critic/root_classifier_dataset.py
     ```
   - Output:
     ```text
     <no output>
     ```
   - Result: passed with exit code 0.
   - Rerun after adding the non-dict inverse-output guard also passed with no output.

24. P1-B RootClassifierDataset cache discovery rerun after edits
   - Command:
     ```bash
     python - <<'PY'
     from kimodo_sceneco.critic.root_classifier_dataset import find_cache_files

     files = find_cache_files("lingo_smplx_cache")
     print("Number of cache files:", len(files))
     print("First files:", files[:5])
     assert len(files) > 0
     assert any(str(f).endswith(".npz") for f in files) or any(str(f).endswith(".pt") for f in files)
     PY
     ```
   - Output:
     ```text
     Number of cache files: 17316
     First files: [PosixPath('lingo_smplx_cache/seg_00000.npz'), PosixPath('lingo_smplx_cache/seg_00001.npz'), PosixPath('lingo_smplx_cache/seg_00002.npz'), PosixPath('lingo_smplx_cache/seg_00003.npz'), PosixPath('lingo_smplx_cache/seg_00004.npz')]
     ```
   - Result: passed with exit code 0.

25. P1-B RootClassifierDataset collate alias check
   - Command:
     ```bash
     python - <<'PY'
     from kimodo_sceneco.critic.root_classifier_dataset import root_classifier_collate_fn, collate_root_classifier
     assert root_classifier_collate_fn is collate_root_classifier
     print('collate alias ok')
     PY
     ```
   - Output:
     ```text
     collate alias ok
     ```
   - Result: passed with exit code 0.

26. P1-C classifier training smoke `py_compile`
   - Phase scope: classifier trainer smoke only; no `KimodoSceneCo` changes, no root generation script changes, and no full training.
   - Command:
     ```bash
     python -m py_compile kimodo_sceneco/critic/train_root_classifier.py
     ```
   - Output:
     ```text
     <no output>
     ```
   - Result: passed with exit code 0.

27. P1-C initial smoke attempt inside sandbox
   - Command:
     ```bash
     export CUDA_VISIBLE_DEVICES=1
     export CHECKPOINT_DIR=$PWD/models

     python kimodo_sceneco/critic/train_root_classifier.py \
       --config configs/root_classifier.yaml \
       --output_dir outputs/root_path_classifier_smoke \
       --batch_size 4 \
       --num_epochs 1 \
       --lr 1e-4 \
       --gpu 0 \
       2>&1 | tee outputs/root_path_classifier_smoke/train.log
     ```
   - Result: the smoke started, loaded datasets, and built the classifier, but the sandbox exposed no CUDA device, so the trainer logged `Device: cpu` and did not reach epoch completion in the allotted time.
   - Follow-up fix: smoke output directories now cap train/val batches and set `num_workers=0` unless explicit override flags are supplied, preventing accidental full training during smoke phases.

28. P1-C classifier training smoke command on physical GPU 1
   - Command:
     ```bash
     export CUDA_VISIBLE_DEVICES=1
     export CHECKPOINT_DIR=$PWD/models

     python kimodo_sceneco/critic/train_root_classifier.py \
       --config configs/root_classifier.yaml \
       --output_dir outputs/root_path_classifier_smoke \
       --batch_size 4 \
       --num_epochs 1 \
       --lr 1e-4 \
       --gpu 0 \
       2>&1 | tee outputs/root_path_classifier_smoke/train.log
     ```
   - Result: passed with exit code 0 after running outside the sandbox so physical GPU 1 was exposed as `cuda:0`.
   - Key output:
     ```text
     Smoke output_dir detected; using max_train_batches=2, max_val_batches=2, num_workers=0.
     Device: cuda:0
     Train: 15584, Val: 1732
     Parameters: 3,297,281
     [Epoch   1/1] train_loss=0.7593 train_acc=0.2500 val_loss=0.6930 val_acc=0.5000 positive_score_mean=0.5069 negative_score_mean=0.5108 AUC=0.4667
     [Epoch   1/1] train_negative_mode_acc=positive:0.4000, reverse_heading:0.0000, shift:0.0000, total:0.2500, wrong_goal:0.0000
     [Epoch   1/1] val_negative_mode_acc=jitter:0.0000, positive:0.8000, shift:0.0000, total:0.5000, wrong_heading:0.0000
     Saved best checkpoint (val_acc=0.5000)
     Done. best_val_acc=0.5000
     ```

29. P1-C smoke artifact verification
   - Command:
     ```bash
     python - <<'PY'
     from pathlib import Path
     out = Path('outputs/root_path_classifier_smoke')
     for name in ['latest.pt', 'best.pt', 'train_log.csv', 'train.log']:
         p = out / name
         print(name, p.exists(), p.stat().st_size if p.exists() else 0)
     assert (out / 'latest.pt').exists()
     assert (out / 'best.pt').exists()
     assert (out / 'train_log.csv').exists()
     PY
     ```
   - Output:
     ```text
     latest.pt True 39634654
     best.pt True 39628086
     train_log.csv True 196
     train.log True 2948
     ```
   - `train_log.csv` content:
     ```text
     epoch,train_loss,train_acc,val_loss,val_acc,positive_score_mean,negative_score_mean,auc
     1,0.7592686712741852,0.25,0.6929715871810913,0.5,0.5068873763084412,0.5108336210250854,0.4666666666666667
     ```
   - Result: passed with exit code 0.

30. P2-A true classifier guidance integration syntax check
   - Phase scope: true classifier guidance integration into `KimodoSceneCo`; dataset and classifier training script were not modified, and full generation was not run.
   - Change summary:
     - Updated `kimodo_sceneco/model/kimodo_model.py` classifier-guidance step to use `x = motion.detach().requires_grad_(True)`, `self.predict_x0(...)`, meter-space root denormalization, classifier BCE loss, root-only gradient masking, gradient clipping, and normal DDIM sampling continuation.
     - Added `grad_norm` to classifier-guidance logs and preserved `energy_*` logs for hybrid mode.
     - Passed external-root state through the classifier-guidance path.
     - Set requested public defaults for `root_classifier_end_step=40` in `__call__`, `_multiprompt`, and `_generate`.
     - Added a minimal `kimodo_sceneco/model/__init__.py` bootstrap so the exact import test can find the sibling `kimodo` package without external `PYTHONPATH`.
   - Command:
     ```bash
     python -m py_compile kimodo_sceneco/model/kimodo_model.py
     ```
   - Output:
     ```text
     <no output>
     ```
   - Result: passed with exit code 0.

31. P2-A KimodoSceneCo import test
   - Initial result: failed before loading `kimodo_model.py` because `kimodo_sceneco/model/__init__.py` imported `scene_encoder.py`, which imports `kimodo.tools`, and the sibling `../kimodo` package was not on `sys.path`.
   - Fix: added a minimal import-path bootstrap in `kimodo_sceneco/model/__init__.py`.
   - Command:
     ```bash
     python - <<'PY'
     from kimodo_sceneco.model.kimodo_model import KimodoSceneCo
     print("KimodoSceneCo import OK")
     PY
     ```
   - Output:
     ```text
     KimodoSceneCo import OK
     ```
   - Result: passed with exit code 0.

32. P2-A root classifier API signature check
   - Command:
     ```bash
     python - <<'PY'
     import inspect
     from kimodo_sceneco.model.kimodo_model import KimodoSceneCo
     required = {
         'root_classifier': None,
         'classifier_guidance_scale': 0.05,
         'classifier_max_grad_norm': 1.0,
         'root_classifier_start_step': 0,
         'root_classifier_end_step': 40,
         'hybrid': False,
         'w_classifier': 1.0,
         'w_energy': 0.3,
     }
     for name in ['__call__', '_multiprompt', '_generate']:
         sig = inspect.signature(getattr(KimodoSceneCo, name))
         for key, expected in required.items():
             param = sig.parameters[key]
             print(name, key, param.default)
             assert param.default == expected
     print('root classifier signatures OK')
     PY
     ```
   - Output:
     ```text
     __call__ root_classifier None
     __call__ classifier_guidance_scale 0.05
     __call__ classifier_max_grad_norm 1.0
     __call__ root_classifier_start_step 0
     __call__ root_classifier_end_step 40
     __call__ hybrid False
     __call__ w_classifier 1.0
     __call__ w_energy 0.3
     _multiprompt root_classifier None
     _multiprompt classifier_guidance_scale 0.05
     _multiprompt classifier_max_grad_norm 1.0
     _multiprompt root_classifier_start_step 0
     _multiprompt root_classifier_end_step 40
     _multiprompt hybrid False
     _multiprompt w_classifier 1.0
     _multiprompt w_energy 0.3
     _generate root_classifier None
     _generate classifier_guidance_scale 0.05
     _generate classifier_max_grad_norm 1.0
     _generate root_classifier_start_step 0
     _generate root_classifier_end_step 40
     _generate hybrid False
     _generate w_classifier 1.0
     _generate w_energy 0.3
     root classifier signatures OK
     ```
   - Result: passed with exit code 0.

33. P2-B classifier-guided generation script syntax check
   - Phase scope: classifier-guided root generation smoke; no full generation was run.
   - Change summary:
     - `scripts/generate_root_classifier_guidance.py` now saves per-sample `.npz` files with `guided_root_5d_norm`, `guided_root_5d_meter`, `target_path_xz`, `text`, `scene_name`, and `source_file`.
     - Added `guidance_log.csv` collection by wrapping `KimodoSceneCo.denoising_step_with_root_classifier_guidance(...)` and recording returned guidance metrics.
     - Cache discovery supports `.npz` and `.pt`, prefers `.npz`, prefers project-local `lingo_smplx_cache`, and filters temporary cache files.
     - `target_path_xz` is built from meter-space decoded root via `extract_root_5d_meter(...)[:, [0, 2]]`, not normalized motion slices.
     - Classifier checkpoint loading defaults to `input_dim=20` when the checkpoint does not include an `input_dim`.
   - Command:
     ```bash
     python -m py_compile scripts/generate_root_classifier_guidance.py
     ```
   - Output:
     ```text
     <no output>
     ```
   - Result: passed with exit code 0.

34. P2-B cache discovery check after script update
   - Command:
     ```bash
     python - <<'PY'
     from scripts.generate_root_classifier_guidance import find_cache_files
     files = find_cache_files('lingo_smplx_cache')
     print(len(files))
     print(files[:3])
     PY
     ```
   - Output:
     ```text
     17316
     [PosixPath('lingo_smplx_cache/seg_00000.npz'), PosixPath('lingo_smplx_cache/seg_00001.npz'), PosixPath('lingo_smplx_cache/seg_00002.npz')]
     ```
   - Result: passed with exit code 0.

35. P2-B classifier-guided root generation smoke on physical GPU 1
   - Command:
     ```bash
     export CUDA_VISIBLE_DEVICES=1
     export CHECKPOINT_DIR=$PWD/models

     python scripts/generate_root_classifier_guidance.py \
       --config configs/root_classifier_guidance.yaml \
       --classifier_ckpt outputs/root_path_classifier_smoke/best.pt \
       --output_dir outputs/root_classifier_guidance_smoke \
       --num_samples 2 \
       --num_denoising_steps 5 \
       --gpu 0 \
       2>&1 | tee outputs/root_classifier_guidance_smoke/generate.log
     ```
   - Result: passed with exit code 0 after running outside the sandbox so physical GPU 1 was exposed as `cuda:0`.
   - Key output:
     ```text
     Loading KimodoSceneCo model
     Loading RootPathSceneClassifier from outputs/root_path_classifier_smoke/best.pt
     Found 17316 cache files
     sample_id=0 step=4 loss_cls=0.5307057499885559 score_valid=0.5881897211074829 loss_total=0.5307057499885559 grad_norm=0.002069593872874975
     sample_id=0 step=0 loss_cls=0.5400139689445496 score_valid=0.5827401280403137 loss_total=0.5400139689445496 grad_norm=0.00017742495401762426
     sample_id=1 step=4 loss_cls=0.525679349899292 score_valid=0.591153621673584 loss_total=0.525679349899292 grad_norm=0.0037872132379561663
     sample_id=1 step=0 loss_cls=0.5386765599250793 score_valid=0.5835199952125549 loss_total=0.5386765599250793 grad_norm=0.0013910789275541902
     Done. Saved 2 guided roots to outputs/root_classifier_guidance_smoke
     ```

36. P2-B smoke artifact verification
   - Command:
     ```bash
     python - <<'PY'
     from pathlib import Path
     import numpy as np
     out = Path('outputs/root_classifier_guidance_smoke')
     expected = [out / 'seg_00000_0000.npz', out / 'seg_00001_0001.npz']
     print('all_npz_count', len(list(out.glob('*.npz'))))
     for p in expected:
         print('checking', p.name, p.exists())
         assert p.exists()
         data = np.load(p, allow_pickle=True)
         print('keys', sorted(data.files))
         for key in ['guided_root_5d_norm','guided_root_5d_meter','target_path_xz','text','scene_name','source_file']:
             assert key in data, (p, key)
         print('source', data['source_file'])
         print('shapes', data['guided_root_5d_norm'].shape, data['guided_root_5d_meter'].shape, data['target_path_xz'].shape)
         assert data['guided_root_5d_norm'].shape[-1] == 5
         assert data['guided_root_5d_meter'].shape[-1] == 5
         assert data['target_path_xz'].shape[-1] == 2
     assert (out / 'guidance_log.csv').exists()
     PY
     ```
   - Output:
     ```text
     all_npz_count 3
     checking seg_00000_0000.npz True
     keys ['guided_root_5d_meter', 'guided_root_5d_norm', 'scene_name', 'source_file', 'target_path_xz', 'text']
     source /home/lzsh2025/kimodo-viser/kimodo_scene_project/lingo_smplx_cache/seg_00000.npz
     shapes (161, 5) (161, 5) (161, 2)
     checking seg_00001_0001.npz True
     keys ['guided_root_5d_meter', 'guided_root_5d_norm', 'scene_name', 'source_file', 'target_path_xz', 'text']
     source /home/lzsh2025/kimodo-viser/kimodo_scene_project/lingo_smplx_cache/seg_00001.npz
     shapes (163, 5) (163, 5) (163, 2)
     ```
   - `guidance_log.csv` starts with:
     ```text
     sample_id,step,loss_cls,score_valid,loss_total,grad_norm
     0,4,0.5307057499885559,0.5881897211074829,0.5307057499885559,0.002069593872874975
     0,3,0.5398397445678711,0.5828416347503662,0.5398397445678711,4.992572212358937e-05
     ```
   - Result: passed with exit code 0.
   - Note: `all_npz_count` is 3 because one stale output from an earlier failed attempt remains in the smoke directory; the two verified fresh outputs are `seg_00000_0000.npz` and `seg_00001_0001.npz`.
