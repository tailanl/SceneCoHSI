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
