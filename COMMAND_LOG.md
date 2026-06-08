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

Notes:
- An earlier smoke attempt used the full YAML classifier size on CPU before CLI precedence was fixed. It was superseded by the passing limited smoke test above.
