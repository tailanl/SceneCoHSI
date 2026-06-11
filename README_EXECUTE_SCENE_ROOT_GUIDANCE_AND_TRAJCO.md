# Execute Scene-Aware Root Guidance and TrajCo Variants

This file is for another AI/agent that will execute the experiments later.

Repository:

```bash
cd /home/lzsh2025/kimodo-viser/kimodo_scene_project
```

Do not claim an experiment is valid just because a process is running or an output file exists. Each stage below has required validation checks.

## Variant A: Scene-Aware Root Classifier Guidance

Purpose: keep the current RootPath classifier guidance, but make the classifier scene-aware by feeding scene SDF/collision information.

### A1. Train 20-d Scene-Aware Root Classifier

Command:

```bash
python kimodo_sceneco/critic/train_root_classifier.py \
  --config configs/root_classifier_scene.yaml \
  --output_dir outputs/root_path_scene_classifier_sdf
```

Required outputs:

```text
outputs/root_path_scene_classifier_sdf/best.pt
outputs/root_path_scene_classifier_sdf/latest.pt
outputs/root_path_scene_classifier_sdf/train_log.csv
outputs/root_path_scene_classifier_sdf/final_metrics.json
```

Validation:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs/root_path_scene_classifier_sdf")
required = ["best.pt", "latest.pt", "train_log.csv", "final_metrics.json"]
for name in required:
    print(name, (root / name).exists())

log = pd.read_csv(root / "train_log.csv")
print(log.tail(3))
last = log.iloc[-1]
print("positive_score_mean", last["positive_score_mean"])
print("negative_score_mean", last["negative_score_mean"])
assert last["positive_score_mean"] > last["negative_score_mean"]
PY
```

Classifier training is only valid if:

- `best.pt`, `latest.pt`, `train_log.csv`, and `final_metrics.json` exist.
- `train_log.csv` contains `train_loss`, `val_loss`, `train_acc` or `val_acc`, `positive_score_mean`, `negative_score_mean`.
- `positive_score_mean > negative_score_mean`.

If not, mark Variant A as `INVALID` or `INCOMPLETE`.

### A2. Generate Scene-Aware Classifier-Guided Roots

Command shape:

```bash
python scripts/generate_root_classifier_guidance.py \
  --config configs/root_classifier_scene_guidance.yaml \
  --classifier_ckpt outputs/root_path_scene_classifier_sdf/best.pt \
  --output_dir outputs/root_classifier_scene_guidance \
  --split val \
  --all
```

Required outputs:

```text
outputs/root_classifier_scene_guidance/*.npz
outputs/root_classifier_scene_guidance/metadata.json
outputs/root_classifier_scene_guidance/guidance_log.csv
outputs/root_classifier_scene_guidance/scene_collision_log.csv
```

Validation:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("outputs/root_classifier_scene_guidance")
npz_files = sorted(root.glob("*.npz"))
print("num_npz", len(npz_files))
assert npz_files, "no root npz outputs"

required_keys = {
    "guided_root_5d_norm",
    "guided_root_5d_meter",
    "target_path_xz",
    "text",
    "scene_name",
    "source_file",
}
for path in npz_files[:10]:
    data = np.load(path, allow_pickle=True)
    missing = required_keys - set(data.files)
    print(path.name, "missing", missing)
    assert not missing

guidance = pd.read_csv(root / "guidance_log.csv")
print(guidance.tail())
for col in ["loss_cls", "score_valid", "grad_norm"]:
    assert col in guidance.columns
    assert guidance[col].notna().any(), col

collision = pd.read_csv(root / "scene_collision_log.csv")
print(collision.tail())
for col in ["root_collision_rate", "root_min_sdf", "root_mean_sdf", "root_sdf_penalty"]:
    assert col in collision.columns
    assert collision[col].notna().any(), col
PY
```

Root generation is only valid if:

- `guidance_log.csv` contains non-empty `loss_cls`, `score_valid`, `grad_norm`.
- Root `.npz` files contain all required keys.
- `scene_collision_log.csv` exists and contains finite collision/SDF metrics.

If `loss_cls`, `score_valid`, or `grad_norm` are missing, classifier guidance is `NOT VERIFIED`.

## Variant B: Stage2 External Root + SceneCo Body + TrajCo Body

Purpose: keep Stage2 external-root body generation, but inject the fixed root trajectory through original TrajCo layers.

Use config:

```text
configs/stage2_root_guided_sceneco_trajco.yaml
```

### B1. Train Stage2 TrajCo Variant

For E4-style generated roots:

```bash
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_root_guided_sceneco_trajco.yaml \
  --output_dir outputs/stage2_E4_sceneco_trajco \
  --path_scene_guided_root_dir outputs/root_classifier_scene_guidance \
  --val_root_dir outputs/root_classifier_scene_guidance
```

For E7-style GT root:

```bash
python train/train_stage2_root_guided_sceneco.py \
  configs/stage2_root_guided_sceneco_trajco.yaml \
  --output_dir outputs/stage2_E7_gtroot_sceneco_trajco \
  --root_mix_gt 1.0 \
  --root_mix_path 0.0 \
  --root_mix_scene 0.0
```

Required Stage2 outputs:

```text
<stage2_output>/train.log
<stage2_output>/checkpoints/best_checkpoint.pt
```

Validation:

```bash
LOG=outputs/stage2_E4_sceneco_trajco/train.log
grep -E "external_root_enabled|use_external_root|TrajCo:|external_root|fallback|missing|loss=|Saved best checkpoint" "$LOG" | tail -80
```

Stage2 is only valid if:

- log shows `external_root_enabled=True` and `use_external_root=True`.
- log shows `TrajCo: enabled=True`.
- log shows body TrajCo layers count is greater than zero.
- external root source is loaded from the expected root directory.
- fallback/missing messages are absent or rare.
- losses are finite.
- checkpoint is saved.

If many fallback or missing messages appear, write:

```text
Stage2 process is running, but experiment is not valid because external root files are not matching dataset sample ids.
```

### B2. Generate Body With Fixed Root and TrajCo

For E4-style roots:

```bash
python scripts/generate_body_from_root.py \
  --root_dir outputs/root_classifier_scene_guidance \
  --output_dir outputs/body_E4_sceneco_trajco \
  --checkpoint outputs/stage2_E4_sceneco_trajco/checkpoints/best_checkpoint.pt \
  --use_trajco \
  --trajco_body
```

For E7-style roots, replace `--root_dir` and checkpoint with the E7 root/checkpoint directory.

Required body outputs:

```text
<body_output>/*.npz
body generation log containing "Root fix max_error"
```

Validation:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path("outputs/body_E4_sceneco_trajco")
files = sorted(root.glob("*.npz"))
print("num_body_npz", len(files))
assert files, "no body outputs"
for path in files[:10]:
    data = np.load(path, allow_pickle=True)
    print(path.name, data.files)
    assert "gen_joints" in data.files
    assert "gen_root" in data.files
PY
```

The body-generation log must contain:

```text
Root fix max_error: <value>
```

Body generation is only valid if `max_error < 1e-5`.

If root fix max error is missing, write:

```text
Body generation ran, but fixed-root correctness is not verified because root fix max error is missing from log.
```

## Metrics

After any generation output exists, run the repository's metrics script used for the current experiment pipeline. The valid result must include:

```text
path_metrics.csv
scene_metrics.csv
```

If generation outputs exist but metrics are missing, mark the experiment `INCOMPLETE`.

## Required Final Report

The executing AI should report:

- Which commands were started.
- Which commands finished.
- Which logs were checked.
- Which checkpoints exist.
- Which `.npz` outputs exist and whether required keys are present.
- Whether classifier guidance is verified by `loss_cls`, `score_valid`, `grad_norm`.
- Whether Stage2 truly used `external_root`.
- Whether TrajCo body layers were active.
- Whether body root fix max error is present and `< 1e-5`.
- Whether metrics exist.
- Final status: `VALID_DONE`, `VALID_RUNNING`, `RUNNING_NOT_VERIFIED`, `INCOMPLETE`, `BLOCKED`, `INVALID`, or `NOT_STARTED`.
