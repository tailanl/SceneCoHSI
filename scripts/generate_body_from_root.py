"""Generate body motion from a fixed/guided root trajectory.

This script takes guided root NPZ files (from generate_root_guidance.py) and
generates body motion by fixing root_slice at every denoising step.

Key behavior:
- Loads base Kimodo via load_model(), wraps in KimodoSceneCo for external_root support.
- Each denoising step: fix_root_each_step forces cur_mot[root_slice] = external_root.
- The CFG denoiser receives external_root / use_external_root=True so the
  body model conditions on the external root (skipping root_model entirely).
- Verifies max_abs(final_root - external_root) < 1e-5.

Usage:
    python scripts/generate_body_from_root.py \
        --root_dir outputs/guidance_path_only \
        --output_dir outputs/guidance_path_body \
        --gpu 0
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent  # kimodo_scene_project/
sys.path.insert(0, str(PROJECT_DIR))  # our code FIRST (overrides kimodo/kimodo_sceneco)
sys.path.insert(1, str(PROJECT_DIR.parent / "kimodo"))  # kimodo/ SECOND

import os

os.environ["CHECKPOINT_DIR"] = str(PROJECT_DIR / "models")

import numpy as np
import torch
from tqdm.auto import tqdm

from kimodo.model.load_model import load_model
from kimodo.sanitize import sanitize_texts
from kimodo.motion_rep.feature_utils import length_to_mask

from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

log = logging.getLogger(__name__)


def load_kimodo_sceneco(
    model_ckpt,
    device,
    checkpoint=None,
    use_trajco=False,
    use_trajco_root=False,
    use_trajco_body=False,
    trajco_type="cross_attn",
    trajco_dropout=0.1,
):
    """Load base Kimodo + wrap in KimodoSceneCo for full external_root support.
    Optionally load a Stage2 fine-tuned checkpoint.

    Returns:
        KimodoSceneCo wrapper with patched denoiser forward (external_root/use_external_root).
    """
    log.info(f"Loading base Kimodo from {model_ckpt}...")
    base_model = load_model(model_ckpt, device="cpu")
    base_model.eval()
    base_model = base_model.to(device)

    model = KimodoSceneCo(
        denoiser=base_model.denoiser.model if hasattr(base_model.denoiser, "model") else base_model.denoiser,
        text_encoder=base_model.text_encoder,
        num_base_steps=1000,
        device=device,
        cfg_type="separated",
        use_trajco=use_trajco,
        use_trajco_root=use_trajco_root,
        use_trajco_body=use_trajco_body,
        traj_dim=5,
        trajco_type=trajco_type,
        trajco_dropout=trajco_dropout,
    )

    if checkpoint is not None:
        log.info(f"Loading Stage2 checkpoint from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    log.info(
        "KimodoSceneCo wrapper loaded (external_root/use_external_root enabled, "
        "trajco=%s root=%s body=%s type=%s)",
        use_trajco,
        use_trajco_root,
        use_trajco_body,
        trajco_type,
    )
    return model


def root_5d_from_meter(root_meter_3d, heading_2d, motion_rep):
    """Convert meter-space root positions + heading to normalized 5D root."""
    root_5d = np.concatenate([root_meter_3d, heading_2d], axis=-1)  # (T, 5)
    root_5d_t = torch.from_numpy(root_5d).float().unsqueeze(0)  # (1, T, 5)
    T = root_5d_t.shape[1]
    full_dim = motion_rep.motion_rep_dim
    full_tensor = torch.zeros(1, T, full_dim)
    full_tensor[:, :, :5] = root_5d_t
    norm_full = motion_rep.normalize(full_tensor)
    return norm_full[:, :, :5]


def heading_from_path(root_xz):
    """Compute heading [cos, sin] from XZ path."""
    if isinstance(root_xz, np.ndarray):
        root_xz = torch.from_numpy(root_xz).float()
    vel = root_xz[1:] - root_xz[:-1]
    theta = torch.atan2(vel[:, 1], vel[:, 0])
    theta = torch.cat([theta, theta[-1:]])
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1).numpy()


def main():
    parser = argparse.ArgumentParser(description="Body generation from fixed root")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, default="kimodo-smplx-rp")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--cfg_weight", type=float, nargs="+", default=[2.0, 2.0])
    parser.add_argument("--fix_root_each_step", action="store_true", default=True)
    parser.add_argument("--use_trajco", action="store_true")
    parser.add_argument("--trajco_root", action="store_true")
    parser.add_argument("--trajco_body", action="store_true")
    parser.add_argument("--trajco_type", type=str, default="cross_attn")
    parser.add_argument("--trajco_dropout", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model with KimodoSceneCo wrapper
    model = load_kimodo_sceneco(
        args.model_ckpt,
        device,
        checkpoint=args.checkpoint,
        use_trajco=args.use_trajco or args.trajco_root or args.trajco_body,
        use_trajco_root=args.trajco_root,
        use_trajco_body=args.trajco_body,
        trajco_type=args.trajco_type,
        trajco_dropout=args.trajco_dropout,
    )

    root_slice = model.motion_rep.root_slice
    cfg_weight = args.cfg_weight
    num_denoising_steps = args.num_denoising_steps
    num_denoising_steps_t = torch.tensor([num_denoising_steps], device=device)

    # Find root NPZ files
    root_dir = Path(args.root_dir)
    npz_files = sorted(root_dir.glob("sample_*.npz"))
    if not npz_files:
        npz_files = sorted(root_dir.glob("seg_*.npz"))
    if not npz_files:
        # Try flat directory
        npz_files = sorted(root_dir.glob("*.npz"))
    if args.max_samples is not None and args.max_samples > 0:
        npz_files = npz_files[: args.max_samples]
    log.info(f"Found {len(npz_files)} root files in {root_dir}")

    root_fix_errors = []

    for npz_file in tqdm(npz_files, desc="Generating body"):
        out_file = output_dir / npz_file.name
        if args.skip_existing and out_file.exists():
            log.info(f"Skipping existing {out_file}")
            continue
        data = np.load(str(npz_file), allow_pickle=True)
        text = str(data.get("text", "walk"))
        scene_name = str(data.get("scene_name", ""))

        # ---- Detect output format ----
        if "guided_root_5d_norm" in data:
            # Classifier-guided format: already normalized 5D root
            guided_root_5d_norm = data["guided_root_5d_norm"]  # (T, 5)
            T = guided_root_5d_norm.shape[0]
            external_root = torch.from_numpy(guided_root_5d_norm).float().unsqueeze(0).to(device)
        elif "gen_root" in data:
            # Energy-guided format: 3D meter-space root
            gen_root_3d = data["gen_root"]  # (T, 3)
            T = gen_root_3d.shape[0]
            gen_root_xz = gen_root_3d[:, [0, 2]]
            heading = heading_from_path(gen_root_xz)
            external_root = root_5d_from_meter(gen_root_3d, heading, model.motion_rep)
            external_root = external_root.to(device)
        else:
            log.warning(f"Skipping {npz_file.name}: no recognized root key found")
            continue

        # Text encoding
        text_clean = sanitize_texts([text])[0]
        text_feat, text_lengths = model.text_encoder([text_clean])
        text_feat = text_feat.to(device)
        B, maxlen = text_feat.shape[:2]
        text_pad_mask = (
            torch.arange(maxlen, device=device).expand(B, maxlen)
            < torch.tensor(text_lengths, device=device)[:, None]
        )

        # Initialize
        lengths = torch.tensor([T], device=device)
        motion_pad_mask = length_to_mask(lengths)
        first_heading_angle = torch.tensor([0.0], device=device)
        motion_mask = torch.zeros(1, T, model.motion_rep.motion_rep_dim, device=device)
        observed_motion = torch.zeros(
            1, T, model.motion_rep.motion_rep_dim, device=device
        )
        traj_feats, traj_mask = None, None
        if getattr(model, "traj_encoder", None) is not None:
            traj_feats, traj_mask = model.encode_traj(external_root, motion_pad_mask)

        cur_mot = torch.randn(1, T, model.motion_rep.motion_rep_dim, device=device)

        # --- Core loop: fix root + denoising with external_root ---
        for i in range(num_denoising_steps - 1, -1, -1):
            t = torch.tensor([i], device=device)

            # Pre-step root fix
            if args.fix_root_each_step:
                cur_mot[..., root_slice] = external_root

            with torch.inference_mode():
                cur_mot = model.denoising_step(
                    motion=cur_mot,
                    pad_mask=motion_pad_mask,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    t=t,
                    first_heading_angle=first_heading_angle,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    num_denoising_steps=num_denoising_steps_t,
                    cfg_weight=cfg_weight,
                    external_root=external_root,
                    use_external_root=True,
                    traj_feats=traj_feats,
                    traj_mask=traj_mask,
                )
            cur_mot = cur_mot.clone()  # exit inference_mode → regular tensor

            # Post-step root fix
            if args.fix_root_each_step:
                cur_mot[..., root_slice] = external_root

        # Decode
        output = model.motion_rep.inverse(cur_mot, is_normalized=True, return_numpy=True)
        gen_joints = output["posed_joints"][0]
        gen_root_out = output["smooth_root_pos"][0]

        # Verify root fix
        root_error = float(
            torch.abs(cur_mot[0, :, 0:5] - external_root[0]).max().cpu().item()
        )
        root_fix_errors.append(root_error)

        if root_error >= 1e-5:
            log.error(
                f"Root fix VIOLATED for {npz_file.name}: "
                f"max_abs(final_root - external_root) = {root_error:.2e} >= 1e-5"
            )

        # Save
        gt_root_xz_out = data.get("gt_root_xz", None)
        if gt_root_xz_out is None:
            gt_root_xz_out = data.get("target_path_xz", None)
        np.savez(
            str(out_file),
            gen_root=gen_root_out,
            gen_joints=gen_joints,
            gt_joints=data.get("gt_joints", None),
            gt_root_xz=gt_root_xz_out,
            text=text,
            scene_name=str(data.get("scene_name", "")),
        )

    if root_fix_errors:
        log.info(
            f"Root fix max_error: {max(root_fix_errors):.2e} | "
            f"mean: {np.mean(root_fix_errors):.2e} | "
            f"all_passed: {all(e < 1e-5 for e in root_fix_errors)}"
        )
    else:
        log.info("No new body files generated.")
    log.info(f"Done! Body results saved to {output_dir}")


if __name__ == "__main__":
    main()
