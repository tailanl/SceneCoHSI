"""Generate root trajectory using analytical energy guidance during DDIM sampling.

This script uses the original Kimodo model and applies hand-written energy
guidance on the root 5D features during sampling to follow a target path and
optionally avoid scene obstacles.

The denoising loop is implemented manually here to allow gradient-based
guidance (the original Kimodo model wraps everything in inference_mode).

Usage:
    # Path-only guidance (baseline)
    python scripts/generate_root_guidance.py \
        --output_dir outputs/guidance_path_only \
        --num_samples 30 --gpu 0

    # Path + Scene guidance
    python scripts/generate_root_guidance.py \
        --output_dir outputs/guidance_path_scene \
        --num_samples 30 --scene_guidance --gpu 0
"""

import argparse, json, logging, sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(1, str(REPO_ROOT))
sys.path.insert(2, str(REPO_ROOT / "kimodo"))

import os
os.environ["CHECKPOINT_DIR"] = str(PROJECT_DIR / "models")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

from kimodo.model.load_model import load_model
from kimodo.sanitize import sanitize_texts
from kimodo.motion_rep.feature_utils import length_to_mask

from kimodo_sceneco.guidance.root_guidance import RootGuidanceConfig, compute_root_guidance_loss
from kimodo_sceneco.guidance.scene_guidance import build_2d_sdf, sample_sdf_2d
from kimodo_sceneco.guidance.path_utils import smooth_path_xz

log = logging.getLogger(__name__)


def load_samples(cache_indices, cache_dir=None):
    """Load dataset samples from LINGO cache."""
    if cache_dir is None:
        cache_dir = PROJECT_DIR / "lingo_smplx_cache"
    else:
        cache_dir = Path(cache_dir)

    joints_file = PROJECT_DIR / "LINGO/dataset/dataset/human_joints_aligned.npy"
    start_idx = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/end_idx.npy")).flatten()

    seg_ranges = {}
    count = 0
    for i in range(len(start_idx)):
        si, ei = int(start_idx[i]), int(end_idx[i])
        if 40 <= ei - si <= 196:
            seg_ranges[count] = (si, ei)
            count += 1

    samples = []
    joints_all = np.load(str(joints_file), mmap_mode="r")
    for ci in cache_indices:
        cache_file = cache_dir / f"seg_{ci:05d}.npz"
        if not cache_file.exists():
            log.warning(f"SKIP seg_{ci:05d}: file not found")
            continue
        data = np.load(str(cache_file), allow_pickle=True)
        T = int(data["length"])
        s, e = seg_ranges.get(ci, (0, T))
        samples.append({
            "cache_idx": ci,
            "text": str(data.get("text", "no-text")),
            "num_frames": T,
            "motion_features": data["motion_features"][:T],
            "gt_joints": joints_all[s:e, :22, :].copy(),
            "scene_name": str(data.get("scene_name", "")),
        })
    return samples


def extract_gt_root_path(motion_rep, features, device=None):
    """Extract GT root XZ path from normalized motion features (meter space)."""
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if device is not None:
        features = features.to(device)
    feat_t = features.unsqueeze(0)
    unnorm = motion_rep.unnormalize(feat_t)
    output = motion_rep.inverse(unnorm, is_normalized=False, return_numpy=True)
    root_pos = output["smooth_root_pos"][0]  # (T, 3)
    return root_pos[:, [0, 2]]  # XZ only


def root_5d_meter_from_output(output):
    """Return meter-space root [x, y, z, heading_cos, heading_sin]."""
    root_pos = np.asarray(output["smooth_root_pos"], dtype=np.float32)
    heading = np.asarray(output["global_root_heading"], dtype=np.float32)
    if root_pos.ndim == 3:
        root_pos = root_pos[0]
    if heading.ndim == 3:
        heading = heading[0]
    if root_pos.ndim != 2 or root_pos.shape[1] < 3:
        raise ValueError(f"smooth_root_pos must have shape (T, >=3), got {root_pos.shape}")
    if heading.ndim != 2 or heading.shape[1] != 2:
        raise ValueError(f"global_root_heading must have shape (T, 2), got {heading.shape}")
    if root_pos.shape[0] != heading.shape[0]:
        raise ValueError(
            f"root/heading length mismatch: root={root_pos.shape}, heading={heading.shape}"
        )
    return np.concatenate([root_pos[:, :3], heading], axis=-1).astype(np.float32)


def load_scene_voxel(scene_name, lingo_root=None):
    """Load scene voxel grid from LINGO dataset."""
    if lingo_root is None:
        lingo_root = PROJECT_DIR / "LINGO/dataset/dataset/Scene"
    scene_dir = Path(lingo_root) / scene_name
    if not scene_dir.exists():
        return None

    for fname in ["semantic_voxel_grid.npy", "voxel_grid.npy"]:
        fpath = scene_dir / fname
        if fpath.exists():
            return np.load(str(fpath))
    return None


def denoising_step_with_guidance(
    model,
    motion,
    pad_mask,
    text_feat,
    text_pad_mask,
    t,
    first_heading_angle,
    motion_mask,
    observed_motion,
    num_denoising_steps,
    use_timesteps,
    map_tensor,
    cfg_weight,
    root_guidance_cfg,
    target_path_xz,
    scene_sdf=None,
    sdf_voxel_size=0.1,
    sdf_grid_origin=(0.0, 0.0, 0.0),
):
    """
    One DDIM step with analytical energy guidance on the root.

    This replaces the original model's denoising_step when guidance is active.
    """
    t_map = map_tensor[t]

    # 1. Get pred_x0 with grad tracking
    x = motion.detach().requires_grad_(True)

    pred_x0 = model.denoiser(
        cfg_weight,
        x,
        pad_mask,
        text_feat,
        text_pad_mask,
        t_map,
        first_heading_angle,
        motion_mask,
        observed_motion,
    )

    # 2. Compute guidance loss (with normalization handling)
    losses = compute_root_guidance_loss(
        pred_x0=pred_x0,
        target_path_xz=target_path_xz,
        root_slice=model.motion_rep.root_slice,
        cfg=root_guidance_cfg,
        scene_sdf=scene_sdf,
        sample_sdf_fn=lambda sdf, pos: sample_sdf_2d(
            sdf, pos, voxel_size=sdf_voxel_size, grid_origin=sdf_grid_origin
        ) if sdf is not None else None,
        motion_rep=model.motion_rep,
        root_is_normalized=True,
    )

    # 3. Gradient update (only root part, clipped)
    grad = torch.autograd.grad(losses["total"], x)[0]
    root_grad = torch.zeros_like(grad)
    root_grad[..., model.motion_rep.root_slice] = grad[..., model.motion_rep.root_slice]
    grad = root_grad
    grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    max_norm = getattr(root_guidance_cfg, "max_grad_norm", 1.0)
    grad = grad * (max_norm / grad_norm).clamp(max=1.0)
    x_guided = motion - root_guidance_cfg.scale * grad
    x_guided = x_guided.detach()

    # 4. DDIM step with guided x
    with torch.no_grad():
        pred_clean = model.denoiser(
            cfg_weight,
            x_guided,
            pad_mask,
            text_feat,
            text_pad_mask,
            t_map,
            first_heading_angle,
            motion_mask,
            observed_motion,
        )
        x_tm1 = model.sampler(use_timesteps, x_guided, pred_clean, t)

    return x_tm1, losses


def main():
    parser = argparse.ArgumentParser(description="Root Classifier Guidance Generation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/guidance_path_only")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val"],
                       help="Use train/val split as defined by seed/ratio")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--scene_guidance", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--target_path_file", type=str, default=None,
                       help="External target path .npy/.npz file (T,2) or directory of per-sample files")
    parser.add_argument("--waypoint_file", type=str, default=None,
                       help="Waypoint file .npy/.npz (K,2), will be interpolated to target length")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

    path_cfg = cfg.get("path_guidance", {})
    scene_cfg = cfg.get("scene_guidance", {})
    gen_cfg = cfg.get("generation", {})

    root_guidance_cfg = RootGuidanceConfig(
        enabled=path_cfg.get("enabled", True),
        w_path=path_cfg.get("w_path", 10.0),
        w_goal=path_cfg.get("w_goal", 20.0),
        w_speed=path_cfg.get("w_speed", 1.0),
        w_smooth=path_cfg.get("w_smooth", 2.0),
        w_jerk=path_cfg.get("w_jerk", 0.5),
        w_heading=path_cfg.get("w_heading", 2.0),
        w_height=path_cfg.get("w_height", 1.0),
        w_scene=scene_cfg.get("w_scene", 5.0) if args.scene_guidance else 0.0,
        scene_margin=scene_cfg.get("scene_margin", 0.10),
        scale=args.guidance_scale or path_cfg.get("scale", 0.03),
        start_step=path_cfg.get("start_step", 0),
        end_step=path_cfg.get("end_step", 50),
    )

    num_denoising_steps = gen_cfg.get("num_denoising_steps", 50)
    cfg_weight = gen_cfg.get("cfg_weight", [2.0, 2.0])

    # Load model
    log.info("Loading Kimodo model...")
    model_ckpt = cfg.get("model", {}).get("checkpoint", "Kimodo-SMPLX-RP-v1")
    model = load_model(model_ckpt, device=device)
    model.eval()

    # Select samples using CACHE-BASED index (same as dataset._load_cached_index)
    import random
    cache_files = sorted((PROJECT_DIR / "lingo_smplx_cache").glob("seg_*.npz"))
    valid_cache = []
    for cf in cache_files:
        if ".tmp" in cf.name:
            continue
        data = np.load(str(cf), allow_pickle=True)
        T = int(data["length"])
        if 40 <= T <= 196:
            valid_cache.append({"cache_path": str(cf), "stem": cf.stem, "length": T})
    
    if args.split is not None:
        rng = random.Random(args.split_seed)
        indices = list(range(len(valid_cache)))
        rng.shuffle(indices)
        n_train = int(len(indices) * args.split_ratio)
        if args.split == "train":
            chosen = sorted(indices[:n_train])
        else:
            chosen = sorted(indices[n_train:])
        valid_cache = [valid_cache[i] for i in chosen]
    
    if args.num_samples == -1:
        args.num_samples = len(valid_cache)
    chosen_samples = valid_cache[args.start_idx:args.start_idx + args.num_samples]
    
    # Build samples from cache
    samples = []
    joints_file = PROJECT_DIR / "LINGO/dataset/dataset/human_joints_aligned.npy"
    joints_all = np.load(str(joints_file), mmap_mode="r")
    start_idx_all = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx_all = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/end_idx.npy")).flatten()
    
    for cs in chosen_samples:
        cf = Path(cs["cache_path"])
        data = np.load(str(cf), allow_pickle=True)
        T = cs["length"]
        ci = int(cs["stem"].split("_")[1])
        s, e = int(start_idx_all[ci]), int(end_idx_all[ci])
        if s < e:
            gt_joints = joints_all[s:s+T, :22, :].copy()
        else:
            gt_joints = np.zeros((T, 22, 3))
        samples.append({
            "cache_idx": ci,
            "cache_stem": cs["stem"],
            "cache_path": str(cf),
            "text": str(data.get("text", "no-text")),
            "num_frames": T,
            "motion_features": data["motion_features"][:T],
            "gt_joints": gt_joints,
            "scene_name": str(data.get("scene_name", "")),
        })

    log.info(f"Loaded {len(samples)} samples | guidance_scale={root_guidance_cfg.scale} | steps={num_denoising_steps}")

    sdf_voxel_size = scene_cfg.get("voxel_size", 0.1)
    sdf_grid_origin = tuple(scene_cfg.get("grid_origin", [0.0, 0.0, 0.0]))

    # Load external target paths
    external_paths = {}
    if args.target_path_file:
        tp = Path(args.target_path_file)
        if tp.is_dir():
            for f in sorted(tp.glob("*.np*")):
                external_paths[f.stem] = np.load(str(f))
        else:
            data = np.load(str(tp))
            if isinstance(data, np.ndarray):
                external_paths["_single"] = data
            elif isinstance(data, dict):
                for k, v in data.items():
                    external_paths[k] = v
        log.info(f"Loaded {len(external_paths)} external target paths")

    if args.waypoint_file:
        waypoints = np.load(str(args.waypoint_file))
        if isinstance(waypoints, np.ndarray) and waypoints.ndim == 2:
            external_paths["_waypoint"] = waypoints
        log.info(f"Loaded waypoints: shape={waypoints.shape}")

    results = []
    for si, sample in enumerate(tqdm(samples, desc="Generating")):
        T = sample["num_frames"]
        text = sanitize_texts([sample["text"]])[0]
        gt_root_xz = extract_gt_root_path(model.motion_rep, sample["motion_features"], device=device)
        target_path_xz = None

        # Determine target path: external > GT
        if external_paths:
            sample_name = f"sample_{si:03d}"
            if sample_name in external_paths:
                raw = external_paths[sample_name]
            elif "_single" in external_paths:
                raw = external_paths["_single"]
            elif "_waypoint" in external_paths:
                raw = external_paths["_waypoint"]
            else:
                raw = None

            if raw is not None:
                if raw.ndim == 2 and raw.shape[1] >= 2:
                    raw_xz = raw[:, :2]
                    # Resample to T if needed
                    if raw_xz.shape[0] != T:
                        raw_t = torch.from_numpy(raw_xz).float().unsqueeze(0)  # (1, K, 2)
                        raw_t = F.interpolate(raw_t.transpose(1, 2), size=T, mode="linear", align_corners=True)
                        raw_xz = raw_t.transpose(1, 2).squeeze(0).numpy()
                    target_path_xz = torch.from_numpy(raw_xz).float().unsqueeze(0).to(device)
                else:
                    target_path_xz = None

        if not external_paths or target_path_xz is None:
            # Extract GT root path as target (in meter space)
            target_path_xz = torch.from_numpy(gt_root_xz).float().unsqueeze(0).to(device)

        target_path_xz = smooth_path_xz(target_path_xz, kernel_size=5)

        # Build scene SDF if requested
        scene_sdf = None
        if args.scene_guidance and sample.get("scene_name"):
            voxel_grid = load_scene_voxel(sample["scene_name"])
            if voxel_grid is not None:
                voxel_grid_t = torch.from_numpy(voxel_grid).to(device)
                scene_sdf = build_2d_sdf(
                    voxel_grid_t,
                    voxel_size=sdf_voxel_size,
                    grid_origin=sdf_grid_origin,
                    device=device,
                )

        # Text encoding
        text_feat, text_lengths = model.text_encoder([text])
        text_feat = text_feat.to(device)
        B, maxlen = text_feat.shape[:2]
        text_pad_mask = torch.arange(maxlen, device=device).expand(B, maxlen) < torch.tensor(text_lengths, device=device)[:, None]

        # Initialize
        lengths = torch.tensor([T], device=device)
        motion_pad_mask = length_to_mask(lengths)
        first_heading_angle = torch.tensor([0.0], device=device)
        motion_mask = torch.zeros(1, T, model.motion_rep.motion_rep_dim, device=device)
        observed_motion = torch.zeros(1, T, model.motion_rep.motion_rep_dim, device=device)

        cur_mot = torch.randn(1, T, model.motion_rep.motion_rep_dim, device=device)

        use_timesteps, map_tensor = model.diffusion.space_timesteps(num_denoising_steps)
        model.diffusion.calc_diffusion_vars(use_timesteps)

        indices = list(range(num_denoising_steps))[::-1]
        step_losses = []

        for i in indices:
            t = torch.tensor([i], device=device)

            if root_guidance_cfg.start_step <= i < root_guidance_cfg.end_step:
                cur_mot, losses = denoising_step_with_guidance(
                    model,
                    cur_mot,
                    motion_pad_mask,
                    text_feat,
                    text_pad_mask,
                    t,
                    first_heading_angle,
                    motion_mask,
                    observed_motion,
                    num_denoising_steps,
                    use_timesteps,
                    map_tensor,
                    cfg_weight,
                    root_guidance_cfg,
                    target_path_xz,
                    scene_sdf=scene_sdf,
                    sdf_voxel_size=sdf_voxel_size,
                    sdf_grid_origin=sdf_grid_origin,
                )
                step_losses.append({k: v.item() for k, v in losses.items()})
            else:
                with torch.no_grad():
                    cur_mot = model.denoising_step(
                        cur_mot, motion_pad_mask, text_feat, text_pad_mask,
                        t, first_heading_angle, motion_mask, observed_motion,
                        torch.tensor([num_denoising_steps], device=device),
                        cfg_weight,
                    )

        # Decode
        output = model.motion_rep.inverse(cur_mot, is_normalized=True, return_numpy=True)
        gen_root = output["smooth_root_pos"][0]  # (T, 3)
        gen_joints = output["posed_joints"][0]    # (T, 22, 3)
        guided_root_5d_norm = (
            cur_mot[0, :, model.motion_rep.root_slice]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        guided_root_5d_meter = root_5d_meter_from_output(output)
        target_path_np = target_path_xz[0].detach().cpu().numpy().astype(np.float32)

        result = {
            "cache_idx": sample["cache_idx"],
            "text": sample["text"],
            "num_frames": T,
            "gen_root": gen_root,
            "gt_root_xz": gt_root_xz,
            "gen_joints": gen_joints,
            "gt_joints": sample["gt_joints"][:T],
            "scene_name": sample.get("scene_name", ""),
        }
        results.append(result)

        np.savez(
            str(output_dir / f"{sample['cache_stem']}.npz"),
            gen_root=gen_root,
            gt_root_xz=gt_root_xz,
            gen_joints=gen_joints,
            gt_joints=sample["gt_joints"][:T],
            text=np.asarray(sample["text"]),
            scene_name=np.asarray(sample.get("scene_name", "")),
            guided_root_5d_norm=guided_root_5d_norm,
            guided_root_5d_meter=guided_root_5d_meter,
            target_path_xz=target_path_np,
            source_file=np.asarray(str(sample.get("cache_path", ""))),
        )

    # Summary
    summary = {
        "num_samples": len(results),
        "guidance_scale": root_guidance_cfg.scale,
        "scene_guidance": args.scene_guidance,
        "config": str(args.config),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Done! {len(results)} samples saved to {output_dir}")


if __name__ == "__main__":
    main()
