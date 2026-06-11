"""Quick fill missing E4 energy-guided root files (cache-based index).

Reads stem list from files, generates only missing roots.
Outputs to same directory as existing roots.
"""

import argparse, logging, sys, os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(1, str(PROJECT_DIR.parent / "kimodo"))
os.environ["CHECKPOINT_DIR"] = str(PROJECT_DIR / "models")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np, torch, yaml
from tqdm.auto import tqdm
from kimodo.model.load_model import load_model
from kimodo.sanitize import sanitize_texts
from kimodo.motion_rep.feature_utils import length_to_mask
from kimodo_sceneco.guidance.root_guidance import RootGuidanceConfig, compute_root_guidance_loss
from kimodo_sceneco.guidance.path_utils import smooth_path_xz

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/guidance_root_scene.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(f"cuda:{args.gpu}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stems
    stems = [line.strip() for line in open(args.stems_file) if line.strip()]
    log.info(f"Need to generate {len(stems)} roots")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    path_cfg = cfg.get("path_guidance", {})
    gen_cfg = cfg.get("generation", {})
    root_guidance_cfg = RootGuidanceConfig(
        enabled=True,
        w_path=path_cfg.get("w_path", 10.0), w_goal=path_cfg.get("w_goal", 20.0),
        w_speed=path_cfg.get("w_speed", 1.0), w_smooth=path_cfg.get("w_smooth", 2.0),
        w_jerk=path_cfg.get("w_jerk", 0.5), w_heading=path_cfg.get("w_heading", 2.0),
        w_height=path_cfg.get("w_height", 1.0), w_scene=0.0,
        scale=path_cfg.get("scale", 0.03),
        start_step=path_cfg.get("start_step", 0), end_step=path_cfg.get("end_step", 50),
    )
    num_denoising_steps = gen_cfg.get("num_denoising_steps", 50)
    cfg_weight = gen_cfg.get("cfg_weight", [2.0, 2.0])

    # Load model
    log.info("Loading model...")
    model_ckpt = cfg.get("model", {}).get("checkpoint", "Kimodo-SMPLX-RP-v1")
    model = load_model(model_ckpt, device=device)
    model.eval()

    cache_dir = PROJECT_DIR / "lingo_smplx_cache"
    joints_file = PROJECT_DIR / "LINGO/dataset/dataset/human_joints_aligned.npy"
    joints_all = np.load(str(joints_file), mmap_mode="r")
    start_idx = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJECT_DIR / "LINGO/dataset/dataset/end_idx.npy")).flatten()

    count = 0
    for stem in tqdm(stems, desc="Generating"):
        cache_file = cache_dir / f"{stem}.npz"
        if not cache_file.exists():
            log.warning(f"SKIP {stem}: not in cache")
            continue
        data = np.load(str(cache_file), allow_pickle=True)
        T = int(data["length"])
        text = str(data.get("text", "walk"))
        scene_name = str(data.get("scene_name", ""))

        # GT root path
        feat = data["motion_features"][:T]
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        unnorm = model.motion_rep.unnormalize(feat_t)
        output = model.motion_rep.inverse(unnorm, is_normalized=False, return_numpy=True)
        gt_root_xz = output["smooth_root_pos"][0][:, [0, 2]]
        target_path_xz = torch.from_numpy(gt_root_xz).float().unsqueeze(0).to(device)
        target_path_xz = smooth_path_xz(target_path_xz, kernel_size=5)

        # Text encoding
        text_clean = sanitize_texts([text])[0]
        text_feat, text_lengths = model.text_encoder([text_clean])
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

        # Adapted denoising loop (inline to avoid importing the full function)
        from scripts.generate_root_guidance import denoising_step_with_guidance
        indices = list(range(num_denoising_steps))[::-1]
        for i in indices:
            t = torch.tensor([i], device=device)
            if root_guidance_cfg.start_step <= i < root_guidance_cfg.end_step:
                cur_mot, _ = denoising_step_with_guidance(
                    model, cur_mot, motion_pad_mask, text_feat, text_pad_mask,
                    t, first_heading_angle, motion_mask, observed_motion,
                    num_denoising_steps, use_timesteps, map_tensor, cfg_weight,
                    root_guidance_cfg, target_path_xz,
                )
            else:
                with torch.no_grad():
                    cur_mot = model.denoising_step(
                        cur_mot, motion_pad_mask, text_feat, text_pad_mask,
                        t, first_heading_angle, motion_mask, observed_motion,
                        torch.tensor([num_denoising_steps], device=device), cfg_weight,
                    )

        # Decode
        output_dec = model.motion_rep.inverse(cur_mot, is_normalized=True, return_numpy=True)
        gen_root = output_dec["smooth_root_pos"][0]
        gen_joints = output_dec["posed_joints"][0]

        ci = int(stem.split("_")[1])
        s, e = int(start_idx[ci]), int(end_idx[ci])
        gt_joints = joints_all[s:s+T, :22, :].copy() if s < e else np.zeros((T, 22, 3))

        np.savez(
            str(output_dir / f"{stem}.npz"),
            gen_root=gen_root,
            gt_root_xz=gt_root_xz,
            gen_joints=gen_joints,
            gt_joints=gt_joints,
            text=np.asarray(text),
            scene_name=np.asarray(scene_name),
            guided_root_5d_norm=cur_mot[0, :, model.motion_rep.root_slice].detach().cpu().numpy().astype(np.float32),
            guided_root_5d_meter=np.concatenate([gen_root[:, :3], np.zeros((gen_root.shape[0], 2))], axis=-1).astype(np.float32),
            target_path_xz=gt_root_xz.astype(np.float32),
            source_file=np.asarray(str(cache_file)),
        )
        count += 1

    log.info(f"Done. Generated {count} missing roots to {output_dir}")


if __name__ == "__main__":
    main()
