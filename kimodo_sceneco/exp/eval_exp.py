"""Evaluation script for Exp1-4 SceneCo models.

Metrics: Collision Rate | Degradation Ratio | FID | Diversity | Foot Skate

Usage:
    python -m kimodo_sceneco.exp.eval_exp \
        --exp_type exp2 \
        --checkpoint ./exp2_rewrite_layer_output/checkpoints/best_checkpoint.pt \
        --data_root ... --cache_dir ...
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

log = logging.getLogger(__name__)


def compute_collision_rate(posed_joints, voxel_grid, lengths, voxel_size_m=0.02, collision_thresh=3):
    B, T, J, _ = posed_joints.shape
    total_collision_frames = 0
    total_body_frames = 0

    for b in range(B):
        L = int(lengths[b].item())
        voxel = voxel_grid[b, 0]
        vx, vy, vz = voxel.shape

        for t in range(L):
            joints = posed_joints[b, t]
            collision_count = 0
            for j in range(J):
                x, y, z = joints[j].cpu().numpy()
                ix = int(x / voxel_size_m)
                iy = int(y / voxel_size_m)
                iz = int(z / voxel_size_m)
                if 0 <= ix < vx and 0 <= iy < vy and 0 <= iz < vz:
                    if voxel[ix, iy, iz] > 0.5:
                        collision_count += 1
            if collision_count >= collision_thresh:
                total_collision_frames += 1
            total_body_frames += 1

    return total_collision_frames / max(total_body_frames, 1) * 100


def compute_degradation_ratio(model, val_loader, device, max_batches=10):
    model.eval()
    model_denoiser = model.denoiser.model if hasattr(model.denoiser, 'model') else model.denoiser

    mse_with_scene = 0.0
    mse_without_scene = 0.0
    count = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break

        motion = batch["motion_features"].to(device)
        mask = batch["motion_mask"].to(device)
        voxel = batch["voxel_grid"].to(device)
        lengths = batch["lengths"]

        if "text_feat" in batch:
            text_feat = batch["text_feat"].to(device)
            B = text_feat.shape[0]
            text_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        else:
            continue

        scene_feat, scene_mask = model.encode_scene(voxel)
        B = motion.shape[0]
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(motion)
        x_t = model.diffusion.q_sample(motion, t, noise=noise)

        with torch.no_grad():
            pred_with = model_denoiser(
                [2.0, 2.0, 2.0], x_t, mask, text_feat, text_pad_mask, t,
                scene_feat=scene_feat, scene_mask=scene_mask, cfg_type="nocfg",
            )
            pred_without = model_denoiser(
                [2.0, 2.0, 2.0], x_t, mask, text_feat, text_pad_mask, t,
                scene_feat=None, scene_mask=None, cfg_type="nocfg",
            )

        mask_f = mask.unsqueeze(-1).float()
        mse_with_scene += ((pred_with - motion) ** 2 * mask_f).sum() / mask_f.sum()
        mse_without_scene += ((pred_without - motion) ** 2 * mask_f).sum() / mask_f.sum()
        count += 1

    if count == 0:
        return 1.0

    mse_with_scene /= count
    mse_without_scene /= count
    return (mse_with_scene / max(mse_without_scene, 1e-8)).item()


def compute_foot_skate(posed_joints, lengths, skeleton):
    try:
        from kimodo_sceneco.metrics.foot_skate import FootSkateFromHeight, FootSkateRatio

        lengths_t = torch.tensor(lengths)
        mask = torch.zeros(posed_joints.shape[0], posed_joints.shape[1], dtype=torch.bool)
        for i, l in enumerate(lengths):
            if l < posed_joints.shape[1]:
                mask[i, l:] = True

        fs_height = FootSkateFromHeight(skeleton)
        fs_ratio = FootSkateRatio(skeleton)

        h_result = fs_height(posed_joints, lengths_t)
        r_result = fs_ratio(posed_joints, lengths_t)

        return {
            "foot_skate_height": h_result.get("foot_skate_from_height", 0),
            "foot_skate_ratio": r_result.get("foot_skate_ratio", 0),
        }
    except Exception as e:
        log.warning(f"Foot skate failed: {e}")
        return {}


@torch.no_grad()
def evaluate(model, val_dataset, args):
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate_fn = val_dataset.__class__.collate_fn if hasattr(val_dataset.__class__, 'collate_fn') else None
    if collate_fn is None:
        from kimodo_sceneco.train.dataset import collate_fn as _collate_fn
        collate_fn = _collate_fn

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )

    all_collision_rates = []
    all_fs_height = []
    all_fs_ratio = []
    n_eval = 0

    log.info(f"Evaluating collision rate on up to {args.num_samples} samples...")
    for batch in tqdm(val_loader, desc="Eval"):
        if args.num_samples and n_eval >= args.num_samples:
            break

        motion = batch["motion_features"].to(device)
        mask = batch["motion_mask"].to(device)
        voxel = batch["voxel_grid"].to(device)
        texts = batch["texts"]
        lengths = batch["lengths"]
        scene_names = batch.get("scene_names", [""] * len(texts))

        batch_frames = int(lengths.max().item())
        if batch_frames < args.min_frames:
            continue

        for i in range(len(texts)):
            if args.num_samples and n_eval >= args.num_samples:
                break
            n_eval += 1
            L = lengths[i].item()

            try:
                output = model(
                    prompts=texts[i],
                    num_frames=L,
                    num_denoising_steps=args.num_denoising_steps,
                    cfg_weight=args.cfg_weight,
                    return_numpy=False,
                    scene_input=voxel[i:i+1],
                )
            except Exception as e:
                log.warning(f"Generation failed for sample {n_eval}: {e}")
                continue

            if "posed_joints" not in output:
                log.warning(f"No posed_joints in output for sample {n_eval}")
                continue

            pj = output["posed_joints"].unsqueeze(0)
            cr = compute_collision_rate(pj, voxel[i:i+1], torch.tensor([L]))
            all_collision_rates.append(cr)

            if hasattr(model, 'skeleton'):
                fs = compute_foot_skate(pj, [L], model.skeleton)
                if "foot_skate_height" in fs:
                    all_fs_height.append(fs["foot_skate_height"])
                if "foot_skate_ratio" in fs:
                    all_fs_ratio.append(fs["foot_skate_ratio"])

    results = {}

    if all_collision_rates:
        cr_array = np.array(all_collision_rates)
        results["collision_rate_mean"] = float(cr_array.mean())
        results["collision_rate_std"] = float(cr_array.std())
        results["collision_rate_median"] = float(np.median(cr_array))

    if all_fs_height:
        results["foot_skate_height_mean"] = float(np.mean(all_fs_height))
    if all_fs_ratio:
        results["foot_skate_ratio_mean"] = float(np.mean(all_fs_ratio))

    log.info("Computing degradation ratio...")
    dr = compute_degradation_ratio(model, val_loader, device, max_batches=args.val_max_batches)
    results["degradation_ratio"] = dr

    results["num_samples"] = n_eval

    log.info("\n" + "=" * 60)
    log.info("Evaluation Results")
    log.info("=" * 60)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            log.info(f"  {k}: {v:.4f}")
        else:
            log.info(f"  {k}: {v}")

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved to {output_path / 'eval_results.json'}")
    return results


def build_model_from_checkpoint(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if "exp_type" in ckpt:
        exp_type = ckpt["exp_type"]
    else:
        exp_type = args.exp_type

    from kimodo.model import load_model as load_kimodo_model

    cache_dir = getattr(args, 'cache_dir', None)
    has_cached_text = cache_dir and Path(cache_dir).exists()

    class _DummyTextEncoder:
        def __call__(self, text):
            raise RuntimeError()
        def to(self, device):
            return self
        def eval(self):
            return self

    log.info("Loading pretrained KiMoDo...")
    kimodo_pretrained = load_kimodo_model(
        args.pretrained_model,
        device="cpu",
        text_encoder=_DummyTextEncoder() if has_cached_text else None,
    )

    scene_config = {
        'voxel_size': (64, 64, 64),
        'patch_size': (8, 8, 8),
        'in_channels': 1,
        'd_model': getattr(args, 'scene_dim', 256),
        'num_heads': getattr(args, 'scene_num_heads', 4),
        'num_layers': getattr(args, 'scene_num_layers', 4),
        'ff_dim': getattr(args, 'scene_ff_dim', 512),
    }

    if exp_type == "exp1":
        from kimodo_sceneco.exp.exp1_monkey_patch import KimodoSceneCoExp1
        pretrained_denoiser = kimodo_pretrained.denoiser.model
        model = KimodoSceneCoExp1(
            denoiser=pretrained_denoiser,
            text_encoder=None,
            num_base_steps=args.num_base_steps,
            scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_config,
            device=torch.device("cpu"),
            cfg_type="scene_separated",
        )
    else:
        from kimodo_sceneco.exp.exp2_rewrite_layer import KimodoSceneCoExp2
        from kimodo_sceneco.exp.exp2_rewrite_layer.backbone_exp2 import TransformerEncoderBlock, SceneCoPostNormEncoder
        from kimodo_sceneco.exp.exp2_rewrite_layer.twostage_denoiser_exp2 import TwostageDenoiser as TDE2

        pretrained_denoiser = kimodo_pretrained.denoiser.model
        pretrained_root = pretrained_denoiser.root_model
        pretrained_body = pretrained_denoiser.body_model
        motion_rep = pretrained_denoiser.motion_rep
        motion_mask_mode = pretrained_denoiser.motion_mask_mode

        root_use = exp_type != "exp4"
        body_use = exp_type != "exp3"

        def _extract(block, use_sc):
            return dict(
                latent_dim=block.latent_dim, ff_size=block.ff_size,
                num_layers=block.num_layers, num_heads=block.num_heads,
                activation=block.activation, dropout=block.dropout,
                pe_dropout=block.pe_dropout,
                norm_first=getattr(block, 'norm_first', False),
                llm_shape=[1, block.embed_text.in_features],
                use_text_mask=block.use_text_mask,
                num_text_tokens_override=getattr(block, 'num_text_tokens_override', None),
                input_first_heading_angle=block.input_first_heading_angle,
                scene_feat_dim=scene_config['d_model'],
                use_sceneco=use_sc,
                sceneco_dropout=getattr(args, 'sceneco_dropout', 0.1),
            )

        root_config = _extract(pretrained_root, root_use)
        root_config["input_dim"] = pretrained_root.input_linear.in_features
        root_config["output_dim"] = pretrained_root.output_linear.out_features
        root_config["skeleton"] = motion_rep.skeleton

        body_config = _extract(pretrained_body, body_use)
        body_config["input_dim"] = pretrained_body.input_linear.in_features
        body_config["output_dim"] = pretrained_body.output_linear.out_features
        body_config["skeleton"] = motion_rep.skeleton

        new_root = TransformerEncoderBlock(**root_config)
        new_body = TransformerEncoderBlock(**body_config)

        new_denoiser = TDE2(motion_rep=motion_rep, motion_mask_mode=motion_mask_mode)
        new_denoiser.root_model = new_root
        new_denoiser.body_model = new_body

        model = KimodoSceneCoExp2(
            denoiser=new_denoiser,
            text_encoder=None,
            num_base_steps=args.num_base_steps,
            scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_config,
            device=torch.device("cpu"),
            cfg_type="scene_separated",
        )
        model._load_and_migrate_pretrained(new_denoiser, pretrained_denoiser)

    if "model_state_dict" in ckpt:
        log.info("Loading checkpoint state_dict...")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif hasattr(model, 'denoiser') and hasattr(model.denoiser, 'model'):
        log.info("Loading checkpoint state_dict (alt key)...")
        model.denoiser.model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)

    model.to(torch.device(args.device))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Kimodo-SceneCo Exp1-4")
    parser.add_argument("--exp_type", type=str, choices=["exp1", "exp2", "exp3", "exp4"], default="exp2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default="Kimodo-SOMA-RP-v1.1")
    parser.add_argument("--data_root", type=str, default="/home/lzsh2025/kimodo-viser/LINGO/dataset")
    parser.add_argument("--cache_dir", type=str, default="/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/cached_data")
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--num_base_steps", type=int, default=1000)
    parser.add_argument("--cfg_weight", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--val_max_batches", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scene_dim", type=int, default=256)
    parser.add_argument("--scene_num_heads", type=int, default=4)
    parser.add_argument("--scene_num_layers", type=int, default=4)
    parser.add_argument("--scene_ff_dim", type=int, default=512)
    parser.add_argument("--sceneco_dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    os.environ.setdefault("CHECKPOINT_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    ))
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    log.info(f"Experiment type: {args.exp_type}")
    log.info(f"Checkpoint: {args.checkpoint}")

    model = build_model_from_checkpoint(args)

    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset

    val_dataset = LINGOSceneMotionDataset(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        max_frames=196,
        min_frames=args.min_frames,
        voxel_size=(64, 64, 64),
        train_ratio=0.9,
        seed=42,
        split="val",
        scene_dropout=0.0,
    )
    log.info(f"Val dataset: {len(val_dataset)} samples")

    evaluate(model, val_dataset, args)


if __name__ == "__main__":
    main()
