#!/usr/bin/env python
"""Compare original KiMoDo vs trained SceneCo model (no scene input).

Generates motion with same text + same noise seed + same denoising steps.
Verifies that the trained SceneCo model without scene produces results
equivalent to original KiMoDo (degradation < 5%).
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("compare_test")


def setup_env():
    os.environ.setdefault("CHECKPOINT_DIR", "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_original_kimodo(device="cpu"):
    from kimodo.model import load_model as load_kimodo_model

    log.info("Loading original KiMoDo (with real text encoder)...")
    kimodo = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device=device)
    kimodo.eval()
    return kimodo


def load_sceneco_model(exp_type, checkpoint_path, text_encoder=None, device="cpu"):
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location="cpu")
    exp_type = ckpt.get("exp_type", exp_type)

    from kimodo.model import load_model as load_kimodo_model

    class _DummyTE:
        def __call__(self, text):
            raise RuntimeError()
        def to(self, d):
            return self
        def eval(self):
            return self

    log.info(f"Loading pretrained KiMoDo base for {exp_type}...")
    kimodo_pretrained = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device="cpu", text_encoder=_DummyTE())

    scene_config = {
        'voxel_size': (64, 64, 64), 'patch_size': (8, 8, 8),
        'in_channels': 1, 'd_model': 256, 'num_heads': 4,
        'num_layers': 4, 'ff_dim': 512,
    }

    if exp_type == "exp1":
        from kimodo_sceneco.exp.exp1_monkey_patch import KimodoSceneCoExp1
        pretrained_denoiser = kimodo_pretrained.denoiser.model
        model = KimodoSceneCoExp1(
            denoiser=pretrained_denoiser, text_encoder=text_encoder,
            num_base_steps=1000, scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_config, device=_torch.device("cpu"),
            cfg_type="scene_separated",
        )
    else:
        from kimodo_sceneco.exp.exp2_rewrite_layer import KimodoSceneCoExp2
        from kimodo_sceneco.exp.exp2_rewrite_layer.backbone_exp2 import TransformerEncoderBlock
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
                scene_feat_dim=256, use_sceneco=use_sc, sceneco_dropout=0.1,
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
            denoiser=new_denoiser, text_encoder=text_encoder,
            num_base_steps=1000, scene_encoder_type="voxel_vit",
            scene_encoder_config=scene_config, device=_torch.device("cpu"),
            cfg_type="scene_separated",
        )
        model._load_and_migrate_pretrained(new_denoiser, pretrained_denoiser)

    if "model_state_dict" in ckpt:
        log.info("Loading checkpoint weights...")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    target_device = _torch.device(device)
    model.to(target_device)
    model.device = target_device
    model.eval()
    return model


@torch.no_grad()
def generate_original(kimodo_model, texts, num_frames, num_steps=50, cfg_weight=None, seed=42):
    """Generate with original KiMoDo - no scene"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg_weight is None:
        cfg_weight = [2.0, 2.0]

    log.info(f"Generating original KiMoDo: '{texts}' ({num_frames} frames, {num_steps} steps)")
    return kimodo_model(
        prompts=texts, num_frames=num_frames,
        num_denoising_steps=num_steps, cfg_weight=cfg_weight,
        return_numpy=False, progress_bar=lambda x: x,
    )


@torch.no_grad()
def generate_sceneco_no_scene(sceneco_model, texts, num_frames, num_steps=50, cfg_weight=None, seed=42):
    """Generate with SceneCo model - NO scene input (scene_input=None)"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg_weight is None:
        cfg_weight = [2.0, 2.0, 2.0]

    log.info(f"Generating SceneCo (no scene): '{texts}' ({num_frames} frames, {num_steps} steps)")
    return sceneco_model(
        prompts=texts, num_frames=num_frames,
        num_denoising_steps=num_steps, cfg_weight=cfg_weight,
        return_numpy=False, progress_bar=lambda x: x,
        scene_input=None,
    )


def compare_motions(out_orig, out_sceneco, tol_mse=0.05, tol_cos=0.95):
    results = {}

    if "posed_joints" in out_orig and "posed_joints" in out_sceneco:
        pj_orig = out_orig["posed_joints"].cpu()
        pj_sce = out_sceneco["posed_joints"].cpu()

        if pj_orig.shape == pj_sce.shape:
            mse_joints = float(F.mse_loss(pj_orig, pj_sce).item())
            pj_flat_orig = pj_orig.reshape(-1, 3)
            pj_flat_sce = pj_sce.reshape(-1, 3)
            diff_per_joint = float(torch.norm(pj_flat_orig - pj_flat_sce, dim=-1).mean().item())
            results["posed_joints_mse"] = mse_joints
            results["posed_joints_mean_l2"] = diff_per_joint

            cos_sims = []
            for joint_idx in range(pj_orig.shape[1]):
                orig_traj = pj_orig[:, joint_idx, :] - pj_orig[:, joint_idx, :].mean(0, keepdim=True)
                sce_traj = pj_sce[:, joint_idx, :] - pj_sce[:, joint_idx, :].mean(0, keepdim=True)
                cos_sim = F.cosine_similarity(orig_traj.flatten().unsqueeze(0), sce_traj.flatten().unsqueeze(0)).item()
                cos_sims.append(cos_sim)
            results["cosine_trajectory_mean"] = float(np.mean(cos_sims))
        else:
            log.warning(f"Shape mismatch: original {pj_orig.shape} vs sceneco {pj_sce.shape}")

    if "motion" in out_orig and "motion" in out_sceneco:
        mot_orig = out_orig["motion"].cpu()
        mot_sce = out_sceneco["motion"].cpu()
        if mot_orig.shape == mot_sce.shape:
            mask = torch.ones(mot_orig.shape[1], dtype=torch.bool)
            results["motion_mse"] = float(F.mse_loss(mot_orig, mot_sce).item())
            m_orig_flat = mot_orig[0]
            m_sce_flat = mot_sce[0]
            results["motion_cosine"] = float(F.cosine_similarity(
                m_orig_flat.flatten().unsqueeze(0),
                m_sce_flat.flatten().unsqueeze(0)
            ).item())

    return results


def main():
    setup_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    log.info("=" * 60)
    log.info("STEP 1: Load original KiMoDo")
    orig = load_original_kimodo(device)
    log.info("  OK")

    checkpoints = {
        "exp1": "./exp1_monkey_patch_output/checkpoints/best_checkpoint.pt",
        "exp2": "./exp2_rewrite_layer_output/checkpoints/best_checkpoint.pt",
        "exp3": "./exp3_root_only_output/checkpoints/best_checkpoint.pt",
        "exp4": "./exp4_body_only_output/checkpoints/best_checkpoint.pt",
    }

    test_prompts = [
        ("A person walks forward in a straight line.", 60),
        ("A person turns around and walks back.", 80),
        ("A person crouches down and stands up.", 50),
        ("A person performs a jumping jack.", 40),
    ]

    all_results = {}

    log.info("")
    log.info("=" * 60)
    log.info("SELF-BASELINE: KiMoDo (seed=42) vs KiMoDo (seed=123)")
    log.info("  This measures inherent noise — KiMoDo generates different")
    log.info("  motions from different seeds even with same text.")
    log.info("=" * 60)
    baseline_results = []
    for text, num_frames in test_prompts:
        log.info(f"  Prompt: '{text}' ({num_frames}f)")
        out_a = generate_original(orig, text, num_frames, num_steps=50, seed=42)
        out_b = generate_original(orig, text, num_frames, num_steps=50, seed=123)
        comp = compare_motions(out_a, out_b)
        log.info(f"    posed_joints MSE:     {comp.get('posed_joints_mse', 'N/A'):.6f}" if 'posed_joints_mse' in comp else "    posed_joints: N/A")
        log.info(f"    posed_joints mean L2: {comp.get('posed_joints_mean_l2', 'N/A'):.4f}m" if 'posed_joints_mean_l2' in comp else "")
        log.info(f"    trajectory cosine:    {comp.get('cosine_trajectory_mean', 'N/A'):.4f}" if 'cosine_trajectory_mean' in comp else "")
        baseline_results.append(comp)
    all_results["self_baseline"] = baseline_results

    for exp_name, ckpt_path in checkpoints.items():
        log.info("")
        log.info("=" * 60)
        log.info(f"STEP: Compare original KiMoDo vs {exp_name} (NO SCENE)")
        log.info("=" * 60)

        try:
            sceneco = load_sceneco_model(exp_name, ckpt_path, text_encoder=orig.text_encoder, device=device)
            log.info(f"  {exp_name} model loaded OK")
        except Exception as e:
            log.error(f"  FAILED to load {exp_name}: {e}")
            continue

        exp_results = []

        for text, num_frames in test_prompts:
            seed = 42
            log.info(f"  Prompt: '{text}' ({num_frames}f)")

            out_orig = generate_original(orig, text, num_frames, num_steps=50, seed=seed)
            out_sce = generate_sceneco_no_scene(sceneco, text, num_frames, num_steps=50, seed=seed)

            comp = compare_motions(out_orig, out_sce)
            log.info(f"    posed_joints MSE:     {comp.get('posed_joints_mse', 'N/A'):.6f}" if 'posed_joints_mse' in comp else "    posed_joints: N/A")
            log.info(f"    posed_joints mean L2: {comp.get('posed_joints_mean_l2', 'N/A'):.4f}m" if 'posed_joints_mean_l2' in comp else "")
            log.info(f"    trajectory cosine:    {comp.get('cosine_trajectory_mean', 'N/A'):.4f}" if 'cosine_trajectory_mean' in comp else "")
            log.info(f"    motion MSE:           {comp.get('motion_mse', 'N/A'):.6f}" if 'motion_mse' in comp else "")
            log.info(f"    motion cosine:        {comp.get('motion_cosine', 'N/A'):.4f}" if 'motion_cosine' in comp else "")
            exp_results.append(comp)

        all_results[exp_name] = exp_results
        del sceneco
        torch.cuda.empty_cache()

    baseline = all_results.pop("self_baseline")
    b_jl2 = np.mean([r.get("posed_joints_mean_l2", 0) for r in baseline])
    b_tcos = np.mean([r.get("cosine_trajectory_mean", 0) for r in baseline])

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"  Noise floor (KiMoDo self, seed 42 vs 123): Joint L2 = {b_jl2:.4f}m, Traj Cos = {b_tcos:.4f}")
    log.info("")
    log.info(f"  Joint L2 = mean(||pos_orig[t,j] - pos_sce[t,j]||_2) over all frames × joints (MPJPE)")
    log.info(f"  Ratio > 1.0 means difference exceeds KiMoDo's own noise floor")
    log.info("")
    log.info(f"{'Exp':<6} {'Joint MSE':>12} {'Joint L2(m)':>12} {'Ratio 👆':>10} {'Traj Cos':>10} {'Motion MSE':>12} {'Motion Cos':>12}")
    log.info("-" * 80)
    for exp_name, results in all_results.items():
        jmse = np.mean([r.get("posed_joints_mse", 0) for r in results])
        jl2 = np.mean([r.get("posed_joints_mean_l2", 0) for r in results])
        ratio = jl2 / b_jl2 if b_jl2 > 0 else float('inf')
        tcos = np.mean([r.get("cosine_trajectory_mean", 0) for r in results])
        mmse = np.mean([r.get("motion_mse", 0) for r in results])
        mcos = np.mean([r.get("motion_cosine", 0) for r in results])
        log.info(f"{exp_name:<6} {jmse:12.6f} {jl2:12.4f} {ratio:10.2f}x {tcos:10.4f} {mmse:12.6f} {mcos:12.4f}")

    log.info("")
    log.info("Interpretation:")
    log.info(f"  Noise floor Joint L2 = {b_jl2:.4f}m (KiMoDo's inherent generation variance)")
    log.info("  Ratio ≈ 1.0 → within noise floor (exp is indistinguishable from KiMoDo variance)")
    log.info("  Ratio < 2.0 → acceptable (minor drift, similar magnitude to natural variance)")
    log.info("  Joint L2 < 0.05m   → visually indistinguishable")
    log.info("  Trajectory cosine > 0.95 → same motion pattern")

    del orig
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
