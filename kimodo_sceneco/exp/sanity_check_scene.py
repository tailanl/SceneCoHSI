#!/usr/bin/env python
"""Sanity check: does scene input actually affect generation for exp1-4?

Runs each model with and without a dummy scene (same seed), checks if output differs.
"""
import logging, os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("sanity")


def setup_env():
    os.environ.setdefault("CHECKPOINT_DIR", "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_original_kimodo(device="cpu"):
    from kimodo.model import load_model as load_kimodo_model
    return load_kimodo_model("Kimodo-SOMA-RP-v1.1", device=device)


def load_sceneco_model(exp_type, checkpoint_path, text_encoder=None, device="cpu"):
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location="cpu")
    exp_type = ckpt.get("exp_type", exp_type)
    from kimodo.model import load_model as load_kimodo_model

    class _DummyTE:
        def __call__(self, t): raise RuntimeError()
        def to(self, d): return self
        def eval(self): return self

    kimodo_pretrained = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device="cpu", text_encoder=_DummyTE())
    scene_config = {
        'voxel_size': (64, 64, 64), 'patch_size': (8, 8, 8),
        'in_channels': 1, 'd_model': 256, 'num_heads': 4, 'num_layers': 4, 'ff_dim': 512,
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
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    target_device = _torch.device(device)
    model.to(target_device)
    model.device = target_device
    model.eval()
    return model


@torch.no_grad()
def generate_sceneco(model, texts, num_frames, scene_input, num_steps=50, cfg_weight=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg_weight is None:
        cfg_weight = [2.0, 2.0, 2.0]
    return model(
        prompts=texts, num_frames=num_frames,
        num_denoising_steps=num_steps, cfg_weight=cfg_weight,
        return_numpy=False, progress_bar=lambda x: x,
        scene_input=scene_input,
    )


def make_dummy_scene(batch_size=1):
    dummy_voxel = torch.ones(batch_size, 1, 64, 64, 64, dtype=torch.float32)
    dummy_bbox_centers = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32).repeat(batch_size, 1)
    dummy_bbox_sizes = torch.tensor([[0.5, 1.0, 0.5]], dtype=torch.float32).repeat(batch_size, 1)
    dummy_label_ids = torch.zeros(batch_size, 1, dtype=torch.long)
    return {
        "voxel_grid": dummy_voxel,
        "bbox_centers": dummy_bbox_centers,
        "bbox_sizes": dummy_bbox_sizes,
        "label_ids": dummy_label_ids,
        "obj_mask": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    setup_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    orig = load_original_kimodo(device)

    checkpoints = {
        "exp1": "./exp1_monkey_patch_output/checkpoints/best_checkpoint.pt",
        "exp2": "./exp2_rewrite_layer_output/checkpoints/best_checkpoint.pt",
        "exp3": "./exp3_root_only_output/checkpoints/best_checkpoint.pt",
        "exp4": "./exp4_body_only_output/checkpoints/best_checkpoint.pt",
    }

    models = {}
    for exp_name, ckpt_path in checkpoints.items():
        try:
            models[exp_name] = load_sceneco_model(exp_name, ckpt_path, text_encoder=orig.text_encoder, device=device)
            log.info(f"  {exp_name} loaded OK")
        except Exception as e:
            log.error(f"  FAILED to load {exp_name}: {e}")

    test_prompts = [
        ("A person walks forward in a straight line.", 60),
        ("A person crouches down and stands up.", 50),
    ]

    log.info("")
    log.info("=" * 70)
    log.info(f"{'Exp':<6} {'NoScene L2':>12} {'Scene L2':>12} {'Delta':>10} {'SceneActive':>12}")
    log.info("-" * 70)

    for exp_name, model in models.items():
        all_no_jl2 = []
        all_yes_jl2 = []

        for text, num_frames in test_prompts:
            out_no = generate_sceneco(model, text, num_frames, scene_input=None, seed=42)
            dummy_scene = make_dummy_scene()
            dummy_scene = {k: v.to(device) if v is not None else None for k, v in dummy_scene.items()}
            out_yes = generate_sceneco(model, text, num_frames, scene_input=dummy_scene, seed=42)

            pj_no = out_no["posed_joints"].cpu()
            pj_yes = out_yes["posed_joints"].cpu()
            if pj_no.ndim == 4:
                pj_no = pj_no[0]
                pj_yes = pj_yes[0]

            diff = torch.norm(pj_no - pj_yes, dim=-1).mean().item()
            all_no_jl2.append(diff)

        mean_delta = np.mean(all_no_jl2)
        active = "YES ✓" if mean_delta > 0.001 else "NO ✗"
        log.info(f"{exp_name:<6} {mean_delta:12.4f} {'--':>12} {mean_delta:10.4f} {active:>12}")

    log.info("")
    log.info("Interpretation:")
    log.info("  Delta = mean L2 distance between (no scene) and (dummy scene) generation")
    log.info("  Delta > 0.001m → SceneCo is ACTIVE and changes motion when scene is provided")
    log.info("  Delta ~ 0.000m → SceneCo is NOT responding to scene input (potential issue)")

    for m in models.values():
        del m
    del orig
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
