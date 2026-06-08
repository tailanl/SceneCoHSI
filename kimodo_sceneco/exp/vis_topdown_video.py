#!/usr/bin/env python
"""2D Top-Down skeleton video in scene for all SceneCo experiments (exp1-4).

Renders skeleton projected to XY plane over scene voxel occupancy,
WITH scene vs WITHOUT scene vs original KiMoDo.
Usage: python -m kimodo_sceneco.exp.vis_topdown_video --gpu 6
"""

import logging
import os
import argparse
import glob

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

log = logging.getLogger("topdown_viz")

CHECKPOINTS = {
    "exp1": "./exp1_monkey_patch_output/checkpoints/best_checkpoint.pt",
    "exp2": "./exp2_rewrite_layer_output/checkpoints/best_checkpoint.pt",
    "exp3": "./exp3_root_only_output/checkpoints/best_checkpoint.pt",
    "exp4": "./exp4_body_only_output/checkpoints/best_checkpoint.pt",
}


def setup_env():
    os.environ.setdefault("CHECKPOINT_DIR",
                          "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/models")
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

    pretrained = load_kimodo_model("Kimodo-SOMA-RP-v1.1", device="cpu",
                                   text_encoder=_DummyTE())
    sc_cfg = {
        'voxel_size': (64, 64, 64), 'patch_size': (8, 8, 8),
        'in_channels': 1, 'd_model': 256, 'num_heads': 4,
        'num_layers': 4, 'ff_dim': 512,
    }

    if exp_type == "exp1":
        from kimodo_sceneco.exp.exp1_monkey_patch import KimodoSceneCoExp1
        model = KimodoSceneCoExp1(
            denoiser=pretrained.denoiser.model, text_encoder=text_encoder,
            num_base_steps=1000, scene_encoder_type="voxel_vit",
            scene_encoder_config=sc_cfg, device=_torch.device("cpu"),
            cfg_type="scene_separated",
        )
    else:
        from kimodo_sceneco.exp.exp2_rewrite_layer import KimodoSceneCoExp2
        from kimodo_sceneco.exp.exp2_rewrite_layer.backbone_exp2 import TransformerEncoderBlock
        from kimodo_sceneco.exp.exp2_rewrite_layer.twostage_denoiser_exp2 import TwostageDenoiser as TDE2

        pd = pretrained.denoiser.model
        mr = pd.motion_rep
        mm = pd.motion_mask_mode

        def _ex(block, use_sc):
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

        rc = _ex(pd.root_model, exp_type != "exp4")
        rc["input_dim"] = pd.root_model.input_linear.in_features
        rc["output_dim"] = pd.root_model.output_linear.out_features
        rc["skeleton"] = mr.skeleton

        bc = _ex(pd.body_model, exp_type != "exp3")
        bc["input_dim"] = pd.body_model.input_linear.in_features
        bc["output_dim"] = pd.body_model.output_linear.out_features
        bc["skeleton"] = mr.skeleton

        new_deno = TDE2(motion_rep=mr, motion_mask_mode=mm)
        new_deno.root_model = TransformerEncoderBlock(**rc)
        new_deno.body_model = TransformerEncoderBlock(**bc)

        model = KimodoSceneCoExp2(
            denoiser=new_deno, text_encoder=text_encoder,
            num_base_steps=1000, scene_encoder_type="voxel_vit",
            scene_encoder_config=sc_cfg, device=_torch.device("cpu"),
            cfg_type="scene_separated",
        )
        model._load_and_migrate_pretrained(new_deno, pd)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    td = _torch.device(device)
    model.to(td)
    model.device = td
    model.eval()
    return model


def get_skeleton_connections(model):
    if hasattr(model.skeleton, 'joint_parents'):
        p = model.skeleton.joint_parents.cpu().numpy()
        return [(int(p[i]), i) for i in range(len(p)) if p[i] >= 0]
    return []


@torch.no_grad()
def gen_orig(model, text, nf, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    return model(prompts=text, num_frames=nf, num_denoising_steps=50,
                 cfg_weight=[2.0, 2.0], return_numpy=False,
                 progress_bar=lambda x: x)


@torch.no_grad()
def gen_sc(model, text, nf, scene_input=None, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    return model(prompts=text, num_frames=nf, num_denoising_steps=50,
                 cfg_weight=[2.0, 2.0, 2.0], return_numpy=False,
                 progress_bar=lambda x: x, scene_input=scene_input)


def load_scene_samples(cache_dir, device, num=6):
    all_files = sorted(glob.glob(os.path.join(cache_dir, "seg_*.npz")))
    import random; random.seed(42)
    selected = sorted(random.sample(all_files, min(num, len(all_files))))
    samples = []
    for fp in selected:
        d = np.load(fp, allow_pickle=True)
        v = torch.from_numpy(d["voxel_grid"]).float()
        if v.ndim == 3: v = v.unsqueeze(0)
        samples.append({
            "voxel_grid": v.to(device), "text": str(d["text"]),
            "length": max(int(d["length"]), 40),
            "scene_name": str(d.get("scene_name", os.path.basename(fp))),
        })
    return samples


def render_2d_video(posed_joints, output_path, title, connections,
                    scene_voxel=None, fps=30):
    T, J, _ = posed_joints.shape
    joints_xy = posed_joints[:, :, :2]

    has_scene = scene_voxel is not None
    if has_scene:
        vox = scene_voxel.squeeze().cpu().numpy()
        occ = np.max(vox, axis=0)
        extent = [0, occ.shape[1] * 0.1, 0, occ.shape[0] * 0.1]
    else:
        occ = None
        extent = None

    all_xy = joints_xy.reshape(-1, 2)
    mx, my = (all_xy[:, 0].mean(), all_xy[:, 1].mean())
    span = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 0.5) * 1.3

    fig, ax = plt.subplots(figsize=(8, 8))
    writer = FFMpegWriter(fps=fps, bitrate=2000)

    with writer.saving(fig, output_path, 100):
        for t in range(T):
            ax.clear()

            if has_scene:
                ax.imshow(occ, origin='lower', extent=extent,
                          cmap='Greys', alpha=0.3, interpolation='nearest')

            xy = joints_xy[t]
            ax.scatter(xy[:, 0], xy[:, 1], c='#1a73e8', s=40, zorder=5,
                       edgecolors='white', linewidth=0.5)

            for pi, ci in connections:
                ax.plot([xy[pi, 0], xy[ci, 0]], [xy[pi, 1], xy[ci, 1]],
                        c='#333333', lw=2.5, alpha=0.8, zorder=4)

            ax.scatter([xy[0, 0]], [xy[0, 1]], c='#d93025', s=80, zorder=6,
                       edgecolors='white', linewidth=1, label='Root')

            ax.set_xlim(mx - span, mx + span)
            ax.set_ylim(my - span, my + span)
            ax.set_aspect('equal')
            ax.set_facecolor('white')
            ax.set_title(f"{title}  (t={t}/{T})", fontsize=10)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.legend(loc='upper right', fontsize=7)

            writer.grab_frame()

    plt.close(fig)
    log.info(f"  Saved → {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="../topdown_videos")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--exp", type=str, default=None,
                        help="exp1/exp2/exp3/exp4/original (default: all)")
    args = parser.parse_args()

    target_exps = [args.exp] if args.exp else list(CHECKPOINTS.keys())
    if args.exp == "original" and not args.exp:
        target_exps = ["original"]

    setup_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    cache_dir = "/home/lzsh2025/kimodo-viser/kimodo/kimodo_sceneco/cached_data"

    log.info("Loading scene samples...")
    samples = load_scene_samples(cache_dir, device, args.num_samples)
    for i, s in enumerate(samples):
        log.info(f"  [{i+1}] {s['scene_name']}: '{s['text']}' len={s['length']}")

    log.info("Loading original KiMoDo...")
    orig = load_original_kimodo(device)
    connections = get_skeleton_connections(orig)
    log.info(f"  {len(connections)} bone connections")

    run_list = [(e, CHECKPOINTS[e]) for e in target_exps if e in CHECKPOINTS]
    if "original" in (target_exps if isinstance(target_exps, list) else [target_exps]) or \
       (args.exp == "original"):
        run_list.append(("original", None))

    for exp_name, ckpt_path in run_list:
        log.info(f"\n{'='*60}")
        log.info(f"Processing {exp_name}...")

        if exp_name == "original":
            model = orig
            log.info(f"  using original KiMoDo")
        else:
            try:
                model = load_sceneco_model(exp_name, ckpt_path,
                                           text_encoder=orig.text_encoder,
                                           device=device)
                log.info(f"  {exp_name} loaded OK")
            except Exception as e:
                log.error(f"  FAILED: {e}")
                continue

        for si, sample in enumerate(samples):
            text = sample["text"]
            nf = sample["length"]
            sn = sample["scene_name"]
            voxel = sample["voxel_grid"].unsqueeze(0)

            tag = f"{exp_name}_{si+1:02d}_{sn}"
            log.info(f"  [{si+1}] {sn}: '{text}'")

            out_orig = gen_orig(orig, text, nf, seed=42)
            pj_orig = out_orig["posed_joints"].cpu().numpy()
            if pj_orig.ndim == 4: pj_orig = pj_orig[0]

            out_yes = gen_sc(model, text, nf, scene_input=voxel, seed=42)
            pj_yes = out_yes["posed_joints"].cpu().numpy()
            if pj_yes.ndim == 4: pj_yes = pj_yes[0]

            out_no = gen_sc(model, text, nf, scene_input=None, seed=42)
            pj_no = out_no["posed_joints"].cpu().numpy()
            if pj_no.ndim == 4: pj_no = pj_no[0]

            disp_yes = np.sqrt(((pj_yes[-1, 0] - pj_yes[0, 0])**2 +
                                (pj_yes[-1, 1] - pj_yes[0, 1])**2))
            disp_no = np.sqrt(((pj_no[-1, 0] - pj_no[0, 0])**2 +
                               (pj_no[-1, 1] - pj_no[0, 1])**2))

            sx = sample["voxel_grid"]
            render_2d_video(pj_orig, f"{args.output_dir}/{tag}_original.mp4",
                            f"[{sn}] original KiMoDo\n{text[:50]}", connections,
                            scene_voxel=sx)

            render_2d_video(pj_yes, f"{args.output_dir}/{tag}_withscene.mp4",
                            f"[{sn}] {exp_name} WITH scene\n{text[:50]}  Δ={disp_yes:.2f}m",
                            connections, scene_voxel=sx)

            render_2d_video(pj_no, f"{args.output_dir}/{tag}_noscene.mp4",
                            f"[{sn}] {exp_name} NO scene\n{text[:50]}  Δ={disp_no:.2f}m",
                            connections, scene_voxel=sx)

        del model; torch.cuda.empty_cache()

    del orig; torch.cuda.empty_cache()
    log.info(f"\nDONE! Videos in {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
