#!/usr/bin/env python
"""Test SceneCo WITH scene input on the old checkpoint (pre-sigmoid fix, alpha≈-0.011).

Loads the step 125000 checkpoint, replaces sigmoid gate with raw alpha,
then generates motions with and without scene, renders side-by-side MP4.

Usage:
    CUDA_VISIBLE_DEVICES=7 /home/lzsh2025/miniconda3/envs/kimodo/bin/python \
        kimodo_scene_project/eval/test_with_scene.py \
        --ckpt kimodo_scene_project/outputs/root_only_sceneco_gpu3/checkpoints/checkpoint_step125000.pt \
        --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_ROOT / "SOMA"))

from kimodo_sceneco.model.backbone import _stable_multihead_attention

os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
os.environ.setdefault("PYTHONHASHSEED", "0")

METER_TO_UNIT = 100

SKELETON_CONNECTIONS = [
    (0, 2), (0, 3), (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (3, 15), (15, 18), (18, 21),
    (2, 16), (16, 19), (19, 22),
    (16, 17), (19, 20), (22, 23),
    (15, 24), (24, 27), (27, 29),
    (18, 25), (25, 28), (28, 30),
    (24, 26), (27, 31), (25, 21), (28, 22),
]

TEST_PROMPTS = [
    ("walk forward in a straight line", 120),
    ("turn around and look back", 100),
    ("sit down on a chair", 100),
    ("run quickly then stop", 100),
    ("walk in a circle", 120),
]

SCENE_LIST = [
    "004", "005", "006", "009", "010",
]


def make_old_forward(module):

    def old_forward(self, h, scene_feat, scene_mask=None):
        if scene_feat is None:
            return h
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        scene_feat = torch.nan_to_num(scene_feat, nan=0.0, posinf=0.0, neginf=0.0)
        scene_kv = self.scene_proj(scene_feat)
        h_norm = self.norm(h)
        h_norm = torch.nan_to_num(h_norm, nan=0.0, posinf=0.0, neginf=0.0)
        q = self.w_q(h_norm)
        kv = self.w_kv(scene_kv)
        k, v = kv.chunk(2, dim=-1)
        key_padding_mask = None
        if scene_mask is not None:
            key_padding_mask = ~scene_mask
        attn_out = _stable_multihead_attention(q, k, v, self.nhead, key_padding_mask=key_padding_mask)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout_layer(attn_out) * self.alpha
        result = h + attn_out
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return old_forward.__get__(module, type(module))


def load_voxel_scene(scene_name, voxel_size=(64, 64, 64)):
    scene_dir = PROJECT_ROOT / "LINGO" / "dataset" / "dataset" / "Scene"
    npz_path = scene_dir / f"{scene_name}.npy"
    if npz_path.exists():
        grid = np.load(str(npz_path))
        if grid.shape != tuple(voxel_size):
            import scipy.ndimage
            zoom = [vs / gs for vs, gs in zip(voxel_size, grid.shape)]
            grid = scipy.ndimage.zoom(grid.astype(np.float32), zoom, order=1)
            grid = (grid > 0.5).astype(np.float32)
        return torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    print(f"  ⚠️  Scene not found: {scene_name}, using zeros")
    return torch.zeros(1, 1, *voxel_size)


def load_sceneco_old(ckpt_path, device):
    print("Loading pretrained Kimodo...")
    from kimodo.model import load_model
    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=device)
    inner_denoiser = pretrained.denoiser
    if hasattr(inner_denoiser, "model"):
        inner_denoiser = inner_denoiser.model

    print("Building KimodoSceneCo...")
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo
    model = KimodoSceneCo(
        denoiser=inner_denoiser, text_encoder=pretrained.text_encoder,
        num_base_steps=1000, scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64), "patch_size": (8, 8, 8),
            "d_model": 256, "num_layers": 4,
        },
        device=device, cfg_type="scene_separated",
        use_in_root_model=True, use_in_body_model=False,
    )
    model = model.to(device)
    model.eval()

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    renamed = {}
    for k, v in state_dict.items():
        if k.startswith("scene_encoder."):
            new_k_root = k.replace("scene_encoder.", "scene_encoder_root.")
            new_k_body = k.replace("scene_encoder.", "scene_encoder_body.")
            renamed[new_k_root] = v.clone()
            renamed[new_k_body] = v.clone()
        else:
            renamed[k] = v
    model.load_state_dict(renamed)

    patched = 0
    for m in model.modules():
        cls_name = type(m).__name__
        if cls_name == "SceneCoLayer" and hasattr(m, "alpha") and m.alpha.numel() == 1:
            m.forward = make_old_forward(m)
            patched += 1

    print(f"Patched {patched} SceneCoLayer(s) back to old (no-sigmoid) forward")
    print(f"Alpha values (raw, no gate):")
    for name, param in model.named_parameters():
        if "alpha" in name and param.numel() == 1:
            print(f"  {name.rsplit('.', 1)[0].rsplit('.', 1)[-1]:20s}: {param.item():.6f}")

    return model


def generate_motion(model, prompt, num_frames, device, voxel_grid=None):
    with torch.no_grad():
        if voxel_grid is not None:
            voxel_grid = voxel_grid.to(device)
        output = model(
            prompts=prompt, num_frames=num_frames,
            num_denoising_steps=50, cfg_weight=[2.0, 2.0, 2.0],
            scene_input=voxel_grid, return_numpy=True,
        )
    return output


def _prepare_3d(posed_joints, root_positions):
    j3d = np.zeros_like(posed_joints)
    j3d[:, :, 0] = posed_joints[:, :, 0] * METER_TO_UNIT
    j3d[:, :, 1] = posed_joints[:, :, 2] * METER_TO_UNIT
    j3d[:, :, 2] = posed_joints[:, :, 1] * METER_TO_UNIT
    r3d = np.zeros_like(root_positions)
    r3d[:, 0] = root_positions[:, 0] * METER_TO_UNIT
    r3d[:, 1] = root_positions[:, 2] * METER_TO_UNIT
    r3d[:, 2] = root_positions[:, 1] * METER_TO_UNIT
    return j3d, r3d


def _draw_skeleton(ax, j3d, r3d, frame_idx, n_joints, color, trail_color, scope):
    jf = j3d[frame_idx]
    for a, b in SKELETON_CONNECTIONS:
        if a < n_joints and b < n_joints:
            ax.plot([jf[a, 0], jf[b, 0]], [jf[a, 1], jf[b, 1]], [jf[a, 2], jf[b, 2]],
                    color=color, linewidth=3, zorder=8, alpha=0.85)
    ax.scatter(jf[:, 0], jf[:, 1], jf[:, 2], c=color, s=40,
               depthshade=False, zorder=10, edgecolors='white', linewidths=0.8)
    if frame_idx > 0:
        s = max(0, frame_idx - 30)
        t = r3d[s:frame_idx + 1]
        if len(t) > 1:
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=trail_color, linewidth=2.5, alpha=0.75, zorder=6)
    ax.set_xlim(scope["x_min"], scope["x_max"])
    ax.set_ylim(scope["y_min"], scope["y_max"])
    ax.set_zlim(scope.get("z_min", 0), scope.get("z_max", 250))
    ax.set_axis_off(); ax.grid(False)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False


def load_scene_pointcloud(scene_name, max_pts=8000):
    """Load actual scene mesh from LINGO and sample to point cloud in rendering coords.

    Loads mesh_low.obj, merges sub-geometries with scene-graph transforms,
    samples surface points, then converts to rendering coordinates
    (x→x, Y↔Z swap, ×METER_TO_UNIT) matching _prepare_3d.
    """
    import trimesh

    mesh_dir = PROJECT_ROOT / "LINGO" / "scene_mesh" / "Scene_mesh"
    mesh_path = mesh_dir / scene_name / "mesh_low.obj"
    if not mesh_path.exists():
        print(f"  ⚠️  Scene mesh not found: {mesh_path}")
        return np.zeros((0, 3), dtype=np.float32)

    scene_obj = trimesh.load(str(mesh_path), force="scene")
    all_verts = []
    all_faces = []
    offset = 0
    for name, geom in scene_obj.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            verts = np.array(geom.vertices)
            transform = scene_obj.graph.get(name)[0]
            if transform is not None:
                verts = trimesh.transform_points(verts, transform)
            faces = np.array(geom.faces) + offset
            all_verts.append(verts)
            all_faces.append(faces)
            offset += len(verts)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32)

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    n_pts = min(max_pts, max(100, len(faces) * 3))
    points, _ = trimesh.sample.sample_surface(mesh, n_pts)

    pts_render = np.zeros_like(points, dtype=np.float32)
    pts_render[:, 0] = points[:, 0] * METER_TO_UNIT
    pts_render[:, 1] = points[:, 2] * METER_TO_UNIT
    pts_render[:, 2] = points[:, 1] * METER_TO_UNIT

    print(f"  Loaded scene pointcloud: {len(pts_render)} points")
    return pts_render


def render_scene_comparison(out_no, out_with, scene_name, output_path,
                            prompt, j_rmse, r_rmse, alpha_val, scene_pts=None, fps=30):
    jn, rn = _prepare_3d(out_no["posed_joints"], out_no["root_positions"])
    js, rs = _prepare_3d(out_with["posed_joints"], out_with["root_positions"])
    nf = min(jn.shape[0], js.shape[0])

    all_pts = [jn.reshape(-1, 3), js.reshape(-1, 3)]
    if scene_pts is not None and len(scene_pts) > 0:
        all_pts.append(scene_pts.reshape(-1, 3))
    all_pts = np.concatenate(all_pts, axis=0)
    center = np.mean(all_pts, axis=0)
    spread = np.max(np.abs(all_pts - center)) + 50
    scope = {
        "x_min": center[0] - spread, "x_max": center[0] + spread,
        "y_min": center[1] - spread, "y_max": center[1] + spread,
        "z_min": center[2] - spread * 0.5, "z_max": center[2] + spread,
    }

    fig = plt.figure(figsize=(20, 10), facecolor="black")
    ax_l = fig.add_subplot(121, projection="3d", facecolor="#0a0a1a")
    ax_r = fig.add_subplot(122, projection="3d", facecolor="#0a0a1a")
    fig.subplots_adjust(wspace=0.02)

    writer = FFMpegWriter(fps=fps)
    writer.setup(fig, str(output_path), dpi=80)
    print(f"  Rendering: {nf} frames → {Path(output_path).name} ...")

    try:
        for fi in tqdm(range(nf), desc=f"  {scene_name[:20]}"):
            ax_l.cla(); ax_r.cla()
            ax_l.set_facecolor("#0a0a1a"); ax_r.set_facecolor("#0a0a1a")

            if scene_pts is not None and len(scene_pts) > 0:
                ax_l.scatter(
                    scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                    c="#606060", s=1.5, alpha=0.3, depthshade=True, zorder=2,
                    label="Scene")
                ax_r.scatter(
                    scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                    c="#606060", s=1.5, alpha=0.3, depthshade=True, zorder=2)

            _draw_skeleton(ax_l, jn, rn, fi, jn.shape[1], "#00DDFF", "#00DDFF", scope)
            _draw_skeleton(ax_r, js, rs, fi, js.shape[1], "#FF8800", "#FF8800", scope)

            ax_l.legend(loc="upper left", fontsize=7, facecolor="#0a0a1a80",
                        edgecolor="white", labelcolor="white")

            ax_l.view_init(elev=90, azim=-90)
            ax_r.view_init(elev=90, azim=-90)
            fig.suptitle(
                f'{scene_name} | "{prompt}" | α={alpha_val:.4f} | '
                f'joints Δ={j_rmse:.3f}m | root Δ={r_rmse:.3f}m',
                color="white", fontsize=10, y=0.02)
            ax_l.set_title("NO Scene", color="#00DDFF", fontsize=12, pad=8, fontweight="bold")
            ax_r.set_title("WITH Scene", color="#FF8800", fontsize=12, pad=8, fontweight="bold")
            writer.grab_frame()
    finally:
        writer.finish()

    plt.close()
    print(f"  ✅ {Path(output_path).name}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default="kimodo_scene_project/outputs/test_with_scene")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Limit N runs (0=all)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cpu"
    print(f"Device: {device}")

    model = load_sceneco_old(str(ckpt_path), device)

    results = []
    run_count = 0

    for prompt, num_frames in TEST_PROMPTS:
        for scene_name in SCENE_LIST:
            if args.limit > 0 and run_count >= args.limit:
                break
            run_count += 1

            print(f"\n{'='*60}")
            print(f'  [{run_count}] "{prompt}" | {scene_name}')
            print(f"{'='*60}")

            voxel = load_voxel_scene(scene_name)
            scene_pts = load_scene_pointcloud(scene_name)

            print("  → Generating WITHOUT scene ...")
            out_no = generate_motion(model, prompt, num_frames, device, voxel_grid=None)

            print("  → Generating WITH scene ...")
            out_with = generate_motion(model, prompt, num_frames, device, voxel_grid=voxel)

            j_rmse = float(np.sqrt(((out_no["posed_joints"] - out_with["posed_joints"]) ** 2).mean()))
            r_rmse = float(np.sqrt(((out_no["root_positions"] - out_with["root_positions"]) ** 2).mean()))
            print(f"  RMSE: joints={j_rmse:.4f}m  root={r_rmse:.4f}m")

            alphas = {}
            for name, param in model.named_parameters():
                if "alpha" in name and param.numel() == 1:
                    alphas[name.split(".")[-1]] = param.item()
            alpha_first = list(alphas.values())[0] if alphas else 0.0

            safe = f"{prompt[:20].replace(' ','_')}_{scene_name}"
            vid = str(output_dir / f"cmp_{safe}.mp4")
            render_scene_comparison(out_no, out_with, scene_name, vid,
                                    prompt, j_rmse, r_rmse, alpha_first,
                                    scene_pts=scene_pts, fps=args.fps)

            results.append({
                "prompt": prompt, "scene": scene_name,
                "joints_rmse": j_rmse, "root_rmse": r_rmse,
                "alpha_raw": alpha_first, "video": vid,
            })

        if args.limit > 0 and run_count >= args.limit:
            break

    with open(output_dir / "scene_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if results:
        all_j = [r["joints_rmse"] for r in results]
        all_r = [r["root_rmse"] for r in results]
        print(f"\n{'='*60}")
        print(f"  SCENE TEST SUMMARY ({len(results)} runs)")
        print(f"{'='*60}")
        print(f"  Avg joints RMSE (scene vs no-scene): {np.mean(all_j):.4f} ± {np.std(all_j):.4f}")
        print(f"  Max joints RMSE:                     {max(all_j):.4f}")
        print(f"  Avg root RMSE:                       {np.mean(all_r):.4f} ± {np.std(all_r):.4f}")
        print(f"  Report: {output_dir}/scene_report.json")
        for r in results:
            print(f"    [{r['scene']}] {r['prompt'][:30]}: joints={r['joints_rmse']:.4f} root={r['root_rmse']:.4f}")
        print("=" * 60)

    print("✅ Scene test complete!")


if __name__ == "__main__":
    main()
