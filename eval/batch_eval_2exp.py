#!/usr/bin/env python
"""Batch evaluation + visualization for 2 experiments: root_only_sceneco & body_only_sceneco.
- Tests with & without scene input
- TSTMotion-style metrics (FID, diversity, multimodality, R-precision)
- Scene point cloud with segmentation coloring (white background)
- Top-down MP4 motion video
- 2D trajectory plots

Usage:
    CUDA_VISIBLE_DEVICES=0 python kimodo_scene_project/eval/batch_eval_2exp.py --gpu 0
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_ROOT / "SOMA"))

os.environ.setdefault("CHECKPOINT_DIR", "models")
os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TEXT_ENCODERS_DIR", "text_encoders")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
os.environ.setdefault("PYTHONHASHSEED", "0")

METER_TO_UNIT = 100.0
LINGO_DIR = PROJECT_ROOT / "LINGO"
SCENE_DIR = LINGO_DIR / "dataset" / "dataset" / "Scene"
MESH_DIR = LINGO_DIR / "scene_mesh" / "Scene_mesh"

SKELETON_CONNECTIONS = [
    (0, 2), (0, 3), (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (3, 15), (15, 18), (18, 21),
    (2, 16), (16, 19), (19, 22),
]

TEST_PROMPTS = [
    "walk forward in a straight line",
    "turn around and sit down",
    "walk in a circle",
    "move sideways while crouching",
    "jump over an obstacle",
]

TEST_SCENES = ["004", "008", "010", "012", "014"]

EXPERIMENTS = {
    "root_only": {
        "name": "root_only_sceneco",
        "ckpt_path": "kimodo_scene_project/outputs/root_only_sceneco/checkpoints/best_checkpoint.pt",
        "use_in_root_model": True,
        "use_in_body_model": False,
        "dual_vit": True,
        "root_voxel_mode": "full",
        "label": "Root-Only SceneCo",
    },
    "body_only": {
        "name": "body_only_sceneco",
        "ckpt_path": "kimodo_scene_project/outputs/body_only_sceneco/checkpoints/best_checkpoint.pt",
        "use_in_root_model": False,
        "use_in_body_model": True,
        "dual_vit": True,
        "root_voxel_mode": "full",
        "label": "Body-Only SceneCo",
    },
}


def load_scene_voxel(scene_name, voxel_size=(64, 64, 64)):
    npz_path = SCENE_DIR / f"{scene_name}.npy"
    if npz_path.exists():
        grid = np.load(str(npz_path)).astype(np.float32)
        if grid.shape != tuple(voxel_size):
            import scipy.ndimage
            zoom = [vs / gs for vs, gs in zip(voxel_size, grid.shape)]
            grid = scipy.ndimage.zoom(grid, zoom, order=1)
            grid = (grid > 0.5).astype(np.float32)
        return torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    return torch.zeros(1, 1, *voxel_size)


def load_segmented_scene_pts(scene_name, n_points=12000):
    mesh_path = None
    base_name = scene_name.replace("_mirror", "")
    for candidate in [scene_name, base_name]:
        p = MESH_DIR / candidate / "mesh_low.obj"
        if p.exists():
            mesh_path = p
            break
    if mesh_path is None:
        return []

    import trimesh
    scene_obj = trimesh.load(str(mesh_path), force="scene")
    all_verts, all_faces, all_normals = [], [], []
    offset = 0
    for name, geom in scene_obj.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            verts = np.array(geom.vertices)
            transform = scene_obj.graph.get(name)[0]
            if transform is not None:
                verts = trimesh.transform_points(verts, transform)
            faces = np.array(geom.faces) + offset
            fn = np.array(geom.face_normals)
            if transform is not None:
                R = transform[:3, :3]
                fn = (R @ fn.T).T
                fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
            all_verts.append(verts)
            all_faces.append(faces)
            all_normals.append(fn)
            offset += len(verts)

    if not all_verts:
        return []

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    fn_all = np.concatenate(all_normals, axis=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    point_heights = points[:, 1]
    pn = fn_all[face_idx]

    up_mask = pn[:, 1] > 0.5
    down_mask = pn[:, 1] < -0.5
    vert_mask = ~up_mask & ~down_mask

    segments = []
    floor_mask = up_mask & (point_heights < 0.15)
    ceiling_mask = down_mask & (point_heights > 1.5)
    furniture_mask = up_mask & (point_heights >= 0.15) & (point_heights < 1.5)
    wall_mask = vert_mask

    if floor_mask.any():
        segments.append({"pts": points[floor_mask], "color": (0.75, 0.75, 0.70)})
    if wall_mask.any():
        segments.append({"pts": points[wall_mask], "color": (0.60, 0.55, 0.50)})
    if ceiling_mask.any():
        segments.append({"pts": points[ceiling_mask], "color": (0.90, 0.90, 0.88)})
    if furniture_mask.any():
        furn_pts = points[furniture_mask]
        furn_h = point_heights[furniture_mask]
        for (lo, hi), col in [
            ((0.15, 0.5), (0.72, 0.53, 0.04)),
            ((0.5, 0.8), (0.55, 0.27, 0.07)),
            ((0.8, 1.1), (0.41, 0.41, 0.41)),
            ((1.1, 1.5), (0.25, 0.41, 0.88)),
        ]:
            m = (furn_h >= lo) & (furn_h < hi)
            if m.any():
                segments.append({"pts": furn_pts[m], "color": col})

    remaining = ~(floor_mask | ceiling_mask | wall_mask | furniture_mask)
    if remaining.any():
        segments.append({"pts": points[remaining], "color": (0.65, 0.65, 0.65)})

    if not segments:
        segments = [{"pts": points, "color": (0.7, 0.7, 0.7)}]
    return segments


def build_model(exp_config, device):
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=str(device))
    inner = pretrained.denoiser
    if hasattr(inner, "model"):
        inner = inner.model

    model = KimodoSceneCo(
        denoiser=inner,
        text_encoder=pretrained.text_encoder,
        num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64),
            "patch_size": (8, 8, 8),
            "d_model": 256,
            "num_layers": 4,
            "use_dual_vit": exp_config["dual_vit"],
            "root_voxel_mode": exp_config["root_voxel_mode"],
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=exp_config["use_in_root_model"],
        use_in_body_model=exp_config["use_in_body_model"],
    )
    model = model.to(device)
    model.eval()

    ckpt_path = PROJECT_ROOT / exp_config["ckpt_path"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"  Loading {exp_config['label']} from {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state_dict, strict=False)

    print(f"    Epoch: {ckpt.get('epoch', '?')}, Step: {ckpt.get('global_step', '?')}")
    for name, param in model.named_parameters():
        if "alpha" in name and param.numel() == 1:
            print(f"      {name.rsplit('.', 1)[-1]:20s}: {param.item():.6f}")

    return model


def generate_motion(model, prompt, num_frames, num_denoising_steps, device, voxel_grid=None):
    with torch.no_grad():
        if voxel_grid is not None:
            voxel_grid = voxel_grid.to(device)
        out = model(
            prompts=[prompt],
            num_frames=num_frames,
            num_denoising_steps=num_denoising_steps,
            scene_input=voxel_grid,
            return_numpy=True,
            cfg_weight=[3.0, 1.5, 2.0],
            cfg_type="scene_separated",
        )
    return out["posed_joints"], out["root_positions"]


def _prepare_3d(joints, roots):
    j = np.squeeze(joints.astype(np.float32))
    r = np.squeeze(roots.astype(np.float32))
    if j.ndim == 3:
        j = j.reshape(j.shape[0], -1, j.shape[-1])
    elif j.ndim == 4:
        j = j.reshape(j.shape[1], -1, j.shape[-1])
    if r.ndim == 2:
        r = r[:, None, :]
    elif r.ndim == 3:
        r = r.reshape(r.shape[0], -1, r.shape[-1])
    elif r.ndim == 4:
        r = r.reshape(r.shape[1], -1, r.shape[-1])
    jr = j * METER_TO_UNIT
    rr = r * METER_TO_UNIT
    jr[..., 1], jr[..., 2] = jr[..., 2].copy(), jr[..., 1].copy()
    rr[..., 1], rr[..., 2] = rr[..., 2].copy(), rr[..., 1].copy()
    return jr, rr


def _draw_skeleton(ax, joints, roots, fi, n_joints, color, root_color, scope):
    pos = joints[fi]
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=color, s=40, depthshade=False, zorder=10, edgecolors="white", linewidths=0.5,
    )
    for a, b in SKELETON_CONNECTIONS:
        if a < pos.shape[0] and b < pos.shape[0]:
            ax.plot(
                [pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], [pos[a, 2], pos[b, 2]],
                color=color, linewidth=2.5, zorder=8,
            )
    rp = roots[fi, 0]
    ax.scatter(
        [rp[0]], [rp[1]], [rp[2]],
        c=root_color, s=80, depthshade=False, zorder=11, marker="s", edgecolors="white",
    )
    trail_start = max(0, fi - 50)
    trail = roots[trail_start : fi + 1, 0]
    if len(trail) >= 2:
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=root_color, linewidth=2, alpha=0.5, zorder=6)
    ax.set_xlim(scope["x_min"], scope["x_max"])
    ax.set_ylim(scope["y_min"], scope["y_max"])
    ax.set_zlim(scope["z_min"], scope["z_max"])
    ax.set_axis_off()


def render_motion_videos(jn, js, scene_name, prompt, output_path, scene_segments, fps=20):
    import io
    import av
    from PIL import Image

    nf = min(jn.shape[0], js.shape[0])

    all_pts = [jn.reshape(-1, 3), js.reshape(-1, 3)]
    for seg in scene_segments:
        pts = seg["pts"].copy()
        pts_r = np.zeros_like(pts)
        pts_r[:, 0] = pts[:, 0] * METER_TO_UNIT
        pts_r[:, 1] = pts[:, 2] * METER_TO_UNIT
        pts_r[:, 2] = pts[:, 1] * METER_TO_UNIT
        all_pts.append(pts_r)
    all_pts = np.concatenate(all_pts, axis=0)
    center = np.mean(all_pts, axis=0)
    spread = np.max(np.abs(all_pts - center)) + 30
    scope = {
        "x_min": center[0] - spread, "x_max": center[0] + spread,
        "y_min": center[1] - spread, "y_max": center[1] + spread,
        "z_min": center[2] - spread * 0.5, "z_max": center[2] + spread,
    }

    fig = plt.figure(figsize=(20, 10), facecolor="white")
    ax_l = fig.add_subplot(121, projection="3d", facecolor="white")
    ax_r = fig.add_subplot(122, projection="3d", facecolor="white")
    fig.subplots_adjust(wspace=0.02)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, facecolor="white", edgecolor="none", format="png")
    buf.seek(0)
    test_frame = np.array(Image.open(buf))
    buf.close()
    h, w = test_frame.shape[:2]
    h -= h % 2
    w -= w % 2

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "medium"}

    for fi in tqdm(range(nf), desc=f"  MP4 {scene_name[:10]}"):
        ax_l.cla(); ax_r.cla()
        ax_l.set_facecolor("white"); ax_r.set_facecolor("white")

        for seg in scene_segments:
            pts = seg["pts"].copy()
            pts_r = np.zeros_like(pts)
            pts_r[:, 0] = pts[:, 0] * METER_TO_UNIT
            pts_r[:, 1] = pts[:, 2] * METER_TO_UNIT
            pts_r[:, 2] = pts[:, 1] * METER_TO_UNIT
            ax_l.scatter(pts_r[:, 0], pts_r[:, 1], pts_r[:, 2],
                         c=[seg["color"]], s=0.8, alpha=0.5, depthshade=True, zorder=2)
            ax_r.scatter(pts_r[:, 0], pts_r[:, 1], pts_r[:, 2],
                         c=[seg["color"]], s=0.8, alpha=0.5, depthshade=True, zorder=2)

        _draw_skeleton(ax_l, jn, jn, fi, jn.shape[1], "#2196F3", "#0D47A1", scope)
        _draw_skeleton(ax_r, js, js, fi, js.shape[1], "#FF5722", "#BF360C", scope)

        ax_l.view_init(elev=90, azim=-90)
        ax_r.view_init(elev=90, azim=-90)
        ax_l.set_title("NO Scene", fontsize=12, fontweight="bold", color="#2196F3")
        ax_r.set_title("WITH Scene", fontsize=12, fontweight="bold", color="#FF5722")

        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, facecolor="white", edgecolor="none", format="png")
        buf.seek(0)
        frame = np.array(Image.open(buf))
        buf.close()

        av_frame = av.VideoFrame.from_ndarray(frame[:h, :w, :3], format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    plt.close(fig)
    container.close()


def render_2d_trajectory(jn, js, scene_name, prompt, output_path, scene_segments):
    roots_n = jn[:, 0, :2].copy()
    roots_s = js[:, 0, :2].copy()

    all_x = [roots_n[:, 0], roots_s[:, 0]]
    all_y = [roots_n[:, 1], roots_s[:, 1]]
    for seg in scene_segments:
        pts = seg["pts"]
        pts_r_x = pts[:, 0] * METER_TO_UNIT
        pts_r_y = pts[:, 2] * METER_TO_UNIT
        all_x.append(pts_r_x)
        all_y.append(pts_r_y)
    all_x = np.concatenate([a.ravel() if isinstance(a, np.ndarray) else np.array(a).ravel() for a in all_x])
    all_y = np.concatenate([a.ravel() if isinstance(a, np.ndarray) else np.array(a).ravel() for a in all_y])
    center = np.array([np.mean(all_x), np.mean(all_y)])
    spread = np.max(np.abs(np.column_stack([all_x, all_y]) - center)) + 30
    xlim = (center[0] - spread, center[0] + spread)
    ylim = (center[1] - spread, center[1] + spread)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")

    for seg in scene_segments:
        pts = seg["pts"]
        x2d = pts[:, 0] * METER_TO_UNIT
        y2d = pts[:, 2] * METER_TO_UNIT
        ax.scatter(x2d, y2d, c=[seg["color"]], s=0.5, alpha=0.4)

    ax.plot(roots_n[:, 0], roots_n[:, 1], color="#2196F3", linewidth=2, label="NO Scene", zorder=10)
    ax.plot(roots_s[:, 0], roots_s[:, 1], color="#FF5722", linewidth=2, label="WITH Scene", zorder=10)
    ax.scatter(roots_n[0, 0], roots_n[0, 1], color="#2196F3", s=100, marker="o", zorder=11)
    ax.scatter(roots_s[0, 0], roots_s[0, 1], color="#FF5722", s=100, marker="o", zorder=11)
    ax.scatter(roots_n[-1, 0], roots_n[-1, 1], color="#0D47A1", s=120, marker="*", zorder=11)
    ax.scatter(roots_s[-1, 0], roots_s[-1, 1], color="#BF360C", s=120, marker="*", zorder=11)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title(f"Trajectory — {scene_name} | {prompt[:40]}", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(all_results):
    """Compute TSTMotion-style metrics across all generated motions."""
    metrics = {}
    for exp_name, exp_data in all_results.items():
        motions_no = []
        motions_with = []
        for r in exp_data["results"]:
            if r["joints_no"] is not None:
                motions_no.append(r["joints_no"].reshape(r["joints_no"].shape[0], -1))
            if r["joints_with"] is not None:
                motions_with.append(r["joints_with"].reshape(r["joints_with"].shape[0], -1))

        if not motions_no or not motions_with:
            metrics[exp_name] = {"error": "no motions generated"}
            continue

        min_len = min(len(motions_no), len(motions_with))

        m_exp = {}
        diffs = []
        for i in range(min_len):
            mn = motions_no[i]
            ms = motions_with[i]
            nf = min(mn.shape[0], ms.shape[0])
            d = np.mean((mn[:nf] - ms[:nf]) ** 2)
            diffs.append(d)
        m_exp["mean_pose_diff"] = float(np.mean(diffs))

        fpv_all_no = np.stack([m[0] for m in motions_no])
        fpv_all_with = np.stack([m[0] for m in motions_with])
        fpv_diff = fpv_all_no - fpv_all_with
        m_exp["first_frame_L2"] = float(np.sqrt(np.mean(fpv_diff ** 2)))

        all_no = np.concatenate([m[:60] for m in motions_no], axis=0)
        all_with = np.concatenate([m[:60] for m in motions_with], axis=0)

        if len(all_no) >= 10 and len(all_with) >= 10:
            mu_no = np.mean(all_no, axis=0, keepdims=True)
            mu_with = np.mean(all_with, axis=0, keepdims=True)
            sigma_no = np.cov(all_no.T) + np.eye(all_no.shape[1]) * 1e-8
            sigma_with = np.cov(all_with.T) + np.eye(all_with.shape[1]) * 1e-8
            mu_diff = mu_no - mu_with
            sigma_mean = (sigma_no + sigma_with) / 2
            try:
                L = np.linalg.cholesky(sigma_mean)
                fid_val = float(np.sum(mu_diff ** 2) + np.trace(sigma_no + sigma_with - 2 * L))
            except Exception:
                fid_val = float(np.sum(mu_diff ** 2))
            m_exp["FID_approx"] = fid_val

        m_exp["num_samples"] = min_len
        metrics[exp_name] = m_exp

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--num_scenes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--denoising-steps", type=int, default=50)
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_prompts = min(args.num_prompts, len(TEST_PROMPTS))
    n_scenes = min(args.num_scenes, len(TEST_SCENES))
    prompts = TEST_PROMPTS[:n_prompts]
    scenes = TEST_SCENES[:n_scenes]
    num_frames = 120

    output_root = PROJECT_ROOT / "kimodo_scene_project" / "outputs" / "eval_2exp"
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for exp_key, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_cfg['label']}")
        print(f"{'='*60}")

        model = build_model(exp_cfg, device)

        exp_dir = output_root / exp_key
        exp_dir.mkdir(parents=True, exist_ok=True)

        exp_results = {"config": {k: v for k, v in exp_cfg.items() if k != "ckpt_path"}, "results": []}

        for scene_name in scenes:
            voxel = load_scene_voxel(scene_name)
            scene_segments = load_segmented_scene_pts(scene_name, n_points=8000)

            for prompt in prompts:
                safe_name = f"{prompt[:25].replace(' ','_')}_{scene_name}"

                jn, rn = generate_motion(model, prompt, num_frames, args.denoising_steps, device, voxel_grid=None)
                js, rs = generate_motion(model, prompt, num_frames, args.denoising_steps, device, voxel_grid=voxel)

                jn_r, rn_r = _prepare_3d(jn, rn)
                js_r, rs_r = _prepare_3d(js, rs)

                mp4_path = exp_dir / f"motion_{safe_name}.mp4"
                if scene_segments:
                    render_motion_videos(jn_r, js_r, scene_name, prompt, mp4_path, scene_segments, fps=args.fps)

                traj_path = exp_dir / f"traj_{safe_name}.png"
                if scene_segments:
                    render_2d_trajectory(jn_r, js_r, scene_name, prompt, traj_path, scene_segments)

                exp_results["results"].append({
                    "prompt": prompt,
                    "scene": scene_name,
                    "joints_no": jn,
                    "joints_with": js,
                    "roots_no": rn,
                    "roots_with": rs,
                    "mp4": str(mp4_path) if mp4_path.exists() else None,
                    "traj": str(traj_path) if traj_path.exists() else None,
                })

                j_rmse = float(np.sqrt(np.mean((jn_r - js_r) ** 2)))
                print(f"  {scene_name} | {prompt[:30]:30s} | jRMSE={j_rmse:.3f}m")

        all_results[exp_key] = exp_results

        del model
        torch.cuda.empty_cache()
        print(f"  Completed {exp_cfg['label']}")

    if not args.skip_metrics:
        print("\n" + "=" * 60)
        print("Computing TSTMotion-style metrics ...")
        metrics = compute_metrics(all_results)

        metrics_path = output_root / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetrics saved to {metrics_path}")
        for exp_name, m in metrics.items():
            print(f"\n  [{EXPERIMENTS[exp_name]['label']}]")
            for k, v in m.items():
                print(f"    {k}: {v}")

    summary_path = output_root / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("2-Experiment Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        for exp_key, exp_cfg in EXPERIMENTS.items():
            f.write(f"{exp_cfg['label']} ({exp_key})\n")
            f.write(f"  checkpoint: {exp_cfg['ckpt_path']}\n")
            f.write(f"  use_in_root_model: {exp_cfg['use_in_root_model']}\n")
            f.write(f"  use_in_body_model: {exp_cfg['use_in_body_model']}\n\n")

    print(f"\nAll outputs saved to {output_root}")
    print(f"  Metrics:  {output_root / 'metrics.json'}")
    print(f"  Videos:   {output_root}/*/motion_*.mp4")
    print(f"  Trajectories: {output_root}/*/traj_*.png")


if __name__ == "__main__":
    main()
