#!/usr/bin/env python
"""Experiment: feed GT local root to Kimodo stage2, compare generated vs GT body.

Fixes v2:
  - Pass actual text for guidance
  - Constrain root Y (height) as well
  - Visualize root trajectory alongside skeleton
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

import os
os.environ["CHECKPOINT_DIR"] = str(PROJECT_ROOT / "kimodo_scene_project/models")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from tqdm import tqdm

from kimodo.model.load_model import load_model
from kimodo.constraints import Root2DConstraintSet
from kimodo.skeleton import SMPLXSkeleton22


class ZeroTextEncoder:
    """Returns zero text embeddings."""
    output_dim = 4096
    llm_dim = 4096
    max_len = 77

    def __call__(self, texts, device=None):
        B = len(texts)
        text_feat = torch.zeros(B, 1, self.output_dim)
        text_length = torch.ones(B, dtype=torch.long)
        if device is not None:
            text_feat = text_feat.to(device)
            text_length = text_length.to(device)
        return text_feat, text_length

    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self


# ── Bone connections for SMPLX22 ──────────────────────────────────────────────
SMPLX_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

BONE_GROUPS = {
    "spine": [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)],
    "left_leg": [(0, 1), (1, 4), (4, 7), (7, 10)],
    "right_leg": [(0, 2), (2, 5), (5, 8), (8, 11)],
    "left_arm": [(9, 13), (13, 16), (16, 18), (18, 20)],
    "right_arm": [(9, 14), (14, 17), (17, 19), (19, 21)],
}
BONE_COLORS = {
    "spine": "#FF8C00", "left_leg": "#4169E1", "right_leg": "#DC143C",
    "left_arm": "#32CD32", "right_arm": "#FF69B4",
}
CONN_TO_GROUP = {}
for g, conns in BONE_GROUPS.items():
    for c in conns:
        CONN_TO_GROUP[c] = g


SCENE_DIR = Path("LINGO/dataset/dataset/Scene")
SCENE_VOXEL_SIZE = 0.1  # meters per voxel in the 64^3 downsampled grid
SCENE_RESOLUTION = 64    # target downsampled resolution from training


def _find_scene_voxel(scene_name):
    """Find and load the scene voxel grid for a given scene name.

    Returns (occupied_xyz, scene_extent_m) where occupied_xyz is [N, 3] in
    world meters and scene_extent_m is the physical extent [Xm, Ym, Zm].
    Follows the same downsampling convention as SceneCo training
    (voxel_size=0.1m, 64^3 resolution).
    """
    if not SCENE_DIR.exists():
        return None
    base_name = scene_name.split("-")[0]
    for suffix in [scene_name, base_name]:
        path = SCENE_DIR / f"{suffix}.npy"
        if path.exists():
            raw = np.load(str(path)).astype(np.float32)
            from scipy.ndimage import zoom as scipy_zoom
            s = raw.shape  # e.g. (300, 100, 400) → axes (Z, Y, X)
            zoom_factors = (SCENE_RESOLUTION / s[0], SCENE_RESOLUTION / s[1], SCENE_RESOLUTION / s[2])
            ds = scipy_zoom(raw, zoom_factors, order=0) > 0.5  # (64, 64, 64) in (Z, Y, X)
            idx = np.argwhere(ds)  # [N, 3] in voxel coords (iz, iy, ix)
            if len(idx) == 0:
                return None
            extent = np.array(s) * SCENE_VOXEL_SIZE  # physical extent in meters
            # Convert voxel idx → world meters (same as training: voxel 0 = world 0)
            world_pts = idx.astype(np.float32) * SCENE_VOXEL_SIZE  # [N, 3] in (Z, Y, X) meters
            # Subsample for performance
            if len(world_pts) > 3000:
                rng = np.random.RandomState(42)
                world_pts = world_pts[rng.choice(len(world_pts), 3000, replace=False)]
            return world_pts  # [N, 3] columns are (Z_m, Y_m, X_m)
    return None


def load_gt_sample(cache_idx=115, cache_dir="lingo_smplx_cache"):
    """Load GT motion, root features, and scene voxels from cache."""
    path = Path(cache_dir) / f"seg_{cache_idx:05d}.npz"
    data = np.load(str(path), allow_pickle=True)
    T = int(data["length"])
    features = data["motion_features"][:T].copy()

    joints_all = np.load("LINGO/dataset/dataset/human_joints_aligned.npy", mmap_mode="r")
    start_idx = np.load("LINGO/dataset/dataset/start_idx.npy").flatten()
    end_idx = np.load("LINGO/dataset/dataset/end_idx.npy").flatten()

    count = 0
    s, e = 0, 0
    for i in range(len(start_idx)):
        si, ei = int(start_idx[i]), int(end_idx[i])
        if 40 <= ei - si <= 196:
            if count == cache_idx:
                s, e = si, ei
                break
            count += 1
    gt_joints = joints_all[s:e, :22, :].copy()

    scene_name = str(data.get("scene_name", ""))
    scene_voxel = _find_scene_voxel(scene_name) if scene_name else None

    return {
        "features": features,
        "T": T,
        "text": str(data.get("text", "motion")),
        "gt_joints": gt_joints,
        "scene_name": scene_name,
        "scene_voxel": scene_voxel,
    }


def extract_root_features(features, motion_rep):
    """Extract and unnormalize global root features (smooth_root_pos + heading)."""
    feat_t = torch.from_numpy(features).float().unsqueeze(0)
    unnorm = motion_rep.unnormalize(feat_t)

    import einops
    smooth_root_pos, global_root_heading, _, _, _, _ = einops.unpack(
        unnorm, motion_rep.ps, "batch time *"
    )
    return smooth_root_pos.squeeze(0), global_root_heading.squeeze(0)


def build_root_constraint(smooth_root_pos, global_root_heading, skeleton, device="cuda"):
    """Build constraints: XZ + Y + heading (full root)."""
    # smooth_root_2d: XZ (dims 0, 2 of smooth_root_pos)
    smooth_root_2d = torch.cat([
        smooth_root_pos[:, 0:1],
        smooth_root_pos[:, 2:3],
    ], dim=-1)
    frame_indices = torch.arange(len(smooth_root_pos), device=device)

    constraint = Root2DConstraintSet(
        skeleton=skeleton,
        frame_indices=frame_indices,
        smooth_root_2d=smooth_root_2d.to(device),
        global_root_heading=global_root_heading.to(device),
    )

    # Wrap to also add root_y_pos constraint
    root_y = smooth_root_pos[:, 1:2].squeeze(-1).to(device)  # [T]

    class ConstraintWithY:
        def __init__(self, inner, fy, fi):
            self.inner = inner
            self.root_y = fy
            self.frame_indices = fi

        def update_constraints(self, data_dict, index_dict):
            self.inner.update_constraints(data_dict, index_dict)
            index_dict.setdefault("root_y_pos", []).append(self.frame_indices)
            data_dict.setdefault("root_y_pos", []).append(self.root_y)

    return [ConstraintWithY(constraint, root_y, frame_indices)]


def _build_observed_motion_and_mask(features, mr, device, use_init_pose=False):
    """Build observed_motion and motion_mask tensors for root constraint.

    When use_init_pose=True, also constrains frame-0 body features (initial pose).
    Returns (observed_motion, motion_mask) both [1, T, D].
    """
    feat_t = torch.from_numpy(features).float().unsqueeze(0)  # [1, T, D]
    norm_feat = mr.normalize(feat_t)  # [1, T, D] normalized

    T = feat_t.shape[1]
    D = feat_t.shape[-1]
    root_slice = mr.root_slice
    body_slice = mr.body_slice

    observed_motion = torch.zeros(1, T, D, device=device)
    motion_mask = torch.zeros(1, T, D, device=device)

    # Root constraint on ALL frames
    observed_motion[..., root_slice] = norm_feat[..., root_slice].to(device)
    motion_mask[..., root_slice] = 1.0

    # Initial pose constraint on frame 0 body features
    if use_init_pose:
        observed_motion[:, 0:1, body_slice] = norm_feat[:, 0:1, body_slice].to(device)
        motion_mask[:, 0:1, body_slice] = 1.0

    return observed_motion, motion_mask


def generate_with_root(model, sample, device, num_denoising_steps=50, use_init_pose=False):
    """Generate body conditioned on GT root trajectory (XZ + Y + heading).

    Optionally also constrains the initial body pose (frame 0).

    After generation, stitches GT root onto the output so the comparison
    isolates body-pose quality rather than root-tracking accuracy.
    """
    T = sample["T"]
    features = sample["features"]
    skel = SMPLXSkeleton22()

    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    mr = KimodoMotionRep(fps=30, stats_path="models/Kimodo-SMPLX-RP-v1/stats/motion", skeleton=skel)

    # Extract GT root for later stitching
    smooth_root_pos, heading = extract_root_features(features, mr)
    gt_root_np = smooth_root_pos.cpu().numpy()  # [T, 3]

    with torch.no_grad():
        if use_init_pose:
            # ── Manual path: build observed_motion/motion_mask directly ──
            observed_motion, motion_mask = _build_observed_motion_and_mask(
                features, mr, device, use_init_pose=True)

            # Use model.text_encoder to encode the text prompt
            text_feat, text_length = model.text_encoder([sample["text"]])
            text_feat = text_feat.to(device)
            B, maxlen = text_feat.shape[:2]
            text_len_t = torch.tensor(text_length, device=device)
            text_pad_mask = torch.arange(maxlen, device=device).expand(B, maxlen) < text_len_t[:, None]

            pad_mask = torch.ones(1, T, dtype=torch.bool, device=device)
            fha = torch.zeros(1, device=device)

            motion_feat = model._generate(
                texts=[sample["text"]],
                max_frames=T,
                num_denoising_steps=num_denoising_steps,
                pad_mask=pad_mask,
                first_heading_angle=fha,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=[2.0, 2.0],
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
            )

            # Decode motion features → joints
            output = model.motion_rep.inverse(
                motion_feat, is_normalized=True, return_numpy=False,
            )
            output = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in output.items()}
        else:
            # ── Original path: use constraint_lst ──
            constraints = build_root_constraint(smooth_root_pos, heading, skel, device)
            print(f"  Text: '{sample['text']}'  |  cfg_weight=[2.0, 2.0]")
            output = model(
                prompts=sample["text"],
                num_frames=T,
                num_denoising_steps=num_denoising_steps,
                constraint_lst=[constraints],
                cfg_weight=[2.0, 2.0],
                return_numpy=True,
            )

    # ── Stitch GT root onto generated output ──────────────────────────────
    gen_root_np = np.squeeze(output["smooth_root_pos"]).copy()  # [T, 3]
    gen_joints = np.squeeze(output["posed_joints"]).copy()  # [T, 22, 3]

    # Remove generated root offset, add GT root offset
    root_delta = gt_root_np - gen_root_np  # [T, 3]
    gen_joints += root_delta[:, None, :]   # broadcast over 22 joints

    output["posed_joints"] = gen_joints[np.newaxis, ...]
    output["smooth_root_pos"] = gt_root_np[np.newaxis, ...]

    return output, smooth_root_pos


# ── Visualization ────────────────────────────────────────────────────────────
def render_comparison(gt_joints, gen_output, gt_root_pos, exp_label, out_path,
                      scene_voxel=None, scene_name=""):
    """Render GT and generated side-by-side with root trajectory and 3D scene."""
    gen_joints = np.squeeze(gen_output["posed_joints"]).copy()
    gen_root_pos = np.squeeze(gen_output.get("smooth_root_pos")).copy()
    T = min(gt_joints.shape[0], gen_joints.shape[0])

    # Save unshifted root for trajectory plotting
    gt_root = gt_root_pos[:T].copy()  # [T, 3] -- unaligned GT root
    gen_root = gen_root_pos[:T].copy()  # [T, 3] -- unaligned generated root

    # Y-align both to ground
    ft_gt = gt_joints[..., 1].min()
    gt_joints[..., 1] -= ft_gt
    gt_root[..., 1] -= ft_gt

    ft_gen = gen_joints[..., 1].min()
    gen_joints[..., 1] -= ft_gen
    gen_root[..., 1] -= ft_gen

    # Pre-compute scene points (static, drawn once per frame)
    scene_pts_plot = None  # (X, Z, Y) for matplotlib
    if scene_voxel is not None:
        # scene_voxel is [N, 3] in (Z_m, Y_m, X_m) — align with person
        scene_x = scene_voxel[:, 2]  # X meters
        scene_y = scene_voxel[:, 1]  # Y meters
        scene_z = scene_voxel[:, 0]  # Z meters
        # Center scene on person's root trajectory mean
        cx = gt_root[:, 0].mean()
        cy = 0.0  # ground
        cz = gt_root[:, 2].mean()
        scene_x = scene_x - scene_x.mean() + cx
        scene_y = scene_y - scene_y.min() + cy
        scene_z = scene_z - scene_z.mean() + cz
        scene_pts_plot = np.stack([scene_x, scene_z, scene_y], axis=-1)  # (X, Z, Y) for ax.plot

    # Compute view bounds (include scene points)
    all_x = [gt_joints[..., 0].ravel(), gen_joints[..., 0].ravel()]
    all_y = [gt_joints[..., 1].ravel(), gen_joints[..., 1].ravel()]
    all_z = [gt_joints[..., 2].ravel(), gen_joints[..., 2].ravel()]
    if scene_pts_plot is not None:
        all_x.append(scene_pts_plot[:, 0])
        all_z.append(scene_pts_plot[:, 1])
        all_y.append(scene_pts_plot[:, 2])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    margin = 0.5
    x_range = [all_x.min() - margin, all_x.max() + margin]
    z_range = [all_z.min() - margin, all_z.max() + margin]
    y_range = [all_y.min() - 0.3, all_y.max() + 0.5]

    fig = plt.figure(figsize=(16, 7), dpi=100, facecolor="white")

    def setup_ax(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.xaxis.line.set_visible(False)
        ax.yaxis.line.set_visible(False)
        ax.zaxis.line.set_visible(False)
        ax.tick_params(axis="x", colors="white", labelcolor="white")
        ax.tick_params(axis="y", colors="white", labelcolor="white")
        ax.tick_params(axis="z", colors="white", labelcolor="white")
        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_zlim(y_range)
        ax.view_init(elev=15, azim=-55)

    def draw_frame(frame):
        plt.clf()
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
        ax_gen = fig.add_subplot(1, 2, 2, projection="3d")

        for ax, title, joints, root_traj in [
            (ax_gt, f"GT (LINGO)", gt_joints, gt_root),
            (ax_gen, f"Generated (root=GT, text='{exp_label[:20]}')", gen_joints, gen_root),
        ]:
            setup_ax(ax)

            # Draw scene voxels (static, semi-transparent)
            if scene_pts_plot is not None:
                ax.scatter(scene_pts_plot[:, 0], scene_pts_plot[:, 1], scene_pts_plot[:, 2],
                           c="#8B4513", s=8, alpha=0.12, marker="s",
                           edgecolors="none")

            jf = joints[min(frame, len(joints) - 1)]

            # Draw skeleton bones
            for c in SMPLX_CONNECTIONS:
                grp = CONN_TO_GROUP[c]
                ax.plot(
                    [jf[c[0], 0], jf[c[1], 0]],
                    [jf[c[0], 2], jf[c[1], 2]],
                    [jf[c[0], 1], jf[c[1], 1]],
                    color=BONE_COLORS[grp], linewidth=3, alpha=0.9,
                )
            ax.scatter(jf[:, 0], jf[:, 2], jf[:, 1],
                       c="black", s=15, alpha=0.7,
                       edgecolors="white", linewidth=0.5)

            # Draw root trajectory (past: fade, future: dim)
            tf = min(frame, T - 1)
            # past trajectory (solid, fading alpha by distance)
            past = max(0, tf - 30)
            for t in range(past, tf + 1):
                alpha = 0.15 + 0.85 * (t - past) / max(1, tf - past)
                ax.scatter(root_traj[t, 0], root_traj[t, 2], root_traj[t, 1],
                           c="#00BCD4", s=8, alpha=alpha)
            # future trajectory (dim dots)
            future_end = min(T, tf + 10)
            for t in range(tf + 1, future_end):
                ax.scatter(root_traj[t, 0], root_traj[t, 2], root_traj[t, 1],
                           c="#00BCD4", s=4, alpha=0.1)

            # Draw full root trail as thin line
            ax.plot(root_traj[:tf+1, 0], root_traj[:tf+1, 2], root_traj[:tf+1, 1],
                    color="#00BCD4", linewidth=1.5, alpha=0.5)

            ax.set_title(title, fontsize=10, fontweight="bold")

    ani = animation.FuncAnimation(fig, lambda f: (draw_frame(f), [])[1],
                                   frames=T, interval=40, blit=False)
    writer = animation.FFMpegWriter(fps=25, bitrate=2000)
    ani.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Quantitative comparison ──────────────────────────────────────────────────
def compute_metrics(gt_joints, gen_joints):
    """Compute MPJPE and bone direction error."""
    T = min(gt_joints.shape[0], gen_joints.shape[0])
    gt = gt_joints[:T]
    gen = gen_joints[:T]

    mpjpe = np.linalg.norm(gt - gen, axis=2).mean(axis=0)
    mean_mpjpe = mpjpe.mean()

    parents = SMPLXSkeleton22().joint_parents
    valid = [j for j in range(22) if int(parents[j]) >= 0]
    pidx = [int(parents[j]) for j in valid]

    gt_dirs = gt[:, valid] - gt[:, pidx]
    gt_dirs = gt_dirs / (np.linalg.norm(gt_dirs, axis=2, keepdims=True) + 1e-8)
    gen_dirs = gen[:, valid] - gen[:, pidx]
    gen_dirs = gen_dirs / (np.linalg.norm(gen_dirs, axis=2, keepdims=True) + 1e-8)

    dot = np.clip((gt_dirs * gen_dirs).sum(axis=2), -1, 1)
    angle_err = np.degrees(np.arccos(dot)).mean()

    return mean_mpjpe, angle_err


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_idx", type=int, nargs="+", default=[115, 174, 89])
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--output_dir", default="kimodo_scene_project/outputs/exp_root_stage2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--init_pose", action="store_true", default=False,
                        help="Constrain the initial (frame-0) body pose to GT")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Kimodo model (ZeroTextEncoder — no text download needed) ...")
    model = load_model(
        "Kimodo-SMPLX-RP-v1",
        device=device,
        text_encoder=ZeroTextEncoder(),
    )
    model.eval()
    print("  Model loaded.")

    for ci in args.cache_idx:
        print(f"\n{'=' * 60}")
        sample = load_gt_sample(ci)
        text = sample["text"]
        scene_name = sample.get("scene_name", "")
        scene_voxel = sample.get("scene_voxel")
        mode_str = "root+init_pose" if args.init_pose else "root_only"
        print(f"Processing seg_{ci:05d} | T={sample['T']} | text='{text}' | scene='{scene_name}' | mode={mode_str}")
        print(f"{'=' * 60}")

        print(f"  Generating with GT root constraint (XZ+Y+heading) + text"
              f"{' + init_pose' if args.init_pose else ''} ...")
        gen, gt_root = generate_with_root(model, sample, device, args.num_denoising_steps,
                                           use_init_pose=args.init_pose)

        gen_joints = np.squeeze(gen["posed_joints"])
        gt_joints = sample["gt_joints"]

        mpjpe, angle_err = compute_metrics(gt_joints, gen_joints)
        print(f"  MPJPE: {mpjpe * 100:.1f} cm  |  Bone angle err: {angle_err:.1f} deg")

        suffix = "_initpose" if args.init_pose else ""
        out_path = out_dir / f"cmp_{ci:05d}{suffix}.mp4"
        render_comparison(gt_joints, gen, gt_root.numpy(), text, out_path,
                          scene_voxel=scene_voxel, scene_name=scene_name)

    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
