#!/usr/bin/env python
"""Compare Kimodo two-stage (root=GT, init_pose, text) vs GT on LINGO scenes.

Renders side-by-side: GT vs Generated, with 3D scene and root trajectory.
"""

import argparse, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

import os
os.environ["CHECKPOINT_DIR"] = str(PROJECT_ROOT / "kimodo_scene_project/models")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import einops

from kimodo.model.load_model import load_model
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton import SMPLXSkeleton22


# ── SMPLX 22-joint skeleton ──────────────────────────────────────────────────
SMPLX_CONNECTIONS = [
    (0,1),(1,4),(4,7),(7,10),       # left leg
    (0,2),(2,5),(5,8),(8,11),       # right leg
    (0,3),(3,6),(6,9),               # spine
    (9,12),(12,15),                   # neck -> head
    (9,13),(13,16),(16,18),(18,20),  # left arm
    (9,14),(14,17),(17,19),(19,21),  # right arm
]
BONE_COLORS = {"spine":"#FF8C00","left_leg":"#4169E1","right_leg":"#DC143C",
               "left_arm":"#32CD32","right_arm":"#FF69B4"}
CONN_TO_GROUP = {}
for grp, conns in [
    ("left_leg", [(0,1),(1,4),(4,7),(7,10)]),
    ("right_leg", [(0,2),(2,5),(5,8),(8,11)]),
    ("spine", [(0,3),(3,6),(6,9),(9,12),(12,15)]),
    ("left_arm", [(9,13),(13,16),(16,18),(18,20)]),
    ("right_arm", [(9,14),(14,17),(17,19),(19,21)]),
]:
    for c in conns:
        CONN_TO_GROUP[c] = grp


# ── Scene helpers ────────────────────────────────────────────────────────────
LINGO_SCENE_DIR = PROJECT_ROOT / "LINGO/dataset/dataset/Scene"

def load_scene_lingo(scene_name, max_height=2.5, max_pts=20000):
    """Load original LINGO hi-res scene and return (pts, colors)."""
    scene_path = LINGO_SCENE_DIR / f"{scene_name}.npy"
    if not scene_path.exists():
        return None, None
    v = np.load(str(scene_path))
    D, H, W = v.shape  # (Z=300, Y=100, X=400)
    voxel_size = 0.02

    # Remove walls
    h_cells = min(int(max_height / voxel_size), H)
    v[:, h_cells:, :] = 0

    oz, oy, ox = np.where(v > 0)
    if len(oz) == 0:
        return None, None

    x_phys = (ox.astype(np.float32) - W / 2) * voxel_size
    y_phys = oy.astype(np.float32) * voxel_size
    z_phys = (oz.astype(np.float32) - D / 2) * voxel_size

    n = len(x_phys)
    if n > max_pts:
        step = max(1, n // max_pts)
        idx = np.arange(0, n, step)
        x_phys, y_phys, z_phys = x_phys[idx], y_phys[idx], z_phys[idx]

    # Color by height
    colors = np.zeros((len(x_phys), 4), dtype=np.float32)
    fl = y_phys < 0.08
    lo = (y_phys >= 0.08) & (y_phys < 0.6)
    mi = (y_phys >= 0.6) & (y_phys < 1.2)
    hi = y_phys >= 1.2
    colors[fl] = [0.45, 0.42, 0.38, 0.35]
    colors[lo] = [0.55, 0.50, 0.42, 0.50]
    colors[mi] = [0.65, 0.55, 0.45, 0.50]
    colors[hi] = [0.60, 0.50, 0.40, 0.40]

    return np.stack([x_phys, z_phys, y_phys], axis=-1), colors


# ── ZeroTextEncoder ──────────────────────────────────────────────────────────
class ZeroTextEncoder:
    output_dim = 4096; llm_dim = 4096; max_len = 77
    def __call__(self, texts, device=None):
        B = len(texts)
        tf = torch.zeros(B, 1, self.output_dim); tl = torch.ones(B, dtype=torch.long)
        if device: tf, tl = tf.to(device), tl.to(device)
        return tf, tl
    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self

# ── Data loading ─────────────────────────────────────────────────────────────
def load_samples(cache_indices):
    """Load GT motion features, joints, text from lingo_smplx_cache."""
    cache_dir = PROJECT_ROOT / "lingo_smplx_cache"
    joints_file = PROJECT_ROOT / "LINGO/dataset/dataset/human_joints_aligned.npy"
    start_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/end_idx.npy")).flatten()

    # Build index: cache_idx -> global frame range
    count = 0
    seg_ranges = {}
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
            print(f"  SKIP seg_{ci:05d}: file not found")
            continue
        data = np.load(str(cache_file), allow_pickle=True)
        T = int(data["length"])
        s, e = seg_ranges.get(ci, (0, T))
        samples.append({
            "cache_idx": ci,
            "text": str(data.get("text", "no-text")),
            "num_frames": T,
            "motion_features": data["motion_features"][:T].copy(),
            "scene_name": str(data.get("scene_name", "")),
            "gt_joints": joints_all[s:e, :22, :].copy(),
        })
    return samples


# ── Generation with Kimodo two-stage ─────────────────────────────────────────
def generate_with_root(model, motion_rep, sample, device, num_steps=50, use_init_pose=False):
    T = sample["num_frames"]
    features = sample["motion_features"]
    feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
    norm_feat = motion_rep.normalize(feat_t)

    D = norm_feat.shape[-1]
    root_slice = motion_rep.root_slice
    body_slice = motion_rep.body_slice

    observed_motion = torch.zeros(1, T, D, device=device)
    motion_mask = torch.zeros(1, T, D, device=device)
    observed_motion[..., root_slice] = norm_feat[..., root_slice]
    motion_mask[..., root_slice] = 1.0

    if use_init_pose:
        observed_motion[:, 0:1, body_slice] = norm_feat[:, 0:1, body_slice]
        motion_mask[:, 0:1, body_slice] = 1.0

    text_feat, text_length = model.text_encoder([sample["text"]], device=device)
    B, maxlen = text_feat.shape[:2]
    text_len_t = torch.tensor(text_length, device=device)
    text_pad_mask = torch.arange(maxlen, device=device).expand(B, maxlen) < text_len_t[:, None]

    pad_mask = torch.ones(1, T, dtype=torch.bool, device=device)
    fha = torch.zeros(1, device=device)

    with torch.no_grad():
        motion_feat = model._generate(
            texts=[sample["text"]], max_frames=T,
            num_denoising_steps=num_steps,
            pad_mask=pad_mask, first_heading_angle=fha,
            motion_mask=motion_mask, observed_motion=observed_motion,
            cfg_weight=[2.0, 2.0],
            text_feat=text_feat, text_pad_mask=text_pad_mask,
        )
        output = motion_rep.inverse(motion_feat, is_normalized=True, return_numpy=True)

    # Stitch GT root onto output
    # Unnormalize GT smooth root — unnormalize returns [B, T, D_full]
    gt_unnorm = motion_rep.unnormalize(feat_t)  # [B=1, T, D]
    s_root_un, *_ = einops.unpack(gt_unnorm, motion_rep.ps, "b t *")
    gt_root_np = s_root_un[0].cpu().numpy()  # [T, 3]

    gen_smooth = np.squeeze(output["smooth_root_pos"])
    gen_joints = np.squeeze(output["posed_joints"])
    delta = gt_root_np - gen_smooth
    gen_joints += delta[:, None, :]

    return gen_joints, gt_root_np


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(gt_joints, gen_joints):
    T = min(gt_joints.shape[0], gen_joints.shape[0])
    gt = gt_joints[:T]
    gen = gen_joints[:T]
    mpjpe = np.linalg.norm(gt - gen, axis=2).mean()
    parents = SMPLXSkeleton22().joint_parents
    valid = [j for j in range(22) if int(parents[j]) >= 0]
    pidx = [int(parents[j]) for j in valid]
    gt_d = gt[:, valid] - gt[:, pidx]
    gt_d = gt_d / (np.linalg.norm(gt_d, axis=2, keepdims=True) + 1e-8)
    gen_d = gen[:, valid] - gen[:, pidx]
    gen_d = gen_d / (np.linalg.norm(gen_d, axis=2, keepdims=True) + 1e-8)
    dot = np.clip((gt_d * gen_d).sum(axis=2), -1, 1)
    angle_err = np.degrees(np.arccos(dot)).mean()
    return mpjpe, angle_err


# ── Rendering ────────────────────────────────────────────────────────────────
def render_comparison(gt_joints, gen_joints, gt_root, gen_root, scene_pts, scene_colors,
                      text, out_path):
    T = min(gt_joints.shape[0], gen_joints.shape[0])

    # Y-align to floor
    ft = min(gt_joints[..., 1].min(), gen_joints[..., 1].min())
    gt_joints[..., 1] -= ft
    gen_joints[..., 1] -= ft
    gt_root[..., 1] -= ft
    gen_root[..., 1] -= ft

    # Viewport
    all_x = np.concatenate([gt_joints[...,0].ravel(), gen_joints[...,0].ravel()])
    all_z = np.concatenate([gt_joints[...,2].ravel(), gen_joints[...,2].ravel()])
    all_y = np.concatenate([gt_joints[...,1].ravel(), gen_joints[...,1].ravel()])
    if scene_pts is not None:
        all_x = np.concatenate([all_x, scene_pts[:,0]])
        all_z = np.concatenate([all_z, scene_pts[:,1]])
        all_y = np.concatenate([all_y, scene_pts[:,2]])

    margin = 0.5
    x_range = [all_x.min()-margin, all_x.max()+margin]
    z_range = [all_z.min()-margin, all_z.max()+margin]
    yb = 0
    yt = all_y.max() + 0.5

    pt_size = 3

    fig = plt.figure(figsize=(16, 7), dpi=120, facecolor="white")

    def setup_ax(ax, title):
        ax.set_facecolor("white")
        ax.grid(False)
        for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
            a.pane.set_visible(False); a.line.set_visible(False)
        ax.set_xlim(x_range); ax.set_ylim(z_range); ax.set_zlim(yb, yt)
        ax.view_init(elev=62, azim=-45)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold", y=1.0)

    def draw_bones(ax, joints):
        jf = joints
        for c in SMPLX_CONNECTIONS:
            grp = CONN_TO_GROUP[c]
            ax.plot([jf[c[0],0], jf[c[1],0]], [jf[c[0],2], jf[c[1],2]], [jf[c[0],1], jf[c[1],1]],
                    color=BONE_COLORS[grp], linewidth=3, alpha=0.9)
        ax.scatter(jf[:,0], jf[:,2], jf[:,1], c="white", s=12, alpha=0.9,
                   edgecolors="black", linewidth=0.5)

    def draw_root_trail(ax, root, frame, color="#00BCD4"):
        if frame > 0:
            ax.plot(root[:frame+1,0], root[:frame+1,2], root[:frame+1,1],
                    color=color, linewidth=3, alpha=0.7)
        ax.scatter(root[frame,0], root[frame,2], root[frame,1],
                   c=color, s=80, marker="o", edgecolors="white", linewidth=1, zorder=10)

    def draw_frame(f):
        plt.clf()
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d", facecolor="white")
        ax_gen = fig.add_subplot(1, 2, 2, projection="3d", facecolor="white")

        for ax, title, joints, root in [
            (ax_gt, "GT (LINGO)", gt_joints[min(f,T-1)], gt_root),
            (ax_gen, f"Generated: \"{text[:30]}\"", gen_joints[min(f,T-1)], gen_root),
        ]:
            setup_ax(ax, title)

            if scene_colors is not None:
                ax.scatter(scene_pts[:,0], scene_pts[:,1], scene_pts[:,2],
                           c=scene_colors, s=pt_size, rasterized=True, depthshade=False)

            draw_bones(ax, joints)
            draw_root_trail(ax, root, f)

    ani = animation.FuncAnimation(fig, draw_frame, frames=T, interval=40)
    writer = animation.FFMpegWriter(fps=25, bitrate=2000)
    ani.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_idx", type=int, nargs="+", default=[0, 2, 5, 8, 11])
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/exp_twostage_cmp")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--init_pose", action="store_true", default=False,
                        help="Constrain frame-0 body pose to GT")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading Kimodo model ...")
    model = load_model("Kimodo-SMPLX-RP-v1", device="cpu", text_encoder=ZeroTextEncoder())
    model = model.to(device).eval()

    skel = SMPLXSkeleton22()
    motion_rep = KimodoMotionRep(
        fps=30, stats_path="models/Kimodo-SMPLX-RP-v1/stats/motion", skeleton=skel)

    # Load samples
    samples = load_samples(args.cache_idx)
    print(f"Loaded {len(samples)} samples")

    for si, sample in enumerate(samples):
        ci = sample["cache_idx"]
        text = sample["text"]
        mode_str = "initpose" if args.init_pose else "root"
        print(f"\n[{si+1}/{len(samples)}] seg_{ci:05d} '{text}' T={sample['num_frames']} mode={mode_str}")

        out_path = out_dir / f"cmp_{ci:05d}_{mode_str}.mp4"
        if out_path.exists():
            print("  Already exists, skip")
            continue

        gen_joints, gt_root = generate_with_root(
            model, motion_rep, sample, device, args.num_denoising_steps,
            use_init_pose=args.init_pose)

        gt_joints = sample["gt_joints"]
        mpjpe, angle_err = compute_metrics(gt_joints, gen_joints)
        print(f"  MPJPE={mpjpe*100:.1f}cm  BoneAngle={angle_err:.1f}deg")

        # Load scene
        scene_pts, scene_colors = load_scene_lingo(sample["scene_name"])

        render_comparison(
            gt_joints, gen_joints, gt_root, gt_root,
            scene_pts, scene_colors, text, out_path)

    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
