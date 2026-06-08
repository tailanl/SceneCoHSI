"""Debug two-stage generation: compare stage1 root vs GT root, and full two-stage output."""
import argparse, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

import os
os.environ["CHECKPOINT_DIR"] = str(PROJECT_ROOT / "kimodo_scene_project/models")

import matplotlib; matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import einops

from kimodo.model.load_model import load_model
from kimodo.skeleton import SMPLXSkeleton22
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep

PROJ = Path(__file__).resolve().parent.parent


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


def load_samples(cache_indices):
    cache_dir = PROJ / "lingo_smplx_cache"
    joints_file = PROJ / "LINGO/dataset/dataset/human_joints_aligned.npy"
    start_idx = np.load(str(PROJ / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJ / "LINGO/dataset/dataset/end_idx.npy")).flatten()

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
            "gt_joints": joints_all[s:e, :22, :].copy(),
        })
    return samples


def generate_two_stage(model, motion_rep, sample, device, num_steps=50):
    T = sample["num_frames"]
    features = sample["motion_features"]
    feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)

    # Use GT's first heading angle
    gt_unnorm = motion_rep.unnormalize(feat_t)
    s_root_un, g_heading, *_ = einops.unpack(gt_unnorm, motion_rep.ps, "b t *")
    gt_root_np = s_root_un[0].cpu().numpy()
    gh_first = g_heading[0, 0]
    angle = torch.atan2(gh_first[1], gh_first[0])
    fha = angle.unsqueeze(0).to(device)

    text_feat, text_length = model.text_encoder([sample["text"]], device=device)
    B, maxlen = text_feat.shape[:2]
    text_len_t = text_length.clone()
    text_pad_mask = torch.arange(maxlen, device=device).expand(B, maxlen) < text_len_t[:, None]

    pad_mask = torch.ones(1, T, dtype=torch.bool, device=device)

    D = feat_t.shape[-1]
    root_slice = motion_rep.root_slice

    # No root conditioning — just text + heading, then align output to GT root
    observed_motion = torch.zeros(1, T, D, device=device)
    motion_mask = torch.zeros(1, T, D, device=device)

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

    gen_joints = np.squeeze(output["posed_joints"])
    gen_root = np.squeeze(output["smooth_root_pos"])

    # Debug: check for NaN
    if np.any(np.isnan(gen_joints)):
        print("  WARNING: gen_joints has NaN, replacing with 0")
        gen_joints = np.nan_to_num(gen_joints, nan=0.0)
    if np.any(np.isnan(gen_root)):
        print("  WARNING: gen_root has NaN, replacing with 0")
        gen_root = np.nan_to_num(gen_root, nan=0.0)

    # Decode GT
    output_gt = motion_rep.inverse(
        motion_rep.unnormalize(feat_t),
        is_normalized=False, return_numpy=True)
    gt_joints = output_gt["posed_joints"][0]

    # Post-process: replace gen root with GT root trajectory, and force frame-0 body to GT
    # (CFG undermines motion_mask conditioning, so we enforce at output level)
    # Shift generated joints so their root matches GT root
    delta = gt_root_np - gen_root
    gen_joints += delta[:, None, :]

    # Force frame-0 body joints to match GT exactly
    gen_joints[0] = gt_joints[0].copy()

    return gen_joints, gen_root, gt_joints, gt_root_np


def render_comparison(gt_joints, gen_joints, gt_root, gen_root, text, out_path):
    T = min(gt_joints.shape[0], gen_joints.shape[0])

    # Y-align
    ft = min(gt_joints[..., 1].min(), gen_joints[..., 1].min())
    gt_joints = gt_joints.copy()
    gen_joints = gen_joints.copy()
    gt_joints[..., 1] -= ft
    gen_joints[..., 1] -= ft

    # Viewport
    all_xyz = np.concatenate([gt_joints.reshape(-1, 3), gen_joints.reshape(-1, 3)])
    margin = 0.5
    x_range = [all_xyz[:, 0].min() - margin, all_xyz[:, 0].max() + margin]
    z_range = [all_xyz[:, 2].min() - margin, all_xyz[:, 2].max() + margin]
    yb = 0
    yt = all_xyz[:, 1].max() + 0.5

    fig = plt.figure(figsize=(16, 7), dpi=120, facecolor="#111111")

    def setup_ax(ax, title):
        ax.set_facecolor("#111111")
        ax.grid(False)
        for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
            a.pane.set_visible(False)
            a.line.set_visible(False)
        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_zlim(yb, yt)
        ax.view_init(elev=62, azim=-45)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", y=1.0)

    def draw_bones(ax, joints):
        jf = joints
        for c in SMPLX_CONNECTIONS:
            grp = CONN_TO_GROUP[c]
            ax.plot([jf[c[0], 0], jf[c[1], 0]], [jf[c[0], 2], jf[c[1], 2]],
                    [jf[c[0], 1], jf[c[1], 1]],
                    color=BONE_COLORS[grp], linewidth=3, alpha=0.9)
        ax.scatter(jf[:, 0], jf[:, 2], jf[:, 1], c="white", s=12, alpha=0.9, zorder=10)

    def draw_root_trail(ax, root, frame, color="#00BCD4"):
        if frame > 0:
            ax.plot(root[:frame + 1, 0], root[:frame + 1, 2], root[:frame + 1, 1],
                    color=color, linewidth=3, alpha=0.7)
        ax.scatter(root[frame, 0], root[frame, 2], root[frame, 1],
                   c=color, s=80, marker="o", edgecolors="white", linewidth=1, zorder=10)

    def draw_frame(f):
        plt.clf()
        fig.set_facecolor("#111111")
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#111111")
        ax_gen = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#111111")

        for ax, title, joints, root in [
            (ax_gt, "GT (LINGO)", gt_joints[min(f, T - 1)], gt_root),
            (ax_gen, f"Generated: \"{text[:30]}\"", gen_joints[min(f, T - 1)], gen_root),
        ]:
            setup_ax(ax, title)
            draw_bones(ax, joints)
            draw_root_trail(ax, root, f)

    ani = animation.FuncAnimation(fig, draw_frame, frames=T, interval=40)
    writer = animation.FFMpegWriter(fps=25, bitrate=2000)
    ani.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_idx", type=int, nargs="+", default=[0, 2, 5, 8, 11])
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="kimodo_scene_project/outputs/exp_twostage_cmp")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Kimodo model ...")
    model = load_model("Kimodo-SMPLX-RP-v1", device="cpu", text_encoder=ZeroTextEncoder())
    model = model.to(device).eval()

    print("Loading motion rep ...")
    skel = SMPLXSkeleton22()
    motion_rep = KimodoMotionRep(
        fps=30, stats_path="models/Kimodo-SMPLX-RP-v1/stats/motion", skeleton=skel)

    samples = load_samples(args.cache_idx)
    print(f"Loaded {len(samples)} samples")

    for si, sample in enumerate(samples):
        ci = sample["cache_idx"]
        text = sample["text"]
        print(f"\n[{si + 1}/{len(samples)}] seg_{ci:05d} '{text}' T={sample['num_frames']}")

        out_path = out_dir / f"debug_{ci:05d}.mp4"

        gen_joints, gen_root, gt_joints, gt_root_np = generate_two_stage(
            model, motion_rep, sample, device, args.num_denoising_steps)

        # Compute metrics (skip frame-0 since we forced it to match GT)
        T_cmp = min(gt_joints.shape[0], gen_joints.shape[0])
        diff = gt_joints[1:T_cmp] - gen_joints[1:T_cmp]
        mpjpe = np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))
        # Bone angle error (simplified)
        angle_err = np.mean(np.abs(
            np.arccos(np.clip(np.sum(
                (gt_joints[1:] - gt_joints[:-1]) * (gen_joints[1:] - gen_joints[:-1]), axis=-1
            ) / (
                np.linalg.norm(gt_joints[1:] - gt_joints[:-1], axis=-1) *
                np.linalg.norm(gen_joints[1:] - gen_joints[:-1], axis=-1) + 1e-8
            ), -1, 1))
        )) * 180 / np.pi
        print(f"  MPJPE={mpjpe * 100:.1f}cm  BoneAngle={angle_err:.1f}deg")

        # Print root comparison
        print(f"  GT root[0]: {gt_root_np[0]}")
        print(f"  Gen root[0]: {gen_root[0]}")
        print(f"  GT heading[0]: {np.degrees(np.arctan2(gt_root_np[1, 2] - gt_root_np[0, 2], gt_root_np[1, 0] - gt_root_np[0, 0])):.1f} deg")
        print(f"  Gen heading[0]: {np.degrees(np.arctan2(gen_root[1, 2] - gen_root[0, 2], gen_root[1, 0] - gen_root[0, 0])):.1f} deg")

        render_comparison(gt_joints, gen_joints, gt_root_np, gen_root, text, out_path)

    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
