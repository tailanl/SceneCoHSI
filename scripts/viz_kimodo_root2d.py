"""Compare Kimodo generation with Root2D constraint vs GT."""

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
from kimodo.constraints import Root2DConstraintSet, FullBodyConstraintSet


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
    cache_dir = PROJECT_ROOT / "lingo_smplx_cache"
    joints_file = PROJECT_ROOT / "LINGO/dataset/dataset/human_joints_aligned.npy"
    start_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/start_idx.npy")).flatten()
    end_idx = np.load(str(PROJECT_ROOT / "LINGO/dataset/dataset/end_idx.npy")).flatten()

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


# ── Generation with Kimodo + Root2D constraint ───────────────────────────────
def generate_with_root_constraint(model, motion_rep, skel, sample, device, num_steps=50):
    T = sample["num_frames"]
    features = sample["motion_features"]
    feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)

    # Decode GT to get smooth_root_2d and heading
    output_gt = motion_rep.inverse(
        motion_rep.unnormalize(feat_t),
        is_normalized=False, return_numpy=True)
    gt_joints = output_gt["posed_joints"][0]
    gt_smooth_root = output_gt["smooth_root_pos"][0]  # (T, 3)

    # Root2D constraint: (X, Z) trajectory for all frames
    smooth_root_2d = torch.from_numpy(gt_smooth_root[:, [0, 2]]).float().to(device)
    frame_indices = torch.arange(T, device=device, dtype=torch.long)

    # Also extract heading as [cos, sin] encoded
    gt_unnorm = motion_rep.unnormalize(feat_t)
    s_root_un, g_heading, *_ = einops.unpack(gt_unnorm, motion_rep.ps, "b t *")
    # g_heading: [B=1, T, 2] = [cos, sin]
    global_heading = g_heading[0].to(device)  # (T, 2)

    constraint = Root2DConstraintSet(
        skeleton=skel,
        frame_indices=frame_indices,
        smooth_root_2d=smooth_root_2d,
        global_root_heading=global_heading,
    )

    # FullBodyConstraintSet for frame-0 body pose (fix initial pose)
    f0 = torch.tensor([0], device=device, dtype=torch.long)
    f0_positions = torch.from_numpy(gt_joints[0:1]).float().to(device)  # (1, J, 3)
    # Identity rotations as quaternions (w=1, x=y=z=0) — not actually used by postprocess
    f0_rots = torch.zeros(1, 22, 4, device=device)
    f0_rots[:, :, 0] = 1.0

    fb_constraint = FullBodyConstraintSet(
        skeleton=skel,
        frame_indices=f0,
        global_joints_positions=f0_positions,
        global_joints_rots=f0_rots,
    )

    constraints = [constraint, fb_constraint]

    # Generate with constraints
    output = model(
        prompts=[sample["text"]],
        num_frames=T,
        num_denoising_steps=num_steps,
        constraint_lst=constraints,
        cfg_weight=[2.0, 2.0],
        return_numpy=True,
    )

    gen_joints = output["posed_joints"][0]
    gen_root = output["smooth_root_pos"]

    return gen_joints, gt_smooth_root, gt_joints


# ── Rendering ────────────────────────────────────────────────────────────────
def render_comparison(gt_joints, gen_joints, gt_root, text, out_path):
    T = min(gt_joints.shape[0], gen_joints.shape[0])

    # Y-align
    ft = min(gt_joints[..., 1].min(), gen_joints[..., 1].min())
    gt_joints = gt_joints.copy()
    gen_joints = gen_joints.copy()
    gt_joints[..., 1] -= ft
    gen_joints[..., 1] -= ft

    # Viewport
    all_xyz = np.concatenate([
        gt_joints.reshape(-1, 3), gen_joints.reshape(-1, 3)
    ])
    margin = 0.5
    x_range = [all_xyz[:,0].min()-margin, all_xyz[:,0].max()+margin]
    z_range = [all_xyz[:,2].min()-margin, all_xyz[:,2].max()+margin]
    yb = 0
    yt = all_xyz[:,1].max() + 0.5

    fig = plt.figure(figsize=(16, 7), dpi=120, facecolor="#111111")

    def setup_ax(ax, title):
        ax.set_facecolor("#111111")
        ax.grid(False)
        for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
            a.pane.set_visible(False); a.line.set_visible(False)
        ax.set_xlim(x_range); ax.set_ylim(z_range); ax.set_zlim(yb, yt)
        ax.view_init(elev=62, azim=-45)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", y=1.0)

    def draw_bones(ax, joints):
        jf = joints
        for c in SMPLX_CONNECTIONS:
            grp = CONN_TO_GROUP[c]
            ax.plot([jf[c[0],0], jf[c[1],0]], [jf[c[0],2], jf[c[1],2]], [jf[c[0],1], jf[c[1],1]],
                    color=BONE_COLORS[grp], linewidth=3, alpha=0.9)
        ax.scatter(jf[:,0], jf[:,2], jf[:,1], c="white", s=12, alpha=0.9, zorder=10)

    def draw_root_trail(ax, root, frame, color="#00BCD4"):
        if frame > 0:
            ax.plot(root[:frame+1,0], root[:frame+1,2], root[:frame+1,1],
                    color=color, linewidth=3, alpha=0.7)
        ax.scatter(root[frame,0], root[frame,2], root[frame,1],
                   c=color, s=80, marker="o", edgecolors="white", linewidth=1, zorder=10)

    def draw_frame(f):
        plt.clf()
        fig.set_facecolor("#111111")
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#111111")
        ax_gen = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#111111")

        for ax, title, joints, root in [
            (ax_gt, "GT (LINGO)", gt_joints[min(f,T-1)], gt_root),
            (ax_gen, f"Generated: \"{text[:30]}\"", gen_joints[min(f,T-1)], gt_root),
        ]:
            setup_ax(ax, title)
            draw_bones(ax, joints)
            draw_root_trail(ax, root, f)

    ani = animation.FuncAnimation(fig, draw_frame, frames=T, interval=40)
    writer = animation.FFMpegWriter(fps=25, bitrate=2000)
    ani.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────
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
        print(f"\n[{si+1}/{len(samples)}] seg_{ci:05d} '{text}' T={sample['num_frames']}")

        out_path = out_dir / f"cmp_root2d_{ci:05d}.mp4"

        gen_joints, gt_root, gt_joints = generate_with_root_constraint(
            model, motion_rep, skel, sample, device, args.num_denoising_steps)

        # Compute metrics
        T_cmp = min(gt_joints.shape[0], gen_joints.shape[0])
        diff = gt_joints[:T_cmp] - gen_joints[:T_cmp]
        mpjpe = np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))
        print(f"  MPJPE={mpjpe*100:.1f}cm")

        render_comparison(gt_joints, gen_joints, gt_root, text, out_path)

    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
