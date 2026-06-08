"""Visualize experiment D (TrajCo only) — top-down scene + skeleton.

Usage:
    CUDA_VISIBLE_DEVICES=0 python kimodo_scene_project/scripts/viz_D_topdown.py \
      --num_samples 15 --gpu 0
"""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "kimodo"))

import os
os.environ["CHECKPOINT_DIR"] = str(Path(__file__).resolve().parent.parent.parent / "kimodo_scene_project/models")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

# ── SMPLX 22-joint skeleton ──────────────────────────────────────────────────
# Joint ordering: 0:pelvis 1:left_hip 2:right_hip 3:spine1 4:left_knee
# 5:right_knee 6:spine2 7:left_ankle 8:right_ankle 9:spine3 10:left_foot
# 11:right_foot 12:neck 13:left_collar 14:right_collar 15:head
# 16:left_shoulder 17:right_shoulder 18:left_elbow 19:right_elbow
# 20:left_wrist 21:right_wrist
SMPLX_CONNECTIONS = [
    (0,1), (1,4), (4,7), (7,10),       # left leg
    (0,2), (2,5), (5,8), (8,11),       # right leg
    (0,3), (3,6), (6,9),                # spine
    (9,12), (12,15),                     # neck -> head
    (9,13), (13,16), (16,18), (18,20),  # left arm
    (9,14), (14,17), (17,19), (19,21),  # right arm
]
BONE_COLORS = {"spine": "#34495e","r_arm": "#e74c3c","l_arm": "#3498db",
               "r_leg": "#e67e22","l_leg": "#2ecc71"}
CONN_TO_GROUP = {
    (0,1):"l_leg", (1,4):"l_leg", (4,7):"l_leg", (7,10):"l_leg",
    (0,2):"r_leg", (2,5):"r_leg", (5,8):"r_leg", (8,11):"r_leg",
    (0,3):"spine", (3,6):"spine", (6,9):"spine",
    (9,12):"spine", (12,15):"spine",
    (9,13):"l_arm", (13,16):"l_arm", (16,18):"l_arm", (18,20):"l_arm",
    (9,14):"r_arm", (14,17):"r_arm", (17,19):"r_arm", (19,21):"r_arm",
}

class ZeroTextEncoder:
    llm_dim=4096; max_len=77; output_dim=4096
    def __call__(self, texts, device=None):
        B=len(texts); f=torch.zeros(B,1,4096); l=torch.ones(B,dtype=torch.long)
        if device: f,l=f.to(device),l.to(device)
        return f,l
    def to(self,d): return self
    def train(self,m=True): return self
    def eval(self): return self


def load_model_D(device):
    from kimodo.model.load_model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo
    pretrained = load_model("Kimodo-SMPLX-RP-v1", device="cpu", text_encoder=ZeroTextEncoder())
    inner = pretrained.denoiser
    if hasattr(inner, "model"): inner = inner.model
    model = KimodoSceneCo(
        denoiser=inner, text_encoder=pretrained.text_encoder, num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={"voxel_size":(64,64,64),"patch_size":(8,8,8),"d_model":256,
                              "num_layers":4,"use_dual_vit":False,"root_voxel_mode":"full"},
        device=device, cfg_type="scene_separated",
        use_in_root_model=False, use_in_body_model=False,
        use_trajco=True, use_trajco_root=False, use_trajco_body=False,
        traj_dim=5, trajco_type="cross_attn",
    ).to(device).eval()
    ckpt_path = "kimodo_scene_project/outputs/trajco_cross_smplx/checkpoints/best_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    return model


LINGO_SCENE_DIR = Path("LINGO/dataset/dataset/Scene")

# Furniture color palette (per height band)
COLORS_FLOOR  = [0.45, 0.42, 0.38, 0.35]
COLORS_LOW    = [0.55, 0.50, 0.42, 0.50]
COLORS_MID    = [0.65, 0.55, 0.45, 0.50]
COLORS_HIGH   = [0.60, 0.50, 0.40, 0.40]


def load_samples(num, seed=42):
    cache_dir = Path("lingo_smplx_cache")
    files = sorted(cache_dir.glob("seg_*.npz"))
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(files), size=min(num, len(files)), replace=False)
    samples = []
    for idx in idxs:
        data = np.load(str(files[idx]), allow_pickle=True)
        T = int(data["length"])
        samples.append({
            "text": str(data.get("text","no-text")),
            "num_frames": T,
            "voxel": data["voxel_grid"],
            "motion_features": data["motion_features"][:T],
            "scene_name": str(data.get("scene_name", "")),
        })
    return samples


def generate(model, sample, device, num_steps=50):
    voxel = torch.from_numpy(sample["voxel"]).float().to(device)
    if voxel.ndim == 4: voxel = voxel.unsqueeze(1)
    elif voxel.ndim == 3: voxel = voxel.unsqueeze(0).unsqueeze(1)

    traj = torch.from_numpy(sample["motion_features"][:, :5]).float().to(device).unsqueeze(0)

    with torch.no_grad():
        out = model(
            prompts=sample["text"], num_frames=sample["num_frames"],
            num_denoising_steps=num_steps,
            cfg_weight=[3.0, 1.5, 2.0], cfg_type="scene_separated",
            scene_input=voxel, traj_input=traj, return_numpy=True,
        )
    return out


def extract_scene_lingo(scene_name, max_height=2.5, max_pts=20000, voxel_size=0.02):
    """Load original LINGO hi-res scene (300x100x400) and extract point cloud.

    LINGO scene:
      Array shape (300, 100, 400) = (Z_depth, Y_height, X_width)
      Physical size: X=8m, Y=2m, Z=6m
      Voxel size: 0.02m
      Floor is at Y index = 0, i.e. physical Y=0m

    Coordinate conversion:
      x_phys = (ox - W/2) * voxel_size   -> center at 0
      y_phys = oy * voxel_size            -> floor at 0
      z_phys = (oz - D/2) * voxel_size    -> center at 0

    Motion data from Kimodo uses (X, Y, Z) with Y=up.
    Matplotlib 3D expects (x, y, z). We plot as:
      scatter(scene_x, scene_z, scene_y)  so Z becomes matplotlib-y (horizontal),
      Y becomes matplotlib-z (vertical).

    Returns points in (X, Z_world, Y_up) order for use with scatter(x, z, y).
    """
    scene_path = LINGO_SCENE_DIR / f"{scene_name}.npy"
    if not scene_path.exists():
        print(f"  WARNING: scene {scene_name}.npy not found")
        return None, None

    v = np.load(str(scene_path))
    # v shape: (Z=300, Y=100, X=400), dtype=bool
    D, H, W = v.shape

    # Remove walls: only keep Y < max_height
    h_cells = min(int(max_height / voxel_size), H)
    v[:, h_cells:, :] = 0

    oz, oy, ox = np.where(v > 0)
    if len(oz) == 0:
        return None, None

    # Physical coordinates
    x_phys = (ox.astype(np.float32) - W / 2) * voxel_size
    y_phys = oy.astype(np.float32) * voxel_size
    z_phys = (oz.astype(np.float32) - D / 2) * voxel_size

    # Downsample to target count
    n = len(x_phys)
    if n > max_pts:
        # Systematic sampling instead of random to preserve structure
        step = max(1, n // max_pts)
        idx = np.arange(0, n, step)
        x_phys, y_phys, z_phys = x_phys[idx], y_phys[idx], z_phys[idx]

    # Color by height
    colors = np.zeros((len(x_phys), 4), dtype=np.float32)
    floor_mask = y_phys < 0.08
    low = (y_phys >= 0.08) & (y_phys < 0.6)
    mid = (y_phys >= 0.6) & (y_phys < 1.2)
    high = y_phys >= 1.2
    colors[floor_mask] = COLORS_FLOOR
    colors[low] = COLORS_LOW
    colors[mid] = COLORS_MID
    colors[high] = COLORS_HIGH

    # Return as (X, Z_world, Y_up) for scatter(x, z, y)
    pts = np.stack([x_phys, z_phys, y_phys], axis=-1)
    return pts, colors


def safe_filename(text, max_len=30):
    s = text.strip().lower().replace(" ", "_").replace("/", "_")
    s = "".join(c for c in s if c.isalnum() or c == "_")
    return s[:max_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="kimodo_scene_project/outputs/viz_D_topdown")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model D ...")
    model = load_model_D(device)

    samples = load_samples(args.num_samples, seed=42)
    print(f"Samples: {len(samples)}")

    for si, sample in enumerate(samples):
        text = sample["text"]
        action_name = safe_filename(text)
        print(f"\n[{si+1}/{len(samples)}] '{text}' ({sample['num_frames']}f)")
        out_path = out_dir / f"D_{si:02d}_{action_name}.mp4"
        if out_path.exists():
            print("  Already exists, skip")
            continue

        gen = generate(model, sample, device, args.num_denoising_steps)
        joints = np.squeeze(gen["posed_joints"]).copy()  # [T, 22, 3]
        root_trail = np.squeeze(gen.get("root_positions", gen.get("smooth_root_pos")))
        if root_trail.ndim == 2:
            root_trail = root_trail[:, :3].copy()
        else:
            root_trail = root_trail[:, 0, :3].copy()

        T = min(sample["num_frames"], joints.shape[0], root_trail.shape[0])

        # Align motion to floor: shift so foot Y matches scene floor (Y=0)
        foot_y = joints[..., 1].min()
        joints[..., 1] -= foot_y
        root_trail[..., 1] -= foot_y

        # Load scene from original LINGO hi-res data
        scene_pts, scene_colors = extract_scene_lingo(sample["scene_name"])

        # Viewport: center on skeleton motion range
        jx, jz, jy = joints[...,0].ravel(), joints[...,2].ravel(), joints[...,1].ravel()
        xm, zm = (jx.min()+jx.max())/2, (jz.min()+jz.max())/2
        rng = max(jx.max()-jx.min(), jz.max()-jz.min(), 3.0) * 1.3
        xl, xr = xm - rng/2, xm + rng/2
        zl, zr = zm - rng/2, zm + rng/2
        yb, yt = 0, jy.max() + 0.8  # Y starts at floor=0

        pt_size = 3

        fig = plt.figure(figsize=(10, 8), dpi=120, facecolor="white")
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        ax.grid(False); ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.view_init(elev=62, azim=-45)

        def draw_frame(f):
            ax.cla()
            ax.set_facecolor("white")
            ax.grid(False)
            for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
                a.pane.set_visible(False); a.line.set_visible(False)
            ax.set_xlim(xl, xr); ax.set_ylim(zl, zr); ax.set_zlim(yb, yt)
            ax.view_init(elev=62, azim=-45)
            ax.set_axis_off()

            # Scene point cloud: scatter(x, z, y) — X horizontal, Z depth, Y up
            if scene_colors is not None:
                ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                           c=scene_colors, s=pt_size, rasterized=True, depthshade=False)

            # Skeleton
            jf = joints[f]
            for c in SMPLX_CONNECTIONS:
                grp = CONN_TO_GROUP[c]
                ax.plot([jf[c[0],0], jf[c[1],0]], [jf[c[0],2], jf[c[1],2]], [jf[c[0],1], jf[c[1],1]],
                        color=BONE_COLORS[grp], linewidth=3, alpha=0.9)

            ax.scatter(jf[:, 0], jf[:, 2], jf[:, 1],
                       c="white", s=14, alpha=0.9, edgecolors="black", linewidth=0.5)

            # Root trail
            if f > 0:
                ax.plot(root_trail[:f+1, 0], root_trail[:f+1, 2], root_trail[:f+1, 1],
                        color="#e67e22", linewidth=4, alpha=0.95)

            ax.scatter(root_trail[f, 0], root_trail[f, 2], root_trail[f, 1],
                       c="#e74c3c", s=130, marker="o", edgecolors="white", linewidth=1.5, zorder=10)

            if f > 0:
                ax.scatter(root_trail[0, 0], root_trail[0, 2], root_trail[0, 1],
                           c="#2ecc71", s=180, marker="*", edgecolors="white", linewidth=1.5, zorder=11)

        ani = animation.FuncAnimation(fig, draw_frame, frames=T, interval=40)
        writer = animation.FFMpegWriter(fps=25, bitrate=2000)
        ani.save(str(out_path), writer=writer, dpi=120)
        plt.close(fig)
        print(f"  Saved {out_path.name}")

    print(f"\nDone — {len(list(out_dir.glob('*.mp4')))} videos in {out_dir}/")


if __name__ == "__main__":
    main()
