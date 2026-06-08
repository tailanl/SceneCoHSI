"""Visualize text-only generated motion (no scene, no trajectory) as MP4 videos."""
from __future__ import annotations
import sys, os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kimodo"))
os.environ["CHECKPOINT_DIR"] = os.path.join(PROJECT_ROOT, "kimodo_scene_project/models")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from kimodo.model.load_model import load_model
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

# SMPLX skeleton connections
SMPLX_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

BONE_GROUPS = {
    "spine":     [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)],
    "left_leg":  [(0, 1), (1, 4), (4, 7), (7, 10)],
    "right_leg": [(0, 2), (2, 5), (5, 8), (8, 11)],
    "left_arm":  [(9, 13), (13, 16), (16, 18), (18, 20)],
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

# Experiment F config
cfg = dict(
    ckpt=os.path.join(PROJECT_ROOT, "kimodo_scene_project/outputs/trajco_cross_root_sceneco_body/checkpoints/best_checkpoint.pt"),
    has_trajco=True, trajco_type="cross_attn",
    use_in_root=False, use_in_body=True,
    use_trajco_root=True, use_trajco_body=False,
)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading pretrained base model...")
pretrained = load_model("Kimodo-SMPLX-RP-v1", device=str(device))

print("Building KimodoSceneCo model...")
model = KimodoSceneCo(
    denoiser=pretrained.denoiser.model, text_encoder=pretrained.text_encoder,
    num_base_steps=1000, scene_encoder_type="voxel_vit",
    scene_encoder_config={"voxel_size":(64,64,64),"patch_size":(8,8,8),"d_model":256,"num_layers":4,"use_dual_vit":False,"root_voxel_mode":"full"},
    device=device, cfg_type="scene_separated",
    use_in_root_model=cfg["use_in_root"],
    use_in_body_model=cfg["use_in_body"],
    use_trajco=True, use_trajco_root=cfg["use_trajco_root"], use_trajco_body=cfg["use_trajco_body"],
    traj_dim=5, trajco_type=cfg["trajco_type"],
).to(device).eval()

print("Loading checkpoint...")
ckpt = torch.load(cfg["ckpt"], map_location=device)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
print("Checkpoint loaded.")

# Test prompts
test_prompts = [
    "A person stands still",
    "A person walks forward",
    "A person sits down on a chair",
]

num_frames = 60
out_dir = Path("kimodo_scene_project/outputs/viz_text_only")
out_dir.mkdir(parents=True, exist_ok=True)

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*60}")

    with torch.no_grad():
        gen = model(
            prompts=prompt, num_frames=num_frames,
            num_denoising_steps=50, cfg_weight=[3.0, 1.5, 2.0], cfg_type="scene_separated",
            scene_input=None,
            traj_input=None,
            text_feat=None,
            return_numpy=True,
        )

    joints = np.squeeze(gen["posed_joints"]).copy()  # (T, 22, 3)
    root_trail = np.squeeze(gen.get("root_positions", gen.get("smooth_root_pos")))  # (T, 3)
    
    if joints.ndim == 2:
        joints = joints.reshape(-1, 22, 3)

    # Y-alignment: FK joints have Y as local (relative to pelvis), add root Y
    if root_trail.ndim == 2 and root_trail.shape[1] > 1:
        root_y = root_trail[:, 1]
    elif root_trail.ndim == 1:
        root_y = root_trail
    else:
        root_y = np.zeros(joints.shape[0])
    
    joints[..., 1] += root_y[:, None]

    # Shift so feet are near Y=0
    foot_y = joints[..., 1].min()
    y_shift = 0.0 - foot_y
    joints[..., 1] += y_shift
    root_trail[..., 1] += y_shift

    T = min(num_frames, joints.shape[0])

    # Viewport
    joints_x = joints[..., 0].ravel()
    joints_z = joints[..., 2].ravel()
    joints_y = joints[..., 1].ravel()

    all_x_min, all_x_max = joints_x.min() - 1.5, joints_x.max() + 1.5
    all_z_min, all_z_max = joints_z.min() - 1.5, joints_z.max() + 1.5
    all_y_min, all_y_max = joints_y.min() - 0.3, joints_y.max() + 0.5

    ground_range = max(all_x_max - all_x_min, all_z_max - all_z_min)
    x_mid = (all_x_min + all_x_max) / 2
    z_mid = (all_z_min + all_z_max) / 2
    all_x_min = x_mid - ground_range / 2
    all_x_max = x_mid + ground_range / 2
    all_z_min = z_mid - ground_range / 2
    all_z_max = z_mid + ground_range / 2

    fig = plt.figure(figsize=(10, 8), dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    safe_fname = prompt.replace(" ", "_")
    out_path = out_dir / f"{safe_fname}.mp4"

    def draw_frame(frame):
        ax.cla()
        ax.set_facecolor("white")
        ax.grid(False)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.xaxis.line.set_visible(False)
        ax.yaxis.line.set_visible(False)
        ax.zaxis.line.set_visible(False)
        ax.set_xlim(all_x_min, all_x_max)
        ax.set_ylim(all_z_min, all_z_max)
        ax.set_zlim(all_y_min, all_y_max)
        ax.view_init(elev=20, azim=-45)
        ax.tick_params(axis='x', which='both', colors='white', labelcolor='white')
        ax.tick_params(axis='y', which='both', colors='white', labelcolor='white')
        ax.tick_params(axis='z', which='both', colors='white', labelcolor='white')

        jf = joints[frame]
        for c in SMPLX_CONNECTIONS:
            grp = CONN_TO_GROUP[c]
            ax.plot([jf[c[0], 0], jf[c[1], 0]],
                    [jf[c[0], 2], jf[c[1], 2]],
                    [jf[c[0], 1], jf[c[1], 1]],
                    color=BONE_COLORS[grp], linewidth=3, alpha=0.9)

        ax.scatter(jf[:, 0], jf[:, 2], jf[:, 1],
                   c="black", s=15, alpha=0.7, edgecolors="white", linewidth=0.5)

        # Root trail
        trail_end = min(frame + 1, len(root_trail))
        if trail_end > 1:
            ax.plot(root_trail[:trail_end, 0],
                    root_trail[:trail_end, 2],
                    root_trail[:trail_end, 1],
                    color="#e67e22", linewidth=3, alpha=0.8)

        ax.scatter(root_trail[frame, 0], root_trail[frame, 2], root_trail[frame, 1],
                   c="#e74c3c", s=100, marker="o", edgecolors="white", linewidth=1.5, zorder=10)

        if frame > 0:
            ax.scatter(root_trail[0, 0], root_trail[0, 2], root_trail[0, 1],
                       c="#2ecc71", s=150, marker="*", edgecolors="white", linewidth=1.5, zorder=11)

        ax.set_title(f"TEXT-ONLY: {prompt} | frame {frame}/{T-1}",
                     fontsize=10, color="black", fontweight="bold")

    def update(f):
        draw_frame(f)
        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=False)
    writer = animation.FFMpegWriter(fps=25, bitrate=2000)
    ani.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    print(f"  Saved {out_path}")

print(f"\nDone → {out_dir}/")
