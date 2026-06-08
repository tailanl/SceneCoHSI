"""Verify: load cached features, run through KimodoMotionRep.inverse(), visualize."""
from __future__ import annotations
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kimodo"))
os.environ["CHECKPOINT_DIR"] = os.path.join(PROJECT_ROOT, "kimodo_scene_project/models")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from pathlib import Path

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton import SMPLXSkeleton22

skel = SMPLXSkeleton22()
mr = KimodoMotionRep(fps=30, stats_path="models/Kimodo-SMPLX-RP-v1/stats/motion", skeleton=skel)

SMPLX_CONNECTIONS = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),
    (9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),(17,19),(18,20),(19,21),
]
BONE_COLORS = {
    (0,3):"#FF8C00",(3,6):"#FF8C00",(6,9):"#FF8C00",(9,12):"#FF8C00",(12,15):"#FF8C00",
    (0,1):"#4169E1",(1,4):"#4169E1",(4,7):"#4169E1",(7,10):"#4169E1",
    (0,2):"#DC143C",(2,5):"#DC143C",(5,8):"#DC143C",(8,11):"#DC143C",
    (9,13):"#32CD32",(13,16):"#32CD32",(16,18):"#32CD32",(18,20):"#32CD32",
    (9,14):"#FF69B4",(14,17):"#FF69B4",(17,19):"#FF69B4",(19,21):"#FF69B4",
}

out_dir = Path("kimodo_scene_project/outputs/viz_cache_inverse")
out_dir.mkdir(parents=True, exist_ok=True)

for seg_id, label in [(115, "walk_forward"), (174, "phone_call"), (89, "stand")]:
    data = np.load(f"lingo_smplx_cache/seg_{seg_id:05d}.npz", allow_pickle=True)
    T = int(data["length"])
    motion = data["motion_features"][:T]  # [T, 273]

    print(f"\nseg_{seg_id:05d} [{label}] T={T}")
    print(f"  features: [{motion[:,:5].min():.3f},{motion[:,:5].max():.3f}]")
    print(f"  rot_data: [{motion[:,71:203].min():.3f},{motion[:,71:203].max():.3f}]")

    feat_t = torch.from_numpy(motion).float().unsqueeze(0)
    out = mr.inverse(feat_t, is_normalized=True, return_numpy=True)
    joints = np.squeeze(out["posed_joints"]).copy()  # [T, 22, 3]
    root_trail = np.squeeze(out.get("root_positions", out.get("smooth_root_pos")))
    if root_trail.ndim != 2:
        root_trail = root_trail[:, 0, :3]

    # Y-alignment: shift so lowest foot is at Y=0
    # (no need to add smooth_root Y — inverse() FK already outputs correct absolute Y with v2 cache)
    foot_y = joints[..., 1].min()
    y_shift = 0.0 - foot_y
    joints[..., 1] += y_shift
    if root_trail.ndim >= 2:
        root_trail[..., 1] += y_shift

    pelvis_y = joints[:, 0, 1].mean()
    head_y = joints[:, 15, 1].mean()
    foot_y_mean = joints[:, [10, 11], 1].mean()
    print(f"  pelvis={pelvis_y:.3f} head={head_y:.3f} foot={foot_y_mean:.3f}")
    print(f"  head-pelvis={head_y-pelvis_y:.3f} pelvis-foot={pelvis_y-foot_y_mean:.3f}")

    # Viewport
    jx, jy, jz = joints[..., 0], joints[..., 1], joints[..., 2]
    ax_mid = (jx.max() - jx.min(), jz.max() - jz.min())
    gr = max(ax_mid) / 2 * 1.2
    xm, zm = (jx.min() + jx.max()) / 2, (jz.min() + jz.max()) / 2

    fig = plt.figure(figsize=(8, 8), dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    def draw_frame(f):
        ax.cla(); ax.set_facecolor("white")
        ax.grid(False)
        for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]: p.set_visible(False)
        for a in [ax.xaxis.line, ax.yaxis.line, ax.zaxis.line]: a.set_visible(False)
        ax.tick_params(colors='white', labelcolor='white')
        ax.set_xlim(xm - gr, xm + gr); ax.set_ylim(zm - gr, zm + gr)
        ax.set_zlim(jy.min() - 0.3, jy.max() + 0.5)
        ax.view_init(elev=20, azim=-45)

        jf = joints[f]
        for c in SMPLX_CONNECTIONS:
            col = BONE_COLORS.get(c, "gray")
            ax.plot([jf[c[0], 0], jf[c[1], 0]],
                    [jf[c[0], 2], jf[c[1], 2]],
                    [jf[c[0], 1], jf[c[1], 1]],
                    color=col, linewidth=3, alpha=0.9)
        ax.scatter(jf[:, 0], jf[:, 2], jf[:, 1], c="black", s=12, alpha=0.7)
        trail_e = min(f + 1, len(root_trail))
        if trail_e > 1:
            ax.plot(root_trail[:trail_e, 0], root_trail[:trail_e, 2], root_trail[:trail_e, 1],
                    color="#e67e22", linewidth=2, alpha=0.8)
        ax.set_title(f"GT cache → inverse: {label} | frame {f}/{T-1}", fontsize=10)

    ani = animation.FuncAnimation(fig, draw_frame, frames=T, interval=40)
    op = out_dir / f"{label}_{seg_id}.mp4"
    ani.save(str(op), writer=animation.FFMpegWriter(fps=25, bitrate=2000), dpi=100)
    plt.close(fig)
    print(f"  Saved {op}")

print(f"\nDone → {out_dir}/")
