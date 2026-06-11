"""Generate comparison videos for experiment results.

For each experiment, renders skeleton animation with root trajectory + scene overlay.
Outputs MP4 videos to outputs/viz_videos/

Usage:
    python scripts/viz_experiment_videos.py --exp E5_v3
    python scripts/viz_experiment_videos.py --all
"""

import argparse, sys, os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt

SMPLX_22_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

EXPERIMENTS = {
    "E1": {
        "name": "EnergyGuidance + Original Body",
        "body_dir": "outputs/e1_energy_guidance_body",
        "config": """Root: energy loss guidance (hand-crafted path/scene losses)
Body: original Kimodo (no SceneCo training)
Root control: weak (PathADE=2.10)
Scene avoidance: poor (CFR=0.53)""",
    },
    "E2": {
        "name": "ClassifierGuidance + Original Body",
        "body_dir": "outputs/e2_classifier_guidance_body",
        "config": """Root: trained RootPathClassifier guidance (19-dim)
Classifier: 4-layer transformer, val_acc=1.0
Body: original Kimodo (no SceneCo training)
Root control: good (PathADE=1.07)
Scene avoidance: best among orig (CFR=0.20)""",
    },
    "E3": {
        "name": "HybridGuidance + Original Body",
        "body_dir": "outputs/e3_hybrid_guidance_body",
        "config": """Root: classifier + energy hybrid guidance
Body: original Kimodo (no SceneCo training)
Root control: moderate (PathADE=1.21)
Scene avoidance: moderate (CFR=0.28)""",
    },
    "E5_v3": {
        "name": "ClassifierGuidance + Stage2 SceneCo",
        "body_dir": "outputs/e5_v3_stage2/val_gen",
        "config": """Root: classifier-guided (same as E2)
Body: Stage2 SceneCo trained (80 epochs)
  - Frozen backbone, train scene_encoder + SceneCo body adapter (145M params)
  - body_mse loss only, root fixed from external
  - scene_dropout=0.1, batch_size=4
Root control: good (PathADE=1.37 on val set)
Scene avoidance: improved (CFR=0.14, -30% vs E2)""",
    },
    "E7_v3": {
        "name": "GT Root + Stage2 SceneCo",
        "body_dir": "outputs/e7_v3_stage2/val_gen",
        "config": """Root: GT root from dataset (upper bound)
Body: Stage2 SceneCo trained (80 epochs)
  - Same config as E5 but with GT root
  - root_mix_gt=0.0, root_mix_path=1.0
Root control: perfect (PathADE=0.0, GT root)
Scene avoidance: CFR=0.34 (scene density limits)""",
    },
}


def load_scene_sdf(scene_name, cache_dir="lingo_smplx_cache"):
    """Load 2D SDF from cached voxel grid."""
    from pathlib import Path
    cache_dir = Path(cache_dir)
    for f in sorted(cache_dir.glob("seg_*.npz"))[:5000]:
        try:
            d = np.load(str(f), allow_pickle=True)
            if str(d.get("scene_name", "")) == scene_name:
                voxel = d["voxel_grid"].astype(np.float32)
                occ_2d = voxel.mean(axis=1)
                binary = occ_2d > 0.5
                dist_out = distance_transform_edt(~binary).astype(np.float32)
                dist_in = distance_transform_edt(binary).astype(np.float32)
                return dist_out - dist_in
        except:
            continue
    return None


def render_video(body_files, out_path, title, num_samples=3, max_frames=196):
    """Render skeleton animation video from body NPZ files."""
    
    # Load samples
    samples = []
    for f in body_files[:num_samples]:
        try:
            d = np.load(str(f), allow_pickle=True)
            gen_root = np.asarray(d["gen_root"], dtype=np.float32)
            gen_joints = np.asarray(d["gen_joints"], dtype=np.float32)
            scene_name = str(d.get("scene_name", ""))
            if gen_joints.shape[1] >= 22:
                gen_joints = gen_joints[:, :22, :]
            samples.append({
                "root": gen_root[:max_frames],
                "joints": gen_joints[:max_frames],
                "scene": scene_name,
                "name": f.stem,
            })
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")

    if not samples:
        print("  No valid samples")
        return

    # Determine grid layout
    ncols = min(len(samples), 3)
    nrows = (len(samples) + ncols - 1) // ncols
    T = max(s["root"].shape[0] for s in samples)

    # Load scene SDF for first sample
    sdf = load_scene_sdf(samples[0]["scene"]) if samples[0]["scene"] else None

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False, subplot_kw={'projection': '3d'} if False else {})
    
    axes_flat = axes.flatten()
    for i in range(len(samples), len(axes_flat)):
        axes_flat[i].set_visible(False)

    connections = SMPLX_22_CONNECTIONS

    def draw_frame(t):
        for ax in axes_flat:
            ax.clear()
        
        for i, s in enumerate(samples):
            ax = axes_flat[i]
            Ti = s["root"].shape[0]
            ti = min(t, Ti - 1)
            
            # Draw skeleton
            joints = s["joints"][ti]
            root = s["root"][ti]
            
            for (j1, j2) in connections:
                if j1 < joints.shape[0] and j2 < joints.shape[0]:
                    ax.plot([joints[j1, 0], joints[j2, 0]],
                           [joints[j1, 2], joints[j2, 2]],
                           'b-', alpha=0.7, linewidth=1.5)
            
            # Root position
            ax.plot(root[0], root[2], 'ro', markersize=6)
            
            # Root trajectory trail
            trail_len = min(ti, 30)
            if trail_len > 1:
                ax.plot(s["root"][max(0,ti-trail_len):ti+1, 0],
                       s["root"][max(0,ti-trail_len):ti+1, 2],
                       'r-', alpha=0.3, linewidth=1)
            
            # Scene SDF contour
            if sdf is not None:
                try:
                    ax.contour(sdf.T, levels=[0], colors='gray', alpha=0.3, linewidths=0.5)
                except:
                    pass
            
            ax.set_title(f'{s["name"]}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_aspect('equal')
            
            # Auto-scale
            margin = 0.5
            ax.set_xlim(s["root"][:, 0].min() - margin, s["root"][:, 0].max() + margin)
            ax.set_ylim(s["root"][:, 2].min() - margin, s["root"][:, 2].max() + margin)

        fig.suptitle(f'{title}\nFrame {t}/{T-1}', fontsize=12)
        plt.tight_layout()
        return []

    ani = FuncAnimation(fig, draw_frame, frames=T, interval=50, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        writer = matplotlib.animation.FFMpegWriter(fps=20, bitrate=1500)
        ani.save(str(out_path), writer=writer, dpi=150)
        print(f"  Saved: {out_path}")
    except Exception as e:
        # fallback to GIF
        gif_path = out_path.with_suffix('.gif')
        ani.save(str(gif_path), writer='pillow', fps=10, dpi=100)
        print(f"  Saved (GIF): {gif_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs/viz_videos")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = args.exp or ["E5_v3", "E7_v3"]

    for eid in exp_ids:
        info = EXPERIMENTS.get(eid)
        if not info:
            print(f"Unknown: {eid}")
            continue
        
        body_dir = Path(info["body_dir"])
        body_files = sorted(body_dir.glob("sample_*.npz"))
        if not body_files:
            body_files = sorted(body_dir.glob("seg_*.npz"))
        if not body_files:
            body_files = sorted(body_dir.glob("*.npz"))
        
        print(f"\n{'='*60}")
        print(f"{eid}: {info['name']}")
        print(f"Samples: {len(body_files)}")
        print(f"Config:\n{info['config']}")
        print(f"{'='*60}")
        
        out_path = out_dir / f"{eid}_comparison.mp4"
        render_video(body_files, out_path, info['name'], num_samples=args.num_samples)


if __name__ == "__main__":
    main()
