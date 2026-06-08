#!/usr/bin/env python
"""Check LINGO dataset: use SOMA raw data for correct GT skeleton rendering.
The issue: KimodoMotionRep.inverse() FK produces twisted poses for LINGO data because
SMPL-X -> SOMA77 -> SOMA30 encoding misaligns rotation axes.
Fix: use SOMA raw poses+transl with SOMA native FK for GT visualization.
"""
import argparse
import os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
sys.path.insert(0, str(PROJECT_ROOT / "SOMA"))

SOMA30_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 7), (6, 8), (6, 9),
    (3, 10), (10, 11), (11, 12), (12, 13), (13, 14), (13, 15),
    (3, 16), (16, 17), (17, 18), (18, 19), (19, 20), (19, 21),
    (0, 22), (22, 23), (23, 24), (24, 25),
    (0, 26), (26, 27), (27, 28), (28, 29),
]
METER_TO_UNIT = 100.0


def load_soma_raw_joints(soma_path):
    """Load SOMA raw data and compute joint positions via SOMA77 FK."""
    d = np.load(str(soma_path), allow_pickle=True)
    poses = d["poses"]       # [T, 77, 3] rotvec
    transl = d["transl"]     # [T, 3] root translation
    joint_orient = d.get("joint_orient", None)  # optional

    T = poses.shape[0]

    from kimodo.skeleton.transforms import rotvec_to_matrix, precompute_joint_orient, apply_joint_orient_local
    from kimodo.skeleton import SOMASkeleton77
    from kimodo.skeleton.kinematics import fk as run_fk
    from kimodo.skeleton import global_rots_to_local_rots

    skel77 = SOMASkeleton77()
    poses_t = torch.from_numpy(poses).float()
    transl_t = torch.from_numpy(transl).float()

    rel_rotmats = rotvec_to_matrix(poses_t)

    if joint_orient is not None:
        jo_t = torch.from_numpy(joint_orient).float()
        n_joints = poses.shape[1]
        if jo_t.shape[0] > n_joints:
            jo_t = jo_t[:n_joints]

        from soma.soma import SOMALayer
        soma_tmp = SOMALayer(
            str(PROJECT_ROOT / "SOMA" / "assets"),
            identity_model_type="smpl",
            device="cpu",
            mode="warp",
        )
        parent_ids = list(soma_tmp.rig_data["joint_parent_ids"])
        if len(parent_ids) > n_joints:
            parent_ids = parent_ids[:n_joints]

        orient_tensor, orient_parent_T = precompute_joint_orient(jo_t, parent_ids)
        abs_rotmats = apply_joint_orient_local(rel_rotmats, orient_tensor, orient_parent_T)
    else:
        abs_rotmats = rel_rotmats

    # Run SOMA77 FK to get correct global joint positions
    global_rots_77, posed_joints_77, _ = run_fk(abs_rotmats, transl_t, skel77)

    return posed_joints_77.numpy(), transl


def map_soma77_to_soma30(pj77, skel30):
    """Map SOMA77 joint positions to SOMA30 using bone names."""
    pj30 = np.zeros((pj77.shape[0], 30, 3))
    name_to_idx30 = {name: i for i, (name, _) in enumerate(skel30.bone_order_names_with_parents)}
    name_to_idx77 = {name: i for i, (name, _) in enumerate(skel77.bone_order_names_with_parents)}

    for j30_name, _ in skel30.bone_order_names_with_parents:
        if j30_name in name_to_idx77:
            j30_idx = name_to_idx30[j30_name]
            j77_idx = name_to_idx77[j30_name]
            pj30[:, j30_idx] = pj77[:, j77_idx]

    return pj30


def draw_skeleton(ax, joints, roots, fi, color, root_color):
    pos = joints[fi]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c=color, s=50, depthshade=False, zorder=10, edgecolors="white", linewidths=0.5)
    for a, b in SOMA30_CONNECTIONS:
        if a < pos.shape[0] and b < pos.shape[0]:
            ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], [pos[a, 2], pos[b, 2]],
                    color=color, linewidth=2.5, zorder=8)
    rp = roots[fi, 0]
    ax.scatter([rp[0]], [rp[1]], [rp[2]],
               c=root_color, s=80, depthshade=False, zorder=11, marker="s", edgecolors="white")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset
    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77

    cache_dir = str(PROJECT_ROOT / "kimodo/kimodo_sceneco/cached_data")
    # Use online dataset to get soma_key; cache_dir still needed for motion_features
    ds_val = LINGOSceneMotionDataset(
        data_root=str(PROJECT_ROOT / "LINGO" / "dataset"),
        max_frames=196, min_frames=40,
        voxel_size=(64, 64, 64),
        train_ratio=0.9, seed=42,
        split="val",
        cache_dir=None,  # Use online mode to get soma_key
    )

    skel30 = SOMASkeleton30()
    skel77 = SOMASkeleton77()

    soma_lingo_dir = PROJECT_ROOT / "soma_converted_all" / "lingo"
    rng = np.random.RandomState(999)
    indices = rng.choice(len(ds_val), size=min(args.num_samples, len(ds_val)), replace=False)

    output_dir = PROJECT_ROOT / "kimodo_scene_project/outputs/dataset_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    for vi, idx in enumerate(sorted(indices)):
        seg = ds_val.segments[idx]
        scene_name = seg.get("scene_name", f"unk_{vi}")
        n_frames = int(seg["length"])
        soma_key = seg.get("soma_key", None)

        # Get text from online dataset
        ds_item = ds_val[idx]
        text = ds_item.get("text", "no-text")
        mf = ds_item["motion_features"]

        print(f"\n[{vi}] scene={scene_name}, frames={n_frames}, text={text[:60]}, soma_key={soma_key}")

        # Try to load SOMA raw data for correct GT
        has_soma = False
        pj30_gt = None
        transl_gt = None
        if soma_key and (soma_lingo_dir / f"{soma_key}_soma.npz").exists():
            try:
                pj77, transl = load_soma_raw_joints(soma_lingo_dir / f"{soma_key}_soma.npz")
                pj30_gt = map_soma77_to_soma30(pj77, skel30)[:n_frames]
                transl_gt = transl[:n_frames].reshape(-1, 1, 3) * METER_TO_UNIT
                has_soma = True
                print(f"  Loaded SOMA raw data: pj77 shape={pj77.shape}")
                # Check body proportions
                h = pj30_gt[0, 6, 1] - min(pj30_gt[0, 24, 1], pj30_gt[0, 29, 1])
                sw = np.linalg.norm(pj30_gt[0, 10] - pj30_gt[0, 16])
                print(f"  SOMA GT height≈{h:.2f}m shoulder_width≈{sw:.2f}m")
            except Exception as e:
                print(f"  SOMA loading failed: {e}")

        if not has_soma or pj30_gt is None:
            print("  No SOMA raw data, skipping (need SOMA for correct GT)")
            continue

        # Render
        frames_to_show = min(10, n_frames)
        step = max(1, n_frames // frames_to_show)
        fig, axes = plt.subplots(2, 5, figsize=(20, 9),
                                 subplot_kw={"projection": "3d"}, facecolor="white")
        axes = axes.flatten()

        all_pts = pj30_gt.reshape(-1, 3)
        all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
        center = np.mean(all_pts, axis=0) if len(all_pts) > 0 else np.zeros(3)
        spread = np.max(np.abs(all_pts - center)) + 40

        for pi in range(frames_to_show):
            fi = min(pi * step, n_frames - 1)
            ax = axes[pi]
            ax.set_facecolor("#FAFAFA")
            j_viz = pj30_gt * METER_TO_UNIT
            r_viz = transl_gt
            draw_skeleton(ax, j_viz, r_viz, fi, "#4CAF50", "#1B5E20")
            ax.set_xlim(center[0] - spread, center[0] + spread)
            ax.set_ylim(center[1] - spread * 0.6, center[1] + spread)
            ax.set_zlim(center[2] - spread * 0.5, center[2] + spread)
            ax.view_init(elev=15, azim=-45)
            ax.set_title(f"t={fi}/{n_frames}", fontsize=9)
            ax.set_axis_off()

        fig.suptitle(f"LINGO Dataset Sample {vi}: {scene_name}\n{text}\n"
                     f"{n_frames} frames, SOMA30 skeleton (SOMA raw GT)",
                     fontsize=11)
        fig.tight_layout()

        out_path = output_dir / f"sample{vi}_{scene_name}.png"
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
