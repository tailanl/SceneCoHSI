"""Check FK joint coordinates for multiple samples."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Check neutral joints
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))
from kimodo.skeleton.definitions import SMPLXSkeleton22
skel = SMPLXSkeleton22()
neutral = skel.neutral_joints.squeeze().numpy()  # (22, 3)

names = ["pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2","l_ankle","r_ankle","spine3","l_foot","r_foot","neck","l_collar","r_collar","head","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist"]

print("=== NEUTRAL JOINTS (from skeleton) ===")
print(f"Pelvis:    X={neutral[0,0]:.4f}, Y={neutral[0,1]:.4f}, Z={neutral[0,2]:.4f}")
print(f"Head:      X={neutral[15,0]:.4f}, Y={neutral[15,1]:.4f}, Z={neutral[15,2]:.4f}")
print(f"L_foot:    X={neutral[10,0]:.4f}, Y={neutral[10,1]:.4f}, Z={neutral[10,2]:.4f}")
print(f"R_foot:    X={neutral[11,0]:.4f}, Y={neutral[11,1]:.4f}, Z={neutral[11,2]:.4f}")

# Spine direction
print(f"\n=== SPINE DIRECTION (neutral) ===")
for i in [0, 3, 6, 9, 12, 15]:
    print(f"  {names[i]:15s}: X={neutral[i,0]:.4f}, Y={neutral[i,1]:.4f}, Z={neutral[i,2]:.4f}")

# Check a few GT samples
print("\n=== GT SAMPLES ===")
for seg in [115, 174, 89]:
    npz = PROJECT_ROOT / f"lingo_smplx_cache/seg_{seg:05d}.npz"
    if not npz.exists():
        continue
    d = np.load(str(npz), allow_pickle=True)
    T = int(d["length"])
    mf = d["motion_features"][:T]

    # smooth_root_pos dims
    print(f"\nseg_{seg:05d}: T={T}, text='{str(d.get('text',''))[:40]}'")
    print(f"  smooth_root X: [{mf[:,0].min():.3f}, {mf[:,0].max():.3f}]")
    print(f"  smooth_root Y: [{mf[:,1].min():.3f}, {mf[:,1].max():.3f}]")
    print(f"  smooth_root Z: [{mf[:,2].min():.3f}, {mf[:,2].max():.3f}]")

    # local joints (dims 5-70 = 22*3)
    lj = mf[:, 5:70].reshape(-1, 22, 3)
    print(f"  local_joints Y pelvis:  [{lj[:,0,1].min():.3f}, {lj[:,0,1].max():.3f}]")
    print(f"  local_joints Y head:    [{lj[:,15,1].min():.3f}, {lj[:,15,1].max():.3f}]")
    print(f"  local_joints Y l_foot:  [{lj[:,10,1].min():.3f}, {lj[:,10,1].max():.3f}]")
    print(f"  local_joints Z spine1:  [{lj[:,3,2].min():.3f}, {lj[:,3,2].max():.3f}]")
    print(f"  local_joints Z head:    [{lj[:,15,2].min():.3f}, {lj[:,15,2].max():.3f}]")

    # Spine direction in local joints
    print(f"  Spine direction (local):")
    for i in [0, 3, 6, 9, 12, 15]:
        j = lj[0, i]
        print(f"    {names[i]:15s}: X={j[0]:.4f}, Y={j[1]:.4f}, Z={j[2]:.4f}")
