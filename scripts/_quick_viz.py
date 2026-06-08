"""Quick debug: generate one sample, save frame 0 as PNG for inspection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kimodo.model.load_model import load_model
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo


class PrecomputedTextEncoder:
    llm_dim = 4096; max_len = 77; output_dim = 4096
    def __call__(self, *a, **kw): raise RuntimeError("use text_feat")
    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Experiment F config
cfg = dict(
    ckpt=str(PROJECT_ROOT / "kimodo_scene_project/outputs/trajco_cross_root_sceneco_body/checkpoints/best_checkpoint.pt"),
    has_trajco=True, trajco_type="cross_attn",
    use_in_root=False, use_in_body=True,
    use_trajco_root=True, use_trajco_body=False,
)

device = torch.device("cuda:0")
pretrained = load_model("Kimodo-SMPLX-RP-v1", device=str(device))
text_encoder = PrecomputedTextEncoder()

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

ckpt = torch.load(cfg["ckpt"], map_location=device)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

# Load sample
data = np.load(str(PROJECT_ROOT / "lingo_smplx_cache/seg_00115.npz"), allow_pickle=True)
T = int(data["length"])
voxel = torch.from_numpy(np.squeeze(data["voxel_grid"])).float().to(device).unsqueeze(0).unsqueeze(0)
text_feat = torch.from_numpy(data["text_feat"]).float().to(device)
traj = torch.from_numpy(data["motion_features"][:T, :5]).float().to(device).unsqueeze(0)

print("Generating...")
with torch.no_grad():
    gen = model(
        prompts=str(data.get("text","")), num_frames=T,
        num_denoising_steps=50, cfg_weight=[3.0,1.5,2.0], cfg_type="scene_separated",
        scene_input=voxel, traj_input=traj,
        text_feat=text_feat, return_numpy=True,
    )

joints = np.squeeze(gen["posed_joints"])        # (T, 22, 3)
root_pos = np.squeeze(gen.get("root_positions", gen.get("smooth_root_pos")))

print(f"joints shape: {joints.shape}")
print(f"root_pos shape: {root_pos.shape}")

# Check raw values
print(f"\n--- Frame 0 joints ---")
for ji, name in enumerate(["pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2","l_ankle","r_ankle","spine3","l_foot","r_foot","neck","l_collar","r_collar","head","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist"]):
    print(f"  {ji:2d} {name:15s}: {joints[0, ji]}")

print(f"\n--- Stats ---")
print(f"joints X: [{joints[...,0].min():.3f}, {joints[...,0].max():.3f}]")
print(f"joints Y: [{joints[...,1].min():.3f}, {joints[...,1].max():.3f}]")
print(f"joints Z: [{joints[...,2].min():.3f}, {joints[...,2].max():.3f}]")
print(f"root   X: [{root_pos[...,0].min():.3f}, {root_pos[...,0].max():.3f}]")
print(f"root   Y: [{root_pos[...,1].min():.3f}, {root_pos[...,1].max():.3f}]")
print(f"root   Z: [{root_pos[...,2].min():.3f}, {root_pos[...,2].max():.3f}]")

# Also check neutral joints for comparison
neutral = torch.load(str(PROJECT_ROOT / "kimodo/kimodo/assets/skeletons/smplx22/joints.p")).squeeze().numpy()
print(f"\nneutral joints Y: [{neutral[:,1].min():.3f}, {neutral[:,1].max():.3f}]")
print(f"neutral pelvis Y: {neutral[0,1]:.3f}")
print(f"neutral head   Y: {neutral[15,1]:.3f}")
print(f"neutral foot   Y: {neutral[10,1]:.3f}")

# Compute per-frame bone lengths to check for jitter
bone_lens = {}
for p,c in [(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]:
    diffs = joints[:,c] - joints[:,p]
    lens = np.sqrt((diffs**2).sum(axis=1))
    print(f"  bone {p}->{c}: len=[{lens.min():.4f}, {lens.max():.4f}] (range={lens.max()-lens.min():.4f})")

# Save frame 0 as scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

j0 = joints[0]
# Plot in standard (X, Y, Z)
ax.scatter(j0[:, 0], j0[:, 1], j0[:, 2], c='red', s=50)
for p, c in [(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]:
    ax.plot([j0[p,0], j0[c,0]], [j0[p,1], j0[c,1]], [j0[p,2], j0[c,2]], 'b-', linewidth=2)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title(f"Frame 0 - raw FK joints (X,Y,Z)")
plt.savefig(str(PROJECT_ROOT / "kimodo_scene_project/outputs/viz_generated/debug_frame0_xyz.png"), dpi=150)
plt.close()

# Plot in (X, Z, Y) - the way renderer maps it
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(j0[:, 0], j0[:, 2], j0[:, 1], c='red', s=50)
for p, c in [(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]:
    ax.plot([j0[p,0], j0[c,0]], [j0[p,2], j0[c,2]], [j0[p,1], j0[c,1]], 'b-', linewidth=2)
ax.set_xlabel("X"); ax.set_ylabel("Z→Y"); ax.set_zlabel("Y→Z (up)")
ax.set_title(f"Frame 0 - renderer mapping (X, Z, Y)")
plt.savefig(str(PROJECT_ROOT / "kimodo_scene_project/outputs/viz_generated/debug_frame0_xzy.png"), dpi=150)
plt.close()

print(f"\nSaved debug images to {PROJECT_ROOT}/kimodo_scene_project/outputs/viz_generated/")
