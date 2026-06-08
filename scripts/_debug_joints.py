"""Quick debug: check generated joints coordinate ranges."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

import numpy as np
import torch
from pathlib import Path

from kimodo.model.load_model import load_model
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo


class PrecomputedTextEncoder:
    llm_dim = 4096; max_len = 77; output_dim = 4096
    def __call__(self, *a, **kw): raise RuntimeError("use text_feat")
    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self


cfg = dict(
    ckpt="kimodo_scene_project/outputs/trajco_cross_root_sceneco_body/checkpoints/best_checkpoint.pt",
    has_trajco=True, trajco_type="cross_attn",
    use_in_root=False, use_in_body=True,
    use_trajco_root=True, use_trajco_body=False,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg["ckpt"] = str(PROJECT_ROOT / cfg["ckpt"])

device = torch.device("cuda:0")
pretrained = load_model("Kimodo-SMPLX-RP-v1", device=str(device))
text_encoder = PrecomputedTextEncoder()

model = KimodoSceneCo(
    pretrained_model=pretrained,
    text_encoder=text_encoder,
    use_in_root_model=cfg["use_in_root"],
    use_in_body_model=cfg["use_in_body"],
    trajco_type=cfg["trajco_type"],
    use_trajco=cfg["has_trajco"],
    use_trajco_root=cfg.get("use_trajco_root", False),
    use_trajco_body=cfg.get("use_trajco_body", False),
).to(device)
ckpt = torch.load(cfg["ckpt"], map_location=device)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# Load a sample
sample = np.load(str(PROJECT_ROOT / "lingo_smplx_cache/seg_00115.npz"), allow_pickle=True)
voxel = torch.from_numpy(np.squeeze(sample["voxel"])).float().to(device).unsqueeze(0).unsqueeze(0)
text_feat = torch.from_numpy(sample["text_feat"]).float().to(device)
duration = torch.tensor(sample["num_frames"], dtype=torch.long, device=device)
traj = torch.from_numpy(sample["motion_features"][:, :5]).float().to(device).unsqueeze(0)
traj_feats, traj_mask = model.encode_traj(traj)

print("Generating...")
with torch.no_grad():
    gen = model(
        duration=duration, max_duration=duration,
        scene_voxel=voxel, traj_input=traj, text_feat=text_feat,
        guidance_strength=3.0, num_steps=50,
    )

joints = gen["posed_joints"].detach().cpu().numpy().squeeze()
root_pos = gen.get("root_positions", gen.get("smooth_root_pos")).detach().cpu().numpy().squeeze()

print(f"\njoints shape: {joints.shape}")
print(f"root shape: {root_pos.shape}")

if root_pos.ndim == 2:  # (T, 3) or (1, T, 3)
    root_trail = root_pos
elif root_pos.ndim == 3:
    root_trail = root_pos[:, 0, :]  # (T, 1, 3) -> (T, 3)
else:
    root_trail = root_pos.reshape(-1, 3)

print(f"\n--- Frame 0 (start) ---")
print(f"pelvis (joint 0): {joints[0, 0]}")
print(f"head (joint 15): {joints[0, 15]}")
print(f"left_foot (joint 10): {joints[0, 10]}")
print(f"root[0]: {root_trail[0]}")
print(f"root_last: {root_trail[-1]}")

print(f"\n--- Stats ---")
print(f"joint Y range: {joints[..., 1].min():.3f} to {joints[..., 1].max():.3f}")
print(f"           (height = {joints[..., 1].max() - joints[..., 1].min():.3f})")
print(f"joint Z range: {joints[..., 2].min():.3f} to {joints[..., 2].max():.3f}")
print(f"root Y range:  {root_trail[..., 1].min():.3f} to {root_trail[..., 1].max():.3f}")
print(f"root Z range:  {root_trail[..., 2].min():.3f} to {root_trail[..., 2].max():.3f}")

print(f"\njoint X range: {joints[..., 0].min():.3f} to {joints[..., 0].max():.3f}")
print(f"root X range:  {root_trail[..., 0].min():.3f} to {root_trail[..., 0].max():.3f}")
