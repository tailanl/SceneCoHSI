"""Test: generate motion with text-only (no scene, no trajectory) to check if pose is correct."""
from __future__ import annotations
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kimodo"))
os.environ["CHECKPOINT_DIR"] = os.path.join(PROJECT_ROOT, "kimodo_scene_project/models")

import numpy as np
import torch

from kimodo.model.load_model import load_model
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

class PrecomputedTextEncoder:
    llm_dim = 4096
    max_len = 77
    output_dim = 4096
    def __call__(self, *a, **kw): raise RuntimeError("use text_feat")
    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self

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

names = ["pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2","l_ankle","r_ankle",
         "spine3","l_foot","r_foot","neck","l_collar","r_collar","head",
         "l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist"]

# Test prompts - simple actions
test_prompts = [
    "A person stands still",
    "A person walks forward",
    "A person sits down on a chair",
]

num_frames = 60

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*60}")

    with torch.no_grad():
        gen = model(
            prompts=prompt, num_frames=num_frames,
            num_denoising_steps=50, cfg_weight=[3.0, 1.5, 2.0], cfg_type="scene_separated",
            scene_input=None,  # NO scene
            traj_input=None,   # NO trajectory
            text_feat=None,    # NO precomputed text feature (model will use its own encoder)
            return_numpy=True,
        )

    joints = np.squeeze(gen["posed_joints"])  # (T, 22, 3)
    sr = np.squeeze(gen.get("smooth_root_pos"))  # (T, 3)
    rp = np.squeeze(gen.get("root_positions"))  # (T, 3)

    print(f"joints shape: {joints.shape}")
    print(f"smooth_root_pos shape: {sr.shape}")
    print(f"root_positions shape: {rp.shape}")

    # Add root Y to joints Y
    if sr.ndim == 2 and sr.shape[1] > 1:
        root_y = sr[:, 1]
    elif sr.ndim == 1:
        root_y = sr
    else:
        root_y = np.zeros(joints.shape[0])
    
    joints[..., 1] += root_y[:, None]

    # Frame 0 analysis
    j0 = joints[0]
    print(f"\nFrame 0 joints (X, Y, Z):")
    for i, n in enumerate(names):
        print(f"  {i:2d} {n:15s}: [{j0[i, 0]:.4f}, {j0[i, 1]:.4f}, {j0[i, 2]:.4f}]")

    # Pose orientation check
    pelvis_y = joints[:, 0, 1].mean()
    head_y = joints[:, 15, 1].mean()
    foot_y = joints[:, [10, 11], 1].mean()
    
    print(f"\n=== Pose Orientation Check ===")
    print(f"Mean pelvis Y: {pelvis_y:.4f}")
    print(f"Mean head Y: {head_y:.4f}")
    print(f"Mean foot Y: {foot_y:.4f}")
    print(f"head Y - pelvis Y: {head_y - pelvis_y:.4f}")
    print(f"pelvis Y - foot Y: {pelvis_y - foot_y:.4f}")

    if head_y - pelvis_y > 0.3:
        print("→ Character appears UPRIGHT (head above pelvis) ✓")
    elif head_y - pelvis_y < -0.3:
        print("→ Character appears INVERTED (head below pelvis) ✗")
    else:
        print("→ Character orientation unclear")
