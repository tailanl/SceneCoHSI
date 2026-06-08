"""Debug: check generated FK joint coordinates in detail."""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add kimodo package to path
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kimodo"))
# Use local models directory instead of incomplete HuggingFace cache
os.environ["CHECKPOINT_DIR"] = os.path.join(PROJECT_ROOT, "kimodo_scene_project/models")

import numpy as np
import torch

from kimodo.model.load_model import load_model
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

class PrecomputedTextEncoder:
    llm_dim = 4096; max_len = 77; output_dim = 4096
    def __call__(self, *a, **kw): raise RuntimeError("use text_feat")
    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cfg = dict(
    ckpt=os.path.join(PROJECT_ROOT, "kimodo_scene_project/outputs/trajco_cross_root_sceneco_body/checkpoints/best_checkpoint.pt"),
    has_trajco=True, trajco_type="cross_attn",
    use_in_root=False, use_in_body=True,
    use_trajco_root=True, use_trajco_body=False,
)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
pretrained = load_model("Kimodo-SMPLX-RP-v1", device=str(device), text_encoder=PrecomputedTextEncoder())
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

names = ["pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2","l_ankle","r_ankle","spine3","l_foot","r_foot","neck","l_collar","r_collar","head","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist"]

for seg in [115, 174, 89]:
    data = np.load(os.path.join(PROJECT_ROOT, f"lingo_smplx_cache/seg_{seg:05d}.npz"), allow_pickle=True)
    T = int(data["length"])
    voxel = torch.from_numpy(np.squeeze(data["voxel_grid"])).float().to(device).unsqueeze(0).unsqueeze(0)
    text_feat = torch.from_numpy(data["text_feat"]).float().to(device)
    traj = torch.from_numpy(data["motion_features"][:T, :5]).float().to(device).unsqueeze(0)

    print(f"\n{'='*60}")
    print(f"seg_{seg:05d}: '{str(data.get('text',''))[:40]}' (T={T})")
    print(f"{'='*60}")

    with torch.no_grad():
        gen = model(
            prompts=str(data.get("text","")), num_frames=T,
            num_denoising_steps=50, cfg_weight=[3.0,1.5,2.0], cfg_type="scene_separated",
            scene_input=voxel, traj_input=traj,
            text_feat=text_feat, return_numpy=True,
        )

    joints = np.squeeze(gen["posed_joints"]).copy()  # (T, 22, 3)
    sr = np.squeeze(gen.get("smooth_root_pos"))  # (T, 5) or (T,)
    rp = np.squeeze(gen.get("root_positions"))  # (T, 3)

    if joints.ndim == 2:
        joints = joints.reshape(-1, 22, 3)
    
    print(f"joints shape: {joints.shape}")
    print(f"smooth_root_pos shape: {sr.shape}")
    print(f"root_positions shape: {rp.shape}")

    # Handle different smooth_root_pos shapes
    if sr.ndim == 2:
        root_y = sr[:, 1] if sr.shape[1] > 1 else np.zeros(sr.shape[0])
        rp_y = rp[:, 1] if rp.ndim == 2 and rp.shape[1] > 1 else np.zeros(rp.shape[0])
    else:
        root_y = sr
        rp_y = rp if rp.ndim == 1 else rp[:, 0]
    
    print(f"\nsmooth_root_pos Y: [{root_y.min():.4f}, {root_y.max():.4f}]")
    print(f"root_positions Y: [{rp_y.min():.4f}, {rp_y.max():.4f}]")
    print(f"joints Y (before root add): [{joints[..., 1].min():.4f}, {joints[..., 1].max():.4f}]")

    # Add root Y to joints Y
    joints[..., 1] += root_y[:, None]
    print(f"joints Y (after root add): [{joints[..., 1].min():.4f}, {joints[..., 1].max():.4f}]")

    # Frame 0
    j0 = joints[0]
    print(f"\nFrame 0 joints (X, Y, Z):")
    for i, n in enumerate(names):
        print(f"  {i:2d} {n:15s}: [{j0[i, 0]:.4f}, {j0[i, 1]:.4f}, {j0[i, 2]:.4f}]")

    # Spine Y direction
    spine_y = [j0[i, 1] for i in [0, 3, 6, 9, 12, 15]]
    print(f"\nSpine Y: {[f'{v:.4f}' for v in spine_y]}")
    print(f"Spine Y diff (head-pelvis): {spine_y[-1] - spine_y[0]:.4f}")

    # Leg Y
    print(f"Pelvis Y: {j0[0, 1]:.4f}")
    print(f"L_knee Y: {j0[4, 1]:.4f}")
    print(f"L_ankle Y: {j0[7, 1]:.4f}")
    print(f"L_foot Y: {j0[10, 1]:.4f}")
    print(f"Leg Y diff (pelvis-foot): {j0[10, 1] - j0[0, 1]:.4f}")

    # Overall
    print(f"\nJoint Y range (all frames): [{joints[..., 1].min():.4f}, {joints[..., 1].max():.4f}]")
    print(f"Height range: {joints[..., 1].max() - joints[..., 1].min():.4f}")

    # Check: is character upright or lying down?
    pelvis_y = joints[:, 0, 1].mean()
    head_y = joints[:, 15, 1].mean()
    foot_y = joints[:, [10, 11], 1].mean()
    spine_z = joints[:, 9, 2].mean()
    head_z = joints[:, 15, 2].mean()
    
    print(f"\n=== Pose Orientation Check ===")
    print(f"Mean pelvis Y: {pelvis_y:.4f}")
    print(f"Mean head Y: {head_y:.4f}")
    print(f"Mean foot Y: {foot_y:.4f}")
    print(f"head Y - pelvis Y: {head_y - pelvis_y:.4f}")
    print(f"pelvis Y - foot Y: {pelvis_y - foot_y:.4f}")
    print(f"Mean spine Z: {spine_z:.4f}")
    print(f"Mean head Z: {head_z:.4f}")
    print(f"head Z - spine Z: {head_z - spine_z:.4f}")
    
    if head_y - pelvis_y > 0.3:
        print("→ Character appears UPRIGHT (head above pelvis)")
    elif abs(head_y - pelvis_y) < 0.3 and abs(head_z - spine_z) > 0.3:
        print("→ Character appears LYING DOWN (head-forward in Z direction)")
    else:
        print("→ Character pose orientation unclear")
