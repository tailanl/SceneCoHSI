"""Quick check: decode generated motion features using cache format."""
from __future__ import annotations
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kimodo"))
os.environ["CHECKPOINT_DIR"] = os.path.join(PROJECT_ROOT, "kimodo_scene_project/models")

import numpy as np
import torch
from kimodo.model.load_model import load_model
from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

class PE:
    llm_dim = 4096; max_len = 77; output_dim = 4096
    def __call__(self, *a, **kw): raise RuntimeError()
    def to(self, d): return self
    def train(self, m=True): return self
    def eval(self): return self

cfg = dict(
    ckpt="kimodo_scene_project/outputs/trajco_cross_root_sceneco_body/checkpoints/best_checkpoint.pt",
    has_trajco=True, trajco_type="cross_attn",
    use_in_root=False, use_in_body=True,
    use_trajco_root=True, use_trajco_body=False,
)
device = torch.device("cuda:0")
print("Loading pretrained...")
pretrained = load_model("Kimodo-SMPLX-RP-v1", device=str(device))
print("Building model...")
model = KimodoSceneCo(
    denoiser=pretrained.denoiser.model, text_encoder=pretrained.text_encoder,
    num_base_steps=1000, scene_encoder_type="voxel_vit",
    scene_encoder_config={"voxel_size":(64,64,64),"patch_size":(8,8,8),"d_model":256,"num_layers":4,"use_dual_vit":False,"root_voxel_mode":"full"},
    device=device, cfg_type="scene_separated",
    use_in_root_model=False, use_in_body_model=True,
    use_trajco=True, use_trajco_root=True, use_trajco_body=False,
    traj_dim=5, trajco_type="cross_attn",
).to(device).eval()
ckpt = torch.load(cfg["ckpt"], map_location=device)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
print("OK\n")

names = ["pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2","l_ankle","r_ankle",
         "spine3","l_foot","r_foot","neck","l_collar","r_collar","head",
         "l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist"]

for seg in [115]:
    data = np.load(f"lingo_smplx_cache/seg_{seg:05d}.npz", allow_pickle=True)
    T = int(data["length"])
    voxel = torch.from_numpy(np.squeeze(data["voxel_grid"])).float().to(device).unsqueeze(0).unsqueeze(0)
    traj = torch.from_numpy(data["motion_features"][:T, :5]).float().to(device).unsqueeze(0)
    text_feat = torch.from_numpy(data["text_feat"]).float().to(device) if "text_feat" in data else None

    (sfr, smr), (sfb, smb) = model.encode_scene(voxel)
    tf_, tm = model.encode_traj(traj)
    pmsk = torch.ones(1, T, dtype=torch.bool, device=device)

    with torch.no_grad():
        raw = model._generate(
            [str(data['text'])], T, num_denoising_steps=50,
            pad_mask=pmsk, first_heading_angle=torch.tensor([0.0], device=device),
            motion_mask=None, observed_motion=None,
            cfg_weight=[3.0, 1.5, 2.0], cfg_type="scene_separated",
            scene_feat_root=sfr, scene_mask_root=smr,
            scene_feat_body=sfb, scene_mask_body=smb,
            traj_feats=tf_, traj_mask=tm, text_feat=text_feat,
        )
        raw = raw[0]

    features = model.motion_rep.unnormalize(raw)
    root_pos = features[:, :3].cpu().numpy()
    local_joints = features[:, 5:71].cpu().numpy().reshape(-1, 22, 3)
    joints = local_joints + root_pos[:, None, :]

    print(f"seg_{seg:05d} [{str(data['text'])[:40]}]")
    print(f"  Frame 0:")
    for i in [0, 3, 6, 9, 12, 15]:
        print(f"    {i:2d} {names[i]:15s}: [{joints[0,i,0]:.4f}, {joints[0,i,1]:.4f}, {joints[0,i,2]:.4f}]")
    for i in [4, 5, 7, 8, 10, 11]:
        print(f"    {i:2d} {names[i]:15s}: [{joints[0,i,0]:.4f}, {joints[0,i,1]:.4f}, {joints[0,i,2]:.4f}]")

    pelvis_y = joints[:, 0, 1].mean()
    head_y = joints[:, 15, 1].mean()
    foot_y = joints[:, [10, 11], 1].mean()
    print(f"  Pelvis Y mean: {pelvis_y:.4f}")
    print(f"  Head Y mean:   {head_y:.4f}")
    print(f"  Foot Y mean:   {foot_y:.4f}")
    print(f"  head-pelvis:   {head_y - pelvis_y:.4f}  (should be > 0)")
    print(f"  pelvis-foot:   {pelvis_y - foot_y:.4f}  (should be > 0)")
    print(f"  -> {'✓ UPRIGHT' if head_y > pelvis_y > foot_y else '✗ INVERTED'}")
