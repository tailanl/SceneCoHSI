"""Debug FK coordinate system: check generated joints in detail."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

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
os.chdir(PROJECT_ROOT)

cfg = dict(
    ckpt=os.path.join(PROJECT_ROOT, "kimodo_scene_project/outputs/trajco_cross_root_sceneco_body/checkpoints/best_checkpoint.pt"),
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

data = np.load(os.path.join(PROJECT_ROOT, "lingo_smplx_cache/seg_00115.npz"), allow_pickle=True)
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

joints = np.squeeze(gen["posed_joints"])
smooth_root = np.squeeze(gen.get("smooth_root_pos"))
root_pos = np.squeeze(gen.get("root_positions"))

print(f"\n=== Joint shape: {joints.shape} ===")
print(f"smooth_root shape: {smooth_root.shape}")
print(f"root_positions shape: {root_pos.shape}")

# Frame 0 joint positions
names = ["pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2","l_ankle","r_ankle","spine3","l_foot","r_foot","neck","l_collar","r_collar","head","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist"]
print(f"\n--- Frame 0 joints (X, Y, Z) ---")
for i, n in enumerate(names):
    print(f"  {i:2d} {n:15s}: [{joints[0,i,0]:.4f}, {joints[0,i,1]:.4f}, {joints[0,i,2]:.4f}]")

# Frame 50 for comparison
print(f"\n--- Frame 50 joints (X, Y, Z) ---")
for i, n in enumerate(names):
    print(f"  {i:2d} {n:15s}: [{joints[50,i,0]:.4f}, {joints[50,i,1]:.4f}, {joints[50,i,2]:.4f}]")

# Check bone lengths for frame 0
# SMPLX22 skeleton: pelvis(0) -> left_hip(1) -> left_knee(4) -> left_ankle(7) -> left_foot(10)
#                   pelvis(0) -> right_hip(2) -> right_knee(5) -> right_ankle(8) -> right_foot(11)
#                   pelvis(0) -> spine1(3) -> spine2(6) -> spine3(9) -> neck(12) -> head(15)
print(f"\n--- Frame 0 bone lengths ---")
bone_pairs = [(0,1,"pelvis->l_hip"),(1,4,"l_hip->l_knee"),(4,7,"l_knee->l_ankle"),(7,10,"l_ankle->l_foot"),
              (0,2,"pelvis->r_hip"),(2,5,"r_hip->r_knee"),(5,8,"r_knee->r_ankle"),(8,11,"r_ankle->r_foot"),
              (0,3,"pelvis->spine1"),(3,6,"spine1->spine2"),(6,9,"spine2->spine3"),
              (9,12,"spine3->neck"),(12,15,"neck->head"),
              (9,13,"spine3->l_collar"),(13,16,"l_collar->l_shoulder"),(16,18,"l_shoulder->l_elbow"),(18,20,"l_elbow->l_wrist"),
              (9,14,"spine3->r_collar"),(14,17,"r_collar->r_shoulder"),(17,19,"r_shoulder->r_elbow"),(19,21,"r_elbow->r_wrist")]

for p, c, name in bone_pairs:
    f0 = np.linalg.norm(joints[0,c] - joints[0,p])
    f50 = np.linalg.norm(joints[50,c] - joints[50,p])
    # Full range
    all_lens = np.linalg.norm(joints[:,c] - joints[:,p], axis=1)
    print(f"  {name:25s}: frame0={f0:.4f}, frame50={f50:.4f}, range=[{all_lens.min():.4f}, {all_lens.max():.4f}]")

# The key question: is the spine roughly aligned with Z or Y?
# If spine is along Z: spine1 Y ≈ pelvis Y, spine1 Z ≠ pelvis Z
# If spine is along Y: spine1 Y ≠ pelvis Y, spine1 Z ≈ pelvis Z
print(f"\n--- Spine alignment check (frame 0) ---")
for ji in [0, 3, 6, 9, 12, 15]:
    n = names[ji]
    j = joints[0, ji]
    print(f"  {n:15s}: X={j[0]:.4f}, Y={j[1]:.4f}, Z={j[2]:.4f}")

# Also check neutral joints
neutral = torch.load(os.path.join(PROJECT_ROOT, "kimodo/kimodo/assets/skeletons/smplx22/joints.p")).squeeze().numpy()
print(f"\n--- Neutral joints ---")
for ji in [0, 3, 6, 9, 12, 15]:
    n = names[ji]
    j = neutral[ji]
    print(f"  {n:15s}: X={j[0]:.4f}, Y={j[1]:.4f}, Z={j[2]:.4f}")
