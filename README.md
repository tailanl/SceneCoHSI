# Kimodo Scene Project

基于 LINGO 数据集的 Kimodo 场景感知人体运动生成实验。

## 目录结构

```
kimodo_scene_project/
├── configs/                     # 训练配置文件 (YAML)
│   ├── trajco_cross_smplx.yaml              # D: 纯轨迹交叉注意力
│   ├── trajco_cross_sceneco_smplx.yaml       # E: 场景+轨迹交叉注意力
│   ├── trajco_cross_root_sceneco_body.yaml   # F: 场景body + 轨迹root
│   ├── trajco_cross_root_body_coarse.yaml    # Dcoarse
│   ├── trajco_cross_root_sceneco_body_coarse.yaml  # Fcoarse
│   ├── trajco_cross_root_body_sceneco_body_clean.yaml  # Hclean
│   └── trajco_cross_root_body_sceneco_body_coarse.yaml # Hcoarse
├── scripts/
│   ├── precompute_smplx_cache.py         # 生成 SMPL-X 缓存数据集
│   ├── _exp_root_stage2.py              # Root Stage2 实验 + 可视化
│   └── visualize_generated_motion.py    # 训练后模型生成可视化
├── train/
│   └── train_sceneco.py                 # 训练入口
├── kimodo_sceneco/                       # SceneCo/TrajCo 模型代码
│   ├── model/
│   │   ├── kimodo_model.py              # KimodoSceneCo 模型
│   │   └── scene_encoder.py             # 场景体素编码器
│   └── train/
│       ├── dataset.py                    # LINGO 数据集加载
│       └── train.py                      # 训练逻辑
├── models/                               # 预训练权重 (Kimodo-SMPLX-RP-v1)
├── outputs/                              # 训练输出和可视化
└── LINGO/                                # 数据链接
    └── dataset/dataset/
        ├── human_joints_aligned.npy      # 对齐后关节 [N, 28, 3]
        ├── human_pose.npy                # SMPL-X pose [N, 63] (axis-angle)
        ├── human_orient.npy              # 根节点朝向 [N, 3]
        ├── transl_aligned.npy            # 根位移 [N, 3]
        ├── start_idx.npy / end_idx.npy   # 片段起止索引
        ├── scene_name.pkl / text_aug.pkl # 场景名 & 文本
        └── Scene/                        # 原始场景体素 (300×100×400)
```

## 1. 数据集生成

### 数据来源

LINGO 数据集为 **SMPL-X 格式**，包含完整 pose 参数（axis-angle），可直接通过 Forward Kinematics 获得精确的全局旋转矩阵。

### 缓存格式

每个 `.npz` 文件包含一个运动片段：

| 字段 | 形状 | 说明 |
|---|---|---|
| `motion_features` | `[T, 273]` | z-score 归一化的 KimodoMotionRep 特征 |
| `voxel_grid` | `[64, 64, 64]` | 下采样后的场景体素（0/1 float） |
| `length` | 标量 | 有效帧数 T (40-196) |
| `scene_name` | 字符串 | 场景标识，如 `005`, `009-1` |
| `text` | 字符串 | 动作描述文本 |

### 特征布局 (273维)

```
[ smooth_root_pos(3) | heading(2) | local_joints(66) | global_rot_data(132) | velocities(66) | foot_contacts(4) ]
```

- **smooth_root_pos**: 根关节世界坐标 (x, y, z)
- **heading**: 朝向角 cos/sin
- **local_joints**: X/Z 相对根关节，Y 为绝对世界高度
- **global_rot_data**: 22 个关节的 6D 连续旋转表示（通过 pose FK 计算，非 Rodriguez 近似）
- **velocities**: 关节速度
- **foot_contacts**: 脚部接触检测（4 个脚关键点）

### 执行缓存生成

```bash
cd kimodo_scene_project
python scripts/precompute_smplx_cache.py \
  --data_root LINGO/dataset \
  --model_dir models/Kimodo-SMPLX-RP-v1 \
  --output_dir lingo_smplx_cache \
  --min_frames 40 \
  --max_frames 196 \
  --voxel_size 64 64 64
```

输出: `lingo_smplx_cache/seg_00000.npz` ~ `seg_17315.npz` (共 17,316 段)

## 2. 训练

### 启动全部实验

```bash
bash launch_all.sh
```

在 tmux 会话中启动 7 个实验（D/E/F/Dcoarse/Fcoarse/Hclean/Hcoarse），每个使用不同 GPU。

### 单独启动

```bash
export CHECKPOINT_DIR=kimodo_scene_project/models

python kimodo_scene_project/train/train_sceneco.py \
  kimodo_scene_project/configs/trajco_cross_smplx.yaml \
  2>&1 | tee kimodo_scene_project/outputs/trajco_cross_smplx/train.log
```

### 实验清单

| 实验 | 配置 | 说明 |
|---|---|---|
| D | `trajco_cross_smplx` | TrajCo 交叉注意力，无场景 |
| E | `trajco_cross_sceneco_smplx` | SceneCo + TrajCo，root/body 都用场景 |
| F | `trajco_cross_root_sceneco_body` | TrajCo root + SceneCo body |
| Dcoarse | `trajco_cross_root_body_coarse` | D 的粗粒度变体 |
| Fcoarse | `trajco_cross_root_sceneco_body_coarse` | F 的粗粒度变体 |
| Hclean | `trajco_cross_root_body_sceneco_body_clean` | 清理版混合训练 |
| Hcoarse | `trajco_cross_root_body_sceneco_body_coarse` | H 的粗粒度变体 |

### 监控训练

```bash
# 查看所有会话
tmux list-sessions

# 查看特定实验日志
tmux capture-pane -t train_D -p | tail -20

# 或直接查看日志文件
tail -20 kimodo_scene_project/outputs/trajco_cross_smplx/train.log
```

## 3. Root Stage2 实验

测试 Kimodo Stage2 在给定 GT root 轨迹下生成 body motion 的能力。

```bash
CUDA_VISIBLE_DEVICES=0 python kimodo_scene_project/scripts/_exp_root_stage2.py \
  --gpu 0 \
  --cache_idx 115 174 89 \
  --num_denoising_steps 50
```

### 工作原理

1. 从缓存加载 GT motion 和场景体素
2. 用 `Root2DConstraintSet` + root Y 约束在 denoising 每步注入 GT root（XZ + heading + Y）
3. Kimodo Stage2 在约束下生成 body motion
4. 后处理将 GT root 拼接到生成结果（delta replacement），确保 root 完全一致
5. 输出对比视频：GT vs 生成，含场景、骨架、root 轨迹

### 输出

```
kimodo_scene_project/outputs/exp_root_stage2/
├── cmp_00115.mp4   # 每段一个对比视频
├── cmp_00174.mp4
└── cmp_00089.mp4
```

视频内容：
- 左侧：GT (LINGO 原始数据)
- 右侧：生成结果（root = GT，body 由模型生成）
- 棕色方块：场景体素（6.3m 房间尺度，半透明）
- 彩色骨架：SMPL-X22 骨骼（按肢体分组着色）
- 青色线条：root 轨迹（历史渐强，前方暗淡）
- 底部显示：MPJPE 和骨骼角度误差

## 4. 训练模型可视化

加载已训练 checkpoint 生成可视化对比视频：

```bash
CUDA_VISIBLE_DEVICES=0 python kimodo_scene_project/scripts/visualize_generated_motion.py \
  --experiments D E F \
  --num_samples 3 \
  --num_denoising_steps 50 \
  --output_dir kimodo_scene_project/outputs/viz_generated
```

输出按实验分目录，每段生成一个 mp4 文件，包含：
- 场景点云（按高度分层着色）
- 人物骨架（肢体分组着色）
- 根轨迹（橙色持续路径）
- 起点/当前位置标记

## 5. 关键实现细节

### SMPL-X FK（而非 Rodriguez 近似）

旧方案用骨骼方向近似旋转，误差高达 47.8°。新方案直接从 SMPL-X pose 参数（axis-angle）通过 FK 计算精确全局旋转，误差降至 2.5°。

```python
# pose.reshape(T, 21, 3)  →  axis_angle_to_matrix  →  FK  →  global_rots
body_rots = axis_angle_to_matrix(body_pose_aa)
root_rot = axis_angle_to_matrix(root_orient_aa)
global_rots, _, _ = kimodo_fk(local_rot_mats, transl_t, skel)
global_rot_data = matrix_to_cont6d(global_rots)
```

### Local Joints Y 修复

X/Z 相对根关节，但 Y 分量保留绝对世界高度（与 `KimodoMotionRep.__call__` 一致）：

```python
local_joints = joints_t - smooth_root_pos[:, None, :]
local_joints[..., 1] = joints_t[..., 1]  # Y = absolute world Y
```

### 场景体素

原始场景体素为 `300×100×400`（轴序 Z,Y,X），下采样到 `64×64×64` 供 SceneCo 使用，体素大小约 0.1m/体素（物理范围约 6.4m）。场景文件位于 `LINGO/dataset/dataset/Scene/{name}.npy`。
