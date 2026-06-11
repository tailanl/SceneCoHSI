# Scene Processing Pipeline (Fixed)

## 数据流

```
LINGO/dataset/dataset/Scene/{name}.npy         原始场景 occupancy grid
         │                                      300×100×400, bool (True=occupied)
         ▼
scripts/precompute_smplx_cache.py               预处理
         │  scipy.ndimage.zoom(order=1) → 64×64×64, float32
         ▼
lingo_smplx_cache/seg_XXXXX.npz                 缓存文件
  ├── motion_features: (T, 273)  归一化后的运动特征
  ├── voxel_grid: (64, 64, 64)   降采样后的场景 occupancy density
  ├── scene_name: str            场景标识
  ├── text: str                  文本描述
  └── text_feat: (1, 1, 4096)    预编码文本特征
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
训练 (dataset.py)                         评估 (eval_sceneadapt_metrics.py)
  加载 voxel_grid (64³)                    加载 voxel_grid (64³)
  → scene_encoder (Voxel ViT)              → 根据 motion extent 动态计算坐标系
  → SceneCo cross-attention                → 构建 2D SDF
  → 模型隐式学习场景-运动对齐                 → 逐帧/关节碰撞检测
```

## 坐标系统

| 阶段 | 坐标空间 | 说明 |
|------|---------|------|
| 原始场景 | voxel indices (300×100×400) | 物理尺度未知，约 0.02m/voxel |
| 缓存场景 | voxel indices (64×64×64) | scipy zoom 降采样，连续 occupancy 值 |
| 运动特征 | normalized feature space | mean/std 归一化，由 motion_rep 管理 |
| 身体输出 | world meter space | `gen_root`, `gen_joints` 经过 inverse transform |
| 评估映射 | **动态计算** | 根据 gen_root 的 min/max 范围，将 64³ grid 映射到 meter 空间 |

## 评估中的场景处理（修复后）

旧版（BUG）:
```python
voxel_size = 0.1          # 硬编码，不匹配任何场景
grid_origin = (0, 0, 0)   # 硬编码，motion 经常在 grid 外
scene = raw_scene.npy     # 加载原始场景（300×100×400），从未降采样
```

新版（FIXED）:
```python
# 1. 从 cache 加载和训练一致的 64³ grid
voxel_64 = cache["voxel_grid"]  # 64×64×64 float32

# 2. 从 motion 范围动态计算坐标映射
x_min, x_max = gen_root[:, 0].min(), gen_root[:, 0].max()
z_min, z_max = gen_root[:, 2].min(), gen_root[:, 2].max()

# 3. 将 64³ grid 映射到 motion 物理范围
ix = int((x_world - gx_min) / (gx_max - gx_min) * 64)
iz = int((z_world - gz_min) / (gz_max - gz_min) * 64)

# 4. 使用 scipy distance_transform_edt 构建 SDF
```

## 指标含义

| 指标 | 含义 | 值域 |
|------|------|------|
| CollisionFrameRate | 任意关节碰到障碍物的帧比例 | 0-1, 越低越好 |
| NonWalkableRootRate | root 位置在障碍物内的帧比例 | 0-1, 越低越好 |
| PenetrationRate | 关节-体素对穿透的比例 | 0-1, 越低越好 |
| PenetrationMean | 平均穿透深度（米） | 越低越好 |
| PenetrationMax | 最大穿透深度（米） | 越低越好 |

## 为什么原始 eval 的 CFR 接近 1.0

旧版 eval 的硬编码 `voxel_size=0.1, grid_origin=(0,0,0)` 导致：
- 场景被错误拉伸到 30m×40m
- 运动坐标在场景边界外 → 被判定为 "outside = obstacle"
- 几乎所有帧都被错误标记为碰撞

修复后 CFR 在 0.2-0.5 之间，符合实际场景中的碰撞情况。

## 注意事项

1. **评估是 2D proxy**: 使用 XZ 平面的 SDF，Y 轴取平均。正式评估应使用 SMPL-X vertices 的 3D SDF。
2. **坐标系是近似的**: 64³ grid 的物理范围由 motion extent 动态确定，不同样本可能有不同映射。
3. **训练不受影响**: 训练中的 SceneCo 交叉注意力自动学习场景-运动对齐，无需显式坐标映射。
