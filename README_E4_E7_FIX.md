# README_E4_E7_FIX

本文档记录 E4 和 E7 的最新修复方案。核心判断标准不变：

补充：本方案中的代码层问题已经在 `README_E4_E7_CODE_FIX.md` 中记录了实际补齐内容。当前仍未完成的是全量 v3 root 重新生成、Stage2 重跑、body generation 和 metrics。

1. Stage2 进程在跑不等于实验有效。
2. `.npz` 文件存在不等于 root artifact 有效。
3. Stage2 必须真实加载 external root，不能 fallback 到 GT root。
4. body generation 必须报告 root fix max error，且 `max_error < 1e-5`。
5. 最终必须有 `path_metrics.csv` 和 `scene_metrics.csv`，才可以讨论 `VALID_DONE`。

## 当前结论

### E4

E4 目前还没有修对。

当前问题：

1. 旧目录 `outputs/e4_energy_guidance_train/path_only` 和
   `outputs/e4_energy_guidance_val/path_only` 是混合目录，包含旧 schema、错误 split、
   不完整文件。
2. `outputs/e4_energy_guidance_train_v2` 只有少量文件，不能用于 Stage2。
3. `outputs/e4_energy_guidance_val_v2` 为空，不能用于 Stage2。
4. 当前 E4 Stage2 进程仍指向旧 `path_only` 目录。
5. E4 fast150 日志出现 fallback：

```text
external_root_sources (first 3): ['path_guided_root', 'path_root_missing_gt_fallback', 'path_root_missing_gt_fallback']
```

6. `scripts/fill_missing_e4_roots.py` 只是在旧目录中补缺失文件，不会重写旧 schema 文件；
   并且它把 `guided_root_5d_meter` 的 heading 后两维写成 0，不适合作为干净有效的 root 集。

### E7

E7 只修对了 root 文件覆盖和 schema，还没有完成有效实验。

当前状态：

1. `outputs/e7_gt_root_v2_train` 数量匹配 train split。
2. `outputs/e7_gt_root_v2_val` 数量匹配 val split。
3. E7 v2 root 文件名、必需 key、shape 已通过覆盖检查。
4. 但现有 E7 Stage2 日志没有使用 `outputs/e7_gt_root_v2_train` 或
   `outputs/e7_gt_root_v2_val`。
5. 旧 E7 Stage2 仍然 fallback：

```text
external_root_sources (first 3): ['path_root_missing_gt_fallback', 'path_root_missing_gt_fallback', 'path_root_missing_gt_fallback']
```

6. `scripts/export_gt_root_for_stage2_v2.py` 仍有一个语义问题：cache 里的
   `motion_features` 已经是 normalized，但脚本又调用了一次 `mr.normalize(feat_t)`。
   因此 `guided_root_5d_norm` 对 Stage2 训练可用，但
   `guided_root_5d_meter` 和 `target_path_xz` 的 meter-space 语义不能直接信。

## 总体修复原则

不要继续在旧目录上补丁式修复。应新建干净目录：

```text
outputs/e4_energy_guidance_train_v3
outputs/e4_energy_guidance_val_v3
outputs/e7_gt_root_v3_train
outputs/e7_gt_root_v3_val
```

当前 Stage2 split 的期望数量是：

```text
train: 15584
val:   1732
```

每个 root `.npz` 文件必须包含：

```text
guided_root_5d_norm
guided_root_5d_meter
target_path_xz
text
scene_name
source_file
```

其中：

```text
guided_root_5d_norm   # (T, 5), normalized Kimodo root feature, Stage2 实际读取这个
guided_root_5d_meter  # (T, 5), meter-space root: x, y, z, heading_cos, heading_sin
target_path_xz        # (T, 2), meter-space target path, x/z
```

## Stage2 external_root 匹配规则

Stage2 dataset 在 `kimodo_sceneco/train/dataset.py` 中按 `source_id` 加载 external root：

```python
npz_path = self.path_guided_root_dir / f"{source_id}.npz"
npz_path = self.path_scene_guided_root_dir / f"{source_id}.npz"
```

cache-based 数据分支中，`source_id` 是 cache 文件名 stem，例如：

```text
seg_00014
```

所以 root 文件必须是：

```text
<root_dir>/seg_00014.npz
```

如果文件名不匹配，Stage2 会 fallback。

## E4 修改方案

### 1. 废弃旧补文件方案

不要把当前旧目录当作有效 E4 root 目录：

```text
outputs/e4_energy_guidance_train/path_only
outputs/e4_energy_guidance_val/path_only
```

不要使用 `fill_missing_e4_roots.py` 作为最终修复方案。它最多只能临时补文件，不能产生干净有效的全量 root set。

### 2. 修改 E4 root 生成逻辑

目标脚本：

```text
scripts/generate_root_guidance.py
```

要求：

1. 使用 cache-based split，和 Stage2 dataset 完全一致。
2. 输出文件名必须是 `seg_XXXXX.npz`。
3. 每个 sample 开始时初始化 `target_path_xz = None`，避免外部 path 分支下未定义。
4. `guided_root_5d_norm` 保存最终送入 Stage2 的 normalized root feature。
5. `guided_root_5d_meter` 必须来自 `motion_rep.inverse(cur_mot, is_normalized=True)` 后的 meter root 和真实 heading。
6. 不允许把 heading 后两维填成全 0。
7. `target_path_xz` 保存实际 guidance 使用的 meter-space target path。
8. 保存完整 root schema。

保存逻辑应类似：

```python
output = model.motion_rep.inverse(cur_mot, is_normalized=True, return_numpy=True)
gen_root = output["smooth_root_pos"][0].astype(np.float32)
heading = output["global_root_heading"][0].astype(np.float32)

guided_root_5d_norm = (
    cur_mot[0, :, model.motion_rep.root_slice]
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)

guided_root_5d_meter = np.concatenate(
    [gen_root[:, :3], heading],
    axis=-1,
).astype(np.float32)

target_path_np = target_path_xz[0].detach().cpu().numpy().astype(np.float32)

np.savez(
    str(output_dir / f"{sample['cache_stem']}.npz"),
    guided_root_5d_norm=guided_root_5d_norm,
    guided_root_5d_meter=guided_root_5d_meter,
    target_path_xz=target_path_np,
    text=np.asarray(sample["text"]),
    scene_name=np.asarray(sample.get("scene_name", "")),
    source_file=np.asarray(str(sample["cache_path"])),
)
```

如果直接使用 `cur_mot[..., root_slice]` 作为 `guided_root_5d_norm`，需要确认它就是最终送入 Stage2 的 normalized root。
更稳妥的方式是参考 classifier guidance 中从 decoded output 重新编码 root 的做法。

### 3. 重新生成 E4 root

重新生成到干净目录：

```text
outputs/e4_energy_guidance_train_v3
outputs/e4_energy_guidance_val_v3
```

验收条件：

```text
train root 文件数 = 15584
val root 文件数   = 1732
missing_stems = 0
extra_stems = 0
schema_missing_files = 0
bad_shape_files = 0
```

### 4. 重新启动 E4 Stage2

E4 Stage2 必须指向 v3 目录：

```text
--path_guided_root_dir outputs/e4_energy_guidance_train_v3
--path_scene_guided_root_dir outputs/e4_energy_guidance_train_v3
--val_root_dir outputs/e4_energy_guidance_val_v3
```

日志必须出现：

```text
Config: external_root_enabled=True, use_external_root=True
external_root_sources ... path_guided_root
```

日志不能出现：

```text
fallback
missing
path_root_missing_gt_fallback
path_scene_root_missing_gt_fallback
```

如果 E4 Stage2 仍出现 fallback，E4 不能算有效实验。

## E7 修改方案

### 1. 修正 GT root 导出脚本

目标脚本可以基于：

```text
scripts/export_gt_root_for_stage2_v2.py
```

建议另存为 v3 或直接修正后输出到 v3 目录。

关键修复点：

cache 里的 `motion_features` 已经是 normalized，不能再次执行：

```python
norm_feat = mr.normalize(feat_t)
output = mr.inverse(norm_feat, is_normalized=True, return_numpy=True)
```

正确写法应是：

```python
feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
output = mr.inverse(feat_t, is_normalized=True, return_numpy=True)

root_meter = output["smooth_root_pos"][0].astype(np.float32)
heading = output["global_root_heading"][0].astype(np.float32)

guided_root_5d_norm = feat[:, :5].astype(np.float32)
guided_root_5d_meter = np.concatenate(
    [root_meter[:, :3], heading],
    axis=-1,
).astype(np.float32)
target_path_xz = root_meter[:, [0, 2]].astype(np.float32)
```

然后保存：

```python
np.savez(
    str(output_dir / f"{source_id}.npz"),
    guided_root_5d_norm=guided_root_5d_norm,
    guided_root_5d_meter=guided_root_5d_meter,
    target_path_xz=target_path_xz,
    text=np.asarray(text),
    scene_name=np.asarray(scene_name),
    source_file=np.asarray(str(cache_file)),
)
```

### 2. 重新导出 E7 root

重新导出到干净目录：

```text
outputs/e7_gt_root_v3_train
outputs/e7_gt_root_v3_val
```

验收条件：

```text
train root 文件数 = 15584
val root 文件数   = 1732
missing_stems = 0
extra_stems = 0
schema_missing_files = 0
bad_shape_files = 0
```

### 3. 重新启动 E7 Stage2

E7 是 GT-root 条件实验。最安全做法是 Stage2 只从 v3 GT root 目录取 external root：

```text
--path_guided_root_dir outputs/e7_gt_root_v3_train
--path_scene_guided_root_dir outputs/e7_gt_root_v3_train
--val_root_dir outputs/e7_gt_root_v3_val
--root_mix_gt 0.0
--root_mix_path 1.0
--root_mix_scene 0.0
```

这样训练时所有 external root 都应来自：

```text
path_guided_root
```

日志不能出现：

```text
fallback
missing
path_root_missing_gt_fallback
path_scene_root_missing_gt_fallback
```

如果 E7 Stage2 仍出现 fallback，E7 不能算有效实验。

## 通用检查脚本

生成 root 后，应运行只读检查，确认数量、stem 覆盖、schema、shape。

检查逻辑应验证：

```text
expected train = 15584
expected val   = 1732
missing_stems = 0
extra_stems = 0
schema_missing_files = 0
bad_shape_files = 0
```

每个 `.npz` 至少应有：

```text
guided_root_5d_norm
guided_root_5d_meter
target_path_xz
text
scene_name
source_file
```

shape 要求：

```text
guided_root_5d_norm:  (T, 5)
guided_root_5d_meter: (T, 5)
target_path_xz:       (T, 2)
```

并且三者的 T 必须相同。

## Stage2 验收标准

E4/E7 Stage2 只有同时满足以下条件，才能算有效：

```text
external_root_enabled=True
use_external_root=True
external_root_sources 使用 path_guided_root 或 path_scene_guided_root
日志中没有 fallback
日志中没有 missing
loss/val_loss 是 finite
checkpoint 存在
```

注意：checkpoint 存在不等于 Stage2 有效。如果 external root fallback，则 checkpoint 也不能证明实验有效。

## Body Generation 验收标准

Stage2 checkpoint 有效之后，才可以做 body generation。

body generation 必须在日志中报告 root fix 误差，例如：

```text
root fix max error: <value>
```

或：

```text
max_error=<value>
```

验收条件：

```text
root fix max error < 1e-5
```

如果 body generation 没有报告 root fix max error，则不能判定 fixed-root correctness。

## Metrics 验收标准

最终必须存在：

```text
path_metrics.csv
scene_metrics.csv
```

如果有 generation output 但没有 metrics，状态只能是 `INCOMPLETE`。

## 推荐修复顺序

1. 先修 E7。
2. 修正 E7 GT root 导出中的二次 normalize。
3. 导出 `outputs/e7_gt_root_v3_train` 和 `outputs/e7_gt_root_v3_val`。
4. 检查 E7 v3 root count/schema/stem/shape。
5. 用 E7 v3 重新跑 Stage2，并确认无 fallback。
6. 再修 E4。
7. 修正 E4 energy root 生成逻辑。
8. 重新生成 `outputs/e4_energy_guidance_train_v3` 和 `outputs/e4_energy_guidance_val_v3`。
9. 检查 E4 v3 root count/schema/stem/shape。
10. 用 E4 v3 重新跑 Stage2，并确认无 fallback。
11. 对有效 Stage2 checkpoint 做 body generation。
12. 检查 root fix max error。
13. 计算 path/scene metrics。

## 最终判定

E4 只有同时满足以下条件，才能标记为有效：

```text
E4 root v3 train/val 完整
E4 Stage2 使用 v3 root
E4 Stage2 无 fallback/missing
E4 checkpoint 存在且 loss finite
E4 body generation 完成
E4 root fix max error < 1e-5
E4 path_metrics.csv 和 scene_metrics.csv 存在
```

E7 只有同时满足以下条件，才能标记为有效：

```text
E7 root v3 train/val 完整
E7 Stage2 使用 v3 root
E7 Stage2 无 fallback/missing
E7 checkpoint 存在且 loss finite
E7 body generation 完成
E7 root fix max error < 1e-5
E7 path_metrics.csv 和 scene_metrics.csv 存在
```

当前已有的旧 E4/E7 输出不能直接用于论文结论或最终对比。
