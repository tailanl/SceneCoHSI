# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""验证 root 只受 Stage1 (root_model) 影响，Stage2 (body_model) 不修改 root。

测试内容:
1. 固定 seed，完整 forward 得到预测结果
2. 只用 stage1 (root_model) 得到 root，与完整 forward 的 root 对比——应完全一致
3. 用完整 forward 的 root 喂给 body_model，与完整 forward 的 body 对比——应完全一致
4. 梯度隔离验证：root loss 只影响 root_model，body loss 不影响 root_model (训练模式下)
5. 量化 root 差异：初始 root vs stage1 预测 root
"""

import contextlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# copy from smoke_test.py
class FakeMotionRep:
    motion_rep_dim = 369
    global_root_dim = 5
    local_root_dim = 4
    body_slice = slice(5, 369)
    normalize_stats = None
    fps = 30

    class skeleton:
        nbjoints = 30
        foot_joint_idx = [7, 8, 10, 11]

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x

    def global_root_to_local_root(self, root_motion, normalized=True, lengths=None):
        B, T, _ = root_motion.shape
        local = torch.zeros(B, T, self.local_root_dim, device=root_motion.device)
        local[:, :, 0] = root_motion[:, :, 3]
        local[:, :, 1] = root_motion[:, :, 0]
        local[:, :, 2] = root_motion[:, :, 2]
        local[:, :, 3] = root_motion[:, :, 1]
        return local

    def create_conditions_from_constraints_batched(self, *args, **kwargs):
        return None, None


def make_forward_inputs(batch_size, max_frames, device, seed=42):
    """创建固定的测试输入，使用指定 seed 保证可复现。"""
    torch.manual_seed(seed)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    x = torch.randn(batch_size, max_frames, 369, generator=g, device=device)
    x_pad_mask = torch.ones(batch_size, max_frames, dtype=torch.bool, device=device)
    lengths = torch.full((batch_size,), max_frames, dtype=torch.long)
    if batch_size > 1:
        lengths[1] = max_frames * 3 // 4
        x_pad_mask[1, lengths[1]:] = False

    text_feat = torch.randn(batch_size, 1, 4096, generator=g, device=device)
    text_pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), generator=g, device=device)
    heading = torch.zeros(batch_size, device=device)
    motion_mask = torch.zeros_like(x)
    observed_motion = torch.zeros_like(x)

    scene_feat = torch.randn(batch_size, 32, 256, generator=g, device=device)
    scene_mask = torch.ones(batch_size, 32, dtype=torch.bool, device=device)

    return {
        "x": x,
        "x_pad_mask": x_pad_mask,
        "text_feat": text_feat,
        "text_feat_pad_mask": text_pad_mask,
        "timesteps": timesteps,
        "first_heading_angle": heading,
        "motion_mask": motion_mask,
        "observed_motion": observed_motion,
        "scene_feat": scene_feat,
        "scene_mask": scene_mask,
    }


def build_denoiser(motion_rep, device="cpu"):
    """构建与 smoke_test 相同的 TwostageDenoiser。"""
    from kimodo_sceneco.model.twostage_denoiser import TwostageDenoiser

    denoiser_kwargs = {
        "latent_dim": 256,
        "ff_size": 512,
        "num_layers": 2,
        "num_heads": 4,
        "activation": "gelu",
        "dropout": 0.1,
        "pe_dropout": 0.1,
        "norm_first": True,
        "use_text_mask": True,
        "llm_shape": [1, 4096],
        "input_first_heading_angle": True,
        "use_sceneco": True,
        "scene_feat_dim": 256,
        "sceneco_dropout": 0.1,
    }

    torch.manual_seed(42)
    denoiser = TwostageDenoiser(
        motion_rep=motion_rep,
        motion_mask_mode="concat",
        **denoiser_kwargs,
    )
    return denoiser.to(device)


# ========== TEST 1: root 仅由 stage1 决定 ==========

def test_root_equals_root_model_output(device="cpu"):
    """验证完整 forward 的 root 部分与单独调用 root_model 的结果完全一致。"""
    log.info("=" * 60)
    log.info("TEST 1: root == root_model 独立输出 (推理模式)")
    log.info("=" * 60)

    motion_rep = FakeMotionRep()
    denoiser = build_denoiser(motion_rep, device)
    denoiser.eval()

    B, T = 2, 60
    inputs = make_forward_inputs(B, T, device, seed=42)

    # ---- 完整 forward ----
    with torch.no_grad():
        output_full = denoiser(**inputs)
    root_full = output_full[:, :, :motion_rep.global_root_dim]  # [B, T, 5]
    body_full = output_full[:, :, motion_rep.global_root_dim:]  # [B, T, 364]

    # ---- 单独调用 root_model ----
    x = inputs["x"]
    motion_mask = inputs["motion_mask"]
    observed_motion = inputs["observed_motion"]
    x_masked = x * (1 - motion_mask) + observed_motion * motion_mask
    x_extended = torch.cat([x_masked, motion_mask], axis=-1)

    with torch.no_grad():
        root_stage1 = denoiser.root_model(
            x_extended,
            inputs["x_pad_mask"],
            inputs["text_feat"],
            inputs["text_feat_pad_mask"],
            inputs["timesteps"],
            inputs["first_heading_angle"],
            scene_feat=inputs["scene_feat"],
            scene_mask=inputs["scene_mask"],
        )  # [B, T, 5]

    root_diff = (root_full - root_stage1).abs().max().item()
    log.info(f"  root_full vs root_stage1 max diff: {root_diff:.10f}")
    assert root_diff < 1e-6, f"root 不一致! diff={root_diff}"
    log.info("  PASSED: root 完全由 root_model (stage1) 决定 ✓")

    return denoiser, motion_rep, inputs, root_stage1, body_full, x_extended


# ========== TEST 2: body 由 stage2 独立复现 ==========

def test_body_equals_body_model_with_full_root(denoiser, motion_rep, inputs,
                                                root_stage1, body_full):
    """用完整 forward 的 root 喂给 body_model，验证 body 输出一致。"""
    log.info("=" * 60)
    log.info("TEST 2: 用 root_stage1 -> local_root -> body_model 复现 body")
    log.info("=" * 60)

    x = inputs["x"]
    motion_mask = inputs["motion_mask"]
    observed_motion = inputs["observed_motion"]
    x_masked = x * (1 - motion_mask) + observed_motion * motion_mask

    lengths = inputs["x_pad_mask"].sum(-1)
    with torch.no_grad():
        root_motion_local = motion_rep.global_root_to_local_root(
            root_stage1, normalized=True, lengths=lengths,
        )

    body_x = x[..., motion_rep.body_slice]  # [B, T, 364]
    x_new = torch.cat([root_motion_local, body_x], axis=-1)
    x_new_extended = torch.cat([x_new, motion_mask], axis=-1)

    with torch.no_grad():
        body_stage2 = denoiser.body_model(
            x_new_extended,
            inputs["x_pad_mask"],
            inputs["text_feat"],
            inputs["text_feat_pad_mask"],
            inputs["timesteps"],
            inputs["first_heading_angle"],
            scene_feat=inputs["scene_feat"],
            scene_mask=inputs["scene_mask"],
        )

    body_diff = (body_full - body_stage2).abs().max().item()
    log.info(f"  body_full vs body_stage2 max diff: {body_diff:.10f}")
    assert body_diff < 1e-6, f"body 不一致! diff={body_diff}"
    log.info("  PASSED: body 可以完全由 root + body_model 复现 ✓")

    return root_motion_local


# ========== TEST 3: 量化 root 差异 (初始 root vs 预测 root) ==========

def test_root_difference_quantified(motion_rep, inputs, device="cpu"):
    """量化初始 root 与 stage1 预测 root 之间的差异。"""
    log.info("=" * 60)
    log.info("TEST 3: 量化初始 root 与 stage1 预测 root 的差异")
    log.info("=" * 60)

    x = inputs["x"]
    root_initial = x[:, :, :motion_rep.global_root_dim]  # 初始噪声中的 root

    # 重新构建 denoiser, 固定 seed 保证可复现
    torch.manual_seed(42)
    denoiser2 = build_denoiser(motion_rep, device)
    denoiser2.eval()

    with torch.no_grad():
        output = denoiser2(**inputs)
    root_pred = output[:, :, :motion_rep.global_root_dim]

    abs_diff = (root_pred - root_initial).abs()
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (root_initial.abs() + 1e-8)).mean().item()

    log.info(f"  初始 root 均值: {root_initial.mean().item():.6f}")
    log.info(f"  预测 root 均值: {root_pred.mean().item():.6f}")
    log.info(f"  平均绝对差异 (MAE): {mean_abs_diff:.6f}")
    log.info(f"  最大绝对差异: {max_abs_diff:.6f}")
    log.info(f"  相对差异 (mean): {rel_diff:.4f}")
    log.info(f"  root 确实被 stage1 显著改变了 (差异 > 0)")
    assert mean_abs_diff > 0, "root 完全没有变化？"

    log.info("  PASSED ✓")
    return root_initial, root_pred


# ========== TEST 4: 梯度隔离 (训练模式) ==========

def test_gradient_isolation(device="cpu"):
    """训练模式下: root loss 只影响 root_model; body loss 不影响 root_model。"""
    log.info("=" * 60)
    log.info("TEST 4: 梯度隔离——训练模式下 root/body 解耦")
    log.info("=" * 60)

    motion_rep = FakeMotionRep()
    torch.manual_seed(42)
    denoiser = build_denoiser(motion_rep, device)
    denoiser.train()  # 训练模式

    B, T = 2, 60
    inputs = make_forward_inputs(B, T, device, seed=99)

    # ===== 4a: 只对 root 部分计算 loss ====
    output = denoiser(**inputs)
    root_pred = output[:, :, :motion_rep.global_root_dim]
    # 构造一个假 target
    root_target = torch.randn_like(root_pred)
    mask_f = inputs["x_pad_mask"].unsqueeze(-1).float()
    root_loss = torch.nn.functional.mse_loss(
        root_pred * mask_f, root_target * mask_f,
    )
    denoiser.zero_grad()
    root_loss.backward(retain_graph=True)

    root_model_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in denoiser.root_model.parameters()
    )
    body_model_has_grad_from_root = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in denoiser.body_model.parameters()
    )

    log.info(f"  4a: root loss → root_model 有梯度: {root_model_has_grad}")
    log.info(f"  4a: root loss → body_model 有梯度: {body_model_has_grad_from_root}")
    assert root_model_has_grad, "root_model 应该收到 root loss 的梯度!"
    assert not body_model_has_grad_from_root, (
        "body_model 不应该从 root loss 收到梯度 (因为 detach)!"
    )

    # ===== 4b: 只对 body 部分计算 loss ====
    denoiser.zero_grad()
    output2 = denoiser(**inputs)
    body_pred = output2[:, :, motion_rep.global_root_dim:]
    body_target = torch.randn_like(body_pred)
    body_loss = torch.nn.functional.mse_loss(
        body_pred * mask_f, body_target * mask_f,
    )
    body_loss.backward()

    body_model_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in denoiser.body_model.parameters()
    )
    root_model_has_grad_from_body = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in denoiser.root_model.parameters()
    )
    log.info(f"  4b: body loss → body_model 有梯度: {body_model_has_grad}")
    log.info(f"  4b: body loss → root_model 有梯度: {root_model_has_grad_from_body}")
    assert body_model_has_grad, "body_model 应该收到 body loss 的梯度!"
    assert not root_model_has_grad_from_body, (
        "root_model 不应该从 body loss 收到梯度 (因为 detach)!"
    )

    log.info("  PASSED: 训练时 root 和 body 梯度完全解耦 ✓")


# ========== TEST 5: root 不会受 body_model 影响 ==========

def test_root_unchanged_by_body_model(device="cpu"):
    """构造两个不同 body_model 权重的 denoiser，验证 root 输出完全相同。"""
    log.info("=" * 60)
    log.info("TEST 5: 不同 body_model 权重下 root 输出不受影响")
    log.info("=" * 60)

    motion_rep = FakeMotionRep()

    # 构建 denoiser A: 固定 seed
    torch.manual_seed(42)
    denoiser_a = build_denoiser(motion_rep, device)
    denoiser_a.eval()

    # 构建 denoiser B: 相同 root_model 权重，但随机修改 body_model 权重
    torch.manual_seed(42)
    denoiser_b = build_denoiser(motion_rep, device)
    denoiser_b.eval()
    # 打乱 body_model 权重
    with torch.no_grad():
        for p in denoiser_b.body_model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    B, T = 2, 60
    inputs = make_forward_inputs(B, T, device, seed=123)

    with torch.no_grad():
        out_a = denoiser_a(**inputs)
        out_b = denoiser_b(**inputs)

    root_a = out_a[:, :, :motion_rep.global_root_dim]
    root_b = out_b[:, :, :motion_rep.global_root_dim]
    body_a = out_a[:, :, motion_rep.global_root_dim:]
    body_b = out_b[:, :, motion_rep.global_root_dim:]

    root_diff = (root_a - root_b).abs().max().item()
    body_diff = (body_a - body_b).abs().max().item()

    log.info(f"  root 差异 (应该为 0): {root_diff:.10f}")
    log.info(f"  body 差异 (应该 > 0): {body_diff:.6f}")
    assert root_diff < 1e-6, f"root 被 body_model 影响了! diff={root_diff}"
    assert body_diff > 0, "body 应该不同 (我们打乱了 body_model 权重)"
    log.info("  PASSED: root 完全不受 body_model 权重影响 ✓")


# ========== 主入口 ==========

def main():
    device = "cpu"
    log.info("=" * 60)
    log.info("Kimodo Two-Stage Root/Body Isolation Test")
    log.info("验证: root 只受 root_model (Stage1) 影响, body_model (Stage2) 不修改 root")
    log.info(f"Device: {device}")
    log.info("=" * 60)
    log.info("")

    # TEST 1: root == root_model 独立输出
    denoiser, motion_rep, inputs, root_stage1, body_full, x_extended = \
        test_root_equals_root_model_output(device)

    # TEST 2: body 可由 root + body_model 复现
    test_body_equals_body_model_with_full_root(
        denoiser, motion_rep, inputs, root_stage1, body_full,
    )

    # TEST 3: 量化 root 差异
    test_root_difference_quantified(motion_rep, inputs, device)

    # TEST 4: 梯度隔离
    test_gradient_isolation(device)

    # TEST 5: 不同 body_model 权重不影响 root
    test_root_unchanged_by_body_model(device)

    log.info("")
    log.info("=" * 60)
    log.info("ALL TESTS PASSED ✓")
    log.info("结论:")
    log.info("  1. root 完全由 root_model (Stage1) 决定，Stage2 不修改 root")
    log.info("  2. body_model 使用 root 作为条件，但不回传梯度给 root_model (训练时)")
    log.info("  3. root_model 和 body_model 在训练时梯度完全解耦")
    log.info("  4. root_model 参与 body 训练仅作为条件提供者 (detached)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
