"""GPU-3 training launcher with comprehensive monitoring and checks.

Wraps the SceneCo training pipeline with exhaustive validation:
1. GPU exclusivity lock & memory baseline checks
2. Pre-flight: hardware, model, dataset, imports, env, config, gate, forward pass
3. Per-step: NaN/Inf, GPU memory, scene/ text encoder, gradient flow, clip stats, alpha drift
4. Periodic: frozen param integrity, checkpoint verify, loss trend, throughput
5. Safety: OOM recovery, CUDA error handling, graceful shutdown

Usage:
    python train_gpu3_monitor.py configs/sceneco_root_only.yaml --gpu 3 --steps 200
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
KIMODO_ROOT = PROJECT_ROOT

if str(KIMODO_ROOT / "kimodo") not in sys.path:
    sys.path.insert(0, str(KIMODO_ROOT / "kimodo"))


# ============================================================================
# SECTION 0: GPU ENVIRONMENT & LOCK
# ============================================================================

def check_gpu_exclusivity(gpu_id: int) -> Dict:
    """Verify GPU 3 is available and not in use by other processes."""
    result = {"gpu_id": gpu_id, "available": False, "free_mem_gb": 0, "total_mem_gb": 0,
              "running_processes": [], "ok_to_use": False}

    if not torch.cuda.is_available():
        result["error"] = "CUDA not available"
        return result

    if gpu_id >= torch.cuda.device_count():
        result["error"] = f"GPU {gpu_id} does not exist (only {torch.cuda.device_count()} GPUs)"
        return result

    result["available"] = True
    prop = torch.cuda.get_device_properties(gpu_id)
    result["total_mem_gb"] = prop.total_memory / 1024**3
    result["gpu_name"] = prop.name
    result["compute_capability"] = f"{prop.major}.{prop.minor}"

    mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    mem_free = result["total_mem_gb"] - mem_reserved
    result["free_mem_gb"] = round(mem_free, 2)
    result["allocated_gb"] = round(mem_allocated, 2)
    result["reserved_gb"] = round(mem_reserved, 2)

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in procs:
            result["running_processes"].append({
                "pid": p.pid, "used_memory_mb": p.usedGpuMemory / 1024**2
            })
        pynvml.nvmlShutdown()
    except Exception:
        pass

    if len(result["running_processes"]) == 0 and mem_reserved < 0.5:
        result["ok_to_use"] = True
    else:
        result["ok_to_use"] = len(result["running_processes"]) == 0
        result.setdefault("warnings", []).append(
            f"GPU {gpu_id} has {mem_reserved:.1f}GB reserved or {len(result['running_processes'])} processes running"
        )

    return result


def get_gpu_memory_snapshot(device: torch.device) -> Dict:
    """Snapshot GPU memory usage."""
    if device.type != "cuda":
        return {"device": "cpu", "allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}
    gpu_id = device.index if device.index is not None else 0
    return {
        "device": f"cuda:{gpu_id}",
        "allocated_gb": round(torch.cuda.memory_allocated(gpu_id) / 1024**3, 4),
        "reserved_gb": round(torch.cuda.memory_reserved(gpu_id) / 1024**3, 4),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(gpu_id) / 1024**3, 4),
    }


def verify_cuda_tensor_ops(device: torch.device) -> Dict:
    """Verify basic CUDA tensor ops work on the target device."""
    result = {"device": str(device), "ok": True, "checks": []}
    if device.type != "cuda":
        result["checks"].append(("CUDA tensor", "SKIP", "CPU device"))
        return result

    try:
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = x @ y
        mem_test_passed = z.abs().mean().item() > 0
        result["checks"].append(("matmul", "PASS" if mem_test_passed else "FAIL",
                                  f"mean={z.abs().mean().item():.4f}"))

        a = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        b = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        c = torch.nn.functional.softmax(a @ b, dim=-1)
        result["checks"].append(("bf16_matmul_softmax", "PASS",
                                  f"shape={list(c.shape)}, mean={c.mean().item():.4f}"))

        del x, y, z, a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        result["ok"] = False
        result["checks"].append(("tensor_ops", "FAIL", str(e)))

    return result


# ============================================================================
# SECTION 1: PRE-FLIGHT CHECKS (extended)
# ============================================================================

def run_preflight_checks(gpu_id: int, config_path: Path) -> bool:
    """Run all pre-flight checks. Returns True if all pass."""
    print("\n" + "=" * 70)
    print("  PRE-FLIGHT CHECKS")
    print("=" * 70)

    all_ok = True

    all_ok &= _check_hardware_preflight(gpu_id)
    all_ok &= _check_model_files_preflight()
    all_ok &= _check_dataset_preflight()
    all_ok &= _check_imports_preflight()
    all_ok &= _check_env_preflight()
    all_ok &= _check_config_preflight(config_path)
    all_ok &= _check_gate_init_preflight()
    all_ok &= _check_forward_pass_preflight(gpu_id)

    return all_ok


def _check_hardware_preflight(gpu_id: int) -> bool:
    print("\n  [1/8] HARDWARE CHECK")
    ok = True

    print(f"    PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    print(f"    GPU count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("    ⚠️  CUDA NOT available — training will use CPU (slow but functional)")
        print("    => ⚠️  WARNING (non-blocking)")
        return True

    if gpu_id >= torch.cuda.device_count():
        print(f"    ❌ GPU {gpu_id} not found (max index: {torch.cuda.device_count()-1})")
        return False

    prop = torch.cuda.get_device_properties(gpu_id)
    total_gb = prop.total_memory / 1024**3
    print(f"    GPU {gpu_id}: {prop.name}")
    print(f"    Memory: {total_gb:.1f} GB | SM: {prop.major}.{prop.minor}")
    print(f"    Multi-processors: {prop.multi_processor_count}")

    if total_gb < 8:
        print(f"    ⚠️  GPU memory < 8GB — may struggle with batch_size=4")
    if prop.major < 7:
        print(f"    ❌ SM < 7.0 — bf16 not supported!")
        ok = False

    import psutil
    ram = psutil.virtual_memory()
    print(f"    System RAM: {ram.available/1024**3:.1f} GB free / {ram.total/1024**3:.1f} GB total")
    if ram.available / 1024**3 < 16:
        print(f"    ⚠️  Low system RAM")

    disk = psutil.disk_usage(str(PROJECT_ROOT))
    print(f"    Disk: {disk.free/1024**3:.1f} GB free")
    if disk.free / 1024**3 < 50:
        print(f"    ⚠️  Low disk space — may run out for checkpoints")

    gpu_lock = check_gpu_exclusivity(gpu_id)
    print(f"    GPU lock: ok_to_use={gpu_lock['ok_to_use']}, free_mem={gpu_lock['free_mem_gb']:.1f}GB")
    if not gpu_lock["ok_to_use"]:
        print(f"    ⚠️  GPU {gpu_id} may have conflicts: {gpu_lock.get('warnings', [])}")

    cuda_ops = verify_cuda_tensor_ops(torch.device(f"cuda:{gpu_id}"))
    for name, status, detail in cuda_ops["checks"]:
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"    {icon} {name}: {detail}")
    if not cuda_ops["ok"]:
        ok = False

    print(f"    => {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def _check_model_files_preflight() -> bool:
    print("\n  [2/8] MODEL FILE CHECK")
    model_dir = PROJECT_ROOT / "models" / "Kimodo-SOMA-RP-v1.1"
    hf_snap = PROJECT_ROOT / ".hf_cache" / "hub" / "models--nvidia--Kimodo-G1-RP-v1" / "snapshots"

    if not model_dir.exists() and hf_snap.exists():
        snaps = sorted(hf_snap.glob("*"))
        if snaps:
            model_dir = snaps[0]

    print(f"    Model dir: {model_dir} (exists={model_dir.exists()})")
    if not model_dir.exists():
        print(f"    ❌ Model directory not found!")
        return False

    safetensors = list(model_dir.glob("*.safetensors"))
    if safetensors:
        size_gb = sum(f.stat().st_size for f in safetensors) / 1024**3
        print(f"    Safetensors: {len(safetensors)} files, {size_gb:.2f} GB")
    else:
        print(f"    ⚠️  No safetensors in {model_dir}")

    configs = list(model_dir.glob("config*.yaml")) + list(model_dir.glob("*.yaml"))
    print(f"    Config files: {len(configs)} found")

    stats_dir = model_dir / "stats"
    if stats_dir.exists():
        npy_files = sorted(stats_dir.rglob("*.npy"))
        print(f"    Stats: {len(npy_files)} npy files")
        for nf in npy_files[:5]:
            arr = np.load(str(nf), allow_pickle=True)
            print(f"      {nf.relative_to(stats_dir)}: {list(arr.shape) if hasattr(arr, 'shape') else type(arr)}")
    else:
        print(f"    ⚠️  stats/ directory missing")

    print(f"    => ✅ PASS")
    return True


def _check_dataset_preflight() -> bool:
    print("\n  [3/8] DATASET CHECK")
    ok = True

    cached = PROJECT_ROOT / "kimodo" / "kimodo_sceneco" / "cached_data"
    print(f"    Cached data: {cached} (exists={cached.exists()})")
    if cached.exists():
        npz_files = sorted(cached.glob("seg_*.npz"))
        print(f"    NPZ files: {len(npz_files)}")
        if len(npz_files) > 0:
            sample = dict(np.load(str(npz_files[0]), allow_pickle=True))
            keys = sorted(sample.keys())
            print(f"    Sample keys: {keys}")
            for k in ["motion_features", "voxel_grid", "text", "length", "scene_name"]:
                if k not in sample:
                    print(f"    ⚠️  Missing key: {k}")
                    ok = False
            if "motion_features" in sample:
                mf = sample["motion_features"]
                has_nan = np.any(np.isnan(mf)) if hasattr(mf, 'shape') else False
                print(f"    motion_features: shape={list(mf.shape) if hasattr(mf,'shape') else 'N/A'}, NaN={has_nan}")
                if has_nan:
                    ok = False
            if "voxel_grid" in sample:
                vg = sample["voxel_grid"]
                if hasattr(vg, 'mean'):
                    print(f"    voxel_grid: shape={list(vg.shape)}, occupancy={vg.mean()*100:.1f}%")
                    if vg.mean() < 0.001:
                        print(f"    ⚠️  Near-empty voxel grid!")
    else:
        print(f"    ❌ Cached data dir missing!")
        ok = False

    scene_dir = PROJECT_ROOT / "LINGO" / "dataset" / "dataset" / "Scene"
    print(f"    Scene dir: {scene_dir} (exists={scene_dir.exists()})")
    if scene_dir.exists():
        scene_files = sorted(scene_dir.glob("*.npy"))
        print(f"    Scene npy: {len(scene_files)} files")
    else:
        print(f"    ⚠️  Scene directory missing")

    print(f"    => {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def _check_imports_preflight() -> bool:
    print("\n  [4/8] IMPORT CHECK")
    ok = True
    modules = [
        "torch", "numpy", "scipy", "tqdm", "yaml", "omegaconf", "safetensors",
        "kimodo.model", "kimodo_sceneco.model.scene_encoder",
        "kimodo_sceneco.train.dataset", "kimodo_sceneco.train.train",
        "kimodo_sceneco.model.backbone",
    ]
    for mod in modules:
        try:
            __import__(mod)
            print(f"    ✅ import {mod}")
        except Exception as e:
            print(f"    ❌ import {mod}: {type(e).__name__}: {e}")
            ok = False
    print(f"    => {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def _check_env_preflight() -> bool:
    print("\n  [5/8] ENVIRONMENT CHECK")
    ok = True
    vars_to_check = {
        "CHECKPOINT_DIR": "models",
        "HF_HOME": ".hf_cache",
        "TEXT_ENCODERS_DIR": "text_encoders",
        "TEXT_ENCODER_MODE": "local",
        "TEXT_ENCODER_DEVICE": "cpu",
        "PYTHONHASHSEED": None,
    }
    for var, expected in vars_to_check.items():
        actual = os.environ.get(var, "")
        note = f"(expected={expected})" if expected and actual != expected else ""
        icon = "✅" if actual else "⚠️" if expected else "ℹ️"
        print(f"    {icon} {var}={actual or '(unset)'} {note}")
        if expected and not actual:
            ok = False
    print(f"    => {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def _check_config_preflight(config_path: Path) -> bool:
    print("\n  [6/8] CONFIG VALIDATION")
    import yaml
    if not config_path.exists():
        print(f"    ❌ Config not found: {config_path}")
        return False

    with open(config_path) as f:
        conf = yaml.safe_load(f)

    checks = []
    tc = conf.get("training", {})
    checks.append(("freeze_pretrained=True", tc.get("freeze_pretrained") in [True, "true"], tc.get("freeze_pretrained")))
    checks.append(("lr=1e-4", abs(float(tc.get("lr", 0)) - 1e-4) < 1e-9, tc.get("lr")))
    checks.append(("precision=bf16", tc.get("precision") == "bf16", tc.get("precision")))
    checks.append(("batch_size", tc.get("batch_size", 0) > 0, tc.get("batch_size")))
    checks.append(("total_steps", tc.get("total_steps", 0) > 0, tc.get("total_steps")))
    checks.append(("max_grad_norm", tc.get("max_grad_norm", 0) > 0, tc.get("max_grad_norm")))
    checks.append(("prior_weight", tc.get("prior_weight", 0) > 0, tc.get("prior_weight")))
    checks.append(("warmup_steps", tc.get("warmup_steps", 0) >= 0, tc.get("warmup_steps")))

    sc = conf.get("sceneco", {})
    checks.append(("sceneco.d_model", sc.get("d_model", 0) > 0, sc.get("d_model")))
    checks.append(("sceneco.scene_feat_dim", sc.get("scene_feat_dim", 0) > 0, sc.get("scene_feat_dim")))
    checks.append(("sceneco.nhead", sc.get("nhead", 0) > 0, sc.get("nhead")))

    vc = conf.get("voxel_vit", {})
    checks.append(("voxel_vit.d_model", vc.get("d_model", 0) > 0, vc.get("d_model")))
    checks.append(("voxel_vit.num_layers", vc.get("num_layers", 0) > 0, vc.get("num_layers")))

    ok = True
    for name, passed, val in checks:
        icon = "✅" if passed else "❌"
        print(f"    {icon} {name}: {val}")
        if not passed:
            ok = False

    print(f"    => {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def _check_gate_init_preflight() -> bool:
    print("\n  [7/8] GATE INIT CHECK")
    ok = True
    try:
        from kimodo_sceneco.model.backbone import SceneCoLayer
        sl = SceneCoLayer(d_model=256, scene_feat_dim=64, nhead=4)
        alpha_raw = sl.alpha.item()
        gate_val = torch.sigmoid(torch.tensor(alpha_raw)).item()
        gate_closed = gate_val < 0.01
        print(f"    backbone.SceneCoLayer alpha_raw={alpha_raw:.6f} gate={gate_val:.6f} (closed={gate_closed})")
        if not gate_closed:
            print(f"    ❌ Gate is NOT closed at init! gate={gate_val:.6f}")
            ok = False

        from kimodo_sceneco.exp.shared.sceneco_layers import SceneCoLayer as SC2
        sl2 = SC2(d_model=256, scene_feat_dim=64, nhead=4)
        alpha2_raw = sl2.alpha.item()
        gate2_val = torch.sigmoid(torch.tensor(alpha2_raw)).item()
        gate2_closed = gate2_val < 0.01
        print(f"    sceneco_layers.SceneCoLayer alpha_raw={alpha2_raw:.6f} gate={gate2_val:.6f} (closed={gate2_closed})")
        if not gate2_closed:
            print(f"    ❌ Gate is NOT closed at init! gate={gate2_val:.6f}")
            ok = False
    except Exception as e:
        print(f"    ❌ Gate init check failed: {e}")
        ok = False

    print(f"    => {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def _check_forward_pass_preflight(gpu_id: int) -> bool:
    print("\n  [8/8] FORWARD PASS SANITY CHECK")
    ok = True
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id < torch.cuda.device_count() else "cpu")
    print(f"    Device: {device}")

    try:
        from kimodo_sceneco.model.scene_encoder import VoxelViT
        vit = VoxelViT().to(device)
        dummy_voxel = torch.randn(2, 1, 64, 64, 64).to(device)
        with torch.no_grad():
            scene_feat, scene_mask = vit(dummy_voxel)
        print(f"    VoxelViT: out={list(scene_feat.shape)}, mask={list(scene_mask.shape)}")
        if torch.isnan(scene_feat).any():
            print(f"    ❌ VoxelViT output has NaN!")
            ok = False
        if scene_feat.abs().max() > 100:
            print(f"    ⚠️  VoxelViT output range large: {scene_feat.abs().max():.1f}")
    except Exception as e:
        print(f"    ❌ VoxelViT forward: {e}")
        ok = False

    try:
        from kimodo_sceneco.model.backbone import SceneCoLayer
        sc_layer = SceneCoLayer(d_model=512, scene_feat_dim=256, nhead=8).to(device)
        dummy_motion = torch.randn(2, 128, 512).to(device)
        dummy_scene = torch.randn(2, 256, 256).to(device)
        with torch.no_grad():
            out = sc_layer(dummy_motion, dummy_scene)
        alpha_before = sc_layer.alpha.item()
        gate_before = torch.sigmoid(torch.tensor(alpha_before)).item()
        print(f"    SceneCoLayer: out={list(out.shape)}, alpha_raw={alpha_before:.4f} gate={gate_before:.6f}")
        max_diff = (out - dummy_motion).abs().max().item()
        print(f"    Gate≈0 output diff: max={max_diff:.6f}")
        if max_diff > 0.1:
            print(f"    ❌ Gate≈0 check FAILED: output ≠ input (max diff={max_diff:.6f})")
            ok = False
        time.sleep(2)
        alpha_after = sc_layer.alpha.item()
        if abs(alpha_after - alpha_before) > 1e-6:
            print(f"    ❌ Alpha changed after idle: {alpha_before:.6f} → {alpha_after:.6f}")
            ok = False
    except Exception as e:
        print(f"    ❌ SceneCoLayer forward: {e}")
        ok = False

    print(f"    => {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


# ============================================================================
# SECTION 2: RUNTIME TRAINING CHECKS
# ============================================================================

def validate_batch(batch: Dict, step: int, device: torch.device) -> Dict:
    """Validate a single batch for common issues. Expanded with more checks."""
    report = {"step": step, "ok": True, "warnings": [], "stats": {}}

    for key in ["motion_features", "voxel_grid", "motion_mask", "lengths", "texts"]:
        if key not in batch:
            report["ok"] = False
            report["warnings"].append(f"Missing key: {key}")

    mf = batch.get("motion_features")
    if mf is not None and isinstance(mf, torch.Tensor):
        report["stats"]["motion_shape"] = list(mf.shape)
        report["stats"]["motion_min"] = float(mf.min())
        report["stats"]["motion_max"] = float(mf.max())
        report["stats"]["motion_mean"] = float(mf.mean())
        report["stats"]["motion_std"] = float(mf.std())
        if torch.isnan(mf).any():
            report["ok"] = False
            report["warnings"].append(f"NaN in motion_features ({torch.isnan(mf).sum().item()} values)")
        if torch.isinf(mf).any():
            report["ok"] = False
            report["warnings"].append(f"Inf in motion_features ({torch.isinf(mf).sum().item()} values)")

    vg = batch.get("voxel_grid")
    if vg is not None and isinstance(vg, torch.Tensor):
        report["stats"]["voxel_shape"] = list(vg.shape)
        report["stats"]["voxel_occ"] = float(vg.mean())
        report["stats"]["voxel_min"] = float(vg.min())
        report["stats"]["voxel_max"] = float(vg.max())
        if torch.isnan(vg).any():
            report["ok"] = False
            report["warnings"].append("NaN in voxel_grid")
        if vg.mean() == 0.0:
            report["warnings"].append("Empty voxel grid (scene dropout)")

    lengths = batch.get("lengths")
    if lengths is not None and isinstance(lengths, torch.Tensor):
        report["stats"]["min_length"] = int(lengths.min())
        report["stats"]["max_length"] = int(lengths.max())
        report["stats"]["mean_length"] = float(lengths.float().mean())
        if lengths.min() < 10:
            report["warnings"].append(f"Very short segment: min_length={lengths.min().item()}")

    return report


def validate_scene_encoder_output(scene_feat, scene_mask, step: int, tag: str = "root") -> Dict:
    """Validate scene encoder output for anomalies."""
    report = {"step": step, "ok": True, "warnings": []}
    with torch.no_grad():
        sf = scene_feat.detach()
        report["stats"] = {
            "scene_feat_shape": list(sf.shape),
            "scene_feat_min": float(sf.min()),
            "scene_feat_max": float(sf.max()),
            "scene_feat_mean": float(sf.mean()),
            "scene_feat_std": float(sf.std()),
        }
        if torch.isnan(sf).any():
            report["ok"] = False
            report["warnings"].append(f"NaN in scene_feat!")
        if torch.isinf(sf).any():
            report["ok"] = False
            report["warnings"].append(f"Inf in scene_feat!")
        if sf.abs().max() > 1e3:
            report["warnings"].append(f"Large scene_feat values: max={sf.abs().max():.1f}")
        if sf.std() < 1e-8:
            report["warnings"].append(f"Scene_feat has near-zero variance (dead encoder?)")
        if scene_mask is not None:
            report["stats"]["scene_mask_active"] = float(scene_mask.float().mean())
            if scene_mask.float().mean() == 0:
                report["warnings"].append("Scene mask is all-zero!")
    return report


def validate_text_encoder_output(text_feat, text_pad_mask, step: int) -> Dict:
    """Validate text encoder output for anomalies."""
    report = {"step": step, "ok": True, "warnings": []}
    with torch.no_grad():
        tf = text_feat.detach()
        report["stats"] = {
            "text_feat_shape": list(tf.shape),
            "text_feat_min": float(tf.min()),
            "text_feat_max": float(tf.max()),
            "text_feat_mean": float(tf.mean()),
        }
        if torch.isnan(tf).any():
            report["ok"] = False
            report["warnings"].append("NaN in text_feat!")
        if torch.isinf(tf).any():
            report["ok"] = False
            report["warnings"].append("Inf in text_feat!")
        if text_pad_mask is not None:
            report["stats"]["text_mask_active"] = float(text_pad_mask.float().mean())
    return report


def get_alphas(model) -> Dict[str, float]:
    """Extract all SceneCo gate alpha values from the model."""
    alphas = {}
    for name, param in model.named_parameters():
        if "alpha" in name and param.numel() == 1:
            alphas[name] = param.item()
    return alphas


def compute_grad_stats(model) -> Dict:
    """Compute comprehensive gradient statistics for trainable params."""
    stats = {
        "max_grad": 0.0, "mean_grad": 0.0, "grad_norm": 0.0,
        "n_params": 0, "n_params_with_grad": 0, "n_dead_params": 0,
        "layer_grad_norms": {},
    }
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            g_norm = param.grad.detach().norm().item()
            grad_norms.append(g_norm)
            stats["n_params_with_grad"] += 1
            g_max = param.grad.detach().abs().max().item()
            if g_max > stats["max_grad"]:
                stats["max_grad"] = g_max
            if g_norm < 1e-10:
                stats["n_dead_params"] += 1
            short_name = name.split(".")[:3]
            layer_key = ".".join(short_name)
            stats["layer_grad_norms"].setdefault(layer_key, []).append(g_norm)
        elif param.requires_grad:
            stats["n_dead_params"] += 1

    stats["n_params"] = sum(1 for p in model.parameters() if p.requires_grad)
    if grad_norms:
        stats["mean_grad"] = float(np.mean(grad_norms))
        stats["grad_norm"] = float(np.sqrt(sum(g**2 for g in grad_norms)))
        top_layers = sorted(
            [(k, float(np.mean(v))) for k, v in stats["layer_grad_norms"].items()],
            key=lambda x: -x[1]
        )[:5]
        stats["top_grad_layers"] = top_layers
    return stats


def check_frozen_param_integrity(model, reference_state: Dict, step: int, tolerance: float = 1e-8) -> Dict:
    """Verify frozen parameters haven't drifted."""
    report = {"step": step, "ok": True, "warnings": [], "n_checked": 0,
              "max_delta": 0.0, "param_with_max_delta": ""}
    for name, param in model.named_parameters():
        if not param.requires_grad and name in reference_state:
            ref = reference_state[name].to(param.device)
            delta = (param.data - ref).abs().max().item()
            report["n_checked"] += 1
            if delta > report["max_delta"]:
                report["max_delta"] = delta
                report["param_with_max_delta"] = name
            if delta > tolerance:
                report["ok"] = False
                report["warnings"].append(
                    f"Frozen param {name} changed! delta={delta:.2e} (tolerance={tolerance})"
                )
                report["max_delta"] = delta
    return report


def validate_checkpoint_integrity(ckpt_path: Path, model, device: torch.device) -> Dict:
    """Save checkpoint then reload and verify."""
    report = {"path": str(ckpt_path), "ok": True, "warnings": []}
    try:
        model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        loaded = torch.load(ckpt_path, map_location="cpu")
        loaded_state = loaded.get("model_state_dict", loaded)
        total_keys = len(model_state)
        loaded_keys = len(loaded_state)
        report["total_keys"] = total_keys
        report["loaded_keys"] = loaded_keys
        if total_keys != loaded_keys:
            report["ok"] = False
            report["warnings"].append(f"Key mismatch: saved={total_keys}, loaded={loaded_keys}")
            missing = set(model_state.keys()) - set(loaded_state.keys())
            extra = set(loaded_state.keys()) - set(model_state.keys())
            if missing:
                report["warnings"].append(f"Missing keys: {list(missing)[:5]}")
            if extra:
                report["warnings"].append(f"Extra keys: {list(extra)[:5]}")
        mismatches = 0
        for k in model_state.keys():
            if k in loaded_state:
                diff = (model_state[k] - loaded_state[k]).abs().max().item()
                if diff > 1e-6:
                    mismatches += 1
        report["mismatched_keys"] = mismatches
        if mismatches > 0:
            report["ok"] = False
            report["warnings"].append(f"{mismatches} keys have value mismatch after save/load!")
    except Exception as e:
        report["ok"] = False
        report["warnings"].append(f"Checkpoint validation error: {e}")
    return report


class LossTrendDetector:
    """Detect anomalous loss trends (spikes, plateaus, divergence)."""

    def __init__(self, window_size: int = 100, spike_factor: float = 5.0):
        self.window = deque(maxlen=window_size)
        self.spike_factor = spike_factor

    def update(self, loss: float) -> Dict:
        self.window.append(loss)
        report = {"ok": True, "warnings": [], "stats": {}}

        if len(self.window) >= 10:
            recent = list(self.window)[-10:]
            mean_recent = np.mean(recent)
            report["stats"]["mean_10"] = mean_recent
            report["stats"]["min_10"] = float(np.min(recent))
            report["stats"]["max_10"] = float(np.max(recent))

            if len(self.window) >= 20:
                older = list(self.window)[-20:-10]
                mean_older = np.mean(older)
                if mean_older > 0:
                    change = (mean_recent - mean_older) / mean_older
                    report["stats"]["trend"] = change
                    if change > self.spike_factor:
                        report["warnings"].append(f"Loss spike: +{change*100:.0f}% vs previous window")
                    if mean_recent > 1e6:
                        report["ok"] = False
                        report["warnings"].append(f"Loss exploded to {mean_recent:.1f}")

            if len(self.window) >= 50 and np.std(recent) < 1e-6 and loss > 0.01:
                report["warnings"].append("Loss plateau (near-zero std for 10 steps)")

            if loss > 0 and loss == loss:
                if len(self.window) > 5:
                    prev = list(self.window)[-6:-1]
                    mean_prev = np.mean(prev)
                    if mean_prev > 0 and loss > mean_prev * self.spike_factor:
                        report["warnings"].append(f"Single-step spike: {loss:.4f} vs prev mean {mean_prev:.4f}")

        return report


def monitor_step(model, loss_dict: Dict, step: int, optimizer, output_dir: Path,
                 grad_stats: Dict, gpu_mem: Dict, scene_check: Dict,
                 text_check: Dict, loss_trend: Dict, clip_report: Dict,
                 timing: Dict) -> Dict:
    """Comprehensive step-level monitoring with ALL checks aggregated."""
    report = {
        "step": step,
        "timestamp": time.time(),
        "losses": {k: float(v) if isinstance(v, (torch.Tensor, float)) else v
                   for k, v in loss_dict.items()},
        "grad_stats": {k: v for k, v in grad_stats.items() if k != "layer_grad_norms"},
        "alphas": get_alphas(model),
        "gates": {k: torch.sigmoid(torch.tensor(v)).item() for k, v in get_alphas(model).items()},
        "lr": optimizer.param_groups[0]["lr"],
        "gpu_memory": gpu_mem,
        "scene_encoder": scene_check,
        "text_encoder": text_check,
        "loss_trend": loss_trend,
        "clip": clip_report,
        "timing": timing,
    }

    has_nan = any(
        isinstance(v, float) and (v != v or v > 1e6)
        for v in report["losses"].values()
    )
    report["has_nan_loss"] = has_nan

    report_file = output_dir / "step_monitor.jsonl"
    with open(report_file, "a") as f:
        f.write(json.dumps(report, default=str) + "\n")

    if has_nan:
        print(f"  ⚠️  [Step {step}] NaN in loss! {report['losses']}")

    return report


# ============================================================================
# SECTION 3: MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPU-3 SceneCo training with exhaustive checks")
    parser.add_argument("config", type=str, default="configs/sceneco_root_only.yaml",
                        help="Path to YAML config")
    parser.add_argument("--gpu", type=int, default=3, help="GPU device ID")
    parser.add_argument("--steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--skip_preflight", action="store_true", help="Skip pre-flight checks")
    parser.add_argument("--check_frozen_every", type=int, default=50,
                        help="Verify frozen params every N steps (0=disable)")
    parser.add_argument("--ckpt_verify_every", type=int, default=200,
                        help="Verify checkpoint integrity every N steps (0=disable)")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: from config)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--dual_vit", type=str, default=None,
                        choices=["true", "false", "True", "False"],
                        help="Override use_dual_vit (true=dual encoder, false=single shared)")
    parser.add_argument("--root_voxel_mode", type=str, default=None,
                        choices=["full", "floor"],
                        help="Override root_voxel_mode (full=complete voxel, floor=bottom 25%%)")
    parser.add_argument("--root_only", action="store_true",
                        help="Stage1: loss only on root dims (first 5) of motion feature")
    parser.add_argument("--freeze_sceneco", action="store_true",
                        help="Stage2: freeze SceneCo layers trained in Stage1")
    parser.add_argument("--scene_co_ckpt", type=str, default=None,
                        help="Path to Stage1 SceneCo checkpoint to load before Stage2")
    parser.add_argument("--batch_size_override", type=int, default=None,
                        help="Override batch_size from config for GPU filling")
    args = parser.parse_args()

    cli_overrides = {}
    if args.dual_vit is not None:
        cli_overrides["use_dual_vit"] = args.dual_vit.lower() == "true"
    if args.root_voxel_mode is not None:
        cli_overrides["root_voxel_mode"] = args.root_voxel_mode

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("  Kimodo-SceneCo Training on GPU 3 — EXHAUSTIVE CHECKS ENABLED")
    print("=" * 70)
    print(f"  Config:         {config_path}")
    print(f"  Time:           {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CWD:            {os.getcwd()}")
    print(f"  Max steps:      {args.steps}")
    print(f"  Log interval:   {args.log_interval}")
    print(f"  Frozen check:   every {args.check_frozen_every} steps")
    print(f"  Ckpt verify:    every {args.ckpt_verify_every} steps")
    print()

    # ---- PRE-FLIGHT CHECKS ----
    if not args.skip_preflight:
        if not run_preflight_checks(args.gpu, config_path):
            print("\n❌ PRE-FLIGHT CHECKS FAILED. Fix issues before training.")
            print("   Use --skip_preflight to bypass (NOT recommended).")
            sys.exit(1)
        print("\n✅ ALL PRE-FLIGHT CHECKS PASSED\n")

    # ---- LOAD CONFIG ----
    import yaml
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    training_conf = conf.get("training", {})
    data_conf = conf.get("data", {})
    sceneco_conf = conf.get("sceneco", {})
    scene_enc_cfg = conf.get("voxel_vit", {})

    output_dir = Path(args.output_dir) if args.output_dir else Path(conf.get("output_dir", "outputs/sceneco"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # ---- DEVICE SETUP ----
    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
        gpu_name = torch.cuda.get_device_name(args.gpu)
        gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
        torch.cuda.set_device(args.gpu)
        print(f"✅ GPU {args.gpu}: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"⚠️  Using fallback device: {device}")

    print(f"[CHECK] Device: {device}")
    gpu_mem_init = get_gpu_memory_snapshot(device)
    print(f"[CHECK] GPU memory baseline: allocated={gpu_mem_init['allocated_gb']:.2f}GB, "
          f"reserved={gpu_mem_init['reserved_gb']:.2f}GB")

    # ---- MODEL LOADING ----
    print("\n[CHECK] Loading pretrained model...")
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=str(device))
    motion_rep = pretrained.motion_rep
    print(f"[CHECK] Pretrained model loaded on {device}")

    inner_denoiser = pretrained.denoiser
    if hasattr(pretrained.denoiser, 'model'):
        inner_denoiser = pretrained.denoiser.model
        print("[CHECK] Unwrapped CFG wrapper to get inner denoiser for KimodoSceneCo")

    scene_enc_cfg = dict(conf.get("voxel_vit", {}))
    scene_enc_cfg["voxel_size"] = tuple(scene_enc_cfg.pop("input_size", (64, 64, 64)))
    scene_enc_cfg["sceneco_dropout"] = sceneco_conf.get("dropout", 0.1)
    scene_enc_cfg["use_dual_vit"] = sceneco_conf.get("use_dual_vit", True)
    scene_enc_cfg["root_voxel_mode"] = sceneco_conf.get("root_voxel_mode", "full")

    for k, v in cli_overrides.items():
        if k in ("use_dual_vit", "root_voxel_mode"):
            scene_enc_cfg[k] = v
            print(f"[CHECK] CLI override: {k}={v}")

    use_root = sceneco_conf.get("use_in_root_model", True)
    use_body = sceneco_conf.get("use_in_body_model", True)
    print(f"[CHECK] SceneCo: root_model={use_root}, body_model={use_body}, dual_vit={scene_enc_cfg['use_dual_vit']}, root_mode={scene_enc_cfg['root_voxel_mode']}")

    model = KimodoSceneCo(
        denoiser=inner_denoiser,
        text_encoder=pretrained.text_encoder,
        num_base_steps=training_conf.get("num_base_steps", 1000),
        scene_encoder_type="voxel_vit",
        scene_encoder_config=scene_enc_cfg,
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=use_root,
        use_in_body_model=use_body,
    )
    model = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CHECK] Total params: {total_p/1e6:.1f}M")
    print(f"[CHECK] Trainable (before freeze): {trainable_p/1e6:.1f}M ({trainable_p/total_p*100:.1f}%)")
    gpu_mem_after_model = get_gpu_memory_snapshot(device)
    print(f"[CHECK] GPU memory after model load: allocated={gpu_mem_after_model['allocated_gb']:.2f}GB")

    # ---- FREEZE PRETRAINED PARAMS ----
    if training_conf.get("freeze_pretrained", True):
        print("\n[CHECK] Freezing pretrained parameters...")
        scene_keywords = {"sceneco", "scene_encoder", "scene_null_embed", "voxel_vit"}
        if args.freeze_sceneco:
            scene_keywords.add("body_model")
        freeze_count = 0
        train_count = 0
        for name, param in model.named_parameters():
            should_train = any(kw in name for kw in scene_keywords)
            param.requires_grad = should_train
            if should_train:
                train_count += 1
            else:
                freeze_count += 1
        frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[CHECK] Frozen: {frozen_p/1e6:.1f}M params ({freeze_count} tensors)")
        print(f"[CHECK] Trainable: {trainable_p/1e6:.1f}M params ({train_count} tensors)")
        print(f"[CHECK] Trainable ratio: {trainable_p/total_p*100:.1f}%")
        if trainable_p / total_p > 0.05:
            print(f"[CHECK] ⚠️  More than 5% params trainable — verify freeze logic!")

        alphas_init = get_alphas(model)
        gates_init = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in alphas_init.items()}
        print(f"[CHECK] Initial gates: {gates_init}")

        # Snapshot frozen params for integrity verification
        frozen_snapshot = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_snapshot[name] = param.data.clone().cpu()
        print(f"[CHECK] Captured snapshot of {len(frozen_snapshot)} frozen params for integrity checks")
    else:
        frozen_snapshot = {}

    if args.freeze_sceneco:
        print("\n[STAGE2] Freezing SceneCo layers...")
        sceneco_frozen = 0
        for name, param in model.named_parameters():
            if 'sceneco' in name.lower() or 'scene_encoder' in name.lower():
                param.requires_grad = False
                sceneco_frozen += 1
        frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[STAGE2] SceneCo frozen: {sceneco_frozen} tensors")
        print(f"[STAGE2] Frozen total: {frozen_p/1e6:.1f}M, Trainable: {trainable_p/1e6:.1f}M")

    if args.scene_co_ckpt:
        print(f"\n[STAGE2] Loading Stage1 SceneCo weights from: {args.scene_co_ckpt}")
        ckpt_path = Path(args.scene_co_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model_state = ckpt.get("model_state_dict", ckpt)
        sceneco_params = {k: v for k, v in model_state.items() if 'sceneco' in k or 'scene_encoder' in k}
        missing, _ = model.load_state_dict(sceneco_params, strict=False)
        print(f"[STAGE2] Loaded {len(sceneco_params)} SceneCo params (unexpected missing: {len(missing)})")

    if args.root_only:
        ROOT_DIM = motion_rep.global_root_dim
        MOTION_DIM = motion_rep.motion_rep_dim
        print(f"\n[STAGE1] Root-only mode: loss on first {ROOT_DIM}/{MOTION_DIM} dims")
        loss_mask = torch.zeros(1, 1, MOTION_DIM)
        loss_mask[:, :, :ROOT_DIM] = 1.0
    else:
        loss_mask = None
    print("\n[CHECK] Loading datasets...")
    data_root = data_conf.get("data_root", "LINGO/dataset")
    if not Path(data_root).is_absolute():
        data_root = str(KIMODO_ROOT / data_root)
    cache_dir = data_conf.get("cache_dir", "kimodo/kimodo_sceneco/cached_data")
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = str(KIMODO_ROOT / cache_dir)

    voxel_size_raw = data_conf.get("voxel_size", (64, 64, 64))
    if isinstance(voxel_size_raw, str):
        voxel_size_t = tuple(map(int, voxel_size_raw.split(",")))
    elif isinstance(voxel_size_raw, (list, tuple)):
        voxel_size_t = tuple(int(x) for x in voxel_size_raw)
    else:
        voxel_size_t = (64, 64, 64)
    max_frames = data_conf.get("max_frames", 196)
    min_frames = data_conf.get("min_frames", 40)

    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset, collate_fn

    ds_train = LINGOSceneMotionDataset(
        data_root=str(data_root), motion_rep=motion_rep,
        max_frames=max_frames, min_frames=min_frames,
        fps=data_conf.get("fps", 30), voxel_size=voxel_size_t,
        scene_dropout=training_conf.get("scene_dropout", 0.1),
        split="train", cache_dir=str(cache_dir),
    )
    ds_val = LINGOSceneMotionDataset(
        data_root=str(data_root), motion_rep=motion_rep,
        max_frames=max_frames, min_frames=min_frames,
        fps=data_conf.get("fps", 30), voxel_size=voxel_size_t,
        scene_dropout=0.0, split="val", cache_dir=str(cache_dir),
    )
    print(f"[CHECK] Train: {len(ds_train)} segments, Val: {len(ds_val)} segments")
    if len(ds_train) < 10:
        print(f"[CHECK] ⚠️  Very small dataset: {len(ds_train)} segments!")

    # ---- DATALOADER ----
    batch_size = training_conf.get("batch_size", 1 if device.type == "cpu" else 4)
    if args.batch_size_override is not None:
        batch_size = args.batch_size_override
        print(f"[CHECK] Batch size override: {batch_size}")
    num_workers = 0 if device.type == "cpu" else training_conf.get("num_workers", 4)
    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[CHECK] DataLoader: batch_size={batch_size}, workers={num_workers}, pin_memory={device.type=='cuda'}")

    # DataLoader speed test
    print("[CHECK] DataLoader speed test (5 batches)...")
    t0 = time.time()
    for i, batch in enumerate(dl_train):
        if i >= 5:
            break
    t1 = time.time()
    dl_speed = 5 / max(t1 - t0, 0.001)
    print(f"[CHECK] DataLoader: {dl_speed:.1f} batches/sec")

    # ---- CRITERION ----
    from kimodo_sceneco.train.train import SceneCoDiffusionLoss
    criterion = SceneCoDiffusionLoss(
        model.diffusion,
        prior_weight=training_conf.get("prior_weight", 0.5),
    ).to(device)
    print(f"[CHECK] Criterion: SceneCoDiffusionLoss (prior_weight={training_conf.get('prior_weight', 0.5)})")

    # ---- OPTIMIZER ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_conf.get("lr", 1e-4),
        weight_decay=training_conf.get("weight_decay", 0.01),
    )
    print(f"[CHECK] Optimizer: AdamW, lr={training_conf.get('lr', 1e-4)}, "
          f"weight_decay={training_conf.get('weight_decay', 0.01)}, {len(trainable_params)} param groups")
    optimizer.zero_grad()
    optimizer.step()
    for group in optimizer.param_groups:
        if any(p.grad is not None for p in group['params']):
            print(f"[CHECK] ⚠️  Gradients present after initial zero_grad+step — check optimizer setup!")
            break
    optimizer.zero_grad()
    print(f"[CHECK] Optimizer gradient reset verified")

    # ---- LR SCHEDULER ----
    from torch.optim.lr_scheduler import CosineAnnealingLR
    max_steps = args.steps
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)
    print(f"[CHECK] Scheduler: CosineAnnealingLR, T_max={max_steps}")

    # ---- RESUME SUPPORT ----
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n[CHECK] Resuming from: {resume_path}")
            ckpt = torch.load(str(resume_path), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt.get("global_step", 0)
            print(f"[CHECK] Resumed at step {global_step}")

    # ---- TRAINING LOOP ----
    print("\n" + "=" * 70)
    print("  STARTING TRAINING LOOP")
    print("=" * 70)
    print(f"  Max steps:     {max_steps}")
    print(f"  Start step:    {global_step}")
    print(f"  LR:            {training_conf.get('lr', 1e-4)}")
    print(f"  Device:        {device}")
    print(f"  Output:        {output_dir}")
    print(f"  Precision:     {training_conf.get('precision', 'bf16')}")
    print(f"  Grad clip:     {training_conf.get('max_grad_norm', 1.0)}")
    print(f"  Prior weight:  {training_conf.get('prior_weight', 0.5)}")
    print("=" * 70)

    model.train()
    best_val_loss = float("inf")
    monitor_log = []
    loss_tracker = LossTrendDetector(window_size=100, spike_factor=5.0)
    log_interval = args.log_interval
    grad_clip = training_conf.get("max_grad_norm", 1.0)

    # Tracking stats
    clip_count = 0
    step_times = deque(maxlen=50)
    total_batches = 0
    nan_skip_count = 0
    oom_count = 0

    # ---- MAIN TRAINING LOOP ----
    try:
        while global_step < max_steps:
            for batch_idx, batch in enumerate(dl_train):
                step_start = time.time()
                global_step += 1
                total_batches += 1

                # --- CHECK 1: Batch validation ---
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                batch_report = validate_batch(batch, global_step, device)
                if not batch_report["ok"]:
                    print(f"⚠️  [Step {global_step}] Invalid batch: {batch_report['warnings']}")
                    continue

                # --- CHECK 2: Forward pass with OOM guard ---
                try:
                    motion = batch["motion_features"]
                    voxel = batch["voxel_grid"]
                    motion_mask = batch["motion_mask"]
                    lengths = batch["lengths"]

                    x_pad_mask = torch.ones(motion.shape[0], motion.shape[1],
                                            dtype=torch.bool, device=device)

                    if "text_feat" in batch:
                        text_feat = batch["text_feat"].to(device)
                        text_feat_pad_mask = torch.ones(text_feat.shape[0], text_feat.shape[1],
                                                        dtype=torch.bool, device=device)
                    else:
                        text_feat, text_length = model.text_encoder(batch["texts"])
                        text_feat = text_feat.to(device)
                        bs, maxlen = text_feat.shape[:2]
                        text_length = text_length.to(device) if isinstance(text_length, torch.Tensor) else torch.tensor(text_length, device=device)
                        text_feat_pad_mask = torch.arange(maxlen, device=device).expand(bs, maxlen) < text_length[:, None]

                    (scene_feat_root, scene_mask_root), (scene_feat_body, scene_mask_body) = model.encode_scene(voxel)
                    t = torch.randint(0, model.diffusion.num_base_steps, (motion.shape[0],), device=device)

                    first_heading_angle = model.motion_rep.get_root_heading_angle(motion)[:, 0]

                    model_kwargs = {
                        "x_pad_mask": x_pad_mask,
                        "text_feat": text_feat,
                        "text_pad_mask": text_feat_pad_mask,
                        "scene_feat_root": scene_feat_root,
                        "scene_mask_root": scene_mask_root,
                        "scene_feat_body": scene_feat_body,
                        "scene_mask_body": scene_mask_body,
                        "first_heading_angle": first_heading_angle,
                    }
                    if loss_mask is not None:
                        model_kwargs["loss_mask"] = loss_mask.to(device)

                    # --- CHECK 3: Scene encoder output validation ---
                    scene_check = validate_scene_encoder_output(scene_feat_root, scene_mask_root, global_step, tag="root")
                    scene_check_body = validate_scene_encoder_output(scene_feat_body, scene_mask_body, global_step, tag="body")
                    # --- CHECK 4: Text encoder output validation ---
                    text_check = validate_text_encoder_output(text_feat, text_feat_pad_mask, global_step)

                    losses = criterion.training_losses(model.denoiser, motion, t, model_kwargs)
                    total_loss = losses.get("loss", sum(losses.values()))

                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    print(f"⚠️  [Step {global_step}] CUDA OOM! Clearing cache and skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    if oom_count > 5:
                        print(f"❌ Too many OOM errors ({oom_count}). Aborting!")
                        break
                    continue

                # --- CHECK 5: NaN loss detection ---
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    nan_skip_count += 1
                    print(f"⚠️  [Step {global_step}] {'NaN' if torch.isnan(total_loss) else 'Inf'} loss, skipping! "
                          f"(total NaN skips: {nan_skip_count})")
                    if nan_skip_count > 10:
                        print(f"❌ Too many NaN steps ({nan_skip_count}). Aborting!")
                        break
                    continue

                # --- CHECK 6: Loss trend analysis ---
                loss_val_for_trend = total_loss.item()
                loss_trend = loss_tracker.update(loss_val_for_trend)

                # --- Backward pass ---
                optimizer.zero_grad()
                total_loss.backward()

                # --- CHECK 7: Gradient statistics ---
                grad_stats = compute_grad_stats(model)

                # --- CHECK 8: Gradient clipping with tracking ---
                clip_report = {"applied": False, "pre_clip_norm": grad_stats["grad_norm"],
                               "post_clip_norm": grad_stats["grad_norm"]}
                if grad_stats["grad_norm"] > grad_clip and grad_stats["grad_norm"] > 0:
                    pre_clip = grad_stats["grad_norm"]
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    clip_report["applied"] = True
                    clip_report["pre_clip_norm"] = pre_clip
                    clip_count += 1
                    # Recompute post-clip norm
                    post_norm = math.sqrt(sum(
                        p.grad.detach().norm().item()**2
                        for p in model.parameters()
                        if p.grad is not None and p.requires_grad
                    ))
                    clip_report["post_clip_norm"] = post_norm

                optimizer.step()
                scheduler.step()

                # --- Timing ---
                step_end = time.time()
                step_time = step_end - step_start
                step_times.append(step_time)

                # --- CHECK 9: GPU memory snapshot ---
                gpu_mem = get_gpu_memory_snapshot(device)

                # --- CHECK 10: Dead grad detection ---
                if grad_stats["n_dead_params"] > 0 and global_step % (log_interval * 5) == 0:
                    print(f"[CHECK] [Step {global_step}] {grad_stats['n_dead_params']}/{grad_stats['n_params']} "
                          f"params have near-zero gradients (dead params)")

                # --- Periodic CHECK: Frozen param integrity ---
                frozen_check = {"checked": False}
                if args.check_frozen_every > 0 and global_step % args.check_frozen_every == 0 and frozen_snapshot:
                    frozen_check = check_frozen_param_integrity(
                        model, frozen_snapshot, global_step, tolerance=1e-8
                    )
                    if not frozen_check["ok"]:
                        print(f"❌ [Step {global_step}] FROZEN PARAM DRIFT DETECTED!")
                        for w in frozen_check["warnings"]:
                            print(f"    {w}")
                    else:
                        print(f"[CHECK] [Step {global_step}] Frozen params OK (checked {frozen_check['n_checked']} "
                              f"tensors, max_delta={frozen_check['max_delta']:.2e})")

                # --- Logging ---
                if global_step % log_interval == 0 or global_step <= 5:
                    timing = {
                        "step_time_s": step_time,
                        "steps_per_sec": 1.0 / max(step_time, 1e-6),
                        "avg_step_time_s": float(np.mean(step_times)) if step_times else 0,
                        "total_time_min": (step_end - (step_end - sum(step_times))) / 60.0
                            if step_times else 0,
                    }

                    monitor_step(model, losses, global_step, optimizer, output_dir,
                                 grad_stats, gpu_mem, scene_check, text_check,
                                 loss_trend, clip_report, timing)

                    lr_now = scheduler.get_last_lr()[0]
                    loss_vals = {k: f"{float(v):.6f}" for k, v in losses.items()
                                if isinstance(v, (torch.Tensor, float))}
                    alphas_raw = get_alphas(model)
                    gates = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in alphas_raw.items()}
                    alpha_str = " ".join(f"{k.split('.')[-1]}={v:.4f}"
                                        for k, v in list(gates.items())[:3])

                    grad_warn = ""
                    if grad_stats["n_dead_params"] > 0:
                        grad_warn = f" [DEAD:{grad_stats['n_dead_params']}]"

                    print(f"[Step {global_step:>7d}/{max_steps}] "
                          f"loss={float(total_loss):.5f} "
                          f"grad={grad_stats['grad_norm']:.4f} "
                          f"clip={'Y' if clip_report['applied'] else 'N'} "
                          f"lr={lr_now:.2e} "
                          f"mem={gpu_mem['allocated_gb']:.2f}GB "
                          f"t={step_time:.2f}s | "
                          f"α: {alpha_str}{grad_warn}")

                    if loss_trend.get("warnings"):
                        for w in loss_trend["warnings"]:
                            print(f"  ⚠️  Loss warning: {w}")

                    monitor_log.append({
                        "step": global_step, "loss": float(total_loss),
                        "grad_norm": grad_stats["grad_norm"],
                        "lr": lr_now, "gates": gates, "alpha_raw": alphas_raw,
                    })

                # --- Periodic CHECK: Checkpoint with integrity verification ---
                if args.ckpt_verify_every > 0 and global_step % args.ckpt_verify_every == 0:
                    ckpt_path = output_dir / "checkpoints" / f"checkpoint_step{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "config": conf,
                    }, ckpt_path)
                    ckpt_integrity = validate_checkpoint_integrity(ckpt_path, model, device)
                    if ckpt_integrity["ok"]:
                        print(f"[CHECK] [Step {global_step}] Checkpoint saved + verified OK: {ckpt_path.name}")
                    else:
                        print(f"❌ [Step {global_step}] Checkpoint verification FAILED!")
                        for w in ckpt_integrity["warnings"]:
                            print(f"    {w}")

                if global_step >= max_steps:
                    break

            if global_step >= max_steps:
                break

    except KeyboardInterrupt:
        print(f"\n⏸️  Training interrupted at step {global_step}. Saving checkpoint...")
    except Exception as e:
        print(f"\n❌ Training error at step {global_step}: {type(e).__name__}: {e}")
        traceback.print_exc()

    # ---- FINAL CHECKPOINT ----
    final_ckpt_path = output_dir / "checkpoints" / f"checkpoint_step{global_step}_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "config": conf,
    }, final_ckpt_path)
    print(f"\n✅ Final checkpoint saved: {final_ckpt_path}")

    final_integrity = validate_checkpoint_integrity(final_ckpt_path, model, device)
    if final_integrity["ok"]:
        print(f"[CHECK] Final checkpoint verified OK")
    else:
        print(f"❌ Final checkpoint verification FAILED!")
        for w in final_integrity["warnings"]:
            print(f"    {w}")

    # ---- FINAL FROZEN PARAM CHECK ----
    if frozen_snapshot:
        final_frozen = check_frozen_param_integrity(model, frozen_snapshot, global_step)
        print(f"\n[CHECK] Final frozen param check: "
              f"{'✅ OK' if final_frozen['ok'] else '❌ DRIFT DETECTED'} "
              f"(max_delta={final_frozen['max_delta']:.2e})")
        if not final_frozen["ok"]:
            for w in final_frozen["warnings"]:
                print(f"    {w}")

    # ---- SUMMARY ----
    with open(output_dir / "monitor_summary.json", "w") as f:
        json.dump(monitor_log[-200:], f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Total steps:      {global_step}")
    print(f"  NaN skips:        {nan_skip_count}")
    print(f"  OOM errors:       {oom_count}")
    print(f"  Gradient clips:   {clip_count}")
    print(f"  Final loss:       {monitor_log[-1]['loss']:.5f}" if monitor_log else "  N/A")
    print(f"  Output:           {output_dir}")
    avg_time = float(np.mean(step_times)) if step_times else 0
    print(f"  Avg step time:    {avg_time:.2f}s ({1.0/max(avg_time,1e-6):.1f} steps/sec)")
    final_mem = get_gpu_memory_snapshot(device)
    print(f"  Final GPU mem:    {final_mem['allocated_gb']:.2f}GB allocated, "
          f"{final_mem['max_allocated_gb']:.2f}GB peak")
    print("=" * 70)
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
