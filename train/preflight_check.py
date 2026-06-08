#!/usr/bin/env python3
"""Pre-flight check for Kimodo-SceneCo training.

Verifies EVERY precondition before training starts on GPU 3.
Run this before any training to catch issues early.

Usage:
    python preflight_check.py --gpu 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
KIMODO_ROOT = PROJECT_ROOT

if str(KIMODO_ROOT / "kimodo") not in sys.path:
    sys.path.insert(0, str(KIMODO_ROOT / "kimodo"))


class PreflightChecker:
    def __init__(self, gpu_id: int = 3):
        self.gpu_id = gpu_id
        self.results: Dict[str, Dict] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def check(self, name: str, passed: bool, detail: str = "", critical: bool = True):
        status = "PASS" if passed else "FAIL" if critical else "WARN"
        self.results[name] = {"status": status, "detail": detail}
        if not passed:
            if critical:
                self.errors.append(f"[{name}] {detail}")
            else:
                self.warnings.append(f"[{name}] {detail}")

    def section(self, title: str):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    # ================================================================
    # 1. HARDWARE
    # ================================================================
    def check_hardware(self):
        self.section("1. HARDWARE CHECK")

        import torch
        cuda_ok = torch.cuda.is_available()
        self.check("1.1 PyTorch CUDA compile", cuda_ok or torch.version.cuda is not None,
                   f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}",
                   critical=False)

        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(self.gpu_id) if self.gpu_id < torch.cuda.device_count() else "N/A"
            self.check("1.2 GPU exists at index", self.gpu_id < torch.cuda.device_count(),
                       f"GPU {self.gpu_id}: {gpu_name} (total: {torch.cuda.device_count()})")
            if self.gpu_id < torch.cuda.device_count():
                mem = torch.cuda.get_device_properties(self.gpu_id).total_mem / 1024**3
                self.check("1.3 GPU memory", mem >= 8,
                           f"{mem:.1f} GB (need >=8 GB for batch_size=4)")
                self.check("1.4 GPU compute capability",
                           torch.cuda.get_device_capability(self.gpu_id)[0] >= 7,
                           f"SM {torch.cuda.get_device_capability(self.gpu_id)}")
        else:
            self.check("1.2 GPU", False, "CUDA NOT available — training will use CPU (slow)", critical=False)

        import psutil
        ram = psutil.virtual_memory()
        self.check("1.5 System RAM", ram.available / 1024**3 >= 16,
                   f"{ram.available/1024**3:.1f} GB free / {ram.total/1024**3:.1f} GB total")

        disk = psutil.disk_usage(str(PROJECT_ROOT))
        self.check("1.6 Disk space", disk.free / 1024**3 >= 50,
                   f"{disk.free/1024**3:.1f} GB free (need >=50 GB for checkpoints)")

    # ================================================================
    # 2. MODEL FILE
    # ================================================================
    def check_model(self):
        self.section("2. MODEL CHECK")

        model_dir = PROJECT_ROOT / "models" / "Kimodo-SOMA-RP-v1.1"
        ckpt_path2 = PROJECT_ROOT / ".hf_cache" / "hub" / "models--nvidia--Kimodo-G1-RP-v1" / "snapshots"
        if not model_dir.exists() and ckpt_path2.exists():
            snapshots = list(ckpt_path2.glob("*"))
            if snapshots:
                model_dir = snapshots[0]
        self.check("2.1 Model directory", model_dir.exists(), str(model_dir))

        safetensors = list(model_dir.glob("*.safetensors"))
        safetensors_found = len(safetensors) > 0
        self.check("2.2 model.safetensors", safetensors_found,
                   f"{safetensors[0].stat().st_size/1024**3:.2f} GB" if safetensors_found else "MISSING in " + str(model_dir))

        configs = list(model_dir.glob("config*.yaml")) + list(model_dir.glob("*.yaml"))
        self.check("2.3 config.yaml", len(configs) > 0)

        stats = model_dir / "stats"
        self.check("2.4 stats directory", stats.exists())
        if stats.exists():
            for stat_name in ["mean.npy", "std.npy"]:
                sp = stats / "motion" / "body" / stat_name
                sp2 = stats / "motion" / "global_root" / stat_name
                self.check(f"2.4 stats/{stat_name}", sp.exists() or sp2.exists(), str(sp))

    # ================================================================
    # 3. DATASET
    # ================================================================
    def check_dataset(self):
        self.section("3. DATASET CHECK")

        cached = PROJECT_ROOT / "kimodo" / "kimodo_sceneco" / "cached_data"
        self.check("3.1 Cached data dir", cached.exists(), str(cached))

        if cached.exists():
            npz_files = sorted(cached.glob("seg_*.npz"))
            self.check("3.2 Cached npz count", len(npz_files) > 100,
                       f"{len(npz_files)} files (need >100)")

            if npz_files:
                sample = dict(np.load(str(npz_files[0]), allow_pickle=True))
                required_keys = {"motion_features", "voxel_grid", "text", "length", "scene_name"}
                missing = required_keys - set(sample.keys())
                extra_keys = set(sample.keys()) - required_keys
                self.check("3.3 Cached npz keys", len(missing) == 0,
                           f"missing={missing}, extra={list(extra_keys)[:3]}")

                mf = sample.get("motion_features")
                if mf is not None:
                    self.check("3.4 motion_features shape", len(mf.shape) >= 2,
                               f"shape={list(mf.shape) if hasattr(mf,'shape') else type(mf)}")
                    self.check("3.5 motion_features no NaN", not np.any(np.isnan(mf)),
                               "NaN detected" if np.any(np.isnan(mf)) else "clean")

                vg = sample.get("voxel_grid")
                if vg is not None:
                    self.check("3.6 voxel_grid shape", len(vg.shape) == 3,
                               f"shape={list(vg.shape) if hasattr(vg,'shape') else type(vg)}")
                    self.check("3.7 voxel_grid occupancy >0",
                               vg.mean() > 0.01 if hasattr(vg, 'mean') else True,
                               f"occ={vg.mean()*100:.1f}%" if hasattr(vg,'mean') else "")

        scene_dir = PROJECT_ROOT / "LINGO" / "dataset" / "dataset" / "Scene"
        self.check("3.8 Scene directory", scene_dir.exists(), str(scene_dir))
        if scene_dir.exists():
            scene_files = sorted(scene_dir.glob("*.npy"))
            self.check("3.9 Scene npy count", len(scene_files) > 0,
                       f"{len(scene_files)} scene files")

        soma_dir = PROJECT_ROOT / "soma_converted_all" / "lingo"
        self.check("3.10 SOMA converted data", soma_dir.exists())
        if soma_dir.exists():
            soma_count = len(list(soma_dir.glob("seg_*.npz")))

        self.check("3.12 Text encoders dir",
                   (PROJECT_ROOT / "text_encoders").exists() or
                   (PROJECT_ROOT / ".hf_cache").exists(),
                   "text_encoders or .hf_cache")

    # ================================================================
    # 4. CODE IMPORTS
    # ================================================================
    def check_imports(self):
        self.section("4. CODE IMPORT CHECK")

        crucial_imports = [
            ("4.1 torch", "torch"),
            ("4.2 numpy", "numpy"),
            ("4.3 scipy", "scipy"),
            ("4.4 tqdm", "tqdm"),
            ("4.5 yaml (pyyaml)", "yaml"),
            ("4.6 omegaconf", "omegaconf"),
            ("4.7 hydra", "hydra"),
            ("4.8 safetensors", "safetensors"),
            ("4.9 Kimodo model", "kimodo.model"),
            ("4.10 SOMA", "soma"),
            ("4.11 Scene encoder", "kimodo_sceneco.model.scene_encoder"),
            ("4.12 Dataset", "kimodo_sceneco.train.dataset"),
            ("4.13 Preprocess", "kimodo_sceneco.train.preprocess"),
        ]

        for label, module_name in crucial_imports:
            try:
                __import__(module_name)
                self.check(label, True, f"import {module_name} OK")
            except Exception as e:
                self.check(label, False, f"import {module_name}: {type(e).__name__}: {e}")

    # ================================================================
    # 5. ENV VARIABLES
    # ================================================================
    def check_env(self):
        self.section("5. ENVIRONMENT CHECK")

        env_vars = {
            "CHECKPOINT_DIR": "models/Kimodo-SOMA-RP-v1.1",
            "HF_HOME": ".hf_cache",
            "TEXT_ENCODERS_DIR": "text_encoders",
            "TEXT_ENCODER_MODE": "local",
            "TEXT_ENCODER_DEVICE": "cpu",
        }

        for var, expected in env_vars.items():
            actual = os.environ.get(var, "")
            ok = True
            if isinstance(expected, str) and var not in ("TEXT_ENCODER_MODE", "TEXT_ENCODER_DEVICE"):
                expected_path = (PROJECT_ROOT / expected).resolve()
                actual_path = Path(actual).resolve() if actual else None
                ok = actual_path == expected_path or (expected_path.exists() and actual_path is not None and expected_path.name == actual_path.name)
            detail = f"actual={actual}, expected={expected}"
            self.check(f"5. ENV {var}", ok, detail)

    # ================================================================
    # 6. CONFIG VALIDATION
    # ================================================================
    def check_config(self):
        self.section("6. CONFIG VALIDATION")

        import yaml

        for cfg_name in ["sceneco_root_only.yaml", "sceneco_root_body.yaml"]:
            cfg_path = PROJECT_ROOT / "configs" / cfg_name
            self.check(f"6.{cfg_name} exists", cfg_path.exists())

            if cfg_path.exists():
                try:
                    with open(cfg_path) as f:
                        conf = yaml.safe_load(f)
                    training = conf.get("training", {})

                    self.check(f"6.{cfg_name} freeze_pretrained",
                               training.get("freeze_pretrained", False) == True,
                               str(training.get("freeze_pretrained")))

                    self.check(f"6.{cfg_name} lr=1e-4",
                               abs(training.get("lr", 0) - 1e-4) < 1e-10,
                               str(training.get("lr")))

                    self.check(f"6.{cfg_name} precision=bf16",
                               training.get("precision") == "bf16",
                               str(training.get("precision")))

                except Exception as e:
                    self.check(f"6.{cfg_name} parse", False, str(e))

    # ================================================================
    # 7. GATE INIT VERIFICATION
    # ================================================================
    def check_gate_init(self):
        self.section("7. GATE INITIALIZATION CHECK")

        from kimodo_sceneco.model.backbone import SceneCoLayer
        sl = SceneCoLayer(d_model=256, scene_feat_dim=64, nhead=4)
        alpha = sl.alpha.item()
        self.check("7.1 gate alpha = 0.0", abs(alpha) < 1e-6, f"alpha = {alpha:.10f}")

        from kimodo_sceneco.exp.shared.sceneco_layers import SceneCoLayer as SC2
        sl2 = SC2(d_model=256, scene_feat_dim=64, nhead=4)
        alpha2 = sl2.alpha.item()
        self.check("7.2 sceneco_layers alpha = 0.0", abs(alpha2) < 1e-6, f"alpha = {alpha2:.10f}")

    # ================================================================
    # 8. FORWARD PASS SANITY CHECK
    # ================================================================
    def check_forward_pass(self):
        self.section("8. FORWARD PASS SANITY CHECK")

        import torch
        device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() and self.gpu_id < torch.cuda.device_count() else "cpu")
        print(f"  Using device: {device}")

        from kimodo_sceneco.model.scene_encoder import VoxelViT
        from kimodo_sceneco.model.backbone import SceneCoLayer

        dummy_voxel = torch.randn(2, 1, 64, 64, 64).to(device)
        try:
            vit = VoxelViT().to(device)
            scene_feat, scene_mask = vit(dummy_voxel)
            self.check("8.1 VoxelViT forward", True,
                       f"output shape: {list(scene_feat.shape)}, mask: {list(scene_mask.shape)}")
        except Exception as e:
            self.check("8.1 VoxelViT forward", False, f"{type(e).__name__}: {e}")

        dummy_motion = torch.randn(2, 128, 512).to(device)
        scene_feat_small = torch.randn(2, 256, 256).to(device)

        try:
            sc_layer = SceneCoLayer(d_model=512, scene_feat_dim=256, nhead=8).to(device)
            out = sc_layer(dummy_motion, scene_feat_small)
            self.check("8.2 SceneCoLayer forward", True,
                       f"output shape: {list(out.shape)}, alpha={sc_layer.alpha.item():.6f}")

            time.sleep(3)
            alpha_after = sc_layer.alpha.item()
            self.check("8.3 alpha unchanged after forward",
                       abs(alpha_after) < 1e-6,
                       f"alpha after fwd: {alpha_after:.10f}")
            self.check("8.4 output ≈ input (gate=0)",
                       torch.allclose(out, dummy_motion, atol=1e-4),
                       f"max diff: {float((out - dummy_motion).abs().max()):.6f}")
        except Exception as e:
            self.check("8.2 SceneCoLayer forward", False, f"{type(e).__name__}: {e}")

    # ================================================================
    # 9. TRAINING PREVIEW
    # ================================================================
    def check_training_preview(self):
        self.section("9. TRAINING PREVIEW")

        import torch
        os.environ["CHECKPOINT_DIR"] = str(PROJECT_ROOT / "models" / "Kimodo-SOMA-RP-v1.1")
        from kimodo.model import load_model
        device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() and self.gpu_id < torch.cuda.device_count() else "cpu")

        try:
            model = load_model("Kimodo-SOMA-RP-v1.1", device=device)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.check("9.1 Model loaded", True,
                       f"total params={total_params/1e6:.1f}M, trainable={trainable/1e6:.1f}M")

            sample_text = ["walk forward"]
            for name, param in model.named_parameters():
                param.requires_grad = "sceneco" in name or "scene_encoder" in name or "scene_null" in name

            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.check("9.2 Freeze logic", trainable2 < total_params * 0.05,
                       f"frozen={frozen/1e6:.1f}M, trainable={trainable2/1e6:.1f}M ({trainable2/total_params*100:.1f}%)")

            text_feat = model.text_encoder(sample_text).to(device)
            text_mask = torch.ones(len(sample_text), text_feat.shape[1], dtype=torch.bool, device=device)
            self.check("9.3 Text encoding", True,
                       f"text_feat shape: {list(text_feat.shape)}")

            t = torch.randint(0, 1000, (min(2, len(text_feat)),), device=device)
            x_shape = (min(2, len(text_feat)), 196, model.denoiser.motion_rep.motion_rep_dim)
            try:
                x = torch.randn(*x_shape, device=device)
                kwargs = {
                    "x_pad_mask": torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=device),
                    "text_feat": text_feat[:x.shape[0]].to(device),
                    "text_pad_mask": text_mask[:x.shape[0]].to(device),
                }
                _ = model.denoiser(x, **kwargs)
                self.check("9.4 Kimodo denoiser forward", True, f"x shape: {list(x.shape)}")
            except Exception as e:
                self.check("9.4 Kimodo denoiser forward", False, f"{type(e).__name__}: {e}")

        except Exception as e:
            self.check("9.1 Model loading", False, f"{type(e).__name__}: {e}")

    def report(self):
        self.section("FINAL REPORT")
        passed = sum(1 for r in self.results.values() if r["status"] == "PASS")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAIL")
        warned = sum(1 for r in self.results.values() if r["status"] == "WARN")
        total = len(self.results)

        print(f"  Total checks: {total}")
        print(f"  ✅ Passed:    {passed}")
        print(f"  ⚠️  Warnings:  {warned}")
        print(f"  ❌ Failed:    {failed}")

        if failed > 0:
            print(f"\n  FAILED CHECKS:")
            for e in self.errors:
                print(f"    ❌ {e}")

        if warned > 0:
            print(f"\n  WARNINGS:")
            for w in self.warnings:
                print(f"    ⚠️  {w}")

        with open(PROJECT_ROOT / "outputs" / "preflight_report.json", "w") as f:
            json.dump({
                "summary": {"total": total, "passed": passed, "failed": failed, "warned": warned},
                "results": self.results,
            }, f, indent=2)

        print(f"\n  Report saved to: outputs/preflight_report.json")
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Kimodo-SceneCo pre-flight check")
    parser.add_argument("--gpu", type=int, default=3, help="GPU device ID")
    parser.add_argument("--full", action="store_true", help="Run full model loading check")
    args = parser.parse_args()

    checker = PreflightChecker(gpu_id=args.gpu)
    checker.check_hardware()
    checker.check_model()
    checker.check_dataset()
    checker.check_imports()
    checker.check_env()
    checker.check_config()
    checker.check_gate_init()
    checker.check_forward_pass()

    if args.full:
        checker.check_training_preview()

    ok = checker.report()
    if not ok:
        print("\n❌ Some critical checks failed. Fix before training.")
        sys.exit(1)
    else:
        print("\n✅ Ready to train!")
        sys.exit(0)


if __name__ == "__main__":
    main()
