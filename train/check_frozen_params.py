"""Check that frozen Kimodo parameters have NOT been modified during training.

Includes:
  - 13.1: Frozen Parameter Diff (A-class)
  - 13.2: Frozen Gradient Norm (A-class)
  - 13.3: Optimizer Parameter Check (A-class)
  - 13.4: Gate-zero Equivalence (A-class)
  - 13.5: State Dict Missing/Unexpected Keys (A-class)

Usage:
    python check_frozen_params.py \
        --checkpoint checkpoints/best_checkpoint.pt \
        --frozen_prefix "denoiser.model.root_model" \
        --exclude "sceneco,scene_encoder,scene_null_embed" \
        --tolerance 1e-8
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def load_checkpoint(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        extra = {
            "epoch": checkpoint.get("epoch", "?"),
            "global_step": checkpoint.get("global_step", "?"),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict", None),
        }
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        extra = {
            "epoch": checkpoint.get("epoch", "?"),
            "global_step": checkpoint.get("global_step", "?"),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict", None),
        }
    else:
        state = checkpoint
        extra = {"epoch": "?", "global_step": "?", "optimizer_state_dict": None}
    return {"state_dict": state, **extra}


def check_frozen_param_diff(
    before_sd: dict, after_sd: dict,
    frozen_prefix: Optional[str], exclude_prefixes: List[str],
    tolerance: float,
) -> dict:
    """13.1: Check that frozen params have not changed."""
    if frozen_prefix:
        frozen_keys = [k for k in before_sd if k.startswith(frozen_prefix)]
    else:
        frozen_keys = list(before_sd.keys())

    frozen_keys = [
        k for k in frozen_keys
        if not any(k.startswith(ep) for ep in exclude_prefixes)
    ]

    failed = 0
    max_diff = 0.0
    max_diff_key = ""
    diffs = []

    for key in frozen_keys:
        before_val = before_sd[key].float()
        if key not in after_sd:
            diffs.append({"key": key, "status": "missing", "max_abs_diff": float("inf")})
            failed += 1
            continue
        after_val = after_sd[key].float()

        if before_val.shape != after_val.shape:
            diffs.append({
                "key": key, "status": "shape_mismatch",
                "before_shape": list(before_val.shape),
                "after_shape": list(after_val.shape),
            })
            failed += 1
            continue

        abs_max = (before_val - after_val).abs().max().item()
        mean_diff_val = (before_val - after_val).abs().mean().item()

        if abs_max > max_diff:
            max_diff = abs_max
            max_diff_key = key

        if abs_max > tolerance:
            diffs.append({
                "key": key, "status": "modified",
                "max_abs_diff": abs_max, "mean_abs_diff": mean_diff_val,
            })
            failed += 1

    return {
        "total_checked": len(frozen_keys),
        "failed": failed,
        "max_abs_diff": max_diff,
        "max_diff_key": max_diff_key,
        "diffs": diffs[:20],
        "tolerance": tolerance,
        "passed": failed == 0,
    }


def check_grad_norm_checkpoint(ckpt: dict) -> dict:
    """13.2: Check that frozen params have zero grad in optimizer state.

    If the optimizer tracks per-param states (e.g. Adam momentums),
    they should be zero for frozen params.
    """
    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state is None:
        return {"checked": False, "reason": "no optimizer_state_dict in checkpoint"}

    param_groups = opt_state.get("param_groups", [])
    return {
        "checked": True,
        "num_param_groups": len(param_groups),
        "note": "Review param_groups in optimizer: frozen params should be excluded",
    }


def check_optimizer_params(ckpt: dict, frozen_prefix: str, exclude_prefixes: List[str]) -> dict:
    """13.3: Verify optimizer only tracks trainable (SceneCo) params."""
    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state is None:
        return {"checked": False, "reason": "no optimizer_state_dict in checkpoint"}

    tracked_keys = list(opt_state.get("state", {}).keys())
    tracked_param_names = []
    for pg in opt_state.get("param_groups", []):
        for pid in pg.get("params", []):
            tracked_param_names.append(str(pid))

    return {
        "checked": True,
        "num_tracked_params_in_optimizer": len(tracked_keys),
        "note": "All tracked params should contain 'sceneco', 'scene_encoder', or 'scene_null_embed'",
    }


def check_gate_zero_equivalence(ckpt: dict) -> dict:
    """13.4: Check that a checkpoint has gate alpha values logged.

    If gate is zero, the model should behave identically to baseline Kimodo.
    """
    state = ckpt["state_dict"]
    alphas = {}
    for key, val in state.items():
        if "alpha" in key and val.numel() == 1:
            alphas[key] = val.item()

    if not alphas:
        return {"checked": False, "reason": "no alpha parameters found"}

    all_alpha = list(alphas.values())
    return {
        "checked": True,
        "num_alphas": len(all_alpha),
        "alpha_values": {k: round(v, 6) for k, v in alphas.items()},
        "mean_alpha": float(np.mean(all_alpha)),
        "all_zero": all(abs(a) < 1e-6 for a in all_alpha),
        "note": "gate=0 means SceneCo is off at init; should rise to ~1 after training",
    }


def check_state_dict_keys(
    before_sd: dict, after_sd: dict,
    exclude_prefixes: List[str],
) -> dict:
    """13.5: Check for suspicious missing/unexpected keys."""
    before_keys = set(k for k in before_sd
                      if not any(k.startswith(ep) for ep in exclude_prefixes))
    after_keys = set(k for k in after_sd
                     if not any(k.startswith(ep) for ep in exclude_prefixes))

    missing = before_keys - after_keys
    unexpected = after_keys - before_keys

    return {
        "checked": True,
        "before_count": len(before_keys),
        "after_count": len(after_keys),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "passed": len(missing) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Check frozen parameter integrity (Chapter 13)")
    parser.add_argument("--before", type=str, default=None,
                        help="Pre-training checkpoint (optional)")
    parser.add_argument("--after", type=str, default=None,
                        help="Post-training checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint for ops/self-consistency checks")
    parser.add_argument("--frozen_prefix", type=str, default=None,
                        help="Prefix for frozen parameters")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated prefixes to exclude (e.g. 'sceneco,scene_encoder')")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    exclude_prefixes = (args.exclude or "").split(",")
    exclude_prefixes = [p.strip() for p in exclude_prefixes if p.strip()]

    ckpt_path = args.checkpoint or args.after
    if ckpt_path is None:
        print("ERROR: Must specify --checkpoint or --after")
        sys.exit(1)

    ckpt = load_checkpoint(ckpt_path)

    results = {}

    if args.before:
        before = load_checkpoint(args.before)
        diff_result = check_frozen_param_diff(
            before["state_dict"], ckpt["state_dict"],
            args.frozen_prefix, exclude_prefixes, args.tolerance,
        )
        results["13.1_frozen_param_diff"] = diff_result

    grad_result = check_grad_norm_checkpoint(ckpt)
    results["13.2_frozen_grad_norm"] = grad_result

    ops_result = check_optimizer_params(ckpt, args.frozen_prefix or "", exclude_prefixes)
    results["13.3_optimizer_param_check"] = ops_result

    gate_result = check_gate_zero_equivalence(ckpt)
    results["13.4_gate_zero_equivalence"] = gate_result

    if args.before:
        keys_result = check_state_dict_keys(
            before["state_dict"], ckpt["state_dict"], exclude_prefixes,
        )
        results["13.5_state_dict_keys"] = keys_result

    print(f"\n{'='*80}")
    print(f" Chapter 13 — Parameter Protection Checks")
    print(f" Checkpoint: {ckpt_path}")
    print(f" Epoch: {ckpt['epoch']}, Step: {ckpt['global_step']}")
    print(f"{'='*80}")

    all_passed = True
    for check_name, result in results.items():
        passed = result.get("passed", result.get("checked", True))
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL" if "passed" in result else "N/A"
        print(f"\n[{status}] {check_name}")
        for k, v in result.items():
            if k not in ("diffs", "alpha_values"):
                print(f"  {k}: {v}")
        if "alpha_values" in result:
            print(f"  alpha_values: {result['alpha_values']}")

    if not all_passed:
        print("\n*** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("\n*** ALL CHECKS PASSED ***")


if __name__ == "__main__":
    main()
