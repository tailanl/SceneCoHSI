"""Compare checkpoints: diff analysis, trainable param count, alpha stats.

Usage:
    python compare_checkpoints.py --checkpoints \
        ckpt1.pt ckpt2.pt ckpt3.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch


def load_checkpoint(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        return {
            "state_dict": checkpoint["model_state_dict"],
            "epoch": checkpoint.get("epoch", "?"),
            "global_step": checkpoint.get("global_step", "?"),
            "args": checkpoint.get("args", {}),
        }
    if "state_dict" in checkpoint:
        return {
            "state_dict": checkpoint["state_dict"],
            "epoch": checkpoint.get("epoch", "?"),
            "global_step": checkpoint.get("global_step", "?"),
        }
    return {
        "state_dict": checkpoint,
        "epoch": "?",
        "global_step": "?",
    }


def count_params(sd: dict) -> int:
    return sum(p.numel() for p in sd.values())


def get_alpha_stats(sd: dict, ckpt_info: Dict) -> Dict:
    alphas = []
    for key, val in sd.items():
        if "alpha" in key and val.numel() == 1:
            alphas.append(val.item())
    if not alphas:
        return {}
    return {
        "alpha_mean": sum(alphas) / len(alphas),
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "alpha_count": len(alphas),
    }


def compare_pair(name_a: str, sd_a: dict, name_b: str, sd_b: dict):
    common_keys = set(sd_a.keys()) & set(sd_b.keys())
    only_a = set(sd_a.keys()) - set(sd_b.keys())
    only_b = set(sd_b.keys()) - set(sd_a.keys())

    diffs = []
    for key in common_keys:
        diff = (sd_a[key].float() - sd_b[key].float()).abs().max().item()
        if diff > 1e-8:
            diffs.append((key, diff))

    diffs.sort(key=lambda x: -x[1])

    return common_keys, only_a, only_b, diffs


def main():
    parser = argparse.ArgumentParser(description="Compare SceneCo checkpoints")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="List of checkpoint paths")
    parser.add_argument("--topk", type=int, default=10,
                        help="Show top-K largest parameter diffs")
    args = parser.parse_args()

    checkpoints = {}
    for cp_path in args.checkpoints:
        name = Path(cp_path).parent.name if "outputs" in cp_path else Path(cp_path).stem
        checkpoints[name] = load_checkpoint(cp_path)

    print(f"\n{'='*80}")
    print(f"{'Checkpoint':<40} {'Epoch':>8} {'Step':>8} {'Params':>12}")
    print(f"{'-'*80}")
    for name, info in checkpoints.items():
        n_params = count_params(info["state_dict"])
        print(f"{name:<40} {str(info['epoch']):>8} {str(info['global_step']):>8} {n_params:>12,}")

    print(f"\n{'='*80}")
    print("Alpha (gate) statistics:")
    print(f"{'-'*80}")
    print(f"{'Checkpoint':<40} {'Mean':>10} {'Min':>10} {'Max':>10} {'Count':>8}")
    print(f"{'-'*80}")
    for name, info in checkpoints.items():
        alpha = get_alpha_stats(info["state_dict"], info)
        if alpha:
            print(f"{name:<40} {alpha['alpha_mean']:>10.5f} {alpha['alpha_min']:>10.5f} {alpha['alpha_max']:>10.5f} {alpha['alpha_count']:>8}")
        else:
            print(f"{name:<40} {'N/A':>10}")

    names = list(checkpoints.keys())
    if len(names) >= 2:
        print(f"\n{'='*80}")
        print(f"Pairwise diffs: {names[0]} vs {names[1]}")
        print(f"{'-'*80}")
        _, only_a, only_b, diffs = compare_pair(
            names[0], checkpoints[names[0]]["state_dict"],
            names[1], checkpoints[names[1]]["state_dict"],
        )
        if only_a:
            print(f"Only in {names[0]}: {len(only_a)} keys")
        if only_b:
            print(f"Only in {names[1]}: {len(only_b)} keys")
        print(f"\nTop-{args.topk} largest diffs:")
        for key, diff in diffs[:args.topk]:
            print(f"  {key:<60} {diff:.2e}")


if __name__ == "__main__":
    main()
