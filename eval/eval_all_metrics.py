"""Comprehensive metric evaluation for all SceneCo experiments.

Generates motions for LINGO val samples with each experiment checkpoint,
computes C-class (CFR, JCR, MeanPen, MaxPen, P95Pen, PFFR, OPIR),
D-class (FootSkate, FootPenetration, FloatingRatio, VelSmooth, AccelJerk, BoneLenErr),
and optionally runs w_scene CFG sweep.

Outputs: all_metrics.json, metric_table.csv, metric_table.txt

Usage:
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH="kimodo:SOMA:$PYTHONPATH" \
    CHECKPOINT_DIR=models/Kimodo-SOMA-RP-v1.1 HF_HOME=.hf_cache \
    TEXT_ENCODERS_DIR=text_encoders TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu \
    python kimodo_scene_project/eval/eval_all_metrics.py \
    --num_samples 5 --output_dir kimodo_scene_project/outputs/metric_eval --gpu 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

FOOT_INDICES = [7, 8, 10, 11]

EXPERIMENT_CONFIGS = {
    "Kimodo_original": {
        "type": "original",
        "description": "Original Kimodo (no scene)",
    },
    "root_only": {
        "type": "sceneco",
        "ckpt": "kimodo_scene_project/outputs/root_only_sceneco/checkpoints/best_checkpoint.pt",
        "description": "SceneCo root only",
        "use_in_root": True, "use_in_body": False,
    },
    "body_only": {
        "type": "sceneco",
        "ckpt": "kimodo_scene_project/outputs/body_only_sceneco/checkpoints/best_checkpoint.pt",
        "description": "SceneCo body only",
        "use_in_root": False, "use_in_body": True,
    },
    "single_vit": {
        "type": "sceneco",
        "ckpt": "kimodo_scene_project/outputs/single_vit_gpu1/checkpoints/checkpoint_step200000_final.pt",
        "description": "Single shared ViT (root+body)",
        "use_in_root": True, "use_in_body": True, "dual_vit": False,
    },
    "dual_vit": {
        "type": "sceneco",
        "ckpt": "kimodo_scene_project/outputs/dual_vit_gpu2/checkpoints/checkpoint_step200000_final.pt",
        "description": "Dual ViT (root+body)",
        "use_in_root": True, "use_in_body": True, "dual_vit": True,
    },
    "dual_vit_floor": {
        "type": "sceneco",
        "ckpt": "kimodo_scene_project/outputs/dual_vit_floor_gpu3/checkpoints/checkpoint_step200000_final.pt",
        "description": "Dual ViT + floor mode",
        "use_in_root": True, "use_in_body": True, "dual_vit": True, "root_voxel_mode": "floor",
    },
    "root_body_merged": {
        "type": "sceneco",
        "ckpt": "kimodo_scene_project/outputs/root_body_merged/checkpoints/best_checkpoint.pt",
        "description": "Root+Body merged training",
        "use_in_root": True, "use_in_body": True, "dual_vit": True,
    },
}

CFG_GRID = {
    "w_text": [1.5, 2.0, 2.5],
    "w_constraint": [1.0, 2.0],
    "w_scene": [0.0, 0.5, 1.0, 1.5, 2.0],
}


def build_model(exp_name, device, device_str):
    from kimodo.model import load_model
    from kimodo_sceneco.model.kimodo_model import KimodoSceneCo

    cfg = EXPERIMENT_CONFIGS[exp_name]

    if cfg["type"] == "original":
        print(f"  Loading Kimodo_original...")
        model = load_model("Kimodo-SOMA-RP-v1.1", device=device_str)
        model.eval()
        return model

    print(f"  Building {exp_name}...")
    pretrained = load_model("Kimodo-SOMA-RP-v1.1", device=device_str)
    inner_denoiser = pretrained.denoiser
    if hasattr(inner_denoiser, "model"):
        inner_denoiser = inner_denoiser.model

    model = KimodoSceneCo(
        denoiser=inner_denoiser,
        text_encoder=pretrained.text_encoder,
        num_base_steps=1000,
        scene_encoder_type="voxel_vit",
        scene_encoder_config={
            "voxel_size": (64, 64, 64),
            "patch_size": (8, 8, 8),
            "d_model": 256,
            "num_layers": 4,
            "use_dual_vit": cfg.get("dual_vit", True),
            "root_voxel_mode": cfg.get("root_voxel_mode", "full"),
        },
        device=device,
        cfg_type="scene_separated",
        use_in_root_model=cfg["use_in_root"],
        use_in_body_model=cfg["use_in_body"],
    )
    model = model.to(device)
    model.eval()

    ckpt_path = cfg["ckpt"]
    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_dataset_samples(num_samples, seed=42):
    from kimodo_sceneco.train.dataset import LINGOSceneMotionDataset

    ds = LINGOSceneMotionDataset(
        data_root=str(PROJECT_ROOT / "LINGO" / "dataset"),
        max_frames=196, min_frames=40,
        voxel_size=(64, 64, 64),
        train_ratio=0.9, seed=42,
        split="val",
        scene_dropout=0.0,
        cache_dir=str(PROJECT_ROOT / "kimodo/kimodo_sceneco/cached_data"),
    )

    rng = np.random.RandomState(seed)
    n = min(num_samples, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)

    samples = []
    for idx in indices:
        seg = ds[int(idx)]
        grid = seg["voxel_grid"].numpy().squeeze()
        if grid.ndim == 3 and grid.shape == (64, 64, 64):
            pass
        elif grid.ndim == 4:
            grid = grid[0]
        samples.append({
            "text": seg.get("text", "no-text"),
            "num_frames": int(seg["length"]),
            "voxel": seg["voxel_grid"],
            "voxel_np": grid,
            "scene_name": seg.get("scene_name", f"scene_{idx}"),
        })
    return samples


def generate_motion(model, exp_name, sample, device, args):
    if exp_name == "Kimodo_original":
        with torch.no_grad():
            output = model(
                prompts=sample["text"],
                num_frames=sample["num_frames"],
                num_denoising_steps=args.num_denoising_steps,
                cfg_weight=[3.0, 1.5],
                cfg_type="separated",
                return_numpy=True,
            )
        return output

    voxel = sample["voxel"].to(device)
    if voxel.ndim == 4:
        voxel = voxel.unsqueeze(1)
    with torch.no_grad():
        output = model(
            prompts=sample["text"],
            num_frames=sample["num_frames"],
            num_denoising_steps=args.num_denoising_steps,
            cfg_weight=[3.0, 1.5, 2.0],
            cfg_type="scene_separated",
            scene_input=voxel,
            return_numpy=True,
        )
    return output


def compute_metrics(output, voxel_np):
    from kimodo_scene_project.eval.eval_scene_metrics import (
        compute_c_class_metrics, compute_d_class_metrics,
    )

    pj = np.squeeze(output["posed_joints"])
    rp = np.squeeze(output["root_positions"])
    srp = np.squeeze(output.get("smooth_root_pos", output.get("root_positions")))

    c = compute_c_class_metrics(pj, rp, voxel_np)

    if pj.ndim == 3:
        d_args = (pj, srp, FOOT_INDICES, voxel_np)
    elif pj.ndim == 4:
        d_args = (pj[0], srp if srp.ndim == 2 else srp[0], FOOT_INDICES, voxel_np)
    else:
        d_args = (pj, srp, FOOT_INDICES, voxel_np)
    d = compute_d_class_metrics(*d_args, fps=30)

    return {**c, **d}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_denoising_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/metric_eval")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", type=str, nargs="*", default=None)
    parser.add_argument("--run_sweep", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_names = args.experiments or list(EXPERIMENT_CONFIGS.keys())
    print(f"Experiments ({len(exp_names)}):\n  " + "\n  ".join(exp_names))

    samples = load_dataset_samples(args.num_samples, seed=args.seed)
    print(f"\nLoaded {len(samples)} val samples:")
    for i, s in enumerate(samples):
        print(f"  [{i}] '{s['text'][:55]}' ({s['num_frames']}f)")

    all_results = {}

    for exp_name in exp_names:
        cfg = EXPERIMENT_CONFIGS[exp_name]
        print(f"\n{'='*60}")
        print(f"▶ {exp_name}: {cfg['description']}")
        print(f"{'='*60}")

        model = build_model(exp_name, device, device_str)

        per_sample = []
        for si, sample in enumerate(samples):
            print(f"  [{si}] '{sample['text'][:45]}' ({sample['num_frames']}f)...", end=" ")
            t0 = time.time()
            output = generate_motion(model, exp_name, sample, device, args)
            gen_time = time.time() - t0

            metrics = compute_metrics(output, sample["voxel_np"])

            entry = {
                "sample_idx": si,
                "scene": sample["scene_name"],
                "text": sample["text"],
                "num_frames": sample["num_frames"],
                "gen_time_s": gen_time,
                **metrics,
            }
            per_sample.append(entry)
            print(f"CFR={metrics['CFR']:.4f} JCR={metrics['JCR']:.4f} "
                  f"MeanPen={metrics['MeanPen']:.4f} FootSkate={metrics['FootSkate']:.4f} "
                  f"({gen_time:.1f}s)")

        del model
        torch.cuda.empty_cache()

        agg = aggregate_metrics(per_sample)
        all_results[exp_name] = {"per_sample": per_sample, "aggregated": agg}

        print(f"  → Agg: CFR={agg['CFR']:.4f} MeanPen={agg['MeanPen']:.4f} "
              f"FootSkate={agg['FootSkate']:.4f} PFFR={agg['PFFR']:.4f} "
              f"BoneLenErr={agg['BoneLenErr']:.4f}")

    if args.run_sweep:
        print(f"\n{'='*60}")
        print(f"▶ w_scene Sweep")
        print(f"{'='*60}")
        sweep_exp = next((e for e in exp_names if e.startswith("dual_vit") and "floor" not in e), exp_names[1])
        print(f"  Running sweep on: {sweep_exp}")
        model = build_model(sweep_exp, device, device_str)
        sweep_data = run_sweep(model, sweep_exp, samples[0], device, args)
        all_results["w_scene_sweep"] = sweep_data
        del model
        torch.cuda.empty_cache()

    all_results["_config"] = {
        "num_samples": len(samples),
        "num_denoising_steps": args.num_denoising_steps,
        "experiments": exp_names,
        "seed": args.seed,
    }

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print_table(all_results, out_dir)

    print(f"\nDone. → {out_dir}/")
    print(f"  all_metrics.json | metric_table.csv | metric_table.txt")


def aggregate_metrics(per_sample):
    metric_keys = [
        "CFR", "JCR", "MeanPen", "MaxPen", "P95Pen", "PFFR", "OPIR",
        "FootSkate", "FootPenetration", "FloatingRatio",
        "VelSmooth", "AccelJerk", "BoneLenErr",
    ]
    agg = {}
    for k in metric_keys:
        vals = [s[k] for s in per_sample if s.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
        agg[f"{k}_std"] = float(np.std(vals)) if vals else None
    agg["avg_gen_time_s"] = float(np.mean([s["gen_time_s"] for s in per_sample]))
    return agg


def run_sweep(model, exp_name, sample, device, args):
    combos = []
    for wt in CFG_GRID["w_text"]:
        for wc in CFG_GRID["w_constraint"]:
            for ws in CFG_GRID["w_scene"]:
                combos.append({"w_text": wt, "w_constraint": wc, "w_scene": ws})

    print(f"  Grid: {len(combos)} combos on '{sample['text'][:40]}' ({sample['num_frames']}f)")

    results = []
    voxel = sample["voxel"].to(device)
    if voxel.ndim == 4:
        voxel = voxel.unsqueeze(1)
    for cfg in tqdm(combos, desc="  sweep"):
        with torch.no_grad():
            output = model(
                prompts=sample["text"],
                num_frames=sample["num_frames"],
                num_denoising_steps=args.num_denoising_steps,
                cfg_weight=[cfg["w_text"], cfg["w_constraint"], cfg["w_scene"]],
                cfg_type="scene_separated",
                scene_input=voxel,
                return_numpy=True,
            )
        metrics = compute_metrics(output, sample["voxel_np"])
        metrics.update(cfg)
        results.append(metrics)

    best_cfr = min(results, key=lambda r: r["CFR"])
    best_pen = min(results, key=lambda r: r["MeanPen"])

    print(f"  Best CFR:      w_scene={best_cfr['w_scene']}, CFR={best_cfr['CFR']:.4f}")
    print(f"  Best MeanPen:  w_scene={best_pen['w_scene']}, MeanPen={best_pen['MeanPen']:.4f}")

    return {"num_combos": len(combos), "best_cfr": best_cfr, "best_pen": best_pen, "results": results}


def print_table(all_results, out_dir):
    exp_names = [k for k in all_results if not k.startswith("_") and not k.startswith("w_scene")]

    c_metrics = ["CFR", "JCR", "MeanPen", "MaxPen", "P95Pen", "PFFR", "OPIR"]
    d_metrics = ["FootSkate", "FootPenetration", "FloatingRatio", "VelSmooth", "AccelJerk", "BoneLenErr"]
    all_m = c_metrics + d_metrics

    header = f"{'Experiment':26s}" + "".join(f" | {m:>10s}" for m in all_m)
    sep = "-" * len(header)

    lines = [f"\n{'='*len(header)}", "SceneCo Experiment Metrics Comparison",
             f"{'='*len(header)}", header, sep]

    for exp_name in exp_names:
        agg = all_results[exp_name]["aggregated"]
        row = f"{exp_name:26s}"
        for m in all_m:
            v = agg.get(m)
            row += f" | {v:10.4f}" if v is not None else f" | {'---':>10s}"
        lines.append(row)

    lines.append(sep)
    lines.append("C-class: CFR(↓) JCR(↓) MeanPen(↓) MaxPen(↓) P95Pen(↓) PFFR(↑) OPIR(↓)")
    lines.append("D-class: FootSkate(↓) FootPen(↓) Float(↓) VelSmooth(↓) AccelJerk(↓) BoneLen(↓)")
    lines.append("=" * len(header))

    table = "\n".join(lines)
    print(table)

    with open(out_dir / "metric_table.txt", "w") as f:
        f.write(table)

    csv = ["experiment," + ",".join(all_m)]
    for exp_name in exp_names:
        agg = all_results[exp_name]["aggregated"]
        vals = [f"{agg.get(m, ''):.6f}" if agg.get(m) is not None else ""
                for m in all_m]
        csv.append(f"{exp_name}," + ",".join(vals))
    with open(out_dir / "metric_table.csv", "w") as f:
        f.write("\n".join(csv) + "\n")

    if "w_scene_sweep" in all_results:
        sweep = all_results["w_scene_sweep"]
        sl = []
        sl.append(f"\n{'='*70}")
        sl.append("w_scene Sweep")
        sl.append(f"{'='*70}")
        sl.append(f"Combos: {sweep['num_combos']}")
        sl.append(f"Best CFR:  w_scene={sweep['best_cfr']['w_scene']} "
                  f"(CFR={sweep['best_cfr']['CFR']:.4f})")
        sl.append(f"Best Pen: w_scene={sweep['best_pen']['w_scene']} "
                  f"(MeanPen={sweep['best_pen']['MeanPen']:.4f})")

        scsv = ["w_text,w_constraint,w_scene,CFR,JCR,MeanPen,MaxPen,PFFR,OPIR"]
        for r in sweep["results"]:
            scsv.append(
                f"{r['w_text']},{r['w_constraint']},{r['w_scene']},"
                f"{r['CFR']:.6f},{r['JCR']:.6f},{r['MeanPen']:.6f},"
                f"{r['MaxPen']:.6f},{r['PFFR']:.6f},{r['OPIR']:.6f}"
            )
        with open(out_dir / "sweep_table.csv", "w") as f:
            f.write("\n".join(scsv) + "\n")

        sweep_text = "\n".join(sl)
        print(sweep_text)
        with open(out_dir / "metric_table.txt", "a") as f:
            f.write(sweep_text)


if __name__ == "__main__":
    main()
