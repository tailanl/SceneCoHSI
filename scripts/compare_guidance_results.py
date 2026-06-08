"""Compare guidance experiment results across methods.

Reads multiple output directories and produces comparison CSV + Markdown table.

Usage:
    python scripts/compare_guidance_results.py \
        --pred_dirs outputs/kimodo_text outputs/guidance_path_only outputs/guidance_path_scene \
        --names baseline path_only path_scene \
        --scene_dir LINGO/dataset/dataset/Scene \
        --output_dir outputs/guidance_comparison
"""

import argparse, csv, logging, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kimodo"))

import numpy as np
from scipy.ndimage import distance_transform_edt

log = logging.getLogger(__name__)

# Import eval modules
from kimodo_scene_project.eval.eval_path_metrics import compute_path_metrics
from kimodo_scene_project.eval.eval_sceneadapt_metrics import compute_scene_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare guidance results")
    parser.add_argument("--pred_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", required=True)
    parser.add_argument("--scene_dir", type=str, default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--output_dir", type=str, default="outputs/guidance_comparison")
    args = parser.parse_args()

    assert len(args.pred_dirs) == len(args.names), "Number of dirs and names must match"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for method_name, pred_dir_str in zip(args.names, args.pred_dirs):
        pred_dir = Path(pred_dir_str)
        npz_files = sorted(pred_dir.glob("sample_*.npz"))
        if not npz_files:
            log.warning(f"No NPZ files in {pred_dir}")
            continue

        for npz_file in npz_files:
            data = np.load(str(npz_file), allow_pickle=True)
            gen_root = data["gen_root"]  # (T, 3)
            gt_root_xz = data.get("gt_root_xz")  # (T, 2)
            gen_joints = data.get("gen_joints")  # (T, 22, 3) or None
            scene_name = str(data.get("scene_name", ""))

            row = {"sample_id": npz_file.stem, "method": method_name}

            # Path metrics
            if gt_root_xz is not None:
                pm = compute_path_metrics(gen_root, gt_root_xz)
                for k, v in pm.items():
                    row[k] = v

            # Scene metrics
            if gen_joints is not None and scene_name:
                sm = compute_scene_metrics(gen_root, gen_joints, scene_name, args.scene_dir)
                for k, v in sm.items():
                    # Skip NaN
                    if np.isnan(v):
                        continue
                    row[k] = v

            all_rows.append(row)

    if not all_rows:
        log.error("No data found!")
        return

    # Collect all possible field names
    fieldnames = ["sample_id", "method"]
    for row in all_rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    # Write CSV
    csv_path = output_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Compute per-method averages and write Markdown
    method_avgs = {}
    for row in all_rows:
        m = row["method"]
        if m not in method_avgs:
            method_avgs[m] = {}
        for k, v in row.items():
            if k in ("sample_id", "method"):
                continue
            if isinstance(v, (int, float)) and not np.isnan(v):
                method_avgs[m].setdefault(k, []).append(v)

    # Key metrics to show in table
    key_metrics = ["PathADE", "PathFDE", "HeadingError", "SpeedStd", "RootJerk",
                   "CollisionFrameRate", "NonWalkableRootRate", "PenetrationRate", "PenetrationMean"]

    md_path = output_dir / "comparison.md"
    with open(md_path, "w") as f:
        f.write("# Guidance Experiment Comparison\n\n")
        f.write("| Method | " + " | ".join(key_metrics) + " |\n")
        f.write("|---" * (len(key_metrics) + 1) + "|\n")

        for m in args.names:
            if m not in method_avgs:
                continue
            avgs = method_avgs[m]
            vals = []
            for km in key_metrics:
                if km in avgs and avgs[km]:
                    vals.append(f"{np.mean(avgs[km]):.4f}")
                else:
                    vals.append("N/A")
            f.write(f"| {m} | " + " | ".join(vals) + " |\n")

        # Comparison sections
        f.write("\n## Key Comparisons\n\n")

        if "baseline" in args.names and "path_only" in args.names:
            f.write("### Path-Guidance vs Baseline\n\n")
            for km in ["PathADE", "PathFDE", "HeadingError"]:
                if km in method_avgs.get("baseline", {}) and km in method_avgs.get("path_only", {}):
                    base = np.mean(method_avgs["baseline"][km])
                    path = np.mean(method_avgs["path_only"][km])
                    delta = (path - base) / (base + 1e-8) * 100
                    f.write(f"- **{km}**: {base:.4f} → {path:.4f} ({delta:+.1f}%)\n")
            f.write("\n")

        if "path_only" in args.names and "path_scene" in args.names:
            f.write("### Path+Scene-Guidance vs Path-Only\n\n")
            for km in ["CollisionFrameRate", "NonWalkableRootRate", "PenetrationRate"]:
                if km in method_avgs.get("path_only", {}) and km in method_avgs.get("path_scene", {}):
                    po = np.mean(method_avgs["path_only"][km])
                    ps = np.mean(method_avgs["path_scene"][km])
                    delta = (ps - po) / (po + 1e-8) * 100
                    f.write(f"- **{km}**: {po:.4f} → {ps:.4f} ({delta:+.1f}%)\n")
            f.write("\n")

    log.info(f"CSV saved: {csv_path}")
    log.info(f"Markdown saved: {md_path}")


if __name__ == "__main__":
    main()
