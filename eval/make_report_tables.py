"""Generate evaluation report tables from metrics JSON files (Chapter 18).

Produces all required tables:
  Table A: Parameter Protection (Chapter 13)
  Table B: Original Kimodo Regression (Chapter 14)
  Table C: Scene Adaptation (Chapter 15)
  Table D: Motion Quality (Chapter 16)
  Table E: Ablation Studies (Chapter 17)

Supports output formats: JSON, CSV, LaTeX.

Usage:
    python make_report_tables.py \
        --metrics_dir outputs/reports \
        --output_dir outputs/reports \
        --format latex
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_metrics(metrics_dir: Path) -> Dict[str, Dict]:
    result = {}
    for json_file in sorted(metrics_dir.glob("*.json")):
        with open(json_file) as f:
            result[json_file.stem] = json.load(f)
    return result


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


class LatexTable:
    def __init__(self, caption: str, label: str, headers: List[str]):
        self.caption = caption
        self.label = label
        self.headers = headers
        self.rows: List[List[str]] = []

    def add_row(self, row: List):
        self.rows.append([_fmt(v) for v in row])

    def render(self) -> str:
        nc = len(self.headers)
        align = "l" + "c" * (nc - 1)
        lines = [
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{{self.caption}}}",
            f"  \\label{{{self.label}}}",
            f"  \\begin{{tabular}}{{{{{align}}}}}",
            "    \\toprule",
        ]
        lines.append("    " + " & ".join(self.headers) + " \\\\")
        lines.append("    \\midrule")
        for row in self.rows:
            lines.append("    " + " & ".join(row) + " \\\\")
        lines.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)


class CSVTable:
    def __init__(self, caption: str, headers: List[str]):
        self.caption = caption
        self.headers = headers
        self.rows: List[List[str]] = []

    def add_row(self, row: List):
        self.rows.append([_fmt(v) for v in row])

    def render(self) -> str:
        lines = ["# " + self.caption]
        lines.append(",".join(self.headers))
        for row in self.rows:
            lines.append(",".join(str(v) for v in row))
        return "\n".join(lines)


TABLE_A_HEADERS = [
    "Check", "Passed", "FrozenParams", "FailedKeys", "MaxAbsDiff",
    "GradNormZero", "OptimizerClean", "AlphasAtInit", "KeysMatch",
]

TABLE_B_HEADERS = [
    "Model", "R@1↑", "FID↓", "Diversity↔", "KeyframeMPJPE↓",
    "EEError↓", "PathError↓", "WaypointError↓",
]

TABLE_C_HEADERS = [
    "Model", "CFR↓", "JCR↓", "MeanPen↓", "P95Pen↓", "OPIR↓",
    "TargetDist↓", "PFFR↑",
]

TABLE_D_HEADERS = [
    "Model", "FootSkate↓", "FootPen↓", "Floating↓", "VelSmooth↓",
    "AccelJerk↓", "BoneLenErr↓",
]

TABLE_E_HEADERS = [
    "Model", "GateZero_CFR", "GateZero_JCR",
    "RandomScene_ΔCFR", "EmptyScene_CFR", "ShuffledScene_ΔCFR",
]

TRAINING_TABLE_HEADERS = [
    "Step", "TrainLoss↓", "ValLoss↓", "PriorLoss↓",
    "SceneLoss↓", "Gate(root)", "Gate(body)", "GradNorm", "CFR_Val↓",
]


def make_table_a(metrics: Dict) -> LatexTable:
    table = LatexTable("Table A: Parameter Protection Checks",
                       "tab:params", TABLE_A_HEADERS)
    for model_name, model_data in metrics.items():
        frozen = model_data.get("13.1_frozen_param_diff", {})
        grad = model_data.get("13.2_frozen_grad_norm", {})
        ops = model_data.get("13.3_optimizer_param_check", {})
        gate = model_data.get("13.4_gate_zero_equivalence", {})
        keys = model_data.get("13.5_state_dict_keys", {})
        table.add_row([
            model_name,
            str(frozen.get("passed", "N/A")),
            frozen.get("total_checked", "N/A"),
            frozen.get("failed", "N/A"),
            _fmt(frozen.get("max_abs_diff")),
            _fmt(grad.get("checked", "N/A")),
            _fmt(ops.get("checked", "N/A")),
            _fmt(gate.get("all_zero", "N/A")),
            _fmt(keys.get("passed", "N/A")),
        ])
    return table


def make_table_b(metrics: Dict) -> LatexTable:
    table = LatexTable("Table B: Original Kimodo Capability Regression",
                       "tab:regression", TABLE_B_HEADERS)
    for model_name, model_data in metrics.items():
        b = model_data.get("B_regression", {})
        table.add_row([
            model_name,
            _fmt(b.get("r_precision")),
            _fmt(b.get("fid")),
            _fmt(b.get("diversity")),
            _fmt(b.get("keyframe_mpjpe")),
            _fmt(b.get("ee_error")),
            _fmt(b.get("path_error")),
            _fmt(b.get("waypoint_error")),
        ])
    return table


def make_table_c(metrics: Dict) -> LatexTable:
    table = LatexTable("Table C: Scene Adaptation Metrics",
                       "tab:scene", TABLE_C_HEADERS)
    for model_name, model_data in metrics.items():
        c = model_data.get("C_scene_adaptation", {})
        table.add_row([
            model_name,
            _fmt(c.get("CFR")),
            _fmt(c.get("JCR")),
            _fmt(c.get("MeanPen")),
            _fmt(c.get("P95Pen")),
            _fmt(c.get("OPIR")),
            _fmt(c.get("TargetDist")),
            _fmt(c.get("PFFR")),
        ])
    return table


def make_table_d(metrics: Dict) -> LatexTable:
    table = LatexTable("Table D: Motion Quality Metrics",
                       "tab:quality", TABLE_D_HEADERS)
    for model_name, model_data in metrics.items():
        d = model_data.get("D_motion_quality", {})
        table.add_row([
            model_name,
            _fmt(d.get("FootSkate")),
            _fmt(d.get("FootPenetration")),
            _fmt(d.get("FloatingRatio")),
            _fmt(d.get("VelSmooth")),
            _fmt(d.get("AccelJerk")),
            _fmt(d.get("BoneLenErr")),
        ])
    return table


def make_table_e(metrics: Dict) -> LatexTable:
    table = LatexTable("Table E: Ablation Studies",
                       "tab:ablation", TABLE_E_HEADERS)
    for model_name, model_data in metrics.items():
        e = model_data.get("E_ablation", {})
        table.add_row([
            model_name,
            _fmt(e.get("gate_zero_cfr")),
            _fmt(e.get("gate_zero_jcr")),
            _fmt(e.get("random_scene_delta_cfr")),
            _fmt(e.get("empty_scene_cfr")),
            _fmt(e.get("shuffled_scene_delta_cfr")),
        ])
    return table


def make_table_training(metrics: Dict) -> LatexTable:
    """Training monitoring table (md §18.4)."""
    table = LatexTable("Table F: Training Monitoring",
                       "tab:training", TRAINING_TABLE_HEADERS)
    train_log = metrics.get("training_log", {})
    steps = train_log.get("steps", [])
    if not steps:
        steps = [1000, 10000, 50000, 100000, 200000]
    for step in steps:
        log_entry = train_log.get(str(step), {})
        table.add_row([
            step,
            _fmt(log_entry.get("train_loss")),
            _fmt(log_entry.get("val_loss")),
            _fmt(log_entry.get("prior_loss")),
            _fmt(log_entry.get("scene_loss")),
            _fmt(log_entry.get("gate_root")),
            _fmt(log_entry.get("gate_body")),
            _fmt(log_entry.get("grad_norm")),
            _fmt(log_entry.get("cfr_val")),
        ])
    return table


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report tables (Chapter 18)")
    parser.add_argument("--metrics_dir", type=str,
                        default="kimodo_scene_project/outputs/reports")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/reports")
    parser.add_argument("--format", type=str, default="csv",
                        choices=["json", "csv", "latex"])
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(metrics_dir)

    if not data:
        print(f"No metrics found in {metrics_dir}; creating template with placeholder.")
        data = {"placeholder": {}}

    table_factories = {
        "table_a_frozen_params": make_table_a,
        "table_b_kimodo_regression": make_table_b,
        "table_c_scene_adaptation": make_table_c,
        "table_d_motion_quality": make_table_d,
        "table_e_ablation": make_table_e,
        "table_f_training_monitor": make_table_training,
    }

    fmt = args.format
    ext = ".tex" if fmt == "latex" else ".csv"

    for name, factory in table_factories.items():
        if fmt == "latex":
            table = factory(data)
            ext_used = ".tex"
        else:
            table_headers = {
                "table_a": TABLE_A_HEADERS,
                "table_b": TABLE_B_HEADERS,
                "table_c": TABLE_C_HEADERS,
                "table_d": TABLE_D_HEADERS,
                "table_e": TABLE_E_HEADERS,
            }
            tkey = name.replace("_frozen_params", "").replace("_kimodo_regression", "").replace("_scene_adaptation", "").replace("_motion_quality", "").replace("_ablation", "")
            headers = table_headers.get(tkey, ["Model", "Metric"])

            if fmt == "latex":
                table = factory(data)
                content = table.render()
            else:
                csv_t = CSVTable(name, headers)
                for model_name, model_data in data.items():
                    csv_t.add_row([model_name] + ["N/A"] * (len(headers) - 1))
                content = csv_t.render()
            ext_used = ".csv"

        if fmt == "latex":
            content = table.render()
            ext_used = ".tex"

        path = out_dir / f"{name}{ext_used}"
        with open(path, "w") as f:
            f.write(content)
        print(f"  Saved: {path}")

    if fmt == "json":
        with open(out_dir / "all_tables.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {out_dir / 'all_tables.json'}")

    print(f"\nReports saved to {out_dir}")


if __name__ == "__main__":
    main()
