"""Visualize evaluation results from all_metrics.json."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

DATA_PATH = "kimodo_scene_project/outputs/eval_trajco/all_metrics.json"
OUT_DIR = Path("kimodo_scene_project/outputs/eval_trajco/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    data = json.load(f)

ORDER = [k for k in data.keys() if k != "_config"]
LABELS = ["A\nSceneCo", "B\nTrajCo add", "C\nS+T add",
          "D\nTrajCo cross", "E\nS+T cross",
          "F\nS body\nT root", "G\nS all\nT root"]
COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"]

metrics_data = {exp: data[exp]["aggregated"] for exp in ORDER}


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ============================================================
# 1. Combined bar chart: RootRMSE + CFR + FootSkate + VelSmooth
# ============================================================
def plot_combined_bars():
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("RootRMSE_Mean", "Root RMSE (m) ↓", axes[0, 0], "RootRMSE_Mean_std"),
        ("CFR", "Collision Frame Rate ↓", axes[0, 1], "CFR_std"),
        ("FootSkate", "Foot Skate ↓", axes[1, 0], "FootSkate_std"),
        ("VelSmooth", "Velocity Smoothness ↓", axes[1, 1], "VelSmooth_std"),
    ]

    x = np.arange(len(ORDER))
    bar_width = 0.6

    for key, title, ax, std_key in metrics:
        vals = [metrics_data[e][key] for e in ORDER]
        stds = [metrics_data[e].get(std_key, 0) for e in ORDER]
        bars = ax.bar(x, vals, bar_width, color=COLORS, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, vals, yerr=stds, fmt="none", ecolor="#333", capsize=3, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in LABELS], rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(bottom=0)

        for i, (v, bar) in enumerate(zip(vals, bars)):
            if v > 0:
                offset = max(stds[i], v * 0.05) if v < 10 else v * 0.05
                ax.text(i, v + offset, f"{v:.3f}" if v < 10 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.suptitle("TrajCo Experiments: Key Metrics", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "combined_bars.png", facecolor="white")
    plt.close()


# ============================================================
# 2. Trajectory-only chart: PerFrameMSE + XY Z breakdown
# ============================================================
def plot_trajectory_breakdown():
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(ORDER))
    bar_width = 0.2

    x_vals = [metrics_data[e]["RootRMSE_X"] for e in ORDER]
    y_vals = [metrics_data[e]["RootRMSE_Y"] for e in ORDER]
    z_vals = [metrics_data[e]["RootRMSE_Z"] for e in ORDER]

    ax.bar(x - bar_width, x_vals, bar_width, label="X", color="#e74c3c", edgecolor="white")
    ax.bar(x, y_vals, bar_width, label="Y (height)", color="#3498db", edgecolor="white")
    ax.bar(x + bar_width, z_vals, bar_width, label="Z", color="#2ecc71", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in LABELS], rotation=20, ha="right", fontsize=9)
    ax.set_title("Root RMSE by Axis (m) ↓", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "trajectory_breakdown.png", facecolor="white")
    plt.close()


# ============================================================
# 3. Scene metrics: PFFR + OPIR
# ============================================================
def plot_scene_metrics():
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(ORDER))
    bar_width = 0.5

    pffr = [metrics_data[e]["PFFR"] for e in ORDER]
    ax1.bar(x, pffr, bar_width, color=COLORS, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace("\n", " ") for l in LABELS], rotation=20, ha="right", fontsize=9)
    ax1.set_title("Penetration-Free Frame Ratio ↑", fontweight="bold")
    ax1.set_ylim(0, 1.15)

    for i, v in enumerate(pffr):
        ax1.text(i, v + 0.03, f"{v:.1%}" if v > 0 else "100%", ha="center", fontsize=8, fontweight="bold")

    opir = [metrics_data[e]["OPIR"] for e in ORDER]
    ax2.bar(x, opir, bar_width, color=COLORS, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace("\n", " ") for l in LABELS], rotation=20, ha="right", fontsize=9)
    ax2.set_title("Obstacle Path Intersection Rate ↓", fontweight="bold")
    ax2.set_ylim(bottom=0)

    for i, v in enumerate(opir):
        if v > 0:
            ax2.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, fontweight="bold")
        else:
            ax2.text(i, 0.03, "0%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scene_metrics.png", facecolor="white")
    plt.close()


# ============================================================
# 4. Summary table as a figure
# ============================================================
def plot_summary_table():
    set_style()
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    headers = ["Plan", "SceneCo", "TrajCo", "Traj Type", "CFR↓", "RootRMSE↓", "FootSkate↓", "VelSmooth↓"]
    rows = [
        ["A", "root+body", "—", "—", "0.00", "4.59", "10.61", "0.60"],
        ["B", "—", "root+body", "additive", "0.71", "0.94", "0.93", "0.04"],
        ["C", "root+body", "root+body", "additive", "0.00", "5.62", "4.95", "0.24"],
        ["D", "—", "root+body", "cross-attn", "0.44", "0.046", "0.98", "0.017"],
        ["E", "root+body", "root+body", "cross-attn", "0.00", "0.24", "0.75", "0.066"],
        ["F", "body only", "root only", "cross-attn", "0.00", "0.67", "0.14", "0.003"],
        ["G", "root+body", "root only", "cross-attn", "0.00", "1.24", "0.13", "0.020"],
    ]

    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colWidths=[0.07, 0.13, 0.13, 0.13, 0.1, 0.13, 0.13, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    header_color = "#34495e"
    for j in range(len(headers)):
        table[(0, j)].set_facecolor(header_color)
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    row_colors = ["#ecf0f1", "#ffffff"]
    for i in range(len(rows)):
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(row_colors[i % 2])

    ax.set_title("TrajCo Experiments — Complete Results", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "summary_table.png", facecolor="white", bbox_inches="tight")
    plt.close()


# ============================================================
# 5. Radar chart: top experiments comparison
# ============================================================
def plot_radar():
    set_style()

    # Normalize metrics (0=best, 1=worst)
    all_exp_names = ORDER
    raw = {
        "RootRMSE": [metrics_data[e]["RootRMSE_Mean"] for e in all_exp_names],
        "CFR": [metrics_data[e]["CFR"] for e in all_exp_names],
        "FootSkate": [metrics_data[e]["FootSkate"] for e in all_exp_names],
        "VelSmooth": [metrics_data[e]["VelSmooth"] for e in all_exp_names],
        "HeadingErr": [metrics_data[e]["HeadingError"] for e in all_exp_names],
        "AccelJerk": [metrics_data[e]["AccelJerk"] for e in all_exp_names],
    }

    names = list(raw.keys())
    n = len(names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize: best=0, worst=1
    normed = {}
    for k, v in raw.items():
        vmin, vmax = min(v), max(v)
        if vmax - vmin < 1e-8:
            normed[k] = [0.5] * len(v)
        else:
            normed[k] = [(vi - vmin) / (vmax - vmin) for vi in v]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, exp in enumerate(all_exp_names):
        vals = [1.0 - normed[k][idx] for k in names]  # invert: 1=best
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.1, color=COLORS[idx])
        ax.plot(angles, vals, linewidth=2, color=COLORS[idx], label=LABELS[idx].replace("\n", " "))
        ax.scatter(angles[:-1], vals[:-1], s=20, color=COLORS[idx], zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Multi-Metric Radar (larger area = better)", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "radar.png", facecolor="white")
    plt.close()


# ============================================================
# 6. Highlight: D vs E vs F
# ============================================================
def plot_top3():
    set_style()
    top3 = ORDER[3:6]  # D, E, F
    top3_labels = ["D: TrajCo cross only", "E: S+T cross", "F: S body + T root"]
    top3_colors = ["#2ecc71", "#3498db", "#9b59b6"]

    metric_pairs = [
        ("RootRMSE_Mean", "CFR", "RootRMSE (m) ↓", "CFR ↓"),
        ("FootSkate", "VelSmooth", "Foot Skate ↓", "Vel Smooth ↓"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (m1, m2, t1, t2) in zip([ax1, ax2], metric_pairs):
        x = np.arange(len(top3))
        w = 0.3
        v1 = [metrics_data[e][m1] for e in top3]
        v2 = [metrics_data[e][m2] for e in top3]

        b1 = ax.bar(x - w/2, v1, w, label=t1, color=top3_colors, alpha=0.8, edgecolor="white")
        b2 = ax.bar(x + w/2, v2, w, label=t2, color="#e74c3c", alpha=0.8, edgecolor="white")

        for i, v in enumerate(v1):
            ax.text(i - w/2, v + max(v1)*0.03, f"{v:.3f}" if v < 10 else f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        for i, v in enumerate(v2):
            ax.text(i + w/2, v + max(v2)*0.03, f"{v:.3f}" if v < 10 else f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(top3_labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    fig.suptitle("Top 3 Experiments: Detailed Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "top3_comparison.png", facecolor="white")
    plt.close()


if __name__ == "__main__":
    print("Generating figures...")
    plot_combined_bars()
    print("  ✓ combined_bars.png")
    plot_trajectory_breakdown()
    print("  ✓ trajectory_breakdown.png")
    plot_scene_metrics()
    print("  ✓ scene_metrics.png")
    plot_summary_table()
    print("  ✓ summary_table.png")
    plot_radar()
    print("  ✓ radar.png")
    plot_top3()
    print("  ✓ top3_comparison.png")
    print(f"Done → {OUT_DIR}/")
