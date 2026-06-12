#!/usr/bin/env python
"""Build a PowerPoint report from the 30-sample LINGO experiment results.

The environment does not provide python-pptx, so this writes a minimal PPTX
package directly using Office Open XML.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(FONT_PATH).exists():
    font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = PROJECT_ROOT / "outputs" / "retrain_mirrorfix50"
REPORT_DIR = RUN_ROOT / "ppt_report"
ASSET_DIR = REPORT_DIR / "assets"
OUT_PPTX = REPORT_DIR / "lingo_root_guidance_30sample_report.pptx"

SLIDE_W = 12192000
SLIDE_H = 6858000
EMU_PER_IN = 914400


def emu(inches: float) -> int:
    return int(round(inches * EMU_PER_IN))


def load_metrics() -> pd.DataFrame:
    sceneco = pd.read_csv(RUN_ROOT / "latest_ckpt_eval" / "summary" / "latest_metrics_summary.csv")
    sceneco["sample_count"] = 30
    sceneco["group"] = "Root指导 + SceneCo身体模型"

    base = pd.read_csv(RUN_ROOT / "eval_viz" / "test_smoke" / "summary_metrics.csv")
    base = base.rename(columns={"experiment": "experiment"})
    base["Pene"] = base["PenetrationMean"]
    base["sample_count"] = base["samples"]
    base["group"] = "原版Kimodo身体模型"

    traj_path = pd.read_csv(PROJECT_ROOT / "outputs" / "B1_E7_sceneco_trajco" / "path_metrics.csv")
    traj_scene = pd.read_csv(PROJECT_ROOT / "outputs" / "B1_E7_sceneco_trajco" / "scene_metrics.csv")
    traj = {
        "experiment": "TrajCo-B1_E7",
        "PathADE": float(pd.to_numeric(traj_path["PathADE"], errors="coerce").mean()),
        "PathFDE": float(pd.to_numeric(traj_path["PathFDE"], errors="coerce").mean()),
        "CollisionFrameRate": float(pd.to_numeric(traj_scene["CollisionFrameRate"], errors="coerce").mean()),
        "NonWalkableRootRate": float(pd.to_numeric(traj_scene["NonWalkableRootRate"], errors="coerce").mean()),
        "PenetrationRate": float(pd.to_numeric(traj_scene["PenetrationRate"], errors="coerce").mean()),
        "PenetrationMean": float(pd.to_numeric(traj_scene["PenetrationMean"], errors="coerce").mean()),
        "Pene": float(pd.to_numeric(traj_scene["PenetrationMean"], errors="coerce").mean()),
        "sample_count": len(traj_scene),
        "group": "已有TrajCo参考",
    }
    rows = []
    keep = [
        "experiment",
        "group",
        "sample_count",
        "PathADE",
        "PathFDE",
        "CollisionFrameRate",
        "NonWalkableRootRate",
        "PenetrationRate",
        "PenetrationMean",
        "Pene",
    ]
    rows.append(base[keep])
    rows.append(sceneco[keep])
    rows.append(pd.DataFrame([traj])[keep])
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(REPORT_DIR / "metrics_for_ppt.csv", index=False)
    return df


def make_metric_charts(df: pd.DataFrame) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = df[df["experiment"].isin(["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"])]
    metrics = ["PathADE", "CollisionFrameRate", "PenetrationRate", "Pene"]
    labels = plot_df["experiment"].tolist()
    colors = [
        "#A7D8B5" if e in {"E1", "E2", "E3"} else
        "#AFCBEF" if e in {"E4", "E5", "E6", "E7"} else
        "#F4D29A"
        for e in labels
    ]
    metric_titles = {
        "PathADE": "Root路径误差 PathADE",
        "CollisionFrameRate": "碰撞帧比例 CFR",
        "PenetrationRate": "穿透关节比例 PenRate",
        "Pene": "场景穿透指标 Pene",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.3))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(plot_df[metric], errors="coerce").fillna(0).to_numpy()
        bars = ax.bar(labels, vals, color=colors, edgecolor="#555555", linewidth=0.6)
        ax.set_title(metric_titles[metric])
        ax.grid(axis="y", alpha=0.25)
        ymax = max(float(vals.max()), 1e-6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("30个测试样本：原版Kimodo vs Root指导SceneCo")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "metric_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    table_cols = ["experiment", "group", "sample_count", "PathADE", "CollisionFrameRate", "PenetrationRate", "Pene"]
    table_df = df[table_cols].copy()
    for col in ["PathADE", "CollisionFrameRate", "PenetrationRate", "Pene"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    fig, ax = plt.subplots(figsize=(14.5, 5.7))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=["实验", "类别", "样本数", "PathADE", "CFR", "PenRate", "Pene"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2F4858")
            cell.set_text_props(color="white", weight="bold")
        elif table_df.iloc[r - 1]["group"] == "原版Kimodo身体模型":
            cell.set_facecolor("#EAF6EE")
        elif table_df.iloc[r - 1]["group"] == "已有TrajCo参考":
            cell.set_facecolor("#EFE5FA")
        else:
            cell.set_facecolor("#F7FAFF")
    ax.set_title("本报告使用的量化指标", fontsize=14, weight="bold", pad=16)
    fig.savefig(ASSET_DIR / "metrics_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_root_guidance_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off")
    boxes = [
        (0.04, 0.64, 0.25, "Root来源", "energy / classifier\nhybrid / GT / raw3d投影"),
        (0.37, 0.64, 0.25, "外部Root", "5D归一化root轨迹\n作为external_root输入"),
        (0.70, 0.64, 0.25, "去噪时固定Root", "每一步前后写回\ncur_motion[root_slice]"),
        (0.20, 0.18, 0.25, "Stage2身体模型", "跳过root预测器\n只生成局部身体动作"),
        (0.53, 0.18, 0.25, "SceneCo条件", "场景特征进入\nbody denoiser"),
        (0.79, 0.38, 0.18, "输出", "固定root + 生成身体\n计算路径/场景指标"),
    ]
    for x, y, w, title, desc in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, 0.20, fc="#E8F1FA", ec="#355C7D", lw=1.5, transform=ax.transAxes))
        ax.text(x + 0.012, y + 0.135, title, fontsize=11.5, weight="bold", transform=ax.transAxes)
        ax.text(x + 0.012, y + 0.045, desc, fontsize=8.8, transform=ax.transAxes, linespacing=1.35)
    arrows = [
        ((0.29, 0.74), (0.37, 0.74)),
        ((0.62, 0.74), (0.70, 0.74)),
        ((0.82, 0.64), (0.40, 0.38)),
        ((0.45, 0.28), (0.53, 0.28)),
        ((0.78, 0.28), (0.79, 0.47)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#333333"})
    ax.text(0.05, 0.08, "Stage2训练混合：E4-E6/E8-E9使用0.3 GT root + 0.7引导root；E7/E10使用GT或投影GT root。", fontsize=10, transform=ax.transAxes)
    fig.savefig(ASSET_DIR / "root_guidance_diagram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_existing_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "experiment_structure.png": RUN_ROOT / "eval_viz" / "figures" / "experiment_structure.png",
        "scene_E2.png": ASSET_DIR / "E2_scene.png",
        "scene_E5.png": ASSET_DIR / "E5_scene.png",
        "scene_E6.png": ASSET_DIR / "E6_scene.png",
        "scene_E7.png": ASSET_DIR / "E7_scene.png",
        "scene_E9.png": ASSET_DIR / "E9_scene.png",
        "scene_E10.png": ASSET_DIR / "E10_scene.png",
    }
    for name, src in sources.items():
        if src.exists() and src.parent != ASSET_DIR:
            shutil.copy2(src, ASSET_DIR / name)


def esc(s: str) -> str:
    return escape(str(s), {'"': "&quot;"})


def text_shape(shape_id: int, x: float, y: float, w: float, h: float, text: str, font_size: int = 20, bold: bool = False, color: str = "1F2933") -> str:
    paragraphs = str(text).split("\n")
    ps = []
    for para in paragraphs:
        lines = para if para else " "
        b = ' b="1"' if bold else ""
        ps.append(
            f'<a:p><a:r><a:rPr lang="en-US" sz="{font_size*100}"{b}>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr><a:t>{esc(lines)}</a:t></a:r></a:p>'
        )
    return f'''
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{"".join(ps)}</p:txBody>
</p:sp>'''


def rect_shape(shape_id: int, x: float, y: float, w: float, h: float, fill: str = "FFFFFF", line: str = "D0D7DE") -> str:
    return f'''
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="Rect {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln></p:spPr>
</p:sp>'''


def pic_shape(shape_id: int, rid: str, x: float, y: float, w: float, h: float) -> str:
    return f'''
<p:pic>
  <p:nvPicPr><p:cNvPr id="{shape_id}" name="Picture {shape_id}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
</p:pic>'''


def slide_xml(elements: list[str]) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {''.join(elements)}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def write_pptx(slides: list[dict]) -> None:
    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_PPTX, "w", compression=zipfile.ZIP_DEFLATED) as z:
        overrides = [
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]
        for idx in range(1, len(slides) + 1):
            overrides.append(f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        z.writestr("[Content_Types].xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
{''.join(overrides)}
</Types>''')
        z.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''')
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        z.writestr("docProps/core.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>LINGO Root Guidance Report</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>''')
        z.writestr("docProps/app.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application><PresentationFormat>16:9</PresentationFormat><Slides>{len(slides)}</Slides></Properties>''')
        slide_ids = []
        pres_rels = []
        for idx in range(1, len(slides) + 1):
            slide_ids.append(f'<p:sldId id="{255+idx}" r:id="rId{idx}"/>')
            pres_rels.append(f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx}.xml"/>')
        z.writestr("ppt/presentation.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldIdLst>{''.join(slide_ids)}</p:sldIdLst><p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>''')
        z.writestr("ppt/_rels/presentation.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(pres_rels)}</Relationships>''')
        media_idx = 1
        for idx, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{idx}.xml", slide["xml"])
            rels = []
            for rid, img_path in slide.get("rels", []):
                target = f"../media/image{media_idx}.png"
                rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>')
                z.write(img_path, f"ppt/media/image{media_idx}.png")
                media_idx += 1
            z.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>''')


def build_slides(df: pd.DataFrame) -> list[dict]:
    slides = []
    sid = 2

    def title(t, sub=""):
        nonlocal sid
        elems = [
            text_shape(sid, 0.6, 0.6, 12.0, 0.8, t, 34, True, "102A43"),
            text_shape(sid + 1, 0.65, 1.55, 11.5, 1.0, sub, 18, False, "334E68"),
            text_shape(sid + 2, 0.65, 6.6, 11.5, 0.3, "outputs/retrain_mirrorfix50/latest_ckpt_eval | 30 samples per experiment unless noted", 9, False, "627D98"),
        ]
        sid += 3
        slides.append({"xml": slide_xml(elems), "rels": []})

    title("LINGO Root Guidance / SceneCo Report", "Repaired dataset run: mirror fix + raw-scene axis fix\nComparison against original Kimodo body and existing TrajCo reference")

    elems = [
        text_shape(sid, 0.45, 0.35, 12.4, 0.45, "Experiment configuration", 28, True),
        text_shape(sid + 1, 0.65, 1.05, 5.9, 4.9,
                   "Original Kimodo body baselines\n"
                   "E1: energy-guided root + original body\n"
                   "E2: classifier-guided root + original body\n"
                   "E3: hybrid root + original body\n\n"
                   "Root-guided SceneCo body\n"
                   "E4: energy root + Stage2 SceneCo\n"
                   "E5: classifier root + Stage2 SceneCo\n"
                   "E6: hybrid root + Stage2 SceneCo\n"
                   "E7: GT root + Stage2 SceneCo", 15),
        text_shape(sid + 2, 6.9, 1.05, 5.7, 4.9,
                   "Raw3D/projected variants\n"
                   "E8: E5 classifier root projected to raw3d walkable region\n"
                   "E9: E6 hybrid root projected to raw3d walkable region\n"
                   "E10: GT root projected to walkable region\n\n"
                   "TrajCo comparisons\n"
                   "Existing B1_E7: Stage2 SceneCo + TrajCo body, full-val reference\n"
                   "New root-stage TrajCo: running; Stage1 root checkpoint only so far", 15),
    ]
    sid += 3
    slides.append({"xml": slide_xml(elems), "rels": []})

    elems = [
        text_shape(sid, 0.45, 0.35, 12.4, 0.45, "How root guidance is injected", 28, True),
        pic_shape(sid + 1, "rId1", 0.65, 1.05, 11.9, 4.9),
        text_shape(sid + 2, 0.65, 6.0, 11.8, 0.55, "Key implementation: external_root is fixed on root_slice during denoising; root_model is skipped for body generation. SceneCo conditions the body denoiser.", 13, False, "334E68"),
    ]
    sid += 3
    slides.append({"xml": slide_xml(elems), "rels": [("rId1", ASSET_DIR / "root_guidance_diagram.png")]})

    elems = [
        text_shape(sid, 0.45, 0.25, 12.4, 0.45, "Quantitative metrics", 28, True),
        pic_shape(sid + 1, "rId1", 0.3, 0.9, 12.7, 5.25),
        text_shape(sid + 2, 0.45, 6.25, 12.0, 0.45, "Pene is reported as PenetrationMean. TrajCo-B1_E7 is full-val and not the same 30-sample subset.", 11, False, "627D98"),
    ]
    sid += 3
    slides.append({"xml": slide_xml(elems), "rels": [("rId1", ASSET_DIR / "metrics_table.png")]})

    elems = [
        text_shape(sid, 0.45, 0.25, 12.4, 0.45, "Metric trends across E1-E10", 28, True),
        pic_shape(sid + 1, "rId1", 0.45, 0.85, 12.1, 5.65),
    ]
    sid += 2
    slides.append({"xml": slide_xml(elems), "rels": [("rId1", ASSET_DIR / "metric_bars.png")]})

    scene_imgs = [
        ("Original Kimodo: E2", "E2_scene.png", 0.55, 1.0),
        ("SceneCo: E6", "E6_scene.png", 3.75, 1.0),
        ("Oracle root: E7", "E7_scene.png", 6.95, 1.0),
        ("Raw3D: E9", "E9_scene.png", 10.15, 1.0),
    ]
    elems = [text_shape(sid, 0.45, 0.25, 12.4, 0.45, "Scene-action visualizations", 28, True)]
    rels = []
    for i, (label, filename, x, y) in enumerate(scene_imgs, start=1):
        elems.append(pic_shape(sid + i, f"rId{i}", x, y, 2.65, 2.65))
        elems.append(text_shape(sid + 10 + i, x, y + 2.72, 2.65, 0.32, label, 11, True))
        rels.append((f"rId{i}", ASSET_DIR / filename))
    elems.append(text_shape(sid + 20, 0.65, 5.95, 12.0, 0.6, "Full mp4 files are under outputs/retrain_mirrorfix50/latest_ckpt_eval/videos/scene_actions and eval_viz/videos/scene_actions.", 11, False, "627D98"))
    sid += 21
    slides.append({"xml": slide_xml(elems), "rels": rels})

    elems = [
        text_shape(sid, 0.45, 0.3, 12.4, 0.45, "Comparison summary", 28, True),
        rect_shape(sid + 1, 0.65, 1.05, 3.8, 4.8, "EAF6EE"),
        text_shape(sid + 2, 0.85, 1.25, 3.4, 4.3, "Original Kimodo body\n\nBest baseline in this sample: E2\nCFR=0.0317, PenRate=0.0096\n\nStrong on collision for the 30-sample baseline, but no SceneCo body adaptation.", 15),
        rect_shape(sid + 3, 4.75, 1.05, 3.8, 4.8, "EAF2FF"),
        text_shape(sid + 4, 4.95, 1.25, 3.4, 4.3, "Root-guided SceneCo\n\nBest current SceneCo variant: E7 oracle root\nCFR=0.0866, PenRate=0.0131\n\nE6 is best among learned guided roots: CFR=0.1504, PenRate=0.0195.", 15),
        rect_shape(sid + 5, 8.85, 1.05, 3.8, 4.8, "F2E9FF"),
        text_shape(sid + 6, 9.05, 1.25, 3.4, 4.3, "TrajCo\n\nExisting B1_E7 full-val reference:\nCFR=0.3382, PenRate=0.0913, Pene=0.0121\n\nNew root-stage TrajCo comparison is still running; no body metrics yet.", 15),
    ]
    sid += 7
    slides.append({"xml": slide_xml(elems), "rels": []})

    elems = [
        text_shape(sid, 0.45, 0.3, 12.4, 0.45, "Takeaways and next steps", 28, True),
        text_shape(sid + 1, 0.8, 1.15, 11.7, 4.8,
                   "1. Root guidance path is now testable and visualizable from one run root.\n"
                   "2. Among current SceneCo variants, E6 is the strongest learned-root setting in the 30-sample report.\n"
                   "3. E7 confirms the body stage can work well when root is oracle/clean.\n"
                   "4. E8-E10 raw3d/projected roots reduce path error, but increase root non-walkable/CFR in this small sample; inspect videos before drawing final conclusions.\n"
                   "5. The 1-hour 300-sample evaluation is running in latest_ckpt_eval_1h and should replace this 30-sample table once complete.", 18),
    ]
    sid += 2
    slides.append({"xml": slide_xml(elems), "rels": []})
    return slides


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    make_metric_charts(df)
    make_root_guidance_diagram()
    copy_existing_assets()
    uno_script = PROJECT_ROOT / "scripts" / "build_lingo_ppt_report_uno.py"
    subprocess.run(["/usr/bin/python3", str(uno_script)], check=True)
    print(json.dumps({"pptx": str(OUT_PPTX), "slides": 8, "metrics": str(REPORT_DIR / "metrics_for_ppt.csv")}, indent=2))


if __name__ == "__main__":
    main()
