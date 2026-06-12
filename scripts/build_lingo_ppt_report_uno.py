#!/usr/bin/env python3
"""Build the LINGO report through LibreOffice UNO.

Run with /usr/bin/python3 so the system `uno` module is available.
This produces a PowerPoint file using LibreOffice's own PPTX exporter,
which is more compatible than a hand-written OOXML package.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import uno
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = PROJECT_ROOT / "outputs" / "retrain_mirrorfix50"
REPORT_DIR = RUN_ROOT / "ppt_report"
ASSET_DIR = REPORT_DIR / "assets"
OUT_PPTX = REPORT_DIR / "lingo_root_guidance_30sample_report.pptx"
OUT_ODP = REPORT_DIR / "lingo_root_guidance_30sample_report.odp"
HOST = "127.0.0.1"
PORT = "2083"

SLIDE_W = 28000
SLIDE_H = 15750


def prop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def color(hex_value: str) -> int:
    return int(hex_value.replace("#", ""), 16)


def mm100(v: float) -> int:
    return int(round(v * 1000))


def file_url(path: Path) -> str:
    return uno.systemPathToFileUrl(str(path.resolve()))


def wait_for_office():
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    url = f"uno:socket,host={HOST},port={PORT};urp;StarOffice.ComponentContext"
    proc = None
    for attempt in range(20):
        try:
            return resolver.resolve(url), proc
        except Exception:
            if attempt == 0:
                proc = subprocess.Popen(
                    [
                        "soffice",
                        "--headless",
                        "--norestore",
                        "--nodefault",
                        "--nofirststartwizard",
                        f"--accept=socket,host={HOST},port={PORT};urp;StarOffice.ServiceManager",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            time.sleep(0.5)
    raise RuntimeError("LibreOffice UNO service did not start")


def set_text_style(shape, size=18, bold=False, font="Noto Sans CJK SC", font_color="#1F2933"):
    shape.CharFontName = font
    shape.CharHeight = float(size)
    shape.CharColor = color(font_color)
    shape.CharWeight = 150.0 if bold else 100.0
    shape.TextWordWrap = True


def add_text(doc, page, x, y, w, h, text, size=18, bold=False, font_color="#1F2933"):
    shape = doc.createInstance("com.sun.star.drawing.TextShape")
    shape.Position = Point(mm100(x), mm100(y))
    shape.Size = Size(mm100(w), mm100(h))
    shape.FillStyle = 0
    shape.LineStyle = 0
    page.add(shape)
    shape.String = text
    set_text_style(shape, size=size, bold=bold, font_color=font_color)
    return shape


def add_box(doc, page, x, y, w, h, fill="#FFFFFF", line="#D0D7DE"):
    shape = doc.createInstance("com.sun.star.drawing.RectangleShape")
    shape.Position = Point(mm100(x), mm100(y))
    shape.Size = Size(mm100(w), mm100(h))
    shape.FillColor = color(fill)
    shape.LineColor = color(line)
    page.add(shape)
    return shape


def add_image(doc, page, path, x, y, w, h):
    shape = doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
    shape.Position = Point(mm100(x), mm100(y))
    shape.Size = Size(mm100(w), mm100(h))
    shape.GraphicURL = file_url(path)
    page.add(shape)
    return shape


def blank_slide(doc, idx):
    pages = doc.getDrawPages()
    if idx == 0:
        page = pages.getByIndex(0)
        while page.getCount():
            page.remove(page.getByIndex(0))
    else:
        page = pages.insertNewByIndex(idx)
    page.Width = mm100(28.0)
    page.Height = mm100(15.75)
    return page


def load_metrics():
    rows = []
    with (REPORT_DIR / "metrics_for_ppt.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def fmt(row, key):
    return f"{float(row[key]):.4f}"


def build_slides(doc, rows):
    page = blank_slide(doc, 0)
    add_text(doc, page, 1.3, 1.4, 25.5, 2.0, "LINGO Root Guidance / SceneCo实验报告", 31, True, "#102A43")
    add_text(
        doc,
        page,
        1.4,
        3.5,
        24.0,
        2.2,
        "数据修复后重跑：mirror修复 + raw-scene坐标轴修复\n"
        "对比对象：原版Kimodo身体模型、加入Root Guidance的SceneCo身体模型、已有TrajCo参考结果",
        18,
        False,
        "#334E68",
    )
    add_text(doc, page, 1.4, 14.7, 24.0, 0.5, "除特别说明外，本报告使用之前的30个测试样本", 9, False, "#627D98")

    page = blank_slide(doc, 1)
    add_text(doc, page, 1.0, 0.45, 25.0, 0.8, "实验配置总览：E1-E10到底分别是什么", 25, True)
    add_text(doc, page, 1.0, 1.28, 25.5, 0.6, "读法：Root来源决定全局轨迹；Body模型决定身体动作生成；SceneCo表示身体生成阶段是否使用场景条件。", 11, False, "#334E68")
    add_box(doc, page, 1.0, 2.05, 26.0, 11.6, "#F7FAFC", "#CBD5E1")
    add_text(
        doc,
        page,
        1.4,
        2.35,
        25.2,
        10.9,
        "原版Kimodo身体模型（不换body，只换root输入）\n"
        "E1：Root=energy引导生成；Body=原版Kimodo Stage2；无SceneCo。\n"
        "E2：Root=训练的root路径分类器/critic引导；Body=原版Kimodo Stage2；无SceneCo。\n"
        "E3：Root=energy + classifier混合引导；Body=原版Kimodo Stage2；无SceneCo。\n\n"
        "Root Guidance + SceneCo身体模型（固定外部root，再让SceneCo生成身体动作）\n"
        "E4：Root=energy引导；Body=重新训练的Stage2 SceneCo；训练混合0.3 GT root + 0.7 guided root。\n"
        "E5：Root=classifier引导；Body=重新训练的Stage2 SceneCo；训练混合0.3 GT root + 0.7 guided root。\n"
        "E6：Root=energy + classifier混合引导；Body=重新训练的Stage2 SceneCo；训练混合0.3 GT root + 0.7 guided root。\n"
        "E7：Root=GT真值root；Body=重新训练的Stage2 SceneCo；这是oracle root上限，不代表可部署方法。\n\n"
        "Raw3D / 可行走区域投影版本（先把root投影到raw3d walkable区域，再生成身体）\n"
        "E8：Root=E5的classifier root + raw3d投影；Body=Stage2 SceneCo。\n"
        "E9：Root=E6的hybrid root + raw3d投影；Body=Stage2 SceneCo。\n"
        "E10：Root=GT root + raw3d投影；Body=Stage2 SceneCo；用于看投影本身的影响。\n\n"
        "TrajCo：已有B1_E7是Stage2 SceneCo + TrajCo body的full-val参考；新root-stage TrajCo仍在跑，当前不放入30样本主表。",
        10.5,
    )

    page = blank_slide(doc, 2)
    add_text(doc, page, 1.0, 0.7, 25.0, 0.9, "Root Guidance怎么加入Kimodo/SceneCo", 25, True)
    add_image(doc, page, ASSET_DIR / "root_guidance_diagram.png", 1.3, 2.1, 25.0, 10.4)
    add_text(
        doc,
        page,
        1.4,
        13.5,
        24.8,
        0.8,
        "关键点：body阶段不再自己预测root，而是在每一步denoising前后把external_root写回root_slice；SceneCo只负责在给定root下生成场景条件身体动作。",
        12,
        False,
        "#334E68",
    )

    page = blank_slide(doc, 3)
    add_text(doc, page, 1.0, 0.5, 25.0, 0.9, "量化指标：30个测试样本", 25, True)
    add_image(doc, page, ASSET_DIR / "metrics_table.png", 0.6, 1.8, 26.8, 11.0)
    add_text(doc, page, 1.0, 13.7, 25.5, 0.6, "Pene按PenetrationMean汇报。TrajCo-B1_E7是full-val参考，不是同一组30个样本，主要用于量级对比。", 10, False, "#627D98")

    page = blank_slide(doc, 4)
    add_text(doc, page, 1.0, 0.5, 25.0, 0.9, "指标趋势：E1-E10对比", 25, True)
    add_image(doc, page, ASSET_DIR / "metric_bars.png", 0.9, 1.6, 25.8, 12.1)

    page = blank_slide(doc, 5)
    add_text(doc, page, 1.0, 0.5, 25.0, 0.9, "场景内动作可视化", 25, True)
    imgs = [
        ("E2 原版Kimodo + classifier root", "E2_scene.png", 1.2),
        ("E6 SceneCo + hybrid root", "E6_scene.png", 7.9),
        ("E7 SceneCo + GT root", "E7_scene.png", 14.6),
        ("E9 SceneCo + raw3d投影root", "E9_scene.png", 21.3),
    ]
    for label, filename, x in imgs:
        add_image(doc, page, ASSET_DIR / filename, x, 2.2, 5.6, 5.6)
        add_text(doc, page, x, 8.1, 5.8, 0.6, label, 11, True)
    add_text(doc, page, 1.2, 13.8, 25.0, 0.6, "完整动作视频在 latest_ckpt_eval/videos/scene_actions 和 eval_viz/videos/scene_actions 下。", 10, False, "#627D98")

    page = blank_slide(doc, 6)
    add_text(doc, page, 1.0, 0.6, 25.0, 0.9, "和原版Kimodo / TrajCo的对比结论", 25, True)
    groups = [
        ("原版Kimodo", "30样本里最好的baseline是E2：\nCFR=0.0317, PenRate=0.0096\n\n解释：原版body在这30个样本上碰撞指标强，但没有使用SceneCo身体阶段。", "#EAF6EE", 1.4),
        ("Root Guidance + SceneCo", "oracle root上限是E7：\nCFR=0.0866, PenRate=0.0131\n\n可部署的learned-root里，E6最好：\nCFR=0.1504, PenRate=0.0195。", "#EAF2FF", 9.7),
        ("TrajCo参考", "已有B1_E7 full-val参考：\nCFR=0.3382, PenRate=0.0913, Pene=0.0121\n\n新root-stage TrajCo还在训练/补实验，当前没有body指标。", "#F2E9FF", 18.0),
    ]
    for title, body, fill, x in groups:
        add_box(doc, page, x, 2.1, 7.4, 10.5, fill)
        add_text(doc, page, x + 0.4, 2.6, 6.7, 1.0, title, 16, True)
        add_text(doc, page, x + 0.4, 4.0, 6.6, 7.6, body, 13)

    page = blank_slide(doc, 7)
    add_text(doc, page, 1.0, 0.6, 25.0, 0.9, "当前结论和下一步", 25, True)
    add_text(
        doc,
        page,
        1.6,
        2.0,
        24.5,
        10.6,
        "1. Root guidance现在已经形成完整链路：外部root -> 固定root_slice -> SceneCo身体生成 -> 场景指标/路径指标。\n"
        "2. 30样本里，learned-root的SceneCo方案中E6最好；E7说明只要root干净，body阶段可以接近oracle表现。\n"
        "3. E8-E10说明raw3d投影能降低路径误差，但在这批样本里会提高CFR/NonWalkable，需要结合视频判断投影是否过度贴边。\n"
        "4. 原版Kimodo E2在这30样本上很强，所以最终结论不能只看30样本；应该用已经完成的300-sample结果替换主表继续比较。\n"
        "5. TrajCo目前只能放已有full-val参考；新root-stage TrajCo完成Stage2 body后再加入同口径比较。",
        17,
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ctx, proc = wait_for_office()
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, tuple())
    rows = load_metrics()
    build_slides(doc, rows)
    doc.storeAsURL(file_url(OUT_ODP), (prop("FilterName", "impress8"), prop("Overwrite", True)))
    doc.storeAsURL(file_url(OUT_PPTX), (prop("FilterName", "Impress MS PowerPoint 2007 XML"), prop("Overwrite", True)))
    doc.close(True)
    if proc is not None:
        proc.terminate()
    print(OUT_PPTX)


if __name__ == "__main__":
    main()
