# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage14.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H, L, T, Bt = 1180, 700, 140, 150, 470
MODES = ["audit_revisit", "quarantine", "quarantine_strike"]
LAB = {"audit_revisit": "阶段13终点", "quarantine": "观察区+抽检", "quarantine_strike": "+误判降级"}
COL = {"audit_revisit": (60, 130, 230), "quarantine": (235, 110, 60), "quarantine_strike": (200, 60, 60)}
def m(mode, k):
    v = [r[k] for r in R if r["mode"] == mode]
    return float(np.mean(v))
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 28), "阶段 14：取消单锚否决权 —— 机制在起作用，但副作用把它吃掉了", font=F(22), fill=(20, 20, 30))
d.text((L - 20, 64), "同源污染 60% + 注入坏锚 / 5 任务 / 3 seeds；纵轴 = 查询准确率", font=F(15), fill=(110, 110, 120))
for gv in np.arange(0, 0.61, 0.10):
    y = Bt - (Bt - T) * gv / 0.60
    d.line([(L - 20, y), (L + 780, y)], fill=(232, 232, 238))
    d.text((L - 62, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
PANELS = [("总体 ACC", 0.60), ("任务0 区域（坏锚作案现场）", 0.14), ("任务1 区域（被误伤）", 0.92)]
for pi, (pt, ymax) in enumerate(PANELS):
    ox = L - 20 + pi * 280
    d.text((ox, 118), pt, font=F(15), fill=(40, 40, 50))
    d.line([(ox, 140), (ox + 200, 140)], fill=(200, 200, 210))
    for i, mode in enumerate(MODES):
        x = ox + i * 66
        key = "ACC_final" if pi == 0 else ("ACC_task0" if pi == 1 else "ACC_task1")
        val = m(mode, key)
        y = Bt - (Bt - T) * min(val, ymax) / ymax
        d.rectangle([x, y, x + 50, Bt], fill=COL[mode], outline=(255, 255, 255))
        d.text((x + 1, y - 19), "%.3f" % val, font=F(12), fill=(50, 50, 60))
        d.text((x - 4, Bt + 10), LAB[mode][:5], font=F(11), fill=(70, 70, 80))
    d.text((ox - 2, Bt + 32), pt.split("（")[0], font=F(13), fill=(90, 90, 100))
d.line([(L - 20, T - 12), (L - 20, Bt)], fill=(110, 110, 120), width=2)
d.line([(L - 20, Bt), (L + 800, Bt)], fill=(110, 110, 120), width=2)
for i, (txt, col) in enumerate([
    ("观察区确实在工作：抽检放行了 31 条被坏锚误拒的干净知识（基线是 0 条）。", (20, 140, 70)),
    ("但量级远远不够 —— 任务0 区域 0.097 -> 0.073，没救回来。", (185, 50, 50)),
    ("更大的副作用：额外的抽检样本让审计**有足够证据**去降级「不完美但有用」的锚，", (185, 50, 50)),
    ("任务1 从 0.85 掉到 0.08（它的锚被撤了，判定退化成最近邻，反而更差）。总体 0.529 -> 0.365。", (185, 50, 50)),
    ("根因：**审计是二元判决**（一致率 >= 0.8 留、< 0.8 撤）。而「撤掉一个不完美的锚」的代价，", (60, 60, 70)),
    ("比「留着一个不完美的锚」更大。正确的做法不是撤销，是**降权**。", (20, 90, 180))]):
    d.text((L - 20, 520 + i * 23), txt, font=F(15), fill=col)
img.save(os.path.join(OUT, "stage14_quarantine.png"))

# 给 stage14 结果补一个"任务1 区域"字段（从 debug 得到，重新算一遍写进 json 供报告引用）
print("saved")
