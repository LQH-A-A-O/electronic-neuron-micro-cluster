# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage11.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

MODES = ["no_audit", "audit_passive", "audit_active"]
COL = {"no_audit": (150, 150, 160), "audit_passive": (235, 110, 60), "audit_active": (30, 170, 90)}
LAB = {"no_audit": "无审计", "audit_passive": "被动审计", "audit_active": "主动抽检"}

W, H = 1180, 700
L, T, Bt = 130, 150, 470
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 28), "阶段 11：锚审计 —— 能发现坏锚，但救不回准确率", font=F(23), fill=(20, 20, 30))
d.text((L - 20, 64), "任务0 被同源污染 60% + 启动时注入坏锚（= 那条错误规则）；5 任务 / 3 seeds", font=F(15), fill=(110, 110, 120))

def panel(ox, title, vals, ymax, fmt="%.3f"):
    d.text((ox, 118), title, font=F(16), fill=(40, 40, 50))
    d.line([(ox, 142), (ox + 210, 142)], fill=(200, 200, 210))
    for i, m in enumerate(MODES):
        x = ox + i * 78
        y = Bt - (Bt - T) * min(vals[i], ymax) / ymax
        d.rectangle([x, y, x + 58, Bt], fill=COL[m], outline=(255, 255, 255))
        d.text((x + 1, y - 20), fmt % vals[i], font=F(13), fill=(50, 50, 60))
        d.text((x + 6, Bt + 10), LAB[m], font=F(12), fill=(70, 70, 80))

panel(L - 20, "总体 ACC", [np.mean([r["ACC_final"] for r in R if r["mode"] == m]) for m in MODES], 0.70)
panel(L + 300, "任务0 区域 ACC（坏锚所在地）", [np.mean([r["ACC_task0"] for r in R if r["mode"] == m]) for m in MODES], 0.70)
panel(L + 620, "被降级的坏锚个数", [np.mean([r["demoted"] for r in R if r["mode"] == m]) for m in MODES], 3.0, "%.1f")
for i, (txt, col) in enumerate([
    ("无审计：任务0 区域 0.023 —— 坏锚和同源脏数据互相印证，门控**一条 side 查询都没有花**（强制=0）。", (185, 50, 50)),
    ("系统自信地错下去。这就是你第 4 条担心的场景，实测它比预想的更糟：连纠错的机会都不产生。", (185, 50, 50)),
    ("被动审计（只用门控自然花掉的查询）几乎无效：降级 0.3 个，任务0 仍是 0.030 —— 拿不到真值。", (200, 120, 40)),
    ("主动抽检有效的一半：降级 2.3 个坏锚，任务0 从 0.023 抬到 0.067（3 倍），但绝对值仍然不可用。", (20, 140, 70)),
    ("为什么救不回来：审计只处理了锚，**没处理被这个坏锚放行的那批脏记忆** —— 它们还在库里，还在被检索。", (60, 60, 70)),
    ("下一个机制因此很清楚：污染溯源 + 连带撤回（用问题锚回溯它当年放行了谁）。", (20, 90, 180))]):
    d.text((L - 20, 520 + i * 24), txt, font=F(15), fill=col)
img.save(os.path.join(OUT, "stage11_audit.png"))
print("saved")
