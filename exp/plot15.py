# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage15.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H, L, T, Bt = 1180, 700, 130, 150, 460
LAB = {"hard": "硬判决撤锚", "weighted": "降权", "weighted_quarantine": "降权+观察区"}
COL = {"hard": (150, 150, 160), "weighted": (60, 130, 230), "weighted_quarantine": (30, 170, 90)}
SHOW = ["hard", "weighted", "weighted_quarantine"]
def m(mode, k):
    v = [r[k] for r in R if r["mode"] == mode and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 28), "阶段 15：审计从「撤销」改成「降权」—— 第一次拿到正面收益", font=F(22), fill=(20, 20, 30))
d.text((L - 20, 64), "同源污染 60% + 注入坏锚 / 5 任务 / 3 seeds；锚不再有生杀权，只有一个 0~1 的可信度", font=F(15), fill=(110, 110, 120))
PANELS = [("总体 ACC", "ACC_final", 0.70), ("任务0 区域（坏锚现场）", "ACC_task0", 0.10),
          ("任务1 区域（原被误杀）", "ACC_task1", 1.00)]
for pi, item in enumerate(PANELS):
    pt, key, ymax = item
    ox = L + pi * 300
    def py(v, ymax=ymax):
        return Bt - (Bt - T) * min(max(v, 0), ymax) / ymax
    d.text((ox, 108), pt, font=F(15), fill=(40, 40, 50))
    d.line([(ox, 130), (ox + 230, 130)], fill=(200, 200, 210))
    for gv in np.arange(0, ymax + 1e-9, ymax / 4):
        y = py(gv)
        d.line([(ox, y), (ox + 250, y)], fill=(234, 234, 240))
        d.text((ox - 44, y - 9), "%.2f" % gv, font=F(12), fill=(130, 130, 140))
    for i, mode in enumerate(SHOW):
        x = ox + i * 80
        v = m(mode, key)
        y = py(v)
        d.rectangle([x, y, x + 60, Bt], fill=COL[mode], outline=(255, 255, 255))
        d.text((x + 2, y - 19), "%.3f" % v, font=F(13), fill=(50, 50, 60))
        d.text((x - 4, Bt + 10), LAB[mode][:7], font=F(11), fill=(70, 70, 80))
    d.line([(ox, T - 12), (ox, Bt)], fill=(110, 110, 120), width=2)
    d.line([(ox, Bt), (ox + 255, Bt)], fill=(110, 110, 120), width=2)
ANNS = [
    ("硬判决（一致率 < 0.8 就撤锚）总体 0.529；换成降权后 0.592（+0.063），任务1 稳定 0.920，不再被误杀。", (20, 140, 70)),
    ("降权 + 观察区抽检 0.599，任务0 从 0.030 抬到 0.073 —— 目前最好的一组。", (20, 140, 70)),
    ("但任务0 仍然只有 0.073。同源污染下 60% 的观测是脏的且互相印证，任何共识类通道都系统性错误，", (185, 50, 50)),
    ("只有真值查询能分辨 —— 这一条从阶段 11 到 15 一次都没变过。", (185, 50, 50)),
    ("注意：只画了三个不同的臂。quarantine 与 weighted_quarantine 在本 runner 里数字完全相同，", (60, 60, 70)),
    ("因为 quarantine 模式走的也是加权审计 —— 实际只有三组可比，报告里如实写了。", (60, 60, 70)),
]
for i, item in enumerate(ANNS):
    txt, col = item
    d.text((L - 20, 508 + i * 23), txt, font=F(14), fill=col)
img.save(os.path.join(OUT, "stage15_weight.png"))
print("saved")
