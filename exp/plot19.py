# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage19.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
def g(m, s, k):
    v = [r[k] for r in R if r["mode"] == m and r["seal"] == s and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
W, H, L, T, Bt = 1180, 760, 150, 165, 480
ARMS = [("park", 0), ("prov", 0), ("prov", 2), ("prov", 5), ("prov", 10)]
LAB = ["一犹豫就查", "溯源 seal=0", "seal=2", "seal=5", "seal=10"]
COL = [(150, 150, 160), (60, 130, 230), (235, 110, 60), (230, 170, 40), (200, 60, 60)]
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "阶段 19：证据溯源 + 独立度传播 —— 方向对了，但盖章把记忆削没了", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "真实 MNIST / 污染 40% / 注入坏锚 / 5 seeds；盖章 = 花 oracle 给「自判」的条目做独立确认", font=F(14), fill=(110, 110, 120))
PANELS = [("总体 ACC", "ACC_final", 0.80, "%.3f"), ("任务0 类 ACC(坏锚现场)", "ACC_task0", 1.00, "%.3f"),
          ("保留记忆条数", "kept", 1100.0, "%.0f")]
for pi, item in enumerate(PANELS):
    pt, key, ymax, fmt = item
    ox = L + pi * 315
    def py(v, ymax=ymax):
        return Bt - (Bt - T) * min(max(v, 0), ymax) / ymax
    d.text((ox - 70, 112), pt, font=F(15), fill=(40, 40, 50))
    d.line([(ox - 70, 134), (ox + 170, 134)], fill=(200, 200, 210))
    for gv in np.arange(0, ymax + 1e-9, ymax / 4):
        y = py(gv)
        d.line([(ox - 70, y), (ox + 170, y)], fill=(234, 234, 240))
        d.text((ox - 132, y - 9), fmt % gv, font=F(12), fill=(130, 130, 140))
    for i, (m, s) in enumerate(ARMS):
        x = ox + i * 44
        v = g(m, s, key)
        y = py(v)
        d.rectangle([x, y, x + 34, Bt], fill=COL[i], outline=(255, 255, 255))
        d.text((x - 6, y - 19), fmt % v, font=F(11), fill=(50, 50, 60))
        d.text((x - 8, Bt + 9), LAB[i][:6], font=F(10), fill=(70, 70, 80))
    d.line([(ox - 70, T - 12), (ox - 70, Bt)], fill=(110, 110, 120), width=2)
    d.line([(ox - 70, Bt), (ox + 180, Bt)], fill=(110, 110, 120), width=2)
ANNS = [
    ("方向是对的：任务0 类（坏锚现场）随盖章预算单调上升：0.723 -> 0.781 -> 0.856 -> 0.913。", (20, 140, 70)),
    ("但总体 ACC 反向崩塌：0.533 -> 0.497 -> 0.474 -> 0.250。看第三幅就明白了 ——", (185, 50, 50)),
    ("保留记忆从 687 掉到 330。盖章会纠正「自判收错了」的条目，而在 40% 污染下这种纠正是大量的；", (185, 50, 50)),
    ("删得越多，记忆覆盖率越低，其他任务越没支撑。「删错」和「留够」在这个场景里直接冲突。", (185, 50, 50)),
    ("性价比反而是不盖章的溯源最好：seal=0 用 190 次查询（基线 690）拿到 0.533，每单位查询效率高 2.8 倍。", (20, 90, 180)),
    ("同一个教训第二次出现：阶段15 学到「审计该降权不该撤销」，这里盖章也该降独立度，而不是删记忆。", (20, 90, 180)),
]
for i, item in enumerate(ANNS):
    d.text((L - 20, 528 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage19_provenance.png"))
print("saved")
