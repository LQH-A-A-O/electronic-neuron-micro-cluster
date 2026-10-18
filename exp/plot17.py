# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage17.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
def g(m, k):
    v = [r[k] for r in R if r["mode"] == m and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
W, H, L, T, Bt = 1180, 720, 150, 150, 460
ARMS = ["one_phase", "two_phase", "two_phase_0"]
LAB = {"one_phase": "一犹豫就查(基线)", "two_phase": "放置", "two_phase_0": "放置+不查"}
COL = {"one_phase": (150, 150, 160), "two_phase": (30, 170, 90), "two_phase_0": (60, 130, 230)}
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "阶段 17：中立票「放置」—— 省了 63% 的 oracle 账单，但准确率掉 0.12", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "真实 MNIST / 污染 40% / 注入坏锚 / 3 seeds；放置 = 先筛确定数据，再用确定批当参照处理中立票", font=F(14), fill=(110, 110, 120))
PANELS = [("总体 ACC", "ACC_final", 0.80, "%.3f"), ("side 查询次数", "side", 750.0, "%.0f"),
          ("放置解决的中立票", "parked_resolved", 550.0, "%.0f")]
for pi, item in enumerate(PANELS):
    pt, key, ymax, fmt = item
    ox = L + pi * 300
    def py(v, ymax=ymax):
        return Bt - (Bt - T) * min(max(v, 0), ymax) / ymax
    d.text((ox - 40, 105), pt, font=F(15), fill=(40, 40, 50))
    d.line([(ox - 40, 127), (ox + 200, 127)], fill=(200, 200, 210))
    for gv in np.arange(0, ymax + 1e-9, ymax / 4):
        y = py(gv)
        d.line([(ox - 40, y), (ox + 215, y)], fill=(234, 234, 240))
        d.text((ox - 88, y - 9), (fmt % gv), font=F(12), fill=(130, 130, 140))
    for i, arm in enumerate(ARMS):
        x = ox + i * 74
        v = g(arm, key)
        y = py(v)
        d.rectangle([x, y, x + 58, Bt], fill=COL[arm], outline=(255, 255, 255))
        d.text((x + 1, y - 19), fmt % v, font=F(12), fill=(50, 50, 60))
        d.text((x - 10, Bt + 9), LAB[arm][:6], font=F(11), fill=(70, 70, 80))
    d.line([(ox - 40, T - 12), (ox - 40, Bt)], fill=(110, 110, 120), width=2)
    d.line([(ox - 40, Bt), (ox + 225, Bt)], fill=(110, 110, 120), width=2)
ANNS = [
    ("放置确实有效：side 查询从 691 砍到 257（-63%），而且用确定批当参照**免费解决了 476/573 张中立票**。", (20, 140, 70)),
    ("但总体 ACC 从 0.717 掉到 0.595，任务0 类从 0.945 掉到 0.863 —— 放置的判决质量不如真值。", (185, 50, 50)),
    ("根因：**「确定批」不等于「正确批」。** 锚和 peer 如果同源偏差，确定批就会系统性地错，", (185, 50, 50)),
    ("拿它当参照只会把偏差固化 —— 和阶段 11 的「同源污染」是同一个病。", (185, 50, 50)),
    ("换个角度看也有价值：每单位查询换来的准确率反而高 2.2 倍（1.04e-3 -> 2.32e-3）。", (20, 90, 180)),
    ("要让放置真正可用，参照批必须先经过一次**独立校验**（低比例抽检过的确定批才算数）。", (20, 90, 180)),
]
for i, item in enumerate(ANNS):
    d.text((L - 20, 512 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage17_parking.png"))
print("saved")
