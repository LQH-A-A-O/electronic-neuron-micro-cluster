# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage21.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H = 1180, 740
L, Rr, T, Bt = 150, 730, 160, 510
px = lambda v: L + (Rr - L) * v / 2100.0
py = lambda v: Bt - (Bt - T) * v / 1.0
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "G1（阶段 21）：门控 vs 主动学习 —— D1 最终通过", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "真实 MNIST / 污染 40% / 5 seeds；横轴 = oracle 查询次数，纵轴 = 总体 ACC", font=F(14), fill=(110, 110, 120))
for gv in np.arange(0, 1.001, 0.2):
    y = py(gv)
    d.line([(L, y), (Rr, y)], fill=(234, 234, 240))
    d.text((L - 52, y - 9), "%.1f" % gv, font=F(13), fill=(120, 120, 130))
for gv in range(0, 2101, 500):
    x = px(gv)
    d.line([(x, Bt), (x, Bt + 6)], fill=(120, 120, 130))
    d.text((x - 20, Bt + 12), str(gv), font=F(13), fill=(120, 120, 130))
d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
d.line([(L, Bt), (Rr + 10, Bt)], fill=(110, 110, 120), width=2)
def a(mode, B, k):
    v = [r[k] for r in R if r["mode"] == mode and r["B"] == B and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
BS = [5, 10, 20]
for mode, col, nm in (("al_random", (150, 150, 160), "随机标注"),
                      ("al_entropy", (235, 110, 60), "熵采样(主动学习)"),
                      ("al_margin", (60, 130, 230), "间隔采样(主动学习)")):
    pts = [(a(mode, b, "side"), a(mode, b, "ACC_final")) for b in BS]
    d.line([(px(x), py(y)) for x, y in pts], fill=col, width=4, joint="curve")
    for (x, y), b in zip(pts, BS):
        d.ellipse([px(x) - 6, py(y) - 6, px(x) + 6, py(y) + 6], fill=col)
        d.text((px(x) - 34, py(y) + 10), "B=%d" % b, font=F(11), fill=col)
    LY = {"al_random": -46, "al_entropy": 6, "al_margin": 34}[mode]
    d.text((px(pts[-1][0]) + 46, py(pts[-1][1]) + LY), nm, font=F(14), fill=col)
gx, gy = a("gate", -1, "side"), a("gate", -1, "ACC_final")
d.rectangle([px(gx) - 9, py(gy) - 9, px(gx) + 9, py(gy) + 9], fill=(30, 170, 90))
d.text((px(gx) + 16, py(gy) - 8), "门控: %d 次 -> %.3f" % (gx, gy), font=F(15), fill=(20, 140, 70))
NEED = 625
d.line([(px(NEED), Bt), (px(NEED), py(gy))], fill=(120, 120, 180), width=2)
d.line([(L, py(gy)), (px(NEED), py(gy))], fill=(120, 120, 180), width=2)
d.text((px(NEED) + 10, py(gy) + 26), "最好的主动学习要花约 %d 次" % NEED, font=F(14), fill=(80, 80, 140))
fx, fy = a("full", -1, "side"), a("full", -1, "ACC_final")
d.ellipse([px(fx) - 7, py(fy) - 7, px(fx) + 7, py(fy) + 7], fill=(200, 60, 60))
d.text((px(fx) - 150, py(fy) - 30), "全标注上界 %.3f" % fy, font=F(14), fill=(185, 50, 50))
for i, item in enumerate([
    ("主动学习在这条赛道上**没有赢**：熵采样 0.738 / 间隔采样 0.739，甚至略低于随机标注的 0.755（B=20）。", (185, 50, 50)),
    ("机制解释：**模型会把噪声学进去，于是被污染的样本反而变得「确定」** —— 熵采样根本挑不中它们。", (60, 60, 70)),
    ("而门控不依赖「模型有多确定」，它依赖「两条证据通道是否互相印证」，所以它能发现污染。", (60, 60, 70)),
    ("D1 最终判定：**通过**。门控 97 次拿 0.706；随机标注要 587 次，最好的主动学习要约 625 次 —— 省 84%。", (20, 140, 70)),
    ("仍未触碰：D2 上界差距(MNIST 83%)、D3 不同性质数据、D4 自然坏锚、D5 超参标定、D6 资源账。", (200, 120, 40)),
]):
    d.text((L - 20, 555 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage21_active_learning.png"))
print("saved")
