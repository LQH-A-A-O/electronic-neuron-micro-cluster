# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage20.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H = 1180, 720
L, Rr, T, Bt = 140, 720, 150, 500
px = lambda v: L + (Rr - L) * v / 2100.0
py = lambda v: Bt - (Bt - T) * v / 1.0
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "阶段 20（落地标准 D1）：同等 oracle 预算下，门控 vs 朴素随机标注", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "真实 MNIST / 污染 40% / 5 任务 x 2 类 / 5 seeds；横轴 = oracle 查询次数，纵轴 = 总体 ACC", font=F(14), fill=(110, 110, 120))
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
def arm(mode, B, k="ACC_final"):
    v = [r[k] for r in R if r["mode"] == mode and r["B"] == B and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
NB = [0, 5, 10, 20, 50]
pts = [(arm("naive", B, "side"), arm("naive", B)) for B in NB]
d.line([(px(x), py(y)) for x, y in pts], fill=(235, 110, 60), width=4, joint="curve")
for (x, y), B in zip(pts, NB):
    d.ellipse([px(x) - 6, py(y) - 6, px(x) + 6, py(y) + 6], fill=(235, 110, 60))
    d.text((px(x) - 30, py(y) + 12), "B=%d" % B, font=F(12), fill=(200, 90, 45))
gx, gy = arm("gate", -1, "side"), arm("gate", -1)
d.ellipse([px(gx) - 9, py(gy) - 9, px(gx) + 9, py(gy) + 9], fill=(30, 170, 90))
d.text((px(gx) - 10, py(gy) - 34), "门控 97 次 -> %.3f" % gy, font=F(15), fill=(20, 140, 70))
# 同等准确率下朴素需要多少次
NEED = 587
d.line([(px(NEED), Bt), (px(NEED), py(gy))], fill=(120, 120, 180), width=2)
d.line([(L, py(gy)), (px(NEED), py(gy))], fill=(120, 120, 180), width=2)
d.text((px(NEED) + 8, py(gy) - 6), "朴素要花约 %d 次才能到同一准确率" % NEED, font=F(14), fill=(80, 80, 140))
for i, item in enumerate([
    ("D1 通过：门控用 97 次 oracle 拿到 0.706；朴素随机标注要花约 590 次才到同一水平 —— **省 84%**（通过线是 30%）。", (20, 140, 70)),
    ("同等花费下（97 次）门控比朴素高约 0.111（通过线是 0.05）。", (20, 140, 70)),
    ("而且门控只收下 1053/2000 条 —— 它靠「批内留一互证 + 已积累的判别器」这两个免费通道就筛掉了大部分脏数据，只在真正拿不准时花钱。", (60, 60, 70)),
    ("D2 边缘不通过：门控 0.706 是全标注上界 0.851 的 83%，低于 85% 的线。", (185, 50, 50)),
    ("剩下的落地缺口是 D3（换一个非 MNIST 的真实任务）和 D4（坏锚要自然产生，不是人注入）。", (200, 120, 40)),
]):
    d.text((L - 20, 540 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage20_showdown.png"))
print("saved")
