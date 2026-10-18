# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
A = json.load(open(os.path.join(OUT, "results_stage20.json"), encoding="utf-8"))
B = json.load(open(os.path.join(OUT, "results_stage20_fmnist.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H = 1180, 730
L, Rr, T, Bt = 140, 720, 150, 500
px = lambda v: L + (Rr - L) * v / 2100.0
py = lambda v: Bt - (Bt - T) * v / 1.0
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "落地标准 D1/D3：门控 vs 朴素随机标注 —— 两个真实数据集上都通过", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "污染 40% / 5 任务 x 2 类 / 5 seeds；横轴 = oracle 查询次数，纵轴 = 总体 ACC", font=F(14), fill=(110, 110, 120))
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
def arm(recs, mode, B_, k):
    v = [r[k] for r in recs if r["mode"] == mode and r["B"] == B_ and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
NB = [0, 5, 10, 20, 50]
for recs, col, name in ((A, (235, 110, 60), "MNIST"), (B, (60, 130, 230), "Fashion-MNIST")):
    pts = [(arm(recs, "naive", b, "side"), arm(recs, "naive", b, "ACC_final")) for b in NB]
    d.line([(px(x), py(y)) for x, y in pts], fill=col, width=4, joint="curve")
    for (x, y), b in zip(pts, NB):
        d.ellipse([px(x) - 6, py(y) - 6, px(x) + 6, py(y) + 6], fill=col)
    gx, gy = arm(recs, "gate", -1, "side"), arm(recs, "gate", -1, "ACC_final")
    d.rectangle([px(gx) - 8, py(gy) - 8, px(gx) + 8, py(gy) + 8], fill=(30, 170, 90))
    d.text((px(gx) + 14, py(gy) - 8), "%s 门控: %d 次 -> %.3f" % (name, gx, gy), font=F(14), fill=(20, 140, 70))
d.text((px(200) - 10, py(0.52) + 6), "朴素随机标注", font=F(15), fill=(180, 90, 45))
for i, item in enumerate([
    ("D1 通过（两个数据集都通过）：MNIST 门控 97 次拿 0.706，朴素要 ~587 次 —— 省 84%；", (20, 140, 70)),
    ("Fashion-MNIST 门控 120 次拿 0.673，朴素要 ~680 次 —— 省 82%。（通过线：省 30%）", (20, 140, 70)),
    ("D2 边缘通过：Fashion-MNIST 上 0.673 / 0.782 = 86%（线是 85%）；MNIST 上 83%，差一点。", (200, 120, 40)),
    ("D3 部分通过：换到第二个真实数据集后核心结论复现，但它仍是 28x28 灰度图 —— 同一种任务，不是同一种数据。", (60, 60, 70)),
    ("仍未触碰的：D4（坏锚要自然产生，不是人注入）、D5（超参标定）、D6（资源账）。", (185, 50, 50)),
]):
    d.text((L - 20, 545 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "landing_D1_D3.png"))
print("saved")
