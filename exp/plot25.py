# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage25.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H = 1180, 760
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((30, 24), "G5（阶段 25）：换到表格数据 —— D3 不通过", font=F(21), fill=(20, 20, 30))
d.text((30, 58), "Forest Covertype（54 维表格特征 / 6 类 / 3 任务 x 2 类 / 污染 40% / 5 seeds）；同一套 D1 对决，只换数据", font=F(13), fill=(110, 110, 120))
def g(mode, B, k):
    v = [r[k] for r in R if r["mode"] == mode and r["B"] == B and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
# 左：ACC vs 花费
L, Rr, T, Bt = 130, 620, 150, 470
px = lambda v: L + (Rr - L) * v / 1250.0
py = lambda v: Bt - (Bt - T) * (v - 0.40) / 0.25
for gv in np.arange(0.40, 0.651, 0.05):
    y = py(gv)
    d.line([(L, y), (Rr, y)], fill=(234, 234, 240))
    d.text((L - 52, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
for gv in range(0, 1251, 250):
    x = px(gv)
    d.line([(x, Bt), (x, Bt + 6)], fill=(120, 120, 130))
    d.text((x - 20, Bt + 12), str(gv), font=F(12), fill=(120, 120, 130))
d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
d.line([(L, Bt), (Rr + 10, Bt)], fill=(110, 110, 120), width=2)
BS = [0, 5, 10, 20, 50]
pts = [(g("naive", b, "side"), g("naive", b, "ACC_final")) for b in BS]
d.line([(px(x), py(y)) for x, y in pts], fill=(235, 110, 60), width=4, joint="curve")
for (x, y), b in zip(pts, BS):
    d.ellipse([px(x) - 5, py(y) - 5, px(x) + 5, py(y) + 5], fill=(235, 110, 60))
    d.text((px(x) - 26, py(y) - 24), "B=%d" % b, font=F(11), fill=(200, 90, 45))
gx, gy = g("gate", -1, "side"), g("gate", -1, "ACC_final")
d.rectangle([px(gx) - 8, py(gy) - 8, px(gx) + 8, py(gy) + 8], fill=(30, 170, 90))
d.text((px(gx) + 14, py(gy) - 8), "门控 180 次 -> %.3f" % gy, font=F(13), fill=(20, 140, 70))
NEED = 223
d.line([(px(NEED), Bt), (px(NEED), py(gy))], fill=(120, 120, 180), width=2)
d.line([(L, py(gy)), (px(NEED), py(gy))], fill=(120, 120, 180), width=2)
d.text((px(NEED) + 8, py(gy) + 20), "朴素只需约 %d 次 —— 只省 19%%" % NEED, font=F(13), fill=(80, 80, 140))
d.text((L, T - 36), "Covertype：ACC vs oracle 花费", font=F(15), fill=(40, 40, 50))
d.text((L, Bt + 40), "oracle 查询次数", font=F(13), fill=(90, 90, 100))
# 右：三个数据集的节省比例
ox = 700
d.text((ox, T - 36), "D1 通过线：省 30%", font=F(15), fill=(40, 40, 50))
items = [("MNIST\n(784 维像素)", 0.84, (30, 170, 90)), ("Fashion-MNIST\n(784 维像素)", 0.82, (30, 170, 90)),
         ("Covertype\n(54 维表格)", 0.19, (200, 60, 60))]
py2 = lambda v: Bt - (Bt - T) * v / 1.0
for i, (lab, val, col) in enumerate(items):
    x = ox + 10 + i * 110
    y = py2(val)
    d.rectangle([x, y, x + 80, Bt], fill=col, outline=(255, 255, 255))
    d.text((x + 4, y - 22), "%.0f%%" % (val * 100), font=F(16), fill=(50, 50, 60))
    for j, ln in enumerate(lab.split("\n")):
        d.text((x - 6, Bt + 12 + j * 16), ln, font=F(11), fill=(70, 70, 80))
d.line([(ox, py2(0.30)), (ox + 340, py2(0.30))], fill=(200, 60, 60), width=3)
d.text((ox + 250, py2(0.30) - 20), "30%", font=F(13), fill=(185, 50, 50))
d.line([(ox, T - 12), (ox, Bt)], fill=(110, 110, 120), width=2)
d.line([(ox, Bt), (ox + 350, Bt)], fill=(110, 110, 120), width=2)
for i, item in enumerate([
    ("结论：**D1 的结论没有迁移到表格数据上**。门控只省 19%（线是 30%），相对「什么都不做」只高 0.042（MNIST 上是 0.142）。", (185, 50, 50)),
    ("我查了两个可能的解释，都**不成立**：", (60, 60, 70)),
    ("  1) 免费通道变差了？不是 —— 留一通道的可分性 Covertype 0.359 vs MNIST 0.425，同一量级。", (60, 60, 70)),
    ("  2) 数据不够？不是 —— 把每任务训练量从 400 提到 1000，节省只从 19% 到 22%。", (60, 60, 70)),
    ("更可能的原因是**头寸变小**：Covertype 从「不做」0.438 到「全标注」0.583 只有 0.145 的空间（MNIST 是 0.287），", (20, 90, 180)),
    ("而门控只吃下其中的 29%（MNIST 是 49%）。**空间小的时候，省标注这件事的绝对值就小。**", (20, 90, 180)),
]):
    d.text((30, 560 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage25_tabular.png"))
print("saved")
