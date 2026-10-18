# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage24.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
DEFAULT = dict(tau=0.60, hilo=0.65, knn_k=5, ridge=1.0)
GRID = {"tau": [0.30, 0.45, 0.60, 0.75, 0.90], "hilo": [0.52, 0.58, 0.65, 0.72, 0.80],
        "knn_k": [2, 3, 5, 8, 12], "ridge": [0.3, 0.5, 1.0, 2.0, 3.0]}
NAME = {"tau": "TAU 锚置信度门槛", "hilo": "HILO 门控决断阈值", "knn_k": "KNN_K 邻居数", "ridge": "RIDGE 岭系数"}
W, H = 1180, 850
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((30, 24), "G4（阶段 24）：超参敏感度扫描 —— D5 判定：不通过（但不是因为脆弱）", font=F(21), fill=(20, 20, 30))
d.text((30, 58), "真实 MNIST / 污染 40% / 5 seeds；D5 通过线 = 相对默认值的最大 |ΔACC| < 0.05", font=F(14), fill=(110, 110, 120))
def acc(p, v):
    x = [r["ACC_final"] for r in R if r["param"] == p and abs(r["value"] - v) < 1e-9]
    return float(np.mean(x)), float(np.std(x))
def side(p, v):
    return float(np.mean([r["side"] for r in R if r["param"] == p and abs(r["value"] - v) < 1e-9]))
for i, (p, vals) in enumerate(GRID.items()):
    ox = 60 + (i % 2) * 570
    oy = 110 + (i // 2) * 290
    d.text((ox, oy), NAME[p], font=F(16), fill=(40, 40, 50))
    base = acc(p, DEFAULT[p])[0]
    px = lambda j: ox + 30 + j * 95
    py = lambda v: oy + 220 - (v - 0.55) / 0.25 * 190
    d.line([(ox + 20, py(base + 0.05)), (ox + 470, py(base + 0.05))], fill=(240, 200, 200), width=1)
    d.line([(ox + 20, py(base - 0.05)), (ox + 470, py(base - 0.05))], fill=(240, 200, 200), width=1)
    d.line([(ox + 20, py(base)), (ox + 470, py(base))], fill=(200, 200, 210), width=1)
    pts = []
    for j, v in enumerate(vals):
        m, s = acc(p, v)
        pts.append((px(j), py(m), v, m, side(p, v)))
    d.line([(x, y) for x, y, _, _, _ in pts], fill=(30, 170, 90), width=3, joint="curve")
    for x, y, v, m, sp in pts:
        isdef = abs(v - DEFAULT[p]) < 1e-9
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(200, 60, 60) if isdef else (30, 170, 90))
        d.text((x - 16, y - 26), "%.3f" % m, font=F(11), fill=(50, 50, 60))
        d.text((x - 22, oy + 232), str(v), font=F(12), fill=(90, 90, 100))
        d.text((x - 22, oy + 250), "%d次" % sp, font=F(10), fill=(140, 140, 150))
    d.text((ox + 30, oy + 272), "绿=取值  红=默认  灰带=±0.05 通过区", font=F(11), fill=(150, 150, 160))
for i, item in enumerate([
    ("TAU 敏感度 0.051（刚好超线）：0.30 -> 0.757 但花 393 次；0.90 -> 0.691 而只花 1 次。**它在拿准确率换查询**。", (185, 50, 50)),
    ("KNN_K 敏感度 0.070（超线）：2 -> 0.637，12 -> 0.727。**单调，而且默认的 5 偏小** —— 免费就能涨 0.02。", (185, 50, 50)),
    ("HILO 敏感度 0.014、RIDGE 0.009：两个都通过。但 HILO 其实**不是一个连续旋钮** ——", (20, 140, 70)),
    ("   因为两个通道都是 0/1 硬票，合成分数只取 {0, 0.4, 0.6, 1.0} 四个值，所以阈值只在 0.6 附近才有意义。", (60, 60, 70)),
    ("🔎 意外收获：HILO <= 0.58 时**oracle 花费降到 0，ACC 只掉 0.014**（0.692 vs 0.706）。", (20, 90, 180)),
    ("   也就是说 97 次查询只买到 +0.014 —— **这套东西可以以「零 oracle」模式运行**，门控的价值几乎全来自免费通道。", (20, 90, 180)),
]):
    d.text((30, 705 + i * 23), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage24_sensitivity.png"))
print("saved")
