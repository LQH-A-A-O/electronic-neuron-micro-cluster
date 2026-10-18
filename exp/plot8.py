# -*- coding: utf-8 -*-
"""阶段 8 出图。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage8.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

ARMS = ["gold", "D1_anchor_q", "D1_mlp", "D1_anchor_t", "B_replay", "D2_anchor_t", "A_naive"]
COL = {"gold": (30, 170, 90), "D1_anchor_q": (20, 150, 210), "D1_mlp": (150, 150, 160),
       "D1_anchor_t": (200, 60, 60), "B_replay": (235, 110, 60), "D2_anchor_t": (140, 40, 40),
       "A_naive": (200, 200, 205)}
LAB = {"gold": "gold 参考", "D1_anchor_q": "λ=经验数量", "D1_mlp": "MLP 判断", "D1_anchor_t": "λ=经验可信度",
       "B_replay": "B 无门控回放", "D2_anchor_t": "λ=可信度 lag2", "A_naive": "A 无记忆"}
def acc(a):
    v = [r["ACC_final"] for r in R if r["arm"] == a]
    return float(np.mean(v)), float(np.std(v))

W, H = 1180, 720
L, Rr, T, Bt = 120, 700, 140, 480
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 10, 28), "阶段 8：把 λ 从「经验数量」换成「经验可信度」—— 假设不成立", font=F(22), fill=(20, 20, 30))
d.text((L - 10, 64), "真实 MNIST / 结构化幻觉 40% / 全部 delay1+回滚（除标注外）/ 3 seeds", font=F(15), fill=(110, 110, 120))
py = lambda v: Bt - (Bt - T) * v / 0.78
for gv in np.arange(0, 0.7801, 0.1):
    y = py(gv)
    d.line([(L, y), (Rr, y)], fill=(232, 232, 238))
    d.text((L - 52, y - 9), "%.1f" % gv, font=F(13), fill=(120, 120, 130))
d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
d.line([(L, Bt), (Rr + 10, Bt)], fill=(110, 110, 120), width=2)
slot = (Rr - L) / len(ARMS)
bw = min(62, slot / 1.7)
for i, a in enumerate(ARMS):
    x = L + slot * (i + 0.5) - bw / 2
    m, s = acc(a)
    y = py(m)
    d.rectangle([x, y, x + bw, Bt], fill=COL[a], outline=(255, 255, 255))
    d.text((x + 1, y - 20), "%.3f" % m, font=F(13), fill=(50, 50, 60))
    d.text((x - 16, Bt + 12), LAB[a], font=F(12), fill=(60, 60, 70))
    d.line([(x + 2, py(m - s)), (x + bw - 2, py(m - s))], fill=(90, 90, 100))
    d.line([(x + 2, py(m + s)), (x + bw - 2, py(m + s))], fill=(90, 90, 100))
for i, (txt, col) in enumerate([
    ("λ=经验数量 0.492  vs  λ=经验可信度 0.471：差 +0.021，3 个种子里只有 2 个方向一致 —— 噪声级别。", (60, 60, 70)),
    ("而且 lag 越大越糟：可信度版 lag2 掉到 0.368，同一个 lag2 用 MLP 判断是 0.507（阶段 7）。", (185, 50, 50)),
    ("机制：λ 只看「已转正」的经验 -> n_verified 变小 -> λ 变小 -> 判断权被交给冷备锚。", (140, 70, 180)),
    ("可是那个锚本身也是从同一批未裁判的温备条目算出来的（污染率 0.417）—— 换汤不换药。", (140, 70, 180))]):
    d.text((L - 10, Bt + 62 + i * 22), txt, font=F(14), fill=col)
img.save(os.path.join(OUT, "stage8_trust_lambda.png"))
print("saved stage8 chart")
