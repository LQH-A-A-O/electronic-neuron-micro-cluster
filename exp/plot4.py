# -*- coding: utf-8 -*-
"""阶段 4 出图：真实 MNIST 上的复验（纯 PIL）。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
recs = json.load(open(os.path.join(OUT, "results_stage4.json"), encoding="utf-8"))

FONT = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(cand):
        FONT = cand; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

W, H = 1180, 700
L, R, T, Bt = 110, 690, 130, 470
COL = {"A_naive": (150, 150, 160), "B_replay": (235, 110, 60),
       "C_tiered_good": (30, 170, 90), "D_tiered_demo": (170, 90, 210),
       "E_clean_replay": (60, 130, 230)}

def chart(title, sub, xs, xlabels, series, anns=(), y0=0.0, y1=1.0, xpos=None):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((L, 30), title, font=F(23), fill=(20, 20, 30))
    d.text((L, 68), sub, font=F(15), fill=(110, 110, 120))
    n = len(xs)
    px = (lambda i: L + (R - L) * i / (n - 1)) if xpos is None else (lambda i: L + (R - L) * xpos[i])
    py = lambda v: Bt - (Bt - T) * (v - y0) / (y1 - y0)
    for gv in np.arange(y0, y1 + 1e-9, (y1 - y0) / 8):
        y = py(gv)
        d.line([(L, y), (R, y)], fill=(232, 232, 238))
        d.text((L - 50, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
    for i, lab in enumerate(xlabels):
        x = px(i)
        d.line([(x, Bt), (x, Bt + 6)], fill=(120, 120, 130))
        d.text((x - 22, Bt + 12), lab, font=F(14), fill=(120, 120, 130))
    d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
    d.line([(L, Bt), (R + 12, Bt)], fill=(110, 110, 120), width=2)
    for name, ys, es in series:
        c = COL[name]
        band = ([(px(i), py(min(y1, ys[i] + es[i]))) for i in range(n)]
                + [(px(i), py(max(y0, ys[i] - es[i]))) for i in range(n)][::-1])
        d.polygon(band, fill=tuple(int(v + (255 - v) * 0.88) for v in c))
    ends = []
    for name, ys, es in series:
        c = COL[name]
        pts = [(px(i), py(ys[i])) for i in range(n)]
        d.line(pts, fill=c, width=4, joint="curve")
        for x, y in pts:
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=c)
        ends.append((pts[-1][1], name, c, ys[-1]))
    ends.sort()
    prev = None
    for ey, name, c, val in ends:
        yy = ey if prev is None else max(prev + 28, ey)
        prev = yy
        d.line([(R + 4, ey), (R + 34, yy)], fill=c, width=2)
        d.text((R + 40, yy - 11), "%-15s %.3f" % (name, val), font=F(15), fill=c)
    for i, (txt, col) in enumerate(anns):
        d.text((L + 6, Bt + 76 + i * 22), txt, font=F(14), fill=col)
    return img, d

def agg(scan, arm, key, val, field="ACC_final"):
    ys, es = [], []
    for v in val:
        m = [r[field] for r in recs if r["scan"] == scan and r["arm"] == arm
             and abs(r.get(key, -99) - v) < 1e-9 and r[field] == r[field]]
        ys.append(float(np.mean(m)) if m else float("nan"))
        es.append(float(np.std(m)) if m else 0.0)
    return np.array(ys), np.array(es)

NO = [0.0, 0.2, 0.4]
ser = [(a, *agg("A", a, "noise", NO)) for a in
       ("A_naive", "B_replay", "C_tiered_good", "D_tiered_demo", "E_clean_replay")]
img, d = chart("阶段 4A：真实 MNIST 上复验阶段 1 的结论",
               "5 任务 x 2 类（共 10 类）/ 784 维原始像素 / 400 步每任务 / 3 seeds；噪声 = 训练流中被翻转的标签比例",
               NO, ["%.0f%%" % (100 * n) for n in NO], ser,
               [("门控在噪声下依然大幅领先：噪声 40% 时 C 比 B 高约 0.23，记忆污染率 0.002 vs 0.44。", (60, 60, 70)),
                ("D（沿用原演示那套只比对 value 的验证器）依旧贴着 B —— 换成真数据后这条结论不变。", (185, 50, 50)),
                ("噪声 0 时 C≈B≈E：门控不花钱。但 C 追平不了上界 E（0.721 vs 0.756），", (60, 60, 70)),
                ("而且阶段 2 的「少量标注就够」在真数据上不成立（见 4B）—— 合成数据低估了 oracle 的价钱。", (60, 60, 70))])
d.text(((L + R) / 2 - 60, Bt + 46), "训练流标签噪声率", font=F(15), fill=(80, 80, 90))
img.save(os.path.join(OUT, "stage4_mnist.png"))

BU = [0, 20, 50, 100, 200]
CUM = [b * 5 for b in BU]
ys, es = agg("B", "C_tiered_good", "budget", BU)
ser2 = [("C_tiered_good", ys, es)]
img2, d2 = chart("阶段 4B：真实 MNIST 上的 oracle 账单",
                 "noise=40%，验证器完美；横轴 = 每任务交给它的干净标注条数（累计 = x5 个任务）",
                 BU, ["%d" % c for c in CUM], ser2,
                 [], y0=0.0, y1=1.0, xpos=[c / 1000.0 for c in CUM])
for ref, name in (("A_naive", "A 无记忆"), ("B_replay", "B 无门控回放")):
    v = [r["ACC_final"] for r in recs if r["scan"] == "B" and r["arm"] == ref]
    yy = d2 and (Bt - (Bt - T) * (float(np.mean(v)) - 0.0) / 1.0)
    d2.line([(L, yy), (R, yy)], fill=COL[ref], width=3)
    d2.text((L + 10, yy - 22), "%s = %.3f" % (name, float(np.mean(v))), font=F(13), fill=COL[ref])
d2.text((L + 6, Bt + 76), "真实数据上预算的边际收益同样是急剧衰减的；", font=F(14), fill=(60, 60, 70))
d2.text((L + 6, Bt + 98), "预算 0 时 C 仍然退化成 A（门控不产生能力，只转移能力）。", font=F(14), fill=(60, 60, 70))
d2.text(((L + R) / 2 - 80, Bt + 46), "累计干净标注条数", font=F(15), fill=(80, 80, 90))
img2.save(os.path.join(OUT, "stage4_mnist_budget.png"))
print("saved stage4 charts")
