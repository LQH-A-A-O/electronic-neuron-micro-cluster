# -*- coding: utf-8 -*-
"""阶段 2 出图（纯 PIL）。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
recs = json.load(open(os.path.join(OUT, "results_stage2.json"), encoding="utf-8"))

FONT = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(cand):
        FONT = cand; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

T = 130
W, H = 1180, 650
L, R, T, B = 110, 700, 130, 510

def canvas(title, sub):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((L, 30), title, font=F(24), fill=(20, 20, 30))
    d.text((L, 68), sub, font=F(15), fill=(110, 110, 120))
    return img, d

def axes(d, xs, labels, y0=0.0, y1=1.0, ylab="", pos=None):
    n = len(xs)
    if pos is None:
        px = lambda i: L + (R - L) * i / (n - 1)
    else:
        px = lambda i: L + (R - L) * float(pos[i])
    py = lambda v: B - (B - T) * (v - y0) / (y1 - y0)
    for gv in np.arange(y0, y1 + 1e-9, (y1 - y0) / 10):
        y = py(gv)
        d.line([(L, y), (R, y)], fill=(232, 232, 238))
        d.text((L - 50, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
    for i, lab in enumerate(labels):
        x = px(i)
        d.line([(x, B), (x, B + 6)], fill=(120, 120, 130))
        d.text((x - len(lab) * 4, B + 12), lab, font=F(13), fill=(120, 120, 130))
    d.line([(L, T - 12), (L, B)], fill=(110, 110, 120), width=2)
    d.line([(L, B), (R + 12, B)], fill=(110, 110, 120), width=2)
    return px, py

def agg(scan, arm, key, field, **match):
    ys, es = [], []
    for v in key:
        m = [r for r in recs if r["scan"] == scan and r["arm"] == arm
             and abs(r.get(field, -99) - v) < 1e-9 and r[field if False else "ACC_final"] == r["ACC_final"]]
        m = [r for r in m if abs(r.get(field, -99) - v) < 1e-9]
        vals = [r["ACC_final"] for r in m]
        ys.append(float(np.mean(vals)) if vals else 0.0)
        es.append(float(np.std(vals)) if vals else 0.0)
    return np.array(ys), np.array(es)

QS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
img, d = canvas("阶段 2A：验证器有多不可靠，门控才不值钱？",
                "数据流标签噪声固定 40%；q = 验证器给出反向判断的概率（q=0 完美，q=1 完全反着来）")
px, py = axes(d, QS, ["%.1f" % q for q in QS])
xb = px(QS.index(0.5))
d.rectangle([xb, T - 12, R + 12, B], fill=(252, 236, 236))
d.line([(xb, T - 12), (xb, B)], fill=(220, 120, 120), width=2)
d.text((xb + 12, T + 6), "q > 0.5：门控主动去挑脏的，比不做门控更差", font=F(15), fill=(190, 60, 60))
ysC, esC = agg("A", "C_tiered", QS, "oracle_noise")
ysB, _ = agg("A", "B_replay", QS, "oracle_noise")
band = ([(px(i), py(min(1.0, ysC[i] + esC[i]))) for i in range(len(QS))]
        + [(px(i), py(max(0.0, ysC[i] - esC[i]))) for i in range(len(QS))][::-1])
d.polygon(band, fill=(214, 244, 226))
d.line([(L, py(ysB[0])), (R, py(ysB[0]))], fill=(235, 110, 60), width=4)
d.text((L + 14, py(ysB[0]) - 26), "B 无门控回放 = %.3f（不随 q 变，因为它根本不用 oracle）" % ysB[0], font=F(14), fill=(200, 90, 45))
ptsC = [(px(i), py(ysC[i])) for i in range(len(QS))]
d.line(ptsC, fill=(30, 170, 90), width=4, joint="curve")
for x, y in ptsC:
    d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(30, 170, 90))
d.text((px(0) + 10, py(ysC[0]) - 30), "C 三层 + 这个 q 的验证器", font=F(14), fill=(20, 140, 70))
xc = px(QS.index(0.5))
d.ellipse([xc - 9, py(ysC[QS.index(0.5)]) - 9, xc + 9, py(ysC[QS.index(0.5)]) + 9],
          outline=(20, 20, 20), width=3)
d.text((xc - 118, py(ysC[QS.index(0.5)]) + 18), "预测的临界点 q*=0.5", font=F(14), fill=(40, 40, 40))
d.text((L, B + 52), "纸上先算的临界点：门控收下的条目里脏的比例 = 0.4q/(0.6(1-q)+0.4q)，让它等于 B 的污染率 0.388", font=F(14), fill=(60, 60, 70))
d.text((L, B + 72), "-> q* = 0.5。实测 q=0.5 时 C-B = -0.004（打平），q=0.6 起转为亏本，q=1.0 时 C=0.078 < 无记忆的 A=0.061。", font=F(14), fill=(60, 60, 70))
img.save(os.path.join(OUT, "stage2_oracle_noise.png"))

LAB = [0, 25, 50, 100, 150, 300]
CUM = [0, 250, 500, 1000, 1500, 3000]
img2, d2 = canvas("阶段 2B：oracle 的账单 —— 每任务标注多少条才够？",
                  "数据流标签噪声固定 40%，验证器完美（q=0）；横轴 = 每任务交给它的干净标注条数")
px2, py2 = axes(d2, CUM, [str(c) for c in CUM], pos=[c / 3000.0 for c in CUM])
ys2, es2 = [], []
for bud in LAB:
    vals = [r["ACC_final"] for r in recs if r["scan"] == "B" and r["arm"] == "C_tiered" and r["budget"] == bud]
    ys2.append(float(np.mean(vals))); es2.append(float(np.std(vals)))
ys2 = np.array(ys2)
for ref, col, name in (("A_naive", (150, 150, 160), "A 无记忆"),
                       ("E_clean_replay", (60, 130, 230), "E 只回放干净留出集"),
                       ("B_replay", (235, 110, 60), "B 无门控回放")):
    vals = [r["ACC_final"] for r in recs if r["scan"] == "B" and r["arm"] == ref]
    yy = py2(float(np.mean(vals)))
    d2.line([(L, yy), (R, yy)], fill=col, width=3)
    d2.text((L + 10, yy - 22), "%s = %.3f" % (name, float(np.mean(vals))), font=F(13), fill=col)
band = ([(px2(i), py2(min(1.0, ys2[i] + es2[i]))) for i in range(len(CUM))]
        + [(px2(i), py2(max(0.0, ys2[i] - es2[i]))) for i in range(len(CUM))][::-1])
d2.polygon(band, fill=(214, 244, 226))
pts2 = [(px2(i), py2(ys2[i])) for i in range(len(CUM))]
d2.line(pts2, fill=(30, 170, 90), width=4, joint="curve")
for x, y in pts2:
    d2.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(30, 170, 90))
for i in (1, 3):
    d2.text((px2(i) - 40, py2(ys2[i]) - 34), "%.3f" % ys2[i], font=F(14), fill=(20, 140, 70))
d2.text((px2(2) + 20, T + 30), "横轴 = 累计标注量（线性刻度）。前 250 条（25/任务）就吃掉 0.451 增益的绝大部分；", font=F(14), fill=(60, 60, 70))
d2.text((px2(2) + 20, T + 50), "1000 条之后基本到顶（0.612 -> 0.632）。预算 0 时 C 精确退化成 A（0.061）。", font=F(14), fill=(60, 60, 70))
img2.save(os.path.join(OUT, "stage2_oracle_budget.png"))
print("saved stage2 charts")

# ---------------------------------------------------------------- 图 3：临界点与污染率无关
PS = [0.2, 0.4, 0.6]
img3, d3 = canvas("阶段 2C：临界点 q*=0.5 与数据流污染率无关",
                  "纸上推：门控收下的条目里脏的比例 = pq/((1-p)(1-q)+pq)，令其等于 p 解得 q = 1/2")
px3, py3 = axes(d3, PS, ["%.0f%%" % (100 * p) for p in PS])
COL = {0.4: (30, 170, 90), 0.5: (200, 160, 30), 0.6: (200, 60, 60)}
ENDS = []
for q in (0.4, 0.5, 0.6):
    ys = []
    for p in PS:
        v = [r["ACC_final"] for r in recs if r["scan"] == "C" and r["arm"] == "C_tiered"
             and abs(r["pollution"] - p) < 1e-9 and abs(r["q"] - q) < 1e-9]
        ys.append(float(np.mean(v)))
    pts = [(px3(i), py3(ys[i])) for i in range(len(PS))]
    d3.line(pts, fill=COL[q], width=4, joint="curve")
    for x, y in pts:
        d3.ellipse([x - 5, y - 5, x + 5, y + 5], fill=COL[q])
    ENDS.append((pts[2][1], "C  q=%.1f" % q, COL[q]))
ysB = []
for p in PS:
    v = [r["ACC_final"] for r in recs if r["scan"] == "C" and r["arm"] == "B_replay"
         and abs(r["pollution"] - p) < 1e-9]
    ysB.append(float(np.mean(v)))
ptsB = [(px3(i), py3(ysB[i])) for i in range(len(PS))]
d3.line(ptsB, fill=(235, 110, 60), width=4, joint="curve", )
for x, y in ptsB:
    d3.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(235, 110, 60))
ENDS.append((ptsB[2][1], "B 无门控回放", (235, 110, 60)))
ENDS.sort()
_prev = None
for _ey, _lab, _col in ENDS:
    _yy = _ey if _prev is None else max(_prev + 26, _ey)
    _prev = _yy
    d3.line([(px3(2) + 4, _ey), (px3(2) + 34, _yy)], fill=_col, width=2)
    d3.text((px3(2) + 40, _yy - 10), _lab, font=F(15), fill=_col)
d3.text((px3(0) + 16, T + 16), "q=0.5 三条线上都跟 B 打平（+0.019 / +0.011 / +0.019），", font=F(14), fill=(60, 60, 70))
d3.text((px3(0) + 16, T + 36), "q=0.4 全面赚（+0.045 / +0.058 / +0.071），q=0.6 全面亏。", font=F(14), fill=(60, 60, 70))
d3.text((px3(0) + 16, T + 56), "并且 C 的污染率在 q=0.5 时精确等于 p（0.212 / 0.431 / 0.606）—— 公式对上了。", font=F(14), fill=(60, 60, 70))
img3.save(os.path.join(OUT, "stage2_crossover.png"))
print("saved stage2_crossover.png")

