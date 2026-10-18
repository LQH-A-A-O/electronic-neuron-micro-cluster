# -*- coding: utf-8 -*-
"""阶段 3 出图（纯 PIL）。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
A = json.load(open(os.path.join(OUT, "results_stage3.json"), encoding="utf-8"))
B = json.load(open(os.path.join(OUT, "results_stage3b.json"), encoding="utf-8"))

FONT = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(cand):
        FONT = cand; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

W, H = 1180, 720
L, R, T, Bt = 110, 690, 130, 470
COL = {"N_no_cold": (150, 150, 160), "F_frozen": (215, 70, 70),
       "S_ema05": (230, 170, 40), "S_ema20": (30, 170, 90), "T_true": (50, 120, 230)}

def chart(title, sub, xs, xlabels, series, anns=(), y0=0.6, y1=1.0):
    # 数值越大越靠上的曲线最后画，避免被压住
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((L, 30), title, font=F(23), fill=(20, 20, 30))
    d.text((L, 68), sub, font=F(15), fill=(110, 110, 120))
    n = len(xs)
    px = lambda i: L + (R - L) * i / (n - 1)
    py = lambda v: Bt - (Bt - T) * (v - y0) / (y1 - y0)
    for gv in np.arange(y0, y1 + 1e-9, (y1 - y0) / 8):
        y = py(gv)
        d.line([(L, y), (R, y)], fill=(232, 232, 238))
        d.text((L - 50, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
    for i, lab in enumerate(xlabels):
        x = px(i)
        d.line([(x, Bt), (x, Bt + 6)], fill=(120, 120, 130))
        d.text((x - 20, Bt + 12), lab, font=F(14), fill=(120, 120, 130))
    d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
    d.line([(L, Bt), (R + 12, Bt)], fill=(110, 110, 120), width=2)
    # 先铺所有误差带，再画所有折线，否则后画的带子会盖住前面的线
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
        d.text((R + 40, yy - 11), "%-11s %.3f" % (name, val), font=F(15), fill=c)
    for i, (txt, col) in enumerate(anns):
        d.text((L + 6, Bt + 74 + i * 22), txt, font=F(14), fill=col)
    return img, d

def agg3(drift, arm, field="ACC_final"):
    v = [r[field] for r in A if r["drift"] == drift and r["arm"] == arm and r[field] == r[field]]
    return (float(np.mean(v)) if v else float("nan"),
            float(np.std(v)) if v else float("nan"))

DR = [0.0, 0.3, 0.6]
ser = []
for name in ("N_no_cold", "F_frozen", "S_ema05", "S_ema20", "T_true"):
    ys = [agg3(d, name)[0] for d in DR]
    es = [agg3(d, name)[1] for d in DR]
    ser.append((name, np.array(ys), np.array(es)))
img, d = chart("阶段 3A：冷备该不该更新？—— 温备充足时，答案让人意外",
               "10 任务 / 25 类 / 每类出现 4 次 / 数据噪声 30% / 5 seeds；冷备以 50% 比例参与回放",
               DR, ["%.1f" % x for x in DR], ser,
               [("T_true = 作弊上界：冷备永远精确等于当前世界的真实原型。它也没赢过不要冷备。", (185, 50, 50)),
                ("漂移越大，所有冷备方案越吃亏；冻结的冷备（构想里的字面设计）掉得最狠。", (60, 60, 70))],
               y0=0.87, y1=0.95)
d.text(((L + R) / 2 - 60, Bt + 46), "世界漂移强度 drift", font=F(15), fill=(80, 80, 90))
img.save(os.path.join(OUT, "stage3_cold_update.png"))

CAPS = [50, 150, 600]
ser2 = []
for name in ("N_no_cold", "F_frozen", "S_ema20", "T_true"):
    ys, es = [], []
    for c in CAPS:
        v = [r["ACC_final"] for r in B if r["cap"] == c and r["arm"] == name]
        ys.append(float(np.mean(v))); es.append(float(np.std(v)))
    ser2.append((name, np.array(ys), np.array(es)))
img2, d2 = chart("阶段 3B：冷备只在「温备装不下」时才值钱 —— 而且必须维护它",
                 "drift=0.6 固定；横轴 = 温备容量（reservoir 上限）；冷备只有 25 条原型却能覆盖全部 25 个类",
                 CAPS, ["%d 条" % c for c in CAPS], ser2,
                 [("温备 50 条时：维护良好的冷备有用（但只到 +0.076），", (60, 60, 70)),
                  ("而冻结的冷备 0.736 比干脆不要冷备 0.766 还差 —— 陈旧的锚点会主动误导。", (185, 50, 50)),
                  ("温备 600 条时：连作弊的真实原型都救不回来（-0.006）。", (60, 60, 70))],
                 y0=0.70, y1=0.96)
d2.text(((L + R) / 2 - 60, Bt + 46), "温备容量（条）", font=F(15), fill=(80, 80, 90))
img2.save(os.path.join(OUT, "stage3_capacity.png"))
print("saved stage3 charts")
