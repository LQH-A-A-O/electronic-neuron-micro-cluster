# -*- coding: utf-8 -*-
"""把阶段 1 的结果画出来（纯 PIL，不装 matplotlib）。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
recs = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))

FONT = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(cand):
        FONT = cand
        break

def F(sz):
    if FONT:
        try:
            return ImageFont.truetype(FONT, sz)
        except Exception:
            pass
    return ImageFont.load_default()

ARMS = [("A_naive", "A 无记忆", (150, 150, 160)),
        ("B_replay", "B 无门控回放", (235, 110, 60)),
        ("C_tiered_good", "C 三层 + 正确验证器", (30, 170, 90)),
        ("D_tiered_demo", "D 三层 + 原演示验证器", (170, 90, 210)),
        ("E_clean_replay", "E 只回放干净留出集（上界）", (60, 130, 230))]
NOISES = [0.0, 0.1, 0.2, 0.3, 0.4]

W, H = 1160, 640
L, R, T, B = 110, 700, 120, 500
px = lambda i: L + (R - L) * i / (len(NOISES) - 1)
py = lambda v: B - (B - T) * v

def series(arm, key):
    ys, es = [], []
    for n in NOISES:
        v = [r[key] for r in recs if r["arm"] == arm and r["noise"] == n and r[key] == r[key]]
        ys.append(float(np.mean(v)) if v else 0.0)
        es.append(float(np.std(v)) if v else 0.0)
    return np.array(ys), np.array(es)

def frame(d, title, subtitle):
    d.text((L, 30), title, font=F(24), fill=(20, 20, 30))
    d.text((L, 66), subtitle, font=F(15), fill=(110, 110, 120))
    for gv in np.arange(0, 1.001, 0.1):
        y = py(gv)
        d.line([(L, y), (R, y)], fill=(230, 230, 236))
        d.text((L - 48, y - 9), "%.1f" % gv, font=F(14), fill=(120, 120, 130))
    for i, n in enumerate(NOISES):
        x = px(i)
        d.line([(x, B), (x, B + 6)], fill=(120, 120, 130))
        d.text((x - 18, B + 12), "%.0f%%" % (100 * n), font=F(14), fill=(120, 120, 130))
    d.line([(L, T - 10), (L, B)], fill=(110, 110, 120), width=2)
    d.line([(L, B), (R + 10, B)], fill=(110, 110, 120), width=2)
    d.text(((L + R) / 2 - 70, B + 46), "训练流标签噪声率", font=F(15), fill=(80, 80, 90))

def draw_curves(d, key, legend_note=True):
    for arm, label, color in ARMS:
        ys, es = series(arm, key)
        band = ([(px(i), py(min(1.0, ys[i] + es[i]))) for i in range(len(ys))]
                + [(px(i), py(max(0.0, ys[i] - es[i]))) for i in range(len(ys))][::-1])
        d.polygon(band, fill=tuple(int(c + (255 - c) * 0.80) for c in color))
    for arm, label, color in ARMS:
        ys, _ = series(arm, key)
        pts = [(px(i), py(ys[i])) for i in range(len(ys))]
        d.line(pts, fill=color, width=4, joint="curve")
        for x, y in pts:
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
    ly = T
    d.text((R + 60, T - 30), "图例（阴影 = ±1 标准差）", font=F(15), fill=(20, 20, 30))
    for arm, label, color in ARMS:
        d.line([(R + 60, ly + 9), (R + 96, ly + 9)], fill=color, width=4)
        d.text((R + 106, ly - 2), label, font=F(15), fill=(40, 40, 50))
        ys, _ = series(arm, key)
        d.text((R + 106, ly + 20), "%s" % ("0%%→%.3f   40%%→%.3f" % (ys[0], ys[-1])), font=F(13), fill=(130, 130, 140))
        ly += 58

img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
frame(d, "阶段 1：三层信任记忆 vs 无门控回放（10 任务 / 50 类 / 3 seeds）",
      "纵轴 = 学完 10 个任务后在全部 50 个类上的平均准确率；噪声 = 训练流里被翻转的标签比例")
draw_curves(d, "ACC_final")
d.text((L, B + 82), "噪声 0 时 C≈B≈E：门控不花钱。噪声 40% 时 C=0.615 逼近上界 E=0.621，B 只有 0.390，", font=F(14), fill=(55, 55, 65))
d.text((L, B + 102), "D 全程贴着 B —— 三层结构本身不值钱，值钱的是验证器。详见 报告-阶段1.md。", font=F(14), fill=(55, 55, 65))
img.save(os.path.join(OUT, "stage1_accuracy.png"))

img2 = Image.new("RGB", (W, H), (255, 255, 255))
d2 = ImageDraw.Draw(img2)
frame(d2, "记忆库污染率：无门控回放 vs 三层门控",
      "污染率 = 记忆库里标签是错的条目占比（只用于事后算账；门控本身看不到真值）")
draw_curves(d2, "pollution_final")
d2.text((L, B + 82), "B 的污染率随噪声线性上升（40% 时 0.383）；C 全程 0.000；D（原演示验证器）与 B 同行（0.397）。", font=F(14), fill=(55, 55, 65))
img2.save(os.path.join(OUT, "stage1_pollution.png"))
print("saved both charts", (W, H))
