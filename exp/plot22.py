# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage22.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
def g(mode, B, k):
    v = [r[k] for r in R if r["mode"] == mode and r["B"] == B and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
W, H = 1180, 760
L, T, Bt = 150, 170, 480
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "G2（阶段 22）：资源账 —— 内存不是问题，但挖出一个部署级的坑", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "真实 MNIST / 5 任务 2000 条观测 / 3 seeds；纯 Python 原型，未优化", font=F(14), fill=(110, 110, 120))
def panel(ox, title, pairs, ymax, fmt, note=""):
    d.text((ox, 110), title, font=F(15), fill=(40, 40, 50))
    d.line([(ox, 132), (ox + 230, 132)], fill=(200, 200, 210))
    py = lambda v: Bt - (Bt - T) * min(max(v, 0), ymax) / ymax
    for i, (lab, val, col) in enumerate(pairs):
        x = ox + 8 + i * 78
        y = py(val)
        d.rectangle([x, y, x + 58, Bt], fill=col, outline=(255, 255, 255))
        d.text((x - 2, y - 19), fmt % val, font=F(12), fill=(50, 50, 60))
        d.text((x - 8, Bt + 10), lab, font=F(11), fill=(70, 70, 80))
    if note:
        d.text((ox, Bt + 34), note, font=F(12), fill=(120, 120, 130))
    d.line([(ox, T - 12), (ox, Bt)], fill=(110, 110, 120), width=2)
    d.line([(ox, Bt), (ox + 250, Bt)], fill=(110, 110, 120), width=2)
GREEN, GRAY, ORANGE = (30, 170, 90), (150, 150, 160), (235, 110, 60)
panel(L - 20, "延迟 µs（越低越好）",
      [("门控观测", g("gate", -1, "us_observe"), GREEN), ("门控预测", g("gate", -1, "us_predict"), GREEN),
       ("朴素观测", g("naive", 0, "us_observe"), GRAY), ("朴素预测", g("naive", 0, "us_predict"), GRAY)],
      20.0, "%.1f", "门控的写路径比朴素慢 43 倍，但绝对值只有 15.8 µs")
panel(L + 320, "内存 MB",
      [("门控库", g("gate", -1, "mem_bytes") / 1e6, GREEN), ("门控峰值", g("gate", -1, "peak_alloc_mb"), GREEN),
       ("朴素库", g("naive", 0, "mem_bytes") / 1e6, GRAY), ("朴素峰值", g("naive", 0, "peak_alloc_mb"), GRAY)],
      14.0, "%.1f", "库只有 1 MB 量级 —— 内存完全不是瓶颈")
# 第三幅：规模效应
ox3 = L + 640
d.text((ox3, 110), "规模效应：库越大，预测越慢？", font=F(15), fill=(40, 40, 50))
d.line([(ox3, 132), (ox3 + 250, 132)], fill=(200, 200, 210))
xs = [np.mean([r["scaling"][k]["buf"] for r in R if r["mode"] == "gate"]) for k in range(5)]
ys = [np.mean([r["scaling"][k]["us_per_sample"] for r in R if r["mode"] == "gate"]) for k in range(5)]
px3 = lambda v: ox3 + 230.0 * v / 1100.0
py3 = lambda v: Bt - (Bt - T) * v / 20.0
for gv in np.arange(0, 20.1, 5):
    y = py3(gv)
    d.line([(ox3, y), (ox3 + 230, y)], fill=(234, 234, 240))
    d.text((ox3 - 40, y - 9), "%.0f" % gv, font=F(12), fill=(130, 130, 140))
d.line([(px3(x), py3(y)) for x, y in zip(xs, ys)], fill=GREEN, width=4, joint="curve")
for x, y in zip(xs, ys):
    d.ellipse([px3(x) - 5, py3(y) - 5, px3(x) + 5, py3(y) + 5], fill=GREEN)
d.line([(ox3, T - 12), (ox3, Bt)], fill=(110, 110, 120), width=2)
d.line([(ox3, Bt), (ox3 + 240, Bt)], fill=(110, 110, 120), width=2)
d.text((ox3, Bt + 10), "库条数 223 -> 1048", font=F(12), fill=(70, 70, 80))
d.text((ox3, Bt + 34), "0.8 -> 16.3 µs，约线性（kNN 的 O(N·d)），没爆炸", font=F(12), fill=(120, 120, 130))
d.text((L, Bt + 70), "🔴 本轮挖出的部署级坑：最初的「观测 811 µs/条」里 **98% 是 BLAS 线程抖动**，不是算法开销。", font=F(15), fill=(185, 50, 50))
d.text((L, Bt + 96), "   129x129 的 inv：默认多线程 BLAS 45.2 ms，把它设成单线程只要 0.34 ms —— **快 133 倍**，一个环境变量。", font=F(14), fill=(60, 60, 70))
d.text((L, Bt + 120), "   设 OPENBLAS_NUM_THREADS=1 之后：观测 811 -> 15.8 µs，重编译 37 -> 1.1 ms，整轮实验 21 秒 -> 2 秒。", font=F(14), fill=(20, 140, 70))
d.text((L, Bt + 150), "   教训：小矩阵上多线程 BLAS 是负优化。这个坑不测资源账永远发现不了 —— 而且它会污染所有性能结论。", font=F(14), fill=(20, 90, 180))
img.save(os.path.join(OUT, "stage22_resources.png"))
print("saved")
