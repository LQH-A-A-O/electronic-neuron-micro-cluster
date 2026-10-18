# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
T2 = json.load(open(os.path.join(OUT, "results_t2_realnoise.json"), encoding="utf-8"))
T3 = json.load(open(os.path.join(OUT, "results_t3_hilo.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H = 1180, 800
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((30, 22), "T2/T3：真实形态的标注噪声 + HILO 从开关变回旋钮", font=F(21), fill=(20, 20, 30))
d.text((30, 56), "真实 MNIST / 污染 40% / 多任务流", font=F(13), fill=(110, 110, 120))
# ---- 左：T2 六种噪声的节省比例 ----
MODELS = ["uniform", "confusion", "similarity", "annotator", "drift", "shortcut"]
LAB = ["均匀\n(旧基线)", "混淆矩阵\n(旧基线)", "相似度驱动", "标注者偏差", "质量漂移", "捷径相关"]
ox, T, Bt = 120, 170, 430
px = lambda i: ox + 20 + i * 88
py = lambda v: Bt - (Bt - T) * v / 1.0
d.text((ox, 108), "① 六种标注噪声下，门控省多少 oracle（线 = 30%）", font=F(15), fill=(40, 40, 50))
for gv in np.arange(0, 1.01, 0.25):
    y = py(gv)
    d.line([(ox, y), (ox + 560, y)], fill=(234, 234, 240))
    d.text((ox - 46, y - 9), "%.0f%%" % (gv * 100), font=F(12), fill=(130, 130, 140))
for i, m in enumerate(MODELS):
    rs = [r for r in T2 if r["model"] == m]
    s = float(np.nanmean([r["saved"] for r in rs]))
    col = (30, 170, 90) if s >= 0.30 else (200, 60, 60)
    if m in ("similarity", "annotator", "drift"):
        col = (60, 130, 230) if s >= 0.30 else (200, 60, 60)
    x = px(i); y = py(s)
    d.rectangle([x, y, x + 62, Bt], fill=col, outline=(255, 255, 255))
    d.text((x - 2, y - 20), "%.0f%%" % (s * 100), font=F(14), fill=(50, 50, 60))
    for j, ln in enumerate(LAB[i].split("\n")):
        d.text((x - 4, Bt + 10 + j * 15), ln, font=F(10), fill=(70, 70, 80))
d.line([(ox, py(0.30)), (ox + 560, py(0.30))], fill=(200, 60, 60), width=3)
d.text((ox + 480, py(0.30) - 18), "30%", font=F(12), fill=(185, 50, 50))
d.line([(ox, T - 12), (ox, Bt)], fill=(110, 110, 120), width=2)
d.line([(ox, Bt), (ox + 560, Bt)], fill=(110, 110, 120), width=2)
# ---- 右：T3 HILO 曲线 ----
ox2, T2y, Bt2 = 750, 170, 430
d.text((ox2 - 40, 108), "② HILO：离散 vs 连续置信度", font=F(15), fill=(40, 40, 50))
HILOS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85]
px2 = lambda i: ox2 + 10 + i * 52
py2 = lambda v: Bt2 - (Bt2 - T2y) * (v - 0.68) / 0.18
for gv in np.arange(0.70, 0.851, 0.05):
    y = py2(gv)
    d.line([(ox2 - 30, y), (ox2 + 360, y)], fill=(234, 234, 240))
    d.text((ox2 - 74, y - 9), "%.2f" % gv, font=F(12), fill=(130, 130, 140))
for mode, col, nm in (("discrete", (150, 150, 160), "离散(0/1硬票)"), ("continuous", (30, 170, 90), "连续置信度")):
    pts = []
    for i, h in enumerate(HILOS):
        v = float(np.mean([r["ACC_final"] for r in T3 if r["mode"] == mode and abs(r["hilo"] - h) < 1e-9]))
        pts.append((px2(i), py2(v), v))
    d.line([(x, y) for x, y, _ in pts], fill=col, width=4, joint="curve")
    for x, y, v in pts:
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)
    d.text((ox2 + 250, py2(pts[-1][2]) + (-24 if mode == "continuous" else 12)), nm, font=F(12), fill=col)
for i, h in enumerate(HILOS):
    d.text((px2(i) - 12, Bt2 + 10), "%.2f" % h, font=F(11), fill=(90, 90, 100))
d.line([(ox2 - 30, T2y - 12), (ox2 - 30, Bt2)], fill=(110, 110, 120), width=2)
d.line([(ox2 - 30, Bt2), (ox2 + 380, Bt2)], fill=(110, 110, 120), width=2)
d.text((ox2 - 30, Bt2 + 30), "离散版跨度 0.009 = 开关   连续版跨度 0.114 = 旋钮", font=F(13), fill=(60, 60, 70))
for i, item in enumerate([
    ("① T2（真实噪声）：**5/6 通过**。我原来的假设被证伪 —— 我以为「翻到最像的类」最难查，实测 82%，和均匀噪声一样。", (20, 140, 70)),
    ("   质量漂移让优势从 82% 掉到 48%，但仍然通过（> 30%）。", (60, 60, 70)),
    ("   ❌ **只有「捷径相关噪声」不通过（23%）**：错误与某个特征相关（批次效应/采集偏差）时，模型能学会这条假规律，", (185, 50, 50)),
    ("      于是脏标签变得「自洽」，门控的 key-value 一致性检查就失效了。**这是真实噪声下唯一被证伪的场景。**", (185, 50, 50)),
    ("② T3：连续置信度把 HILO 从开关变成了真旋钮（跨度 0.009 -> 0.114），而且整条性价比曲线被抬高：", (20, 90, 180)),
    ("   376 次查询 -> 0.776，575 次 -> 0.792，1470 次 -> 0.832（全标注上界 0.851）。离散版在 97 次只有 0.725。", (20, 90, 180)),
]):
    d.text((30, 545 + i * 23), item[0], font=F(13), fill=item[1])
img.save(os.path.join(OUT, "t2t3_noise_and_hilo.png"))
print("saved")
