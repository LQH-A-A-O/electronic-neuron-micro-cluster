# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
T5 = json.load(open(os.path.join(OUT, "results_t5_auditmem.json"), encoding="utf-8"))
T6 = json.load(open(os.path.join(OUT, "results_t6_shortcut.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
W, H = 1180, 780
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((30, 22), "T5/T6：审计打到记忆上就管用了；捷径噪声方向复现，但结论不稳", font=F(20), fill=(20, 20, 30))
# ---- 左：T5 ----
ox, T, Bt = 130, 160, 400
py = lambda v: Bt - (Bt - T) * v / 0.65
d.text((ox - 60, 105), "① T5 持续污染下：审计打到记忆条目上", font=F(15), fill=(40, 40, 50))
for gv in np.arange(0, 0.651, 0.15):
    y = py(gv)
    d.line([(ox - 40, y), (ox + 300, y)], fill=(234, 234, 240))
    d.text((ox - 86, y - 9), "%.2f" % gv, font=F(12), fill=(130, 130, 140))
MODES = ["no_audit", "audit_tool", "audit_mem"]
LAB = ["无审计", "只评锚\n(旧机制)", "打到记忆\n(新机制)"]
for i, m in enumerate(MODES):
    rs = [r for r in T5 if r["mode"] == m]
    a = float(np.mean([r["ACC_final"] for r in rs]))
    x = ox + i * 92
    col = (200, 60, 60) if a < 0.4 else (30, 170, 90)
    d.rectangle([x, py(a), x + 64, Bt], fill=col, outline=(255, 255, 255))
    d.text((x - 4, py(a) - 20), "%.3f" % a, font=F(14), fill=(50, 50, 60))
    for j, ln in enumerate(LAB[i].split("\n")):
        d.text((x - 4, Bt + 10 + j * 15), ln, font=F(11), fill=(70, 70, 80))
    d.text((x - 16, Bt + 44), "坏锚 %.1f 轮" % np.mean([r["bad_rounds"] for r in rs]), font=F(10), fill=(150, 150, 160))
d.line([(ox - 40, T - 12), (ox - 40, Bt)], fill=(110, 110, 120), width=2)
d.line([(ox - 40, Bt), (ox + 300, Bt)], fill=(110, 110, 120), width=2)
d.text((ox - 60, Bt + 74), "0.352 -> 0.579（+0.227）；坏锚 28.2 -> 0.2 轮", font=F(13), fill=(20, 140, 70))
# ---- 右：T6 ----
ox2, T2y, Bt2 = 690, 160, 400
d.text((ox2 - 40, 105), "② T6：同花费下 ΔACC（线 = +0.05）", font=F(15), fill=(40, 40, 50))
lo, hi = -0.09, 0.11
py2 = lambda v: Bt2 - (Bt2 - T2y) * (v - lo) / (hi - lo)
for gv in [-0.05, 0.0, 0.05, 0.10]:
    y = py2(gv)
    d.line([(ox2 - 30, y), (ox2 + 340, y)], fill=(234, 234, 240))
    d.text((ox2 - 74, y - 9), "%+.2f" % gv, font=F(12), fill=(130, 130, 140))
d.line([(ox2 - 30, py2(0.05)), (ox2 + 340, py2(0.05))], fill=(200, 60, 60), width=2)
DSS = ["adult", "ml100k", "cancer"]
for i, ds in enumerate(DSS):
    for j, m in enumerate(["uniform", "shortcut"]):
        rs = [r for r in T6 if r["ds"] == ds and r["model"] == m]
        v = float(np.nanmean([r["delta_acc"] for r in rs]))
        x = ox2 - 30 + i * 118 + j * 54
        y = py2(v)
        col = (150, 150, 160) if m == "uniform" else (235, 110, 60)
        base = py2(0.0)
        d.rectangle([x, min(y, base), x + 44, max(y, base)], fill=col, outline=(255, 255, 255))
        d.text((x - 4, y - 20 if v >= 0 else y + 4), "%+.3f" % v, font=F(11), fill=(50, 50, 60))
    d.text((ox2 - 30 + i * 118, Bt2 + 10), ds, font=F(12), fill=(70, 70, 80))
d.line([(ox2 - 30, T2y - 12), (ox2 - 30, Bt2)], fill=(110, 110, 120), width=2)
d.line([(ox2 - 30, Bt2), (ox2 + 350, Bt2)], fill=(110, 110, 120), width=2)
d.text((ox2 - 30, Bt2 + 32), "灰=均匀噪声   橙=捷径噪声（三个数据集上橙都比灰低）", font=F(12), fill=(110, 110, 120))
for i, item in enumerate([
    ("① **T5 是决定性的正面结果**：把每轮抽检拿到的真值打到**记忆条目**上（不一致就降权），持续污染下总体 ACC", (20, 140, 70)),
    ("   从 0.352 提到 **0.579（+0.227）**，而只评锚的旧机制收益是 0.0000。**坏锚轮数顺带从 28.2 掉到 0.2** ——", (20, 140, 70)),
    ("   记忆洗干净了，锚的问题自动消失。F23「审计只评锚不评记忆」被一个修复实验证实。", (20, 140, 70)),
    ("② T6：捷径噪声让门控变差这个**方向在三个数据集上都复现**（adult +0.018->-0.009、ml100k -0.006->-0.059、cancer +0.082->-0.045）。", (60, 60, 70)),
    ("   但两个坏消息：(a) **「节省比例」这个指标极不稳** —— adult 在阶段26 是「省 40% 通过」，这里换成 ΔACC 是「+0.018 不通过」；", (185, 50, 50)),
    ("   (b) **维度假说也不牢** —— cancer 只有 30 维却通过（+0.082）。所以我上一轮说的「边界在维度」**站不住**。", (185, 50, 50)),
]):
    d.text((30, 535 + i * 23), item[0], font=F(13), fill=item[1])
img.save(os.path.join(OUT, "t5t6_auditmem_shortcut.png"))
print("saved")
