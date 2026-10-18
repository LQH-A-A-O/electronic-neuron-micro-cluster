# -*- coding: utf-8 -*-
"""阶段 9 / 10 出图。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R9 = json.load(open(os.path.join(OUT, "results_stage9.json"), encoding="utf-8"))
R10 = json.load(open(os.path.join(OUT, "results_stage10.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

W, H = 1180, 660
L, R, T, Bt = 130, 700, 140, 470

# ---------------- 图 1：阶段 9 ----------------
DEL = [40, 80, 160, 400, 800]
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 28), "阶段 9：把「待裁判窗口」拧小 —— 有用，但拧到底也只到 0.58", font=F(22), fill=(20, 20, 30))
d.text((L - 20, 64), "真实 MNIST / 结构化幻觉 40% / 标签按观测条数延迟到达 / 每 80 条观测做一次巩固+裁判 / 3 seeds", font=F(14), fill=(110, 110, 120))
py = lambda v: Bt - (Bt - T) * (v - 0.10) / (0.78 - 0.10)
for gv in np.arange(0.10, 0.781, 0.10):
    y = py(gv)
    d.line([(L, y), (R, y)], fill=(232, 232, 238))
    d.text((L - 52, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
d.line([(L, Bt), (R + 10, Bt)], fill=(110, 110, 120), width=2)
xs = [L + (R - L) * i / (len(DEL) - 1) for i in range(len(DEL))]
ys, es = [], []
for dl in DEL:
    v = [r["ACC_final"] for r in R9 if r["delay"] == dl]
    ys.append(float(np.mean(v))); es.append(float(np.std(v)))
for ref, col, nm in (("B_replay", (235, 110, 60), "B 无门控回放 0.481"), ("A_naive", (150, 150, 160), "A 无记忆 0.125")):
    v = np.mean([r["ACC_final"] for r in R9 if r.get("arm") == ref])
    d.line([(L, py(v)), (R, py(v))], fill=col, width=3)
    d.text((L + 8, py(v) - 22), nm, font=F(14), fill=col)
band = ([(xs[i], py(ys[i] + es[i])) for i in range(len(DEL))] + [(xs[i], py(ys[i] - es[i])) for i in range(len(DEL))][::-1])
d.polygon(band, fill=(214, 244, 226))
d.line(list(zip(xs, [py(v) for v in ys])), fill=(30, 170, 90), width=4, joint="curve")
for i, x in enumerate(xs):
    d.ellipse([x - 6, py(ys[i]) - 6, x + 6, py(ys[i]) + 6], fill=(30, 170, 90))
    d.text((x - 20, py(ys[i]) - 34), "%.3f" % ys[i], font=F(13), fill=(20, 140, 70))
    d.text((x - 26, Bt + 12), str(DEL[i]), font=F(14), fill=(90, 90, 100))
d.text(((L + R) / 2 - 80, Bt + 40), "标签延迟（观测条数）", font=F(15), fill=(80, 80, 90))
for i, (txt, col) in enumerate([
    ("窗口 800 条 -> 0.472；压到 80 条 -> 0.581，污染率 0.323 -> 0.231。阶段 8 指出的杠杆是对的。", (60, 60, 70)),
    ("但 40/80/160 已经打平（0.564/0.581/0.579）—— 收益在 160 条附近饱和，再拧没用。", (185, 50, 50)),
    ("而且离 gold 的 0.719 仍差 0.14：**窗口不是唯一的瓶颈**，标签本身的延迟无法被窗口补偿。", (60, 60, 70))]):
    d.text((L - 20, Bt + 74 + i * 22), txt, font=F(14), fill=col)
img.save(os.path.join(OUT, "stage9_window.png"))

# ---------------- 图 2：阶段 10 ----------------
MODES = ["B_gold", "C_peer", "D_matrix"]
LAB = {"B_gold": "B 全量 side\n(阶段1方案)", "C_peer": "C 只有热备互证", "D_matrix": "D 三通道矩阵"}
COL = {"B_gold": (150, 150, 160), "C_peer": (235, 110, 60), "D_matrix": (30, 170, 90)}
img2 = Image.new("RGB", (W, H), (255, 255, 255))
d2 = ImageDraw.Draw(img2)
d2.text((L - 20, 28), "阶段 10：三通道矩阵交叉验证 + 把成熟知识编译成工具", font=F(22), fill=(20, 20, 30))
d2.text((L - 20, 64), "5 个任务 = 5 条局部多项式规则 / 污染 40% / 查询时不告诉盒子 / 3 seeds；三组都允许编译，差别只在门控", font=F(14), fill=(110, 110, 120))
# 左：ACC
LX, RX = L, L + 250
acc = [np.mean([r["ACC_final"] for r in R10 if r["mode"] == m]) for m in MODES]
for i, m in enumerate(MODES):
    x = LX + i * 78
    y = 470 - (470 - 180) * acc[i] / 0.8
    d2.rectangle([x, y, x + 58, 470], fill=COL[m], outline=(255, 255, 255))
    d2.text((x + 2, y - 20), "%.3f" % acc[i], font=F(13), fill=(50, 50, 60))
    for j, ln in enumerate(LAB[m].split("\n")):
        d2.text((x - 6, 482 + j * 16), ln, font=F(12), fill=(70, 70, 80))
d2.text((LX, 140), "查询准确率", font=F(15), fill=(40, 40, 50))
d2.line([(LX, 160), (LX + 230, 160)], fill=(200, 200, 210))
# 右：side 查询次数
side = [np.mean([r["side_queries"] for r in R10 if r["mode"] == m]) for m in MODES]
for i, m in enumerate(MODES):
    x = RX + i * 78
    y = 470 - (470 - 180) * side[i] / 1200
    d2.rectangle([x, y, x + 58, 470], fill=COL[m], outline=(255, 255, 255))
    d2.text((x + 2, y - 20), "%.0f" % side[i], font=F(13), fill=(50, 50, 60))
d2.text((RX, 140), "side 查询次数（oracle 账单）", font=F(15), fill=(40, 40, 50))
d2.line([(RX, 160), (RX + 230, 160)], fill=(200, 200, 210))
d2.text((RX + 46, 500), "← 少 40%", font=F(16), fill=(20, 140, 70))
for i, (txt, col) in enumerate([
    ("三通道矩阵（D）准确率 0.661 与全量验证（B）的 0.671 统计上不可区分，但 side 查询从 1000 降到 604（-40%）。", (60, 60, 70)),
    ("关掉冷备锚只留热备互证（C）几乎不省钱（927）—— 说明真正省账单的是**已编译工具**，不是热备互证。", (185, 50, 50)),
    ("这正是你说的第 5 条：成熟知识编译成代码后，它本身就是一条同步、免费、覆盖自己定义域的验证通道。", (20, 140, 70))]):
    d2.text((L - 20, 552 + i * 22), txt, font=F(14), fill=col)
img2.save(os.path.join(OUT, "stage10_tool.png"))
print("saved stage9/10 charts")
