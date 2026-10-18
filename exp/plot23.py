# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
A = json.load(open(os.path.join(OUT, "results_stage23_b50.json"), encoding="utf-8"))
B = json.load(open(os.path.join(OUT, "results_stage23_b00.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
def v(recs, mode, k):
    x = [r[k] for r in recs if r["mode"] == mode and r[k] == r[k]]
    return float(np.mean(x)) if x else float("nan")
W, H = 1180, 740
L, T, Bt = 160, 175, 500
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "G3（阶段 23）：坏锚能不能自然产生？审计能不能抓到？", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "真实 MNIST / 只注入「数据源级错标」（任务0 前 200 条 80% 标签反转），不注入任何锚 / 5 seeds", font=F(13), fill=(110, 110, 120))
def panel(ox, title, pairs, ymax, fmt, note, line=None):
    d.text((ox, 112), title, font=F(15), fill=(40, 40, 50))
    d.line([(ox, 134), (ox + 210, 134)], fill=(200, 200, 210))
    py = lambda t: Bt - (Bt - T) * min(max(t, 0), ymax) / ymax
    for i, (lab, val, col) in enumerate(pairs):
        x = ox + 10 + i * 78
        y = py(val)
        d.rectangle([x, y, x + 56, Bt], fill=col, outline=(255, 255, 255))
        d.text((x - 2, y - 19), fmt % val, font=F(12), fill=(50, 50, 60))
        d.text((x - 6, Bt + 10), lab, font=F(11), fill=(70, 70, 80))
    if line is not None:
        y = py(line)
        d.line([(ox, y), (ox + 230, y)], fill=(200, 60, 60), width=2)
        d.text((ox + 150, y - 18), "判定线 %.2f" % line, font=F(11), fill=(185, 50, 50))
    d.text((ox, Bt + 34), note, font=F(12), fill=(120, 120, 130))
    d.line([(ox, T - 12), (ox, Bt)], fill=(110, 110, 120), width=2)
    d.line([(ox, Bt), (ox + 240, Bt)], fill=(110, 110, 120), width=2)
GREEN, GRAY, ORANGE = (30, 170, 90), (150, 150, 160), (235, 110, 60)
panel(L - 20, "工具在「自己管辖的类」上的最低准确率",
      [("无审计", v(A, "no_audit", "min_tool_acc"), GRAY), ("有审计", v(A, "audit", "min_tool_acc"), GREEN)],
      0.6, "%.3f", "2 类问题，随机=0.5；0.325 比瞎猜还差 -> 它在系统性地反着判", line=0.5)
panel(L + 300, "被抓住的坏轮数（共 3.6 轮）",
      [("可信度带平滑", v(A, "audit", "caught"), ORANGE), ("可信度无平滑", v(B, "audit", "caught"), GREEN)],
      4.0, "%.1f", "指数平滑太慢，坏锚只发作 3.6 轮，来不及跌到判定线")
panel(L + 600, "审计给出的最低可信度",
      [("带平滑", v(A, "audit", "min_cred"), ORANGE), ("无平滑", v(B, "audit", "min_cred"), GREEN)],
      1.0, "%.3f", "0.658 = 0.5x1.0 + 0.5x0.325 —— 它看见了，只是被平滑拖住", line=0.5)
for i, item in enumerate([
    ("D4 前半段 ✅ **坏锚确实能自然产生**：工具在自己管辖的类上准确率掉到 0.325（比 2 类瞎猜还差）。", (20, 140, 70)),
    ("       这是 23 个阶段里第一次 —— 之前 11~19 的坏锚全是人直接塞进工具箱的。", (60, 60, 70)),
    ("D4 后半段 ✅（有条件）**审计能抓到，但必须去掉可信度的指数平滑**：抓到 0/3.6 -> 2.4/3.6，可信度 0.658 -> 0.312。", (20, 140, 70)),
    ("🚩 但它**对最终准确率完全没有影响**（两臂都是 0.669）—— 因为坏锚只发作 3.6/40 轮，记忆检索兜住了。", (185, 50, 50)),
    ("   所以当前设置下，审计的价值是**保险**（防止长期错下去），不是**收益**（提高准确率）。", (60, 60, 70)),
    ("另一个教训：**降权是对的，但权重的更新不能慢**。指数平滑会让它错过短期发作（阶段15 只说了该降权，没说该多快降）。", (20, 90, 180)),
]):
    d.text((L - 20, 545 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage23_natural_bad_anchor.png"))
print("saved")
