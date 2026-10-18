# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage18.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()
def g(m, k):
    v = [r[k] for r in R if r["mode"] == m and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
W, H, L, T, Bt = 1180, 740, 160, 160, 470
ARMS = ["one_phase", "two_phase", "park_verified", "park_verified_mem"]
LAB = {"one_phase": "一犹豫就查", "two_phase": "放置", "park_verified": "放置+批校验",
       "park_verified_mem": "放置+真值参照"}
COL = {"one_phase": (150, 150, 160), "two_phase": (235, 110, 60),
       "park_verified": (230, 170, 40), "park_verified_mem": (30, 170, 90)}
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "阶段 18：参照批的独立校验 —— 批级校验没用，「真值参照」才管用", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "真实 MNIST / 污染 40% / 注入坏锚 / 5 seeds；所有放置策略都先筛确定数据、再处理中立票", font=F(14), fill=(110, 110, 120))
PANELS = [("总体 ACC", "ACC_final", 0.80, "%.3f"), ("任务0 类 ACC(坏锚所在)", "ACC_task0", 1.00, "%.3f"),
          ("side 查询次数", "side", 750.0, "%.0f")]
for pi, item in enumerate(PANELS):
    pt, key, ymax, fmt = item
    ox = L + pi * 315
    def py(v, ymax=ymax):
        return Bt - (Bt - T) * min(max(v, 0), ymax) / ymax
    d.text((ox - 70, 110), pt, font=F(15), fill=(40, 40, 50))
    d.line([(ox - 70, 132), (ox + 170, 132)], fill=(200, 200, 210))
    for gv in np.arange(0, ymax + 1e-9, ymax / 4):
        y = py(gv)
        d.line([(ox - 70, y), (ox + 170, y)], fill=(234, 234, 240))
        d.text((ox - 132, y - 9), fmt % gv, font=F(12), fill=(130, 130, 140))
    for i, arm in enumerate(ARMS):
        x = ox + i * 64
        v = g(arm, key)
        y = py(v)
        d.rectangle([x, y, x + 50, Bt], fill=COL[arm], outline=(255, 255, 255))
        d.text((x - 1, y - 19), fmt % v, font=F(11), fill=(50, 50, 60))
        d.text((x - 8, Bt + 9), LAB[arm][:5], font=F(10), fill=(70, 70, 80))
    d.line([(ox - 70, T - 12), (ox - 70, Bt)], fill=(110, 110, 120), width=2)
    d.line([(ox - 70, Bt), (ox + 180, Bt)], fill=(110, 110, 120), width=2)
ANNS = [
    ("批级抽检确实在触发（15 个批被判不可信），但准确率没动：0.544 vs 放置的 0.546。", (200, 120, 40)),
    ("因为否掉本批之后，参照退回到历史记忆 —— 而历史记忆是同一套门控选出来的，偏差还在原地。", (185, 50, 50)),
    ("控制组给出答案：参照池只装真值验证过的记忆，任务0 类从 0.804 抬到 0.904，总体 0.546 -> 0.595。", (20, 140, 70)),
    ("结论：偏差不在某一批里，在整个记忆里。系统内的任何参照（本批确定样本、历史记忆、锚、peer）", (20, 140, 70)),
    ("都来自同一套选择过程，共享同一个偏差。唯一的独立来源是 oracle 本身。", (20, 140, 70)),
    ("这已是第三次撞同一堵墙：阶段11 同源污染 -> 阶段17 确定批!=正确批 -> 阶段18 换参照也逃不掉。", (60, 60, 70)),
]
for i, item in enumerate(ANNS):
    d.text((L - 20, 520 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage18_verify.png"))
print("saved")
