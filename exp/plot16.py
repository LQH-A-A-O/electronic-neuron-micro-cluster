# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R = json.load(open(os.path.join(OUT, "results_stage16.json"), encoding="utf-8"))
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
W, H, L, T, Bt = 1180, 700, 120, 150, 460
ARMS = ["no_anchor", "hard", "weighted", "B_replay", "A_naive"]
LAB = {"no_anchor": "不注入坏锚", "hard": "硬判决撤锚", "weighted": "降权",
       "B_replay": "B回放", "A_naive": "A无记忆"}
COL = {"no_anchor": (60, 130, 230), "hard": (150, 150, 160), "weighted": (30, 170, 90),
       "B_replay": (235, 110, 60), "A_naive": (200, 200, 205)}
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((L - 20, 26), "阶段 16：把「坏锚 -> 审计 -> 撤销 vs 降权」整条链路搬到真实 MNIST", font=F(21), fill=(20, 20, 30))
d.text((L - 20, 60), "5 任务 x 2 类 / 128 维固定随机投影 / 污染 40% / 注入一个「打乱标签」的坏锚 / 3 seeds", font=F(14), fill=(110, 110, 120))
PANELS = [("总体 ACC", "ACC_final", 0.80), ("最终剩余锚数", "tools", 45.0), ("side 查询次数(oracle 账单)", "side", 900.0)]
for pi, item in enumerate(PANELS):
    pt, key, ymax = item
    ox = L + pi * 300
    def py(v, ymax=ymax):
        return Bt - (Bt - T) * min(max(v, 0), ymax) / ymax
    d.text((ox, 100), pt, font=F(15), fill=(40, 40, 50))
    d.line([(ox, 122), (ox + 240, 122)], fill=(200, 200, 210))
    for gv in np.arange(0, ymax + 1e-9, ymax / 4):
        y = py(gv)
        d.line([(ox, y), (ox + 255, y)], fill=(234, 234, 240))
        d.text((ox - 46, y - 9), "%.0f" % gv if ymax > 2 else "%.2f" % gv, font=F(12), fill=(130, 130, 140))
    for i, arm in enumerate(ARMS):
        x = ox + i * 52
        v = g(arm, key)
        if v != v:
            continue
        y = py(v)
        d.rectangle([x, y, x + 40, Bt], fill=COL[arm], outline=(255, 255, 255))
        fmt = "%.3f" if key == "ACC_final" else "%.0f"
        d.text((x - 2, y - 18), fmt % v, font=F(11), fill=(50, 50, 60))
        d.text((x - 4, Bt + 9), LAB[arm][:4], font=F(10), fill=(70, 70, 80))
    d.line([(ox, T - 12), (ox, Bt)], fill=(110, 110, 120), width=2)
    d.line([(ox, Bt), (ox + 265, Bt)], fill=(110, 110, 120), width=2)
ANNS = [
    ("核心结论在真实数据上复现：降权 0.739 > 硬判决 0.719，而且降权 >= 不注入坏锚的 0.735（都在噪声内）。", (20, 140, 70)),
    ("硬判决的病灶在真实数据上更明显：38 个锚被砍到 2.3 个 —— 它几乎撤掉了全部工具。", (185, 50, 50)),
    ("（准确率没崩是因为记忆检索兜住了；但这说明撤销是'宁可错杀'。）", (185, 50, 50)),
    ("新代价（合成域里看不到）：降权把 side 查询从 309 推到 856，2.8 倍。", (200, 120, 40)),
    ("机制：低可信度的锚不给硬票、而是给 0.5「中立票」，于是 score 频繁落进 [0.35,0.65] 犹豫区，每次都触发一次验证。", (60, 60, 70)),
    ("下一步很清楚：把「中立票 0.5」改成「真弃权 None」—— 让没底气的锚**闭嘴**，而不是和稀泥。", (20, 90, 180)),
]
for i, item in enumerate(ANNS):
    d.text((L - 20, 512 + i * 24), item[0], font=F(14), fill=item[1])
img.save(os.path.join(OUT, "stage16_real_audit.png"))
print("saved")
