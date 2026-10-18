# -*- coding: utf-8 -*-
"""阶段 6 / 7 出图（PIL 分组柱状）。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R6 = json.load(open(os.path.join(OUT, "results_stage6.json"), encoding="utf-8"))
R7 = json.load(open(os.path.join(OUT, "results_stage7.json"), encoding="utf-8"))

FONT = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(cand):
        FONT = cand; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

W, H = 1180, 700
L, R, T, Bt = 120, 700, 140, 480
CN = {"MLP_gated": (150, 150, 160), "K_warm": (235, 110, 60), "K_cold": (60, 130, 230),
      "K_fixed": (170, 90, 210), "K_adapt": (30, 170, 90), "K_adapt_ema": (20, 200, 130)}
SH = {"MLP_gated": "MLP判断", "K_warm": "纯温备", "K_cold": "纯冷备", "K_fixed": "固定0.5",
      "K_adapt": "自适应", "K_adapt_ema": "自适应+EMA"}


def draw(title, sub, groups, y0, y1, anns, barcap=6):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((L - 10, 28), title, font=F(22), fill=(20, 20, 30))
    d.text((L - 10, 64), sub, font=F(15), fill=(110, 110, 120))
    py = lambda v: Bt - (Bt - T) * (v - y0) / (y1 - y0)
    for gv in np.arange(y0, y1 + 1e-9, (y1 - y0) / 8):
        y = py(gv)
        d.line([(L, y), (R, y)], fill=(232, 232, 238))
        d.text((L - 52, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
    d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
    d.line([(L, Bt), (R + 10, Bt)], fill=(110, 110, 120), width=2)
    slot = (R - L) / len(groups)
    bw = min(46, slot / (barcap + 1.6))
    for gi, (glabel, bars) in enumerate(groups):
        cx = L + slot * (gi + 0.5)
        x0 = cx - (len(bars) * (bw + 7)) / 2
        for bi, (key, val) in enumerate(bars):
            x = x0 + bi * (bw + 7)
            y = py(max(y0, val))
            d.rectangle([x, y, x + bw, Bt], fill=CN[key], outline=(255, 255, 255))
            if val > y0 + 0.015:
                d.text((x + 1, y - 19), "%.3f" % val, font=F(11), fill=(50, 50, 60))
            d.text((x - 1, Bt + 9), SH[key][:5], font=F(10), fill=(90, 90, 100))
        d.text((cx - len(glabel) * 6, Bt + 32), glabel, font=F(16), fill=(40, 40, 50))
    for i, (txt, col) in enumerate(anns):
        d.text((L - 10, Bt + 66 + i * 22), txt, font=F(14), fill=col)
    return img, d


def a6(drift, cap, arm, key="ACC"):
    v = [r[key] for r in R6 if r["drift"] == drift and r["cap"] == cap and r["arm"] == arm and r[key] == r[key]]
    return float(np.mean(v)) if v else float("nan")

ORDER = ["MLP_gated", "K_warm", "K_cold", "K_fixed", "K_adapt", "K_adapt_ema"]
g = [("温备 60 条 / drift=0.6", [(a, a6(0.6, 60, a)) for a in ORDER]),
     ("温备 300 条 / drift=0.6", [(a, a6(0.6, 300, a)) for a in ORDER])]
img, d = draw("阶段 6：冷备当「判断的锚」，温备当「累积的经验」",
              "25 类 / 每类出现 4 次 / 数据噪声 30% / 3 seeds；p(y|x) = λ(x)·p_warm + (1-λ(x))·p_cold，λ 由 x 附近的经验量决定",
              g, 0.70, 1.01,
              [("温备越小，冷备越值钱：自适应仲裁在 60 条时 0.975，纯温备只有 0.889，MLP 判断 0.752。", (60, 60, 70)),
               ("温备够大时自适应仲裁自动退化成「信经验」（0.998 ≈ 纯温备 0.999）—— 正是设计意图。", (60, 60, 70)),
               ("注意：纯冷备（冻结、从不更新）在 60 条时是 0.966，也远好于纯温备 —— 见下方机制图。", (185, 50, 50))])
img.save(os.path.join(OUT, "stage6_anchor.png"))

BK = ["bucket_fresh", "bucket_mid", "bucket_old"]
g2 = [(SH[a], [(BK[0], a6(0.6, 60, a, "bucket_fresh")), (BK[1], a6(0.6, 60, a, "bucket_mid")),
               (BK[2], a6(0.6, 60, a, "bucket_old"))]) for a in ORDER]
CN2 = {"bucket_fresh": (30, 170, 90), "bucket_mid": (230, 170, 40), "bucket_old": (215, 70, 70)}
img2 = Image.new("RGB", (W, H), (255, 255, 255))
d2 = ImageDraw.Draw(img2)
d2.text((L - 10, 28), "阶段 6 机制图：冷备赢在「类别覆盖」，不在「信息新鲜度」", font=F(22), fill=(20, 20, 30))
d2.text((L - 10, 64), "按「这个类最后一次出现距今多少个任务」分桶（drift=0.6，温备 60 条，3 seeds）", font=F(15), fill=(110, 110, 120))
py = lambda v: Bt - (Bt - T) * (v - 0.6) / (1.02 - 0.6)
for gv in np.arange(0.6, 1.021, 0.05):
    y = py(gv)
    d2.line([(L, y), (R, y)], fill=(232, 232, 238))
    d2.text((L - 52, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
d2.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
d2.line([(L, Bt), (R + 10, Bt)], fill=(110, 110, 120), width=2)
slot = (R - L) / len(g2)
bw = min(46, slot / 3.9)
for gi, (glabel, bars) in enumerate(g2):
    cx = L + slot * (gi + 0.5)
    x0 = cx - (3 * (bw + 7)) / 2
    for bi, (bk, val) in enumerate(bars):
        x = x0 + bi * (bw + 7)
        y = py(max(0.6, val))
        d2.rectangle([x, y, x + bw, Bt], fill=CN2[bk], outline=(255, 255, 255))
        d2.text((x + 1, y - 19), "%.3f" % val, font=F(11), fill=(50, 50, 60))
    d2.text((cx - len(glabel) * 7, Bt + 12), glabel, font=F(15), fill=(40, 40, 50))
d2.rectangle([L + 4, Bt + 42, L + 20, Bt + 56], fill=CN2["bucket_fresh"])
d2.text((L + 26, Bt + 40), "新鲜(0-1)", font=F(13), fill=(60, 60, 70))
d2.rectangle([L + 130, Bt + 42, L + 146, Bt + 56], fill=CN2["bucket_mid"])
d2.text((L + 152, Bt + 40), "中等(2-4)", font=F(13), fill=(60, 60, 70))
d2.rectangle([L + 256, Bt + 42, L + 272, Bt + 56], fill=CN2["bucket_old"])
d2.text((L + 278, Bt + 40), "陈旧(5-9)", font=F(13), fill=(60, 60, 70))
for i, (txt, col) in enumerate([
        ("纯温备在陈旧类上崩到 0.654 —— 温备容量有限，老类被淘汰掉了；纯冷备是 0.997（它从不淘汰）。", (185, 50, 50)),
        ("自适应仲裁 0.993：新鲜类上信经验，陈旧类上退回先验，两头都不塌。", (20, 140, 70))]):
    d2.text((L - 10, Bt + 72 + i * 22), txt, font=F(14), fill=col)
img2.save(os.path.join(OUT, "stage6_mechanism.png"))

AR7 = ["gold", "delay1(坏)", "delay1+回滚", "delay2+回滚", "delay3+回滚", "B_replay", "A_naive"]
C7 = {"gold": (30, 170, 90), "delay1(坏)": (215, 70, 70), "delay1+回滚": (230, 170, 40),
      "delay2+回滚": (240, 190, 70), "delay3+回滚": (250, 210, 110), "B_replay": (235, 110, 60),
      "A_naive": (150, 150, 160)}
def a7(name):
    v = [r["ACC_final"] for r in R7 if r["arm"] == name]
    return float(np.mean(v))
img3 = Image.new("RGB", (W, H), (255, 255, 255))
d3 = ImageDraw.Draw(img3)
d3.text((L - 10, 28), "阶段 7：把裁判推迟 —— 延迟标签从「完全无用」救回可用", font=F(22), fill=(20, 20, 30))
d3.text((L - 10, 64), "真实 MNIST / 结构化幻觉 40% / 3 seeds；先收下（provisional），标签到了再撤回被证伪的", font=F(15), fill=(110, 110, 120))
py3 = lambda v: Bt - (Bt - T) * v / 0.78
for gv in np.arange(0, 0.7801, 0.1):
    y = py3(gv)
    d3.line([(L, y), (R, y)], fill=(232, 232, 238))
    d3.text((L - 52, y - 9), "%.1f" % gv, font=F(13), fill=(120, 120, 130))
d3.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
d3.line([(L, Bt), (R + 10, Bt)], fill=(110, 110, 120), width=2)
slot = (R - L) / len(AR7)
bw = min(64, slot / 1.7)
for i, name in enumerate(AR7):
    x = L + slot * (i + 0.5) - bw / 2
    val = a7(name)
    y = py3(val)
    d3.rectangle([x, y, x + bw, Bt], fill=C7[name], outline=(255, 255, 255))
    d3.text((x + 2, y - 20), "%.3f" % val, font=F(13), fill=(50, 50, 60))
    d3.text((x - 6, Bt + 12), name, font=F(13), fill=(60, 60, 70))
for i, (txt, col) in enumerate([
        ("delay1 什么都不做 = 0.125，精确等于完全无记忆（验证池只覆盖旧类，新类永远验不了）。", (185, 50, 50)),
        ("加回滚后 0.471 / 0.507 / 0.523 —— 追平并略超「不做门控的回放」0.481，但离 gold 0.719 还差一截。", (20, 140, 70)),
        ("意外：lag 越大反而略好。这个顺序我目前没有确证的解释，只报告现象。", (140, 70, 180))]):
    d3.text((L - 10, Bt + 60 + i * 22), txt, font=F(14), fill=col)
img3.save(os.path.join(OUT, "stage7_rollback.png"))
print("saved stage6/7 charts")
