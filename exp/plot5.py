# -*- coding: utf-8 -*-
"""阶段 5 出图：柱状对比（PIL）。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
recs = json.load(open(os.path.join(OUT, "results_stage5.json"), encoding="utf-8"))

FONT = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(cand):
        FONT = cand; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

W, H = 1180, 660
L, R, T, Bt = 120, 700, 140, 500

def barv(v):  # 0..1 -> 像素高度
    return (Bt - T) * v

def draw(title, sub, groups, anns):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((L - 10, 30), title, font=F(23), fill=(20, 20, 30))
    d.text((L - 10, 68), sub, font=F(15), fill=(110, 110, 120))
    for gv in np.arange(0, 1.001, 0.2):
        y = Bt - barv(gv)
        d.line([(L, y), (R, y)], fill=(232, 232, 238))
        d.text((L - 52, y - 9), "%.1f" % gv, font=F(13), fill=(120, 120, 130))
    d.line([(L, T - 12), (L, Bt)], fill=(110, 110, 120), width=2)
    d.line([(L, Bt), (R + 10, Bt)], fill=(110, 110, 120), width=2)
    n = len(groups)
    slot = (R - L) / n
    bw = min(52, slot / 5.2)
    for gi, (glabel, bars) in enumerate(groups):
        cx = L + slot * (gi + 0.5)
        x0 = cx - (len(bars) * (bw + 8)) / 2
        for bi, (sname, val, col) in enumerate(bars):
            x = x0 + bi * (bw + 8)
            y = Bt - barv(max(0.0, val))
            d.rectangle([x, y, x + bw, Bt], fill=col, outline=(255, 255, 255))
            if val > 0.02:
                d.text((x + 2, y - 20), "%.3f" % val, font=F(12), fill=(50, 50, 60))
            d.text((x - 1, Bt + 9), sname, font=F(11), fill=(90, 90, 100))
        d.text((cx - len(glabel) * 5, Bt + 32), glabel, font=F(16), fill=(40, 40, 50))
    for i, (txt, col) in enumerate(anns):
        d.text((L - 10, Bt + 92 + i * 22), txt, font=F(14), fill=col)
    return img, d

C = {"A_naive": (150, 150, 160), "B_replay": (235, 110, 60), "C_tiered_good": (30, 170, 90),
     "D_tiered_demo": (170, 90, 210), "gold": (30, 170, 90), "delay1": (215, 70, 70),
     "delay2": (215, 120, 70), "consensus": (170, 90, 210)}
NAME = {"A_naive": "A 无记忆", "B_replay": "B 无门控回放", "C_tiered_good": "C 门控",
        "D_tiered_demo": "D 原演示验证器", "gold": "gold 即时标注", "delay1": "delay1 迟到1任务",
        "delay2": "delay2 迟到2任务", "consensus": "consensus 零标注自洽"}

def acc5a(mode, arm):
    v = [r["ACC_final"] for r in recs if r["scan"] == "A" and r["mode"] == mode and r["arm"] == arm]
    return float(np.mean(v)), float(np.std(v))

groups = []
for mode, mlabel in (("uniform", "均匀幻觉 40%"), ("structured", "结构化幻觉 40%")):
    SHORT = {"A_naive": "A无记忆", "B_replay": "B回放", "C_tiered_good": "C门控", "D_tiered_demo": "D原演示"}
    bars = [(SHORT[a], acc5a(mode, a)[0], C[a]) for a in
            ("A_naive", "B_replay", "C_tiered_good", "D_tiered_demo")]
    groups.append((mlabel, bars))
img, d = draw("阶段 5A：换成真实模型的「混淆方向」，门控一样挡得住",
              "MNIST / 5 任务 x 2 类 / 错误率都固定 40% / 3 seeds；结构化 = 按弱 MLP 的混淆矩阵犯错（4->9、5->3、7->2…）",
              groups,
              [("左：均匀随机翻转（阶段 1~4 用的假威胁模型）。右：结构化混淆 —— 门控的漏放率两者都是 0.003。", (60, 60, 70)),
               ("结论与预测相反：错误是随机还是混淆，门控根本不在乎 —— 它只看 key 和 value 匹不匹配。", (185, 50, 50)),
               ("所以真正能骗过门控的只有两种：key 本身就错了，或者错误恰好自洽。", (60, 60, 70))])
img.save(os.path.join(OUT, "stage5_hallucination.png"))

def acc5b(kind):
    v = [r["ACC_final"] for r in recs if r["scan"] == "B" and r["arm"] == "C_tiered_good" and r["kind"] == kind]
    return float(np.mean(v)), float(np.std(v))

groups2 = [("C 用不同的 oracle（结构化幻觉 40%）",
            [(k, acc5b(k)[0], C[k]) for k in ("gold", "delay1", "delay2", "consensus")] +
            [("B回放", float(np.mean([r["ACC_final"] for r in recs if r["scan"] == "B" and r["arm"] == "B_replay"])), C["B_replay"])])]
img2, d2 = draw("阶段 5B：不要钱的验证信号 —— 只有一条能用",
                "结构化幻觉 40%，全部由同一个门控（theta=0.7）处理，只换验证信号来源",
                groups2,
                [("gold（标签立刻可用）0.719。", (20, 140, 70)),
                 ("delay1 / delay2 = 0.125，和完全无记忆一模一样：验证池只覆盖旧类，新类永远验不了。", (185, 50, 50)),
                 ("consensus（零标注自洽）0.240，比不做门控的 0.481 还差 —— 污染率 0.404 ≈ 输入，一点没清掉。", (140, 70, 180)),
                 ("落地结论：只有「同步、且覆盖当前类」的验证信号（环境奖励/可机械校验）能直接用。", (60, 60, 70))])
img2.save(os.path.join(OUT, "stage5_free_oracle.png"))
print("saved stage5 charts")
