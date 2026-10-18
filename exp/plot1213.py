# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
R12 = json.load(open(os.path.join(OUT, "results_stage12.json"), encoding="utf-8"))
R13 = json.load(open(os.path.join(OUT, "results_stage13.json"), encoding="utf-8"))
FONT = None
for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(c):
        FONT = c; break
def F(sz):
    if FONT:
        try: return ImageFont.truetype(FONT, sz)
        except Exception: pass
    return ImageFont.load_default()

W, H, L, T, Bt = 1180, 700, 140, 150, 470

def bars(title, sub, groups, ymax, anns, fname, width=64):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((L - 20, 28), title, font=F(23), fill=(20, 20, 30))
    d.text((L - 20, 64), sub, font=F(14), fill=(110, 110, 120))
    for gv in np.arange(0, ymax + 1e-9, ymax / 6):
        y = Bt - (Bt - T) * gv / ymax
        d.line([(L - 20, y), (L + 780, y)], fill=(232, 232, 238))
        d.text((L - 62, y - 9), "%.2f" % gv, font=F(13), fill=(120, 120, 130))
    x = L - 20
    for gname, items in groups:
        if gname:
            cx = x + (len(items) * (width + 10)) / 2
            d.text((cx - len(gname) * 6, Bt + 34), gname, font=F(16), fill=(40, 40, 50))
        for lab, val, col in items:
            y = Bt - (Bt - T) * min(val, ymax) / ymax
            d.rectangle([x, y, x + width, Bt], fill=col, outline=(255, 255, 255))
            d.text((x + 2, y - 19), "%.3f" % val, font=F(13), fill=(50, 50, 60))
            d.text((x - 2, Bt + 12), lab, font=F(12), fill=(70, 70, 80))
            x += width + 10
        x += 46
    d.line([(L - 20, T - 12), (L - 20, Bt)], fill=(110, 110, 120), width=2)
    d.line([(L - 20, Bt), (L + 800, Bt)], fill=(110, 110, 120), width=2)
    for i, (txt, col) in enumerate(anns):
        d.text((L - 20, 540 + i * 23), txt, font=F(15), fill=col)
    img.save(os.path.join(OUT, fname))
    return img

def a12(m, k="ACC_final"):
    v = [r[k] for r in R12 if r["mode"] == m and r[k] == r[k]]
    return float(np.mean(v)) if v else 0.0

def a13(m, k="ACC_task0"):
    v = [r[k] for r in R13 if r["mode"] == m and r[k] == r[k]]
    return float(np.mean(v)) if v else 0.0

C = {"gold_mem": (60, 130, 230), "gold_tool": (30, 170, 90), "toolonly": (235, 110, 60),
     "B_replay": (150, 150, 160), "A_naive": (200, 200, 205)}
C13 = {"no_audit": (150, 150, 160), "audit_active": (235, 110, 60),
       "audit_trace": (30, 170, 90), "audit_revisit": (60, 130, 230)}
L12 = {"gold_mem": "gold+记忆", "gold_tool": "gold+编译", "toolonly": "零标注+编译",
       "B_replay": "B回放", "A_naive": "A无记忆"}
L13 = {"no_audit": "无审计", "audit_active": "主动抽检", "audit_trace": "+连带撤回", "audit_revisit": "+被拒重审"}

bars("阶段 12：把「工具化」搬到真实数据（MNIST）上",
     "MNIST / 784 维先随机投影到 128 维 / 5 任务 x 2 类 / 污染 40% / 3 seeds；工具 = 每类一个线性判别式（全库共 1290 个参数）",
     [("", [(L12[m], a12(m), C[m]) for m in ("gold_mem", "gold_tool", "toolonly", "B_replay", "A_naive")])],
     0.90,
     [("gold+记忆 0.814 vs gold+编译 0.779：真实知识上编译**有损**，掉 3.5 个点（合成多项式域上是零损失）。", (185, 50, 50)),
      ("但零标注的工具链（自举 80 条 -> 编译 -> 用工具验证后续）拿到 0.612，碾压无门控回放的 0.451。", (20, 140, 70)),
      ("离 gold 的 0.814 还差 0.20：差距主要来自自举那一小批（80 条）里就有 40% 是脏的，工具的质量被它拖住。", (60, 60, 70))],
     "stage12_real_tool.png")

bars("阶段 13：溯源 + 连带撤回 + 被拒重审（任务0 区域准确率）",
     "同源污染 60% + 注入坏锚；任务0 区域 = 坏锚的作案现场。0.023 是「系统自信地错」的基线",
     [("", [(L13[m], a13(m), C13[m]) for m in ("no_audit", "audit_active", "audit_trace", "audit_revisit")])],
     0.16,
     [("链路是通的：0.023 -> 0.067（主动抽检）-> 0.100（连带撤回 48 条从犯记忆）-> 0.097（重审捞回 18 条）。", (20, 140, 70)),
      ("但绝对值仍然不可用。根因不是审计不够狠，而是：**坏锚在入口处就把正确知识拒了**，", (185, 50, 50)),
      ("事后能删错、能捞回零星几条，**补不回被拒掉的那一大批**。审计能止损，不能追回。", (185, 50, 50))],
     "stage13_trace.png")
print("saved stage12/13 charts")
