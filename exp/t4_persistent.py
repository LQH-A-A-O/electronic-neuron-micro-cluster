# -*- coding: utf-8 -*-
"""T4：**持续污染**下的审计 —— 它能不能从"保险"变成"收益"？

阶段 23 的结论与它的边界
------------------------------------------------
阶段 23 里坏锚确实自然产生了（工具在管辖类上掉到 0.325），审计去掉指数平滑后也抓到了
（2.4/3.6 轮）。**但最终准确率一点没变**（两臂都是 0.669）。

当时的解释：污染窗口是**有限**的（前 200 条），坏锚很短暂，工具会随干净数据自我修复，
所以审计只是"保险"。

这一轮把那个边界推掉
------------------------------------------------
让**污染贯穿整个流**（每个任务的每一条都按数据源级错标），坏锚就无法自愈。
三种持续程度：
    once        只有任务 0 的前 200 条（= 阶段 23，对照组）
    all_task0   任务 0 全程错标
    every_task  每个任务都全程错标

对每种程度比 no_audit / audit（可信度无平滑）。
判据：**审计能不能让最终 ACC 显著不同**（> 0.03 才算"收益"）。
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp.stage23 as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
S.CRED_BLEND = 0.0
SEEDS = [0, 1, 2, 3, 4]
LEVELS = ["once", "all_task0", "every_task"]


def corrupt(st, level, seed):
    rng = np.random.default_rng(4242 + seed)
    for ti, task in enumerate(st.tasks):
        a, b = task.classes
        y = task.y_train.copy()
        inv = np.where(y == a, b, a)
        if level == "once":
            if ti != 0:
                continue
            n = 200
        elif level == "all_task0":
            if ti != 0:
                continue
            n = len(y)
        else:
            n = len(y)
        frac = 0.8 if level != "every_task" else 0.8
        hit = rng.choice(n, size=int(n * frac), replace=False)
        y[hit] = inv[hit]
        task.y_train = y
    return st


def run(level, mode, seed):
    st = S.make_stream(seed)
    st = corrupt(st, level, seed)
    R = S.R23(mode, seed)
    accs = []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        px, py = [], []
        for j in range(t + 1):
            px.extend(list(st.tasks[j].x_holdout)); py.extend(list(st.tasks[j].y_holdout))
        for s0 in range(0, len(task.x_train), S.ROUND):
            R.round(task.x_train[s0:s0 + S.ROUND], task.y_train[s0:s0 + S.ROUND], tf,
                    np.array(px), np.array(py))
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    th = np.array(R.tool_hist, dtype=float)
    ch = np.array(R.tool_cred_hist, dtype=float)
    bad = np.where(th < 0.5)[0]
    caught = int(np.sum(ch[bad] < 0.5)) if len(bad) else 0
    return dict(level=level, mode=mode, seed=seed, ACC_final=accs[-1], side=R.side,
                bad_rounds=int(len(bad)), caught=caught,
                min_tool_acc=float(np.nanmin(th)) if len(th) else float("nan"),
                min_cred=float(np.nanmin(ch)) if len(ch) else float("nan"))


if __name__ == "__main__":
    res = []
    for level in LEVELS:
        for seed in SEEDS:
            for mode in ("no_audit", "audit"):
                res.append(run(level, mode, seed))
            print("%-11s seed=%d done" % (level, seed)); sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_t4_persistent.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 96)
    print("T4：持续污染下审计是『保险』还是『收益』（真实 MNIST / 5 seeds）")
    print("=" * 96)
    print("%-12s %-12s %-12s %-14s %-13s %-11s %s"
          % ("污染程度", "无审计 ACC", "有审计 ACC", "审计带来的差", "坏锚轮数", "被抓到", "最低可信度"))
    for level in LEVELS:
        na = [r for r in res if r["level"] == level and r["mode"] == "no_audit"]
        au = [r for r in res if r["level"] == level and r["mode"] == "audit"]
        a0 = np.mean([r["ACC_final"] for r in na]); a1 = np.mean([r["ACC_final"] for r in au])
        print("%-12s %-12.3f %-12.3f %-14s %-13.1f %-11.1f %.3f"
              % (level, a0, a1, "%+.4f" % (a1 - a0), np.mean([r["bad_rounds"] for r in na]),
                 np.mean([r["caught"] for r in au]), np.nanmean([r["min_cred"] for r in au])))
    print()
    for level in LEVELS:
        na = [r for r in res if r["level"] == level and r["mode"] == "no_audit"]
        au = [r for r in res if r["level"] == level and r["mode"] == "audit"]
        d = np.mean([r["ACC_final"] for r in au]) - np.mean([r["ACC_final"] for r in na])
        print("%-12s 审计收益 %+.4f -> %s" % (level, d, "收益" if d > 0.03 else "保险（不改变结果）"))
