# -*- coding: utf-8 -*-
"""T6：捷径相关噪声的失效，能不能在**第二个数据集**上复现？

T2 在 MNIST 上发现：错误挂在一个特征上时（批次效应/采集偏差），门控优势从 82% 垮到 23%。
但那是**单点结论**。这一轮换成 adult（197 维表格）和 ml100k（32 维推荐）再试一次。

对照：同样两个数据集上跑均匀翻转，作为"应该通过"的基线。
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp.stage26 as S26
from exp.stage25 import R25

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
ROUND = S26.ROUND
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["adult", "ml100k", "cancer"]
MODELS = ["uniform", "shortcut"]


def repollute(tasks, model, seed):
    """把 build() 里默认的均匀污染换成目标噪声模型。"""
    rng = np.random.default_rng(31337 + seed)
    for task in tasks:
        y = task.y_train_true.copy()
        cls = sorted(set(y.tolist()))
        n = len(y)
        k = int(n * 0.4)
        if model == "uniform":
            idx = rng.choice(n, size=k, replace=False)
            for i in idx:
                others = [c for c in cls if c != y[i]]
                y[i] = others[rng.integers(0, len(others))]
        elif model == "shortcut":
            # 与标签无关的确定性规则：某个特征落在高分位就翻
            v = task.x_train[:, 0]
            thr = float(np.quantile(v, 0.6))
            for i in np.flatnonzero(v >= thr)[:k]:
                others = [c for c in cls if c != y[i]]
                y[i] = others[rng.integers(0, len(others))]
        else:
            raise ValueError(model)
        task.y_train = y
    return tasks


def run(name, model, seed):
    tasks = S26.build(name, seed)
    tasks = repollute(tasks, model, seed)
    ncls = int(max(t.y_test.max() for t in tasks)) + 1
    accs = {}
    for mode, B in [("gate", -1)] + [("naive", b) for b in S26.BUDGETS]:
        R = R25(mode, seed, B if B >= 0 else -1)
        if mode == "gate":
            R.mode = "gate"
        a = []
        for t, task in enumerate(tasks):
            hv_x, hv_y = task.x_holdout, task.y_holdout
            tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
            for s0 in range(0, len(task.x_train), ROUND):
                R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
            accs["acc%d" % t] = [float((R.predict(tasks[j].x_test) == tasks[j].y_test).mean()) for j in range(t + 1)]
            a.append(float(np.mean(accs["acc%d" % t])))
        accs["%s_%d" % (mode, B)] = (float(a[-1]), R.side)
    ga, gs = accs["gate_-1"]
    xs = [(accs["naive_%d" % b][1], accs["naive_%d" % b][0]) for b in S26.BUDGETS]
    need = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if y0 <= ga <= y1 and y1 > y0:
            need = x0 + (ga - y0) / (y1 - y0) * (x1 - x0); break
    saved = (need - gs) / need if need == need and need > 0 else float("nan")
    # 更稳的指标：在"门控实际花费"这个点上，朴素能拿到多少 ACC —— 然后比差值
    nat = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if x0 <= gs <= x1 and x1 > x0:
            nat = y0 + (gs - x0) / (x1 - x0) * (y1 - y0); break
    if nat != nat:
        nat = xs[0][1] if gs <= xs[0][0] else (xs[-1][1] if gs >= xs[-1][0] else float("nan"))
    return dict(ds=name, model=model, seed=seed, gate_acc=ga, gate_side=gs, need=need, saved=saved,
                naive_at_gate=nat, delta_acc=(ga - nat) if nat == nat else float("nan"),
                raw=accs["naive_0"][0], full=accs["naive_50"][0],
                headroom=accs["naive_50"][0] - accs["naive_0"][0])


if __name__ == "__main__":
    res = []
    for name in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                r = run(name, model, seed); res.append(r)
                print("%-9s %-9s seed=%d 门控 %.3f@%d 省 %s 头寸 %.3f"
                      % (name, model, seed, r["gate_acc"], r["gate_side"],
                         ("%.0f%%" % (100 * r["saved"])) if r["saved"] == r["saved"] else "n/a", r["headroom"]))
                sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_t6_shortcut.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 84)
    print("T6：捷径噪声在别的数据集上复现吗（通过线 = 省 30%）")
    print("=" * 84)
    print("%-10s %-10s %-12s %-12s %-12s %-12s %s"
          % ("数据集", "噪声", "门控 ACC", "同花费朴素", "ΔACC", "省 oracle", "判定(ΔACC>=0.05)"))
    for name in DATASETS:
        for model in MODELS:
            rs = [r for r in res if r["ds"] == name and r["model"] == model]
            d = float(np.nanmean([r["delta_acc"] for r in rs]))
            s = float(np.nanmean([r["saved"] for r in rs]))
            print("%-10s %-10s %-12.3f %-12.3f %-12s %-12s %s"
                  % (name, model, np.mean([r["gate_acc"] for r in rs]),
                     np.nanmean([r["naive_at_gate"] for r in rs]),
                     ("%+.4f" % d) if d == d else "n/a",
                     ("%.0f%%" % (100 * s)) if s == s else "n/a",
                     "通过" if (d == d and d >= 0.05) else "不通过"))
