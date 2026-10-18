# -*- coding: utf-8 -*-
"""P1：**D1 也用稳指标重跑一遍** —— 如果它也缩水，整个判定都要重写。

为什么要重跑
------------------------------------------------
T6/T7 发现"省 oracle 的比例"这个指标**极不稳**（同一配置能给出 40% / 7% / -83% 三种结论），
而 D1 的"省 84%"**正是用这个指标算出来的**。

这一轮用 ΔACC（同花费下的准确率差）重测 D1，种子数从 5 提到 10。
判据沿用 D1 原文：**省 >= 30% 或 ΔACC >= 0.05**，但两个都报，看它们是否一致。
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.stage20 import R20
from exp.hallucination import inject, weak_model_confusion
from exp.realdata import RealClassStream, load_mnist
from exp.stage20 import featurize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
ROUND, N_TASK, CPT = 50, 5, 2
N_TRAIN, N_HOLD, N_TEST = 400, 200, 200
BUDGETS = [0, 5, 10, 20, 50]
SEEDS = list(range(10))
MN = load_mnist()
CONF, _ = weak_model_confusion()


def make_stream(seed):
    st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=N_TASK, classes_per_task=CPT,
                         n_train=N_TRAIN, n_holdout=N_HOLD, n_test=N_TEST,
                         label_noise=0.0, seed=1000 + seed)
    rng = np.random.default_rng(777 + seed)
    for task in st:
        task.y_train = inject(task.y_train_true, task.classes, 0.4, rng, "structured", CONF)
        task.x_train = featurize(task.x_train)
        task.x_holdout = featurize(task.x_holdout)
        task.x_test = featurize(task.x_test)
    return st


def run_arm(mode, B, st, seed):
    R = R20(mode, seed, B)
    if mode == "gate":
        R.mode = "gate"
    accs = []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    return float(accs[-1]), R.side


if __name__ == "__main__":
    res = []
    for seed in SEEDS:
        st = make_stream(seed)
        row = dict(seed=seed)
        row["gate"] = run_arm("gate", -1, st, seed)
        for b in BUDGETS:
            row["n%d" % b] = run_arm("naive", b, st, seed)
        res.append(row)
        print("seed=%d gate %.3f@%d" % (seed, row["gate"][0], row["gate"][1])); sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_p1_d1delta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def m(k, i):
        return float(np.mean([r[k][i] for r in res]))

    ga, gs = m("gate", 0), m("gate", 1)
    xs = [(m("n%d" % b, 1), m("n%d" % b, 0)) for b in BUDGETS]
    nat = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if x0 <= gs <= x1 and x1 > x0:
            nat = y0 + (gs - x0) / (x1 - x0) * (y1 - y0); break
    need = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if y0 <= ga <= y1 and y1 > y0:
            need = x0 + (ga - y0) / (y1 - y0) * (x1 - x0); break
    saved = (need - gs) / need if need == need and need > 0 else float("nan")
    print()
    print("=" * 80)
    print("P1：D1 用 ΔACC 重测（真实 MNIST / 污染 40% / 10 seeds）")
    print("=" * 80)
    print("门控 %.3f @ %d 次；同花费朴素 %.3f；ΔACC %+.4f" % (ga, gs, nat, ga - nat))
    print("朴素要 %s 次才追上 -> 省 %s" % (("%.0f" % need) if need == need else "n/a",
                                    ("%.0f%%" % (100 * saved)) if saved == saved else "n/a"))
    print("朴素曲线: " + "  ".join("B=%d %d次 %.3f" % (b, xs[i][0], xs[i][1]) for i, b in enumerate(BUDGETS)))
    print()
    print("D1 原文判据（省 >=30%% 或 ΔACC >=0.05）：省 %s / ΔACC %s -> %s"
          % (("通过" if saved >= 0.30 else "不通过"), ("通过" if ga - nat >= 0.05 else "不通过"),
             "通过" if (saved >= 0.30 or ga - nat >= 0.05) else "不通过"))
