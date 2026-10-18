# -*- coding: utf-8 -*-
"""阶段 3B：冷备什么时候才值钱？—— 温备容量 × 冷备更新规则的交互。

3A 的结论是负的：在温备容量 600 的前提下，加冷备（无论冻结还是 EMA，
甚至给它当前世界的真实原型）都没有正收益。那冷备是不是纯粹冗余？

一个合理的猜想：冷备的价值不在"信任层级"，而在"紧凑的类别覆盖" ——
25 个类只需要 25 条原型就能全覆盖，而温备要装下同样覆盖面得花几百条。
所以当**温备装不下**时，冷备才该显出价值。

于是把温备容量从 600 压到 50，看冷备的曲线会不会翻正。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.learners import TieredLearner
from exp.recurring import RecurringDriftStream
from exp.run import train_one_task
from mm import NeighborValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DIM = 20
STEPS = 300
DRIFT = 0.6
SEEDS = [0, 1, 2, 3, 4]
CAPS = [50, 150, 600]
ARMS = [
    ("N_no_cold", dict(cold_mode="none"), False),
    ("F_frozen", dict(cold_mode="frozen"), False),
    ("S_ema20", dict(cold_mode="ema", cold_alpha=0.20), False),
    ("T_true", dict(cold_mode="frozen"), True),
]


def run_one(kw, is_true, stream, seed, cap):
    learner = TieredLearner(DIM, seed=seed, capacity=cap,
                            validator=NeighborValidator(np.zeros((0, DIM)), np.zeros(0)),
                            theta=0.7, **kw)
    for t, task in enumerate(stream):
        train_one_task(learner, task, t, steps_per_task=STEPS)
        if is_true:
            for c in task.classes:
                learner.cold_centroids[int(c)] = stream.centers_at[t][int(c)].copy()
    pred = learner.predict(stream.xt)
    return dict(ACC_final=float((pred == stream.yt).mean()),
                store=len(learner.mem.warm), cold=len(learner.cold_centroids))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for cap in CAPS:
        for seed in SEEDS:
            st = RecurringDriftStream(n_classes=25, repeats=4, classes_per_task=10,
                                      n_train=500, n_holdout=200, n_test=100,
                                      drift=DRIFT, label_noise=0.3, seed=1000 + seed)
            for name, kw, is_true in ARMS:
                r = run_one(kw, is_true, st, seed, cap)
                recs.append(dict(arm=name, cap=cap, seed=seed, **r))
        print("cap=%d 完成  %.1f 秒" % (cap, time.time() - t0))
        sys.stdout.flush()
    with open(os.path.join(OUT, "results_stage3b.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 92)
    print("阶段 3B：温备容量 x 冷备（drift=0.6，数据噪声 30%，5 seeds）")
    print("=" * 92)
    print("%-12s" % "温备容量", end="")
    for name, _, _ in ARMS:
        print(" | %-14s" % name, end="")
    print(" | 相对 N 的最大增益")
    print("-" * 92)
    for cap in CAPS:
        base = np.mean([r["ACC_final"] for r in recs if r["cap"] == cap and r["arm"] == "N_no_cold"])
        best = -9
        row = ""
        for name, _, _ in ARMS:
            v = np.mean([r["ACC_final"] for r in recs if r["cap"] == cap and r["arm"] == name])
            row += " | %.3f±%.3f  " % (v, np.std([r["ACC_final"] for r in recs if r["cap"] == cap and r["arm"] == name]))
            if name != "N_no_cold":
                best = max(best, v - base)
        print("%-12s%s | %+.3f" % ("%d 条" % cap, row, best))
    print()
    print("温备实际装了多少条（reservoir 上限就是容量）")
    for cap in CAPS:
        print("  cap=%-4d -> 温备 %.0f 条" % (cap, np.mean([r["store"] for r in recs if r["cap"] == cap])))
    print()
    print("总耗时 %.1f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage3b.json")))


if __name__ == "__main__":
    main()
