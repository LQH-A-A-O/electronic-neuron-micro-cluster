# -*- coding: utf-8 -*-
"""阶段 3：冷备到底该不该更新？

背景：阶段 1/2 里冷备一直是空的 —— 因为 class-incremental 每个类只出现一次，
"更新规则"根本触发不了。这一阶段换成「类重复出现 + 世界漂移」的流，才谈得上这个问题。

六组，全部共用同一套「温备 + 验证门控」，唯一变量是冷备：

  N_no_cold   冷备空着（阶段 1/2 基线）
  F_frozen    类第一次被验证通过时的原型，之后**一个字都不改**（构想里的字面设计）
  S_ema05     同一类的原型按 alpha=0.05 缓慢吸收新观测
  S_ema20     alpha=0.20
  T_true      **作弊上界**：冷备永远精确等于当前真实原型（模型不可能做到，用来回答
              "冷备这个东西本身到底有没有用；如果有用，我们的维护方式差多远"）
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
CAP = 600
STEPS = 300
DATA_NOISE = 0.3

ARMS = [
    ("N_no_cold", dict(cold_mode="none")),
    ("F_frozen", dict(cold_mode="frozen")),
    ("S_ema05", dict(cold_mode="ema", cold_alpha=0.05)),
    ("S_ema20", dict(cold_mode="ema", cold_alpha=0.20)),
    ("T_true", dict(cold_mode="frozen")),          # 冷备由外部每任务重写成真实原型
]
TRUE_ARMS = {"T_true"}
DRIFTS = [0.0, 0.3, 0.6]
SEEDS = [0, 1, 2, 3, 4]


def safe_nanmean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def run_one(arm_kw, arm_name, stream, seed):
    learner = TieredLearner(DIM, seed=seed, capacity=CAP,
                            validator=NeighborValidator(np.zeros((0, DIM)), np.zeros(0)),
                            theta=0.7, **arm_kw)
    for t, task in enumerate(stream):
        train_one_task(learner, task, t, steps_per_task=STEPS)
        if arm_name in TRUE_ARMS:
            # 作弊：直接把"当前世界的真实原型"塞进冷备
            for c in task.classes:
                learner.cold_centroids[int(c)] = stream.centers_at[t][int(c)].copy()
    pred = learner.predict(stream.xt)
    rep = learner.memory_report()
    cold = learner.cold_centroids
    if cold:
        stale = safe_nanmean([np.linalg.norm(cold[c] - stream.centers_final[c]) for c in cold])
        init_d = safe_nanmean([np.linalg.norm(stream.centers_start[c] - stream.centers_final[c]) for c in cold])
    else:
        stale, init_d = float("nan"), float("nan")
    return dict(
        ACC_final=float((pred == stream.yt).mean()),
        buckets=stream.age_bucket_accuracy(pred),
        cold_classes=len(cold), cold_updates=rep["cold_updates"],
        cold_staleness=stale, center_drift=init_d,
        store=rep["store"], rejected=rep["rejected_total"],
    )


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for drift in DRIFTS:
        for seed in SEEDS:
            st = RecurringDriftStream(n_classes=25, repeats=4, classes_per_task=10,
                                      n_train=500, n_holdout=200, n_test=100,
                                      drift=drift, label_noise=DATA_NOISE, seed=1000 + seed)
            for name, kw in ARMS:
                r = run_one(kw, name, st, seed)
                recs.append(dict(arm=name, drift=drift, seed=seed, **r))
        print("drift=%.1f 完成  %.1f 秒" % (drift, time.time() - t0))
        sys.stdout.flush()
    with open(os.path.join(OUT, "results_stage3.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 100)
    print("阶段 3：冷备更新规则（10 任务 / 25 类 / 每类出现 4 次 / 数据噪声 30% / 5 seeds）")
    print("=" * 100)
    print("%-12s" % "漂移/冷备", end="")
    for name, _ in ARMS:
        print(" | %-13s" % name, end="")
    print()
    print("-" * 100)
    for drift in DRIFTS:
        print("%-12s" % ("drift=%.1f" % drift), end="")
        for name, _ in ARMS:
            v = [r["ACC_final"] for r in recs if r["drift"] == drift and r["arm"] == name]
            print(" | %.3f±%.3f  " % (np.mean(v), np.std(v)), end="")
        print()

    print()
    print("冷备原型与当前真实原型的距离（越小越跟得上世界）")
    print("-" * 100)
    print("%-12s %-14s %-14s %-14s %s" % ("冷备", "drift=0", "drift=0.3", "drift=0.6", "覆盖类数(25)"))
    for name, _ in ARMS:
        row = []
        for drift in DRIFTS:
            rs = [r for r in recs if r["drift"] == drift and r["arm"] == name]
            v = safe_nanmean([r["cold_staleness"] for r in rs])
            row.append("%.2f" % v if v == v else "  -  ")
        cc = np.mean([r["cold_classes"] for r in recs if r["arm"] == name])
        print("%-12s %-14s %-14s %-14s %.0f" % (name, row[0], row[1], row[2], cc))

    print()
    print("按「类最后一次出现距今多久」分桶（drift=0.6）")
    print("-" * 100)
    print("%-12s %-16s %-16s %-16s" % ("冷备", "新鲜(0-1)", "中等(2-4)", "陈旧(5-9)"))
    for name, _ in ARMS:
        rs = [r for r in recs if r["drift"] == 0.6 and r["arm"] == name]
        print("%-12s %-16s %-16s %-16s" % (name,
              "%.3f" % safe_nanmean([r["buckets"]["新鲜(0-1)"] for r in rs]),
              "%.3f" % safe_nanmean([r["buckets"]["中等(2-4)"] for r in rs]),
              "%.3f" % safe_nanmean([r["buckets"]["陈旧(5-9)"] for r in rs])))
    print()
    for drift in DRIFTS:
        rs = [r for r in recs if r["drift"] == drift]
        print("  drift=%.1f：真实原型从最初走到最后，平均位移 = %.2f"
              % (drift, safe_nanmean([r["center_drift"] for r in rs])))
    print()
    print("总耗时 %.1f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage3.json")))


if __name__ == "__main__":
    main()
