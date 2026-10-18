# -*- coding: utf-8 -*-
"""阶段 4：把阶段 1/2 的结论搬到真实数据（MNIST）上复验。

模型、优化器、门控、对照组全部沿用阶段 1 的同一份实现 —— 只换数据。
  A_naive         无记忆
  B_replay        无门控回放
  C_tiered_good   三层信任记忆 + 看 key 的验证器
  D_tiered_demo   三层信任记忆 + 原演示那套只看 value 的验证器
  E_clean_replay  只回放干净留出集（上界，不可部署）
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.learners import (CleanReplayLearner, NaiveLearner, ReservoirLearner, TieredLearner)
from exp.realdata import RealClassStream, load_mnist
from exp.run import run_trajectory, summarize
from mm import NeighborValidator, ValueOnlyValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
CAP = 600
STEPS = 400
N_CLASSES = 10
SEEDS = [0, 1, 2]
NOISES = [0.0, 0.2, 0.4]
BUDGETS = [0, 20, 50, 100, 200]
ARMS = ["A_naive", "B_replay", "C_tiered_good", "D_tiered_demo", "E_clean_replay"]

MN = load_mnist()


def stream(seed, noise, holdout=200):
    return RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=5, classes_per_task=2,
                           n_train=400, n_holdout=holdout, n_test=200,
                           label_noise=noise, seed=1000 + seed)


def build(arm, dim, seed, holdout_budget=None):
    if arm == "A_naive":
        return NaiveLearner(dim, seed=seed)
    if arm == "B_replay":
        return ReservoirLearner(dim, seed=seed, capacity=CAP)
    if arm == "E_clean_replay":
        return CleanReplayLearner(dim, seed=seed, capacity=CAP)
    if arm == "C_tiered_good":
        return TieredLearner(dim, seed=seed, capacity=CAP,
                             validator=NeighborValidator(np.zeros((0, dim)), np.zeros(0)),
                             theta=0.7, holdout_budget=holdout_budget)
    if arm == "D_tiered_demo":
        return TieredLearner(dim, seed=seed, capacity=CAP,
                             validator=ValueOnlyValidator(list(range(N_CLASSES))),
                             theta=0.7, holdout_budget=holdout_budget)
    raise KeyError(arm)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    print("4A：真实 MNIST，5 任务 x 2 类，400 步/任务，3 seeds")
    for noise in NOISES:
        for seed in SEEDS:
            st = stream(seed, noise)
            for arm in ARMS:
                L = build(arm, st.dim, seed)
                acc, led = run_trajectory(L, st, steps_per_task=STEPS)
                recs.append(dict(scan="A", arm=arm, noise=noise, seed=seed, **summarize(acc, led)))
        print("  noise=%.1f 完成  %.0f 秒" % (noise, time.time() - t0)); sys.stdout.flush()

    print("4B：真实 MNIST，oracle 预算扫描（noise=0.4）")
    for budget in BUDGETS:
        for seed in SEEDS:
            st = stream(seed, 0.4)
            L = build("C_tiered_good", st.dim, seed, holdout_budget=budget)
            acc, led = run_trajectory(L, st, steps_per_task=STEPS)
            s = summarize(acc, led)
            s["oracle_labels"] = int(led[-1].get("oracle_labels", 0))
            recs.append(dict(scan="B", arm="C_tiered_good", budget=budget, noise=0.4,
                             seed=seed, **s))
        print("  budget=%d 完成  %.0f 秒" % (budget, time.time() - t0)); sys.stdout.flush()
    for arm in ("A_naive", "B_replay"):
        for seed in SEEDS:
            st = stream(seed, 0.4)
            L = build(arm, st.dim, seed)
            acc, led = run_trajectory(L, st, steps_per_task=STEPS)
            recs.append(dict(scan="B", arm=arm, budget=-1, noise=0.4, seed=seed,
                             **summarize(acc, led)))

    json.dump(recs, open(os.path.join(OUT, "results_stage4.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print("=" * 96)
    print("阶段 4A：真实 MNIST 上的复验（准率 / 对比）")
    print("=" * 96)
    print("%-8s" % "噪声", end="")
    for a in ARMS:
        print(" | %-15s" % a, end="")
    print()
    print("-" * 96)
    for noise in NOISES:
        print("%-8s" % ("%.0f%%" % (100 * noise)), end="")
        for a in ARMS:
            v = [r["ACC_final"] for r in recs if r["scan"] == "A" and r["noise"] == noise and r["arm"] == a]
            print(" | %.3f±%.3f    " % (np.mean(v), np.std(v)), end="")
        print()
    print()
    print("%-8s %-16s %-12s %-12s %-12s" % ("噪声", "arm(40%)", "BWT", "污染率", "拒收"))
    for noise in NOISES:
        for a in ARMS:
            rs = [r for r in recs if r["scan"] == "A" and r["noise"] == noise and r["arm"] == a]
            pol = [r["pollution_final"] for r in rs if r["pollution_final"] == r["pollution_final"]]
            print("%-8s %-16s %+.3f        %-12s %.0f"
                  % ("%.0f%%" % (100 * noise), a, np.mean([r["BWT"] for r in rs]),
                     ("%.3f" % np.mean(pol)) if pol else "  -  ",
                     np.mean([r["rejected_total"] for r in rs])))

    print()
    print("=" * 96)
    print("阶段 4B：真实 MNIST 上 oracle 预算（noise=40%）")
    print("=" * 96)
    print("%-14s %-14s %-16s" % ("每任务标注", "累计标注", "ACC"))
    for budget in BUDGETS:
        rs = [r for r in recs if r["scan"] == "B" and r["arm"] == "C_tiered_good" and r["budget"] == budget]
        print("%-14d %-14.0f %.3f±%.3f" % (budget, np.mean([r["oracle_labels"] for r in rs]),
                                           np.mean([r["ACC_final"] for r in rs]),
                                           np.std([r["ACC_final"] for r in rs])))
    for arm in ("A_naive", "B_replay"):
        rs = [r for r in recs if r["scan"] == "B" and r["arm"] == arm]
        print("%-14s %-14s %.3f±%.3f" % ("参照 " + arm, "0",
              np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage4.json")))


if __name__ == "__main__":
    main()
