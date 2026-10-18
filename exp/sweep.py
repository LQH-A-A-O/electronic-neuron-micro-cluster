# -*- coding: utf-8 -*-
"""阶段 1 主实验：4 组学习器 × 标签噪声档 × 多随机种子。

A_naive        无记忆（遗忘下界）
B_replay       无门控回放（照单全收）
C_tiered_good  三层信任记忆 + 正确验证器（本次要检验的主角）
D_tiered_demo  三层信任记忆 + 原演示那套"只看 value"的验证器（反例）
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.learners import (NaiveLearner, ReservoirLearner, TieredLearner,
                          CleanReplayLearner)
from exp.run import run_trajectory, summarize
from exp.taskstream import TaskStream
from mm import NeighborValidator, ValueOnlyValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DIM = 20
N_TASKS = 10
N_CLASSES = N_TASKS * 5
CAP = 600
STEPS = 300


def build_arm(arm, seed):
    if arm == "A_naive":
        return NaiveLearner(DIM, seed=seed)
    if arm == "B_replay":
        return ReservoirLearner(DIM, seed=seed, capacity=CAP)
    if arm == "C_tiered_good":
        return TieredLearner(DIM, seed=seed, capacity=CAP,
                             validator=NeighborValidator(np.zeros((0, DIM)), np.zeros(0)),
                             theta=0.7)
    if arm == "E_clean_replay":
        return CleanReplayLearner(DIM, seed=seed, capacity=CAP)
    if arm == "D_tiered_demo":
        # 复刻原演示：只问"value 是不是已知答案之一"
        return TieredLearner(DIM, seed=seed, capacity=CAP,
                             validator=ValueOnlyValidator(list(range(N_CLASSES))),
                             theta=0.7)
    raise KeyError(arm)


ARMS = ["A_naive", "B_replay", "C_tiered_good", "D_tiered_demo", "E_clean_replay"]
NOISES = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 1, 2]


def main():
    os.makedirs(OUT, exist_ok=True)
    records = []
    t0 = time.time()
    for noise in NOISES:
        for seed in SEEDS:
            stream = TaskStream(n_tasks=N_TASKS, label_noise=noise,
                                center_scale=2.0, cluster_spread=1.8,
                                seed=1000 + seed)
            for arm in ARMS:
                learner = build_arm(arm, seed)
                acc, ledger = run_trajectory(learner, stream, steps_per_task=STEPS)
                s = summarize(acc, ledger)
                records.append(dict(arm=arm, noise=noise, seed=seed, **s,
                                    ledger=ledger))
        print("noise=%.1f 完成  累计 %.1f 秒" % (noise, time.time() - t0))
        sys.stdout.flush()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 88)
    print("阶段 1 结果（10 任务 / 50 类 / 3 seeds，均值±标准差）")
    print("=" * 88)
    hdr = "%-14s" % "噪声率"
    for a in ARMS:
        hdr += " | %-19s" % a
    print(hdr)
    print("-" * 88)
    for noise in NOISES:
        line = "%-14s" % ("%.0f%%" % (100 * noise))
        for a in ARMS:
            rs = [r for r in records if r["noise"] == noise and r["arm"] == a]
            accs = np.array([r["ACC_final"] for r in rs])
            line += " | %.3f±%.3f        " % (accs.mean(), accs.std())
        print(line)

    print()
    print("=" * 88)
    print("同一批跑的：遗忘量 BWT / 记忆污染率 / 门控拒收量")
    print("=" * 88)
    print("%-8s %-14s %-18s %-18s %-14s" % ("噪声", "arm", "BWT", "污染率", "拒收条数"))
    print("-" * 88)
    for noise in NOISES:
        for a in ARMS:
            rs = [r for r in records if r["noise"] == noise and r["arm"] == a]
            bwt = np.mean([r["BWT"] for r in rs])
            pol = [r["pollution_final"] for r in rs if r["pollution_final"] == r["pollution_final"]]
            pols = "%.3f" % np.mean(pol) if pol else "  -  "
            rej = np.mean([r["rejected_total"] for r in rs])
            print("%-8s %-14s %+.3f              %-18s %.0f" % ("%.0f%%" % (100 * noise), a, bwt, pols, rej))
    print()
    print("总耗时 %.1f 秒，原始记录 %s" % (time.time() - t0, os.path.join(OUT, "results.json")))


if __name__ == "__main__":
    main()
