# -*- coding: utf-8 -*-
"""阶段 2：把 oracle 的账单摆到台面上。

扫描 A —— oracle 有多可靠？
  用 NoisyOracleValidator 以概率 q 把验证判断反着给，扫 q = 0 .. 1。
  理论预测（先写在纸上再跑）：门控收下的条目里脏的比例是
      P(dirty|accepted) = 0.4q / (0.6(1-q) + 0.4q)
  它等于 B 的污染率 0.4 时就是临界点 -> q = 0.5。
  q > 0.5 时三层门控应当**比完全不做门控更差**（因为它主动去挑脏的）。

扫描 B —— oracle 要花多少钱？
  固定完美 oracle，把"每任务交给验证器的带标注样本数"从 0 扫到 300。
  预算 0 -> 验证池为空 -> 门控全拒 -> C 退化成 A（无记忆）。这张账单必须明码标价。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.learners import (CleanReplayLearner, NaiveLearner, ReservoirLearner,
                          TieredLearner)
from exp.run import run_trajectory, summarize
from exp.taskstream import TaskStream
from mm import NeighborValidator, NoisyOracleValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DIM = 20
N_TASKS = 10
CAP = 600
STEPS = 300
DATA_NOISE = 0.4                       # 固定在最考验门控的那一档
SEEDS = [0, 1, 2]
ORACLE_NOISES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
BUDGETS = [0, 25, 50, 100, 150, 300]
POLLUTIONS = [0.2, 0.4, 0.6]
Q_NEAR = [0.4, 0.5, 0.6]


def stream_for(seed, n_holdout=300, data_noise=None):
    return TaskStream(n_tasks=N_TASKS, label_noise=DATA_NOISE if data_noise is None else data_noise,
                      center_scale=2.0, cluster_spread=1.8,
                      n_holdout=n_holdout, seed=1000 + seed)


def run_scan_a():
    recs = []
    for q in ORACLE_NOISES:
        for seed in SEEDS:
            st = stream_for(seed)
            # 参照：不做门控
            b = ReservoirLearner(DIM, seed=seed, capacity=CAP)
            acc, led = run_trajectory(b, st, steps_per_task=STEPS)
            recs.append(dict(scan="A", arm="B_replay", oracle_noise=q, seed=seed,
                             **summarize(acc, led)))
            # 主角：门控 + 打嗝的 oracle
            nv = NeighborValidator(np.zeros((0, DIM)), np.zeros(0))
            noisy = NoisyOracleValidator(nv, noise=q, seed=5000 + seed * 17 + int(q * 100))
            c = TieredLearner(DIM, seed=seed, capacity=CAP, validator=noisy, theta=0.7)
            acc, led = run_trajectory(c, st, steps_per_task=STEPS)
            recs.append(dict(scan="A", arm="C_tiered", oracle_noise=q, seed=seed,
                             **summarize(acc, led)))
        print("  扫描A q=%.1f 完成" % q)
        sys.stdout.flush()
    return recs


def run_scan_b():
    recs = []
    for seed in SEEDS:
        st = stream_for(seed)
        for arm, mk in (("B_replay", lambda: ReservoirLearner(DIM, seed=seed, capacity=CAP)),
                        ("E_clean_replay", lambda: CleanReplayLearner(DIM, seed=seed, capacity=CAP)),
                        ("A_naive", lambda: NaiveLearner(DIM, seed=seed))):
            acc, led = run_trajectory(mk(), st, steps_per_task=STEPS)
            recs.append(dict(scan="B", arm=arm, budget=-1, seed=seed, **summarize(acc, led)))
    for budget in BUDGETS:
        for seed in SEEDS:
            st = stream_for(seed)
            nv = NeighborValidator(np.zeros((0, DIM)), np.zeros(0))
            c = TieredLearner(DIM, seed=seed, capacity=CAP, validator=nv,
                              theta=0.7, holdout_budget=budget)
            acc, led = run_trajectory(c, st, steps_per_task=STEPS)
            s = summarize(acc, led)
            s["oracle_labels"] = int(led[-1].get("oracle_labels", 0))
            recs.append(dict(scan="B", arm="C_tiered", budget=budget, seed=seed, **s))
        print("  扫描B 预算=%d 完成" % budget)
        sys.stdout.flush()
    return recs


def run_scan_c():
    """临界点 q* 到底跟数据流的污染率 p 有没有关系？

    纸上推：门控收下的条目里脏的比例 = pq / ((1-p)(1-q) + pq)。
    让它等于不做门控时的污染率 p，解出来 q = 1/2 —— 与 p 无关。
    这个推论如果成立，就等于给出了一条"验证器合格线"：错误率低于 50% 就不亏。
    """
    recs = []
    for p in POLLUTIONS:
        for seed in SEEDS:
            st = stream_for(seed, data_noise=p)
            b = ReservoirLearner(DIM, seed=seed, capacity=CAP)
            acc, led = run_trajectory(b, st, steps_per_task=STEPS)
            recs.append(dict(scan="C", arm="B_replay", pollution=p, q=-1, seed=seed,
                             **summarize(acc, led)))
            for q in Q_NEAR:
                nv = NeighborValidator(np.zeros((0, DIM)), np.zeros(0))
                noisy = NoisyOracleValidator(nv, noise=q, seed=9000 + seed * 31 + int(q * 100))
                c = TieredLearner(DIM, seed=seed, capacity=CAP, validator=noisy, theta=0.7)
                acc, led = run_trajectory(c, st, steps_per_task=STEPS)
                recs.append(dict(scan="C", arm="C_tiered", pollution=p, q=q, seed=seed,
                                 **summarize(acc, led)))
        print("  扫描C p=%.1f 完成" % p)
        sys.stdout.flush()
    return recs


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print("阶段 2 开始（数据流标签噪声固定 %.0f%%，10 任务，3 seeds）" % (100 * DATA_NOISE))
    a = run_scan_a()
    b = run_scan_b()
    c = run_scan_c()
    with open(os.path.join(OUT, "results_stage2.json"), "w", encoding="utf-8") as f:
        json.dump(a + b + c, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 92)
    print("扫描 A：oracle 可靠度 vs 门控收益（q = 验证器给出反向判断的概率）")
    print("=" * 92)
    print("%-8s %-10s %-16s %-14s %-16s %-10s" % ("q", "arm", "ACC", "污染率", "接受率", "拒收/收下"))
    print("-" * 92)
    for q in ORACLE_NOISES:
        for arm in ("B_replay", "C_tiered"):
            rs = [r for r in a if r["oracle_noise"] == q and r["arm"] == arm]
            acc = np.mean([r["ACC_final"] for r in rs])
            pol = [r["pollution_final"] for r in rs if r["pollution_final"] == r["pollution_final"]]
            pols = "%.3f" % np.mean(pol) if pol else "  -  "
            rej = np.mean([r["rejected_total"] for r in rs])
            sto = np.mean([r["store_final"] for r in rs])
            print("%-8.2f %-10s %.3f±%.3f     %-14s %-16s %.0f/%.0f"
                  % (q, arm, acc, np.std([r["ACC_final"] for r in rs]), pols, "-", rej, sto))
        rr = [r for r in a if r["oracle_noise"] == q and r["arm"] == "C_tiered"]
        bb = [r for r in a if r["oracle_noise"] == q and r["arm"] == "B_replay"]
        adv = np.mean([r["ACC_final"] for r in rr]) - np.mean([r["ACC_final"] for r in bb])
        print("%-8s C - B = %+.3f %s" % ("", adv, "  <-- 门控开始亏本" if adv < -0.02 else ""))

    print()
    print("=" * 92)
    print("扫描 B：oracle 预算（每任务交给验证器的干净标注条数）")
    print("=" * 92)
    print("%-12s %-16s %-16s %-14s" % ("预算/任务", "累计标注", "ACC", "拒收条数"))
    print("-" * 92)
    for budget in [-1] + BUDGETS:
        rs = [r for r in b if r["budget"] == budget and r["arm"] == "C_tiered"]
        if rs:
            acc = np.mean([r["ACC_final"] for r in rs])
            std = np.std([r["ACC_final"] for r in rs])
            lab = np.mean([r.get("oracle_labels", 0) for r in rs])
            rej = np.mean([r["rejected_total"] for r in rs])
            print("%-12s %-16s %.3f±%.3f      %.0f"
                  % (str(budget) + (" (全给)" if budget == 150 else ""), "%.0f" % lab, acc, std, rej))
    for arm in ("A_naive", "B_replay", "E_clean_replay"):
        rs = [r for r in b if r["arm"] == arm]
        acc = np.mean([r["ACC_final"] for r in rs])
        print("%-12s %-16s %.3f±%.3f      %s" % ("参照 " + arm, "0", acc,
              np.std([r["ACC_final"] for r in rs]), ""))
    print()
    print("=" * 92)
    print("扫描 C：临界点 q* 跟数据流污染率 p 有没有关系？（纸上推：q* = 0.5，与 p 无关）")
    print("=" * 92)
    print("%-10s %-10s %-16s %-14s %-16s %s" % ("污染率 p", "oracle q", "C 的 ACC", "C-B", "C 的污染率", "判定"))
    print("-" * 92)
    for p in POLLUTIONS:
        bacc = np.mean([r["ACC_final"] for r in c if r["pollution"] == p and r["arm"] == "B_replay"])
        bpol = np.mean([r["pollution_final"] for r in c if r["pollution"] == p and r["arm"] == "B_replay"])
        for q in Q_NEAR:
            rs = [r for r in c if r["pollution"] == p and r["arm"] == "C_tiered" and abs(r["q"] - q) < 1e-9]
            acc = np.mean([r["ACC_final"] for r in rs])
            pol = np.mean([r["pollution_final"] for r in rs])
            verdict = "打平" if abs(acc - bacc) < 0.02 else ("赚" if acc > bacc else "亏")
            print("%-10s %-10s %.3f±%.3f      %+.3f         %.3f            %s"
                  % ("%.1f" % p, "%.1f" % q, acc, np.std([r["ACC_final"] for r in rs]),
                     acc - bacc, pol, verdict))
        print("%-10s %-10s %.3f                %s   (B 的污染率 %.3f)"
              % ("%.1f" % p, "B 参照", bacc, "-", bpol))
    print()
    print("总耗时 %.1f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage2.json")))


if __name__ == "__main__":
    main()
