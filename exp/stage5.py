# -*- coding: utf-8 -*-
"""阶段 5：把结论往"能落地"推两步。

5A  结构化幻觉 —— 均匀随机翻转是假的威胁模型。换成真实模型的混淆方向后，门控还挡得住吗？
5B  免费 oracle —— 现实里没人同步给你标好 holdout。三种"不要钱"的验证信号能不能顶上？
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.learners import NaiveLearner, ReservoirLearner, TieredLearner
from exp.realdata import RealClassStream, load_mnist
from exp.run import run_trajectory, summarize
from mm import ConsensusValidator, DelayedLabelValidator, NeighborValidator, ValueOnlyValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
CAP, STEPS, P = 600, 400, 0.4
SEEDS = [0, 1, 2]
MN = load_mnist()


def make_stream(seed):
    return RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=5, classes_per_task=2,
                           n_train=400, n_holdout=200, n_test=200,
                           label_noise=0.0, seed=1000 + seed)


def poison(st, mode, p, conf, seed):
    rng = np.random.default_rng(777 + seed)
    for task in st:
        task.y_train = inject(task.y_train_true, task.classes, p, rng, mode, conf)
    return st


def new_validator(kind, dim, conf):
    if kind == "gold":
        return NeighborValidator(np.zeros((0, dim)), np.zeros(0))
    if kind == "delay1":
        return DelayedLabelValidator(NeighborValidator(np.zeros((0, dim)), np.zeros(0)), lag=1)
    if kind == "delay2":
        return DelayedLabelValidator(NeighborValidator(np.zeros((0, dim)), np.zeros(0)), lag=2)
    if kind == "consensus":
        return ConsensusValidator(k=8, bootstrap_tasks=1)
    if kind == "value_only":
        return ValueOnlyValidator(list(range(10)))
    raise KeyError(kind)


def run(arm, kind, st, seed):
    dim = st.dim
    if arm == "A_naive":
        L = NaiveLearner(dim, seed=seed)
    elif arm == "B_replay":
        L = ReservoirLearner(dim, seed=seed, capacity=CAP)
    else:
        L = TieredLearner(dim, seed=seed, capacity=CAP, theta=0.7,
                          validator=new_validator(kind, dim, None))
    acc, led = run_trajectory(L, st, steps_per_task=STEPS)
    s = summarize(acc, led)
    s["false_accept"] = led[-1].get("false_accept", float("nan"))
    s["false_reject"] = led[-1].get("false_reject", float("nan"))
    return s


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    conf, wacc = weak_model_confusion()
    print("弱模型准确率 = %.4f" % wacc)
    off = conf.copy(); np.fill_diagonal(off, 0)
    print("它最常混淆的对（真->错，前 8 个）：")
    pairs = [(off[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    for v, i, j in sorted(pairs, reverse=True)[:8]:
        print("   %d -> %d   %.3f" % (i, j, v))
    sys.stdout.flush()

    recs = []
    print("\n5A：均匀幻觉 vs 结构化幻觉（错误率都固定 40%）")
    for mode in ("uniform", "structured"):
        for seed in SEEDS:
            for arm in ("A_naive", "B_replay", "C_tiered_good", "D_tiered_demo"):
                st = poison(make_stream(seed), mode, P, conf, seed)
                kind = "value_only" if arm == "D_tiered_demo" else "gold"
                recs.append(dict(scan="A", mode=mode, arm=arm, seed=seed, kind=kind,
                                 **run(arm, kind, st, seed)))
        print("  mode=%s 完成 %.0f 秒" % (mode, time.time() - t0)); sys.stdout.flush()

    print("\n5B：免费 oracle（结构化幻觉 40%）")
    for kind in ("gold", "delay1", "delay2", "consensus"):
        for seed in SEEDS:
            st = poison(make_stream(seed), "structured", P, conf, seed)
            recs.append(dict(scan="B", mode="structured", arm="C_tiered_good", kind=kind,
                             seed=seed, **run("C_tiered_good", kind, st, seed)))
        print("  oracle=%s 完成 %.0f 秒" % (kind, time.time() - t0)); sys.stdout.flush()
    for arm in ("A_naive", "B_replay"):
        for seed in SEEDS:
            st = poison(make_stream(seed), "structured", P, conf, seed)
            recs.append(dict(scan="B", mode="structured", arm=arm, kind="-", seed=seed,
                             **run(arm, "gold", st, seed)))

    json.dump(recs, open(os.path.join(OUT, "results_stage5.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print("=" * 100)
    print("阶段 5A：结构化幻觉 vs 均匀幻觉（错误率均 40%%，MNIST，3 seeds）")
    print("=" * 100)
    print("%-14s %-16s %-16s %-16s %s" % ("幻觉类型", "A 无记忆", "B 无门控回放", "C 门控", "D 原演示验证器"))
    for mode in ("uniform", "structured"):
        row = "%-14s" % mode
        for arm in ("A_naive", "B_replay", "C_tiered_good", "D_tiered_demo"):
            v = [r["ACC_final"] for r in recs if r["scan"] == "A" and r["mode"] == mode and r["arm"] == arm]
            row += " %-16s" % ("%.3f±%.3f" % (np.mean(v), np.std(v)))
        print(row)
    print()
    print("门控自己的混淆矩阵（越靠左越好：漏放率 = 脏的进了温备；误拒率 = 干净的没进去）")
    print("-" * 100)
    print("%-14s %-14s %-14s %-14s %s" % ("幻觉类型", "漏放率", "误拒率", "污染率", "C 的 ACC"))
    for mode in ("uniform", "structured"):
        rs = [r for r in recs if r["scan"] == "A" and r["mode"] == mode and r["arm"] == "C_tiered_good"]
        print("%-14s %-14s %-14s %-14s %.3f"
              % (mode, "%.3f" % np.mean([r["false_accept"] for r in rs]),
                 "%.3f" % np.mean([r["false_reject"] for r in rs]),
                 "%.3f" % np.mean([r["pollution_final"] for r in rs]),
                 np.mean([r["ACC_final"] for r in rs])))

    print()
    print("=" * 100)
    print("阶段 5B：不要钱的验证信号（结构化幻觉 40%）")
    print("=" * 100)
    print("%-14s %-16s %-14s %-14s %s" % ("oracle", "ACC", "漏放率", "误拒率", "污染率"))
    for kind in ("gold", "delay1", "delay2", "consensus"):
        rs = [r for r in recs if r["scan"] == "B" and r["arm"] == "C_tiered_good" and r["kind"] == kind]
        print("%-14s %.3f±%.3f      %-14s %-14s %.3f"
              % (kind, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 "%.3f" % np.mean([r["false_accept"] for r in rs]),
                 "%.3f" % np.mean([r["false_reject"] for r in rs]),
                 np.mean([r["pollution_final"] for r in rs])))
    for arm in ("A_naive", "B_replay"):
        rs = [r for r in recs if r["scan"] == "B" and r["arm"] == arm]
        print("%-14s %.3f±%.3f   (参照)" % (arm, np.mean([r["ACC_final"] for r in rs]),
                                            np.std([r["ACC_final"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage5.json")))


if __name__ == "__main__":
    main()
