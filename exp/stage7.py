# -*- coding: utf-8 -*-
"""阶段 7：事后回滚式门控 —— 让"延迟标签"这条路走得通。

阶段 5 的发现：delay1/delay2 的准确率精确等于"完全没有记忆"（0.125），
根因是验证池只覆盖旧类，新类永远验不了 -> 全拒 -> 温备空。

修法不是"更宽容的验证器"，而是**把裁判推迟**：
  先收下（provisional），等标签到了再重新裁判，没过的撤回。
这正是构想里"冷备用于校正"那条路径 —— 第一次真正实现。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.learners import NaiveLearner, ReservoirLearner, TieredLearner
from exp.realdata import RealClassStream, load_mnist
from exp.run import run_trajectory, summarize
from mm import DelayedLabelValidator, NeighborValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
CAP, STEPS, P = 600, 400, 0.4
SEEDS = [0, 1, 2]
MN = load_mnist()
CONF, WACC = weak_model_confusion()


def stream(seed):
    st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=5, classes_per_task=2,
                         n_train=400, n_holdout=200, n_test=200,
                         label_noise=0.0, seed=1000 + seed)
    rng = np.random.default_rng(777 + seed)
    for task in st:
        task.y_train = inject(task.y_train_true, task.classes, P, rng, "structured", CONF)
    return st


ARMS = [
    ("gold", 0, False),
    ("delay1(坏)", 1, False),
    ("delay1+回滚", 1, True),
    ("delay2+回滚", 2, True),
    ("delay3+回滚", 3, True),
]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        for name, lag, rb in ARMS:
            st = stream(seed)
            v = DelayedLabelValidator(NeighborValidator(np.zeros((0, st.dim)), np.zeros(0)), lag=lag)
            L = TieredLearner(st.dim, seed=seed, capacity=CAP, validator=v, theta=0.7,
                              admit_unverifiable=rb)
            acc, led = run_trajectory(L, st, steps_per_task=STEPS)
            s = summarize(acc, led)
            s["rollback_evicted"] = led[-1].get("rollback_evicted", 0)
            recs.append(dict(arm=name, lag=lag, rb=rb, seed=seed, **s))
        for arm in ("A_naive", "B_replay"):
            st = stream(seed)
            L = NaiveLearner(st.dim, seed=seed) if arm == "A_naive" else \
                ReservoirLearner(st.dim, seed=seed, capacity=CAP)
            acc, led = run_trajectory(L, st, steps_per_task=STEPS)
            recs.append(dict(arm=arm, lag=-1, rb=False, seed=seed, **summarize(acc, led)))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage7.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print("=" * 96)
    print("阶段 7：事后回滚式门控（真实 MNIST / 结构化幻觉 40% / 3 seeds）")
    print("=" * 96)
    print("%-16s %-18s %-14s %-14s %s" % ("oracle", "ACC", "温备污染率", "拒收", "回滚撤回"))
    for name, lag, rb in ARMS:
        rs = [r for r in recs if r["arm"] == name]
        pol = [r["pollution_final"] for r in rs if r["pollution_final"] == r["pollution_final"]]
        print("%-16s %.3f±%.3f        %-14s %-14.0f %.0f"
              % (name, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 ("%.3f" % np.mean(pol)) if pol else "  -  ",
                 np.mean([r["rejected_total"] for r in rs]),
                 np.mean([r.get("rollback_evicted", 0) for r in rs])))
    for arm in ("A_naive", "B_replay"):
        rs = [r for r in recs if r["arm"] == arm]
        print("%-16s %.3f±%.3f   (参照)" % (arm, np.mean([r["ACC_final"] for r in rs]),
                                            np.std([r["ACC_final"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage7.json")))


if __name__ == "__main__":
    main()
