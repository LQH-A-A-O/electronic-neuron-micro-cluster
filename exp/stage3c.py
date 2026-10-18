# -*- coding: utf-8 -*-
"""阶段 3C：把"稀释"这个嫌疑排掉 —— 扫冷备参与回放的比例。

3A/3B 里冷备固定占回放的 50%。温备 600 条时这可能是"稀释了温备"而不是"冷备没用"。
所以把 cold_share 从 0.1 扫到 0.5，看三档容量下的结论会不会变。
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.learners import TieredLearner
from exp.recurring import RecurringDriftStream
from exp.run import train_one_task
from mm import NeighborValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DIM, STEPS, DRIFT, NOISE = 20, 300, 0.6, 0.3
SEEDS = [0, 1, 2]
CAPS = [50, 600]
SHARES = [0.1, 0.25, 0.5]
ARMS = [("F_frozen", dict(cold_mode="frozen"), False),
        ("T_true", dict(cold_mode="frozen"), True)]


def run_one(kw, is_true, stream, seed, cap, share):
    learner = TieredLearner(DIM, seed=seed, capacity=cap,
                            validator=NeighborValidator(np.zeros((0, DIM)), np.zeros(0)),
                            theta=0.7, cold_share=share, **kw)
    for t, task in enumerate(stream):
        train_one_task(learner, task, t, steps_per_task=STEPS)
        if is_true:
            for c in task.classes:
                learner.cold_centroids[int(c)] = stream.centers_at[t][int(c)].copy()
    return float((learner.predict(stream.xt) == stream.yt).mean())


def main():
    t0 = time.time()
    recs = []
    for cap in CAPS:
        for seed in SEEDS:
            st = RecurringDriftStream(n_classes=25, repeats=4, classes_per_task=10,
                                      n_train=500, n_holdout=200, n_test=100,
                                      drift=DRIFT, label_noise=NOISE, seed=1000 + seed)
            # 参照：完全不要冷备（与 share 无关）
            n = TieredLearner(DIM, seed=seed, capacity=cap,
                              validator=NeighborValidator(np.zeros((0, DIM)), np.zeros(0)),
                              theta=0.7, cold_mode="none")
            for t, task in enumerate(st):
                train_one_task(n, task, t, steps_per_task=STEPS)
            recs.append(dict(arm="N_no_cold", cap=cap, share=-1, seed=seed,
                             ACC_final=float((n.predict(st.xt) == st.yt).mean())))
            for share in SHARES:
                for name, kw, is_true in ARMS:
                    recs.append(dict(arm=name, cap=cap, share=share, seed=seed,
                                     ACC_final=run_one(kw, is_true, st, seed, cap, share)))
        print("cap=%d 完成 %.1f 秒" % (cap, time.time() - t0)); sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage3c.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 86)
    print("阶段 3C：冷备参与回放的比例（drift=0.6，3 seeds）")
    print("=" * 86)
    for cap in CAPS:
        base = np.mean([r["ACC_final"] for r in recs if r["cap"] == cap and r["arm"] == "N_no_cold"])
        print("温备 %d 条   参照 N_no_cold = %.3f" % (cap, base))
        print("  %-10s %-14s %-14s %s" % ("冷备占比", "F_frozen", "T_true", ""))
        for share in SHARES:
            f = np.mean([r["ACC_final"] for r in recs if r["cap"] == cap and r["arm"] == "F_frozen" and r["share"] == share])
            t = np.mean([r["ACC_final"] for r in recs if r["cap"] == cap and r["arm"] == "T_true" and r["share"] == share])
            print("  %-10s %.3f (%+.3f) %.3f (%+.3f)"
                  % ("%.0f%%" % (100 * share), f, f - base, t, t - base))
        print()
    print("总耗时 %.1f 秒" % (time.time() - t0))


if __name__ == "__main__":
    main()
