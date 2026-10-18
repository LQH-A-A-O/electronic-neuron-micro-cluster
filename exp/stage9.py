# -*- coding: utf-8 -*-
"""阶段 9：把「待裁判窗口」这个旋钮拧小 —— 阶段 8 指出的真正的杠杆。

阶段 7/8 的结论：延迟标签的污染是结构性的，重新加权绕不开它；
真正的杠杆是**减少同时挂着的未裁判条目**。
阶段 7 里这个残留是 404 条（= 整整一个任务），因为标签按"任务"整批到达。

这一阶段把标签改成**按观测条数**延迟到达（现实里反馈本来就是一条条回来的），
于是窗口长度变成一个连续可调的旋钮：

    delay = 40 / 80 / 160 / 400 / 800 条观测
    每 80 条观测做一次「巩固 + 事后裁判」

对照：阶段 7 的 delay=1 个任务（400 条训练样本）应当在 delay=400 处复现（≈0.471）。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.learners import NaiveLearner, ReservoirLearner, TieredLearner
from exp.realdata import RealClassStream, load_mnist
from mm import NeighborValidator, ObservationDelayValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
CAP, P = 600, 0.4
BLOCK, EVERY, STEPS_PER_BLOCK, BATCH = 40, 80, 30, 32
SEEDS = [0, 1, 2]
DELAYS = [40, 80, 160, 400, 800]
MN = load_mnist()
CONF, WACC = weak_model_confusion()


def make_stream(seed):
    st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=5, classes_per_task=2,
                         n_train=400, n_holdout=200, n_test=200,
                         label_noise=0.0, seed=1000 + seed)
    rng = np.random.default_rng(777 + seed)
    for task in st:
        task.y_train = inject(task.y_train_true, task.classes, P, rng, "structured", CONF)
    return st


def run_delayed(L, stream):
    """L 的验证器必须是 ObservationDelayValidator；池子完全由"迟到的标签"喂。"""
    T = len(stream)
    acc = np.full((T, T), np.nan)
    n_rounds = 0
    for t, task in enumerate(stream):
        L.begin_task(task)                       # holdout_budget=0 -> 不推任何标签
        n = len(task.x_train)
        rng = np.random.default_rng(12345 + t)
        seen = 0
        for s0 in range(0, n, BLOCK):
            sl = slice(s0, min(s0 + BLOCK, n))
            xb, yb, ytb = task.x_train[sl], task.y_train[sl], task.y_train_true[sl]
            L.absorb_batch(xb, yb, ytb, t)
            L.validator.push(xb, yb)             # 标签进延迟队列
            seen += len(xb)
            for _ in range(STEPS_PER_BLOCK):
                idx = rng.integers(0, len(xb), size=BATCH)
                rb = L.sample_replay(L.batch_replay) if L.batch_replay else None
                if rb is None:
                    L.model.step(xb[idx], yb[idx])
                else:
                    L.model.step(np.vstack([xb[idx], rb[0]]), np.concatenate([yb[idx], rb[1]]))
            if seen % EVERY == 0:
                rep = L.mem.consolidate(
                    admit_unverifiable=True,
                    truth_fn=lambda e: L.truth_of.get(id(e), (0, 0))[0] != L.truth_of.get(id(e), (0, 0))[1])
                L.gate_diag.append(rep)
                L.rollback_evicted += L.mem.revalidate()["evicted"]
                n_rounds += 1
        for j in range(t + 1):
            p = L.predict(stream.tasks[j].x_test)
            acc[t, j] = float((p == stream.tasks[j].y_test).mean())
    return acc, n_rounds


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for delay in DELAYS:
        for seed in SEEDS:
            st = make_stream(seed)
            v = ObservationDelayValidator(NeighborValidator(np.zeros((0, st.dim)), np.zeros(0)), delay=delay)
            L = TieredLearner(st.dim, seed=seed, capacity=CAP, validator=v, theta=0.7,
                              admit_unverifiable=True, holdout_budget=0)
            acc, n_rounds = run_delayed(L, st)
            T = acc.shape[0]
            final = np.nanmean(acc[T - 1])
            prov = sum(1 for e in L.mem.warm if e.provisional)
            pol = L.pollution()
            recs.append(dict(delay=delay, seed=seed, ACC_final=float(final),
                             provisional_left=prov, warm=len(L.mem.warm),
                             pollution=pol, evicted=L.rollback_evicted, rounds=n_rounds))
        print("delay=%d 完成 %.0f 秒" % (delay, time.time() - t0)); sys.stdout.flush()

    for arm in ("A_naive", "B_replay"):
        for seed in SEEDS:
            st = make_stream(seed)
            L = NaiveLearner(st.dim, seed=seed) if arm == "A_naive" else \
                ReservoirLearner(st.dim, seed=seed, capacity=CAP)
            from exp.run import run_trajectory, summarize
            acc, led = run_trajectory(L, st, steps_per_task=400)
            recs.append(dict(delay=-1, seed=seed, arm=arm, **summarize(acc, led)))

    json.dump(recs, open(os.path.join(OUT, "results_stage9.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print("=" * 96)
    print("阶段 9：待裁判窗口 vs 准确率（真实 MNIST / 结构化幻觉 40% / 3 seeds）")
    print("=" * 96)
    print("%-14s %-18s %-16s %-16s %s" % ("标签延迟(条)", "ACC", "待裁判残留", "污染率", "回滚撤回"))
    for delay in DELAYS:
        rs = [r for r in recs if r["delay"] == delay]
        print("%-14d %.3f±%.3f        %-16.0f %-16.3f %.0f"
              % (delay, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["provisional_left"] for r in rs]),
                 np.nanmean([r["pollution"] for r in rs]),
                 np.mean([r["evicted"] for r in rs])))
    for arm in ("B_replay", "A_naive"):
        rs = [r for r in recs if r.get("arm") == arm]
        print("%-14s %.3f±%.3f   (参照)" % (arm, np.mean([r["ACC_final"] for r in rs]),
                                            np.std([r["ACC_final"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage9.json")))


if __name__ == "__main__":
    main()
