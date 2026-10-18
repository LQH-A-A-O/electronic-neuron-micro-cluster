# -*- coding: utf-8 -*-
"""跑一条持续学习轨迹，返回准确率矩阵和记忆账本。"""

from __future__ import annotations

import numpy as np


def train_one_task(learner, task, t, steps_per_task=300, batch_current=32):
    """在线吃一个任务：先按一遍全量入记忆，再跑固定步数的梯度更新。"""
    learner.begin_task(task)
    learner.absorb_batch(task.x_train, task.y_train, task.y_train_true, t)
    n = len(task.x_train)
    rng = np.random.default_rng(12345 + t)
    for _ in range(steps_per_task):
        idx = rng.integers(0, n, size=batch_current)
        Xc, yc = task.x_train[idx], task.y_train[idx]
        rb = learner.sample_replay(learner.batch_replay) if learner.batch_replay else None
        if rb is None:
            learner.model.step(Xc, yc)
        else:
            Xr, yr = rb
            learner.model.step(np.vstack([Xc, Xr]), np.concatenate([yc, yr]))
    learner.end_task(task)


def run_trajectory(learner, stream, steps_per_task=300, batch_current=32, verbose=False):
    T = len(stream)
    acc = np.full((T, T), np.nan)
    ledger = []

    for t, task in enumerate(stream):
        train_one_task(learner, task, t, steps_per_task, batch_current)

        for j in range(t + 1):
            p = learner.predict(stream.tasks[j].x_test)
            acc[t, j] = float((p == stream.tasks[j].y_test).mean())

        rep = learner.memory_report()
        ledger.append({
            "task": t,
            "acc_now": float(acc[t, t]),
            "acc_mean_seen": float(np.nanmean(acc[t, : t + 1])),
            "pollution": learner.pollution(),
            **rep,
        })
        if verbose:
            print("  task %2d  本任务 acc=%.3f  已见平均=%.3f  记忆=%s 污染=%s"
                  % (t, acc[t, t], np.nanmean(acc[t, : t + 1]), rep,
                     ("%.3f" % ledger[-1]["pollution"]) if ledger[-1]["pollution"] == ledger[-1]["pollution"] else "-"))
    return acc, ledger


def summarize(acc, ledger, n_tasks=None):
    T = acc.shape[0]
    final = acc[T - 1]
    acc_final = float(np.nanmean(final))
    diag = np.array([acc[j, j] for j in range(T)])
    bwt = float(np.nanmean(final - diag))
    tail = float(np.nanmean([acc[j, j] for j in range(max(0, T - 3), T)]))
    pol = [l["pollution"] for l in ledger]
    pol = [p for p in pol if p == p]
    return {
        "ACC_final": acc_final,
        "BWT": bwt,
        "ACC_new_tail3": tail,
        "pollution_final": pol[-1] if pol else float("nan"),
        "store_final": ledger[-1].get("store", 0),
        "rejected_total": ledger[-1].get("rejected_total", 0),
    }
