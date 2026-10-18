# -*- coding: utf-8 -*-
"""重复出现 + 概念漂移的任务流。

为什么阶段 3 需要这个
------------------------------------------------
阶段 1/2 用的是 class-incremental：每个类只出现一次。
在那种流里"冷备该不该更新"根本触发不了 —— 没有第二次观测。

而构想里最悬而未决的一条恰恰是（原对话最后一句）：
    "冷备份永远不变吗？若不进化，是否永远回滚到原始状态？"
要回答它，必须让同一个类**反复出现**，并且**世界本身在慢慢移动**。
于是"固化"才可能变成"陈旧"，"缓慢吸收"才有意义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from exp.taskstream import Task


class RecurringDriftStream:
    def __init__(self, n_classes=50, dim=20, n_tasks=10, classes_per_task=10,
                 repeats=2, n_train=300, n_holdout=150, n_test=100,
                 label_noise=0.3, center_scale=3.0, cluster_spread=1.5,
                 drift=0.0, seed=0, n_final_test=100):
        assert n_classes * repeats == n_tasks * classes_per_task, "排课数不匹配"
        self.cfg = dict(n_classes=n_classes, dim=dim, n_tasks=n_tasks,
                        classes_per_task=classes_per_task, repeats=repeats,
                        n_train=n_train, n_holdout=n_holdout, n_test=n_test,
                        label_noise=label_noise, center_scale=center_scale,
                        cluster_spread=cluster_spread, drift=drift, seed=seed)
        self.dim = dim
        self.n_classes_total = n_classes
        rng = np.random.default_rng(seed)
        self.centers_start = rng.normal(0, center_scale, size=(n_classes, dim))

        slots = np.repeat(np.arange(n_classes), repeats)
        rng.shuffle(slots)
        self.schedule = slots.reshape(n_tasks, classes_per_task)

        centers = self.centers_start.copy()
        self.tasks: List[Task] = []
        self.drift_steps = []
        self.centers_at: List[np.ndarray] = []      # 每个任务开始那一刻的真实原型
        for t in range(n_tasks):
            classes = sorted(int(c) for c in set(self.schedule[t]))
            self.centers_at.append(centers.copy())
            self.tasks.append(self._make_task(t, classes, centers, rng))
            if drift > 0:
                step = rng.normal(0, drift, size=centers.shape)
                centers = centers + step
                self.drift_steps.append(step)
        self.centers_final = centers

        # 最终测试集：全部类，用**漂移之后**的原型生成 = 当前世界的真值
        y = np.repeat(np.arange(n_classes), n_final_test)
        x = centers[y] + rng.normal(0, cluster_spread, size=(len(y), dim))
        self.xt = x
        self.yt = y.astype(int)
        self.last_seen = {c: max(t for t in range(n_tasks) if c in set(self.schedule[t]))
                          for c in range(n_classes)}
        self.age = np.array([n_tasks - 1 - self.last_seen[c] for c in range(n_classes)])

    def _draw(self, classes, n, centers, rng):
        spread = self.cfg["cluster_spread"]
        y = rng.choice(classes, size=n)
        return centers[y] + rng.normal(0, spread, size=(n, self.dim)), y

    def _make_task(self, t, classes, centers, rng) -> Task:
        c = self.cfg
        x_tr, y_true = self._draw(classes, c["n_train"], centers, rng)
        y_tr = y_true.copy()
        p = c["label_noise"]
        if p > 0:
            flip = rng.random(len(y_tr)) < p
            other = rng.choice(classes, size=int(flip.sum()))
            bad = other == y_true[flip]
            while bad.any():
                other[bad] = rng.choice(classes, size=int(bad.sum()))
                bad = other == y_true[flip]
            y_tr[flip] = other
        x_h, y_h = self._draw(classes, c["n_holdout"], centers, rng)
        x_te, y_te = self._draw(classes, c["n_test"], centers, rng)
        return Task(t, classes, x_tr, y_tr, y_true, x_h, y_h, x_te, y_te)

    def per_class_accuracy(self, pred: np.ndarray) -> np.ndarray:
        """按类返回准确率（顺序与 0..n_classes-1 一致）。"""
        out = np.full(self.n_classes_total, np.nan)
        for c in range(self.n_classes_total):
            m = self.yt == c
            if m.any():
                out[c] = float((pred[m] == self.yt[m]).mean())
        return out

    def age_bucket_accuracy(self, pred: np.ndarray):
        """按"这个类最后一次出现距今多少个任务"分桶，回答'冷备对陈旧知识有没有用'。"""
        per = self.per_class_accuracy(pred)
        buckets = {}
        for name, lo, hi in (("新鲜(0-1)", 0, 1), ("中等(2-4)", 2, 4), ("陈旧(5-9)", 5, 99)):
            m = (self.age >= lo) & (self.age <= hi)
            buckets[name] = float(np.nanmean(per[m])) if m.any() else float("nan")
        return buckets

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
