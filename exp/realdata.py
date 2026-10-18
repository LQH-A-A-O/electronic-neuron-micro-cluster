# -*- coding: utf-8 -*-
"""真实数据的持续学习任务流：MNIST class-incremental。

和合成流的唯一区别就是数据来源。模型、优化器、门控、对照组全部沿用阶段 1 的实现，
这样"结论是否只在合成高斯上成立"才是干净的对照。
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from exp.taskstream import Task

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_mnist():
    d = DATA
    x = np.load(os.path.join(d, "mnist_train.npy")).astype(np.float32)
    y = np.load(os.path.join(d, "mnist_train_y.npy")).astype(np.int64)
    xt = np.load(os.path.join(d, "mnist_test.npy")).astype(np.float32)
    yt = np.load(os.path.join(d, "mnist_test_y.npy")).astype(np.int64)
    return x, y, xt, yt


class RealClassStream:
    def __init__(self, x_tr, y_tr, x_te, y_te, n_tasks=5, classes_per_task=2,
                 n_train=400, n_holdout=200, n_test=200, label_noise=0.0,
                 seed=0, standardize=True):
        self.dim = int(x_tr.shape[1])
        x_tr = x_tr / 255.0
        x_te = x_te / 255.0
        if standardize:
            mu = x_tr.mean(axis=0, keepdims=True)
            sd = x_tr.std(axis=0, keepdims=True) + 1e-6
            x_tr = (x_tr - mu) / sd
            x_te = (x_te - mu) / sd
        self.cfg = dict(n_tasks=n_tasks, classes_per_task=classes_per_task,
                        n_train=n_train, n_holdout=n_holdout, n_test=n_test,
                        label_noise=label_noise, seed=seed)
        self.n_classes_total = n_tasks * classes_per_task
        rng = np.random.default_rng(seed)
        # 每个类抽一个固定池子，避免各任务互相抢样本
        self.tasks: List[Task] = []
        for t in range(n_tasks):
            classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
            self.tasks.append(self._make_task(t, classes, x_tr, y_tr, x_te, y_te, rng))

    def _draw(self, pool_x, pool_y, classes, n, rng):
        idx_parts = []
        for c in classes:
            ci = np.flatnonzero(pool_y == c)
            take = max(1, n // len(classes))
            idx_parts.append(rng.choice(ci, size=take, replace=len(ci) < take))
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
        return pool_x[idx], pool_y[idx].copy()

    def _make_task(self, t, classes, x_tr, y_tr, x_te, y_te, rng) -> Task:
        c = self.cfg
        x_train, y_true = self._draw(x_tr, y_tr, classes, c["n_train"], rng)
        y_train = y_true.copy()
        p = c["label_noise"]
        if p > 0:
            flip = rng.random(len(y_train)) < p
            other = rng.choice(classes, size=int(flip.sum()))
            bad = other == y_true[flip]
            while bad.any():
                other[bad] = rng.choice(classes, size=int(bad.sum()))
                bad = other == y_true[flip]
            y_train[flip] = other
        x_h, y_h = self._draw(x_te, y_te, classes, c["n_holdout"], rng)
        x_t, y_t = self._draw(x_te, y_te, classes, c["n_test"], rng)
        return Task(t, classes, x_train, y_train, y_true, x_h, y_h, x_t, y_t)

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
