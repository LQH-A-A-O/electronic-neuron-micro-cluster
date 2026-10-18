# -*- coding: utf-8 -*-
"""合成持续学习任务流：class-incremental + 可注入标签噪声。

设计要点
- 每个任务带来 5 个全新的类，共 T 个任务 -> 总类数 5T
- 测试时不提供任务 id（class-incremental），所以遗忘会非常显著
- 标签噪声只加在"训练流"上；验证集/测试集标签永远干净
  => 噪声 = 我们人为制造的"幻觉"，验证集 = 环境提供的 oracle
- 每个任务额外给一份干净的 holdout，这就是三层记忆架构要付的账单
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Task:
    task_id: int
    classes: List[int]
    x_train: np.ndarray
    y_train: np.ndarray          # 带噪标签
    y_train_true: np.ndarray     # 真实标签（只用于事后算污染率）
    x_holdout: np.ndarray
    y_holdout: np.ndarray        # 干净，交给验证器
    x_test: np.ndarray
    y_test: np.ndarray           # 干净

    @property
    def n_classes(self) -> int:
        return len(self.classes)


class TaskStream:
    def __init__(self, n_tasks=10, classes_per_task=5, dim=20,
                 n_train=300, n_holdout=150, n_test=150,
                 label_noise=0.0, cluster_spread=1.0, center_scale=4.0,
                 seed=0):
        self.cfg = dict(n_tasks=n_tasks, classes_per_task=classes_per_task, dim=dim,
                        n_train=n_train, n_holdout=n_holdout, n_test=n_test,
                        label_noise=label_noise, cluster_spread=cluster_spread,
                        center_scale=center_scale, seed=seed)
        self.n_classes_total = n_tasks * classes_per_task
        rng = np.random.default_rng(seed)
        # 每个类一个固定原型，任务之间共享同一套空间
        self.centers = rng.normal(0, center_scale, size=(self.n_classes_total, dim))
        self.tasks: List[Task] = []
        for t in range(n_tasks):
            classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
            self.tasks.append(self._make_task(t, classes, rng))

    def _draw(self, classes, n, rng, spread):
        y = rng.choice(classes, size=n)
        x = self.centers[y] + rng.normal(0, spread, size=(n, self.cfg["dim"]))
        return x, y

    def _make_task(self, t, classes, rng) -> Task:
        c = self.cfg
        x_tr, y_true = self._draw(classes, c["n_train"], rng, c["cluster_spread"])
        y_tr = y_true.copy()
        p = c["label_noise"]
        if p > 0:
            flip = rng.random(c["n_train"]) < p
            other = rng.choice([k for k in classes], size=int(flip.sum()))
            # 保证翻转后的标签不等于原标签
            bad = other == y_true[flip]
            while bad.any():
                other[bad] = rng.choice(classes, size=int(bad.sum()))
                bad = other == y_true[flip]
            y_tr[flip] = other
        x_h, y_h = self._draw(classes, c["n_holdout"], rng, c["cluster_spread"])
        x_te, y_te = self._draw(classes, c["n_test"], rng, c["cluster_spread"])
        return Task(t, classes, x_tr, y_tr, y_true, x_h, y_h, x_te, y_te)

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
