# -*- coding: utf-8 -*-
"""三种学习器：唯一差别是"记忆怎么管"，网络、数据、梯度步数全部相同。

  A  naive        : 无记忆，只学当前任务
  B  replay       : 无门控回放（照单全收，reservoir 采样）
  C  tiered       : 三层信任记忆 + 验证门控，只回放"经验证"的部分

B 和 C 每个梯度步的回放量一模一样，所以任何差异只能来自"门控筛掉了什么"。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.mlp import MLP
from mm import NeighborValidator, TieredMemory


def t_has_warm(learner) -> bool:
    return getattr(learner, "mem", None) is not None and len(learner.mem.warm) > 0


class Learner:
    name = "?"
    uses_memory = False

    def __init__(self, dim, seed, capacity=600, batch_replay=16):
        self.model = MLP(dim, seed=seed)
        self.rng = np.random.default_rng(seed + 999)
        self.capacity = capacity
        self.batch_replay = batch_replay
        self.n_seen = 0

    def begin_task(self, task):
        # 按"出现过的最大类号"扩容：class-incremental 与"类重复出现"两种流都适用
        need = max(task.classes) + 1
        if self.model.n_out < need:
            self.model.expand(need - self.model.n_out)

    def absorb_batch(self, X, y, y_true, task_id):
        pass

    def end_task(self, task):
        pass

    def sample_replay(self, n):
        return None

    def predict(self, X):
        return self.model.predict(X)

    def memory_report(self):
        return {}

    # 污染率：记忆里有多少条记的是错标签（只用于事后算账）
    def pollution(self):
        return float("nan")


class NaiveLearner(Learner):
    """A：不记忆。作为"忘得多厉害"的下界。"""
    name = "A_naive"
    uses_memory = False

    def __init__(self, dim, seed, capacity=600, batch_replay=16):
        super().__init__(dim, seed, capacity, batch_replay=0)


class ReservoirLearner(Learner):
    """B：无门控回放。看到什么存什么，脏的也存。"""
    name = "B_replay"
    uses_memory = True

    def __init__(self, dim, seed, capacity=600, batch_replay=16):
        super().__init__(dim, seed, capacity, batch_replay)
        self.buf_x = []
        self.buf_y = []
        self.buf_true = []

    def absorb_batch(self, X, y, y_true, task_id):
        for i in range(len(X)):
            self.n_seen += 1
            if len(self.buf_x) < self.capacity:
                self.buf_x.append(X[i]); self.buf_y.append(int(y[i])); self.buf_true.append(int(y_true[i]))
            else:
                # reservoir sampling：保证对全历史均匀
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.capacity:
                    self.buf_x[j] = X[i]; self.buf_y[j] = int(y[i]); self.buf_true[j] = int(y_true[i])

    def sample_replay(self, n):
        if not self.buf_x:
            return None
        idx = self.rng.integers(0, len(self.buf_x), size=n)
        return np.array([self.buf_x[i] for i in idx]), np.array([self.buf_y[i] for i in idx])

    def memory_report(self):
        return {"store": len(self.buf_x), "rejected_total": 0}

    def pollution(self):
        if not self.buf_x:
            return float("nan")
        bad = sum(1 for a, b in zip(self.buf_y, self.buf_true) if a != b)
        return bad / len(self.buf_x)


class CleanReplayLearner(ReservoirLearner):
    """E：作弊上界 —— 直接把环境给的干净留出集当回放池用。

    它不是可部署方案（真实场景里你只有带噪的数据流），
    但它的存在是为了回答一个刺人的问题：
    "既然你已经有 oracle 了，为什么不干脆只回放 oracle 给的数据？"
    如果 E <= C，说明门控确实从'脏流'里捞出了额外价值。
    """
    name = "E_clean_replay"

    def absorb_batch(self, X, y, y_true, task_id):
        for i in range(len(X)):
            if int(y[i]) != int(y_true[i]):
                continue                      # 只收干净的
            self.n_seen += 1
            if len(self.buf_x) < self.capacity:
                self.buf_x.append(X[i]); self.buf_y.append(int(y[i])); self.buf_true.append(int(y_true[i]))
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.capacity:
                    self.buf_x[j] = X[i]; self.buf_y[j] = int(y[i]); self.buf_true[j] = int(y_true[i])


class TieredLearner(Learner):
    """C：三层信任记忆。热备照单全收，门控决定谁能进温备并参与回放。"""
    name = "C_tiered"
    uses_memory = True

    def __init__(self, dim, seed, capacity=600, batch_replay=16,
                 theta=0.7, validator=None, sim_floor=0.5, holdout_budget=None,
                 cold_mode="none", cold_alpha=0.05, cold_share=0.5,
                 admit_unverifiable=False):
        super().__init__(dim, seed, capacity, batch_replay)
        if validator is None:
            raise ValueError("TieredLearner 必须显式传入验证器（这就是这套架构的 oracle 账单）")
        self.validator = validator
        self.sim_floor = sim_floor
        # 每任务最多给验证器喂多少条带标注的留出样本（None = 全给）。
        # 这是阶段 2 的"oracle 预算"旋钮：标注量是这套架构唯一的真实成本。
        self.holdout_budget = holdout_budget
        self.oracle_labels_used = 0
        self.mem = TieredMemory(
            validator=self.validator,
            theta=theta,
            sim_floor=sim_floor,
            capacity_hot=1024,
            capacity_warm=capacity,
            read_order=("warm", "cold"),
        )
        # ------- 冷备：阶段 3 的主角 -------
        # cold_mode = none   冷备空着（阶段 1/2 的基线）
        #           = frozen 每个类第一次被验证通过时的原型，之后**永不修改**
        #           = ema    同一个类的原型按 alpha 慢慢吸收新观测
        # cold_share：回放时有多大比例抽自冷备（0 = 冷备不参与）
        self.admit_unverifiable = admit_unverifiable
        self.rollback_evicted = 0
        self.cold_mode = cold_mode
        self.cold_alpha = cold_alpha
        self.cold_share = cold_share
        self.cold_centroids = {}
        self.cold_updates = 0
        if hasattr(self.validator, "bind"):
            self.validator.bind(self.mem)
        # 门控诊断（每次巩固的混淆矩阵）
        self.gate_diag = []
        self.truth_of = {}

    def begin_task(self, task):
        super().begin_task(task)
        # 环境持续提供干净留出集 -> 这是这套架构要付的账单
        b = self.holdout_budget
        xh, yh = (task.x_holdout, task.y_holdout) if b is None else (task.x_holdout[:b], task.y_holdout[:b])
        self.oracle_labels_used += len(xh)
        self.validator.extend_pool(xh, yh)
        # 新一批标签刚刚到达 -> 先对上一批"先收下"的条目做事后裁判
        if self.admit_unverifiable and t_has_warm(self):
            self.rollback_evicted += self.mem.revalidate()["evicted"]

    def absorb_batch(self, X, y, y_true, task_id):
        for i in range(len(X)):
            self.n_seen += 1
            e = self.mem.observe(X[i], int(y[i]), tag=str(task_id))
            self.truth_of[id(e)] = (int(y[i]), int(y_true[i]))

    def end_task(self, task):
        rep = self.mem.consolidate(
            admit_unverifiable=self.admit_unverifiable,
            truth_fn=lambda e: self.truth_of.get(id(e), (0, 0))[0] != self.truth_of.get(id(e), (0, 0))[1]
        )
        self.gate_diag.append(rep)
        self._update_cold(task)

    def _update_cold(self, task):
        """把本任务里"通过验证"的条目按类压成原型，再按 cold_mode 决定怎么进冷备。"""
        if self.cold_mode == "none":
            return
        by_class = {}
        for e in self.mem.warm:
            if e.tag == str(task.task_id):
                by_class.setdefault(int(e.value), []).append(e.key)
        for c, keys in by_class.items():
            cen = np.mean(keys, axis=0)
            if c not in self.cold_centroids:
                self.cold_centroids[c] = cen
                self.cold_updates += 1
            elif self.cold_mode == "ema":
                a = self.cold_alpha
                self.cold_centroids[c] = (1.0 - a) * self.cold_centroids[c] + a * cen
                self.cold_updates += 1
            # frozen：既然已经"固化"，就一个字都不改

    def _sample_cold(self, n):
        classes = list(self.cold_centroids.keys())
        pick = self.rng.choice(len(classes), size=n)
        return (np.array([self.cold_centroids[classes[i]] for i in pick]),
                np.array([classes[i] for i in pick]))

    def sample_replay(self, n):
        pool = self.mem.warm
        use_cold = self.cold_mode != "none" and self.cold_centroids and self.cold_share > 0
        if not pool and not use_cold:
            return None
        if not use_cold:
            idx = self.rng.integers(0, len(pool), size=n)
            return (np.array([pool[i].key for i in idx]),
                    np.array([pool[i].value for i in idx]))
        n_cold = int(round(n * self.cold_share))
        n_warm = n - n_cold
        parts_x, parts_y = [], []
        if n_cold > 0:
            cx, cy = self._sample_cold(n_cold)
            parts_x.append(cx); parts_y.append(cy)
        if n_warm > 0:
            idx = self.rng.integers(0, len(pool), size=n_warm)
            parts_x.append(np.array([pool[i].key for i in idx]))
            parts_y.append(np.array([pool[i].value for i in idx]))
        return np.vstack(parts_x), np.concatenate(parts_y)

    def memory_report(self):
        s = self.mem.stats()
        fa = [r.false_accept_rate for r in self.gate_diag]
        fr = [r.false_reject_rate for r in self.gate_diag]
        fa = [v for v in fa if v == v]
        fr = [v for v in fr if v == v]
        return {"store": s["warm"], "rejected_total": s["rejected_total"],
                "accepted_total": s["accepted_total"],
                "cold": len(self.cold_centroids), "cold_updates": self.cold_updates,
                "oracle_labels": self.oracle_labels_used,
                "false_accept": float(np.mean(fa)) if fa else float("nan"),
                "false_reject": float(np.mean(fr)) if fr else float("nan"),
                "rollback_evicted": self.rollback_evicted}

    def pollution(self):
        pool = self.mem.warm
        if not pool:
            return float("nan")
        bad = 0
        for e in pool:
            rec = self.truth_of.get(id(e))
            if rec is not None and rec[0] != rec[1]:
                bad += 1
        return bad / len(pool)
