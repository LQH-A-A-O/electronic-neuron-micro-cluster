# -*- coding: utf-8 -*-
"""阶段 12：把「工具化」搬到真实数据上 —— MNIST。

为什么必须做这个
------------------------------------------------
阶段 10 的"编译成工具"是在**合成多项式域**上做的，那里编译是**无损**的：
一条稀疏多项式就是知识本身，拟合出来就等于知识。
真实图像显然不是这样。所以这一阶段问三个问题：

  Q1 真实知识能编译成什么样的紧凑函数？（原型 / 线性判别）
  Q2 编译损失多大？（判定准确率 vs 不编译的记忆检索）
  Q3 **编译出来的函数能不能当免费验证通道，替掉 gold oracle？**

四组对照
------------------------------------------------
  gold_mem    gold holdout 门控 + 保留经验证样本 + kNN 判定      （阶段 4 的基线）
  gold_tool   gold holdout 门控 + **编译成线性判别式并丢掉样本**  （隔离 Q2 的编译损失）
  toolonly    **零 gold**：先无条件信一小批 -> 编译成工具 -> 用工具验证后续  （Q3）
  B_replay / A_naive  参照

特征：MNIST 原始 784 维先做固定的随机投影到 128 维（所有组共用同一投影，
所以组间比较仍然干净），判定与编译都在这个 128 维空间里做。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.learners import NaiveLearner, ReservoirLearner
from exp.realdata import RealClassStream, load_mnist
from exp.run import run_trajectory, summarize
from mm import NeighborValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
P_DIM = 128
N_CLASSES = 10
POLLUTE = 0.4
BOOT = 80                 # toolonly 每个任务无条件信任的前多少条
REFIT_EVERY = 40
KNN_K = 5
SEEDS = [0, 1, 2]
MN = load_mnist()
CONF, WACC = weak_model_confusion()
_rng = np.random.default_rng(2026)
PROJ = _rng.normal(0, 1.0 / np.sqrt(784), (784, P_DIM))


def featurize(X):
    Z = np.asarray(X, dtype=np.float64) @ PROJ
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def make_stream(seed):
    st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=5, classes_per_task=2,
                         n_train=400, n_holdout=200, n_test=200,
                         label_noise=0.0, seed=1000 + seed)
    rng = np.random.default_rng(777 + seed)
    for task in st:
        task.y_train = inject(task.y_train_true, task.classes, POLLUTE, rng, "structured", CONF)
        task.x_train = featurize(task.x_train)
        task.x_holdout = featurize(task.x_holdout)
        task.x_test = featurize(task.x_test)
    st.dim = P_DIM
    return st


class Tool:
    """把一批经验证的样本编译成每类一个线性判别式（含偏置）。"""

    def __init__(self, X, y, classes, ridge=1.0):
        self.classes = list(classes)
        self.W = np.zeros((len(self.classes), X.shape[1] + 1))
        A = np.hstack([X, np.ones((len(X), 1))])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        for i, c in enumerate(self.classes):
            t = (y == c).astype(np.float64) * 2.0 - 1.0
            self.W[i] = np.linalg.solve(G, A.T @ t)

    def predict(self, X):
        A = np.hstack([X, np.ones((len(X), 1))])
        return np.array(self.classes)[np.argmax(A @ self.W.T, axis=1)]

    @property
    def n_params(self):
        return self.W.size


class R12:
    def __init__(self, mode, seed):
        self.mode = mode
        self.accepted_x, self.accepted_y = [], []
        self.tool = None
        self.gold_used = 0
        self.seen_classes = []
        self.n_seen_in_task = 0
        if mode != "toolonly":
            self.validator = None

    # ---- 判定 ----
    def predict(self, X):
        if self.tool is not None and self.mode != "gold_mem":
            return self.tool.predict(X)
        if not self.accepted_x:
            return np.zeros(len(X), dtype=int)
        A = np.array(self.accepted_x)
        ay = np.array(self.accepted_y)
        S = X @ A.T
        k = min(KNN_K, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        return np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])

    # ---- 编译 ----
    def refit(self):
        if len(self.accepted_x) < 40:
            return
        X = np.array(self.accepted_x)
        y = np.array(self.accepted_y)
        cls = sorted(set(y.tolist()))
        if len(cls) < 2:
            return
        self.tool = Tool(X, y, cls)

    # ---- 一个任务 ----
    def run_task(self, task, validator):
        rng = np.random.default_rng(11 + task.task_id)
        xo, yo = task.x_train, task.y_train
        n = len(xo)
        n_gold = 0
        for s0 in range(0, n, 50):
            xb, yb = xo[s0:s0 + 50], yo[s0:s0 + 50]
            for i in range(len(xb)):
                x, y = xb[i], int(yb[i])
                if self.mode == "toolonly":
                    if self.n_seen_in_task < BOOT or self.tool is None:
                        keep = True                       # 自举：先无条件信
                    else:
                        keep = int(self.tool.predict(x[None, :])[0]) == y
                else:
                    # gold 通道：用留出集投票（要花 1 次标注）
                    self.gold_used += 1
                    keep = validator.score(x, y) >= 0.7
                self.n_seen_in_task += 1
                if keep:
                    self.accepted_x.append(x)
                    self.accepted_y.append(y)
            if (s0 // 50) % 2 == 1:
                self.refit()
        self.refit()
        self.n_seen_in_task = 0


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode in ("gold_mem", "gold_tool", "toolonly"):
            R = R12(mode, seed)
            accs, accs_nogold = [], []
            for t, task in enumerate(st):
                v = NeighborValidator(np.zeros((0, P_DIM)), np.zeros(0))
                v.extend_pool(task.x_holdout, task.y_holdout)   # 每个任务一份 gold 留出集
                R.run_task(task, v)
                a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean())
                     for j in range(t + 1)]
                accs.append(float(np.mean(a)))
            n_samp = len(R.accepted_x)
            recs.append(dict(mode=mode, seed=seed, ACC_final=accs[-1], acc_curve=accs,
                             gold_used=R.gold_used, kept=n_samp,
                             params=(R.tool.n_params if R.tool else 0),
                             tool_classes=(len(R.tool.classes) if R.tool else 0)))
        for arm in ("B_replay", "A_naive"):
            st2 = make_stream(seed)
            L = NaiveLearner(P_DIM, seed=seed) if arm == "A_naive" else \
                ReservoirLearner(P_DIM, seed=seed, capacity=600)
            for t, task in enumerate(st2):
                pass
            acc, led = run_trajectory(L, st2, steps_per_task=400)
            recs.append(dict(mode=arm, seed=seed, ACC_final=summarize(acc, led)["ACC_final"],
                             gold_used=0, kept=0, params=0, tool_classes=0))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage12.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 100)
    print("阶段 12：真实数据（MNIST，128 维随机投影）上的工具化 / 3 seeds")
    print("=" * 100)
    print("%-12s %-16s %-16s %-16s %s" % ("组", "最终 ACC", "gold 标注次数", "保留样本数", "工具参数量"))
    for mode in ("gold_mem", "gold_tool", "toolonly", "B_replay", "A_naive"):
        rs = [r for r in recs if r["mode"] == mode]
        if not rs:
            continue
        print("%-12s %.3f±%.3f        %-16.0f %-16.0f %.0f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["gold_used"] for r in rs]), np.mean([r["kept"] for r in rs]),
                 np.mean([r["params"] for r in rs])))
    print()
    for mode in ("gold_mem", "gold_tool", "toolonly"):
        rs = [r for r in recs if r["mode"] == mode and "acc_curve" in r]
        cur = np.mean([r["acc_curve"] for r in rs], axis=0)
        print("  %-10s 逐任务 ACC: %s" % (mode, " ".join("%.3f" % v for v in cur)))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage12.json")))


if __name__ == "__main__":
    main()
