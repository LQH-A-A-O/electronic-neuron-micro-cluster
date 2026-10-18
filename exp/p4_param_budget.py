# -*- coding: utf-8 -*-
"""P4：**同样准确率下，门控能让参数量小多少？** —— "3B 够不够用"的迷你版。

为什么这么设计
------------------------------------------------
项目新定位是「电子神经元超小集群模型（单机部署）」，硬约束是**每次激活 <= 3B**。
那么真正要回答的问题是：
    **门控清洗过的记忆，能不能让"同样质量"所需的模型容量显著变小？**

这里把"模型容量"做成可控的旋钮：把 128 维特征随机投影到 d' 维（d' 从 4 到 256），
再训练一个 d' x C 的线性判别器。参数量 = d' x C，是可控且单调的容量代理。

三组
    full   用真标签训练（上界）
    raw    用 40% 脏标签训练（不做门控）
    gate   先过门控，再用留下来的记忆训练

要看的
  (1) 对每个容量档，gate 比 raw 高多少 ACC？
  (2) 要达到同一个目标 ACC，gate 比 raw 省多少参数量？  <- 这就是"能不能把小模型做小"
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.realdata import RealClassStream, load_mnist
from exp.stage20 import featurize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
POLLUTE, N_TASK, CPT = 0.4, 5, 2
N_TRAIN, N_HOLD, N_TEST = 400, 200, 200
TAU, HILO = 0.60, 0.65
DIMS = [4, 8, 16, 32, 64, 128, 256]
SEEDS = list(range(5))
TARGET = 0.62          # 目标准确率（用来算"省多少参数"）
MN = load_mnist()
CONF, _ = weak_model_confusion()
_rng = np.random.default_rng(2026)
PROJ = _rng.normal(0, 1.0 / np.sqrt(784), (784, 128))


def make_stream(seed):
    st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=N_TASK, classes_per_task=CPT,
                         n_train=N_TRAIN, n_holdout=N_HOLD, n_test=N_TEST,
                         label_noise=0.0, seed=1000 + seed)
    rng = np.random.default_rng(777 + seed)
    for task in st:
        task.y_train = inject(task.y_train_true, task.classes, POLLUTE, rng, "structured", CONF)
        task.x_train = featurize(task.x_train)
        task.x_holdout = featurize(task.x_holdout)
        task.x_test = featurize(task.x_test)
    return st


def loo_scores(X, y, classes, ridge=1.0):
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    h = np.clip(np.sum((A @ Ginv) * A, axis=1), 0.0, 0.999)
    P = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        w = Ginv @ (A.T @ t)
        P[:, ci] = t - (t - A @ w) / (1.0 - h)
    return P


def gate_filter(X, y):
    """用阶段 20/21 那条已验证的门控筛一遍，返回保留下来的下标。"""
    cls = sorted(set(y.tolist()))
    if len(cls) < 2 or len(X) < 10:
        return np.arange(len(X))
    P = loo_scores(X, y, cls)
    pred = np.array(cls)[P.argmax(axis=1)]
    keep = []
    for i in range(len(X)):
        sc = 1.0 if pred[i] == y[i] else 0.0       # 只有同伴通道可用（冷启动）
        if sc >= HILO:
            keep.append(i)
    return np.array(keep)


def fit_acc(Xtr, ytr, Xte, yte, ncls, dsub, ridge=1.0):
    if len(Xtr) < 4 or len(set(ytr.tolist())) < 2:
        return float("nan")
    rng = np.random.default_rng(7)
    P = rng.normal(0, 1.0 / np.sqrt(Xtr.shape[1]), (Xtr.shape[1], dsub))
    A = np.hstack([Xtr @ P, np.ones((len(Xtr), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    cls = sorted(set(ytr.tolist()))
    W = np.zeros((len(cls), A.shape[1]))
    for i, c in enumerate(cls):
        t = (ytr == c).astype(np.float64) * 2.0 - 1.0
        W[i] = Ginv @ (A.T @ t)
    At = np.hstack([Xte @ P, np.ones((len(Xte), 1))])
    pred = np.array(cls)[(At @ W.T).argmax(axis=1)]
    return float((pred == yte).mean())


def run(seed):
    st = make_stream(seed)
    Xtr = np.vstack([t.x_train for t in st]); Yraw = np.concatenate([t.y_train for t in st])
    Ytrue = np.concatenate([t.y_train_true for t in st])
    Xte = np.vstack([t.x_test for t in st]); Yte = np.concatenate([t.y_test for t in st])
    ncls = int(max(Yte.max(), Ytrue.max())) + 1
    idx = gate_filter(Xtr, Yraw)
    out = {}
    for d in DIMS:
        out[d] = dict(full=fit_acc(Xtr, Ytrue, Xte, Yte, ncls, d),
                      raw=fit_acc(Xtr, Yraw, Xte, Yte, ncls, d),
                      gate=fit_acc(Xtr[idx], Yraw[idx], Xte, Yte, ncls, d),
                      keep=len(idx))
    return out


if __name__ == "__main__":
    res = [run(s) for s in SEEDS]
    json.dump([{str(k): v for k, v in r.items()} for r in res],
              open(os.path.join(OUT, "results_p4_budget.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def m(d, a):
        return float(np.mean([r[d][a] for r in res]))

    print()
    print("=" * 88)
    print("P4：准确率 vs 参数量（真实 MNIST / 污染 40% / 5 seeds；参数量 = d' x 类别数）")
    print("=" * 88)
    print("%-8s %-12s %-12s %-12s %-12s %s" % ("d'", "参数量", "真标签", "脏标签", "门控", "门控-脏标签"))
    tab = {}
    for d in DIMS:
        f, r_, g = m(d, "full"), m(d, "raw"), m(d, "gate")
        tab[d] = (f, r_, g)
        print("%-8d %-12d %-12.3f %-12.3f %-12.3f %+.4f" % (d, d * 10, f, r_, g, g - r_))
    print()
    print("目标准确率 %.2f 时的最小参数量（线性插值）" % TARGET)
    for a, nm in (("raw", "脏标签"), ("gate", "门控")):
        need = None
        for d in DIMS:
            if tab[d][{"raw": 1, "gate": 2}[a]] >= TARGET:
                need = d * 10; break
        print("  %-8s %s" % (nm, ("%d 参数 (d'=%d)" % (need, need // 10)) if need else "在扫描范围内达不到"))
    nr = next((d * 10 for d in DIMS if tab[d][1] >= TARGET), None)
    ng = next((d * 10 for d in DIMS if tab[d][2] >= TARGET), None)
    if nr and ng:
        print("  -> 门控省了 %.0f%% 的参数" % (100 * (1 - ng / nr)))
