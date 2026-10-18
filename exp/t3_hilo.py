# -*- coding: utf-8 -*-
"""T3：把 HILO 从"开关"改回"旋钮"。

阶段 24 的发现
------------------------------------------------
两个通道都输出 **0/1 硬票**，所以合成分数只取 {0, 0.4, 0.6, 1.0} 四个离散值，
"犹豫区" (1-HILO, HILO) 只有 HILO > 0.60 时才可能命中 —— HILO 在 0.65/0.72/0.80
上给出的**是同一个决策函数**。所以它不是一个真的阈值。

怎么改
------------------------------------------------
让两个通道输出**连续置信度**：
    锚通道  a = 工具对"观测到的那个标签"的 softmax 概率       -> [0,1] 连续
    同伴通道 p = 留一打分对"观测标签"的 softmax 概率            -> [0,1] 连续
合成分数 sc = w*a + (1-w)*p 也就连续了。

要回答
  (1) HILO 现在是不是真的旋钮（不同取值给出不同行为）？
  (2) 连续版的最优点能不能超过离散版的 0.706？
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.realdata import RealClassStream, load_mnist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
P_DIM, N_CLASSES, POLLUTE, ROUND = 128, 10, 0.4, 50
N_TASK, CPT = 5, 2
N_TRAIN, N_HOLD, N_TEST = 400, 200, 200
TAU, KNN_K = 0.60, 8
SEEDS = [0, 1, 2, 3, 4]
HILOS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85]
W_A = 0.6
MN = load_mnist()
CONF, _ = weak_model_confusion()
_rng = np.random.default_rng(2026)
PROJ = _rng.normal(0, 1.0 / np.sqrt(784), (784, P_DIM))


def featurize(X):
    Z = np.asarray(X, dtype=np.float64) @ PROJ
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


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


def fit_tool(X, y, classes, ridge=1.0):
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    W = np.zeros((len(classes), A.shape[1]))
    for i, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        W[i] = Ginv @ (A.T @ t)
    return W


def softmax(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)


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


class R3:
    def __init__(self, mode, seed, hilo):
        self.mode, self.hilo = mode, hilo        # mode: 'discrete' | 'continuous'
        self.rng = np.random.default_rng(900 + seed)
        self.bx, self.by = [], []
        self.side = 0
        self.W = None
        self.classes = []
        self.amb_frac = []

    def refit(self):
        if len(self.by) >= 60:
            cl = sorted(set(self.by))
            if len(cl) >= 2:
                self.W = fit_tool(np.array(self.bx), np.array(self.by), cl)
                self.classes = cl

    def round(self, xb, yb, truth_fn):
        n = len(xb)
        cl = sorted(set(yb.tolist()))
        if len(cl) >= 2 and n >= 10:
            P = loo_scores(xb, yb, cl)
            if self.mode == "continuous":
                Q = softmax(P * 2.0)
                pos = {c: j for j, c in enumerate(cl)}
                pp = np.array([Q[i, pos[int(yb[i])]] for i in range(n)])
            else:
                pred = np.array(cl)[P.argmax(axis=1)]
                pp = np.array([1.0 if pred[i] == yb[i] else 0.0 for i in range(n)])
        else:
            pp = np.full(n, np.nan)
        if self.W is not None:
            Pt = softmax(np.hstack([xb, np.ones((n, 1))]) @ self.W.T)
            covered = Pt.max(axis=1) >= TAU
            pos = {c: j for j, c in enumerate(self.classes)}
            aa = np.array([Pt[i, pos[int(yb[i])]] if int(yb[i]) in pos else np.nan for i in range(n)])
            aa = np.where(covered, aa, np.nan)
        else:
            aa = np.full(n, np.nan)
        amb = 0
        for i in range(n):
            a, p = aa[i], pp[i]
            parts = []
            if a == a:
                parts.append((W_A, a))
            if p == p:
                parts.append(((1 - W_A) if a == a else 1.0, p))
            if not parts:
                sc = None
            else:
                tot = sum(w for w, _ in parts)
                sc = sum(w * v for w, v in parts) / max(tot, 1e-9)
            if sc is None or (1.0 - self.hilo) < sc < self.hilo:
                ok = truth_fn(xb[i]) == int(yb[i]); self.side += 1; amb += 1
            else:
                ok = sc >= self.hilo
            if ok:
                self.bx.append(xb[i]); self.by.append(int(yb[i]))
        self.amb_frac.append(amb / max(n, 1))
        self.refit()

    def predict(self, X):
        out = np.zeros(len(X), dtype=int)
        rest = np.ones(len(X), dtype=bool)
        if self.W is not None:
            Pt = softmax(np.hstack([X, np.ones((len(X), 1))]) @ self.W.T)
            ms = Pt.max(axis=1) >= TAU
            if ms.any():
                out[ms] = np.array(self.classes)[Pt.argmax(axis=1)][ms]
            rest = ~ms
        if rest.any() and self.bx:
            A = np.array(self.bx); ay = np.array(self.by)
            S = X[rest] @ A.T
            k = min(KNN_K, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            out[rest] = np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])
        return out


def run(mode, hilo, st, seed):
    R = R3(mode, seed, hilo)
    accs = []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    return dict(mode=mode, hilo=hilo, seed=seed, ACC_final=accs[-1], side=R.side,
                amb=float(np.mean(R.amb_frac)) if R.amb_frac else 0.0, kept=len(R.by))


if __name__ == "__main__":
    res = []
    for seed in SEEDS:
        st = make_stream(seed)
        for h in HILOS:
            for mode in ("discrete", "continuous"):
                res.append(run(mode, h, st, seed))
        print("seed=%d done" % seed); sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_t3_hilo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 78)
    print("T3：HILO 是开关还是旋钮（真实 MNIST / 污染 40% / 5 seeds）")
    print("=" * 78)
    print("%-6s %-14s %-14s %-12s %s" % ("HILO", "离散 ACC", "连续 ACC", "连续 side", "连续犹豫比例"))
    for h in HILOS:
        ds = [r for r in res if r["mode"] == "discrete" and r["hilo"] == h]
        cs = [r for r in res if r["mode"] == "continuous" and r["hilo"] == h]
        print("%-6.2f %-14.3f %-14.3f %-12.0f %.3f"
              % (h, np.mean([r["ACC_final"] for r in ds]), np.mean([r["ACC_final"] for r in cs]),
                 np.mean([r["side"] for r in cs]), np.mean([r["amb"] for r in cs])))
    for mode in ("discrete", "continuous"):
        accs = {}
        for h in HILOS:
            accs[h] = np.mean([r["ACC_final"] for r in res if r["mode"] == mode and r["hilo"] == h])
        sp = max(accs.values()) - min(accs.values())
        print("%-14s HILO 变动引起的 ACC 跨度 = %.4f  -> %s"
              % (mode, sp, "是旋钮" if sp > 0.02 else "是开关"))
