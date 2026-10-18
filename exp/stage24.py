# -*- coding: utf-8 -*-
"""阶段 24（G4）：关键超参敏感度扫描 —— D5。

为什么必须做
------------------------------------------------
到阶段 23 为止，所有关键超参都是**随手定的**，而且从没扫过。
一个"调一调就崩"的机制不能上线 —— 用户不可能知道我们手气好。

扫哪四个
------------------------------------------------
只扫**用户真正要自己设**的四个（都在部署路径上）：
  TAU       锚的置信度门槛：只有它的 softmax 置信度超过它，才算"管得着 x"
  HILO      门控决断阈值：score >= HILO 收，<= 1-HILO 拒，中间才花 oracle
  KNN_K     记忆检索的邻居数（工具不覆盖时的回退）
  RIDGE     线性工具的岭系数

判定
------------------------------------------------
对每个参数，测 5 个取值（默认值 ±50% 及更宽），记录 ACC 与 oracle 花费。
  **敏感度 = 相对默认值的最大 |ΔACC|**
  D5 通过线：四个参数全部 < 0.05
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.realdata import RealClassStream, load_mnist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
P_DIM, N_CLASSES, POLLUTE, ROUND = 128, 10, 0.4, 50
SEEDS = [0, 1, 2, 3, 4]
DEFAULT = dict(tau=0.60, hilo=0.65, knn_k=5, ridge=1.0)
GRID = {
    "tau": [0.30, 0.45, 0.60, 0.75, 0.90],
    "hilo": [0.52, 0.58, 0.65, 0.72, 0.80],
    "knn_k": [2, 3, 5, 8, 12],
    "ridge": [0.3, 0.5, 1.0, 2.0, 3.0],
}
MN = load_mnist()
CONF, _ = weak_model_confusion()
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


class LinTool:
    __slots__ = ("W", "classes")

    def __init__(self, X, y, classes, ridge=1.0):
        self.classes = list(classes)
        A = np.hstack([X, np.ones((len(X), 1))])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        Ginv = np.linalg.inv(G)
        self.W = np.zeros((len(self.classes), A.shape[1]))
        for i, c in enumerate(self.classes):
            t = (y == c).astype(np.float64) * 2.0 - 1.0
            self.W[i] = Ginv @ (A.T @ t)

    def _proba(self, X):
        X = np.atleast_2d(X)
        A = np.hstack([X, np.ones((len(X), 1))])
        M = A @ self.W.T
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


class R24:
    def __init__(self, seed, tau, hilo, knn_k, ridge):
        self.tau, self.hilo, self.knn_k, self.ridge = tau, hilo, knn_k, ridge
        self.rng = np.random.default_rng(900 + seed)
        self.bx, self.by = [], []
        self.side = 0
        self.tool = None

    def refit(self):
        if len(self.bx) < 60:
            return
        X = np.array(self.bx); y = np.array(self.by)
        cls = sorted(set(y.tolist()))
        if len(cls) >= 2:
            self.tool = LinTool(X, y, cls, ridge=self.ridge)

    def round(self, xb, yb, truth_fn):
        n = len(xb)
        cls = sorted(set(yb.tolist()))
        if len(cls) >= 2 and n >= 10:
            P = loo_scores(xb, yb, cls, ridge=self.ridge)
            pred = np.array(cls)[P.argmax(axis=1)]
        else:
            pred = None
        if self.tool is not None:
            Pt = self.tool._proba(xb)
            tpred = np.array(self.tool.classes)[Pt.argmax(axis=1)]
            tcover = Pt.max(axis=1) >= self.tau
        else:
            tpred, tcover = None, np.zeros(n, dtype=bool)
        for i in range(n):
            x, y = xb[i], int(yb[i])
            a = (1.0 if int(tpred[i]) == y else 0.0) if (tpred is not None and tcover[i]) else None
            p = None if pred is None else (1.0 if pred[i] == y else 0.0)
            parts = []
            if a is not None:
                parts.append((0.6, a))
            if p is not None:
                parts.append((0.4 if a is not None else 1.0, p))
            sc = None
            if parts:
                tot = sum(w for w, _ in parts)
                sc = sum(w * v for w, v in parts) / max(tot, 1e-9)
            if sc is None or (1.0 - self.hilo) < sc < self.hilo:
                ok = truth_fn(x) == y
                self.side += 1
            else:
                ok = sc >= self.hilo
            if ok:
                self.bx.append(x); self.by.append(y)
        self.refit()

    def predict(self, X):
        out = np.zeros(len(X), dtype=int)
        rest = np.ones(len(X), dtype=bool)
        if self.tool is not None:
            Pt = self.tool._proba(X)
            ms = Pt.max(axis=1) >= self.tau
            if ms.any():
                out[ms] = np.array(self.tool.classes)[Pt.argmax(axis=1)][ms]
            rest = ~ms
        if rest.any() and self.bx:
            A = np.array(self.bx); ay = np.array(self.by)
            S = X[rest] @ A.T
            k = min(self.knn_k, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            out[rest] = np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])
        return out


def run_cfg(st, seed, cfg):
    R = R24(seed, cfg["tau"], cfg["hilo"], cfg["knn_k"], cfg["ridge"])
    accs = []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    return dict(ACC_final=accs[-1], side=R.side, kept=len(R.bx))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for pname, vals in GRID.items():
            for v in vals:
                cfg = dict(DEFAULT)
                cfg[pname] = v
                r = run_cfg(st, seed, cfg)
                recs.append(dict(param=pname, value=v, seed=seed, **r))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage24.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 100)
    print("阶段 24（G4）：超参敏感度（真实 MNIST / 污染 40% / 5 seeds；D5 通过线 = 敏感度 < 0.05）")
    print("=" * 100)
    print("%-8s %-9s %-16s %-14s %s" % ("参数", "取值", "ACC", "oracle 花费", "相对默认的 ΔACC"))
    verdict = {}
    for pname, vals in GRID.items():
        base = [r["ACC_final"] for r in recs if r["param"] == pname and r["value"] == DEFAULT[pname]]
        bacc = np.mean(base)
        worst = 0.0
        for v in vals:
            rs = [r for r in recs if r["param"] == pname and r["value"] == v]
            acc = np.mean([r["ACC_final"] for r in rs])
            sp = np.mean([r["side"] for r in rs])
            mark = "  <-默认" if abs(v - DEFAULT[pname]) < 1e-9 else ""
            d = acc - bacc
            if abs(v - DEFAULT[pname]) > 1e-9:
                worst = max(worst, abs(d))
            print("%-8s %-9s %.3f±%.3f      %-14.0f %+.4f%s"
                  % (pname, v, acc, np.std([r["ACC_final"] for r in rs]), sp, d, mark))
        verdict[pname] = worst
        print("%-8s %-9s 敏感度 = %.4f  -> %s" % (pname, "", worst, "通过" if worst < 0.05 else "不通过"))
        print()
    print("D5 判定：%s" % ("通过" if all(v < 0.05 for v in verdict.values()) else "不通过"))
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage24.json")))


if __name__ == "__main__":
    main()
