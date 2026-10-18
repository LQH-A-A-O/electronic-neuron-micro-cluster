# -*- coding: utf-8 -*-
"""阶段 22（G2）：资源账 —— D6。

21 个阶段里从没测过的东西：内存、每样本延迟、更新成本、吞吐。
真实系统第一眼看的就是这些，而不是 ACC 的第三位小数。

测什么
------------------------------------------------
  1. 在线观测路径：每收一条样本，门控判定 + 可能花一次 oracle 的**系统自身**耗时
     （oracle 本身的时间不算在内 —— 那在现实里是人或外部服务，成本另计）
  2. 预测路径：每个查询的耗时（这是线上真正的热路径）
  3. 更新成本：每次巩固 + 重编译的耗时
  4. 内存：库字节数 + 工具参数量 + tracemalloc 峰值
  5. 规模效应：库从几十条涨到两千条时，预测延迟怎么变

诚实前提
------------------------------------------------
这是**纯 Python 原型**，没有任何优化。测出来的绝对值不代表产品，但**相对关系**
（门控 vs 朴素、随规模的走势）是有意义的，而且"要不要动手优化"这个判断需要这些数。
"""
from __future__ import annotations

import json, os, sys, time, tracemalloc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.realdata import RealClassStream, load_mnist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
P_DIM, N_CLASSES, POLLUTE, ROUND = 128, 10, 0.4, 50
TAU, KNN_K, HILO = 0.60, 5, 0.65
SEEDS = [0, 1, 2]
BUDGETS = [0, 10]
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

    def covers(self, x):
        return float(self._proba(x)[0].max()) >= TAU

    def predict(self, X):
        return np.array(self.classes)[self._proba(X).argmax(axis=1)]


def loo_scores(X, y, classes, ridge=1.0):
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    h = np.clip(np.sum((A @ Ginv) * A, axis=1), 0.0, 0.999)   # 走 BLAS，比 einsum 快得多
    P = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        w = Ginv @ (A.T @ t)
        P[:, ci] = t - (t - A @ w) / (1.0 - h)
    return P


class R22:
    def __init__(self, mode, seed, B):
        self.mode, self.B = mode, B
        self.rng = np.random.default_rng(900 + seed)
        self.bx, self.by = [], []
        self.side = 0
        self.tool = None
        self.t_observe = 0.0     # 累计（不含 oracle）
        self.t_oracle = 0.0      # 累计（含 oracle 的调用本身）
        self.t_predict = 0.0
        self.t_refit = 0.0
        self.t_loo = 0.0
        self.n_predict = 0
        self.n_observe = 0
        self.n_refit = 0
        self.scaling = []

    def refit(self):
        t0 = time.perf_counter()
        if len(self.bx) >= 60:
            X = np.array(self.bx); y = np.array(self.by)
            cls = sorted(set(y.tolist()))
            if len(cls) >= 2:
                self.tool = LinTool(X, y, cls)
        self.t_refit += time.perf_counter() - t0
        self.n_refit += 1

    def ch_anchor(self, x, y):
        if self.tool is None or not self.tool.covers(x):
            return None
        return 1.0 if int(self.tool.predict(x)[0]) == y else 0.0

    def round(self, xb, yb, truth_fn):
        n = len(xb)
        t0 = time.perf_counter()
        if self.mode == "gate":
            cls = sorted(set(yb.tolist()))
            if len(cls) >= 2 and n >= 10:
                tl = time.perf_counter()
                P = loo_scores(xb, yb, cls)
                self.t_loo += time.perf_counter() - tl
            else:
                P = None
            pred = np.array(cls)[P.argmax(axis=1)] if P is not None else None
            # 批处理：工具对整个 round 一次算完，别在循环里逐条调用 numpy
            if self.tool is not None:
                Pt = self.tool._proba(xb)
                tcls = np.array(self.tool.classes)
                tpred = tcls[Pt.argmax(axis=1)]
                tcover = Pt.max(axis=1) >= TAU
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
                self.t_observe += time.perf_counter() - t0
                if sc is None or 0.35 < sc < HILO:
                    ts = time.perf_counter()
                    ok = truth_fn(x) == y
                    self.t_oracle += time.perf_counter() - ts
                    self.side += 1
                else:
                    ok = sc >= HILO
                if ok:
                    self.bx.append(x); self.by.append(y)
                t0 = time.perf_counter()
        else:
            ys = np.array(yb).copy()
            if self.B > 0:
                for i in self.rng.choice(n, size=min(self.B, n), replace=False):
                    i = int(i)
                    ts = time.perf_counter()
                    ys[i] = truth_fn(xb[i])
                    self.t_oracle += time.perf_counter() - ts
                    self.side += 1
            self.bx.extend(list(xb)); self.by.extend(list(ys))
        self.t_observe += time.perf_counter() - t0
        self.n_observe += n
        self.refit()

    def predict(self, X):
        t0 = time.perf_counter()
        out = np.zeros(len(X), dtype=int)
        rest = np.ones(len(X), dtype=bool)
        if self.tool is not None:
            Pt = self.tool._proba(X)
            ms = Pt.max(axis=1) >= TAU
            if ms.any():
                out[ms] = np.array(self.tool.classes)[Pt.argmax(axis=1)][ms]
            rest = ~ms
        if rest.any() and self.bx:
            A = np.array(self.bx); ay = np.array(self.by)
            S = X[rest] @ A.T
            k = min(KNN_K, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            out[rest] = np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])
        self.t_predict += time.perf_counter() - t0
        self.n_predict += len(X)
        return out

    def mem_bytes(self):
        b = 0
        for v in self.bx:
            b += np.asarray(v).nbytes
        b += np.asarray(self.by).nbytes if self.by else 0
        if self.tool is not None:
            b += self.tool.W.nbytes
        return b


def run_arm(mode, B, st, seed):
    R = R22(mode, seed, B)
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        # 规模效应：每个任务结束后测一次单样本预测延迟
        xq = task.x_test
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            R.predict(xq)
            best = min(best, (time.perf_counter() - t0) / len(xq))
        R.scaling.append(dict(task=t, buf=len(R.bx), us_per_sample=best * 1e6))
    # 峰值分配单独测一次：tracemalloc 会让每次分配慢好几倍，
    # 一直开着会把"观测延迟"测成假数字（这个坑本轮真踩了）。
    tracemalloc.start()
    R.predict(st.tasks[-1].x_test)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return dict(mode=mode, B=B, seed=seed, side=R.side, kept=len(R.bx),
                mem_bytes=R.mem_bytes(), peak_alloc_mb=peak / 1e6,
                us_observe=R.t_observe / max(R.n_observe, 1) * 1e6,
                us_oracle=R.t_oracle / max(R.side, 1) * 1e6,
                us_predict=R.t_predict / max(R.n_predict, 1) * 1e6,
                us_loo=R.t_loo / max(R.n_observe, 1) * 1e6,
                ms_refit=R.t_refit / max(R.n_refit, 1) * 1e3,
                refits=R.n_refit, scaling=R.scaling)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        recs.append(run_arm("gate", -1, st, seed))
        for B in BUDGETS:
            recs.append(run_arm("naive", B, st, seed))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage22.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 108)
    print("阶段 22（G2）：资源账（真实 MNIST / 污染 40% / 3 seeds；纯 Python 原型，未优化）")
    print("=" * 108)
    print("%-14s %-10s %-11s %-13s %-11s %-13s %-12s %-11s %s"
          % ("组", "库条数", "库字节", "观测 µs/条", "其中LOO", "预测 µs/条", "重编译 ms", "峰值MB", "oracle"))
    for mode, B in [("gate", -1), ("naive", 0), ("naive", 10)]:
        rs = [r for r in recs if r["mode"] == mode and r["B"] == B]
        name = "gate(门控)" if mode == "gate" else "naive B=%d/轮" % B
        print("%-14s %-10.0f %-11.0f %-13.2f %-11.2f %-13.2f %-12.3f %-11.1f %.0f"
              % (name, np.mean([r["kept"] for r in rs]), np.mean([r["mem_bytes"] for r in rs]),
                 np.mean([r["us_observe"] for r in rs]), np.mean([r.get("us_loo", 0) for r in rs]),
                 np.mean([r["us_predict"] for r in rs]),
                 np.mean([r["ms_refit"] for r in rs]), np.mean([r["peak_alloc_mb"] for r in rs]),
                 np.mean([r["side"] for r in rs])))
    print()
    print("门控组的规模效应（库越大，预测越慢？）")
    print("-" * 108)
    print("%-10s %-14s %s" % ("任务", "库条数", "预测 µs/样本"))
    rs = [r for r in recs if r["mode"] == "gate"]
    for k in range(5):
        bufs = [r["scaling"][k]["buf"] for r in rs]
        us = [r["scaling"][k]["us_per_sample"] for r in rs]
        print("%-10d %-14.0f %.2f" % (k, np.mean(bufs), np.mean(us)))
    print()
    print("吞吐（按单样本预测延迟折算，单核）")
    up = np.mean([r["us_predict"] for r in rs])
    print("  门控组 %.2f µs/查询 -> 约 %.0f 查询/秒/核" % (up, 1e6 / up))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage22.json")))


if __name__ == "__main__":
    main()
