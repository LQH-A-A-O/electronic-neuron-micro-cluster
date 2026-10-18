# -*- coding: utf-8 -*-
"""阶段 25（G5）：把 D1 对决搬到**表格数据**上 —— D3 真版。

为什么必须做
------------------------------------------------
22 个阶段的结论全部建立在 28x28 灰度图上（MNIST / Fashion-MNIST）。
"换一个数据集"和"换一种数据性质"是两回事 —— 前面只是换了第二张图。

这一轮换成 **Forest Covertype**：54 维表格特征（海拔/坡度/土壤类型/日照…），6 个均衡类别。
它和像素的差别是本质的：没有空间结构、特征异构、量纲各异。

测什么
------------------------------------------------
原封不动重跑阶段 20 的对决：**同等 oracle 预算下，门控 vs 朴素随机标注。**
唯一变化 = 数据。

  gate       门控：锚/peer 都明确就自己判，犹豫才花 oracle
  naive B    全部收下；其中随机 B 条用真值覆盖标签；B=50/轮 即全标注上界
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
DIM, N_CLASSES, N_TASK, CPT = 54, 6, 3, 2
POLLUTE, ROUND = 0.4, 50
N_TRAIN = int(os.environ.get("NT", "400"))
N_HOLD, N_TEST = 200, 200
TAU, KNN_K, HILO = 0.60, 5, 0.65
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [0, 5, 10, 20, 50]


def load_cov():
    g = lambda n: np.load(os.path.join(DATA, n))
    return (g("cov_tr.npy"), g("cov_tr_y.npy"), g("cov_va.npy"),
            g("cov_va_y.npy"), g("cov_te.npy"), g("cov_te_y.npy"))


class Task:
    pass


def build_tasks(seed):
    Xtr, ytr, Xva, yva, Xte, yte = load_cov()
    rng = np.random.default_rng(777 + seed)
    tasks = []
    for t in range(N_TASK):
        classes = [CPT * t + i for i in range(CPT)]
        def draw(X, y, n):
            parts = []
            for c in classes:
                ci = np.flatnonzero(y == c)
                parts.append(rng.choice(ci, size=max(1, n // len(classes)), replace=False))
            idx = np.concatenate(parts)
            rng.shuffle(idx)
            return X[idx].astype(np.float64), y[idx].copy()
        xt, yt = draw(Xtr, ytr, N_TRAIN)
        y_true = yt.copy()
        yt = inject(y_true, classes, POLLUTE, rng, "uniform")   # 表格数据用均匀翻转
        xh, yh = draw(Xva, yva, N_HOLD)
        xe, ye = draw(Xte, yte, N_TEST)
        k = Task(); k.task_id = t; k.classes = classes
        k.x_train, k.y_train, k.y_train_true = xt, yt, y_true
        k.x_holdout, k.y_holdout = xh, yh
        k.x_test, k.y_test = xe, ye
        tasks.append(k)
    return tasks


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


class R25:
    def __init__(self, mode, seed, B):
        self.mode, self.B = mode, B
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
            self.tool = LinTool(X, y, cls)

    def ch_anchor(self, x, y):
        if self.tool is None or self.tool._proba(x)[0].max() < TAU:
            return None
        return 1.0 if int(np.array(self.tool.classes)[self.tool._proba(x).argmax()]) == y else 0.0

    def round(self, xb, yb, truth_fn):
        n = len(xb)
        if self.mode == "gate":
            cls = sorted(set(yb.tolist()))
            if len(cls) >= 2 and n >= 10:
                P = loo_scores(xb, yb, cls)
                pred = np.array(cls)[P.argmax(axis=1)]
            else:
                pred = None
            if self.tool is not None:
                Pt = self.tool._proba(xb)
                tpred = np.array(self.tool.classes)[Pt.argmax(axis=1)]
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
                if sc is None or (1.0 - HILO) < sc < HILO:
                    ok = truth_fn(x) == y
                    self.side += 1
                else:
                    ok = sc >= HILO
                if ok:
                    self.bx.append(x); self.by.append(y)
        else:
            ys = np.array(yb).copy()
            B = n if self.mode == "full" else self.B
            if B > 0:
                for i in self.rng.choice(n, size=min(B, n), replace=False):
                    i = int(i)
                    ys[i] = truth_fn(xb[i]); self.side += 1
            self.bx.extend(list(xb)); self.by.extend(list(ys))
        self.refit()

    def predict(self, X):
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
        return out


def run_arm(mode, B, tasks, seed):
    R = R25(mode, seed, B)
    accs = []
    for t, task in enumerate(tasks):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        a = [float((R.predict(tasks[j].x_test) == tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    return dict(mode=mode, B=B, seed=seed, ACC_final=accs[-1], side=R.side, kept=len(R.bx))


ARMS = [("gate", -1), ("full", -1)] + [("naive", b) for b in BUDGETS]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        tasks = build_tasks(seed)
        for mode, B in ARMS:
            recs.append(run_arm(mode, B, tasks, seed))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage25.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 92)
    print("阶段 25（G5 / D3 真版）：表格数据 Covertype 上的 D1 对决（5 seeds）")
    print("=" * 92)
    print("%-20s %-16s %-14s %s" % ("组", "总体 ACC", "oracle 花费", "每单位花费的 ACC"))
    for mode, B in ARMS:
        rs = [r for r in recs if r["mode"] == mode and r["B"] == B]
        acc = np.mean([r["ACC_final"] for r in rs]); sd = np.std([r["ACC_final"] for r in rs])
        sp = np.mean([r["side"] for r in rs])
        name = {"gate": "gate(门控)", "full": "full(全标注上界)"}.get(mode, "naive B=%d/轮" % B)
        print("%-20s %.3f±%.3f       %-14.0f %.5f" % (name, acc, sd, sp, acc / max(sp, 1)))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage25.json")))


if __name__ == "__main__":
    main()
