# -*- coding: utf-8 -*-
"""阶段 20（落地标准 L2）：同等 oracle 预算下，门控 vs 朴素随机标注。

这是可能**一票否决整个方向**的实验
------------------------------------------------
前面 19 个阶段的对手都是"无门控回放"（B_replay = 0.451），那是个很弱的对手。
真实的对手是：**给我同样的 oracle 预算，我随机抽一批样本标注，然后监督学习。**

如果朴素做法在同等花费下赢过门控，那这套架构就没有存在的理由 ——
它花的钱一样多，做的判断还不如随机花钱。

统一预测器（所有组共用，唯一变量是"每轮往库里放什么、花多少钱"）
------------------------------------------------
    当前工具（每轮用库里的数据重拟合一个线性判别器）
    工具没覆盖的样本 -> 退回 kNN(k=5)

各组
    gate          我们的门控：锚/peer 都明确就自己判，犹豫才花 oracle
    naive_B       全部收下；其中随机 B 条用真值覆盖标签（B ∈ {0,5,10,20,50}）
                  B=50 就是"每条都标"的上界
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
TAU, KNN_K, HILO = 0.60, 5, 0.65
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [0, 5, 10, 20, 50]
DS = os.environ.get("DS", "mnist")


def load_ds():
    d = os.path.join(ROOT, "data")
    p = "fmnist" if DS == "fmnist" else "mnist"
    return (np.load(os.path.join(d, p + "_train.npy")).astype(np.float32),
            np.load(os.path.join(d, p + "_train_y.npy")).astype(np.int64),
            np.load(os.path.join(d, p + "_test.npy")).astype(np.float32),
            np.load(os.path.join(d, p + "_test_y.npy")).astype(np.int64))


MN = load_ds()
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
    h = np.clip(np.einsum("ij,jk,ik->i", A, Ginv, A), 0.0, 0.999)
    P = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        w = Ginv @ (A.T @ t)
        P[:, ci] = t - (t - A @ w) / (1.0 - h)
    return P


class R20:
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
        if self.tool is None or not self.tool.covers(x):
            return None
        return 1.0 if int(self.tool.predict(x)[0]) == y else 0.0

    def round(self, xb, yb, truth_fn):
        if self.mode == "gate":
            cls = sorted(set(yb.tolist()))
            P = loo_scores(xb, yb, cls) if len(cls) >= 2 and len(xb) >= 10 else None
            pred = np.array(cls)[P.argmax(axis=1)] if P is not None else None
            for i in range(len(xb)):
                x, y = xb[i], int(yb[i])
                a = self.ch_anchor(x, y)
                p = None if pred is None else (1.0 if pred[i] == y else 0.0)
                parts = []
                if a is not None:
                    parts.append((0.6, a))
                if p is not None:
                    parts.append((0.4 if a is not None else 1.0, p))
                if not parts:
                    sc = None
                else:
                    tot = sum(w for w, _ in parts)
                    sc = sum(w * v for w, v in parts) / max(tot, 1e-9)
                if sc is None or 0.35 < sc < HILO:
                    ok = truth_fn(x) == y
                    self.side += 1
                else:
                    ok = sc >= HILO
                if ok:
                    self.bx.append(x); self.by.append(y)
        else:
            ys = np.array(yb).copy()
            if self.B > 0:
                k = min(self.B, len(xb))
                for i in self.rng.choice(len(xb), size=k, replace=False):
                    i = int(i)
                    ys[i] = truth_fn(xb[i]); self.side += 1
            self.bx.extend(list(xb)); self.by.extend(list(ys))
        self.refit()

    def predict(self, X):
        out = np.zeros(len(X), dtype=int)
        rest = np.ones(len(X), dtype=bool)
        if self.tool is not None:
            ms = np.array([self.tool.covers(x) for x in X])
            if ms.any():
                out[ms] = self.tool.predict(X[ms])
            rest = ~ms
        if rest.any() and self.bx:
            A = np.array(self.bx); ay = np.array(self.by)
            S = X[rest] @ A.T
            k = min(KNN_K, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            out[rest] = np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])
        return out


def run_arm(mode, B, st, seed):
    R = R20(mode, seed, B)
    accs = []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    return dict(mode=mode, B=B, seed=seed, ACC_final=accs[-1], side=R.side, kept=len(R.bx))


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
    out_name = "results_stage20.json" if DS == "mnist" else "results_stage20_%s.json" % DS
    json.dump(recs, open(os.path.join(OUT, out_name), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 96)
    print("阶段 20：同等 oracle 预算下，门控 vs 朴素随机标注（数据集 = %s / 污染 40%% / 5 seeds）" % DS)
    print("=" * 96)
    print("%-16s %-16s %-16s %-14s %s" % ("组", "总体 ACC", "oracle 花费", "每单位花费的 ACC", "库里条数"))
    for mode, B in [("gate", -1)] + [("naive", b) for b in BUDGETS]:
        rs = [r for r in recs if r["mode"] == mode and r["B"] == B]
        acc = np.mean([r["ACC_final"] for r in rs]); sd = np.std([r["ACC_final"] for r in rs])
        sp = np.mean([r["side"] for r in rs])
        name = "gate(门控)" if mode == "gate" else "naive B=%d/轮" % B
        print("%-16s %.3f±%.3f       %-16.0f %-14.5f %.0f"
              % (name, acc, sd, sp, acc / max(sp, 1), np.mean([r["kept"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, out_name)))


if __name__ == "__main__":
    main()
