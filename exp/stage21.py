# -*- coding: utf-8 -*-
"""阶段 21（G1）：对上主动学习基线 —— D1 的最终审判。

为什么必须做
------------------------------------------------
阶段 20 的朴素对手是"**随机**抽样本去标注"。它赢了 84%。
但主动学习（active learning）正是干这个的：**用模型的不确定性挑最值得标的样本**。
如果一个熵采样基线在同等预算下追上门控，那 D1 就不成立 —— 门控没有必要存在。

设计
------------------------------------------------
一切沿用阶段 20：同一个流、同一个预测器（线性判别器 + kNN 回退）、同一批 seeds。
唯一变量 = **每轮花 B 次 oracle 去标哪些样本**：

  gate        门控：锚/peer 都明确就自己判，犹豫才花钱
  al_random   随机挑 B 条（阶段 20 的对手）
  al_entropy  挑**预测熵最大**的 B 条
  al_margin   挑**top1-top2 间隔最小**的 B 条
  full        B=50/轮，全部标注（上界）
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
BUDGETS = [5, 10, 20]
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
    h = np.clip(np.einsum("ij,jk,ik->i", A, Ginv, A), 0.0, 0.999)
    P = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        w = Ginv @ (A.T @ t)
        P[:, ci] = t - (t - A @ w) / (1.0 - h)
    return P


class R21:
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

    # ---- 主动学习的不确定性打分 ----
    def uncertainty(self, X):
        if self.tool is not None:
            P = self.tool._proba(X)
            cls = np.array(self.tool.classes)
            # 只保留已知类别的概率质量
            ent = -np.sum(P * np.log(np.maximum(P, 1e-12)), axis=1)
            ps = np.sort(P, axis=1)
            margin = ps[:, -1] - ps[:, -2]
            return ent, margin
        # 还没有工具 -> 用 kNN 投票分布
        n = len(X)
        ent = np.full(n, np.log(N_CLASSES))
        margin = np.zeros(n)
        if self.bx:
            A = np.array(self.bx); ay = np.array(self.by)
            S = X @ A.T
            k = min(KNN_K, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            for i, r in enumerate(idx):
                cnt = np.bincount(ay[r], minlength=N_CLASSES).astype(float)
                p = cnt / cnt.sum()
                ent[i] = -np.sum(p * np.log(np.maximum(p, 1e-12)))
                ps = np.sort(p)
                margin[i] = ps[-1] - ps[-2]
        return ent, margin

    def round(self, xb, yb, truth_fn):
        n = len(xb)
        if self.mode == "gate":
            cls = sorted(set(yb.tolist()))
            P = loo_scores(xb, yb, cls) if len(cls) >= 2 and n >= 10 else None
            pred = np.array(cls)[P.argmax(axis=1)] if P is not None else None
            for i in range(n):
                x, y = xb[i], int(yb[i])
                a = self.ch_anchor(x, y)
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
                if sc is None or 0.35 < sc < HILO:
                    ok = truth_fn(x) == y
                    self.side += 1
                else:
                    ok = sc >= HILO
                if ok:
                    self.bx.append(x); self.by.append(y)
        else:
            ys = np.array(yb).copy()
            B = self.B
            if self.mode == "full":
                B = n
            if B > 0:
                if self.mode == "al_random":
                    pick = self.rng.choice(n, size=min(B, n), replace=False)
                elif self.mode == "al_entropy":
                    ent, _ = self.uncertainty(xb)
                    pick = np.argsort(-ent)[:B]
                elif self.mode == "al_margin":
                    _, mg = self.uncertainty(xb)
                    pick = np.argsort(mg)[:B]
                else:
                    pick = np.arange(min(B, n))
                for i in pick:
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
    R = R21(mode, seed, B)
    accs = []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    return dict(mode=mode, B=B, seed=seed, ACC_final=accs[-1], side=R.side, kept=len(R.bx))


ARMS = [("gate", -1), ("full", -1)] + [(m, b) for m in ("al_random", "al_entropy", "al_margin")
                                       for b in BUDGETS]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode, B in ARMS:
            recs.append(run_arm(mode, B, st, seed))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage21.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 96)
    print("阶段 21（G1）：门控 vs 主动学习（真实 MNIST / 污染 40% / 5 seeds）")
    print("=" * 96)
    print("%-18s %-16s %-14s %s" % ("组", "总体 ACC", "oracle 花费", "每单位花费的 ACC"))
    for mode, B in ARMS:
        rs = [r for r in recs if r["mode"] == mode and r["B"] == B]
        acc = np.mean([r["ACC_final"] for r in rs]); sd = np.std([r["ACC_final"] for r in rs])
        sp = np.mean([r["side"] for r in rs])
        name = {"gate": "gate(门控)", "full": "full(全标注上界)"}.get(mode, "%s B=%d/轮" % (mode, B))
        print("%-18s %.3f±%.3f       %-14.0f %.5f" % (name, acc, sd, sp, acc / max(sp, 1)))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage21.json")))


if __name__ == "__main__":
    main()
