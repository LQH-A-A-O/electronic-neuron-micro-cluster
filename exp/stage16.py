# -*- coding: utf-8 -*-
"""阶段 16：把「坏锚 -> 审计 -> 撤销 vs 降权」整条链路搬到真实数据（MNIST）。

前面 11~15 全部在合成多项式域 + 人为注入的坏锚上做。这一阶段只换域，机制不动。

真实数据上"锚"怎么定义
------------------------------------------------
合成域里锚 = 一条局部多项式 + 一个盒子（定义域）。
MNIST 没有盒子，所以锚 = **一个线性判别器 + 一个置信度门槛**：
只有当它在 x 上的 softmax 置信度超过门槛，才算"它管得着 x"（covers）。

坏锚怎么注入
------------------------------------------------
用**打乱标签**的训练集拟合一个线性判别器 —— 它在任务 0 的两个类上会**自信地错**，
和合成域里"用错规则的锚"是同一个东西。

六组
------------------------------------------------
  hard        审计判错就**撤销**锚（阶段 13/14）
  weighted    审计结果转成 0~1 **可信度**，锚只有发言权重（阶段 15）
  外加 no_anchor（不注入坏锚）作为"本来能拿多少"的上界
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.learners import NaiveLearner, ReservoirLearner
from exp.realdata import RealClassStream, load_mnist
from exp.run import run_trajectory, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
P_DIM = 128
N_CLASSES = 10
POLLUTE = 0.4
ROUND = 50
AUDIT_MIN = 12
CRED_BLEND = 0.5
CRED_USE = 0.25
TAU = 0.60
FORCE_PER_ROUND = 4
SEEDS = [0, 1, 2]
KNN_K = 5
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
    __slots__ = ("W", "classes", "cred")

    def __init__(self, X, y, classes, ridge=1.0):
        self.classes = list(classes)
        A = np.hstack([X, np.ones((len(X), 1))])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        self.W = np.zeros((len(self.classes), A.shape[1]))
        for i, c in enumerate(self.classes):
            t = (y == c).astype(np.float64) * 2.0 - 1.0
            self.W[i] = np.linalg.solve(G, A.T @ t)
        self.cred = 1.0

    def _proba(self, X):
        A = np.hstack([X, np.ones((len(X), 1))])
        M = A @ self.W.T
        M = M - M.max(axis=1, keepdims=True)
        E = np.exp(M)
        return E / E.sum(axis=1, keepdims=True)

    def covers(self, x):
        return float(self._proba(np.asarray(x).reshape(1, -1))[0].max()) >= TAU

    def predict(self, X):
        return np.array(self.classes)[self._proba(np.atleast_2d(X)).argmax(axis=1)]


class R16:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(900 + seed)
        self.tools = []
        self.accepted_x, self.accepted_y = [], []
        self.audit_x, self.audit_y = [], []
        self.side = 0
        self.demoted = 0
        self.cred_updates = 0
        self.kept_all = []

    # ---------------- 编译 / 注入 ----------------
    def compile_tool(self):
        if len(self.accepted_x) < 60:
            return
        X = np.array(self.accepted_x)
        y = np.array(self.accepted_y)
        cls = sorted(set(y.tolist()))
        if len(cls) < 2:
            return
        self.tools.append(LinTool(X, y, cls))

    def inject_bad(self, X, classes):
        y = np.array(classes)[self.rng.integers(0, len(classes), len(X))]
        self.tools.append(LinTool(X, y, classes))

    # ---------------- 三条通道 ----------------
    def ch_anchor(self, x, y):
        votes = []
        for tl in self.tools:
            if not tl.covers(x):
                continue
            v = 1.0 if int(tl.predict(x)[0]) == y else 0.0
            if self.mode == "hard":
                votes.append(v)
            else:
                w = tl.cred
                votes.append(w * v + (1.0 - w) * 0.5)
        return float(np.mean(votes)) if votes else None

    def ch_peer(self, x, y, px, py):
        if len(px) < 10:
            return None
        cls = sorted(set(py.tolist()))
        if len(cls) < 2:
            return None
        tl = LinTool(px, py, cls, ridge=1.0)
        return 1.0 if int(tl.predict(x)[0]) == y else 0.0

    # ---------------- 审计 ----------------
    def run_audit(self, side_fn):
        if not self.tools or len(self.audit_x) < AUDIT_MIN:
            return
        X = np.array(self.audit_x)
        Y = np.array(self.audit_y)
        kept = []
        for tl in self.tools:
            if len(tl.classes) < 2:
                kept.append(tl); continue
            P = tl._proba(X)
            agree = float((np.array(tl.classes)[P.argmax(axis=1)] == Y).mean())
            if self.mode == "hard":
                self.demoted += 1
                self.cred_updates += 1
                if agree >= 0.8:
                    kept.append(tl)
            else:
                tl.cred = CRED_BLEND * tl.cred + (1.0 - CRED_BLEND) * agree
                self.cred_updates += 1
                kept.append(tl)
        self.tools = kept

    # ---------------- 判定 ----------------
    def predict(self, X):
        if self.tools:
            if self.mode == "hard":
                use = self.tools
            else:
                use = [tl for tl in self.tools if tl.cred >= CRED_USE]
            if use:
                acc = np.zeros(len(X))
                wsum = np.zeros(len(X))
                for tl in use:
                    w = 1.0 if self.mode == "hard" else max(tl.cred, 1e-6)
                    acc += w * tl.predict(X)
                    wsum += w
                # 只有多数锚"管得着"的样本才用锚，否则退回记忆
                cover = np.zeros(len(X))
                for tl in use:
                    cover += np.array([1.0 if tl.covers(x) else 0.0 for x in X])
                ok = cover > 0
                out = np.zeros(len(X), dtype=int)
                if ok.any():
                    out[ok] = np.round(acc[ok] / np.maximum(wsum[ok], 1e-9)).astype(int)
                rest = ~ok
                if rest.any() and self.accepted_x:
                    A = np.array(self.accepted_x); ay = np.array(self.accepted_y)
                    S = X[rest] @ A.T
                    k = min(KNN_K, S.shape[1])
                    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
                    out[rest] = np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])
                return out
        if not self.accepted_x:
            return np.zeros(len(X), dtype=int)
        A = np.array(self.accepted_x); ay = np.array(self.accepted_y)
        S = X @ A.T
        k = min(KNN_K, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        return np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])


def run_arm(mode, st, seed, inject_bad):
    R = R16(mode, seed)
    if inject_bad:
        t0 = st.tasks[0]
        R.inject_bad(t0.x_holdout[:200], t0.classes)
    accs, acc_c0 = [], []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            xb = task.x_train[s0:s0 + ROUND]
            yb = task.y_train[s0:s0 + ROUND]
            forced = set(R.rng.choice(len(xb), size=min(FORCE_PER_ROUND, len(xb)), replace=False).tolist())
            for i in range(len(xb)):
                x, y = xb[i], int(yb[i])
                if i in forced:
                    yt = tf(x); R.side += 1
                    R.audit_x.append(x.copy()); R.audit_y.append(yt)
                a = R.ch_anchor(x, y)
                p = R.ch_peer(x, y, np.delete(xb, i, axis=0), np.delete(yb, i))
                if a is None and p is None:
                    yt = tf(x); R.side += 1
                    R.audit_x.append(x.copy()); R.audit_y.append(yt)
                    ok = yt == y
                else:
                    parts = []
                    if a is not None:
                        parts.append((0.6, a))
                    if p is not None:
                        parts.append((0.4 if a is not None else 1.0, p))
                    tot = sum(w for w, _ in parts)
                    sc = sum(w * v for w, v in parts) / max(tot, 1e-9)
                    if 0.35 <= sc <= 0.65:
                        yt = tf(x); R.side += 1
                        R.audit_x.append(x.copy()); R.audit_y.append(yt)
                        ok = yt == y
                    else:
                        ok = sc >= 0.5
                if ok:
                    R.accepted_x.append(x); R.accepted_y.append(y)
            R.run_audit(tf)
            R.compile_tool()
        a, c0 = [], []
        for j in range(t + 1):
            pred = R.predict(st.tasks[j].x_test)
            hit = pred == st.tasks[j].y_test
            a.append(float(hit.mean()))
            if j == 0:
                c0.append(float(hit.mean()))
        accs.append(float(np.mean(a)))
        acc_c0.append(float(np.mean(c0)))
    return dict(mode=mode, seed=seed, ACC_final=accs[-1], ACC_task0=acc_c0[-1],
                side=R.side, demoted=R.demoted, tools=len(R.tools),
                min_cred=(min(tl.cred for tl in R.tools) if R.tools else float("nan")),
                kept=len(R.accepted_x))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode, inj in (("no_anchor", False), ("hard", True), ("weighted", True)):
            recs.append(run_arm(mode, st, seed, inj))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
        for arm in ("B_replay", "A_naive"):
            st2 = make_stream(seed)
            L = NaiveLearner(P_DIM, seed=seed) if arm == "A_naive" else \
                ReservoirLearner(P_DIM, seed=seed, capacity=600)
            acc, led = run_trajectory(L, st2, steps_per_task=400)
            recs.append(dict(mode=arm, seed=seed, ACC_final=summarize(acc, led)["ACC_final"],
                             ACC_task0=float("nan"), side=0, demoted=0, tools=0,
                             min_cred=float("nan"), kept=0))

    json.dump(recs, open(os.path.join(OUT, "results_stage16.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 100)
    print("阶段 16：真实 MNIST 上的「坏锚 -> 审计 -> 撤销 vs 降权」（3 seeds）")
    print("=" * 100)
    print("%-12s %-16s %-16s %-12s %-12s %s" % ("组", "总体 ACC", "任务0 类 ACC", "撤销锚数", "最低可信度", "side 查询"))
    for mode in ("no_anchor", "hard", "weighted", "B_replay", "A_naive"):
        rs = [r for r in recs if r["mode"] == mode]
        print("%-12s %.3f±%.3f        %.3f           %-12.1f %-12.3f %.0f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.nanmean([r["ACC_task0"] for r in rs]),
                 np.mean([r["demoted"] for r in rs]), np.nanmean([r["min_cred"] for r in rs]),
                 np.mean([r["side"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage16.json")))


if __name__ == "__main__":
    main()
