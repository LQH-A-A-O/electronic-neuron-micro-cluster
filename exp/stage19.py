# -*- coding: utf-8 -*-
"""阶段 19：证据溯源 + 独立度传播 —— 「真值参照」到底要花多少钱？

阶段 18 的结论
------------------------------------------------
参照池如果只装"真值验证过的记忆"，坏锚现场能从 0.804 救到 0.904。
但那是个**刻意构造的上界**：它假设每条记忆都能追查来源、而且验证是免费的。

这一轮把它变成一个**有成本的机制**：
  - 每条被收下的记忆带一个 0~1 的 **独立度 indep**
        oracle 直接确认         -> 1.0
        由参照工具推出           -> LAMBDA × 参照池的平均独立度（打折传播）
        由共识（锚/peer）直接定   -> CONS_FLOOR（很低，因为它没有独立根据）
  - 参照池 = 独立度 >= INDEP_MIN 的条目
  - 每轮额外花 SEAL 次 oracle 去**给随机抽到的"自判"条目盖章**
        （盖过章并且 oracle 同意的 -> 独立度直接拉到 1.0）

要回答的问题
------------------------------------------------
**一小笔"盖章"预算，能不能撑起一个可用的独立参照？**
扫 SEAL ∈ {0, 2, 5, 10} 每轮（一轮 50 条候选 = 0% / 4% / 10% / 20% 的候选被盖章）。
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
AUDIT_MIN, CRED_BLEND, CRED_USE, TAU = 12, 0.5, 0.25, 0.60
FORCE_PER_ROUND, KNN_K = 4, 5
HILO = 0.65
LAMBDA, CONS_FLOOR, INDEP_MIN = 0.9, 0.15, 0.5
SEEDS = [0, 1, 2, 3, 4]
SEALS = [0, 2, 5, 10]
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
        Ginv = np.linalg.inv(G)
        self.W = np.zeros((len(self.classes), A.shape[1]))
        for i, c in enumerate(self.classes):
            t = (y == c).astype(np.float64) * 2.0 - 1.0
            self.W[i] = Ginv @ (A.T @ t)
        self.cred = 1.0

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


class R19:
    def __init__(self, mode, seed, seal):
        self.mode = mode
        self.seal = seal
        self.rng = np.random.default_rng(900 + seed)
        self.tools = []
        self.accepted_x, self.accepted_y, self.accepted_indep = [], [], []
        self.audit_x, self.audit_y = [], []
        self.side = 0
        self.parked = 0
        self.parked_resolved = 0
        self.sealed = 0

    def compile_tool(self):
        if len(self.accepted_x) < 60:
            return
        X = np.array(self.accepted_x); y = np.array(self.accepted_y)
        cls = sorted(set(y.tolist()))
        if len(cls) >= 2:
            self.tools.append(LinTool(X, y, cls))

    def inject_bad(self, X, classes):
        y = np.array(classes)[self.rng.integers(0, len(classes), len(X))]
        self.tools.append(LinTool(X, y, classes))

    def ch_anchor(self, x, y):
        votes = []
        for tl in self.tools:
            if tl.cred < CRED_USE or not tl.covers(x):
                continue
            v = 1.0 if int(tl.predict(x)[0]) == y else 0.0
            votes.append(tl.cred * v + (1.0 - tl.cred) * 0.5)
        return float(np.mean(votes)) if votes else None

    @staticmethod
    def combine(a, p):
        parts = []
        if a is not None:
            parts.append((0.6, a))
        if p is not None:
            parts.append((0.4 if a is not None else 1.0, p))
        if not parts:
            return None
        tot = sum(w for w, _ in parts)
        return sum(w * v for w, v in parts) / max(tot, 1e-9)

    def ref_pool(self):
        """参照池 = 独立度够高的记忆。"""
        if self.mode == "park":
            return np.array(self.accepted_x) if self.accepted_x else np.zeros((0, P_DIM)), \
                   np.array(self.accepted_y) if self.accepted_y else np.zeros(0, dtype=int)
        keep = [i for i, v in enumerate(self.accepted_indep) if v >= INDEP_MIN]
        if not keep:
            return np.zeros((0, P_DIM)), np.zeros(0, dtype=int)
        return np.array([self.accepted_x[i] for i in keep]), \
               np.array([self.accepted_y[i] for i in keep])

    def gate_round(self, xb, yb, side_fn):
        n = len(xb)
        cls = sorted(set(yb.tolist()))
        P = loo_scores(xb, yb, cls) if len(cls) >= 2 and n >= 10 else None
        pred_i = np.array(cls)[P.argmax(axis=1)] if P is not None else None
        resolved, src, parked = {}, {}, []
        for i in range(n):
            x, y = xb[i], int(yb[i])
            a = self.ch_anchor(x, y)
            p = None if pred_i is None or len(cls) < 2 else (1.0 if pred_i[i] == y else 0.0)
            sc = self.combine(a, p)
            if sc is None:
                parked.append(i); continue
            if sc >= HILO:
                resolved[i] = True; src[i] = "cons"
            elif sc <= 1.0 - HILO:
                resolved[i] = False; src[i] = "cons"
            else:
                parked.append(i)
        ref_x, ref_y = self.ref_pool()
        ref_indep = float(np.mean([v for v in self.accepted_indep if v >= INDEP_MIN])) \
            if self.mode != "park" and any(v >= INDEP_MIN for v in self.accepted_indep) else CONS_FLOOR
        reftool = LinTool(ref_x, ref_y, sorted(set(ref_y.tolist()))) \
            if len(ref_y) >= 10 and len(set(ref_y.tolist())) >= 2 else None
        self.parked += len(parked)
        for i in parked:
            x, y = xb[i], int(yb[i])
            if self.mode == "park":
                self.side += 1
                self.audit_x.append(np.asarray(x).copy()); self.audit_y.append(side_fn(x))
                resolved[i] = (self.audit_y[-1] == y); src[i] = "side"
                continue
            a = self.ch_anchor(x, y)
            p2 = None
            if reftool is not None:
                p2 = 1.0 if int(reftool.predict(x)[0]) == y else 0.0
            sc2 = self.combine(a, p2)
            if sc2 is not None and (sc2 >= HILO or sc2 <= 1.0 - HILO):
                resolved[i] = sc2 >= HILO; src[i] = "ref"; self.parked_resolved += 1
                continue
            self.side += 1
            self.audit_x.append(np.asarray(x).copy()); self.audit_y.append(side_fn(x))
            resolved[i] = (self.audit_y[-1] == y); src[i] = "side"
        # ---- 盖章：花 SEAL 次 oracle 给"自判"的条目做独立确认 ----
        indep = {}
        for i in range(n):
            if not resolved.get(i, False):
                continue
            if src.get(i) == "side":
                indep[i] = 1.0
            elif src.get(i) == "cons":
                indep[i] = CONS_FLOOR
            else:
                indep[i] = LAMBDA * ref_indep
        cand = [i for i in range(n) if resolved.get(i, False) and src.get(i) != "side"]
        if self.mode == "prov" and self.seal > 0 and cand:
            k = min(self.seal, len(cand))
            for i in self.rng.choice(cand, size=k, replace=False):
                i = int(i)
                self.side += 1
                self.audit_x.append(np.asarray(xb[i]).copy()); self.audit_y.append(side_fn(xb[i]))
                self.sealed += 1
                if self.audit_y[-1] == int(yb[i]):
                    indep[i] = 1.0
                else:
                    resolved[i] = False          # 盖章发现自判错了 -> 纠回
        for i in range(n):
            if resolved.get(i, False):
                self.accepted_x.append(xb[i]); self.accepted_y.append(int(yb[i]))
                self.accepted_indep.append(indep.get(i, CONS_FLOOR))

    def run_audit(self):
        if not self.tools or len(self.audit_x) < AUDIT_MIN:
            return
        X = np.array(self.audit_x); Y = np.array(self.audit_y)
        for tl in self.tools:
            if len(tl.classes) < 2:
                continue
            agree = float((np.array(tl.classes)[tl._proba(X).argmax(axis=1)] == Y).mean())
            tl.cred = CRED_BLEND * tl.cred + (1.0 - CRED_BLEND) * agree

    def predict(self, X):
        use = [tl for tl in self.tools if tl.cred >= CRED_USE]
        out = np.zeros(len(X), dtype=int)
        cover = np.zeros(len(X)); accv = np.zeros(len(X)); wsum = np.zeros(len(X))
        for tl in use:
            pr = tl.predict(X)
            ms = np.array([tl.covers(x) for x in X])
            cover += ms
            accv += np.where(ms, tl.cred * pr, 0.0)
            wsum += np.where(ms, tl.cred, 0.0)
        ok = cover > 0
        if ok.any():
            out[ok] = np.round(accv[ok] / np.maximum(wsum[ok], 1e-9)).astype(int)
        rest = ~ok
        if rest.any() and self.accepted_x:
            A = np.array(self.accepted_x); ay = np.array(self.accepted_y)
            S = X[rest] @ A.T
            k = min(KNN_K, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            out[rest] = np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])
        return out


def run_arm(mode, seal, st, seed):
    R = R19(mode, seed, seal)
    t0 = st.tasks[0]
    R.inject_bad(t0.x_holdout[:200], t0.classes)
    accs, acc_c0 = [], []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            xb = task.x_train[s0:s0 + ROUND]; yb = task.y_train[s0:s0 + ROUND]
            for i in R.rng.choice(len(xb), size=min(FORCE_PER_ROUND, len(xb)), replace=False):
                x = xb[int(i)]
                R.side += 1
                R.audit_x.append(x.copy()); R.audit_y.append(tf(x))
            R.gate_round(xb, yb, tf)
            R.run_audit(); R.compile_tool()
        a, c0 = [], []
        for j in range(t + 1):
            pred = R.predict(st.tasks[j].x_test)
            hit = pred == st.tasks[j].y_test
            a.append(float(hit.mean()))
            if j == 0:
                c0.append(float(hit.mean()))
        accs.append(float(np.mean(a))); acc_c0.append(float(np.mean(c0)))
    return dict(mode=mode, seal=seal, seed=seed, ACC_final=accs[-1], ACC_task0=acc_c0[-1],
                side=R.side, sealed=R.sealed, kept=len(R.accepted_x),
                high_indep=sum(1 for v in R.accepted_indep if v >= INDEP_MIN))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode, seal in (("park", 0), ("prov", 0), ("prov", 2), ("prov", 5), ("prov", 10)):
            recs.append(run_arm(mode, seal, st, seed))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage19.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 104)
    print("阶段 19：证据溯源 + 独立度传播（真实 MNIST / 5 seeds）")
    print("=" * 104)
    print("%-18s %-16s %-16s %-12s %-12s %s" % ("组", "总体 ACC", "任务0 类 ACC", "盖章次数", "side 查询", "高独立度记忆"))
    for mode, seal in (("park", 0), ("prov", 0), ("prov", 2), ("prov", 5), ("prov", 10)):
        rs = [r for r in recs if r["mode"] == mode and r["seal"] == seal]
        name = "%s(seal=%d)" % (mode, seal)
        print("%-18s %.3f±%.3f       %.3f          %-12.0f %-12.0f %.0f"
              % (name, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.nanmean([r["ACC_task0"] for r in rs]), np.mean([r["sealed"] for r in rs]),
                 np.mean([r["side"] for r in rs]), np.mean([r["high_indep"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage19.json")))


if __name__ == "__main__":
    main()
