# -*- coding: utf-8 -*-
"""阶段 17：中立票「放置」—— 先筛确定数据，再处理中立票。

用户的想法
------------------------------------------------
上一轮（阶段 16）在真实数据上暴露出：降权把低可信度的锚变成一张 **0.5 的中立票**，
于是 score 频繁落进犹豫区，**每次都触发一次真值查询**（side 从 309 涨到 856，2.8 倍）。

用户的提法不是"弃权"，而是**放置**：
  1. 先把**确定**的候选筛完（锚有明确意见、或 peer 有明确意见）
  2. 等到确定的那批筛完，再回头处理中立票 —— 此时手里已经有**一批干净的确定样本**，
     用它做参照去判中立票，比在混合批次上瞎猜准得多，也省得多

三组
  one_phase    阶段 16 的加权做法：一遇到犹豫就花 side（基线）
  two_phase    **放置**：确定批先走，再用确定批当参照处理中立票，仍不确定才花 side
  two_phase_0  放置，但**完全不对中立票花 side**（看放置本身能走多远）

性能修复
------------------------------------------------
阶段 16 每个候选都要做一次留一岭回归（1206 秒/3 seeds）。这里改用**帽矩阵快捷式**：
一次求逆就得到整批的留一预测（ŷ_i^LOO = y_i - e_i/(1 - h_ii)）。
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
FORCE_PER_ROUND, SEEDS, KNN_K = 4, [0, 1, 2, 3, 4], 5
SPOT_FRAC = 0.15      # 对"确定批"的独立抽检比例
ERR_MAX = 0.20        # 抽检估计出的错误率超过它，整批取消当参照的资格
HILO = 0.65          # 分数高于它=确定收，低于 1-HILO=确定拒，中间=中立票
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


def ridge_fit(X, y, classes, ridge=1.0):
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    W = np.zeros((len(classes), A.shape[1]))
    for i, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        W[i] = Ginv @ (A.T @ t)
    return W, Ginv


def loo_scores(X, y, classes, ridge=1.0):
    """一次求逆得到整批的留一预测（帽矩阵快捷式）。返回 (n, n_classes)。"""
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    h = np.einsum("ij,jk,ik->i", A, Ginv, A)
    h = np.clip(h, 0.0, 0.999)
    P = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        w = Ginv @ (A.T @ t)
        e = t - A @ w
        P[:, ci] = t - e / (1.0 - h)
    return P


class LinTool:
    __slots__ = ("W", "classes", "cred")

    def __init__(self, X, y, classes, ridge=1.0):
        self.classes = list(classes)
        self.W, _ = ridge_fit(X, y, self.classes, ridge)
        self.cred = 1.0

    def _proba(self, X):
        A = np.hstack([np.atleast_2d(X), np.ones((len(np.atleast_2d(X)), 1))])
        M = A @ self.W.T
        M = M - M.max(axis=1, keepdims=True)
        E = np.exp(M)
        return E / E.sum(axis=1, keepdims=True)

    def covers(self, x):
        return float(self._proba(x)[0].max()) >= TAU

    def predict(self, X):
        return np.array(self.classes)[self._proba(X).argmax(axis=1)]


class R17:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(900 + seed)
        self.tools, self.accepted_x, self.accepted_y = [], [], []
        self.audit_x, self.audit_y = [], []
        self.side = 0
        self.parked_total = 0
        self.spot_queries = 0
        self.batches_rejected = 0
        self.verified_x, self.verified_y = [], []    # 只装"由真值验证过"的记忆（控制组用）
        self.parked_resolved = 0
        self.n_p1 = 0
        self.n_p2 = 0

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

    def gate_round(self, xb, yb, side_fn):
        n = len(xb)
        cls = sorted(set(yb.tolist()))
        P = loo_scores(xb, yb, cls) if len(cls) >= 2 and n >= 10 else None
        pred_i = np.array(cls)[P.argmax(axis=1)] if P is not None else None

        resolved = {}
        via_side = set()
        parked = []
        # ---- 第一段：确定数据先筛 ----
        for i in range(n):
            x, y = xb[i], int(yb[i])
            a = self.ch_anchor(x, y)
            p = None if pred_i is None or len(cls) < 2 else (1.0 if pred_i[i] == y else 0.0)
            sc = self.combine(a, p)
            if sc is None:
                parked.append(i); continue
            if sc >= HILO:
                resolved[i] = True
            elif sc <= 1.0 - HILO:
                resolved[i] = False
            else:
                parked.append(i)
        self.n_p1 += len(resolved)
        # ---- 独立校验：确定批到底可不可信？----
        # 10% 的抽检没法给单条背书，但能估出**这一批**的错误率；
        # 错误率超阈值就整批取消"当参照"的资格（这就是"独立校验"）。
        reliable = True
        if self.mode == "park_verified":
            acc_idx = [i for i, ok in resolved.items() if ok]
            if acc_idx:
                k = max(1, int(round(SPOT_FRAC * len(acc_idx))))
                pick = self.rng.choice(acc_idx, size=min(k, len(acc_idx)), replace=False)
                wrong = 0
                for i in pick:
                    i = int(i)
                    self.side += 1
                    self.audit_x.append(np.asarray(xb[i]).copy())
                    self.audit_y.append(side_fn(xb[i]))
                    self.spot_queries += 1
                    if self.audit_y[-1] != int(yb[i]):
                        wrong += 1
                if wrong / len(pick) > ERR_MAX:
                    reliable = False
                    self.batches_rejected += 1
        # ---- 第二段：用"确定收下"的那批当参照，处理中立票 ----
        ref_idx = [i for i, ok in resolved.items() if ok] if reliable else []
        ref_x = np.array([xb[i] for i in ref_idx]) if ref_idx else np.zeros((0, P_DIM))
        ref_y = np.array([yb[i] for i in ref_idx]) if ref_idx else np.zeros(0, dtype=int)
        if self.mode == "park_verified_mem":
            # 控制组：参照池只用"由真值验证过"的记忆 —— 构造一个真正独立的参照
            if self.verified_x:
                ref_x = np.vstack([ref_x, np.array(self.verified_x)])
                ref_y = np.concatenate([ref_y, np.array(self.verified_y)])
        elif len(self.accepted_x):
            ref_x = np.vstack([ref_x, np.array(self.accepted_x)])
            ref_y = np.concatenate([ref_y, np.array(self.accepted_y)])
        reftool = None
        if len(ref_y) >= 10 and len(set(ref_y.tolist())) >= 2:
            reftool = LinTool(ref_x, ref_y, sorted(set(ref_y.tolist())))

        self.parked_total += len(parked)
        for i in parked:
            x, y = xb[i], int(yb[i])
            if self.mode == "one_phase":
                # 基线（阶段 16 的加权做法）：一遇中立票就立刻花 side，不放置
                self.side += 1
                self.audit_x.append(np.asarray(x).copy())
                self.audit_y.append(side_fn(x))
                resolved[i] = (self.audit_y[-1] == y)
                via_side.add(i)
                self.n_p2 += 1
                continue
            a = self.ch_anchor(x, y)
            p2 = None
            if reftool is not None:
                p2 = 1.0 if int(reftool.predict(x)[0]) == y else 0.0
            sc2 = self.combine(a, p2)
            if sc2 is not None and (sc2 >= HILO or sc2 <= 1.0 - HILO):
                resolved[i] = sc2 >= HILO
                self.parked_resolved += 1
                self.n_p2 += 1
                continue
            if self.mode == "two_phase_0":
                resolved[i] = False
                continue
            # 放置到最后仍然中立，才花一次真值
            self.side += 1
            self.audit_x.append(np.asarray(x).copy())
            self.audit_y.append(side_fn(x))
            resolved[i] = (self.audit_y[-1] == y)
            via_side.add(i)
            self.n_p2 += 1
        for i in range(n):
            if resolved.get(i, False):
                self.accepted_x.append(xb[i])
                self.accepted_y.append(int(yb[i]))
                if i in via_side:
                    self.verified_x.append(xb[i])
                    self.verified_y.append(int(yb[i]))

    def run_audit(self):
        if not self.tools or len(self.audit_x) < AUDIT_MIN:
            return
        X = np.array(self.audit_x); Y = np.array(self.audit_y)
        for tl in self.tools:
            if len(tl.classes) < 2:
                continue
            P = tl._proba(X)
            agree = float((np.array(tl.classes)[P.argmax(axis=1)] == Y).mean())
            tl.cred = CRED_BLEND * tl.cred + (1.0 - CRED_BLEND) * agree

    def predict(self, X):
        use = [tl for tl in self.tools if tl.cred >= CRED_USE]
        out = np.zeros(len(X), dtype=int)
        cover = np.zeros(len(X))
        accv = np.zeros(len(X)); wsum = np.zeros(len(X))
        if use:
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


def run_arm(mode, st, seed):
    R = R17(mode, seed)
    t0 = st.tasks[0]
    R.inject_bad(t0.x_holdout[:200], t0.classes)
    accs, acc_c0 = [], []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            xb = task.x_train[s0:s0 + ROUND]
            yb = task.y_train[s0:s0 + ROUND]
            for i in R.rng.choice(len(xb), size=min(FORCE_PER_ROUND, len(xb)), replace=False):
                x = xb[int(i)]
                R.side += 1
                R.audit_x.append(x.copy()); R.audit_y.append(tf(x))
            R.gate_round(xb, yb, tf)
            R.run_audit()
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
                side=R.side, tools=len(R.tools), kept=len(R.accepted_x),
                min_cred=(min(tl.cred for tl in R.tools) if R.tools else float("nan")),
                parked=R.parked_total, parked_resolved=R.parked_resolved,
                spot=R.spot_queries, rejected_batches=R.batches_rejected)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode in ("one_phase", "two_phase", "park_verified", "park_verified_mem"):
            recs.append(run_arm(mode, st, seed))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage18.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 100)
    print("阶段 18：放置 + 参照批的独立校验（真实 MNIST / 污染 40% / 注入坏锚 / 5 seeds）")
    print("=" * 100)
    print("%-14s %-15s %-14s %-11s %-11s %s" % ("组", "总体 ACC", "任务0 类 ACC", "side 查询", "抽检查询", "被否的批 / 放置解决"))
    for mode in ("one_phase", "two_phase", "park_verified", "park_verified_mem"):
        rs = [r for r in recs if r["mode"] == mode]
        print("%-14s %.3f±%.3f       %.3f          %-11.0f %-11.0f %.1f / %.0f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.nanmean([r["ACC_task0"] for r in rs]), np.mean([r["side"] for r in rs]),
                 np.mean([r.get("spot", 0) for r in rs]),
                 np.mean([r.get("rejected_batches", 0) for r in rs]),
                 np.mean([r["parked_resolved"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage18.json")))


if __name__ == "__main__":
    main()
