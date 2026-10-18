# -*- coding: utf-8 -*-
"""阶段 8：把 λ 从「经验数量」改成「经验可信度」—— 回滚 + 锚定的交叉实验。

阶段 7 留下的问题
------------------------------------------------
回滚把延迟标签从 0.125 救到 0.471，但温备里永远挂着一批"待裁判"的条目（污染率 0.41），
而且回滚本身分不出"已经转正的经验"和"还挂着待审的经验" —— 它一视同仁。

阶段 8 的假设
------------------------------------------------
把这两类经验区分开：λ(x) 只由**已转正**的经验量决定，待裁判的条目降权。

    p = λ(x)·p_warm + (1-λ(x))·p_cold
    λ = n_verified(x) / (n_verified(x) + k0)

三组对照（全部 delay1 + 回滚，同一个验证器、同一个开关）
------------------------------------------------
  D1_mlp         MLP 判断（阶段 7 的基线）
  D1_anchor_q    kNN 判断，λ 用**全部**经验量（= 阶段 6 的 λ）
  D1_anchor_t    kNN 判断，λ 只用**已转正**的经验量，待裁判条目权重 0.4

  D1_anchor_q -> D1_anchor_t 这一步之差，就是"把数量换成可信度"的全部效果。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject, weak_model_confusion
from exp.learners import NaiveLearner, ReservoirLearner, TieredLearner
from exp.realdata import RealClassStream, load_mnist
from exp.run import run_trajectory, summarize
from mm import DelayedLabelValidator, NeighborValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
CAP, STEPS, P = 600, 400, 0.4
SEEDS = [0, 1, 2]
MN = load_mnist()
CONF, WACC = weak_model_confusion()


class _NoopModel:
    n_out = 0
    def expand(self, n): pass
    def step(self, X, y): return 0.0
    def predict(self, X): return np.zeros(len(X), dtype=int)


class AnchorVerifiedLearner(TieredLearner):
    """kNN 判断 + 冷备锚，λ 可以只看"已转正"的经验。"""

    def __init__(self, dim, seed, capacity, validator, lam_mode="quantity",
                 k=16, k0=8, temp=0.05, prov_weight=0.4):
        super().__init__(dim, seed=seed, capacity=capacity, validator=validator, theta=0.7,
                         admit_unverifiable=True, cold_mode="ema", cold_alpha=0.2, cold_share=0.5)
        self.model = _NoopModel()
        self.lam_mode = lam_mode
        self.k = k
        self.k0 = k0
        self.temp = temp
        self.prov_weight = prov_weight
        self.n_classes = 0

    def begin_task(self, task):
        super().begin_task(task)
        self.n_classes = max(self.n_classes, max(task.classes) + 1)

    def _proba(self, X, keys, values, n_classes, ew=None, ver=None):
        n = len(X)
        if len(keys) == 0:
            return np.zeros((n, n_classes)), np.zeros(n), np.zeros(n)
        K = np.asarray(keys, dtype=np.float64)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Kn = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)
        S = Xn @ Kn.T
        kk = min(self.k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        rows = np.arange(n)[:, None]
        s = S[rows, idx]
        v = values[idx]
        ok = s >= self.sim_floor
        w = np.exp((s - 1.0) / self.temp) * ok
        if ew is not None:
            w = w * ew[idx]
        P = np.zeros((n, n_classes))
        np.add.at(P, (np.repeat(np.arange(n), kk), v.ravel()), w.ravel())
        if ver is None:
            nsup_all = ok.sum(axis=1).astype(float)
            nsup_ver = nsup_all
        else:
            nsup_all = ok.sum(axis=1).astype(float)
            nsup_ver = (ok * (ver[idx] > 0)).sum(axis=1).astype(float)
        tot = P.sum(axis=1, keepdims=True)
        return P / np.maximum(tot, 1e-12), nsup_all, nsup_ver

    def _cold_arrays(self):
        cs = sorted(self.cold_centroids.keys())
        if not cs:
            return None, None
        return np.array([self.cold_centroids[c] for c in cs]), np.array(cs)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        nc = self.n_classes
        warm = self.mem.warm
        wk = [e.key for e in warm]
        wv = np.array([int(e.value) for e in warm], dtype=int) if warm else np.zeros(0, dtype=int)
        if self.lam_mode == "trust" and warm:
            ew = np.array([1.0 if not e.provisional else self.prov_weight for e in warm])
            ver = np.array([0.0 if e.provisional else 1.0 for e in warm])
        else:
            ew = None
            ver = None
        Pw, nsup_all, nsup_ver = self._proba(X, wk, wv, nc, ew, ver)
        ck, cv = self._cold_arrays()
        Pc = self._proba(X, ck, cv, nc)[0] if ck is not None else np.zeros((len(X), nc))
        nsup = nsup_ver if self.lam_mode == "trust" else nsup_all
        lam = nsup / (nsup + self.k0)
        P = lam[:, None] * Pw + (1 - lam[:, None]) * Pc
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        return P.argmax(axis=1)


ARMS = ["gold", "D1_mlp", "D1_anchor_q", "D1_anchor_t", "D2_anchor_t"]


def stream(seed):
    st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=5, classes_per_task=2,
                         n_train=400, n_holdout=200, n_test=200,
                         label_noise=0.0, seed=1000 + seed)
    rng = np.random.default_rng(777 + seed)
    for task in st:
        task.y_train = inject(task.y_train_true, task.classes, P, rng, "structured", CONF)
    return st


def build(name, dim, seed):
    nv = lambda: NeighborValidator(np.zeros((0, dim)), np.zeros(0))
    if name == "gold":
        return TieredLearner(dim, seed=seed, capacity=CAP, validator=nv(), theta=0.7)
    if name == "D1_mlp":
        v = DelayedLabelValidator(nv(), lag=1)
        return TieredLearner(dim, seed=seed, capacity=CAP, validator=v, theta=0.7,
                             admit_unverifiable=True)
    if name == "D1_anchor_q":
        v = DelayedLabelValidator(nv(), lag=1)
        return AnchorVerifiedLearner(dim, seed=seed, capacity=CAP, validator=v, lam_mode="quantity")
    if name == "D1_anchor_t":
        v = DelayedLabelValidator(nv(), lag=1)
        return AnchorVerifiedLearner(dim, seed=seed, capacity=CAP, validator=v, lam_mode="trust")
    if name == "D2_anchor_t":
        v = DelayedLabelValidator(nv(), lag=2)
        return AnchorVerifiedLearner(dim, seed=seed, capacity=CAP, validator=v, lam_mode="trust")
    raise KeyError(name)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        for name in ARMS:
            st = stream(seed)
            L = build(name, st.dim, seed)
            acc, led = run_trajectory(L, st, steps_per_task=STEPS)
            s = summarize(acc, led)
            prov = sum(1 for e in getattr(L.mem, "warm", []) if getattr(e, "provisional", False))
            s["provisional_left"] = prov
            s["rollback_evicted"] = led[-1].get("rollback_evicted", 0)
            recs.append(dict(arm=name, seed=seed, **s))
        for arm in ("A_naive", "B_replay"):
            st = stream(seed)
            L = NaiveLearner(st.dim, seed=seed) if arm == "A_naive" else \
                ReservoirLearner(st.dim, seed=seed, capacity=CAP)
            acc, led = run_trajectory(L, st, steps_per_task=STEPS)
            recs.append(dict(arm=arm, seed=seed, **summarize(acc, led)))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage8.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print("=" * 96)
    print("阶段 8：λ 用「数量」还是「可信度」？（真实 MNIST / 结构化幻觉 40% / 3 seeds）")
    print("=" * 96)
    print("%-16s %-18s %-14s %-14s %s" % ("组", "ACC", "污染率", "待裁判残留", "回滚撤回"))
    for name in ARMS + ["B_replay", "A_naive"]:
        rs = [r for r in recs if r["arm"] == name]
        pol = [r["pollution_final"] for r in rs if r["pollution_final"] == r["pollution_final"]]
        print("%-16s %.3f±%.3f        %-14s %-14s %.0f"
              % (name, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 ("%.3f" % np.mean(pol)) if pol else "  -  ",
                 "%.0f" % np.mean([r.get("provisional_left", 0) for r in rs]),
                 np.mean([r.get("rollback_evicted", 0) for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage8.json")))


if __name__ == "__main__":
    main()
