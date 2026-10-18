# -*- coding: utf-8 -*-
"""阶段 6：冷备作为"判断的锚"，温备作为"累积的经验" —— 自适应仲裁。

用户的构想
------------------------------------------------
"把冷备做成非动态的（冻结的）基，与门控结合去判断；结合温备数据形成类似人类一样的
 经验累积式的判断。"

和阶段 3 的关键区别
------------------------------------------------
阶段 3 里冷备是**回放素材**（把冷备条目混进训练 batch），冻结果然有害。
这里冷备是**判断时的先验**：决策时把两个来源按"经验够不够"自适应地加权。

    p(y|x) = λ(x) · p_warm(y|x) + (1 − λ(x)) · p_cold(y|x)
    λ(x)   = n_support(x) / (n_support(x) + k0)

n_support(x) = x 附近的**经验证的**温备条数。
人类式的直觉：这个领域我经验多，就信经验；没经验，就退回底层先验。
温备是容量有限的（会被淘汰），冷备**从不淘汰** —— 所以冷备恰好补在温备没覆盖的地方。

六组对照（同一个门控、同一个流，只换"怎么判断"）
------------------------------------------------
  MLP_gated    阶段 3 的基线：MLP 判断（记忆只用于回放）
  K_warm       纯温备判断（λ=1）
  K_cold       纯冷备判断（λ=0，冻结的先验）
  K_fixed      固定 λ=0.5
  K_adapt      **经验累积式自适应 λ**（冷备冻结 —— 用户的字面版本）
  K_adapt_ema  同上，但冷备跟着世界 EMA 更新

可证伪预测
------------------------------------------------
  1. 温备容量小（60）时，K_adapt 明显优于 K_warm（冷备补上了被淘汰的覆盖）；
     容量大（300）时两者接近（冷备冗余，与阶段 3B 一致）。
  2. 漂移下，**冻结**的冷备会拖累 K_adapt；K_adapt_ema 应更好。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.learners import TieredLearner
from exp.recurring import RecurringDriftStream
from exp.run import train_one_task
from mm import NeighborValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DIM = 20
STEPS = 300
SEEDS = [0, 1, 2]
DRIFTS = [0.0, 0.6]
CAPS = [60, 300]


class _NoopModel:
    """kNN 判断路线不需要训练网络；用它把 train_one_task 的梯度步变成空操作。"""
    n_out = 0
    def expand(self, n): pass
    def step(self, X, y): return 0.0
    def predict(self, X): return np.zeros(len(X), dtype=int)


class AnchorLearner(TieredLearner):
    """把三类记忆的仲裁搬进判断路径。"""

    def __init__(self, dim, seed, capacity, validator, lam_mode="adapt",
                 k=16, k0=8, temp=0.05, cold_mode="frozen", cold_alpha=0.2, cold_share=0.5):
        super().__init__(dim, seed=seed, capacity=capacity, validator=validator, theta=0.7,
                         cold_mode=cold_mode, cold_alpha=cold_alpha, cold_share=cold_share)
        self.model = _NoopModel()
        self.lam_mode = lam_mode
        self.k = k
        self.k0 = k0
        self.temp = temp
        self.n_classes = 0

    def begin_task(self, task):
        super().begin_task(task)
        self.n_classes = max(self.n_classes, max(task.classes) + 1)

    # ---------------- 概率来源 ----------------
    def _proba(self, X, keys, values, n_classes):
        if len(keys) == 0:
            return np.zeros((len(X), n_classes)), np.zeros(len(X))
        K = np.asarray(keys, dtype=np.float64)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Kn = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)
        S = Xn @ Kn.T
        k = min(self.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(len(X))[:, None]
        s = S[rows, idx]
        v = values[idx]
        w = np.exp((s - 1.0) / self.temp) * (s >= self.sim_floor)
        P = np.zeros((len(X), n_classes))
        np.add.at(P, (np.repeat(np.arange(len(X)), k), v.ravel()), w.ravel())
        nsup = (s >= self.sim_floor).sum(axis=1).astype(float)
        tot = P.sum(axis=1, keepdims=True)
        return P / np.maximum(tot, 1e-12), nsup

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
        ck, cv = self._cold_arrays()
        Pw, nsup = self._proba(X, wk, wv, nc)
        Pc = self._proba(X, ck, cv, nc)[0] if ck is not None else np.zeros((len(X), nc))
        if self.lam_mode == "warm":
            lam = np.ones(len(X))
        elif self.lam_mode == "cold":
            lam = np.zeros(len(X))
        elif self.lam_mode == "fixed":
            lam = np.full(len(X), 0.5)
        else:                                  # adapt：经验累积式
            lam = nsup / (nsup + self.k0)
        P = lam[:, None] * Pw + (1 - lam[:, None]) * Pc
        tot = P.sum(axis=1, keepdims=True)
        P = P / np.maximum(tot, 1e-12)
        return P.argmax(axis=1)


ARMS = [
    ("MLP_gated",   None),
    ("K_warm",      dict(lam_mode="warm")),
    ("K_cold",      dict(lam_mode="cold")),
    ("K_fixed",     dict(lam_mode="fixed")),
    ("K_adapt",     dict(lam_mode="adapt")),
    ("K_adapt_ema", dict(lam_mode="adapt")),
]


def build(name, kw, seed, cap):
    v = NeighborValidator(np.zeros((0, DIM)), np.zeros(0))
    if name == "MLP_gated":
        return TieredLearner(DIM, seed=seed, capacity=cap, validator=v, theta=0.7)
    cm = "ema" if name == "K_adapt_ema" else "frozen"
    return AnchorLearner(DIM, seed=seed, capacity=cap, validator=v,
                         cold_mode=cm, cold_alpha=0.2, cold_share=0.5, **kw)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for drift in DRIFTS:
        for cap in CAPS:
            for seed in SEEDS:
                st = RecurringDriftStream(n_classes=25, repeats=4, classes_per_task=10,
                                          n_train=500, n_holdout=200, n_test=100,
                                          drift=drift, label_noise=0.3, seed=1000 + seed)
                for name, kw in ARMS:
                    L = build(name, kw, seed, cap)
                    for t, task in enumerate(st):
                        train_one_task(L, task, t, steps_per_task=STEPS)
                    pred = L.predict(st.xt)
                    acc = float((pred == st.yt).mean())
                    bk = st.age_bucket_accuracy(pred)
                    recs.append(dict(arm=name, drift=drift, cap=cap, seed=seed,
                                     ACC=acc, bucket_fresh=bk["新鲜(0-1)"],
                                     bucket_mid=bk["中等(2-4)"], bucket_old=bk["陈旧(5-9)"],
                                     warm=len(L.mem.warm), cold=len(L.cold_centroids)))
            print("  drift=%.1f cap=%d 完成 %.0f 秒" % (drift, cap, time.time() - t0))
            sys.stdout.flush()
    json.dump(recs, open(os.path.join(OUT, "results_stage6.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print("=" * 100)
    print("阶段 6：冷备当锚 + 经验累积式仲裁（25 类 / 每类出现 4 次 / 数据噪声 30% / 3 seeds）")
    print("=" * 100)
    for drift in DRIFTS:
        print("drift=%.1f" % drift)
        print("  %-12s %-8s %s" % ("温备容量", "arm", "ACC"))
        for cap in CAPS:
            for name, _ in ARMS:
                v = [r["ACC"] for r in recs if r["drift"] == drift and r["cap"] == cap and r["arm"] == name]
                print("  %-12s %-12s %.3f±%.3f" % (cap, name, np.mean(v), np.std(v)))
        print()
    print("按「类最后一次出现距今多久」分桶（drift=0.6, cap=60，温备最吃紧的一档）")
    print("-" * 100)
    print("%-14s %-14s %-14s %-14s %s" % ("arm", "新鲜(0-1)", "中等(2-4)", "陈旧(5-9)", "温备/冷备条目"))
    for name, _ in ARMS:
        rs = [r for r in recs if r["drift"] == 0.6 and r["cap"] == 60 and r["arm"] == name]
        f = lambda k: np.nanmean([r[k] for r in rs])
        print("%-14s %-14.3f %-14.3f %-14.3f %.0f/%.0f"
              % (name, f("bucket_fresh"), f("bucket_mid"), f("bucket_old"),
                 np.mean([r["warm"] for r in rs]), np.mean([r["cold"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage6.json")))


if __name__ == "__main__":
    main()
