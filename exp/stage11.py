# -*- coding: utf-8 -*-
"""阶段 11：锚审计 —— 冷备里的错误由谁来发现？

用户的第 4 条
------------------------------------------------
"因为已知数据也有可能是有错误的，所以单独建立另一套比对验证机制。"
当前实现里冷备锚一旦编译就不可质疑 —— 这是整个架构最后一个不需要自检的部件。

设计时冒出来的更锋利的问题
------------------------------------------------
如果冷备里的错误和脏数据**同源**，那么"用新证据反查旧锚"这条路会自己断掉：
  - 坏锚说"真值是 A"
  - 数据流里的脏样本恰好也是 A（同源污染）
  - 门控看到两边一致 -> 断定是"干净数据"，**一条 side 查询都不花**
  - 于是审计池里永远没有这些区域的真值 -> 被动审计什么也发现不了
**系统会自信地错下去。**

于是三组对照
------------------------------------------------
  no_audit       只有门控（阶段 10 的 D 组），锚永不被质疑
  audit_passive  用门控**自然花掉**的 side 查询积累审计池，回测旧锚
  audit_active   **定期主动抽检**：不管门控怎么判，每轮对每个锚覆盖区强制花预算取真值

场景
------------------------------------------------
任务 0 的污染率 0.6，而且脏样本**全部来自同一条错误规则 W_wrong**；
系统启动时被塞进一个锚 = W_wrong（覆盖任务 0 的盒子）。
这模拟的就是"已知数据本身是错的，而且错误还会吸引同源的脏数据"。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.stage10 import (NF, N_OBS, N_QUERY, N_TASK, RADIUS, ROUND, TOL, Stream,
                         Tool, omp_fit, phi)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SEEDS = [0, 1, 2]
POLLUTE_T0 = 0.6
FORCE_PER_ROUND = 3          # 每轮对每个锚覆盖区强制抽检几条
AUDIT_MIN = 12               # 判定一个锚需要的最少真值条数（证据不足不许降级）
QUAR_SAMPLE = 0.5            # 观察区每轮抽检比例
STRIKE_K = 8                 # 一个锚累计误判多少次就自动降级
QUAR_MODES = ("quarantine", "quarantine_strike", "weighted_quarantine")
FORCE_MODES = ("audit_active", "audit_trace", "audit_revisit", "weighted", "weighted_quarantine")


def make_stream(seed):
    st = Stream(1000 + seed)
    rng = np.random.default_rng(4242 + seed)
    # 任务 0 的污染：全部来自同一条错误规则（= 要被审计的那条）
    xo, yo, dirty = st.obs[0]
    wrong = st.ws[1 % N_TASK]
    bad = rng.random(N_OBS) < POLLUTE_T0
    yo = yo.copy()
    for i in np.flatnonzero(bad):
        yo[i] = float((phi(xo[i:i + 1]) @ wrong).ravel()[0]) + rng.normal(0, 0.02)
    st.obs[0] = (xo, yo, bad)
    st.wrong_rule = wrong
    return st


class AuditRunner:
    def __init__(self, mode, seed, stream, rng_seed=777):
        self.mode = mode
        # 分开的随机流：加一个新机制不应该扰动其他机制的随机轨迹，
        # 否则两组会因为"多消耗了几个随机数"而走上完全不同的路径（踩过）。
        self.rng = np.random.default_rng(rng_seed + seed)
        self.rng_force = np.random.default_rng(rng_seed + seed + 10000)
        self.rng_quar = np.random.default_rng(rng_seed + seed + 20000)
        self.tools = []
        self.accepted_x, self.accepted_y = [], []
        self.audit_x, self.audit_y = [], []
        self.side_queries = 0
        self.forced_queries = 0
        self.demoted = 0
        self.recompiled = 0
        self.rejected = 0
        self.compiled = 0
        self.sponsors = []      # 与 accepted_x 对齐：每条记忆是被哪些锚担保进来的
        self.retracted = 0
        self.pending = []       # 被拒候选：(x, y, 否决它的锚)。坏锚被查出来后要重审这批
        self.revisited = 0
        self.quarantine = []    # 观察区：被单个锚否决的候选，不直接丢，等着被抽检
        self.recovered = 0      # 抽检后发现"其实是干净的"、放行回来的条数
        self.strikes = {}       # 锚的误判计数
        self.strike_demoted = 0
        # 启动时注入坏锚：覆盖任务 0 的盒子，但用的是错误规则
        self.tools.append(Tool(stream.wrong_rule.copy(), stream.centers[0], RADIUS * 2.0))

    # ------------------------------------------------ 三条通道
    def ch_anchor(self, x, y):
        votes = [1.0 if abs(tl.predict(x) - y) < TOL else 0.0
                 for tl in self.tools if tl.covers(x)]
        return float(np.mean(votes)) if votes else None

    def ch_peer(self, x, y, peers_x, peers_y):
        if len(peers_x) < 8:
            return None
        k = np.max(np.abs(peers_x - x), axis=1) <= 2.0 * RADIUS
        if int(k.sum()) < 8:
            return None
        w, r2 = omp_fit(peers_x[k], peers_y[k])
        if r2 < 0.5:
            return None
        return float(np.exp(-(abs(float((phi(x) @ w).ravel()[0]) - y) / TOL) ** 2))

    def side(self, x, truth_fn):
        self.side_queries += 1
        yt = float(truth_fn(x))
        self.audit_x.append(np.asarray(x).copy())
        self.audit_y.append(yt)
        return yt

    # ------------------------------------------------ 审计
    def process_quarantine(self, truth_fn):
        """观察区抽检：**对"被拒"的候选也随机取样**，而不是只在门控犹豫时才花预算。

        这是本轮的关键 —— 原来 score=0 属于"两个通道一致说它脏"，系统就自信地拒了，
        从不回头验证自己的拒绝，于是好数据静默流失。
        """
        if not self.quarantine:
            return
        n = max(1, int(round(len(self.quarantine) * QUAR_SAMPLE)))
        n = min(n, len(self.quarantine))
        pick = set(self.rng_quar.choice(len(self.quarantine), size=n, replace=False).tolist())
        keep = []
        for i, (x, y, veto) in enumerate(self.quarantine):
            if i not in pick:
                keep.append((x, y, veto))
                continue
            yt = float(truth_fn(x))
            self.side_queries += 1
            self.audit_x.append(np.asarray(x).copy())
            self.audit_y.append(yt)
            if abs(yt - y) < TOL:
                # 抽检发现它其实是干净的 -> 否决它的锚误判了
                self.accepted_x.append(x)
                self.accepted_y.append(y)
                self.sponsors.append([])
                self.recovered += 1
                if self.mode == "quarantine_strike":
                    for tl in veto:
                        self.strikes[id(tl)] = self.strikes.get(id(tl), 0) + 1
        self.quarantine = keep
        if self.mode == "quarantine_strike":
            for tl in list(self.tools):   # 只有这个 mode 才真的撤销锚
                if self.strikes.get(id(tl), 0) >= STRIKE_K:
                    self.tools.remove(tl)
                    self.strike_demoted += 1
                    km = [tl not in sp for sp in self.sponsors]
                    before = len(self.accepted_x)
                    self.accepted_x = [v for v, kk in zip(self.accepted_x, km) if kk]
                    self.accepted_y = [v for v, kk in zip(self.accepted_y, km) if kk]
                    self.sponsors = [v for v, kk in zip(self.sponsors, km) if kk]
                    self.retracted += before - len(self.accepted_x)
                    self.strikes.pop(id(tl), None)

    def run_audit(self, truth_fn=None):
        if self.mode == "no_audit" or not self.tools or len(self.audit_x) < AUDIT_MIN:
            return
        X = np.array(self.audit_x)
        Y = np.array(self.audit_y)
        kept = []
        for tl in self.tools:
            k = np.max(np.abs(X - tl.center), axis=1) <= tl.radius
            if int(k.sum()) < AUDIT_MIN:
                kept.append(tl)
                continue
            pred = np.array([tl.predict(x) for x in X[k]])
            agree = float((np.abs(pred - Y[k]) < TOL).mean())
            if agree >= 0.8:
                kept.append(tl)
                continue
            self.demoted += 1                      # 这个锚有问题 -> 降级
            if self.mode in ("audit_trace", "audit_revisit"):
                # 连带撤回：所有由这个坏锚担保进来的记忆，一并拿下
                keep_m = [tl not in sp for sp in self.sponsors]
                before = len(self.accepted_x)
                self.accepted_x = [v for v, kk in zip(self.accepted_x, keep_m) if kk]
                self.accepted_y = [v for v, kk in zip(self.accepted_y, keep_m) if kk]
                self.sponsors = [v for v, kk in zip(self.sponsors, keep_m) if kk]
                self.retracted += before - len(self.accepted_x)
            if self.mode == "audit_revisit" and truth_fn is not None:
                # 关键：坏锚当年**拒收**过的候选，现在要重新审一遍
                rest, rev = [], []
                for p in self.pending:
                    (rev if any(t2 is tl for t2 in p[2]) else rest).append(p)
                self.pending = rest
                for (px_, py_, _v) in rev:
                    a = None
                    for t2 in self.tools:
                        if t2 is tl or not t2.covers(px_):
                            continue
                        a = 1.0 if abs(t2.predict(px_) - py_) < TOL else 0.0
                        break
                    if a is None:
                        yt = float(truth_fn(px_))
                        self.side_queries += 1
                        self.audit_x.append(np.asarray(px_).copy())
                        self.audit_y.append(yt)
                        ok = abs(yt - py_) < TOL
                    else:
                        ok = a >= 0.5
                    if ok:
                        self.accepted_x.append(px_)
                        self.accepted_y.append(py_)
                        self.sponsors.append([])
                        self.revisited += 1
            if int(k.sum()) >= AUDIT_MIN + 2:
                w, r2 = omp_fit(X[k], Y[k], max_terms=4)
                if r2 > 0.99:
                    kept.append(Tool(w, tl.center, tl.radius))   # 用真值重编译
                    self.recompiled += 1
        self.tools = kept

    def maybe_force_sample(self, xs, truth_fn):
        """主动抽检：不管门控怎么判，每个锚覆盖区每轮强制取一条真值。"""
        if self.mode not in FORCE_MODES:
            return
        for tl in list(self.tools):
            idx = [i for i in range(len(xs)) if tl.covers(xs[i])]
            if not idx:
                continue
            take = min(FORCE_PER_ROUND, len(idx))
            for i in self.rng_force.choice(idx, size=take, replace=False):
                self.forced_queries += 1
                self.side(xs[int(i)], truth_fn)       # 强制取真值

    # ------------------------------------------------ 一个回合
    def process_round(self, xs, ys, truth_fn):
        self.maybe_force_sample(xs, truth_fn)
        keep_x, keep_y, keep_s = [], [], []
        for i in range(len(xs)):
            x, y = xs[i], ys[i]
            px = np.delete(xs, i, axis=0)
            py = np.delete(ys, i)
            a = self.ch_anchor(x, y)
            p = self.ch_peer(x, y, px, py)
            if a is None and p is None:
                score = 1.0 if abs(self.side(x, truth_fn) - y) < TOL else 0.0
            else:
                parts = []
                if a is not None:
                    parts.append((0.6, a))
                if p is not None:
                    parts.append((0.4 if a is not None else 1.0, p))
                tot = sum(wt for wt, _ in parts)
                score = sum(wt * v for wt, v in parts) / max(tot, 1e-9)
                if 0.35 <= score <= 0.65:
                    score = 1.0 if abs(self.side(x, truth_fn) - y) < TOL else 0.0
            if score >= 0.5:
                keep_x.append(x)
                keep_y.append(y)
                # 记下"谁担保了这条" —— 审计判出坏锚时要用它做连带撤回
                keep_s.append([tl for tl in self.tools
                               if tl.covers(x) and abs(tl.predict(x) - y) < TOL])
            else:
                self.rejected += 1
                veto = [tl for tl in self.tools
                        if tl.covers(x) and abs(tl.predict(x) - y) >= TOL]
                if veto:
                    self.pending.append((np.asarray(x).copy(), y, veto))
                    if self.mode in QUAR_MODES:
                        self.quarantine.append((np.asarray(x).copy(), y, veto))
        if self.mode in QUAR_MODES:
            self.process_quarantine(truth_fn)
        self.accepted_x.extend(keep_x)
        self.accepted_y.extend(keep_y)
        self.sponsors.extend(keep_s)
        self.run_audit(truth_fn)

    def try_compile(self, center):
        if len(self.accepted_x) < 30:
            return
        X = np.array(self.accepted_x); y = np.array(self.accepted_y)
        k = np.max(np.abs(X - center), axis=1) <= RADIUS * 1.6
        if int(k.sum()) < 30:
            return
        w, r2 = omp_fit(X[k], y[k], max_terms=5)
        if r2 > 0.995 and len(np.flatnonzero(np.abs(w) > 1e-6)) <= 5:
            self.tools.append(Tool(w, center, RADIUS * 2.0))
            self.compiled += 1
            self.accepted_x = list(X[~k]); self.accepted_y = list(y[~k])

    def predict(self, X):
        out = np.zeros(len(X))
        ax = np.array(self.accepted_x) if self.accepted_x else np.zeros((0, 4))
        ay = np.array(self.accepted_y) if self.accepted_y else np.zeros(0)
        for i, x in enumerate(X):
            hit = [tl for tl in self.tools if tl.covers(x)]
            if hit:
                out[i] = float(np.mean([tl.predict(x) for tl in hit]))
            elif len(ax):
                j = int(np.argmin(np.max(np.abs(ax - x), axis=1)))
                out[i] = float(ay[j])
        return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode in ("no_audit", "audit_passive", "audit_active"):
            R = AuditRunner(mode, seed, st)
            accs, acc_t0 = [], []
            for t in range(N_TASK):
                xo, yo, _ = st.obs[t]
                tf = lambda x, _t=t: float((phi(x) @ st.ws[_t]).ravel()[0])
                for s0 in range(0, N_OBS, ROUND):
                    xb, yb = xo[s0:s0 + ROUND], yo[s0:s0 + ROUND]
                    R.process_round(xb, yb, tf)
                    R.try_compile(st.centers[t])
                a, a0 = [], []
                for j in range(t + 1):
                    xq, yq = st.queries[j]
                    pred = R.predict(xq)
                    hit = np.abs(pred - yq) < TOL
                    a.append(float(hit.mean()))
                    if j == 0:
                        a0.append(float(hit.mean()))
                accs.append(float(np.mean(a)))
                acc_t0.append(float(np.mean(a0)))
            recs.append(dict(mode=mode, seed=seed, ACC_final=accs[-1], ACC_task0=acc_t0[-1],
                             side=R.side_queries, forced=R.forced_queries,
                             demoted=R.demoted, recompiled=R.recompiled,
                             tools_left=len(R.tools)))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage11.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 100)
    print("阶段 11：锚审计（任务 0 被同源污染 60% + 启动时注入坏锚；5 任务 / 3 seeds）")
    print("=" * 100)
    print("%-15s %-18s %-18s %-20s %-12s %s" % ("组", "总体 ACC", "任务0 区域 ACC", "side 查询(其中强制)", "降级/重编译", "剩余锚"))
    for mode in ("no_audit", "audit_passive", "audit_active"):
        rs = [r for r in recs if r["mode"] == mode]
        print("%-15s %.3f±%.3f          %.3f±%.3f          %-20s %-12s %.1f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["ACC_task0"] for r in rs]), np.std([r["ACC_task0"] for r in rs]),
                 "%.0f (%.0f)" % (np.mean([r["side"] for r in rs]), np.mean([r["forced"] for r in rs])),
                 "%.1f / %.1f" % (np.mean([r["demoted"] for r in rs]), np.mean([r["recompiled"] for r in rs])),
                 np.mean([r["tools_left"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage11.json")))


if __name__ == "__main__":
    main()
