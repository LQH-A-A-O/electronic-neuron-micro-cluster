# -*- coding: utf-8 -*-
"""阶段 10：三通道矩阵交叉验证 + 把成熟知识**编译成工具**。

用户的想法
------------------------------------------------
1. 用冷备和热备相互交叉验证，绕开温备可能带来的错误
2. 加一条"完全侧载"的验证通道（对成熟数据和新数据做数学验证）
3. 侧载验证与冷备验证再做交叉对比，提高正确率
4. 已知数据（冷备）也可能有错 —— 所以要单独一套比对机制
5. **把成熟内容工具化**：某种数学运算既然能用代码实现，就不必再存它的"原理"，只存代码结构

第 5 条是把前四条串起来的关键：编译出来的工具**本身就是一条免费的验证通道**，
它同步、覆盖自己的定义域 —— 正好是阶段 5B 认定"唯一能落地"的那种信号。

任务域
------------------------------------------------
稀疏二阶多项式规则，作用在 R^4 的**局部盒子**里（一个盒子 = 一个"概念"）：
    y = phi(x) · w_t + eps,   x 落在盒子 t 内
污染：p 比例的观测被换成**另一条规则的输出**（结构化噪声）。
查询时不告诉你是哪个盒子 —— 系统必须自己判断 x 落在谁的管辖范围里。

三条验证通道（矩阵）
------------------------------------------------
  anchor（冷备）  已编译工具在 x 处的预测与候选 y 是否相符；没有工具覆盖 -> 弃权
  peer （热备互证）**留一法**：用同批其他候选拟合出的函数，回头检验这一条
  side （侧载）   直接问环境要 x 的真值 —— 要花预算，这是唯一有代价的通道

三组对照
------------------------------------------------
  B_gold     所有候选都用 side 验证（阶段 1 的方案）—— 准，但账单最高
  C_peer     只有 peer（热备互证）+ 模糊时才花 side，**不编译**
  D_compile  三通道 + **编译工具**：攒够已确认的样本就拟合出表达式，冻结成 anchor
             编译成功后对应记忆直接丢弃（只留代码结构）

主指标：查询准确率 与 **总 side 查询次数**。要回答的是：
      编译能不能在保住准确率的同时，把 oracle 的账单降下来。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DIM = 4
NF = 15
RADIUS = 1.1
TOL = 0.25
NOISE_P = 0.4
N_TASK, N_OBS, N_QUERY = 5, 200, 100
ROUND = 40
SEEDS = [0, 1, 2]


def phi(X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.stack([np.ones(len(X)), x1, x2, x3, x4,
                     x1 * x2, x1 * x3, x1 * x4, x2 * x3, x2 * x4, x3 * x4,
                     x1 * x1, x2 * x2, x3 * x3, x4 * x4], axis=1)


def omp_fit(X, y, max_terms=4):
    """正交匹配追踪：从 15 个候选特征里挑出最多 max_terms 项。"""
    P = phi(X)
    resid = y.astype(np.float64).copy()
    sel = []
    n_terms = min(max_terms, P.shape[1])
    for _ in range(n_terms):
        corr = np.abs(P.T @ resid)
        order = np.argsort(-corr)
        j = next((int(k) for k in order if int(k) not in sel), None)
        if j is None:
            break
        sel.append(j)
        w, *_ = np.linalg.lstsq(P[:, sel], y, rcond=None)
        resid = y - P[:, sel] @ w
    if not sel:
        return np.zeros(NF), -1e9
    w, *_ = np.linalg.lstsq(P[:, sel], y, rcond=None)
    pred = P[:, sel] @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    full = np.zeros(NF)
    full[sel] = w
    return full, 1.0 - ss_res / ss_tot


class Tool:
    """一个编译好的工具：一条表达式 + 它的定义域（盒子）。"""
    __slots__ = ("w", "center", "radius")

    def __init__(self, w, center, radius):
        self.w = w
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = radius

    def covers(self, x):
        return bool(np.max(np.abs(np.asarray(x) - self.center)) <= self.radius)

    def predict(self, x):
        return float((phi(x) @ self.w).ravel()[0])


class Stream:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        # 先把所有盒子与规则一次性建好：污染要引用"另一条规则"，必须全部存在
        self.centers, self.ws = [], []
        for t in range(N_TASK):
            while True:
                c = rng.uniform(-4, 4, size=DIM)
                if all(np.max(np.abs(c - s)) > 2.9 for s in self.centers):
                    break
            self.centers.append(c)
            k = int(rng.integers(3, 5))
            idx = rng.choice(np.arange(1, NF), size=k, replace=False)
            w = np.zeros(NF)
            w[idx] = rng.uniform(-2.0, 2.0, size=k) * np.where(rng.random(k) < 0.5, -1, 1)
            self.ws.append(w)
        self.obs, self.queries = [], []
        for t in range(N_TASK):
            c = self.centers[t]
            w = self.ws[t]
            xo = c + rng.uniform(-RADIUS, RADIUS, size=(N_OBS, DIM))
            yo = phi(xo) @ w + rng.normal(0, 0.02, N_OBS)
            dirty = rng.random(N_OBS) < NOISE_P
            for i in np.flatnonzero(dirty):
                j = int(rng.integers(0, N_TASK))
                while j == t:
                    j = int(rng.integers(0, N_TASK))
                yo[i] = float((phi(xo[i:i + 1]) @ self.ws[j]).ravel()[0]) + rng.normal(0, 0.02)
            self.obs.append((xo, yo, dirty))
            xq = c + rng.uniform(-RADIUS, RADIUS, size=(N_QUERY, DIM))
            yq = phi(xq) @ w + rng.normal(0, 0.02, N_QUERY)
            self.queries.append((xq, yq))


class Runner:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 555)
        self.tools = []
        self.accepted_x, self.accepted_y = [], []
        self.side_queries = 0
        self.compiled = 0
        self.rejected = 0

    # ---------------- 三条通道 ----------------
    def ch_anchor(self, x, y):
        if self.mode != "D_matrix":
            return None                      # 消融：关掉冷备锚这一通道
        votes = []
        for tl in self.tools:
            if tl.covers(x):
                votes.append(1.0 if abs(tl.predict(x) - y) < TOL else 0.0)
        if not votes:
            return None
        return float(np.mean(votes))

    def ch_peer(self, x, y, peers_x, peers_y):
        """留一法：用同批其他候选拟合一个函数，回头检验这一条。"""
        if len(peers_x) < 8:
            return None
        k = np.max(np.abs(peers_x - x), axis=1) <= 2.0 * RADIUS
        if int(k.sum()) < 8:
            return None
        w, r2 = omp_fit(peers_x[k], peers_y[k])
        if r2 < 0.5:
            return None
        resid = abs(float((phi(x) @ w).ravel()[0]) - y)
        return float(np.exp(-(resid / TOL) ** 2))

    def ch_side(self, x, y, truth_fn):
        self.side_queries += 1
        return 1.0 if abs(float(truth_fn(x)) - y) < TOL else 0.0   # truth_fn 已保证返回标量

    # ---------------- 一个回合 ----------------
    def process_round(self, xs, ys, truth_fn):
        keep_x, keep_y = [], []
        for i in range(len(xs)):
            x, y = xs[i], ys[i]
            peers_x = np.delete(xs, i, axis=0)
            peers_y = np.delete(ys, i)
            a = self.ch_anchor(x, y)
            if self.mode == "B_gold":
                score = self.ch_side(x, y, truth_fn)
            else:
                p = self.ch_peer(x, y, peers_x, peers_y)
                if a is None and p is None:
                    # 两条免费通道都弃权 -> 只能花预算问环境
                    score = self.ch_side(x, y, truth_fn)
                else:
                    parts = []
                    if a is not None:
                        parts.append((0.6, a))
                    if p is not None:
                        parts.append((0.4 if a is not None else 1.0, p))
                    tot = sum(wt for wt, _ in parts)
                    score = sum(wt * v for wt, v in parts) / max(tot, 1e-9)
                    if 0.35 <= score <= 0.65:
                        score = self.ch_side(x, y, truth_fn)
            if score >= 0.5:
                keep_x.append(x)
                keep_y.append(y)
            else:
                self.rejected += 1
        return (np.array(keep_x).reshape(-1, DIM), np.array(keep_y)) if keep_x else \
               (np.zeros((0, DIM)), np.zeros(0))

    def try_compile(self, center):
        """攒够确认过的样本就编译成工具，然后把那批记忆丢掉（只留代码结构）。"""
        if len(self.accepted_x) < 30:
            return
        X = np.array(self.accepted_x)
        y = np.array(self.accepted_y)
        k = np.max(np.abs(X - center), axis=1) <= RADIUS * 1.6
        if int(k.sum()) < 30:
            return
        w, r2 = omp_fit(X[k], y[k], max_terms=5)
        if r2 > 0.995 and len(np.flatnonzero(np.abs(w) > 1e-6)) <= 5:
            self.tools.append(Tool(w, center, RADIUS * 2.0))
            self.compiled += 1
            keep = ~k
            self.accepted_x = list(X[keep])
            self.accepted_y = list(y[keep])

    def predict(self, X):
        out = np.zeros(len(X))
        ax = np.array(self.accepted_x) if self.accepted_x else np.zeros((0, DIM))
        ay = np.array(self.accepted_y) if self.accepted_y else np.zeros(0)
        for i, x in enumerate(X):
            hit = [tl for tl in self.tools if tl.covers(x)]
            if hit:
                out[i] = float(np.mean([tl.predict(x) for tl in hit]))
            elif len(ax):
                j = int(np.argmin(np.max(np.abs(ax - x), axis=1)))
                out[i] = float(ay[j])
            else:
                out[i] = 0.0
        return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = Stream(1000 + seed)
        for mode in ("B_gold", "C_peer", "D_matrix"):
            R = Runner(mode, seed)
            accs = []
            for t in range(N_TASK):
                xo, yo, dirty = st.obs[t]
                truth = {}
                xs_all = st.obs[t][0]
                for s0 in range(0, N_OBS, ROUND):
                    xb = xo[s0:s0 + ROUND]
                    yb = yo[s0:s0 + ROUND]
                    idx = np.arange(s0, min(s0 + ROUND, N_OBS))
                    tf = lambda x, _idx=idx: float((phi(x) @ st.ws[t]).ravel()[0])
                    kx, ky = R.process_round(xb, yb, tf)
                    if len(kx):
                        R.accepted_x.extend(list(kx))
                        R.accepted_y.extend(list(ky))
                    R.try_compile(st.centers[t])   # 三组都允许编译，差别只在门控怎么决定收谁
                # 评测：已见过的所有盒子里各抽查询
                a = []
                for j in range(t + 1):
                    xq, yq = st.queries[j]
                    pred = R.predict(xq)
                    a.append(float((np.abs(pred - yq) < TOL).mean()))
                accs.append(float(np.mean(a)))
            recs.append(dict(mode=mode, seed=seed, ACC_final=accs[-1], acc_curve=accs,
                             side_queries=R.side_queries, compiled=R.compiled,
                             rejected=R.rejected))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage10.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 96)
    print("阶段 10：三通道矩阵交叉验证 + 工具编译（5 任务 / 局部多项式规则 / 污染 40% / 3 seeds）")
    print("=" * 96)
    print("%-12s %-18s %-18s %-14s %s" % ("组", "最终查询 ACC", "side 查询次数", "编译出的工具", "被拒候选"))
    for mode in ("B_gold", "C_peer", "D_matrix"):
        rs = [r for r in recs if r["mode"] == mode]
        print("%-12s %.3f±%.3f          %-18.0f %-14.1f %.0f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["side_queries"] for r in rs]),
                 np.mean([r["compiled"] for r in rs]), np.mean([r["rejected"] for r in rs])))
    print()
    for mode in ("B_gold", "C_peer", "D_matrix"):
        rs = [r for r in recs if r["mode"] == mode]
        cur = np.mean([r["acc_curve"] for r in rs], axis=0)
        print("  %-10s 逐任务 ACC: %s" % (mode, " ".join("%.3f" % v for v in cur)))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage10.json")))


if __name__ == "__main__":
    main()
