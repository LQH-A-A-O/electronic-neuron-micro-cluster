# -*- coding: utf-8 -*-
"""T5：让审计**追溯到记忆条目本身** —— 直接打 F23。

T4 发现的问题
------------------------------------------------
持续污染下审计抓到 27.2/28.2 轮坏锚、把工具可信度压到 0.093，
**而最终准确率差是 0.0000**。原因：审计只给"工具"打分，
工具被停用后预测退回 kNN —— 而 kNN 读的正是那份被污染的库。**锚的锅甩掉了，记忆的锅还在。**

这一轮怎么改
------------------------------------------------
让每轮强制抽检拿到的真值**直接打到记忆条目上**：
    抽到 (x, 真值 y)
    -> 找库里离 x 最近的 K 条记忆
    -> 标签和 y 不一致的那些 **降权**（乘 0.5，设下限）
预测改成**加权 kNN**，并且工具只用权重够高的条目重拟合。

三组
    no_audit     审计关掉（权重要么全 1，要么不更新）
    audit_tool   只有工具可信度（= 阶段 23 的机制，对照）
    audit_mem    真值打到记忆条目上（本轮的新机制）
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp.stage23 as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
S.CRED_BLEND = 0.0
SEEDS = [0, 1, 2, 3, 4]
K_HIT = 8          # 一次抽检影响最近的多少条记忆
DECAY = 0.5        # 不一致时权重乘多少
W_FLOOR = 0.05
W_USE = 0.5        # 权重低于它的条目不进工具、算作被剔除


def corrupt_all(st, seed):
    """持续污染：每个任务全程 80% 标签反转。"""
    rng = np.random.default_rng(4242 + seed)
    for task in st.tasks:
        a, b = task.classes
        y = task.y_train.copy()
        inv = np.where(y == a, b, a)
        hit = rng.choice(len(y), size=int(len(y) * 0.8), replace=False)
        y[hit] = inv[hit]
        task.y_train = y
    return st


class R5:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(900 + seed)
        self.bx, self.by, self.bw = [], [], []
        self.audit_x, self.audit_y = [], []
        self.tool = None
        self.side = 0
        self.tool_hist, self.tool_cred_hist = [], []
        self.n_demote = 0

    def refit(self):
        keep = [i for i, w in enumerate(self.bw) if w >= W_USE]
        if len(keep) < 60:
            self.tool = None; return
        X = np.array([self.bx[i] for i in keep]); y = np.array([self.by[i] for i in keep])
        cls = sorted(set(y.tolist()))
        if len(cls) >= 2:
            self.tool = S.LinTool(X, y, cls)

    def run_audit(self):
        if self.mode == "no_audit" or self.tool is None:
            return
        if len(self.audit_x) < S.AUDIT_MIN:
            return
        X = np.array(self.audit_x[-S.AUDIT_WINDOW:]); Y = np.array(self.audit_y[-S.AUDIT_WINDOW:])
        m = np.isin(Y, self.tool.classes)
        if int(m.sum()) >= S.AUDIT_MIN:
            p = self.tool._proba(X[m])
            agree = float((np.array(self.tool.classes)[p.argmax(axis=1)] == Y[m]).mean())
            self.tool.cred = agree                      # 无平滑
        if self.mode == "audit_mem" and self.bx:
            A = np.array(self.bx); ay = np.array(self.by)
            for x, y in zip(X, Y):
                d = np.linalg.norm(A - x, axis=1)
                k = min(K_HIT, len(d))
                idx = np.argpartition(d, k - 1)[:k]
                for i in idx:
                    if ay[i] != y:
                        self.bw[i] = max(W_FLOOR, self.bw[i] * DECAY)
                        self.n_demote += 1

    def round(self, xb, yb, truth_fn, probe_x, probe_y):
        n = len(xb)
        for i in self.rng.choice(n, size=min(S.FORCE, n), replace=False):
            i = int(i)
            self.side += 1
            self.audit_x.append(xb[i].copy()); self.audit_y.append(truth_fn(xb[i]))
        cls = sorted(set(yb.tolist()))
        if len(cls) >= 2 and n >= 10:
            P = S.loo_scores(xb, yb, cls)
            pred = np.array(cls)[P.argmax(axis=1)]
        else:
            pred = None
        if self.tool is not None and (self.mode == "no_audit" or self.tool.cred >= S.CRED_USE):
            Pt = self.tool._proba(xb)
            tpred = np.array(self.tool.classes)[Pt.argmax(axis=1)]
            tcover = Pt.max(axis=1) >= S.TAU
        else:
            tpred, tcover = None, np.zeros(n, dtype=bool)
        for i in range(n):
            x, y = xb[i], int(yb[i])
            a = (1.0 if int(tpred[i]) == y else 0.0) if (tpred is not None and tcover[i]) else None
            if a is not None and self.mode != "no_audit":
                cr = self.tool.cred if self.mode == "audit_tool" else 1.0
                a = cr * a + (1.0 - cr) * 0.5
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
            if sc is None or 0.35 < sc < S.HILO:
                ok = truth_fn(x) == y
                self.side += 1
            else:
                ok = sc >= S.HILO
            if ok:
                self.bx.append(x); self.by.append(y); self.bw.append(1.0)
        self.refit()
        self.run_audit()
        if self.tool is not None and len(probe_y):
            m = np.isin(probe_y, self.tool.classes)
            self.tool_hist.append(float((self.tool.predict(probe_x[m]) == probe_y[m]).mean()) if m.any() else float("nan"))
            self.tool_cred_hist.append(getattr(self.tool, "cred", 1.0))
        else:
            self.tool_hist.append(float("nan")); self.tool_cred_hist.append(float("nan"))

    def predict(self, X):
        out = np.zeros(len(X), dtype=int)
        rest = np.ones(len(X), dtype=bool)
        use = self.tool is not None and (self.mode == "no_audit" or self.tool.cred >= S.CRED_USE)
        if use:
            Pt = self.tool._proba(X)
            ms = Pt.max(axis=1) >= S.TAU
            if ms.any():
                out[ms] = np.array(self.tool.classes)[Pt.argmax(axis=1)][ms]
            rest = ~ms
        if rest.any() and self.bx:
            A = np.array(self.bx); ay = np.array(self.by); w = np.array(self.bw)
            Sx = X[rest] @ A.T
            k = min(S.KNN_K, Sx.shape[1])
            idx = np.argpartition(-Sx, k - 1, axis=1)[:, :k]
            res = []
            for r in idx:
                if self.mode == "audit_mem":
                    cnt = np.bincount(ay[r], weights=w[r], minlength=S.N_CLASSES)
                else:
                    cnt = np.bincount(ay[r], minlength=S.N_CLASSES).astype(float)
                res.append(int(cnt.argmax()))
            out[rest] = np.array(res)
        return out


def run(mode, seed):
    st = corrupt_all(S.make_stream(seed), seed)
    R = R5(mode, seed)
    accs = []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        px, py = [], []
        for j in range(t + 1):
            px.extend(list(st.tasks[j].x_holdout)); py.extend(list(st.tasks[j].y_holdout))
        for s0 in range(0, len(task.x_train), S.ROUND):
            R.round(task.x_train[s0:s0 + S.ROUND], task.y_train[s0:s0 + S.ROUND], tf,
                    np.array(px), np.array(py))
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    th = np.array(R.tool_hist, dtype=float); ch = np.array(R.tool_cred_hist, dtype=float)
    bad = np.where(th < 0.5)[0]
    return dict(mode=mode, seed=seed, ACC_final=accs[-1], side=R.side,
                bad_rounds=int(len(bad)), caught=int(np.sum(ch[bad] < 0.5)) if len(bad) else 0,
                min_cred=float(np.nanmin(ch)) if len(ch) else float("nan"),
                demote=R.n_demote, kept=len(R.bx),
                w_lt=sum(1 for x in R.bw if x < W_USE))


if __name__ == "__main__":
    res = []
    for seed in SEEDS:
        for mode in ("no_audit", "audit_tool", "audit_mem"):
            res.append(run(mode, seed))
        print("seed=%d done" % seed); sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_t5_auditmem.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 96)
    print("T5：审计追溯到记忆条目（持续污染 every_task / 真实 MNIST / 5 seeds）")
    print("=" * 96)
    print("%-14s %-14s %-12s %-12s %-12s %s"
          % ("组", "总体 ACC", "坏锚轮数", "被抓到", "最低可信度", "被降权条目"))
    base = None
    for mode in ("no_audit", "audit_tool", "audit_mem"):
        rs = [r for r in res if r["mode"] == mode]
        a = np.mean([r["ACC_final"] for r in rs])
        if base is None: base = a
        print("%-14s %.3f±%.3f    %-12.1f %-12.1f %-12.3f %.0f"
              % (mode, a, np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["bad_rounds"] for r in rs]), np.mean([r["caught"] for r in rs]),
                 np.nanmean([r["min_cred"] for r in rs]), np.mean([r["w_lt"] for r in rs])))
    print()
    for mode in ("audit_tool", "audit_mem"):
        rs = [r for r in res if r["mode"] == mode]
        d = np.mean([r["ACC_final"] for r in rs]) - base
        print("%-14s 相对 no_audit 的收益 %+.4f -> %s" % (mode, d, "收益" if d > 0.03 else "惰性"))
