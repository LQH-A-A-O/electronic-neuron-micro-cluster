# -*- coding: utf-8 -*-
"""阶段 23（G3）：坏锚能不能自然产生？审计能不能抓到它？—— D4。

19 个阶段的坏锚全是人注入的
------------------------------------------------
阶段 11~19 里，"坏锚"是一个 Tool(W_wrong, ...) 被直接塞进工具箱的。
那测的是**我们自己的构造**，不是系统的失效。这是整个项目最大的信任缺口。

这一轮换成自然产生
------------------------------------------------
不注入任何锚。只注入**数据源级别的系统性错标**：
  任务 0 的前 200 条观测里，80% 的标签被**整体反转**（a<->b）。
这是一段"被错标的数据源"，是真实世界最常见的污染形态 —— 一致的、成片的、看起来自洽的。

后果应该是：门控的两个免费通道（批内互证 + 已编译的工具）**同时被这段数据说服**，
于是它照单全收，然后用这批数据**编译出一个错的工具** —— 坏锚自然产生。

两组
  no_audit      审计关掉：工具一旦编译就永远 cred=1（阶段 14 之前的行为）
  audit         审计开着：每轮强制抽检若干条取真值，回测工具 -> 更新 cred（阶段 15/16 的机制）

怎么判定"自然产生了坏锚"
------------------------------------------------
每轮结束后，用一个**只有测量才可见**的探针集（当前任务的 holdout，真标签）测当前工具的准确率。
  tool_acc < 0.5 的轮次 = 坏锚发作的轮次
审计是否抓到 = 那些轮次里 cred 有没有被压下去
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
CRED_BLEND = float(os.environ.get("CB", "0.0"))   # 0 -> 可信度直接等于当前窗口一致率（无平滑）
AUDIT_MIN, CRED_USE = 8, 0.25
FORCE = 10
AUDIT_WINDOW = 60             # 审计只用最近这么多条证据（滑动窗口，别用全局累积）
BAD_WINDOW, BAD_FRAC = 200, 0.8
SEEDS = [0, 1, 2, 3, 4]
MN = load_mnist()
CONF, _ = weak_model_confusion()
_rng = np.random.default_rng(2026)
PROJ = _rng.normal(0, 1.0 / np.sqrt(784), (784, P_DIM))


def featurize(X):
    Z = np.asarray(X, dtype=np.float64) @ PROJ
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def source_corruption(st, seed):
    """数据源级错标：任务 0 的前 BAD_WINDOW 条里，80% 的标签整体反转。"""
    rng = np.random.default_rng(4242 + seed)
    t0 = st.tasks[0]
    a, b = t0.classes
    y = t0.y_train.copy()
    inv = np.where(y == a, b, a)
    hit = rng.choice(BAD_WINDOW, size=int(BAD_WINDOW * BAD_FRAC), replace=False)
    y[hit] = inv[hit]
    t0.y_train = y
    return st


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
    return source_corruption(st, seed)


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
    h = np.clip(np.sum((A @ Ginv) * A, axis=1), 0.0, 0.999)
    P = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        w = Ginv @ (A.T @ t)
        P[:, ci] = t - (t - A @ w) / (1.0 - h)
    return P


class R23:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(900 + seed)
        self.bx, self.by = [], []
        self.audit_x, self.audit_y = [], []
        self.tool = None
        self.side = 0
        self.tool_hist = []
        self.tool_cred_hist = []

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
        v = 1.0 if int(self.tool.predict(x)[0]) == y else 0.0
        w = self.tool.cred if self.mode == "audit" else 1.0
        return w * v + (1.0 - w) * 0.5

    def run_audit(self):
        if self.mode != "audit" or self.tool is None or len(self.audit_x) < AUDIT_MIN:
            return
        # 只取最近一段证据，并且只算落在这个锚"管辖类"里的那些
        X = np.array(self.audit_x[-AUDIT_WINDOW:])
        Y = np.array(self.audit_y[-AUDIT_WINDOW:])
        m = np.isin(Y, self.tool.classes)
        if int(m.sum()) < AUDIT_MIN:
            return
        p = self.tool._proba(X[m])
        agree = float((np.array(self.tool.classes)[p.argmax(axis=1)] == Y[m]).mean())
        self.tool.cred = CRED_BLEND * self.tool.cred + (1.0 - CRED_BLEND) * agree

    def round(self, xb, yb, truth_fn, probe_x, probe_y):
        n = len(xb)
        for i in self.rng.choice(n, size=min(FORCE, n), replace=False):
            i = int(i)
            self.side += 1
            self.audit_x.append(xb[i].copy()); self.audit_y.append(truth_fn(xb[i]))
        cls = sorted(set(yb.tolist()))
        if len(cls) >= 2 and n >= 10:
            P = loo_scores(xb, yb, cls)
            pred = np.array(cls)[P.argmax(axis=1)]
        else:
            P, pred = None, None
        use_tool = self.tool is not None and (self.mode != "audit" or self.tool.cred >= CRED_USE)
        if use_tool:
            Pt = self.tool._proba(xb)
            tpred = np.array(self.tool.classes)[Pt.argmax(axis=1)]
            tcover = Pt.max(axis=1) >= TAU
        else:
            tpred, tcover = None, np.zeros(n, dtype=bool)
        for i in range(n):
            x, y = xb[i], int(yb[i])
            a = (1.0 if int(tpred[i]) == y else 0.0) if (tpred is not None and tcover[i]) else None
            if a is not None and self.mode == "audit":
                w = self.tool.cred
                a = w * a + (1.0 - w) * 0.5
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
        self.refit()
        self.run_audit()
        if self.tool is not None and len(probe_y):
            # 只在这个工具"声称管得着"的类上测 —— 否则会把"还没学到新任务"误判成"工具坏了"
            m = np.isin(probe_y, self.tool.classes)
            if m.any():
                pr = self.tool.predict(probe_x[m])
                self.tool_hist.append(float((pr == probe_y[m]).mean()))
            else:
                self.tool_hist.append(float("nan"))
            self.tool_cred_hist.append(getattr(self.tool, "cred", 1.0))
        else:
            self.tool_hist.append(float("nan"))
            self.tool_cred_hist.append(float("nan"))

    def predict(self, X):
        out = np.zeros(len(X), dtype=int)
        rest = np.ones(len(X), dtype=bool)
        use = self.tool is not None and (self.mode != "audit" or self.tool.cred >= CRED_USE)
        if use:
            Pt = self.tool._proba(X)
            ms = Pt.max(axis=1) >= TAU
            if ms.any():
                out[ms] = np.array(self.tool.classes)[Pt.argmax(axis=1)][ms]
            rest = ~ms
        if rest.any() and self.bx:
            A = np.array(self.bx); ay = np.array(self.by)
            S = X[rest] @ A.T
            k = min(KNN_K, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            out[rest] = np.array([np.bincount(ay[r], minlength=N_CLASSES).argmax() for r in idx])
        return out


def run_arm(mode, st, seed):
    R = R23(mode, seed)
    accs = []
    px, py = [], []
    for t, task in enumerate(st):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        px.extend(list(hv_x)); py.extend(list(hv_y))      # 探针池：所有已见任务的真标签
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf,
                    np.array(px), np.array(py))
        a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    th = np.array(R.tool_hist, dtype=float)
    ch = np.array(R.tool_cred_hist, dtype=float)
    bad = np.where(th < 0.5)[0]
    caught = int(np.sum(ch[bad] < 0.5)) if len(bad) else 0
    return dict(mode=mode, seed=seed, ACC_final=accs[-1], side=R.side,
                bad_rounds=int(len(bad)), caught=caught,
                min_tool_acc=float(np.nanmin(th)) if len(th) else float("nan"),
                min_cred=float(np.nanmin(ch)) if len(ch) else float("nan"),
                kept=len(R.bx))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode in ("no_audit", "audit"):
            recs.append(run_arm(mode, st, seed))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()
    tag = "b%02d" % int(CRED_BLEND * 100)
    out_name = os.path.join(OUT, "results_stage23_%s.json" % tag)
    json.dump(recs, open(out_name, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 104)
    print("阶段 23（G3）：坏锚自然产生 + 审计能否抓到（真实 MNIST / 5 seeds）")
    print("=" * 104)
    print("%-12s %-15s %-14s %-13s %-13s %s" % ("组", "总体 ACC", "坏锚发作轮数", "其中被抓到", "工具最低准确率", "最低可信度"))
    for mode in ("no_audit", "audit"):
        rs = [r for r in recs if r["mode"] == mode]
        print("%-12s %.3f±%.3f       %-14.1f %-13.1f %-13.3f %.3f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["bad_rounds"] for r in rs]), np.mean([r["caught"] for r in rs]),
                 np.nanmean([r["min_tool_acc"] for r in rs]), np.nanmean([r["min_cred"] for r in rs])))
    print()
    print("CRED_BLEND = %.2f" % CRED_BLEND)
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, out_name))


if __name__ == "__main__":
    main()
