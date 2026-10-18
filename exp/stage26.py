# -*- coding: utf-8 -*-
"""阶段 26（D3 扩展）：把 D1 对决批量跑到多个**不同性质**的数据集上。

    python exp/stage26.py <数据集名> [种子数]
    python exp/stage26.py all 3

数据集在 data/ds_<name>_X.npy / _y.npy（由 fetch_datasets.py 生成）。
判定沿用 D1 的通过线：**同等准确率下省 >= 30% 的 oracle**。
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import inject
from exp.stage25 import R25, LinTool, loo_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data")
POLLUTE, ROUND, KNN_FALLBACK = 0.4, 50, 5
BUDGETS = [0, 5, 10, 20, 50]


class Task:
    pass


def build(name, seed, n_task=None):
    X = np.load(os.path.join(DATA, "ds_%s_X.npy" % name)).astype(np.float64)
    y = np.load(os.path.join(DATA, "ds_%s_y.npy" % name)).astype(np.int64)
    rng = np.random.default_rng(700 + seed)
    perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
    n = len(X); ntr, nho = int(n * 0.6), int(n * 0.2)
    Xtr, ytr = X[:ntr], y[:ntr]
    Xho, yho = X[ntr:ntr + nho], y[ntr:ntr + nho]
    Xte, yte = X[ntr + nho:], y[ntr + nho:]
    cls = sorted(set(y.tolist()))
    if len(cls) <= 2:
        groups = [cls, cls]                       # 二类：按顺序切 2 段 = 域增量
        n_task = n_task or 2
    else:
        groups = [cls[i:i + 2] for i in range(0, len(cls) - 1, 2)]
        n_task = n_task or len(groups)
    per = max(40, min(400, ntr // (n_task * 2)))
    tasks = []
    for t in range(n_task):
        g = groups[t % len(groups)]
        def draw(A, B, k):
            parts = []
            for c in g:
                ci = np.flatnonzero(B == c)
                parts.append(rng.choice(ci, size=max(1, k // len(g)), replace=False))
            idx = np.concatenate(parts); rng.shuffle(idx)
            return A[idx], B[idx].copy()
        xt, yt = draw(Xtr, ytr, per)
        y_true = yt.copy()
        yt = inject(y_true, g, POLLUTE, rng, "uniform")
        xh, yh = draw(Xho, yho, per // 2)
        xe, ye = draw(Xte, yte, per // 2)
        k = Task(); k.task_id = t; k.classes = g
        k.x_train, k.y_train, k.y_train_true = xt, yt, y_true
        k.x_holdout, k.y_holdout = xh, yh
        k.x_test, k.y_test = xe, ye
        tasks.append(k)
    return tasks


def run_arm(mode, B, tasks, seed, ncls):
    R = R25(mode, seed, B)
    import exp.stage25 as S
    S.N_CLASSES = ncls
    accs = []
    for t, task in enumerate(tasks):
        hv_x, hv_y = task.x_holdout, task.y_holdout
        tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
        for s0 in range(0, len(task.x_train), ROUND):
            R.round(task.x_train[s0:s0 + ROUND], task.y_train[s0:s0 + ROUND], tf)
        a = [float((R.predict(tasks[j].x_test) == tasks[j].y_test).mean()) for j in range(t + 1)]
        accs.append(float(np.mean(a)))
    return dict(mode=mode, B=B, seed=seed, ACC_final=accs[-1], side=R.side, kept=len(R.bx))


def summarize(recs):
    def m(mode, B, k):
        v = [r[k] for r in recs if r["mode"] == mode and r["B"] == B and r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")
    gate_acc, gate_side = m("gate", -1, "ACC_final"), m("gate", -1, "side")
    xs = [(m("naive", b, "side"), m("naive", b, "ACC_final")) for b in BUDGETS]
    need = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if y0 <= gate_acc <= y1 and y1 > y0:
            need = x0 + (gate_acc - y0) / (y1 - y0) * (x1 - x0); break
    saved = (need - gate_side) / need if need == need and need > 0 else float("nan")
    # 更稳的判据：在门控实际花费那一点上，朴素随机标注能拿到多少 ACC
    nat = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if x0 <= gate_side <= x1 and x1 > x0:
            nat = y0 + (gate_side - x0) / (x1 - x0) * (y1 - y0); break
    return dict(gate_acc=gate_acc, gate_side=gate_side, need=need, saved=saved,
                naive_at_gate=nat, delta_acc=(gate_acc - nat) if nat == nat else float("nan"),
                raw=m("naive", 0, "ACC_final"), full=m("naive", 50, "ACC_final"),
                headroom=m("naive", 50, "ACC_final") - m("naive", 0, "ACC_final"))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    names = ([f[3:-6] for f in sorted(os.listdir(DATA)) if f.startswith("ds_") and f.endswith("_X.npy")]
             if which == "all" else [which])
    allrec, rows = {}, []
    for name in names:
        recs = []
        for seed in range(nseed):
            tasks = build(name, seed)
            ncls = int(max(t.y_test.max() for t in tasks)) + 1
            for mode, B in [("gate", -1), ("full", -1)] + [("naive", b) for b in BUDGETS]:
                recs.append(run_arm(mode, B, tasks, seed, ncls))
        allrec[name] = recs
        rows.append((name, summarize(recs)))
        print("%-9s 门控 %d次->%.3f  朴素需 %s 次  省 %s  头寸 %.3f"
              % (name, rows[-1][1]["gate_side"], rows[-1][1]["gate_acc"],
                 ("%.0f" % rows[-1][1]["need"]) if rows[-1][1]["need"] == rows[-1][1]["need"] else "n/a",
                 ("%.0f%%" % (100 * rows[-1][1]["saved"])) if rows[-1][1]["saved"] == rows[-1][1]["saved"] else "n/a",
                 rows[-1][1]["headroom"]))
        sys.stdout.flush()
    json.dump({"recs": allrec, "summary": {n: s for n, s in rows}},
              open(os.path.join(OUT, "results_stage26.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 84)
    print("D3 批量：同等准确率下省多少 oracle（通过线 = 30%）")
    print("=" * 84)
    print("%-10s %-10s %-12s %-12s %-10s %-10s %s"
          % ("数据集", "门控ACC", "同花费朴素", "ΔACC", "省oracle", "头寸", "判定(ΔACC>=0.05)"))
    for name, s in rows:
        d = s.get("delta_acc", float("nan"))
        ok = (d == d) and d >= 0.05
        print("%-10s %-10.3f %-12.3f %-12s %-10s %-10.3f %s"
              % (name, s["gate_acc"],
                 s.get("naive_at_gate", float("nan")) if s.get("naive_at_gate", float("nan")) == s.get("naive_at_gate", float("nan")) else float("nan"),
                 ("%+.4f" % d) if d == d else "n/a",
                 ("%.0f%%" % (100 * s["saved"])) if s["saved"] == s["saved"] else "n/a",
                 s["headroom"], "通过" if ok else "不通过"))
