# -*- coding: utf-8 -*-
"""上线前体检：给一套数据，判定这套机制在你这儿值不值得上。

它把 25 个阶段的结论压缩成一次调用 —— 因为**大多数场景不该用它**，
而一个不会说"不该用"的工具是没用的。

关键：这里走的是**阶段 20/21 实际验证过的那条门控**（留一互证 + 已编译的工具 + 犹豫才花钱），
不是别的近似实现。第一版我图省事用了 TieredMemory 的默认路径，
结果在 MNIST（已知有效）上判成"边际" —— 错的工具比没有更糟。

用法
------------------------------------------------
你需要给出一批**已经标过**的数据（retrospective）：
    X_train/y_obs  线上收到的（可能被污染）
    y_true         这批数据你后来核对的真值（体检时当 oracle 用，不参与决策）
    X_test/y_test  独立的测试集

判据（都来自实测阶段）
------------------------------------------------
  1. 验证器漏放率 < 0.5          （阶段 2：与污染率无关，是硬合格线）
  2. 任务头寸 >= 0.15            （阶段 25：空间小时省标注的绝对收益就小）
  3. 门控吃掉头寸 >= 50%         （阶段 20-25：MNIST 49~84%，Covertype 29%）
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _proba(W, classes, X):
    A = np.hstack([np.atleast_2d(X), np.ones((len(np.atleast_2d(X)), 1))])
    M = A @ W.T
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)


def _fit(X, y, classes, ridge):
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    W = np.zeros((len(classes), A.shape[1]))
    for i, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        W[i] = Ginv @ (A.T @ t)
    return W


def _loo(X, y, classes, ridge):
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


def _knn_acc(Xq, yq, Xref, yref, k):
    if len(Xref) == 0:
        return float("nan")
    S = Xq @ Xref.T
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    nc = int(max(yq.max(), yref.max())) + 1
    pred = np.array([np.bincount(yref[r], minlength=nc).argmax() for r in idx])
    return float((pred == yq).mean())


def deployability_check(
    X_train, y_obs, y_true, X_test, y_test,
    tau: float = 0.60, hilo: float = 0.65, knn_k: int = 8,
    ridge: float = 1.0, round_size: int = 50,
) -> Dict[str, Any]:
    X = np.asarray(X_train, dtype=np.float64)
    y_obs = np.asarray(y_obs); y_true = np.asarray(y_true)
    Xq = np.asarray(X_test, dtype=np.float64); yq = np.asarray(y_test)
    n, d = X.shape
    classes_all = sorted(set(y_true.tolist()))

    raw_acc = _knn_acc(Xq, yq, X, y_obs, knn_k)
    full_acc = _knn_acc(Xq, yq, X, y_true, knn_k)
    headroom = full_acc - raw_acc

    bx, by = [], []
    W = None
    classes = []
    n_clean = n_dirty = acc_clean = acc_dirty = 0
    oracle = 0
    for s0 in range(0, n, round_size):
        xb = X[s0:s0 + round_size]; yb = y_obs[s0:s0 + round_size]
        yt = y_true[s0:s0 + round_size]
        cls = sorted(set(yb.tolist()))
        pred = None
        if len(cls) >= 2 and len(xb) >= 10:
            P = _loo(xb, yb, cls, ridge)
            pred = np.array(cls)[P.argmax(axis=1)]
        if W is not None:
            Pt = _proba(W, classes, xb)
            tpred = np.array(classes)[Pt.argmax(axis=1)]
            tcover = Pt.max(axis=1) >= tau
        else:
            tpred, tcover = None, np.zeros(len(xb), dtype=bool)
        for i in range(len(xb)):
            y = int(yb[i]); dirty = y != int(yt[i])
            a = (1.0 if int(tpred[i]) == y else 0.0) if (tpred is not None and tcover[i]) else None
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
            if sc is None or (1.0 - hilo) < sc < hilo:
                ok = (y == int(yt[i]))      # 花一次 oracle（体检时真值可得）
                oracle += 1
            else:
                ok = sc >= hilo
            if dirty:
                n_dirty += 1; acc_dirty += int(not ok)
            else:
                n_clean += 1; acc_clean += int(ok)
            if ok:
                bx.append(xb[i]); by.append(y)
        if len(by) >= 60:
            classes = sorted(set(by))
            if len(classes) >= 2:
                W = _fit(np.array(bx), np.array(by), classes, ridge)

    KX = np.array(bx) if bx else np.zeros((0, d))
    KY = np.array(by) if by else np.zeros(0, dtype=int)
    gate_acc = _knn_acc(Xq, yq, KX, KY, knn_k)
    captured = (gate_acc - raw_acc) / headroom if headroom > 1e-9 else float("nan")
    # acc_dirty 累计的是"脏样本被拒"的次数 -> 漏放率 = 1 - 它
    fa = 1.0 - (acc_dirty / n_dirty) if n_dirty else float("nan")
    fr = 1.0 - (acc_clean / n_clean) if n_clean else float("nan")

    reasons = []
    if n_dirty == 0:
        reasons.append("这批数据里没有脏标签，验证器无法评估")
    elif fa > 0.5:
        reasons.append("验证器漏放率 %.2f > 0.5：用它比不做门控更差（阶段 2）" % fa)
    if headroom < 0.15:
        reasons.append("头寸只有 %.3f（全标注也只比不做高这么多）：省标注的绝对收益太小（阶段 25）" % headroom)
    if captured == captured and captured < 0.5:
        reasons.append("门控只吃下头寸的 %.0f%%（< 50%%）：经济性不达标（阶段 20-25）" % (100 * captured))
    if d < 64:
        reasons.append("维度只有 %d：实测低维数据上门控明显变弱（阶段 25）" % d)
    verdict = "可以用" if not reasons else ("不该用" if len(reasons) >= 2 else "边际，谨慎")
    return dict(n=n, dim=d, raw_acc=raw_acc, gate_acc=gate_acc, full_acc=full_acc,
                headroom=headroom, captured=captured, kept=len(bx), oracle=oracle,
                false_accept_rate=fa, false_reject_rate=fr,
                p_dirty=n_dirty / max(n, 1), verdict=verdict, reasons=reasons)


def format_deployability(r: Dict[str, Any]) -> str:
    g = lambda v: ("%.3f" % v) if v == v else "n/a"
    lines = [
        "上线前体检报告",
        "  样本 %d 条 / %d 维 / 脏标签 %.0f%%" % (r["n"], r["dim"], 100 * r["p_dirty"]),
        "  什么都不做      %s" % g(r["raw_acc"]),
        "  门控（过滤后）  %s   （留下 %d 条，花了 %d 次 oracle）" % (g(r["gate_acc"]), r["kept"], r["oracle"]),
        "  全标注上界      %s" % g(r["full_acc"]),
        "  头寸            %s   （全标注 - 不做）" % g(r["headroom"]),
        "  门控吃下头寸    %s" % (("%.0f%%" % (100 * r["captured"])) if r["captured"] == r["captured"] else "n/a"),
        "  验证器漏放/误拒 %s / %s" % (g(r["false_accept_rate"]), g(r["false_reject_rate"])),
        "",
        "  判定：%s" % r["verdict"],
    ]
    for x in r["reasons"]:
        lines.append("    - " + x)
    return "\n".join(lines)
