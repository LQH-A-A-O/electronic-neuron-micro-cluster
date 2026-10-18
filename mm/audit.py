# -*- coding: utf-8 -*-
"""上线前的体检：先量你的验证器，再决定要不要用这套门控。

阶段 2 的判据（先推公式、再实测验证过）：
    验证器的错误率（漏放 + 误拒）如果超过 50%，门控比不做门控更差；
    q=1（验证器完全反着来）时甚至比"完全没有记忆"还差。

所以使用顺序永远是：**先把这段跑一遍，看 verdict，再决定接不接。**
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .memory import GateReport, TieredMemory, Validator


def audit_gate(keys: np.ndarray, values: Sequence[Any], true_values: Sequence[Any],
               validator: Validator, theta: float = 0.7,
               truth_fn=None) -> dict:
    """只做一次巩固，把门控的诊断指标打出来。

    keys/values   : 你的候选记忆（可以是真实历史数据，也可以是回放数据）
    true_values   : 事后才知道的真值；**只用于统计，绝不参与打分**
    validator     : 你要评估的那个验证器（打分时它看不到 true_values）
    """
    tv = list(true_values)
    if truth_fn is None:
        truth_fn = lambda e: e.value != tv[int(e.step)]      # noqa: E731
    mem = TieredMemory(validator=validator, theta=theta,
                       capacity_hot=None, capacity_warm=None)
    for i, (k, v) in enumerate(zip(keys, values)):
        mem.observe(k, v, step=i)
    rep: GateReport = mem.consolidate(truth_fn=truth_fn)

    fa = rep.false_accept_rate
    fr = rep.false_reject_rate
    n_dirty = rep.accepted_dirty + rep.rejected_dirty
    n_clean = rep.accepted_clean + rep.rejected_clean
    err = np.nanmean([fa, fr]) if (fa == fa or fr == fr) else float("nan")

    if fa != fa:
        verdict = "无法评估（这批样本里没有脏数据）"
    elif fa > 0.5:
        verdict = "不要用：漏放率 > 50%，门控会比不做门控更差"
    elif fa < 0.2:
        verdict = "可以用：漏放率 < 20%"
    else:
        verdict = "边际可用：建议提高 theta 或换更好的验证信号"

    return {
        "n_candidates": rep.n_candidates,
        "n_dirty": n_dirty, "n_clean": n_clean,
        "accept_rate": rep.accept_rate,
        "false_accept_rate": fa,        # 漏放：脏的进了温备
        "false_reject_rate": fr,        # 误拒：干净的没进去
        "validator_error_rate": float(err),
        "accepted_dirty": rep.accepted_dirty,
        "warm_size": len(mem.warm),
        "theta": theta,
        "verdict": verdict,
    }


def format_audit(r: dict) -> str:
    return "\n".join([
        "门控体检报告",
        "  候选记忆      %d 条（脏 %d / 净 %d）" % (r["n_candidates"], r["n_dirty"], r["n_clean"]),
        "  theta         %.2f" % r["theta"],
        "  接受率        %.1f%%" % (100 * r["accept_rate"]),
        "  漏放率        %.3f   (脏的进了温备 —— 越低越好)" % r["false_accept_rate"],
        "  误拒率        %.3f   (干净的没进去 —— 只浪费，不致命)" % r["false_reject_rate"],
        "  验证器错误率  %.3f   (阶段 2 的判据线 = 0.5)" % r["validator_error_rate"],
        "  结论          %s" % r["verdict"],
    ])
