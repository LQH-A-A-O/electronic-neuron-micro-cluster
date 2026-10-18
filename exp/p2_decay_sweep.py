# -*- coding: utf-8 -*-
"""P2：T5 的降权参数扫描 —— +0.227 有多脆？

T5 把审计打到记忆条目上，持续污染下 0.352 -> 0.579。但那三个参数是随手定的：
    一次抽检影响最近的 K_HIT 条     （默认 8）
    不一致时权重乘 DECAY            （默认 0.5）
    权重的下限 W_FLOOR              （默认 0.05）
这一轮扫前两个。判据：**ΔACC 相对 no_audit 是否稳定为正**。
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp.t5_audit_mem as T5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SEEDS = [0, 1, 2, 3, 4]
K_HITS = [3, 8, 16]
DECAYS = [0.3, 0.5, 0.8]

if __name__ == "__main__":
    base = []
    for seed in SEEDS:
        base.append(T5.run("no_audit", seed))
    b = float(np.mean([r["ACC_final"] for r in base]))
    print("no_audit 基线 ACC = %.3f" % b); sys.stdout.flush()
    res = []
    for k in K_HITS:
        for d in DECAYS:
            T5.K_HIT = k; T5.DECAY = d
            for seed in SEEDS:
                r = T5.run("audit_mem", seed)
                r.update(K_HIT=k, DECAY=d)
                res.append(r)
            rs = [r for r in res if r["K_HIT"] == k and r["DECAY"] == d]
            a = float(np.mean([r["ACC_final"] for r in rs]))
            print("K_HIT=%-3d DECAY=%.1f  ACC %.3f  ΔACC %+.4f  坏锚 %.1f 轮"
                  % (k, d, a, a - b, np.mean([r["bad_rounds"] for r in rs]))); sys.stdout.flush()
    json.dump({"baseline": b, "base_recs": base, "recs": res},
              open(os.path.join(OUT, "results_p2_decay.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 76)
    print("P2：降权参数敏感度（持续污染 every_task / 真实 MNIST / 5 seeds）")
    print("=" * 76)
    print("no_audit 基线 %.3f" % b)
    print("%-10s %-10s %-12s %-12s %s" % ("K_HIT", "DECAY", "ACC", "ΔACC", "判定(ΔACC>0.03)"))
    accs = []
    for k in K_HITS:
        for d in DECAYS:
            rs = [r for r in res if r["K_HIT"] == k and r["DECAY"] == d]
            a = float(np.mean([r["ACC_final"] for r in rs])); accs.append(a)
            print("%-10d %-10.1f %-12.3f %-12s %s" % (k, d, a, "%+.4f" % (a - b),
                                                      "收益" if a - b > 0.03 else "惰性"))
    print()
    print("ACC 跨度 = %.4f（最小 %.3f / 最大 %.3f）" % (max(accs) - min(accs), min(accs), max(accs)))
