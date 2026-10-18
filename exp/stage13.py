# -*- coding: utf-8 -*-
"""阶段 13：污染溯源 + 连带撤回。

阶段 11 发现：主动抽检能**发现**坏锚（降级 0.3 -> 2.3），但救不回准确率 ——
因为审计只撤了锚，**没撤被这个坏锚放行的那批脏记忆**。从犯还在库里。

机制：给每条被收下的记忆记下"当时是哪个锚担保的"；
审计判定某个锚有问题时，把它担保过的记忆一并撤回。
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.stage10 import N_OBS, N_TASK, ROUND, TOL, phi
from exp.stage11 import AuditRunner, make_stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SEEDS = [0, 1, 2]
MODES = ["no_audit", "audit_active", "audit_trace", "audit_revisit"]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode in MODES:
            R = AuditRunner(mode, seed, st)
            accs, acc_t0 = [], []
            for t in range(N_TASK):
                xo, yo, _ = st.obs[t]
                tf = lambda x, _t=t: float((phi(x) @ st.ws[_t]).ravel()[0])
                for s0 in range(0, N_OBS, ROUND):
                    R.process_round(xo[s0:s0 + ROUND], yo[s0:s0 + ROUND], tf)
                    R.try_compile(st.centers[t])
                a, a0 = [], []
                for j in range(t + 1):
                    xq, yq = st.queries[j]
                    hit = np.abs(R.predict(xq) - yq) < TOL
                    a.append(float(hit.mean()))
                    if j == 0:
                        a0.append(float(hit.mean()))
                accs.append(float(np.mean(a)))
                acc_t0.append(float(np.mean(a0)))
            recs.append(dict(mode=mode, seed=seed, ACC_final=accs[-1], ACC_task0=acc_t0[-1],
                             side=R.side_queries, forced=R.forced_queries,
                             demoted=R.demoted, recompiled=R.recompiled,
                             retracted=R.retracted, revisited=R.revisited,
                             kept=len(R.accepted_x),
                             tools_left=len(R.tools)))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage13.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 104)
    print("阶段 13：污染溯源 + 连带撤回（同源污染 60% + 注入坏锚 / 5 任务 / 3 seeds）")
    print("=" * 104)
    print("%-14s %-15s %-15s %-9s %-9s %-9s %s" % ("组", "总体 ACC", "任务0 区域 ACC", "降级", "撤回", "重审收下", "剩余记忆"))
    for mode in MODES:
        rs = [r for r in recs if r["mode"] == mode]
        print("%-14s %.3f±%.3f      %.3f±%.3f      %-9.1f %-9.0f %-9.0f %.0f"
              % (mode,
                 np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["ACC_task0"] for r in rs]), np.std([r["ACC_task0"] for r in rs]),
                 np.mean([r["demoted"] for r in rs]), np.mean([r["retracted"] for r in rs]),
                 np.mean([r.get("revisited", 0) for r in rs]),
                 np.mean([r["kept"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage13.json")))


if __name__ == "__main__":
    main()
