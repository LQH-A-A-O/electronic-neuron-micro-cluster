# -*- coding: utf-8 -*-
"""阶段 14：取消单锚否决权 —— 直接对着阶段 13 的根因。

阶段 13 的结论：坏锚在**入口**就把正确知识拒了，事后审计补不回来。
根因在门控的取样策略：只在 score 落在 [0.35,0.65] 的"犹豫区"才花预算，
而 score=0（锚和 peer 一致说它脏）属于"自信的拒绝"，**从来不被验证**。

于是三组对照：
  audit_revisit      阶段 13 的终点（0.100）
  quarantine         单锚否决 -> 观察区；**对观察区按比例抽检**，抽检干净的放行
  quarantine_strike  同上 + 锚的误判累计到阈值就自动降级（并连带撤回它的从犯）
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
MODES = ["audit_revisit", "quarantine", "quarantine_strike"]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode in MODES:
            R = AuditRunner(mode, seed, st)
            accs, acc_t0, acc_t1 = [], [], []
            for t in range(N_TASK):
                xo, yo, _ = st.obs[t]
                tf = lambda x, _t=t: float((phi(x) @ st.ws[_t]).ravel()[0])
                for s0 in range(0, N_OBS, ROUND):
                    R.process_round(xo[s0:s0 + ROUND], yo[s0:s0 + ROUND], tf)
                    R.try_compile(st.centers[t])
                a, a0, a1 = [], [], []
                for j in range(t + 1):
                    xq, yq = st.queries[j]
                    hit = np.abs(R.predict(xq) - yq) < TOL
                    a.append(float(hit.mean()))
                    if j == 0:
                        a0.append(float(hit.mean()))
                    if j == 1:
                        a1.append(float(hit.mean()))
                accs.append(float(np.mean(a)))
                acc_t0.append(float(np.mean(a0)))
                if a1:
                    acc_t1.append(float(np.mean(a1)))
            recs.append(dict(mode=mode, seed=seed, ACC_final=accs[-1], ACC_task0=acc_t0[-1],
                             ACC_task1=(acc_t1[-1] if acc_t1 else float("nan")),
                             side=R.side_queries, demoted=R.demoted, retracted=R.retracted,
                             recovered=R.recovered, strike_demoted=R.strike_demoted,
                             quar_left=len(R.quarantine), kept=len(R.accepted_x),
                             tools_left=len(R.tools)))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage14.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 104)
    print("阶段 14：取消单锚否决权（同源污染 60% + 注入坏锚 / 5 任务 / 3 seeds）")
    print("=" * 104)
    print("%-18s %-15s %-15s %-9s %-9s %-9s %s" % ("组", "总体 ACC", "任务0 区域 ACC", "抽检放行", "误判降级", "side 查询", "剩余记忆"))
    for mode in MODES:
        rs = [r for r in recs if r["mode"] == mode]
        print("%-18s %.3f±%.3f      %.3f±%.3f      %-9.0f %-9.1f %-9.0f %.0f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["ACC_task0"] for r in rs]), np.std([r["ACC_task0"] for r in rs]),
                 np.mean([r["recovered"] for r in rs]), np.mean([r["strike_demoted"] for r in rs]),
                 np.mean([r["side"] for r in rs]), np.mean([r["kept"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage14.json")))


if __name__ == "__main__":
    main()
