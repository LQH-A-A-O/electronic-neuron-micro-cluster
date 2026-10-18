# -*- coding: utf-8 -*-
"""阶段 15：审计从「撤销」改成「降权」—— 阶段 14 指出的、目前最便宜的一步。

阶段 14 的发现：审计是**二元判决**（一致率 >= 0.8 留、< 0.8 撤），
而"撤掉一个不完美但有用的锚"的代价（判定退回最近邻）**比留着它更大** ——
额外的抽检证据反而让审计误杀了任务 1 的锚，总体从 0.529 掉到 0.365。

改法：不给锚"生杀"，给它一个 0~1 的**可信度权重**，随审计一致率连续变化。
  ch_anchor 里：  有效票 = cred * 票 + (1 - cred) * 0.5    （cred=0 = 弃权，不是反对）
  predict   里：  各锚的预测按 cred 加权平均
这样坏锚会被"稀释到没声音"，而不是被"处决"。

四组对照
  audit_revisit        阶段 13 终点（硬判决）
  weighted             只降权，不撤销
  quarantine           阶段 14 的观察区（对照）
  weighted_quarantine  **降权 + 观察区抽检**（应该兼得两者之长）
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.stage10 import N_OBS, N_TASK, ROUND, TOL, Tool, phi
from exp.stage11 import AUDIT_MIN, AuditRunner, make_stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SEEDS = [0, 1, 2]
MODES = ["hard", "weighted", "quarantine", "weighted_quarantine"]
CRED_BLEND = 0.5
CRED_USE = 0.25          # 可信度低于这个值，预测时当它不存在


class CTool(Tool):
    __slots__ = ("cred",)


class WRunner(AuditRunner):
    """把锚的"留/撤"换成连续可信度。"""

    def __init__(self, mode, seed, stream):
        super().__init__(mode, seed, stream)
        self.tools = [self._cred(t) for t in self.tools]
        self.cred_updates = 0

    @staticmethod
    def _cred(t):
        c = CTool(t.w, t.center, t.radius)
        c.cred = 1.0
        return c

    def try_compile(self, center):
        n0 = len(self.tools)
        super().try_compile(center)
        if len(self.tools) > n0:
            self.tools = list(self.tools[:n0]) + [self._cred(t) for t in self.tools[n0:]]

    # ---------------- 加权通道 ----------------
    def ch_anchor(self, x, y):
        if self.mode == "hard":
            return AuditRunner.ch_anchor(self, x, y)
        votes = []
        for tl in self.tools:
            if tl.covers(x):
                v = 1.0 if abs(tl.predict(x) - y) < TOL else 0.0
                w = getattr(tl, "cred", 1.0)
                votes.append(w * v + (1.0 - w) * 0.5)
        return float(np.mean(votes)) if votes else None

    def predict(self, X):
        if self.mode == "hard":
            return AuditRunner.predict(self, X)
        out = np.zeros(len(X))
        ax = np.array(self.accepted_x) if self.accepted_x else np.zeros((0, 4))
        ay = np.array(self.accepted_y) if self.accepted_y else np.zeros(0)
        for i, x in enumerate(X):
            # 可信度掉到地板以下 = 这个锚没资格说话，退回记忆检索
            hit = [tl for tl in self.tools
                   if tl.covers(x) and getattr(tl, "cred", 1.0) >= CRED_USE]
            if hit:
                ws = np.array([max(getattr(tl, "cred", 1.0), 1e-6) for tl in hit])
                ps = np.array([tl.predict(x) for tl in hit])
                out[i] = float((ws * ps).sum() / ws.sum())
            elif len(ax):
                j = int(np.argmin(np.max(np.abs(ax - x), axis=1)))
                out[i] = float(ay[j])
        return out

    # ---------------- 审计 -> 降权 ----------------
    def run_audit(self, truth_fn=None):
        if self.mode == "hard":
            return super().run_audit(truth_fn)
        if not self.tools or len(self.audit_x) < AUDIT_MIN:
            return
        X = np.array(self.audit_x)
        Y = np.array(self.audit_y)
        for tl in self.tools:
            k = np.max(np.abs(X - tl.center), axis=1) <= tl.radius
            if int(k.sum()) < AUDIT_MIN:
                continue
            pred = np.array([tl.predict(x) for x in X[k]])
            agree = float((np.abs(pred - Y[k]) < TOL).mean())
            old = getattr(tl, "cred", 1.0)
            tl.cred = CRED_BLEND * old + (1.0 - CRED_BLEND) * agree
            self.cred_updates += 1


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for seed in SEEDS:
        st = make_stream(seed)
        for mode in MODES:
            R = WRunner(mode, seed, st)
            accs, acc_t0, acc_t1, mincred = [], [], [], []
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
                if R.tools:
                    mincred.append(min(getattr(tl, "cred", 1.0) for tl in R.tools))
            recs.append(dict(mode=mode, seed=seed, ACC_final=accs[-1], ACC_task0=acc_t0[-1],
                             ACC_task1=(acc_t1[-1] if acc_t1 else float("nan")),
                             min_cred=(mincred[-1] if mincred else float("nan")),
                             tools=len(R.tools), side=R.side_queries,
                             demoted=R.demoted, strike_demoted=R.strike_demoted,
                             recovered=R.recovered, kept=len(R.accepted_x)))
        print("seed=%d 完成 %.0f 秒" % (seed, time.time() - t0)); sys.stdout.flush()

    json.dump(recs, open(os.path.join(OUT, "results_stage15.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 104)
    print("阶段 15：审计从「撤销」改成「降权」（同源污染 60% + 注入坏锚 / 5 任务 / 3 seeds）")
    print("=" * 104)
    print("%-20s %-14s %-12s %-12s %-11s %-9s %s"
          % ("组", "总体 ACC", "任务0 区域", "任务1 区域", "最低可信度", "抽检放行", "剩余锚"))
    for mode in MODES:
        rs = [r for r in recs if r["mode"] == mode]
        print("%-20s %.3f±%.3f    %.3f       %.3f       %-11.3f %-9.0f %.1f"
              % (mode, np.mean([r["ACC_final"] for r in rs]), np.std([r["ACC_final"] for r in rs]),
                 np.mean([r["ACC_task0"] for r in rs]), np.nanmean([r["ACC_task1"] for r in rs]),
                 np.nanmean([r["min_cred"] for r in rs]),
                 np.mean([r["recovered"] for r in rs]), np.mean([r["tools"] for r in rs])))
    print()
    print("总耗时 %.0f 秒 -> %s" % (time.time() - t0, os.path.join(OUT, "results_stage15.json")))


if __name__ == "__main__":
    main()
