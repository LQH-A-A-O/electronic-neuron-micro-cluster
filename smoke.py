# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp.taskstream import TaskStream
from exp.learners import NaiveLearner, ReservoirLearner, TieredLearner
from exp.run import run_trajectory, summarize
from mm import NeighborValidator

t0 = time.time()
for noise in (0.0, 0.3):
    st = TaskStream(n_tasks=6, label_noise=noise, seed=0)
    print("=== label_noise = %.1f ===" % noise)
    for name, make in (
        ("A_naive", lambda: NaiveLearner(20, seed=0)),
        ("B_replay", lambda: ReservoirLearner(20, seed=0, capacity=600)),
        ("C_tiered", lambda: TieredLearner(20, seed=0, capacity=600,
                                           validator=NeighborValidator(np.zeros((0,20)), np.zeros(0)))),
    ):
        L = make()
        acc, led = run_trajectory(L, st, steps_per_task=300)
        s = summarize(acc, led)
        print("  %-10s ACC=%.3f  BWT=%+.3f  新任务=%.3f  污染=%.3f  温备=%d  拒掉=%d"
              % (name, s["ACC_final"], s["BWT"], s["ACC_new_tail3"], s["pollution_final"],
                 s["store_final"], s["rejected_total"]))
print("耗时 %.1f 秒" % (time.time() - t0))
