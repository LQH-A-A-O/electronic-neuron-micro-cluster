# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp.taskstream import TaskStream
from exp.learners import NaiveLearner, ReservoirLearner, TieredLearner
from exp.run import run_trajectory, summarize
from mm import NeighborValidator

print("%-22s %-10s %-8s %-8s %-8s" % ("难度(center/spread)", "learner", "ACC", "BWT", "新任务"))
for cs, sp in ((4.0, 1.0), (2.5, 1.5), (2.0, 1.8), (1.6, 2.0)):
    st = TaskStream(n_tasks=10, label_noise=0.0, center_scale=cs, cluster_spread=sp, seed=0)
    row = []
    for name, make in (
        ("A_naive", lambda: NaiveLearner(20, seed=0)),
        ("B_replay", lambda: ReservoirLearner(20, seed=0, capacity=600)),
        ("C_tiered", lambda: TieredLearner(20, seed=0, capacity=600,
                                           validator=NeighborValidator(np.zeros((0,20)), np.zeros(0)))),
    ):
        L = make()
        acc, led = run_trajectory(L, st, steps_per_task=300)
        s = summarize(acc, led)
        print("%-22s %-10s ACC=%.3f BWT=%+.3f 新=%.3f" % ("%.1f/%.1f" % (cs, sp), name,
              s["ACC_final"], s["BWT"], s["ACC_new_tail3"]))
