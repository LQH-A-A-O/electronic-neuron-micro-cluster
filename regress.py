# -*- coding: utf-8 -*-
"""回归检查：阶段 2 改动之后，阶段 1 的数字没变。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp.taskstream import TaskStream
from exp.learners import ReservoirLearner, TieredLearner
from exp.run import run_trajectory, summarize
from mm import NeighborValidator

st = TaskStream(n_tasks=10, label_noise=0.4, center_scale=2.0, cluster_spread=1.8, seed=1000)
b = ReservoirLearner(20, seed=0, capacity=600)
acc, led = run_trajectory(b, st, steps_per_task=300)
print("B_replay  ACC=%.3f (阶段1 记录 0.390)  污染=%.3f" % (summarize(acc, led)["ACC_final"], summarize(acc, led)["pollution_final"]))
c = TieredLearner(20, seed=0, capacity=600, validator=NeighborValidator(np.zeros((0, 20)), np.zeros(0)), theta=0.7)
acc, led = run_trajectory(c, st, steps_per_task=300)
s = summarize(acc, led)
print("C_tiered  ACC=%.3f (阶段1 记录 0.615)  污染=%.3f  拒收=%d" % (s["ACC_final"], s["pollution_final"], s["rejected_total"]))
