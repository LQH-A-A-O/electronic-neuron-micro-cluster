# -*- coding: utf-8 -*-
"""阶段 4 冒烟：真实 MNIST 上跑一遍，看耗时和门控接受率。"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.realdata import load_mnist, RealClassStream
from exp.learners import NaiveLearner, ReservoirLearner, TieredLearner
from exp.run import run_trajectory, summarize
from mm import NeighborValidator

x, y, xt, yt = load_mnist()
print("MNIST", x.shape, xt.shape)
for noise in (0.0, 0.4):
    st = RealClassStream(x, y, xt, yt, n_tasks=5, classes_per_task=2,
                         n_train=400, n_holdout=200, n_test=200,
                         label_noise=noise, seed=1000)
    print("=== 真数据 label_noise=%.1f  dim=%d ===" % (noise, st.dim))
    for name, mk in (
        ("A_naive", lambda: NaiveLearner(st.dim, seed=0)),
        ("B_replay", lambda: ReservoirLearner(st.dim, seed=0, capacity=600)),
        ("C_tiered", lambda: TieredLearner(st.dim, seed=0, capacity=600,
                                           validator=NeighborValidator(np.zeros((0, st.dim)), np.zeros(0)), theta=0.7)),
    ):
        t0 = time.time()
        L = mk()
        acc, led = run_trajectory(L, st, steps_per_task=400)
        s = summarize(acc, led)
        print("  %-9s ACC=%.3f BWT=%+.3f 污染=%.3f 收下=%d 拒掉=%d  %.1fs"
              % (name, s["ACC_final"], s["BWT"], s["pollution_final"],
                 s["store_final"], s["rejected_total"], time.time() - t0))
