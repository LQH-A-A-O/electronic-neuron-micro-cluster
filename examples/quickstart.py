# -*- coding: utf-8 -*-
"""30 秒看懂这套东西在干什么，以及怎么用。

    python examples/quickstart.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm import NeighborValidator, TieredMemory, audit_gate, format_audit

rng = np.random.default_rng(0)
DIM, N_CLASS, N = 16, 5, 600
centers = rng.normal(0, 3.0, (N_CLASS, DIM))

# 一批"已知正确"的历史数据 —— 这就是你要付的唯一成本（验证池）
pool_y = rng.integers(0, N_CLASS, 400)
pool_x = centers[pool_y] + rng.normal(0, 1.0, (400, DIM))

# 在线流：40% 的标签是错的（幻觉 / 数据源污染 / 模型自我强化）
y_true = rng.integers(0, N_CLASS, N)
keys = centers[y_true] + rng.normal(0, 1.0, (N, DIM))
values = y_true.copy()
dirty = rng.random(N) < 0.4
for i in np.flatnonzero(dirty):
    values[i] = (y_true[i] + rng.integers(1, N_CLASS)) % N_CLASS

print("在线流里有 %d/%d = %.0f%% 条记忆是错的\n" % (dirty.sum(), N, 100 * dirty.mean()))

# ---------------- 第一件事：先量验证器，再决定用不用 ----------------
r = audit_gate(keys, values, y_true,
               NeighborValidator(pool_x, pool_y, sim_floor=0.5, min_support=3),
               theta=0.7)
print(format_audit(r))

# ---------------- 然后才是接进记忆层 ----------------
mem = TieredMemory(validator=NeighborValidator(pool_x, pool_y, sim_floor=0.5, min_support=3),
                   theta=0.7, capacity_hot=None, capacity_warm=None)
for k, v in zip(keys, values):
    mem.observe(k, v)                 # 热备：照单全收
rep = mem.consolidate()               # 门控：决定谁能进温备
print("\n巩固结果：收下 %d 条，拒掉 %d 条（热备已清空）" % (rep.n_accepted, rep.n_rejected))
print("现在 replay_pool() 里是 %d 条经验证的记忆，可以直接拿去回放/检索。" % len(mem.replay_pool()))

# 对照：不做门控会怎样
naive_wrong = int(dirty.sum())
print("\n对照 —— 如果照单全收（无门控回放），温备里会有 %d/%d = %.0f%% 是错的；"
      % (naive_wrong, N, 100 * dirty.mean()))
print("      过门控之后，温备里错的是 %d 条。" % r["accepted_dirty"])
