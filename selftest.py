# -*- coding: utf-8 -*-
"""阶段 0 自检：证明这套门控真的会做事，而不是像原演示那样走个过场。

每一条断言都对应原演示的一个具体缺陷。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm import (TieredMemory, NeighborValidator, ValueOnlyValidator,
                NoisyOracleValidator, cosine)

N_CLASS = 5
DIM = 20
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (("  " + detail) if detail else ""))


def make_pool(n, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(N_CLASS, DIM))
    y = rng.integers(0, N_CLASS, size=n)
    x = centers[y] + rng.normal(0, 1.0, size=(n, DIM))
    return x, y, centers


def make_hot(centers, n, dirty_frac, seed):
    """造一批待巩固的热备条目：dirty 的那些 key 是真的、标签是错的（幻觉）。"""
    rng = np.random.default_rng(seed)
    hot = []
    n_dirty = int(round(n * dirty_frac))
    truth = {}
    for i in range(n):
        c = int(rng.integers(0, N_CLASS))
        key = centers[c] + rng.normal(0, 1.0, size=DIM)
        if i < n_dirty:
            wrong = int((c + 1 + rng.integers(0, N_CLASS - 1)) % N_CLASS)
            hot.append((key, wrong, c, True))     # (key, 记下的值, 真值, 是否脏)
        else:
            hot.append((key, c, c, False))
    return hot


print("=" * 72)
print("阶段 0 自检：三层信任记忆 + 验证门控")
print("=" * 72)

pool_x, pool_y, centers = make_pool(500, seed=1)
hot = make_hot(centers, 200, dirty_frac=0.40, seed=2)
n_dirty = sum(1 for h in hot if h[3])

# ---------------------------------------------------------------- 检查 1
print("\n[1] 正确验证器（到留出池看 key 落点的真标签）能不能挡住 40% 的脏标签？")
v = NeighborValidator(pool_x, pool_y, sim_floor=0.5, min_support=3, topk=8)
mem = TieredMemory(v, theta=0.7)
for key, val, _t, _d in hot:
    mem.observe(key, val)
rep = mem.consolidate(truth_fn=lambda e: False)
# 重新跑一遍，这次带真实的脏判据
mem2 = TieredMemory(NeighborValidator(pool_x, pool_y), theta=0.7)
truth_map = {}
for key, val, t, d in hot:
    e = mem2.observe(key, val)
    truth_map[id(e)] = d
rep = mem2.consolidate(truth_fn=lambda e: truth_map[id(e)])
print("      候选 %d 条（其中脏 %d 条），收下 %d 条，拒掉 %d 条，接受率 %.1f%%"
      % (rep.n_candidates, n_dirty, rep.n_accepted, rep.n_rejected, 100 * rep.accept_rate))
dirty_in = rep.accepted_dirty if rep.accepted_dirty >= 0 else -1
print("      被收下的条目里，脏的只有 %d 条" % dirty_in)
check("脏标签几乎全被拦住", dirty_in <= 0.05 * max(rep.n_accepted, 1),
      "收下 %d 条，脏 %d 条" % (rep.n_accepted, dirty_in))
check("干净条目大部分被收下", rep.n_accepted >= 0.8 * (rep.n_candidates - n_dirty),
      "净 %d 条，收下 %d 条" % (rep.n_candidates - n_dirty, rep.n_accepted))

# ---------------------------------------------------------------- 检查 2
print("\n[2] 原演示那套验证器（只看 value 像不像已知答案）会发生什么？")
vo = ValueOnlyValidator(known_values=list(range(N_CLASS)))
mem3 = TieredMemory(vo, theta=0.7)
tm3 = {}
for key, val, _t, d in hot:
    e = mem3.observe(key, val)
    tm3[id(e)] = d
rep3 = mem3.consolidate(truth_fn=lambda e: tm3[id(e)])
print("      接受率 %.1f%%，收下 %d 条，其中脏 %d 条（脏占比 %.1f%%）"
      % (100 * rep3.accept_rate, rep3.n_accepted,
         rep3.accepted_dirty, 100.0 * rep3.accepted_dirty / max(rep3.n_accepted, 1)))
check("复现了原演示的失败：门控全放行", rep3.accept_rate > 0.99)
check("并且把脏记忆原样收进温备", rep3.accepted_dirty == n_dirty)

# ---------------------------------------------------------------- 检查 3
print("\n[3] 无法验证时应当拒绝，而不是默认通过")
mem4 = TieredMemory(NeighborValidator(np.zeros((0, DIM)), np.zeros(0)), theta=0.7)
mem4.observe(centers[0], 0)
rep4 = mem4.consolidate()
check("留出池为空 -> 全部不可验证 -> 一条都不收",
      rep4.n_accepted == 0 and rep4.n_unverifiable == 1,
      "accepted=%d unverifiable=%d" % (rep4.n_accepted, rep4.n_unverifiable))

# ---------------------------------------------------------------- 检查 4
print("\n[4] 读优先级与冷备只读")
mem5 = TieredMemory(NeighborValidator(pool_x, pool_y), theta=0.7)
mem5.load_cold([(centers[0], 0), (centers[1], 1)])
mem5.observe(centers[2], 2)
mem5.consolidate()
got = mem5.read(centers[2])
check("热备清空后，读命中的是温备", got is not None and got[1] == "warm", str(got))
got_cold = mem5.read(centers[0])
check("冷备里没有的、温备也没有的，会落到 cold", got_cold is not None and got_cold[1] == "cold",
      str(got_cold))
check("冷备条数不受门控影响", len(mem5.cold) == 2, "cold=%d" % len(mem5.cold))

# ---------------------------------------------------------------- 检查 5
print("\n[5] 验证器被打嗝（不可靠 oracle）时，门控会随之退化吗？")
for noise in (0.0, 0.2, 0.5):
    nv = NoisyOracleValidator(NeighborValidator(pool_x, pool_y), noise=noise, seed=7)
    m = TieredMemory(nv, theta=0.7)
    tm = {}
    for key, val, _t, d in hot:
        e = m.observe(key, val)
        tm[id(e)] = d
    r = m.consolidate(truth_fn=lambda e: tm[id(e)])
    print("      oracle 噪声 %.0f%% -> 接受率 %.1f%%，收下 %d 条，其中脏 %d 条"
          % (100 * noise, 100 * r.accept_rate, r.n_accepted, r.accepted_dirty))

print("\n" + "=" * 72)
print("PASS %d / FAIL %d" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项: " + ", ".join(FAIL))
print("=" * 72)
sys.exit(1 if FAIL else 0)
