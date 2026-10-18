# -*- coding: utf-8 -*-
import sys, time
import numpy as np
sys.path.insert(0, ".")
from exp.stage22 import loo_scores
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 128)); y = rng.integers(0, 2, 50); cls = [0, 1]
for _ in range(3): loo_scores(X, y, cls)
t0 = time.perf_counter()
for _ in range(20): loo_scores(X, y, cls)
print("loo_scores 单次 %.2f ms" % ((time.perf_counter() - t0) / 20 * 1e3))
A = np.hstack([X, np.ones((50, 1))])
G = A.T @ A + np.eye(129)
t0 = time.perf_counter()
for _ in range(20): np.linalg.inv(G)
print("inv(129x129) 单次 %.2f ms" % ((time.perf_counter() - t0) / 20 * 1e3))
t0 = time.perf_counter()
for _ in range(20): np.linalg.solve(G, np.eye(129))
print("solve(129x129, I) 单次 %.2f ms" % ((time.perf_counter() - t0) / 20 * 1e3))
t0 = time.perf_counter()
for _ in range(20): np.linalg.cholesky(G)
print("cholesky 单次 %.2f ms" % ((time.perf_counter() - t0) / 20 * 1e3))
t0 = time.perf_counter()
for _ in range(20): A @ G
print("matmul 50x129 @ 129x129 单次 %.2f ms" % ((time.perf_counter() - t0) / 20 * 1e3))
