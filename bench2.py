# -*- coding: utf-8 -*-
import sys, time, os
import numpy as np
rng = np.random.default_rng(0)
G = rng.normal(size=(129, 129)); G = G @ G.T + 129 * np.eye(129)
def t(fn, n=50):
    fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1e3
print("threads env: OPENBLAS=%s OMP=%s" % (os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("OMP_NUM_THREADS")))
print("inv          %.2f ms" % t(lambda: np.linalg.inv(G)))
print("solve(G,I)   %.2f ms" % t(lambda: np.linalg.solve(G, np.eye(129))))
print("cholesky     %.2f ms" % t(lambda: np.linalg.cholesky(G)))
print("qr(179x129)  %.2f ms" % t(lambda: np.linalg.qr(rng.normal(size=(179, 129)), mode="reduced")))
