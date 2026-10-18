# -*- coding: utf-8 -*-
"""用两个已知答案的数据集验证体检工具：它必须给出**不同且正确**的判定。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm import deployability_check, format_deployability
from exp.hallucination import inject, weak_model_confusion
from exp.realdata import load_mnist

MN = load_mnist(); CONF, _ = weak_model_confusion()
rng = np.random.default_rng(2026)
PROJ = rng.normal(0, 1.0 / np.sqrt(784), (784, 128))
def feat(X):
    Z = np.asarray(X, dtype=np.float64) @ PROJ
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

rng2 = np.random.default_rng(1)
cls = [0, 1]
idx = np.concatenate([np.flatnonzero(MN[1] == c)[:300] for c in cls])
X = feat(MN[0][idx]); yt = MN[1][idx].copy()
yo = inject(yt, cls, 0.4, rng2, "structured", CONF)
it = np.concatenate([np.flatnonzero(MN[3] == c)[:150] for c in cls])
print("================ MNIST (128维像素特征)  已知答案：门控有效")
print(format_deployability(deployability_check(X, yo, yt, feat(MN[2][it]), MN[3][it])))

g = lambda n: np.load(os.path.join("data", n))
Xtr, ytr = g("cov_tr.npy")[:800].astype(np.float64), g("cov_tr_y.npy")[:800]
Xte, yte = g("cov_te.npy")[:400].astype(np.float64), g("cov_te_y.npy")[:400]
yt2 = ytr.copy()
yo2 = inject(yt2, sorted(set(yt2.tolist())), 0.4, rng2, "uniform")
print()
print("================ Covertype (54维表格)  已知答案：经济性不达标")
print(format_deployability(deployability_check(Xtr, yo2, yt2, Xte, yte)))
