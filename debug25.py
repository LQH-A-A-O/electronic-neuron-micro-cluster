# -*- coding: utf-8 -*-
"""为什么门控在表格数据上变弱了？量一下两个免费通道的"可分性"。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp.stage25 as S25
from exp.hallucination import inject, weak_model_confusion
from exp.realdata import RealClassStream, load_mnist

def sep_loo(X, y_true, y_obs, classes, ridge=1.0):
    """留一通道：干净样本 vs 脏样本 被判"标签一致"的比例差。"""
    P = S25.loo_scores(X, y_obs, classes, ridge=ridge)
    pred = np.array(classes)[P.argmax(axis=1)]
    agree = (pred == y_obs)
    dirty = y_obs != y_true
    return agree[~dirty].mean(), agree[dirty].mean()

print("数据        维度  干净一致率  脏一致率  可分性(差)")
# Covertype
tasks = S25.build_tasks(0)
for t, tk in enumerate(tasks):
    c, d = sep_loo(tk.x_train, tk.y_train_true, tk.y_train, tk.classes)
    print("Covertype   %-5d %-11.3f %-9.3f %.3f" % (S25.DIM, c, d, c - d))
# MNIST
MN = load_mnist(); CONF, _ = weak_model_confusion()
_rng = np.random.default_rng(2026)
PROJ = _rng.normal(0, 1.0 / np.sqrt(784), (784, 128))
def feat(X):
    Z = np.asarray(X, dtype=np.float64) @ PROJ
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=5, classes_per_task=2,
                     n_train=400, n_holdout=200, n_test=200, label_noise=0.0, seed=1000)
rng = np.random.default_rng(777)
for tk in st:
    yt = inject(tk.y_train_true, tk.classes, 0.4, rng, "structured", CONF)
    c, d = sep_loo(feat(tk.x_train), tk.y_train_true, yt, tk.classes)
    print("MNIST       %-5d %-11.3f %-9.3f %.3f" % (128, c, d, c - d))
