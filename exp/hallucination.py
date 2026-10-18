# -*- coding: utf-8 -*-
"""三种"幻觉"的生成方式。

阶段 1~4 用的都是「均匀随机翻转」—— 把标签随机换成另一个类。
这是最偷懒的噪声模型：真实的模型不会把 0 认成 7，它会在**相似的类之间**犯迷糊（4/9、3/5、7/1…）。

这里做两件事：
  1. 训一个"弱模型"，把它的**混淆矩阵**读出来 —— 那就是真实错误的分布
  2. 用这个混淆矩阵生成**结构化幻觉**：错误率仍是 p（与均匀噪声严格可比），
     但犯错的方向变成真实的混淆方向
另外提供第三种：直接把弱模型的预测当标签（"伪标签"），错误率由模型自己决定。
"""
from __future__ import annotations

import os

import numpy as np

from exp.mlp import MLP

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def weak_model_confusion(seed=0, n_train=12000, steps=1500, hidden=(128, 64), cache=True):
    """训一个小 MLP，返回 (混淆矩阵 C 10x10, 准确率)。C[i,j] = 真 i 被认成 j 的比例。"""
    path = os.path.join(DATA, "confusion_mlp.npz")
    if cache and os.path.exists(path):
        z = np.load(path)
        return z["conf"], float(z["acc"])
    x = np.load(os.path.join(DATA, "mnist_train.npy")).astype(np.float32)[:n_train] / 255.0
    y = np.load(os.path.join(DATA, "mnist_train_y.npy")).astype(np.int64)[:n_train]
    xt = np.load(os.path.join(DATA, "mnist_test.npy")).astype(np.float32)[:5000] / 255.0
    yt = np.load(os.path.join(DATA, "mnist_test_y.npy")).astype(np.int64)[:5000]
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    x, xt = (x - mu) / sd, (xt - mu) / sd
    model = MLP(x.shape[1], hidden=hidden, seed=seed, lr=3e-3)
    model.expand(10)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(x), size=64)
        model.step(x[idx], y[idx])
    pred = model.predict(xt)
    conf = np.zeros((10, 10))
    for c in range(10):
        m = yt == c
        if m.any():
            conf[c] = np.bincount(pred[m], minlength=10) / m.sum()
    acc = float((pred == yt).mean())
    if cache:
        np.savez(path, conf=conf, acc=acc)
    return conf, acc


def _fallback(classes, c, rng):
    others = [k for k in classes if k != c]
    return int(rng.choice(others)) if others else int(c)


def inject(y_true, classes, p, rng, mode, conf=None):
    """按 mode 生成脏标签。p = 被污染的样本比例。"""
    y = np.array(y_true).copy()
    if mode == "none" or p <= 0:
        return y
    hit = rng.random(len(y)) < p
    for i in np.flatnonzero(hit):
        c = int(y_true[i])
        if mode == "uniform":
            y[i] = _fallback(classes, c, rng)
        elif mode == "structured":
            row = np.zeros(10)
            row[list(classes)] = conf[c, list(classes)]
            row[c] = 0.0
            s = row.sum()
            if s <= 1e-9:
                y[i] = _fallback(classes, c, rng)
            else:
                y[i] = int(rng.choice(10, p=row / s))
        else:
            raise KeyError(mode)
    return y


def pseudolabel_batch(model, x, mu, sd):
    return model.predict((x / 255.0 - mu) / sd)
