# -*- coding: utf-8 -*-
"""纯 numpy 小 MLP：共享主干 + 可动态扩容的分类头。

不依赖 torch/sklearn —— 工具箱里的 numpy 就够，跑得也快。
"""

from __future__ import annotations

import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


class MLP:
    def __init__(self, dim, hidden=(128, 64), seed=0, lr=3e-3, weight_decay=0.0):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        self.W1 = rng.normal(0, np.sqrt(2.0 / dim), (dim, h1))
        self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = np.zeros((h2, 0))
        self.b3 = np.zeros(0)
        self.n_out = 0
        self.lr = lr
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        self._t = 0

    # ------------------------------------------------------------ 结构
    def expand(self, n_new, seed=0):
        rng = np.random.default_rng(1000 + self.n_out + seed)
        add = rng.normal(0, 0.01, (self.W3.shape[0], n_new))
        self.W3 = np.concatenate([self.W3, add], axis=1)
        self.b3 = np.concatenate([self.b3, np.zeros(n_new)])
        self.n_out += n_new

    # ------------------------------------------------------------ 前向
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        logits = a2 @ self.W3 + self.b3
        return logits, (X, z1, a1, z2, a2)

    def predict(self, X):
        if self.n_out == 0:
            return np.zeros(len(X), dtype=int)
        return np.argmax(self.forward(X)[0], axis=1)

    # ------------------------------------------------------------ 反向
    def step(self, X, y):
        n = len(X)
        logits, (X_, z1, a1, z2, a2) = self.forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        dz = probs.copy()
        dz[np.arange(n), y] -= 1.0
        dz /= n

        gW3 = a2.T @ dz
        gb3 = dz.sum(axis=0)
        da2 = dz @ self.W3.T
        dz2 = da2 * (z2 > 0)
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        gW1 = X_.T @ dz1
        gb1 = dz1.sum(axis=0)

        grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2, W3=gW3, b3=gb3)
        self._adam(grads)
        return float(-np.log(np.maximum(probs[np.arange(n), y], 1e-12)).mean())

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            if self.weight_decay and k.startswith("W"):
                g = g + self.weight_decay * getattr(self, k)
            m = self._m.get(k)
            v = self._v.get(k)
            if m is None or m.shape != g.shape:
                m = np.zeros_like(g)
                v = np.zeros_like(g)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            self._m[k] = m
            self._v[k] = v
            mh = m / (1 - beta1 ** self._t)
            vh = v / (1 - beta2 ** self._t)
            setattr(self, k, getattr(self, k) - self.lr * mh / (np.sqrt(vh) + eps))
