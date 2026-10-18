# -*- coding: utf-8 -*-
"""把 MNIST 拉下来并导出成 npy，之后实验就不再依赖 torch。"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

xs, ys = os.path.join(DATA, "mnist_train.npy"), os.path.join(DATA, "mnist_train_y.npy")
if os.path.exists(xs) and os.path.exists(ys):
    print("已存在，跳过下载")
else:
    from torchvision import datasets
    tr = datasets.MNIST(root=os.path.join(DATA, "raw"), train=True, download=True)
    te = datasets.MNIST(root=os.path.join(DATA, "raw"), train=False, download=True)
    np.save(xs, tr.data.numpy().reshape(-1, 784).astype(np.uint8))
    np.save(ys, tr.targets.numpy().astype(np.int64))
    np.save(os.path.join(DATA, "mnist_test.npy"), te.data.numpy().reshape(-1, 784).astype(np.uint8))
    np.save(os.path.join(DATA, "mnist_test_y.npy"), te.targets.numpy().astype(np.int64))
    print("下载完成")

x = np.load(xs); y = np.load(ys)
xt = np.load(os.path.join(DATA, "mnist_test.npy")); yt = np.load(os.path.join(DATA, "mnist_test_y.npy"))
print("train", x.shape, x.dtype, "labels", np.bincount(y))
print("test ", xt.shape, "labels", np.bincount(yt))
print("像素范围", x.min(), x.max(), "均值", round(float(x.mean()), 2))
