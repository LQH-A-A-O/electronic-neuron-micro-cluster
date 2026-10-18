# -*- coding: utf-8 -*-
"""拉 Fashion-MNIST，导出 npy（和 MNIST 同管线，用来验 D3：结论不是 MNIST 的特例）。"""
import os
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
xs, ys = os.path.join(DATA, "fmnist_train.npy"), os.path.join(DATA, "fmnist_train_y.npy")
if not (os.path.exists(xs) and os.path.exists(ys)):
    from torchvision import datasets
    tr = datasets.FashionMNIST(root=os.path.join(DATA, "raw_f"), train=True, download=True)
    te = datasets.FashionMNIST(root=os.path.join(DATA, "raw_f"), train=False, download=True)
    np.save(xs, tr.data.numpy().reshape(-1, 784).astype(np.uint8))
    np.save(ys, tr.targets.numpy().astype(np.int64))
    np.save(os.path.join(DATA, "fmnist_test.npy"), te.data.numpy().reshape(-1, 784).astype(np.uint8))
    np.save(os.path.join(DATA, "fmnist_test_y.npy"), te.targets.numpy().astype(np.int64))
x = np.load(xs); y = np.load(ys)
print("Fashion-MNIST train", x.shape, "labels", np.bincount(y))
