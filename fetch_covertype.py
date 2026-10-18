# -*- coding: utf-8 -*-
"""下载并准备 Forest Covertype（表格数据，54 维，7 类）—— G5 用。

为什么用它：22 个阶段全在 28x28 灰度图上。Covertype 是**表格数据**，
特征是"海拔/坡度/土壤类型/日照"这类真实地理量测，和像素完全不同的性质。
"""
import gzip, os, io
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
gz = os.path.join(DATA, "covtype.data.gz")
raw = os.path.join(DATA, "covtype.data")
if not os.path.exists(raw):
    if not os.path.exists(gz):
        raise SystemExit("先把 covtype.data.gz 下到 data/")
    with gzip.open(gz, "rb") as f:
        data = f.read()
    open(raw, "wb").write(data)
arr = np.loadtxt(raw, delimiter=",", dtype=np.float64)
X, y = arr[:, :54], arr[:, 54].astype(np.int64)
print("全量:", X.shape, "类别分布:", np.bincount(y)[1:])
rng = np.random.default_rng(0)
PER = 1200
KEEP = [2, 1, 3, 7, 6, 5]          # 取 6 个类（丢掉最稀有的第 4 类），按样本量排序
idx = []
for c in KEEP:
    ci = np.flatnonzero(y == c)
    idx.append(rng.choice(ci, size=min(PER, len(ci)), replace=False))
idx = np.concatenate(idx)
rng.shuffle(idx)
X, y = X[idx], y[idx]
# 重新编号成 0..5
m = {c: i for i, c in enumerate(KEEP)}
y = np.array([m[int(t)] for t in y], dtype=np.int64)
mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-9
X = ((X - mu) / sd).astype(np.float32)
n = len(X)
tr, te = int(n * 0.6), int(n * 0.85)
np.save(os.path.join(DATA, "cov_tr.npy"), X[:tr]);  np.save(os.path.join(DATA, "cov_tr_y.npy"), y[:tr])
np.save(os.path.join(DATA, "cov_te.npy"), X[tr:te]); np.save(os.path.join(DATA, "cov_te_y.npy"), y[tr:te])
np.save(os.path.join(DATA, "cov_va.npy"), X[te:]);   np.save(os.path.join(DATA, "cov_va_y.npy"), y[te:])
print("子样本:", X.shape, "类别分布:", np.bincount(y))
print("划分: train %d / holdout %d / test %d" % (tr, te - tr, n - te))
