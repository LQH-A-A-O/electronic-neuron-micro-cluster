# -*- coding: utf-8 -*-
"""P3：**真实人工标注噪声** —— 用 CIFAR-10N 的真人混淆矩阵，施加到 CIFAR-10 图像上。

26 个阶段的污染全是人工注入的。这一轮的噪声来自
**CIFAR-10N（ICLR 2022）里 5 万个样本被真人在 MTurk 上重新标注的结果**：
总噪声率 40.2%，而且结构极其"人类"——
    automobile <-> truck   24.7% / 28.2%
    cat <-> dog            20.6% / 18.3%
    deer -> horse          15.2%
    airplane <-> ship       8.5% /  8.2%
**错标全部落在语义相邻的类上。** 这正是门控最难的一类污染（T2 的"相似度驱动"）。

设计：把 5 个任务按"两类之间的真实混淆强度"从高到低排序，
看门控的优势是不是随混淆强度变化 —— 这是以前从没问过的问题。
"""
from __future__ import annotations

import collections, io, json, os, sys, tarfile
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data"); RAW = os.path.join(DATA, "raw")
P_DIM, N_TRAIN, N_HOLD, N_TEST = 128, 400, 200, 200
POLLUTE, ROUND, BUDGETS = 0.4, 50, [0, 5, 10, 20, 50]
SEEDS = list(range(5))
TAU, HILO, KNN_K = 0.60, 0.65, 8
NAMES = "airplane automobile bird cat deer dog frog horse ship truck".split()
_proj = np.random.default_rng(2026).normal(0, 1.0 / np.sqrt(3072), (3072, P_DIM))

# 128 维随机投影在 CIFAR-10 上是瞎猜（真值上界只有 0.52）。换成 ImageNet 预训练 ResNet18 的特征。
import torch
from torchvision.models import resnet18
_W = os.path.join(DATA, "resnet18.pth")
if not os.path.exists(_W):
    os.replace("/tmp/rt.pth" if os.path.exists("/tmp/rt.pth") else _W, _W) if os.path.exists("/tmp/rt.pth") else None
_net = resnet18(weights=None)
_net.load_state_dict(torch.load(_W, map_location="cpu"))
_net.fc = torch.nn.Identity()
_net.eval()
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _resnet_feat(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2 and X.shape[1] == 3072:
        # 关键：图像是 HWC 展平的，必须先 reshape 成 (n,32,32,3) 再转置成 CHW，
        # 直接 reshape(-1,3,32,32) 会把像素拆错，特征等于噪声（上一版就栽在这）。
        X = X.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
    t = torch.from_numpy(X)
    if t.max() > 2.0:
        t = t / 255.0
    t = (t - _MEAN) / _STD
    out = []
    with torch.no_grad():
        for i in range(0, len(t), 256):
            out.append(_net(t[i:i + 256]).numpy())
    F = np.vstack(out)
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)

# 按真实互相混淆强度（P(i->j)+P(j->i)）从高到低排
P = np.load(os.path.join(DATA, "real_human_confusion.npy"))
PAIRS = sorted([(i, j) for i in range(10) for j in range(i + 1, 10)],
               key=lambda ij: -(P[ij[0], ij[1]] + P[ij[1], ij[0]]))


def featurize(X):
    return _resnet_feat(X).astype(np.float64)


def load_images():
    """只取需要的类；tarball 里是 train/<class>/<k>.jpg。"""
    need = set()
    for i, j in PAIRS[:5]:
        need.add(NAMES[i]); need.add(NAMES[j])
    out = collections.defaultdict(list)
    t = tarfile.open(os.path.join(RAW, "cifar10img.tar.gz"), "r:gz")
    for m in t:
        parts = m.name.split("/")
        if len(parts) == 4 and parts[1] == "train" and parts[2] in need and m.name.endswith(".jpg"):
            im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
            out[parts[2]].append(np.asarray(im, dtype=np.float32).ravel())
    return {k: np.array(v) for k, v in out.items()}


def make_task(cls, imgs, seed):
    rng = np.random.default_rng(700 + seed)
    xs, ys = [], []
    for ci in cls:
        a = imgs[NAMES[ci]]
        idx = rng.permutation(len(a))[:N_TRAIN + N_HOLD + N_TEST]
        xs.append(a[idx]); ys.append(np.full(len(idx), ci))
    X = np.vstack(xs); y = np.concatenate(ys)
    o = rng.permutation(len(y)); X, y = X[o], y[o]
    n1 = N_TRAIN; n2 = N_TRAIN + N_HOLD
    return X[:n1], y[:n1], X[n1:n2], y[n1:n2], X[n2:], y[n2:]


def human_noise(y_true, cls, p, rng):
    """按**真实人工混淆矩阵**在任务的两个类之间产生错标。"""
    y = y_true.copy()
    hit = rng.random(len(y)) < p
    for k in np.flatnonzero(hit):
        c = int(y_true[k])
        other = [x for x in cls if x != c]
        w = np.array([P[c, o] for o in other], dtype=float)
        if w.sum() <= 0:
            y[k] = other[0]
        else:
            y[k] = other[int(rng.choice(len(other), p=w / w.sum()))]
    return y


def loo_scores(X, y, classes, ridge=1.0):
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Gi = np.linalg.inv(G)
    h = np.clip(np.sum((A @ Gi) * A, axis=1), 0.0, 0.999)
    Q = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2 - 1
        w = Gi @ (A.T @ t)
        Q[:, ci] = t - (t - A @ w) / (1 - h)
    return Q


def knn(Xq, yq, Xr, yr, k=KNN_K):
    if len(Xr) == 0:
        return float("nan")
    S = Xq @ Xr.T
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    nc = int(max(yq.max(), yr.max())) + 1
    return float((np.array([np.bincount(yr[r], minlength=nc).argmax() for r in idx]) == yq).mean())


def run_task(cls, imgs, seed, mode):
    Xt, yt_true, Xh, yh, Xe, ye = make_task(cls, imgs, seed)
    Xt = featurize(Xt); Xh = featurize(Xh); Xe = featurize(Xe)
    rng = np.random.default_rng(31337 + seed)
    yt = human_noise(yt_true, list(cls), POLLUTE, rng)
    yh_obs = human_noise(yh, list(cls), POLLUTE, rng)
    tf = lambda x: int(yh[int(np.argmin(np.linalg.norm(Xh - x, axis=1)))])
    bx, by = [], []
    side = 0; n_d = n_c = d_acc = c_rej = 0
    if mode == "gate":
        Q = loo_scores(Xt, yt, list(cls))
        pred = np.array(cls)[Q.argmax(axis=1)]
        for i in range(len(Xt)):
            p = 1.0 if pred[i] == yt[i] else 0.0
            if (1 - HILO) < p < HILO:
                ok = truth = (yt[i] == tf(Xt[i])); side += 1; yh_t = tf(Xt[i])
            else:
                ok = p >= HILO; truth = tf(Xt[i])
            dirty = yt[i] != truth
            if dirty: n_d += 1; d_acc += int(ok)
            else: n_c += 1; c_rej += int(not ok)
            if ok: bx.append(Xt[i]); by.append(int(yt[i]))
    elif mode == "full":
        bx = list(Xt); by = list(yt_true)          # 真值上界：用干净标签训练
    else:
        bx = list(Xt); by = list(yt)               # 原始：用脏标签训练
    return knn(Xe, ye, np.array(bx), np.array(by)), side, bx, by, (d_acc / n_d if n_d else float("nan")), (c_rej / n_c if n_c else float("nan"))


if __name__ == "__main__":
    print("加载图像 ..."); sys.stdout.flush()
    imgs = load_images()
    print("  " + ", ".join("%s %d" % (k, len(v)) for k, v in imgs.items()))
    print()
    print("按真实互相混淆强度排序的前 5 对：")
    for k, (i, j) in enumerate(PAIRS[:5]):
        print("  %d) %-11s <-> %-11s  真实互混 %.1f%% (%.1f%% / %.1f%%)"
              % (k + 1, NAMES[i], NAMES[j], 100 * (P[i, j] + P[j, i]), 100 * P[i, j], 100 * P[j, i]))
    res = []
    for k, (i, j) in enumerate(PAIRS[:5]):
        for seed in SEEDS:
            g, gs, _, _, gfa, gfr = run_task((i, j), imgs, seed, "gate")
            r0, _, _, _, _, _ = run_task((i, j), imgs, seed, "raw")
            r1, _, _, _, _, _ = run_task((i, j), imgs, seed, "full")
            res.append(dict(pair="%s/%s" % (NAMES[i], NAMES[j]), conf=float(P[i, j] + P[j, i]),
                            seed=seed, gate=g, raw=r0, full=r1, side=gs, fa=gfa, fr=gfr))
        print("  完成 %-22s" % ("%s/%s" % (NAMES[i], NAMES[j]))); sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_p3_realconfusion.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 92)
    print("P3：真实人工标注噪声下的门控（CIFAR-10 图像 / 真实人类混淆矩阵 / 污染 40% / 5 seeds）")
    print("=" * 92)
    print("%-24s %-10s %-10s %-10s %-10s %-10s %s"
          % ("任务（按混淆强度）", "互混强度", "门控", "原始", "真值上界", "门控-原始", "门控漏放/误拒"))
    for k, (i, j) in enumerate(PAIRS[:5]):
        rs = [r for r in res if r["pair"] == "%s/%s" % (NAMES[i], NAMES[j])]
        g = np.mean([r["gate"] for r in rs]); r0 = np.mean([r["raw"] for r in rs])
        r1 = np.mean([r["full"] for r in rs])
        print("%-24s %-10.1f%% %-10.3f %-10.3f %-10.3f %+.4f     %.3f / %.3f"
              % ("%s/%s" % (NAMES[i], NAMES[j]), 100 * (P[i, j] + P[j, i]), g, r0, r1, g - r0,
                 np.nanmean([r["fa"] for r in rs]), np.nanmean([r["fr"] for r in rs])))
