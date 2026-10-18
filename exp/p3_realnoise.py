# -*- coding: utf-8 -*-
"""P3：**真实人工标注错误** —— 26 个阶段以来一直缺的那一块。

为什么这一条最要紧
------------------------------------------------
26 个阶段的"污染"**全是人工注入的**（均匀翻转 / 混淆矩阵 / 数据源错标 …）。
我反复在报告里写"这不是真实世界的标注错误"。
现在用 **CIFAR-10N**：同一批图像被人类（Amazon MTurk）重新标注过，
其中 \`worse_label\` 是三轮里最脏的一份 —— **标签错误是真人犯的，不是模拟的。**

数据
    CIFAR-10 图像（cave.cs.toronto.edu）
    CIFAR-10N 的真实人工标签 data/CIFAR-10_human.pt（含 clean_label / worse_label / aggre_label）

要回答
  (1) 真实人工噪声长什么样？（噪声率、类别混淆结构）
  (2) 门控在**真实噪声**上的漏放率 / 误拒率是多少？（以前只在模拟噪声上报过）
  (3) 门控能不能把真实噪声清掉，拿到多少 ΔACC？
"""
from __future__ import annotations

import io, json, os, pickle, sys, tarfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data"); RAW = os.path.join(DATA, "raw")
P_DIM, N_TASK, CPT = 128, 5, 2
N_TRAIN, N_HOLD, N_TEST = 400, 200, 200
ROUND, BUDGETS = 50, [0, 5, 10, 20, 50]
SEEDS = list(range(5))
TAU, HILO, KNN_K = 0.60, 0.65, 8
_proj = np.random.default_rng(2026).normal(0, 1.0 / np.sqrt(3072), (3072, P_DIM))


def featurize(X):
    Z = np.asarray(X, dtype=np.float64) @ _proj
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def load_cifar():
    p = os.path.join(RAW, "cifar10.tar.gz")
    with tarfile.open(p, "r:gz") as tf:
        names = [n for n in tf.getnames() if "data_batch" in n]
        Xs, ys = [], []
        for n in sorted(names):
            d = pickle.load(io.BytesIO(tf.extractfile(n).read()), encoding="latin1")
            Xs.append(np.asarray(d["data"], dtype=np.float32)); ys.append(np.asarray(d["labels"]))
    return np.vstack(Xs), np.concatenate(ys)


def load_cifar10n():
    p = os.path.join(RAW, "cifar10n.tar.gz")
    with tarfile.open(p, "r:gz") as tf:
        n = [x for x in tf.getnames() if x.endswith("CIFAR-10_human.pt")][0]
        d = pickle.load(io.BytesIO(tf.extractfile(n).read()))
    return d


def loo_scores(X, y, classes, ridge=1.0):
    A = np.hstack([X, np.ones((len(X), 1))])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    Ginv = np.linalg.inv(G)
    h = np.clip(np.sum((A @ Ginv) * A, axis=1), 0.0, 0.999)
    P = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(classes):
        t = (y == c).astype(np.float64) * 2.0 - 1.0
        w = Ginv @ (A.T @ t)
        P[:, ci] = t - (t - A @ w) / (1.0 - h)
    return P


def knn_acc(Xq, yq, Xr, yr, k=KNN_K):
    if len(Xr) == 0:
        return float("nan")
    S = Xq @ Xr.T
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    nc = int(max(yq.max(), yr.max())) + 1
    pred = np.array([np.bincount(yr[r], minlength=nc).argmax() for r in idx])
    return float((pred == yq).mean())


def report_noise(clean, noisy, names):
    m = clean != noisy
    print("真实人工噪声（worse_label）")
    print("  总噪声率 %.1f%%（%d / %d）" % (100 * m.mean(), m.sum(), len(m)))
    print("  每个真实类别的被标错率：")
    for c in range(10):
        s = clean == c
        print("    %-10s n=%-6d 错 %.1f%%   最常被标成 %s" %
              (names.get(c, str(c)), int(s.sum()), 100 * m[s].mean(),
               names.get(int(np.bincount(noisy[s], minlength=10).argmax()), "?")))
    print("  混淆矩阵（行=真类，列=人工标成的类，只显示最大三列）：")
    C = np.zeros((10, 10), dtype=int)
    for a, b in zip(clean, noisy):
        C[a, b] += 1
    for a in range(10):
        row = C[a].copy(); row[a] = 0
        top = np.argsort(-row)[:3]
        print("    %-10s -> %s" % (names.get(a, str(a)),
                                   ", ".join("%s %d" % (names.get(int(t), "?"), row[t]) for t in top)))
    return m


def run(seed, X, y_clean, y_noisy):
    rng = np.random.default_rng(900 + seed)
    accs = {}
    n_dirty = n_clean = n_dirty_acc = n_clean_rej = 0
    for mode, B in [("gate", -1)] + [("naive", b) for b in BUDGETS]:
        buf_x, buf_y = [], []
        side = 0
        W = None; classes = []
        accs_by_task = []
        for t in range(N_TASK):
            cls = [CPT * t, CPT * t + 1]
            sel = np.concatenate([rng.choice(np.flatnonzero(y_clean == c), size=N_TRAIN // CPT, replace=False) for c in cls])
            Xt, yt_true, yt_obs = X[sel], y_clean[sel].copy(), y_noisy[sel].copy()
            if mode == "gate":
                P = loo_scores(Xt, yt_obs, cls)
                pred = np.array(cls)[P.argmax(axis=1)]
                if W is not None:
                    Pt = np.exp(np.hstack([Xt, np.ones((len(Xt), 1))]) @ W.T)
                    ms = Pt.max(axis=1) >= TAU
                else:
                    Pt, ms = None, np.zeros(len(Xt), dtype=bool)
                for i in range(len(Xt)):
                    a = None
                    if W is not None and ms[i]:
                        pos = {c: j for j, c in enumerate(classes)}
                        if int(yt_obs[i]) in pos:
                            p_ = np.exp(np.hstack([Xt[i:i + 1], np.ones((1, 1))]) @ W.T)[0]
                            p_ = p_ / p_.sum()
                            a = float(p_[pos[int(yt_obs[i])]])
                    p = 1.0 if pred[i] == yt_obs[i] else 0.0
                    sc = p if a is None else (0.6 * a + 0.4 * p)
                    if (1 - HILO) < sc < HILO:
                        ok = (yt_obs[i] == yt_true[i]); side += 1
                    else:
                        ok = sc >= HILO
                    dirty = yt_obs[i] != yt_true[i]
                    if dirty:
                        n_dirty += 1; n_dirty_acc += int(ok)
                    else:
                        n_clean += 1; n_clean_rej += int(not ok)
                    if ok:
                        buf_x.append(Xt[i]); buf_y.append(int(yt_obs[i]))
            else:
                b = len(Xt) if mode == "full" else B
                ys = yt_obs.copy()
                if b > 0:
                    for i in rng.choice(len(Xt), size=min(b, len(Xt)), replace=False):
                        ys[int(i)] = yt_true[int(i)]; side += 1
                buf_x.extend(list(Xt)); buf_y.extend(list(ys))
            if len(buf_y) >= 60:
                cl = sorted(set(buf_y))
                if len(cl) >= 2:
                    A_ = np.hstack([np.array(buf_x), np.ones((len(buf_x), 1))])
                    G = A_.T @ A_ + np.eye(A_.shape[1])
                    Gi = np.linalg.inv(G)
                    classes = cl
                    W = np.zeros((len(cl), A_.shape[1]))
                    for ci_, c in enumerate(cl):
                        tt = (np.array(buf_y) == c).astype(np.float64) * 2 - 1
                        W[ci_] = Gi @ (A_.T @ tt)
        # 全部任务训完后一次性评估
        keep_x = np.array(buf_x); keep_y = np.array(buf_y)
        hits = []
        for t in range(N_TASK):
            cls = [CPT * t, CPT * t + 1]
            te = np.concatenate([np.flatnonzero(y_clean == c)[:N_TEST // CPT] for c in cls])
            hits.append(float((np.array([_pred(keep_x, keep_y, X[te][i], classes, W) for i in range(len(te))]) == y_clean[te]).mean()))
        accs["%s_%d" % (mode, B)] = (float(np.mean(hits)), side)
    ga, gs = accs["gate_-1"]
    xs = [(accs["naive_%d" % b][1], accs["naive_%d" % b][0]) for b in BUDGETS]
    nat = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if x0 <= gs <= x1 and x1 > x0:
            nat = y0 + (gs - x0) / (x1 - x0) * (y1 - y0); break
    return dict(seed=seed, gate_acc=ga, gate_side=gs, naive_at_gate=nat,
                delta_acc=(ga - nat) if nat == nat else float("nan"),
                raw=accs["naive_0"][0], full=accs["naive_50"][0],
                gate_fa=(n_dirty_acc / n_dirty if n_dirty else float("nan")),
                gate_fr=(n_clean_rej / n_clean if n_clean else float("nan")),
                n_dirty=n_dirty, n_clean=n_clean)


def _pred(keep_x, keep_y, x, classes, W):
    if W is not None:
        p = np.exp(np.hstack([x[None, :], np.ones((1, 1))]) @ W.T)[0]
        p = p / p.sum()
        if p.max() >= TAU:
            return int(np.array(classes)[p.argmax()])
    if len(keep_x) == 0:
        return -1
    d = keep_x @ x
    k = min(KNN_K, len(d))
    idx = np.argpartition(-d, k - 1)[:k]
    return int(np.bincount(keep_y[idx], minlength=10).argmax())


NAMES = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
         5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

if __name__ == "__main__":
    if not os.path.exists(os.path.join(RAW, "cifar10.tar.gz")):
        raise SystemExit("缺 CIFAR-10")
    X, y_clean = load_cifar()
    hum = load_cifar10n()
    y_noisy = np.asarray(hum["worse_label"])
    print("CIFAR-10 train", X.shape, " 人工标签 worse_label", y_noisy.shape)
    mask = report_noise(y_clean, y_noisy, NAMES)
    Xf = featurize(X)
    dset = dict(X=Xf, y_clean=y_clean, y_noisy=y_noisy)
    np.save(os.path.join(DATA, "cifar10n_X128.npy"), Xf)
    np.save(os.path.join(DATA, "cifar10n_yclean.npy"), y_clean)
    np.save(os.path.join(DATA, "cifar10n_ynoisy.npy"), y_noisy)
    res = [run(s, Xf, y_clean, y_noisy) for s in SEEDS]
    json.dump(res, open(os.path.join(OUT, "results_p3_realnoise.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 84)
    print("P3：真实人工标注噪声（CIFAR-10N worse_label / 5 seeds）")
    print("=" * 84)
    ga = np.mean([r["gate_acc"] for r in res]); na = np.nanmean([r["naive_at_gate"] for r in res])
    print("  不做门控（原始人工标签） %.3f" % np.mean([r["raw"] for r in res]))
    print("  门控                  %.3f @ %.0f 次" % (ga, np.mean([r["gate_side"] for r in res])))
    print("  同花费下的朴素随机标注 %.3f" % na)
    print("  全对（clean_label 上界） %.3f" % np.mean([r["full"] for r in res]))
    print("  ΔACC                  %+.4f" % (ga - na))
    print("  门控对真实噪声的漏放率   %.3f  （脏样本 %d 条）" % (np.nanmean([r["gate_fa"] for r in res]), np.mean([r["n_dirty"] for r in res])))
    print("  门控对干净标签的误拒率   %.3f  （干净样本 %d 条）" % (np.nanmean([r["gate_fr"] for r in res]), np.mean([r["n_clean"] for r in res])))
