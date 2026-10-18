# -*- coding: utf-8 -*-
"""T2：**真实形态的标注噪声** —— 项目最根本的信任缺口。

25 个阶段的"污染"全是人工注入的：均匀翻转、按弱模型混淆矩阵翻转。
真实世界的标注错误不长这样。这里换成六种更像真的：

    uniform     均匀翻到别的类（旧基线）
    confusion   按弱模型的混淆矩阵翻（阶段 1-25 用的就是这个）
    similarity  **翻到"特征上最像"的另一个类** —— 人会混淆相似的东西，这类错最难查
    annotator   **标注者偏差**：5 个标注者各有自己的混淆习惯，样本随机派给人
    drift       **质量漂移**：每轮的污染率在 0.1~0.7 之间随机变
    shortcut    **捷径相关**：按一个与特征相关的确定性规则翻（批次效应/采集偏差）

要回答的问题：**门控的优势在这六种噪声下还成立吗？**
（判据同 D1：同等准确率下省 >= 30% 的 oracle）
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.hallucination import weak_model_confusion
from exp.realdata import RealClassStream, load_mnist
from exp.stage20 import R20, featurize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
P_DIM, POLLUTE, ROUND, N_TASK, CPT = 128, 0.4, 50, 5, 2
N_TRAIN, N_HOLD, N_TEST = 400, 200, 200
SEEDS = [0, 1, 2]
BUDGETS = [0, 5, 10, 20, 50]
MODELS = ["uniform", "confusion", "similarity", "annotator", "drift", "shortcut"]
MN = load_mnist()
CONF, _ = weak_model_confusion()


def flip_uniform(y, classes, idx, rng):
    out = y.copy()
    for i in idx:
        others = [c for c in classes if c != y[i]]
        out[i] = others[rng.integers(0, len(others))]
    return out


def flip_confusion(y, classes, idx, rng, conf):
    """conf 是 10x10 的混淆矩阵 conf[真类, 预测类]，和 hallu.inject 的用法一致。"""
    out = y.copy()
    for i in idx:
        c = int(y[i])
        row = np.zeros(10)
        row[list(classes)] = conf[c, list(classes)]
        row[c] = 0.0
        s = row.sum()
        if s <= 1e-9:
            others = [k for k in classes if k != c]
            out[i] = others[rng.integers(0, len(others))]
        else:
            out[i] = int(rng.choice(10, p=row / s))
    return out


def flip_similarity(X, y, classes, idx, rng):
    """翻到特征上最像的另一个类 —— 这类错在特征空间里"看起来是对的"。"""
    cents = {c: X[y == c].mean(0) for c in classes}
    out = y.copy()
    for i in idx:
        d = {c: float(np.linalg.norm(X[i] - cents[c])) for c in classes if c != y[i]}
        out[i] = min(d, key=d.get)
    return out


def flip_annotator(X, y, classes, idx, rng, n_ann=5):
    """5 个标注者，各有偏向：第 k 个人倾向于把类 a 记成类 b。"""
    bias = []
    for k in range(n_ann):
        a = classes[k % len(classes)]
        b = classes[(k + 1 + rng.integers(0, len(classes) - 1)) % len(classes)]
        while b == a:
            b = classes[rng.integers(0, len(classes))]
        bias.append((a, b))
    out = y.copy()
    for i in idx:
        a, b = bias[int(rng.integers(0, n_ann))]
        if y[i] == a:
            out[i] = b
        else:
            others = [c for c in classes if c != y[i]]
            out[i] = others[rng.integers(0, len(others))]
    return out


def build_tasks(seed, model):
    st = RealClassStream(MN[0], MN[1], MN[2], MN[3], n_tasks=N_TASK, classes_per_task=CPT,
                         n_train=N_TRAIN, n_holdout=N_HOLD, n_test=N_TEST,
                         label_noise=0.0, seed=1000 + seed)
    rng = np.random.default_rng(555 + seed)
    for task in st:
        task.x_train = featurize(task.x_train)
        task.x_holdout = featurize(task.x_holdout)
        task.x_test = featurize(task.x_test)
        X, y, cls = task.x_train, task.y_train_true.copy(), task.classes
        n = len(y)
        k = int(n * POLLUTE)
        idx = rng.choice(n, size=k, replace=False)
        if model == "uniform":
            y2 = flip_uniform(y, cls, idx, rng)
        elif model == "confusion":
            y2 = flip_confusion(y, cls, idx, rng, CONF)
        elif model == "similarity":
            y2 = flip_similarity(X, y, cls, idx, rng)
        elif model == "annotator":
            y2 = flip_annotator(X, y, cls, idx, rng)
        elif model == "drift":
            y2 = y.copy()                            # 逐轮在 run() 里按变化的污染率覆盖
        elif model == "shortcut":
            y2 = y.copy()
            # 按"第一个特征维度的高分位"这个与标签无关的规则翻 —— 批次效应
            v = X[:, 0]
            thr = float(np.quantile(v, 1 - POLLUTE))
            sel = np.flatnonzero(v >= thr)
            for i in sel:
                y2[i] = [c for c in cls if c != y[i]][int(rng.integers(0, len(cls) - 1))]
        else:
            raise ValueError(model)
        task.y_train = y2
        task.noise_model = model
    return st


def run(model, seed):
    st = build_tasks(seed, model)
    recs = {}
    for mode, B in [("gate", -1)] + [("naive", b) for b in BUDGETS]:
        R = R20(mode, seed, B if B >= 0 else -1)
        if mode == "gate":
            R.mode = "gate"
        accs = []
        for t, task in enumerate(st):
            hv_x, hv_y = task.x_holdout, task.y_holdout
            tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
            for s0 in range(0, len(task.x_train), ROUND):
                xb = task.x_train[s0:s0 + ROUND]; yb = task.y_train[s0:s0 + ROUND]
                if model == "drift":
                    # 标注质量逐轮漂移：污染率在 0.1~0.7 之间随机
                    r2 = np.random.default_rng(seed * 100000 + t * 1000 + s0)
                    yb = task.y_train_true[s0:s0 + ROUND].copy()
                    pp = float(r2.uniform(0.1, 0.7))
                    idx = np.flatnonzero(r2.random(len(yb)) < pp)
                    yb = flip_uniform(yb, sorted(set(yb.tolist())), idx, r2)
                R.round(xb, yb, tf)
            a = [float((R.predict(st.tasks[j].x_test) == st.tasks[j].y_test).mean()) for j in range(t + 1)]
            accs.append(float(np.mean(a)))
        recs["%s_%d" % (mode, B)] = (float(accs[-1]), R.side)
    ga, gs = recs["gate_-1"]
    xs = [(recs["naive_%d" % b][1], recs["naive_%d" % b][0]) for b in BUDGETS]
    need = float("nan")
    for i in range(len(xs) - 1):
        (x0, y0), (x1, y1) = xs[i], xs[i + 1]
        if y0 <= ga <= y1 and y1 > y0:
            need = x0 + (ga - y0) / (y1 - y0) * (x1 - x0); break
    saved = (need - gs) / need if need == need and need > 0 else float("nan")
    return dict(model=model, seed=seed, gate_acc=ga, gate_side=gs, need=need, saved=saved,
                raw=recs["naive_0"][0], full=recs["naive_50"][0],
                headroom=recs["naive_50"][0] - recs["naive_0"][0])


if __name__ == "__main__":
    res = []
    for model in MODELS:
        for seed in SEEDS:
            r = run(model, seed); res.append(r)
            print("%-11s seed=%d  门控 %.3f@%d  省 %s  头寸 %.3f"
                  % (model, seed, r["gate_acc"], r["gate_side"],
                     ("%.0f%%" % (100 * r["saved"])) if r["saved"] == r["saved"] else "n/a", r["headroom"]))
            sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_t2_realnoise.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print("=" * 84)
    print("T2：六种标注噪声下的门控优势（真实 MNIST / 污染 40% / 3 seeds；通过线 = 省 30%）")
    print("=" * 84)
    print("%-12s %-12s %-12s %-14s %-10s %s" % ("噪声模型", "门控 ACC", "省 oracle", "朴素需要", "头寸", "判定"))
    for model in MODELS:
        rs = [r for r in res if r["model"] == model]
        s = np.nanmean([r["saved"] for r in rs]); h = np.nanmean([r["headroom"] for r in rs])
        print("%-12s %-12.3f %-12s %-14s %-10.3f %s"
              % (model, np.mean([r["gate_acc"] for r in rs]),
                 ("%.0f%%" % (100 * s)) if s == s else "n/a",
                 ("%.0f" % np.nanmean([r["need"] for r in rs])) if not np.all(np.isnan([r["need"] for r in rs])) else "n/a",
                 h, "通过" if (s == s and s >= 0.30) else "不通过"))
