# -*- coding: utf-8 -*-
"""T1：重建 power / ml100k / sms 的任务构造 —— 首轮头寸 <= 0.016，等于在噪声上比较。

三个坏在哪
    power    用"过去 60 分钟的 7 个通道"预测"下一分钟负荷的分位" -> 头寸 -0.012（全标注比不做还差）
    ml100k   用户类型偏好 + 电影类型 -> 评分档，信号太弱 -> 头寸 0.013
    sms      256 维无 IDF 哈希词袋 + 只有 1400 条 -> 头寸 0.016

怎么修（提高头寸，而不是调门控）
    power    改预测**一天中的时段**（4 段），日周期是真实可预测的
    ml100k   对评分矩阵做截断 SVD，取用户因子 + 电影因子（各 16 维），标签改成"喜欢/不喜欢"
    sms      词袋提到 1024 维 + IDF 加权 + 二元组，让 spam 可分
"""
import io, json, os, re, sys, zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根，不是 exp/
DATA = os.path.join(ROOT, "data"); RAW = os.path.join(DATA, "raw")
sys.path.insert(0, ROOT)
from fetch_datasets import fetch, read_zip, std, balance


def fix_power():
    z = fetch("power")
    cand = [n for n in z.namelist() if "household_power" in n.lower() and n.lower().endswith(".txt")]
    raw = z.read(cand[0]).decode("latin-1")
    rows = [l.split(";") for l in raw.splitlines()[1:] if l.count(";") == 8 and "?" not in l]
    A = np.array([[float(v) for v in r[2:]] for r in rows])
    hour = np.array([int(r[1].split(":")[0]) for r in rows])
    W = 60
    Xs, ys = [], []
    for i in range(0, len(A) - W - 1, 3):
        w = A[i:i + W]
        # 关键：用**窗口的统计量**而不是展平。420 维展平让 kNN 完全失效（头寸只有 0.046）。
        f = []
        for j in range(w.shape[1]):
            c = w[:, j]
            f += [c.mean(), c.std(), c.min(), c.max(), c[-1] - c[0], c[-10:].mean() - c[:10].mean()]
        Xs.append(f)
        ys.append(hour[i + W] // 6)          # 0-5 / 6-11 / 12-17 / 18-23 -> 4 段
    Xs = np.array(Xs); y = np.array(ys, dtype=np.int64)
    return "timeseries", balance(std(Xs), y, per=800, max_classes=4)


def fix_ml100k():
    z = fetch("ml100k")
    u = np.array([[int(v) for v in l.decode("latin-1").split("\t")[:3]]
                  for l in read_zip(z, "u.data").splitlines() if l.strip()])
    items = []
    for l in read_zip(z, "u.item").splitlines():
        if not l.strip():
            continue
        p = l.decode("latin-1").split("|")
        items.append([float(v) for v in p[5:24]])
    items = np.array(items)
    nu, ni = int(u[:, 0].max()), int(u[:, 1].max())
    R = np.zeros((nu, ni))
    M = np.zeros((nu, ni), dtype=bool)
    for uu, ii, r in u[:, :3]:
        R[uu - 1, ii - 1] = r; M[uu - 1, ii - 1] = True
    # 用观察到的评分做截断 SVD（缺失填该电影均分）
    Rf = R.copy()
    mu_item = np.array([R[M[:, j], j].mean() if M[:, j].any() else 3.0 for j in range(ni)])
    Rf[~M] = np.tile(mu_item, (nu, 1))[~M]
    U, S, Vt = np.linalg.svd(Rf - Rf.mean(), full_matrices=False)
    k = 16
    uf, itf = U[:, :k] * S[:k], Vt[:k].T
    u_idx, i_idx, rate = u[:, 0] - 1, u[:, 1] - 1, u[:, 2]
    X = np.hstack([uf[u_idx], itf[i_idx]])
    y = (rate >= 4).astype(np.int64)               # 喜欢 / 不喜欢
    return "recommendation", balance(std(X), y, per=800, max_classes=2)


def fix_sms():
    z = fetch("sms")
    cand = [n for n in z.namelist() if n.lower().endswith("spam.csv") or "SMSSpamCollection" in n]
    raw = z.read(cand[0]).decode("latin-1")
    ys, texts = [], []
    for line in raw.splitlines():
        p = line.split("\t")
        if len(p) < 2 or p[0].strip().lower() not in ("spam", "ham"):
            continue
        ys.append(1 if p[0].strip().lower() == "spam" else 0); texts.append(p[1])
    # 1024 维哈希词袋让 kNN 失效（头寸 -0.066）。降到 256 维，并补上几条短信的强特征。
    D = 256
    tok = lambda t: ["<s>"] + re.findall(r"[a-z0-9']+", t.lower())
    docs = [tok(t) for t in texts]
    df = np.zeros(D)
    for d in docs:
        for w in set(d + ["%s_%s" % (a, b) for a, b in zip(d, d[1:])]):
            df[hash(w) % D] += 1
    idf = np.log((1 + len(docs)) / (1 + df))
    X = np.zeros((len(docs), D))
    for i, d in enumerate(docs):
        for w in d + ["%s_%s" % (a, b) for a, b in zip(d, d[1:])]:
            X[i, hash(w) % D] += 1.0
    X = np.log1p(X) * idf
    extra = np.array([[len(t), len(t.split()), sum(ch.isdigit() for ch in t),
                       sum(ch.isupper() for ch in t) / max(len(t), 1),
                       t.count("!"), t.count("$"), t.count("£"),
                       1.0 if re.search(r"\b(free|win|winner|prize|urgent|call|txt|text|claim)\b", t.lower()) else 0.0,
                       1.0 if "http" in t.lower() else 0.0] for t in texts], dtype=float)
    X = np.hstack([X, extra])
    return "text", balance(std(X), np.array(ys, dtype=np.int64), per=700, max_classes=2)


FIX = {"power": fix_power, "ml100k": fix_ml100k, "sms": fix_sms}
if __name__ == "__main__":
    mf = os.path.join(DATA, "ds_manifest.json")
    man = json.load(open(mf, encoding="utf-8"))
    for name, fn in FIX.items():
        try:
            nature, (X, y) = fn()
            np.save(os.path.join(DATA, "ds_%s_X.npy" % name), X)
            np.save(os.path.join(DATA, "ds_%s_y.npy" % name), y)
            man[name] = dict(nature=nature, n=int(len(X)), dim=int(X.shape[1]),
                             classes=int(len(set(y.tolist()))), rebuilt="T1",
                             dist=[int(v) for v in np.bincount(y)])
            print("OK   %-8s %-16s n=%-6d dim=%-5d classes=%d" % (name, nature, len(X), X.shape[1], len(set(y.tolist()))))
        except Exception as e:
            print("FAIL %-8s %s: %s" % (name, type(e).__name__, str(e)[:150]))
        sys.stdout.flush()
    json.dump(man, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
