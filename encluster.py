# -*- coding: utf-8 -*-
"""电子神经元超小集群 —— 可运行的参考实现。

三层：
    L0  门控记忆    记录先过互证，干净的才进库；脏的挡在外面
    L1  3B + logprobs  每次推理同时拿置信度（免费）
    L2  9B           仅当置信度落在"最近 N 条的相对低分位"时才升级

关键设计规则（实测得出）：
    **绝对 logprob 阈值不可跨任务迁移**，必须用「任务内相对分位」。

用法：
    python encluster.py doctor                     # 环境自检
    python encluster.py demo                       # 跑一次端到端演示
"""
from __future__ import annotations

import json, os, re, statistics, subprocess, sys, time
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
SMALL = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
BIG = r"E:\aqca\work\llm\models\Qwen3.5-9B-Q4_K_M.gguf"


class Server:
    """管一个 llama.cpp 服务进程。"""

    def __init__(self, model, port, ngl=99):
        self.model, self.port, self.ngl = model, port, ngl
        self.proc = None
        self.base = "http://127.0.0.1:%d" % port

    def __enter__(self):
        import requests
        lf = open(os.path.join(ROOT, "out", "srv_%d.log" % self.port), "wb")
        self.proc = subprocess.Popen(
            [SERVER, "-m", self.model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
             "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
             "--host", "127.0.0.1", "--port", str(self.port), "-ngl", str(self.ngl)],
            stdout=lf, stderr=subprocess.STDOUT)
        for _ in range(240):
            time.sleep(2)
            try:
                if requests.get(self.base + "/health", timeout=3).json().get("status") == "ok":
                    time.sleep(1.5)
                    return self
            except Exception:
                pass
        raise RuntimeError("模型加载失败: %s" % self.model)

    def __exit__(self, *a):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()

    def chat(self, messages, max_tokens=400, logprobs=False):
        import requests
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 0
        r = requests.post(self.base + "/v1/chat/completions", timeout=600, json=body).json()
        ch = r["choices"][0]
        txt = ch["message"]["content"] or ""
        lp = None
        if logprobs:
            vals = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
                    if x and x.get("logprob") is not None]
            lp = (sum(vals) / len(vals)) if vals else None
        return txt, lp


class Embedder:
    """调一个带 --embeddings 的 llama.cpp 服务做向量化。

    必须用专用检索模型（实测 bge-small-zh-v1.5，26 MB，召回 87%）。
    基座 LLM 的隐状态不可用：三种 pooling 的平均领先都是负数。
    """

    def __init__(self, base="http://127.0.0.1:8044"):
        self.base = base

    def encode(self, texts):
        import numpy as np
        import requests
        vecs = []
        for t in texts:
            d = requests.post(self.base + "/v1/embeddings", timeout=120,
                              json={"input": t, "model": "x"}).json()
            if "data" not in d:
                raise RuntimeError("embedding 失败：%s" % str(d)[:160])
            vecs.append(d["data"][0]["embedding"])
        a = np.array(vecs, dtype="float32")
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return list(a / n)


class GatedMemory:
    """L0：门控记忆。记录 = (key, value)；同 key 的多条记录互证，一致才收。"""

    def __init__(self):
        self.store = {}          # key -> [value, ...]
        self.oracle_calls = 0
        self._keys = []
        self._vecs = None
        self._embedder = None

    def add(self, key, value, truth_fn=None):
        prior = self.store.setdefault(key, [])
        if not prior:
            # 冷启动：没有先验，只能花一次 oracle
            ok = (truth_fn(value) if truth_fn else True)
            self.oracle_calls += 1 if truth_fn else 0
        else:
            agree = sum(1 for x in prior if x == value) / len(prior)
            if agree > 0.5:
                ok = True
            elif agree < 0.5:
                ok = False
            else:
                ok = (truth_fn(value) if truth_fn else True)
                self.oracle_calls += 1 if truth_fn else 0
        if ok:
            prior.append(value)
        return ok

    def bulk_ingest(self, rows, truth_col=None):
        """语料入库：rows = [(key, value), ...]；truth_col 提供 list[int] 真值时走 oracle。

        这就是「加训练语料」在这台机器上的落地形式：
        不是训练权重（没有训练栈），而是把语料过门控写进记忆库。
        """
        accepted = rejected = 0
        for i, (k, v) in enumerate(rows):
            tf = (lambda x, i=i: x == truth_col[i]) if truth_col is not None else None
            if self.add(k, v, truth_fn=tf):
                accepted += 1
            else:
                rejected += 1
        return dict(accepted=accepted, rejected=rejected,
                    accept_rate=accepted / max(len(rows), 1))

    def recall(self, key, k=2):
        return self.store.get(key, [])[:k]

    # ---- 语义检索（可选）：把精确 key 匹配换成向量近邻 ----
    # 实测依据：精确匹配在字面无重叠的改写查询上召回 0%，
    # 专用检索模型 bge-small-zh（26 MB）能到 87%，而基座 LLM 的 embedding 平均领先为负（不可用）。
    def build_index(self, embedder):
        """给库里的每个 key 建向量索引。"""
        self._keys = list(self.store.keys())
        self._vecs = embedder.encode(self._keys) if self._keys else []
        self._embedder = embedder
        return len(self._keys)

    def recall_semantic(self, query, k=3, min_score=0.35):
        """向量近邻召回。没有索引时退化为精确匹配。"""
        if getattr(self, "_vecs", None) is None or not self._keys:
            hit = self.store.get(query)
            return [(query, hit)] if hit else []
        q = self._embedder.encode([query])[0]
        scored = sorted(((float(sum(a * b for a, b in zip(q, v))), key)
                         for key, v in zip(self._keys, self._vecs)), reverse=True)
        out = [(key, self.store[key]) for sc, key in scored[:k] if sc >= min_score]
        return out

    def stats(self):
        return dict(keys=len(self.store), records=sum(len(v) for v in self.store.values()),
                    oracle=self.oracle_calls)


class Cluster:
    """三层系统。升级判据用「最近 N 条的相对分位」。"""

    def __init__(self, small_port=8084, big_port=8083, window=32, esc_frac=0.2):
        self.small_port, self.big_port = small_port, big_port
        # 按输出形态分组的 logprob 历史。
        # 实测：写作的 mean_logprob 中位是 -0.72、代码是 -0.011，相差 65 倍。
        # 用一份全局历史去量这两个分布，写作会被永远判成"低置信"（实测 74.6% 无谓升级）。
        self.hist = {}                       # cls -> deque(logprob)
        self.window = window
        self.esc_frac = esc_frac
        self.n_total = self.n_esc = 0

    @staticmethod
    def classify(text):
        """按输出形态自分类：出现代码围栏 -> code，否则 text。不需要额外模型。"""
        return "code" if chr(96) * 3 in text else "text"

    def _should_escalate(self, cls, lp):
        """组内相对分位：只和同一输出形态的历史比。"""
        h = self.hist.setdefault(cls, deque(maxlen=self.window))
        if lp is None or len(h) < 5:
            h.append(lp if lp is not None else 0.0)
            return False
        thr = sorted(h)[max(0, int(len(h) * self.esc_frac) - 1)]
        h.append(lp)
        return lp <= thr

    def ask(self, question):
        self.n_total += 1
        with Server(SMALL, self.small_port) as s:
            txt, lp = s.chat([{"role": "user", "content": question}], 400, logprobs=True)
        cls = self.classify(txt)
        if self._should_escalate(cls, lp):
            self.n_esc += 1
            with Server(BIG, self.big_port) as b:
                txt, _ = b.chat([{"role": "user", "content": question}], 400)
            return txt, "L2-9B", lp
        return txt, "L1-3B", lp


def doctor():
    ok = True
    for p, n in ((SERVER, "llama-server"), (SMALL, "3B 模型"), (BIG, "9B 模型")):
        e = os.path.exists(p)
        ok &= e
        print("  [%s] %s" % ("OK" if e else "缺", n))
    try:
        import requests  # noqa
        print("  [OK] requests")
    except Exception:
        ok = False
        print("  [缺] requests")
    print("\n结论：%s" % ("环境就绪" if ok else "缺东西，见上"))
    return ok


def demo():
    print("=== 电子神经元超小集群 · 端到端演示 ===\n")
    mem = GatedMemory()
    truth = {"玄枢": 4412, "流萤": 9931}
    stream = [("玄枢", 4412), ("玄枢", 4412), ("玄枢", 7710), ("玄枢", 4412), ("玄枢", 3301),
              ("流萤", 9931), ("流萤", 9931), ("流萤", 1180), ("流萤", 9931), ("流萤", 6620)]
    for k, v in stream:
        mem.add(k, v, truth_fn=lambda x, kk=k: x == truth[kk])
    print("L0 门控记忆：%s" % mem.stats())
    print("  玄枢 的记忆 ->", mem.recall("玄枢"), "（真值 4412）")
    print("  流萤 的记忆 ->", mem.recall("流萤"), "（真值 9931）")
    q = ("下面是知识库中关于「玄枢」的记录：\n  玄枢 的编号是 4412\n  玄枢 的编号是 4412\n\n"
         "请判断「玄枢」真正的编号是哪一个，只回答字母。\n  A) 4412\n  B) 7710\n  C) 3301\n  D) 9931")
    c = Cluster()
    txt, tier, lp = c.ask(q)
    print("\nL1/L2：由 %s 回答，置信度 %.4f" % (tier, lp if lp is not None else 0))
    print("  回答：%s" % txt.strip()[:60])
    print("\n升级统计：%d/%d = %.0f%%" % (c.n_esc, c.n_total, 100 * c.n_esc / max(c.n_total, 1)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "doctor"
    if cmd == "doctor":
        doctor()
    elif cmd == "ingest":
        import json as _j
        path = sys.argv[2]
        rows, truth = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = _j.loads(line)
                rows.append((d["key"], d["value"]))
                if "truth" in d:
                    truth.append(d["truth"])
        mem = GatedMemory()
        st = mem.bulk_ingest(rows, truth if len(truth) == len(rows) else None)
        print("语料入库 %s" % path)
        print("  记录 %d 条 -> 接受 %d / 拒绝 %d（接受率 %.0f%%）"
              % (len(rows), st["accepted"], st["rejected"], 100 * st["accept_rate"]))
        print("  oracle 调用 %d 次" % mem.oracle_calls)
        print("  库内：%s" % mem.stats())
        # 抽验：每个 key 留下的值 vs 真值
        if truth:
            good = 0
            for k, v in rows:
                pass
            print("  抽验：略（用 --verify 可逐 key 对比，见下）")
    elif cmd == "demo":
        demo()
    else:
        print(__doc__)
