# -*- coding: utf-8 -*-
"""P5：**门控清洗过的记忆，能不能让 <=3B 的小模型答得更准？**

这是这套记忆层第一座通向真实模型的桥
------------------------------------------------
前面 26 个阶段的"模型"都是一个线性判别器 + kNN，从来没接过生成式模型。
而项目定位是「电子神经元超小集群模型（单机部署，每次激活 <= 3B）」，
所以真正要回答的是：**把记忆洗干净，小模型会不会少答错？**

设计
------------------------------------------------
在这个实验室里造一个可控的"知识库"：
    100 个虚构实体，每个有唯一编号（4 位数）
    流里每个实体出现 5 次 = 500 条记录，其中 40% 被污染成错误编号
三份记忆：
    raw    照单全收（40% 错）
    gate   过门控（与已接受记录不一致就拒；拿不准才花 oracle）
    clean  只留真值（上界）
查询：每个实体抽 2 条记忆记录塞进提示词，让模型从 4 个选项里选编号。
    2 条记录是因为：**2 条如果互相矛盾，模型没有依据可依** —— 这样"记忆干不干净"才会体现到分数上。
如果模型能自己投票取胜，那 raw 和 gate 就没区别，这个实验也就没意义。
"""
from __future__ import annotations

import json, os, random, re, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
BASE = "http://127.0.0.1:8090"
N_ENT, N_REP, POLLUTE, K_SHOW = 100, 5, 0.4, 2
N_QUERY_PER = 2
SEED = 20261017

rnd = random.Random(SEED)
NAMES = ["玄枢", "流萤", "苍梧", "砚台", "寒汀", "拾光", "白露", "沉舟", "青釉", "归墟",
         "燃犀", "问渠", "照野", "漱石", "听风", "渡鸦", "拾贝", "临渊", "叠翠", "拂晓"]
ents = []
for i in range(N_ENT):
    ents.append(dict(id=i,
                     name="%s-%03d" % (NAMES[i % len(NAMES)], i),
                     code=rnd.randrange(1000, 9999)))

# 造流
stream = []
for e in ents:
    vals = [e["code"]] * N_REP
    k = int(N_REP * POLLUTE)
    wrong = rnd.sample(range(N_REP), k)
    for j in wrong:
        vals[j] = rnd.randrange(1000, 9999)
    rnd.shuffle(vals)
    for v in vals:
        stream.append(dict(ent=e["id"], val=v))
rnd.shuffle(stream)


def build_memory(mode):
    """gate: 与已接受记录不一致就拒；没有先验或票数打平 -> 花一次 oracle。"""
    acc = {e["id"]: [] for e in ents}
    oracle = 0
    for r in stream:
        ent, v = r["ent"], r["val"]
        truth = ents[ent]["code"]
        prior = acc[ent]
        if mode == "raw":
            prior.append(v); continue
        if mode == "clean":
            prior.append(truth); continue
        if not prior:
            ok = (v == truth); oracle += 1
        else:
            agree = sum(1 for x in prior if x == v) / len(prior)
            if agree > 0.5:
                ok = True
            elif agree < 0.5:
                ok = False
            else:
                ok = (v == truth); oracle += 1
        if ok:
            prior.append(v)
    return acc, oracle


def ask(prompt, options, letters="ABCD"):
    sys_p = "你是一个只做选择题的助手。只输出一个大写字母，不要任何解释。"
    for attempt in range(3):
        try:
            r = requests.post(BASE + "/v1/chat/completions", timeout=120, json={
                "messages": [{"role": "system", "content": sys_p},
                             {"role": "user", "content": prompt}],
                "max_tokens": 8, "temperature": 0})
            txt = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"[ABCD]", txt.upper())
            return m.group(0) if m else "?"
        except Exception:
            time.sleep(2)
    return "?"


def run_arm(mode, mem, oracle=0):
    hit = tot = 0
    t0 = time.time()
    for e in ents:
        pool = mem[e["id"]]
        for _ in range(N_QUERY_PER):
            if len(pool) == 0:
                shown = []
            elif len(pool) == 1:
                shown = [pool[0], pool[0]]
            else:
                shown = rnd.sample(pool, K_SHOW)
            correct = e["code"]
            opts = {correct}
            while len(opts) < 4:
                opts.add(rnd.randrange(1000, 9999))
            opts = list(opts); rnd.shuffle(opts)
            letters = "ABCD"
            ci = letters[opts.index(correct)]
            lines = "\n".join("  %d. %s 的编号是 %d" % (i + 1, e["name"], v)
                              for i, v in enumerate(shown)) or "  （没有记录）"
            body = ("下面是知识库中关于「%s」的记录：\n%s\n\n"
                    "请判断「%s」真正的编号是哪一个，只回答字母。\n" % (e["name"], lines, e["name"])
                    + "\n".join("  %s) %d" % (letters[i], o) for i, o in enumerate(opts)))
            got = ask(body, opts)
            hit += int(got == ci); tot += 1
    return dict(mode=mode, acc=hit / max(tot, 1), q=tot, oracle=oracle,
                mem_size=sum(len(v) for v in mem.values()), secs=time.time() - t0)


if __name__ == "__main__":
    print("健康:", requests.get(BASE + "/health", timeout=10).json())
    print("实体 %d 个，流 %d 条，污染 %.0f%%，每实体查 %d 次" % (N_ENT, len(stream), POLLUTE * 100, N_QUERY_PER))
    res = []
    for mode in ("clean", "raw", "gate"):
        mem, oracle = build_memory(mode)
        r = run_arm(mode, mem, oracle)
        res.append(r)
        print("  %-6s 题数 %d  记忆 %d 条  oracle %d  准确率 %.3f  (%.0f 秒)"
              % (mode, r["q"], r["mem_size"], r["oracle"], r["acc"], r["secs"]))
        sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_p5_bridge.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    d = {r["mode"]: r for r in res}
    print()
    print("=" * 72)
    print("P5：清洗过的记忆 vs 原始记忆（Qwen2.5-1.5B-Instruct Q4_K_M）")
    print("=" * 72)
    print("  上界（只留真值） %.3f" % d["clean"]["acc"])
    print("  原始记忆         %.3f" % d["raw"]["acc"])
    print("  门控清洗         %.3f   (oracle %d 次)" % (d["gate"]["acc"], d["gate"]["oracle"]))
    print("  -> 门控相对原始记忆 %+.4f" % (d["gate"]["acc"] - d["raw"]["acc"]))
    print("  -> 门控相对上界     %+.4f" % (d["gate"]["acc"] - d["clean"]["acc"]))
