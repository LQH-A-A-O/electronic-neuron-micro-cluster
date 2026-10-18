# -*- coding: utf-8 -*-
"""P5b：门控记忆 + Qwen2.5-3B（选定的模型），规模放大到 300 实体。

与 P5 的唯一区别：目标模型从 1.5B 换成 3B，实体数 100 -> 300，每题查 2 次。
设计不变：每实体 5 条记录、40% 污染、查询时抽 2 条塞进提示词。

    python exp/p5b_3b.py
"""
from __future__ import annotations
import json, os, random, re, subprocess, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8088; BASE = "http://127.0.0.1:%d" % PORT
MODEL = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
N_ENT, N_REP, POLLUTE, K_SHOW, NQ = 300, 5, 0.4, 2, 2
SEED = 20261017
rnd = random.Random(SEED)
NAMES = ["玄枢", "流萤", "苍梧", "砚台", "寒汀", "拾光", "白露", "沉舟", "青釉", "归墟",
         "燃犀", "问渠", "照野", "漱石", "听风", "渡鸦", "拾贝", "临渊", "叠翠", "拂晓"]

ents = [dict(id=i, name="%s-%03d" % (NAMES[i % len(NAMES)], i), code=rnd.randrange(1000, 9999))
        for i in range(N_ENT)]
stream = []
for e in ents:
    vals = [e["code"]] * N_REP
    k = int(N_REP * POLLUTE)
    for j in rnd.sample(range(N_REP), k):
        vals[j] = rnd.randrange(1000, 9999)
    rnd.shuffle(vals)
    for v in vals:
        stream.append(dict(ent=e["id"], val=v))
rnd.shuffle(stream)


def build_memory(mode):
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


def ask(prompt):
    r = requests.post(BASE + "/v1/chat/completions", timeout=180, json={
        "messages": [{"role": "system", "content": "你是一个只做选择题的助手。只输出一个大写字母，不要任何解释。"},
                     {"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0})
    m = re.search(r"[ABCD]", (r.json()["choices"][0]["message"]["content"] or "").upper())
    return m.group(0) if m else "?"


def run_arm(mode, mem, oracle=0):
    hit = tot = 0
    t0 = time.time()
    for e in ents:
        pool = mem[e["id"]]
        for _ in range(NQ):
            shown = [] if not pool else ([pool[0], pool[0]] if len(pool) == 1 else rnd.sample(pool, K_SHOW))
            correct = e["code"]
            opts = {correct}
            while len(opts) < 4:
                opts.add(rnd.randrange(1000, 9999))
            opts = list(opts); rnd.shuffle(opts)
            L = "ABCD"
            ci = L[opts.index(correct)]
            lines = "\n".join("  %d. %s 的编号是 %d" % (i + 1, e["name"], v) for i, v in enumerate(shown)) or "  （没有记录）"
            body = ("下面是知识库中关于「%s」的记录：\n%s\n\n请判断「%s」真正的编号是哪一个，只回答字母。\n"
                    % (e["name"], lines, e["name"])
                    + "\n".join("  %s) %d" % (L[i], o) for i, o in enumerate(opts)))
            hit += int(ask(body) == ci); tot += 1
    return dict(mode=mode, acc=hit / max(tot, 1), q=tot, oracle=oracle,
                mem_size=sum(len(v) for v in mem.values()), secs=time.time() - t0)


if __name__ == "__main__":
    lf = open(os.path.join(OUT, "p5b_srv.log"), "wb")
    p = subprocess.Popen([SERVER, "-m", MODEL, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            time.sleep(2)
            try:
                if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                    time.sleep(2); break
            except Exception:
                pass
        print("实体 %d  记录 %d  污染 %.0f%%  每题查 %d 次" % (N_ENT, len(stream), POLLUTE * 100, NQ))
        res = []
        for mode in ("clean", "raw", "gate"):
            mem, oracle = build_memory(mode)
            r = run_arm(mode, mem, oracle); res.append(r)
            print("  %-6s 记忆 %4d 条  oracle %3d  准确率 %.3f  (%.0f 秒)"
                  % (mode, r["mem_size"], r["oracle"], r["acc"], r["secs"]))
            sys.stdout.flush()
        d = {r["mode"]: r for r in res}
        print()
        print("=" * 66)
        print("P5b：门控记忆 + Qwen2.5-3B（300 实体，规模放大）")
        print("=" * 66)
        print("  只留真值（上界） %.3f" % d["clean"]["acc"])
        print("  原始记忆         %.3f" % d["raw"]["acc"])
        print("  门控清洗         %.3f   (oracle %d)" % (d["gate"]["acc"], d["gate"]["oracle"]))
        print("  -> 相对原始记忆 %+.4f" % (d["gate"]["acc"] - d["raw"]["acc"]))
        json.dump(res, open(os.path.join(OUT, "results_p5b_3b.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
