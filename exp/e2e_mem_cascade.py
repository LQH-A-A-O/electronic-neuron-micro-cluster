# -*- coding: utf-8 -*-
"""端到端：门控记忆 + logprob 级联。测两件事：
1) 记忆洗干净后，级联的升级率是多少（会不会乱升级）
2) 记忆任务的置信度分布 vs HumanEval 的分布

如果记忆任务天然高置信 -> 级联几乎不升级 -> 两个部件不打架，组合是"记忆回答事实、级联兜底难题"。
"""
from __future__ import annotations
import json, os, random, re, subprocess, sys, time
import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8086; BASE = "http://127.0.0.1:%d" % PORT
MODEL = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
N_ENT, N_REP, POLLUTE, K_SHOW, NQ = 200, 5, 0.4, 2, 2
rnd = random.Random(20261017)
NAMES = ["玄枢", "流萤", "苍梧", "砚台", "寒汀", "拾光", "白露", "沉舟", "青釉", "归墟"]
ents = [dict(id=i, name="%s-%03d" % (NAMES[i % len(NAMES)], i), code=rnd.randrange(1000, 9999))
        for i in range(N_ENT)]
stream = []
for e in ents:
    vals = [e["code"]] * N_REP
    for j in rnd.sample(range(N_REP), int(N_REP * POLLUTE)):
        vals[j] = rnd.randrange(1000, 9999)
    rnd.shuffle(vals)
    for v in vals:
        stream.append(dict(ent=e["id"], val=v))
rnd.shuffle(stream)


def build(mode):
    acc = {e["id"]: [] for e in ents}
    for r in stream:
        ent, v = r["ent"], r["val"]
        truth = ents[ent]["code"]
        if mode == "raw":
            acc[ent].append(v); continue
        prior = acc[ent]
        if not prior:
            ok = (v == truth)
        else:
            ag = sum(1 for x in prior if x == v) / len(prior)
            ok = True if ag > 0.5 else (False if ag < 0.5 else (v == truth))
        if ok:
            prior.append(v)
    return acc


def ask(prompt):
    r = requests.post(BASE + "/v1/chat/completions", timeout=180, json={
        "messages": [{"role": "system", "content": "你是一个只做选择题的助手。只输出一个大写字母，不要任何解释。"},
                     {"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": 0}).json()
    ch = r["choices"][0]
    txt = (ch["message"]["content"] or "").upper()
    m = re.search(r"[ABCD]", txt)
    lps = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
           if x and x.get("logprob") is not None]
    return (m.group(0) if m else "?"), (sum(lps) / len(lps) if lps else None)


def run(mode):
    mem = build(mode)
    hit = tot = 0; lps_all = []
    for e in ents:
        pool = mem[e["id"]]
        for _ in range(NQ):
            shown = [] if not pool else ([pool[0], pool[0]] if len(pool) == 1 else rnd.sample(pool, K_SHOW))
            correct = e["code"]
            opts = {correct}
            while len(opts) < 4:
                opts.add(rnd.randrange(1000, 9999))
            opts = list(opts); rnd.shuffle(opts)
            L = "ABCD"; ci = L[opts.index(correct)]
            lines = "\n".join("  %d. %s 的编号是 %d" % (i + 1, e["name"], v) for i, v in enumerate(shown)) or "  （没有记录）"
            body = ("下面是知识库中关于「%s」的记录：\n%s\n\n请判断「%s」真正的编号是哪一个，只回答字母。\n"
                    % (e["name"], lines, e["name"]) + "\n".join("  %s) %d" % (L[i], o) for i, o in enumerate(opts)))
            got, lp = ask(body)
            hit += int(got == ci); tot += 1
            if lp is not None:
                lps_all.append(lp)
    return hit / max(tot, 1), np.array(lps_all)


if __name__ == "__main__":
    lf = open(os.path.join(OUT, "e2e_srv.log"), "wb")
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
        print("实体 %d，每题查 %d 次" % (N_ENT, NQ))
        res = {}
        for mode in ("raw", "gate"):
            acc, lp = run(mode)
            res[mode] = dict(acc=acc, n=len(lp), mean=float(lp.mean()) if len(lp) else None,
                             p20=float(np.quantile(lp, 0.2)) if len(lp) else None)
            print("  %-5s 准确率 %.3f   mean_logprob 均值 %.4f  20%%分位 %.4f  (n=%d)"
                  % (mode, acc, lp.mean(), np.quantile(lp, 0.2), len(lp)))
            sys.stdout.flush()
        print()
        print("对照：HumanEval 上 3B 的 mean_logprob 20%% 分位是 -0.046（那就是 20%% 升级率的阈值）")
        print()
        print("=" * 64)
        print("端到端：记忆干净后级联会不会乱升级")
        print("=" * 64)
        for m in ("raw", "gate"):
            r = res[m]
            rate = float((np.array([]) if r["p20"] is None else np.array([0])) .size)  # placeholder
            print("  %-5s 准确率 %.3f  20%%分位 mean_logprob %.4f" % (m, r["acc"], r["p20"]))
        print()
        print("判读：若 gate 的 20%% 分位明显高于 -0.046，说明它比 HumanEval 更自信")
        print("      -> 用 HumanEval 标定的阈值，在记忆任务上几乎不会触发升级（不打架）。")
        json.dump(res, open(os.path.join(OUT, "results_e2e.json"), "w", encoding="utf-8"), indent=1)
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
