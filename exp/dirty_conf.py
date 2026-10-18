# -*- coding: utf-8 -*-
"""脏记忆 + 高置信的对照：级联能不能救"抄错"？

逐题记录 (对/错, mean_logprob)，然后看：
 1) 错误是否集中在低置信区（若否 -> 级联抓不到）
 2) 用 HumanEval 的阈值 -0.046，能拦截多少比例的错题
"""
from __future__ import annotations
import json, os, random, re, subprocess, sys, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8085; BASE = "http://127.0.0.1:%d" % PORT
MODEL = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
N_ENT, N_REP, POLLUTE, K_SHOW, NQ = 200, 5, 0.4, 2, 2
THR = -0.046
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
raw = {e["id"]: [] for e in ents}
for r in stream:
    raw[r["ent"]].append(r["val"])


def ask(prompt):
    r = requests.post(BASE + "/v1/chat/completions", timeout=180, json={
        "messages": [{"role": "system", "content": "你是一个只做选择题的助手。只输出一个大写字母，不要任何解释。"},
                     {"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": 0}).json()
    ch = r["choices"][0]
    m = re.search(r"[ABCD]", (ch["message"]["content"] or "").upper())
    lps = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
           if x and x.get("logprob") is not None]
    return (m.group(0) if m else "?"), (sum(lps) / len(lps) if lps else None)


if __name__ == "__main__":
    lf = open(os.path.join(OUT, "dirty_srv.log"), "wb")
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
        rows = []
        for e in ents:
            pool = raw[e["id"]]
            for _ in range(NQ):
                shown = rnd.sample(pool, K_SHOW) if len(pool) >= K_SHOW else pool
                correct = e["code"]
                opts = {correct}
                while len(opts) < 4:
                    opts.add(rnd.randrange(1000, 9999))
                opts = list(opts); rnd.shuffle(opts)
                L = "ABCD"; ci = L[opts.index(correct)]
                lines = "\n".join("  %d. %s 的编号是 %d" % (i + 1, e["name"], v) for i, v in enumerate(shown))
                body = ("下面是知识库中关于「%s」的记录：\n%s\n\n请判断「%s」真正的编号是哪一个，只回答字母。\n"
                        % (e["name"], lines, e["name"]) + "\n".join("  %s) %d" % (L[i], o) for i, o in enumerate(opts)))
                got, lp = ask(body)
                if lp is not None:
                    rows.append(dict(ok=int(got == ci), lp=lp))
        json.dump(rows, open(os.path.join(OUT, "results_dirty_conf.json"), "w", encoding="utf-8"), indent=1)
        y = np.array([r["ok"] for r in rows]); lp = np.array([r["lp"] for r in rows])
        n = len(rows)
        print("脏记忆（raw）下的 %d 道题：准确率 %.3f" % (n, y.mean()))
        print()
        q = np.quantile(lp, [0.25, 0.5, 0.75])
        print("按 mean_logprob 四分位：")
        for i, (lo, hi) in enumerate([(-99, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 99)]):
            m = (lp > lo) & (lp <= hi)
            print("  组%d %-16s n=%-4d 通过率 %.3f" % (i + 1, "(%.4f,%.4f]" % (lo if lo > -99 else -99, hi), m.sum(), y[m].mean()))
        print()
        esc = lp <= THR
        err = y == 0
        print("用 HumanEval 阈值 %.3f：" % THR)
        print("  升级率 %.1f%%（%d/%d）" % (100 * esc.mean(), esc.sum(), n))
        print("  错题共 %d 道，其中被阈值抓到 %d 道 = %.1f%%" % (err.sum(), (err & esc).sum(), 100 * (err & esc).sum() / max(err.sum(), 1)))
        print("  升级后能救回的错题上限 = %d 道 -> 准确率上限 %.3f" % ((err & esc).sum(), (y.sum() + (err & esc).sum()) / n))
        print()
        print("判读：若『被抓到的错题比例』很低，说明级联对抄错几乎无效，只有门控记忆能救。")
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
