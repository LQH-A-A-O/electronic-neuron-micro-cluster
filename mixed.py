# -*- coding: utf-8 -*-
"""混合负载：记忆题与代码题交替进同一个系统，看路由器怎么分流。

预期（若成立）：记忆题几乎不升级，代码题约 20% 升级。
"""
import json, os, random, re, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"
P3, P9 = 8064, 8063

def launch(model, port):
    lf = open(os.path.join(ROOT, "out", "mix_%d.log" % port), "wb")
    p = subprocess.Popen([SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                return p, lf
        except Exception:
            pass
    p.terminate(); lf.close(); raise SystemExit("load fail %d" % port)

def chat(port, msg, lp=False):
    b = {"messages": [{"role": "user", "content": msg}], "max_tokens": 200, "temperature": 0}
    if lp:
        b["logprobs"] = True; b["top_logprobs"] = 0
    d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600, json=b).json()
    ch = d["choices"][0]
    v = None
    if lp:
        vals = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
                if x and x.get("logprob") is not None]
        v = sum(vals) / len(vals) if vals else None
    return ch["message"]["content"] or "", v

# 记忆题：实体编号选择题
rnd = random.Random(7)
MEM = []
for i in range(6):
    code = 1000 + i * 111
    opts = [code, code + 7, code - 13, code + 250]
    rnd.shuffle(opts); L = "ABCD"
    ci = L[opts.index(code)]
    MEM.append((("下面是知识库中关于「玄枢-%03d」的记录：\n  %d. 玄枢-%03d 的编号是 %d\n  %d. 玄枢-%03d 的编号是 %d\n\n"
                 "请判断真正的编号，只回答字母。\n" % (i, 1, i, code, 2, i, code))
                + "\n".join("  %s) %d" % (L[j], o) for j, o in enumerate(opts)), ci))

# 代码题
CODE = [
 "写一个 Python 函数 is_prime(n)，返回 n 是否为质数。只输出代码块。",
 "写一个 Python 函数 reverse_words(s)，翻转句子中单词顺序。只输出代码块。",
 "写一个 Python 函数 gcd(a,b) 求最大公约数。只输出代码块。",
 "写一个 Python 函数 count_vowels(s) 统计元音个数。只输出代码块。",
 "写一个 Python 函数 sum_digits(n) 求各位数字之和。只输出代码块。",
 "写一个 Python 函数 unique_keep_order(lst)。只输出代码块。",
]

if __name__ == "__main__":
    p3, lf3 = launch(M3, P3)
    p9 = None
    try:
        hist = []
        rows = []
        order = ["M", "C"] * 6
        for kind, i in zip(order, list(range(6)) * 2):
            if kind == "M":
                q, gold = MEM[i]
                txt, lp = chat(P3, q, lp=True)
                ok = bool(re.search(gold, txt.upper()))
            else:
                q = CODE[i]
                txt, lp = chat(P3, q, lp=True)
                ok = None  # 代码不判分，只看分流
            esc = False
            if lp is not None and len(hist) >= 4:
                thr = sorted(hist)[max(0, int(len(hist) * 0.2) - 1)]
                esc = lp <= thr
            hist.append(lp if lp is not None else 0.0)
            if esc:
                if p9 is None:
                    p9, lf9 = launch(M9, P9)
                txt, _ = chat(P9, q)
            rows.append(dict(kind=kind, lp=lp, esc=esc, ok=ok))
            print("  %s 题%d  置信度 %+.4f  升级 %s%s"
                  % ("记忆" if kind == "M" else "代码", i,
                     lp if lp is not None else 0, "是" if esc else "否",
                     "" if ok is None else ("  判对 %s" % ok)))
            sys.stdout.flush()
        m = [r for r in rows if r["kind"] == "M"]
        c = [r for r in rows if r["kind"] == "C"]
        print()
        print("=" * 60)
        print("混合负载（6 记忆 + 6 代码交替）")
        print("=" * 60)
        print("  记忆题：升级 %d/%d   判对 %d/%d"
              % (sum(r["esc"] for r in m), len(m), sum(bool(r["ok"]) for r in m), len(m)))
        print("  代码题：升级 %d/%d" % (sum(r["esc"] for r in c), len(c)))
        json.dump(rows, open(os.path.join(ROOT, "out", "results_mixed.json"), "w", encoding="utf-8"), indent=1)
    finally:
        for p, lf in ((p3, lf3),):
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
        if p9 is not None:
            p9.terminate()
            try: p9.wait(timeout=20)
            except Exception: p9.kill()
