# -*- coding: utf-8 -*-
"""长压力负载：两个服务常驻，连打 N 条混合请求，测延迟分布与显存漂移。

关键：不再每条请求重启服务（那样会把 3~7 秒的加载算进延迟，不真实）。
"""
import json, os, random, re, statistics, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"
P3, P9 = 8062, 8061
N = 40
WINDOW, ESC_FRAC = 20, 0.2      # 调控旋钮：历史窗口 / 升级分位

def free():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip())

def launch(model, port):
    lf = open(os.path.join(ROOT, "out", "load_%d.log" % port), "wb")
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
    p.terminate(); lf.close(); raise SystemExit("load fail")

def chat(port, msg, lp=False, mx=120):
    b = {"messages": [{"role": "user", "content": msg}], "max_tokens": mx, "temperature": 0}
    if lp:
        b["logprobs"] = True; b["top_logprobs"] = 0
    d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600, json=b).json()
    ch = d["choices"][0]
    v = None
    if lp:
        vals = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
                if x and x.get("logprob") is not None]
        v = sum(vals) / len(vals) if vals else None
    return ch["message"]["content"] or "", v, d.get("usage", {}).get("completion_tokens", 0)

# 两类请求
rnd = random.Random(11)
MEMQ = []
for i in range(20):
    code = 2000 + i * 37
    opts = [code, code + 5, code - 9, code + 400]
    rnd.shuffle(opts); L = "ABCD"
    ci = L[opts.index(code)]
    MEMQ.append((("下面是知识库中关于「流萤-%03d」的记录：\n  1. 流萤-%03d 的编号是 %d\n  2. 流萤-%03d 的编号是 %d\n\n"
                  "请判断真正的编号，只回答字母。\n" % (i, i, code, i, code))
                 + "\n".join("  %s) %d" % (L[j], o) for j, o in enumerate(opts)), ci))
CODEQ = ["写一个 Python 函数 %s。只输出代码块。" % f for f in
         ("is_even(n)", "factorial(n)", "sum_list(xs)", "max_of(a,b)", "count_words(s)", "reverse(s)",
          "is_leap(y)", "abs_diff(a,b)", "min_max(xs)", "repeat(s,n)")]

if __name__ == "__main__":
    p3, lf3 = launch(M3, P3)
    p9, lf9 = launch(M9, P9)
    try:
        print("两个服务常驻，空闲显存 %d MiB" % free())
        hist, recs, errs = [], [], 0
        t_start = time.perf_counter()
        for k in range(N):
            kind = "M" if k % 2 == 0 else "C"
            if kind == "M":
                q, gold = MEMQ[(k // 2) % len(MEMQ)]
            else:
                q = CODEQ[(k // 2) % len(CODEQ)]
            t0 = time.perf_counter()
            try:
                txt, lp, nt = chat(P3, q, lp=True, mx=150)
            except Exception:
                errs += 1; continue
            esc = False
            if lp is not None and len(hist) >= 5:
                thr = sorted(hist)[max(0, int(len(hist) * ESC_FRAC) - 1)]
                esc = lp <= thr
            hist.append(lp if lp is not None else 0.0)
            tier = "L1"
            if esc:
                try:
                    txt, _, nt2 = chat(P9, q, mx=150)
                    nt += nt2; tier = "L2"
                except Exception:
                    errs += 1
            dt = time.perf_counter() - t0
            ok = None
            if kind == "M":
                ok = bool(re.search(gold, txt.upper()))
            recs.append(dict(k=k, kind=kind, tier=tier, lp=lp, sec=dt, tok=nt, ok=ok, free=free()))
            if (k + 1) % 10 == 0:
                print("  %d/%d  升级 %d  错 %d  累计 %.0f 秒"
                      % (k + 1, N, sum(1 for r in recs if r["tier"] == "L2"), errs, time.perf_counter() - t_start))
                sys.stdout.flush()
        lat = sorted(r["sec"] for r in recs)
        p50 = statistics.median(lat)
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        m = [r for r in recs if r["kind"] == "M"]
        print()
        print("=" * 66)
        print("长压力负载（%d 条混合请求，两服务常驻）" % len(recs))
        print("=" * 66)
        print("  总耗时 %.0f 秒   平均 %.2f 秒/条" % (time.perf_counter() - t_start, sum(lat) / len(lat)))
        print("  延迟 P50 %.2f 秒   P95 %.2f 秒   最大 %.2f 秒" % (p50, p95, max(lat)))
        print("  升级率 %.0f%%（%d/%d）   异常 %d 次"
              % (100 * sum(1 for r in recs if r["tier"] == "L2") / max(len(recs), 1),
                 sum(1 for r in recs if r["tier"] == "L2"), len(recs), errs))
        print("  记忆题判对 %d/%d" % (sum(1 for r in m if r["ok"]), len(m)))
        f0, f1 = recs[0]["free"], recs[-1]["free"]
        print("  显存空闲：开始 %d MiB -> 结束 %d MiB（变化 %+d）" % (f0, f1, f1 - f0))
        print("  总生成 token %d" % sum(r["tok"] for r in recs))
        json.dump(recs, open(os.path.join(ROOT, "out", "results_loadtest.json"), "w", encoding="utf-8"), indent=1)
    finally:
        for p, lf in ((p3, lf3), (p9, lf9)):
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
