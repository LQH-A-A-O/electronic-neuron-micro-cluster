# -*- coding: utf-8 -*-
"""2 小时长测（带分组阈值修复）：验证修复后的长期行为。

与上一版 soakt.py 的唯一区别：置信度历史按输出形态分组（修复 74.6% 无谓升级）。
"""
import json, os, re, subprocess, sys, threading, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"
P3, P9 = 8050, 8049
WINDOW, ESC_FRAC = 20, 0.2
FENCE = chr(96) * 3
LOG = os.path.join(ROOT, "out", "soak2.jsonl")
VLOG = os.path.join(ROOT, "out", "soak2_vram.jsonl")
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

def free():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip())
    except Exception:
        return -1

def launch(model, port):
    lf = open(os.path.join(ROOT, "out", "soak2_%d.log" % port), "wb")
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

def chat(port, msg, lp=False, mx=200):
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

CODEQ = ["写一个 Python 函数 %s，只输出代码块。" % f for f in
         ("is_even(n)", "factorial(n)", "sum_list(xs)", "max_of(a,b)", "count_words(s)", "reverse(s)",
          "is_leap(y)", "abs_diff(a,b)", "min_max(xs)", "repeat(s,n)", "gcd(a,b)", "is_prime(n)",
          "fib(n)", "flatten(xs)", "unique(xs)", "char_count(s)", "power(a,b)", "avg(xs)")]
WRITEQ = ["写一段 %d 到 %d 字关于「%s」的说明。" % (a, b, t) for a, b, t in
          ((150, 200, "缓存"), (200, 250, "并发"), (180, 220, "日志"), (160, 210, "加密"),
           (150, 200, "索引"), (200, 250, "重试"), (180, 220, "限流"), (160, 210, "快照"))]

if __name__ == "__main__":
    p3, lf3 = launch(M3, P3)
    p9, lf9 = launch(M9, P9)
    stop = threading.Event()
    def vloop():
        while not stop.is_set():
            with open(VLOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(t=time.time(), free=free())) + "\n")
            stop.wait(30)
    threading.Thread(target=vloop, daemon=True).start()
    hist = {"code": [], "text": []}
    n = err = esc_n = toks = 0
    t0 = time.time(); t_end = t0 + HOURS * 3600
    print("长测（分组阈值修复版）启动：%.1f 小时，空闲显存 %d MiB" % (HOURS, free())); sys.stdout.flush()
    try:
        while time.time() < t_end:
            kind = "C" if n % 3 else "W"
            q = CODEQ[n % len(CODEQ)] if kind == "C" else WRITEQ[n % len(WRITEQ)]
            ts = time.time()
            try:
                txt, lp, nt = chat(P3, q, lp=True)
            except Exception:
                err += 1; n += 1; continue
            cls = "code" if FENCE in txt else "text"
            h = hist[cls]
            esc = False
            if lp is not None and len(h) >= 5:
                t = sorted(h)[max(0, int(len(h) * ESC_FRAC) - 1)]
                esc = lp <= t
            h.append(lp if lp is not None else 0.0)
            tier = "L1"
            if esc:
                esc_n += 1
                try:
                    _, _, nt2 = chat(P9, q); nt += nt2; tier = "L2"
                except Exception:
                    err += 1
            toks += nt
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(n=n, kind=kind, cls=cls, tier=tier, lp=lp,
                                        sec=time.time() - ts, tok=nt, free=free())) + "\n")
            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                print("  %d 条  %.0f 分钟  升级 %d  错 %d  %.0f tok/s  空闲 %d MiB"
                      % (n, el / 60, esc_n, err, toks / el, free())); sys.stdout.flush()
        print("完成：%d 条，升级 %d，错 %d，%.0f 分钟" % (n, esc_n, err, (time.time() - t0) / 60))
    finally:
        stop.set()
        for p, lf in ((p3, lf3), (p9, lf9)):
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
