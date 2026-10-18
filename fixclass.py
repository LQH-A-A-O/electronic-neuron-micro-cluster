# -*- coding: utf-8 -*-
"""修复验证：按输出形态自分类，每类用自己的置信度分布定阈值。

修复前：全局一份 history -> 写作天然低 logprob -> 74.6% 无谓升级
修复后：代码/写作各自一份 history -> 各自取自己的 20% 分位
"""
import json, os, re, statistics, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"
P3, P9 = 8056, 8055
FENCE = chr(96) * 3
ESC_FRAC = 0.2

CODEQ = ["写一个 Python 函数 %s，只输出代码块。" % f for f in
         ("is_even(n)", "factorial(n)", "sum_list(xs)", "max_of(a,b)", "count_words(s)", "reverse(s)",
          "is_leap(y)", "abs_diff(a,b)", "min_max(xs)", "repeat(s,n)", "gcd(a,b)", "is_prime(n)",
          "fib(n)", "flatten(xs)", "unique(xs)", "char_count(s)", "power(a,b)", "avg(xs)",
          "is_upper(s)", "square(n)")]
WRITEQ = ["写一段 %d 到 %d 字关于「%s」的说明。" % (a, b, t) for a, b, t in
          ((150, 200, "缓存"), (200, 250, "并发"), (180, 220, "日志"), (160, 210, "加密"),
           (150, 200, "索引"), (200, 250, "重试"), (180, 220, "限流"), (160, 210, "快照"),
           (150, 200, "回滚"), (200, 250, "灰度"), (180, 220, "埋点"), (160, 210, "熔断"),
           (150, 200, "分片"), (200, 250, "副本"), (180, 220, "幂等"), (160, 210, "超时"),
           (150, 200, "序列化"), (200, 250, "压缩"), (180, 220, "校验"), (160, 210, "迁移"))]

def launch(model, port):
    lf = open(os.path.join(ROOT, "out", "fix_%d.log" % port), "wb")
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

def chat(port, msg, lp=False, mx=400):
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
    return ch["message"]["content"] or "", v

def classify(txt):
    """按输出形态自分类：含代码块 -> code，否则 text。"""
    return "code" if FENCE in txt else "text"

if __name__ == "__main__":
    p3, lf3 = launch(M3, P3)
    p9, lf9 = launch(M9, P9)
    try:
        hist = {"code": [], "text": []}      # 修法核心：每类一份历史
        rows = []
        jobs = [("W", q) for q in WRITEQ] + [("C", q) for q in CODEQ]
        for i, (kind, q) in enumerate(jobs):
            t0 = time.perf_counter()
            txt, lp = chat(P3, q, lp=True)
            cls = classify(txt)
            esc = False
            h = hist[cls]
            if lp is not None and len(h) >= 5:
                thr = sorted(h)[max(0, int(len(h) * ESC_FRAC) - 1)]
                esc = lp <= thr
            h.append(lp if lp is not None else 0.0)
            tier = "L1"
            if esc:
                txt2, _ = chat(P9, q)
                tier = "L2"
            rows.append(dict(kind=kind, cls=cls, lp=lp, esc=esc, tier=tier, sec=time.perf_counter() - t0))
            if (i + 1) % 10 == 0:
                print("  %d/%d" % (i + 1, len(jobs))); sys.stdout.flush()
        # 对比：修复前（全局一份历史）
        gal = [r["lp"] for r in rows if r["lp"] is not None]
        print()
        print("=" * 64)
        print("修复验证（写作 20 + 代码 20，按输出形态自分类）")
        print("=" * 64)
        for k, nm in (("W", "写作"), ("C", "代码")):
            s = [r for r in rows if r["kind"] == k]
            e = sum(1 for r in s if r["esc"])
            e2 = 0
            g = []
            # 重放：用全局历史算"修复前"的升级
            for r in sorted(rows, key=lambda x: 0):
                pass
            ghist = []
            for r in rows:
                if r["lp"] is not None and len(ghist) >= 5:
                    t = sorted(ghist)[max(0, int(len(ghist) * ESC_FRAC) - 1)]
                    if r["kind"] == k and r["lp"] <= t:
                        e2 += 1
                ghist.append(r["lp"] if r["lp"] is not None else 0.0)
            print("  %-4s %2d 条  修复后升级 %2d (%3.0f%%)   修复前升级 %2d (%3.0f%%)"
                  % (nm, len(s), e, 100 * e / len(s), e2, 100 * e2 / len(s)))
        lat = sorted(r["sec"] for r in rows)
        print()
        print("  总升级 %d/%d (%.0f%%)   P50 %.2f 秒   P95 %.2f 秒"
              % (sum(1 for r in rows if r["esc"]), len(rows),
                 100 * sum(1 for r in rows if r["esc"]) / len(rows),
                 statistics.median(lat), lat[int(len(lat) * 0.95)]))
        json.dump(rows, open(os.path.join(ROOT, "out", "results_fixclass.json"), "w", encoding="utf-8"), indent=1)
    finally:
        for p, lf in ((p3, lf3), (p9, lf9)):
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
