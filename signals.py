# -*- coding: utf-8 -*-
"""信号验证：mean_logprob vs min_logprob，哪个更能分开"对"与"错"。

数据：20 道代码（真跑断言判对错）+ 20 道写作（机械评分，满分 3 -> >=2 记为"对"）。
对每个信号算：最低四分位组的通过率 vs 最高四分位组的通过率（分离度）。
"""
import json, os, re, statistics, subprocess, sys, tempfile, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
P3 = 8054
FENCE = chr(96) * 3

CODE = [
 ("def is_even(n)", "返回 n 是否为偶数", "assert is_even(2)\nassert not is_even(3)"),
 ("def factorial(n)", "返回 n 的阶乘", "assert factorial(0)==1\nassert factorial(5)==120"),
 ("def sum_list(xs)", "返回列表元素之和", "assert sum_list([1,2,3])==6\nassert sum_list([])==0"),
 ("def count_words(s)", "返回单词个数（按空白切分）", "assert count_words('a b c')==3"),
 ("def reverse(s)", "返回反转字符串", "assert reverse('abc')=='cba'"),
 ("def is_leap(y)", "判断闰年", "assert is_leap(2024)\nassert not is_leap(2023)"),
 ("def abs_diff(a,b)", "返回差的绝对值", "assert abs_diff(3,7)==4"),
 ("def min_max(xs)", "返回 (最小值, 最大值)", "assert min_max([3,1,4])==(1,4)"),
 ("def repeat(s,n)", "返回 s 重复 n 次", "assert repeat('ab',3)=='ababab'"),
 ("def gcd(a,b)", "返回最大公约数", "assert gcd(12,18)==6"),
 ("def is_prime(n)", "返回 n 是否为质数", "assert is_prime(7)\nassert not is_prime(8)"),
 ("def fib(n)", "返回第 n 个斐波那契数", "assert fib(0)==0\nassert fib(10)==55"),
 ("def flatten(xs)", "展平一层嵌套", "assert flatten([[1,2],[3]])==[1,2,3]"),
 ("def char_count(s)", "返回字符串长度", "assert char_count('abc')==3"),
 ("def power(a,b)", "返回 a 的 b 次方", "assert power(2,10)==1024"),
 ("def avg(xs)", "返回平均值", "assert avg([1,2,3])==2"),
 ("def is_upper(s)", "判断是否全大写", "assert is_upper('ABC')\nassert not is_upper('Abc')"),
 ("def square(n)", "返回平方", "assert square(7)==49"),
 ("def triple(n)", "返回三倍", "assert triple(4)==12"),
 ("def negate(n)", "返回相反数", "assert negate(5)==-5"),
]
WRITE = [
 ("写一段 150 到 200 字关于「缓存」的说明。", [], (150, 200), None),
 ("写一段 200 到 250 字关于「并发」的说明，并包含 2 个以 ## 开头的小标题。", [], (200, 250), 2),
 ("写一段 180 到 220 字关于「日志」的说明。", [], (180, 220), None),
 ("写一段 160 到 210 字关于「加密」的说明，并包含恰好 1 个以 ## 开头的小标题。", [], (160, 210), 1),
 ("写一段 150 到 200 字关于「索引」的说明。", [], (150, 200), None),
 ("写一段 200 到 250 字关于「重试」的说明，并包含「退避」「失败」两个词。", ["退避", "失败"], (200, 250), None),
 ("写一段 180 到 220 字关于「限流」的说明，并包含「令牌」「桶」两个词。", ["令牌", "桶"], (180, 220), None),
 ("写一段 160 到 210 字关于「快照」的说明。", [], (160, 210), None),
 ("写一段 150 到 200 字关于「回滚」的说明，并包含 2 个以 ## 开头的小标题。", [], (150, 200), 2),
 ("写一段 200 到 250 字关于「灰度」的说明。", [], (200, 250), None),
 ("写一段 180 到 220 字关于「埋点」的说明，并包含「事件」「采集」两个词。", ["事件", "采集"], (180, 220), None),
 ("写一段 160 到 210 字关于「熔断」的说明。", [], (160, 210), None),
 ("写一段 150 到 200 字关于「分片」的说明，并包含恰好 1 个以 ## 开头的小标题。", [], (150, 200), 1),
 ("写一段 200 到 250 字关于「副本」的说明。", [], (200, 250), None),
 ("写一段 180 到 220 字关于「幂等」的说明，并包含「重复」「结果」两个词。", ["重复", "结果"], (180, 220), None),
 ("写一段 160 到 210 字关于「超时」的说明。", [], (160, 210), None),
 ("写一段 150 到 200 字关于「序列化」的说明，并包含 2 个以 ## 开头的小标题。", [], (150, 200), 2),
 ("写一段 200 到 250 字关于「压缩」的说明。", [], (200, 250), None),
 ("写一段 180 到 220 字关于「校验」的说明，并包含「哈希」「一致」两个词。", ["哈希", "一致"], (180, 220), None),
 ("写一段 160 到 210 字关于「迁移」的说明。", [], (160, 210), None),
]

def launch(model, port):
    lf = open(os.path.join(ROOT, "out", "sig_%d.log" % port), "wb")
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

def chat(port, msg, mx=500):
    d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600, json={
        "messages": [{"role": "user", "content": msg}], "max_tokens": mx, "temperature": 0,
        "logprobs": True, "top_logprobs": 0}).json()
    ch = d["choices"][0]
    lps = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
           if x and x.get("logprob") is not None]
    return (ch["message"]["content"] or "",
            (sum(lps) / len(lps)) if lps else None,
            min(lps) if lps else None)

def strip_f(t):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, t, re.S)
    return m[0] if m else t

def run_py(code, asserts):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code + "\n\n" + asserts + "\n"); p = f.name
    try:
        return subprocess.run([sys.executable, p], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(p)
        except Exception: pass

def score_write(txt, words, rng, nh):
    s = 0
    if all(w in txt for w in words): s += 1
    n = len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", txt))
    if rng[0] <= n <= rng[1]: s += 1
    heads = len(re.findall(r"^##\s", txt, re.M))
    if nh is None or heads == nh: s += 1
    return s

if __name__ == "__main__":
    p3, lf3 = launch(M3, P3)
    try:
        rows = []
        for fn, desc, asserts in CODE:
            txt, m, mn = chat(P3, "用 Python 写一个函数，签名是 %s，%s。只输出一个代码块。" % (fn, desc))
            rows.append(dict(kind="C", mean=m, min=mn, ok=int(run_py(strip_f(txt), asserts))))
        for q, words, rng, nh in WRITE:
            txt, m, mn = chat(P3, q)
            rows.append(dict(kind="W", mean=m, min=mn, ok=int(score_write(txt, words, rng, nh) >= 2)))
        print("  采完 %d 条（代码通过 %d/20，写作达标 %d/20）"
              % (len(rows), sum(r["ok"] for r in rows if r["kind"] == "C"),
                 sum(r["ok"] for r in rows if r["kind"] == "W")))
        json.dump(rows, open(os.path.join(ROOT, "out", "results_signals.json"), "w", encoding="utf-8"), indent=1)
        print()
        print("=" * 70)
        print("信号分离度（每题：对/错 与 两个置信度信号）")
        print("=" * 70)
        for sig in ("mean", "min"):
            for scope in ("all", "W", "C"):
                sub = [r for r in rows if (scope == "all" or r["kind"] == scope) and r[sig] is not None]
                if len(sub) < 8: continue
                v = np.array([r[sig] for r in sub]); y = np.array([r["ok"] for r in sub])
                q1, q3 = np.quantile(v, 0.25), np.quantile(v, 0.75)
                lo = y[v <= q1]; hi = y[v >= q3]
                sep = (hi.mean() if len(hi) else float("nan")) - (lo.mean() if len(lo) else float("nan"))
                print("  %-5s %-4s  低分组通过 %.2f  高分组通过 %.2f  分离度 %+.3f  (n=%d)"
                      % (sig, scope, lo.mean() if len(lo) else -1, hi.mean() if len(hi) else -1, sep, len(sub)))
    finally:
        p3.terminate()
        try: p3.wait(timeout=20)
        except Exception: p3.kill()
        lf3.close()
