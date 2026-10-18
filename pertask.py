# -*- coding: utf-8 -*-
"""分任务类型的负载测试：写作 / 代码推理，各自判分。

写作：要素齐全 + 字数区间 + 小标题数（各 1 分）
代码：真跑官方式断言（写函数 -> 抽代码块 -> exec -> assert）
"""
import json, os, re, statistics, subprocess, sys, tempfile, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"
P3, P9 = 8060, 8059
FENCE = chr(96) * 3
WINDOW, ESC_FRAC = 20, 0.2

WRITE = [
 ("写一段 200 到 260 字的产品介绍，必须包含「本地部署」「隐私」「离线」三个词，并包含恰好 3 个小标题（每个以 ## 开头）。",
  ["本地部署", "隐私", "离线"], (200, 260), 3),
 ("写一段 150 到 200 字的技术说明，必须包含「延迟」「吞吐」「显存」三个词，不要使用小标题。",
  ["延迟", "吞吐", "显存"], (150, 200), 0),
 ("用 120 到 180 字解释什么是向量数据库，必须包含「相似度」和「索引」两个词。",
  ["相似度", "索引"], (120, 180), None),
 ("写一段 200 到 280 字的开源项目 README 开头，必须包含「许可证」「贡献」两个词，并包含恰好 2 个以 ## 开头的小标题。",
  ["许可证", "贡献"], (200, 280), 2),
 ("写一段 180 到 240 字的复盘，必须包含「指标」「原因」「改进」三个词。",
  ["指标", "原因", "改进"], (180, 240), None),
]

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
]

def launch(model, port):
    lf = open(os.path.join(ROOT, "out", "pt_%d.log" % port), "wb")
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

def chat(port, msg, lp=False, mx=600):
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

def one(kind, item, hist, p3, p9):
    if kind == "W":
        q, words, rng, nh = item
        prompt = q
    else:
        fn, desc, asserts = item
        prompt = "用 Python 写一个函数，签名是 %s，%s。只输出一个代码块。" % (fn, desc)
    t0 = time.perf_counter()
    txt, lp = chat(p3, prompt, lp=True, mx=700)
    esc = False
    if lp is not None and len(hist) >= 5:
        thr = sorted(hist)[max(0, int(len(hist) * ESC_FRAC) - 1)]
        esc = lp <= thr
    hist.append(lp if lp is not None else 0.0)
    tier = "L1"
    if esc:
        txt, _ = chat(p9, prompt, mx=700)
        tier = "L2"
    dt = time.perf_counter() - t0
    if kind == "W":
        sc, mx_ = score_write(txt, words, rng, nh), 3
    else:
        sc, mx_ = int(run_py(strip_f(txt), item[2])), 1
    return dict(kind=kind, tier=tier, lp=lp, sec=dt, score=sc, max=mx_)

if __name__ == "__main__":
    p3, lf3 = launch(M3, P3)
    p9, lf9 = launch(M9, P9)
    try:
        out = {}
        for kind, items in (("W", WRITE), ("C", CODE)):
            hist, recs = [], []
            for it in items:
                r = one(kind, it, hist, P3, P9); recs.append(r)
                print("  %s 分 %d/%d  升级 %s  %.2f 秒"
                      % ("写作" if kind == "W" else "代码", r["score"], r["max"], r["tier"], r["sec"]))
                sys.stdout.flush()
            lat = sorted(r["sec"] for r in recs)
            out[kind] = dict(score=sum(r["score"] for r in recs), max=sum(r["max"] for r in recs),
                             p50=statistics.median(lat), p95=lat[-1],
                             esc=sum(1 for r in recs if r["tier"] == "L2"), n=len(recs))
        print()
        print("=" * 62)
        print("分任务类型负载（各 5/10 条，两服务常驻）")
        print("=" * 62)
        for kind, nm in (("W", "写作"), ("C", "代码推理")):
            o = out[kind]
            print("  %-6s 得分 %d/%d   升级 %d/%d   P50 %.2f 秒   P95 %.2f 秒"
                  % (nm, o["score"], o["max"], o["esc"], o["n"], o["p50"], o["p95"]))
        json.dump(out, open(os.path.join(ROOT, "out", "results_pertask.json"), "w", encoding="utf-8"), indent=1)
    finally:
        for p, lf in ((p3, lf3), (p9, lf9)):
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
