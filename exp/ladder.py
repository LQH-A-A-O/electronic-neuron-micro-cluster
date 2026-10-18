# -*- coding: utf-8 -*-
"""分层稠密级联：1.5B（便宜）-> 3B（次便宜）-> 可选第三层。

与 MoE 方案的区别只有一条，但很关键：
    升级到 3B 只贵约 1.5 倍（100 -> 66 tok/s），
    升级到 35B-A3B 贵 10 倍（100 -> 10 tok/s）。
同一个互证判据，成本结构完全不同。

    python exp/ladder.py phase1 60        # 1.5B 采 K=3
    python exp/ladder.py phase2 0.5       # 3B 只跑互证失败的
"""
from __future__ import annotations
import gzip, json, os, re, subprocess, sys, tarfile, tempfile, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8094; BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3
K = 3
SMALL = r"E:\aqca\work\llm\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
MID = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
CACHE = os.path.join(OUT, "ladder_samples.json")


def load_he():
    with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
        n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
        return [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]


def start(model):
    lf = open(os.path.join(OUT, "ladder_srv.log"), "ab")
    p = subprocess.Popen([SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(2)
        try:
            if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                time.sleep(2); return p, lf
        except Exception:
            pass
    p.terminate(); lf.close(); raise SystemExit("加载失败")


def gen(prompt, mx=450, temp=0.9):
    r = requests.post(BASE + "/v1/chat/completions", timeout=900, json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": temp})
    return r.json()["choices"][0]["message"]["content"] or ""


def strip_fence(txt):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, txt, re.S)
    if m:
        return m[0]
    return re.sub(r"^\s*" + FENCE + r"(?:python)?\s*\n", "", txt)


def run_he(prob, completion):
    code = completion
    if prob["prompt"].strip().split("\n")[0] not in code:
        code = prob["prompt"] + code
    prog = code + "\n\n" + prob["test"] + "\n\ncheck(%s)\n" % prob["entry_point"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass


def toks(s):
    s = re.sub(r"#.*", "", s)
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[+\-*/%=<>!]+", s))


def sim(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(len(A | B), 1)


def agreement(samples):
    """与 cascade.py 完全相同的判据（代码规范化后完全相等），保证两轮可比。"""
    if not samples:
        return 0.0
    c = {}
    for s in samples:
        k = re.sub(r"\s+", "", re.sub(r"#.*", "", s))
        c[k] = c.get(k, 0) + 1
    return max(c.values()) / len(samples)


def phase1(n):
    he = load_he()[:n]
    p, lf = start(SMALL)
    try:
        rows = []
        t0 = time.perf_counter()
        for i, pr in enumerate(he):
            prompt = pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"
            s = []
            for _ in range(K):
                try: s.append(strip_fence(gen(prompt, 450, 0.9)))
                except Exception: s.append("")
            rows.append(dict(idx=i, samples=s, agree=agreement(s),
                             base_pass=int(run_he(pr, s[0])), mid=None, esc=0))
            if (i + 1) % 15 == 0:
                print("  phase1 %d/%d  平均互证 %.3f" % (i + 1, n, sum(r["agree"] for r in rows) / len(rows)))
                sys.stdout.flush()
        json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        print("phase1 完成 %.0f 秒" % (time.perf_counter() - t0))
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)


def phase2(thr):
    rows = json.load(open(CACHE, encoding="utf-8"))
    he = load_he()[:len(rows)]
    esc = [r for r in rows if r["agree"] < thr]
    print("升级 %d/%d = %.0f%%（阈值 %.2f）" % (len(esc), len(rows), 100 * len(esc) / len(rows), thr))
    p, lf = start(MID)
    try:
        for j, r in enumerate(esc):
            try:
                r["mid"] = strip_fence(gen(he[r["idx"]]["prompt"] + "\n\n只补全这个函数体，不要任何解释。", 450, 0.0))
                r["esc"] = 1
            except Exception:
                r["mid"] = None
            if (j + 1) % 10 == 0:
                print("  phase2 %d/%d" % (j + 1, len(esc))); sys.stdout.flush()
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)
    json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)

    n = len(rows)
    base = sum(r["base_pass"] for r in rows) / n
    casc = sum((run_he(he[r["idx"]], r["mid"]) if r["mid"] else r["base_pass"]) for r in rows) / n
    ef = len(esc) / n
    # 成本折算：1.5B ~100 tok/s，3B ~66 tok/s，每解 450 token
    cheap_s = 450 / 100.0
    mid_s = 450 / 66.0
    lat_single = cheap_s
    lat_bo = K * cheap_s
    lat_casc = K * cheap_s + ef * mid_s
    print()
    print("=" * 78)
    print("分层稠密级联（HumanEval 前 %d 题，K=%d，阈值 %.2f）" % (n, K, thr))
    print("=" * 78)
    print("  1.5B 单采样        pass@1 = %.3f   延迟 ≈ %.1f 秒/题   (实测基线)" % (base, lat_single))
    print("  1.5B K 采样        pass@1 = %.3f   延迟 ≈ %.1f 秒/题   (等于多花 %dx 换不出质量)" % (base, lat_bo, K))
    print("  分层稠密(1.5B->3B) pass@1 = %.3f   延迟 ≈ %.1f 秒/题   升级 %.0f%%" % (casc, lat_casc, 100 * ef))
    print("  -> 相对单采样 %+.3f，延迟 %.1fx" % (casc - base, lat_casc / lat_single))
    print()
    print("  对照（同题、上轮实测）：1.5B->35B-A3B MoE  pass@1 = 0.967  延迟 ≈ 45 秒/题（约 10x）")
    json.dump(dict(n=n, base=base, casc=casc, esc_frac=ef, thr=thr, K=K,
                   lat_single=lat_single, lat_casc=lat_casc),
              open(os.path.join(OUT, "results_ladder.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "phase1":
        phase1(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    else:
        phase2(float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
