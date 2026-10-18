# -*- coding: utf-8 -*-
"""3B -> 9B 的升级率与三层级联的实账。

    python exp/ladder3.py phase1     # 3B 采 K=3，算互证度
    python exp/ladder3.py phase2 0.34
全部 164 题（不用子集 —— 子集把我坑过一次）。
"""
from __future__ import annotations
import gzip, json, os, re, subprocess, sys, tarfile, tempfile, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8091; BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3
K = 3
MID = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
BIG = r"E:\aqca\work\llm\models\Qwen3.5-9B-Q4_K_M.gguf"
CACHE = os.path.join(OUT, "ladder3_samples.json")


def load_he():
    with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
        n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
        return [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]


def start(model):
    lf = open(os.path.join(OUT, "ladder3_srv.log"), "ab")
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
    p.terminate(); lf.close(); raise SystemExit("load failed")


def gen(prompt, mx=450, temp=0.9):
    r = requests.post(BASE + "/v1/chat/completions", timeout=900, json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": temp})
    return r.json()["choices"][0]["message"]["content"] or ""


def strip_fence(t):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, t, re.S)
    return m[0] if m else re.sub(r"^\s*" + FENCE + r"(?:python)?\s*\n", "", t)


def run_he(prob, code):
    c = code if prob["prompt"].strip().split("\n")[0] in code else prob["prompt"] + code
    prog = c + "\n\n" + prob["test"] + "\n\ncheck(%s)\n" % prob["entry_point"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass


def agreement(samples):
    c = {}
    for x in samples:
        k = re.sub(r"\s+", "", re.sub(r"#.*", "", x))
        c[k] = c.get(k, 0) + 1
    return max(c.values()) / len(samples)


def phase1():
    he = load_he()
    p, lf = start(MID)
    try:
        rows = []
        t0 = time.perf_counter()
        for i, pr in enumerate(he):
            prompt = pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"
            s = [strip_fence(gen(prompt, 450, 0.9)) for _ in range(K)]
            rows.append(dict(idx=i, samples=s, agree=agreement(s),
                             base_pass=int(run_he(pr, s[0])), big=None))
            if (i + 1) % 20 == 0:
                print("  %d/%d 平均互证 %.3f" % (i + 1, len(he), sum(r["agree"] for r in rows) / len(rows)))
                sys.stdout.flush()
        json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        print("phase1 完成 %.0f 秒" % (time.perf_counter() - t0))
        print("3B 单采样 pass@1 = %.3f" % (sum(r["base_pass"] for r in rows) / len(rows)))
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)


def phase2(thr):
    rows = json.load(open(CACHE, encoding="utf-8"))
    he = load_he()
    esc = [r for r in rows if r["agree"] < thr]
    print("升级 %d/%d = %.0f%%" % (len(esc), len(rows), 100 * len(esc) / len(rows)))
    p, lf = start(BIG)
    try:
        for j, r in enumerate(esc):
            try:
                r["big"] = strip_fence(gen(he[r["idx"]]["prompt"] + "\n\n只补全这个函数体，不要任何解释。", 450, 0.0))
            except Exception:
                r["big"] = None
            if (j + 1) % 20 == 0:
                print("  %d/%d" % (j + 1, len(esc))); sys.stdout.flush()
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)
    json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    n = len(rows)
    base = sum(r["base_pass"] for r in rows) / n
    casc = sum((run_he(he[r["idx"]], r["big"]) if r["big"] else r["base_pass"]) for r in rows) / n
    ef = len(esc) / n
    lat = K * 450 / 66.0 + ef * 450 / 25.0
    print()
    print("=" * 74)
    print("三层级联实账（HumanEval 全 164 题，3B K=3 -> 9B）")
    print("=" * 74)
    print("  3B 单采样         pass@1 = %.3f   延迟 ≈ %.1f 秒/题" % (base, 450 / 66.0))
    print("  3B->9B 级联       pass@1 = %.3f   延迟 ≈ %.1f 秒/题  升级 %.0f%%" % (casc, lat, 100 * ef))
    print("  9B 单独（实测）    pass@1 = 0.768   延迟 ≈ %.1f 秒/题" % (450 / 25.0))
    json.dump(dict(n=n, base=base, casc=casc, esc_frac=ef, thr=thr, lat=lat),
              open(os.path.join(OUT, "results_ladder3.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "phase1":
        phase1()
    else:
        phase2(float(sys.argv[2]) if len(sys.argv) > 2 else 0.34)
