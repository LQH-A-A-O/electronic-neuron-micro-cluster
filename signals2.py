# -*- coding: utf-8 -*-
"""信号验证（修正版）：用真有错题的 HumanEval 164 题，同时记录 mean 与 min logprob。

上一版用人工小题目，20/20 全对 -> 天花板效应，分离度全是 0，验证作废。
"""
import gzip, json, os, re, subprocess, sys, tarfile, tempfile, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
PORT = 8053
FENCE = chr(96) * 3

def load_he():
    with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
        n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
        return [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]

def strip_f(t):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, t, re.S)
    return m[0] if m else re.sub(r"^\s*" + FENCE + r"(?:python)?\s*\n", "", t)

def run_he(prob, code):
    c = code if prob["prompt"].strip().split("\n")[0] in code else prob["prompt"] + code
    prog = c + "\n\n" + prob["test"] + "\n\ncheck(%s)\n" % prob["entry_point"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog); p = f.name
    try:
        return subprocess.run([sys.executable, p], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(p)
        except Exception: pass

if __name__ == "__main__":
    he = load_he()
    lf = open(os.path.join(OUT, "sig2_srv.log"), "wb")
    p = subprocess.Popen([SERVER, "-m", M3, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            time.sleep(1)
            try:
                if requests.get("http://127.0.0.1:%d/health" % PORT, timeout=2).json().get("status") == "ok":
                    time.sleep(2); break
            except Exception:
                pass
        rows = []
        for i, pr in enumerate(he):
            r = requests.post("http://127.0.0.1:%d/v1/chat/completions" % PORT, timeout=600, json={
                "messages": [{"role": "user", "content": pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"}],
                "max_tokens": 450, "temperature": 0, "logprobs": True, "top_logprobs": 0}).json()
            ch = r["choices"][0]
            code = strip_f(ch["message"]["content"] or "")
            lps = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
                   if x and x.get("logprob") is not None]
            rows.append(dict(idx=i,
                             mean=(sum(lps) / len(lps)) if lps else None,
                             mn=min(lps) if lps else None,
                             p10=(float(np.percentile(lps, 10)) if lps else None),
                             ok=int(run_he(pr, code))))
            if (i + 1) % 20 == 0:
                print("  %d/%d 通过 %d" % (i + 1, len(rows), sum(x["ok"] for x in rows))); sys.stdout.flush()
        json.dump(rows, open(os.path.join(OUT, "results_signals2.json"), "w", encoding="utf-8"), indent=1)
        print()
        print("=" * 72)
        print("信号分离度（HumanEval 164 题，3B 单采样）")
        print("=" * 72)
        y = np.array([r["ok"] for r in rows])
        print("  基线 pass@1 = %.3f（错 %d 题）" % (y.mean(), (1 - y).sum()))
        print()
        for sig in ("mean", "mn", "p10"):
            v = np.array([r[sig] for r in rows if r[sig] is not None])
            yy = np.array([r["ok"] for r in rows if r[sig] is not None])
            q1, q3 = np.quantile(v, 0.25), np.quantile(v, 0.75)
            lo, hi = yy[v <= q1], yy[v >= q3]
            print("  %-5s 低四分位通过 %.3f   高四分位通过 %.3f   分离度 %+.3f"
                  % (sig, lo.mean(), hi.mean(), hi.mean() - lo.mean()))
        print()
        print("  各信号在 20%% 升级率下的『错题覆盖率』：")
        for sig in ("mean", "mn", "p10"):
            v = np.array([r[sig] for r in rows if r[sig] is not None])
            yy = np.array([r["ok"] for r in rows if r[sig] is not None])
            thr = np.quantile(v, 0.20)
            esc = v <= thr
            print("    %-5s 升级 %2d 题，其中错题 %2d / 总错题 %d = %.0f%%"
                  % (sig, esc.sum(), ((~yy.astype(bool)) & esc).sum(), (~yy.astype(bool)).sum(),
                     100 * ((~yy.astype(bool)) & esc).sum() / max((~yy.astype(bool)).sum(), 1)))
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
