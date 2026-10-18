# -*- coding: utf-8 -*-
"""logprob 置信度能否分开"答对"与"答错"？—— 免费置信度级联的关键验证。

单次推理就带 logprobs，所以这个判据是**免费的**（不像多采样要 ×K）。
"""
import gzip, json, os, re, subprocess, sys, tarfile, tempfile, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8089; BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3
MID = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
CACHE = os.path.join(OUT, "logprob_conf.json")


def load_he():
    with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
        n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
        return [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]


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


if __name__ == "__main__":
    he = load_he()
    lf = open(os.path.join(OUT, "logprob_conf_srv.log"), "wb")
    p = subprocess.Popen([SERVER, "-m", MID, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
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
        t0 = time.perf_counter()
        for i, pr in enumerate(he):
            try:
                r = requests.post(BASE + "/v1/chat/completions", timeout=600, json={
                    "messages": [{"role": "user", "content": pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"}],
                    "max_tokens": 450, "temperature": 0, "logprobs": True, "top_logprobs": 0}).json()
                ch = r["choices"][0]
                code = strip_fence(ch["message"]["content"] or "")
                lps = [x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or [])
                       if x and x.get("logprob") is not None]
                rows.append(dict(idx=i, mean_lp=(sum(lps) / len(lps)) if lps else None,
                                 min_lp=min(lps) if lps else None, n_lp=len(lps),
                                 ok=int(run_he(pr, code))))
            except Exception:
                rows.append(dict(idx=i, mean_lp=None, min_lp=None, n_lp=0, ok=0))
            if (i + 1) % 20 == 0:
                good = sum(x["ok"] for x in rows)
                print("  %d/%d  通过 %d (%.1f%%)" % (i + 1, len(rows), good, 100 * good / (i + 1)))
                sys.stdout.flush()
        json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        print("完成 %.0f 秒" % (time.perf_counter() - t0))
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
