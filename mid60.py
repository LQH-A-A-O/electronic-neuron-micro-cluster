# -*- coding: utf-8 -*-
"""补测 3B 在全部 60 题上的单独成绩（级联里它只跑过 45 道升级题）。"""
import json, os, re, subprocess, sys, tarfile, gzip, tempfile, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8092; BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3
MID = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"

with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
    n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
    he = [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]
rows = json.load(open(os.path.join(OUT, "ladder_samples.json"), encoding="utf-8"))


def strip_fence(txt):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, txt, re.S)
    return m[0] if m else re.sub(r"^\s*" + FENCE + r"(?:python)?\s*\n", "", txt)


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


lf = open(os.path.join(OUT, "mid60_srv.log"), "wb")
p = subprocess.Popen([SERVER, "-m", MID, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                      "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                      "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                     stdout=lf, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        time.sleep(2)
        try:
            if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                time.sleep(2); break
        except Exception:
            pass
    ok = 0
    for j, r in enumerate(rows):
        if r.get("mid"):
            good = run_he(he[r["idx"]], r["mid"])
        else:
            out = requests.post(BASE + "/v1/chat/completions", timeout=600, json={
                "messages": [{"role": "user", "content": he[r["idx"]]["prompt"] + "\n\n只补全这个函数体，不要任何解释。"}],
                "max_tokens": 450, "temperature": 0}).json()["choices"][0]["message"]["content"] or ""
            r["mid"] = strip_fence(out)
            good = run_he(he[r["idx"]], r["mid"])
        ok += int(good)
        if (j + 1) % 15 == 0:
            print("  %d/%d  3B 通过 %d (%.1f%%)" % (j + 1, len(rows), ok, 100 * ok / (j + 1)))
            sys.stdout.flush()
    print("3B 单独在这 60 题上 pass@1 = %.3f (%d/%d)" % (ok / len(rows), ok, len(rows)))
    json.dump(rows, open(os.path.join(OUT, "ladder_samples.json"), "w", encoding="utf-8"), ensure_ascii=False)
finally:
    p.terminate()
    try: p.wait(timeout=20)
    except Exception: p.kill()
    lf.close()
