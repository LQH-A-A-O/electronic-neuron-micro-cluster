# -*- coding: utf-8 -*-
"""跑一道 HumanEval，把模型的原始输出和失败原因都打出来。"""
import gzip, json, os, subprocess, sys, tarfile, tempfile, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8096; BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3
with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
    n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
    he = [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]
lf = open(os.path.join(ROOT, "out", "debug_he_srv.log"), "wb")
cmd = [SERVER, "-m", r"E:\aqca\work\llm\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", "-c", "4096",
       "-np", "1", "-fa", "on", "--jinja", "--temp", "0", "--host", "127.0.0.1", "--port", str(PORT)]
p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
try:
    for _ in range(90):
        time.sleep(2)
        try:
            if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok": break
        except Exception: pass
    time.sleep(2)
    pr = he[0]
    for tag, stop in [("有 stop", ["\n" + FENCE]), ("无 stop", None)]:
        body = {"messages": [{"role": "user", "content": pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"}],
                "max_tokens": 400, "temperature": 0}
        if stop: body["stop"] = stop
        out = requests.post(BASE + "/v1/chat/completions", timeout=300, json=body).json()["choices"][0]["message"]["content"]
        print("=== %s 原始输出 ===" % tag); print(repr(out[:500])); print()
        c = out
        import re
        m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, out, re.S)
        if m: c = m[0]
        if pr["prompt"].strip().split("\n")[0] not in c:
            c = pr["prompt"] + c
        prog = c + "\n\n" + pr["test"] + "\n\ncheck(%s)\n" % pr["entry_point"]
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(prog); path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=20)
        print("%s -> 退出码 %s" % (tag, r.returncode))
        if r.returncode != 0:
            print("  stderr 尾:", r.stderr.decode("utf-8", "replace").strip().split("\n")[-1][:160])
        os.unlink(path)
finally:
    p.terminate()
    try: p.wait(timeout=15)
    except Exception: p.kill()
    lf.close()
