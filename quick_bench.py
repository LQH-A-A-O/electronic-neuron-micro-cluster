# -*- coding: utf-8 -*-
"""给一个 gguf 测 tok/s（起服务->测->关）。用法: python quick_bench.py <名字> <路径> [额外参数...]"""
import json, os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8098; BASE = "http://127.0.0.1:%d" % PORT
name, model, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
lf = open(os.path.join(ROOT, "out", "quick_%s.log" % name), "wb")
cmd = [SERVER, "-m", model, "-c", "4096", "-np", "1", "-fa", "on", "--jinja",
       "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
       "--host", "127.0.0.1", "--port", str(PORT), "--alias", name] + extra
p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
try:
    for _ in range(180):
        time.sleep(2)
        try:
            if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                break
        except Exception:
            pass
    else:
        print("加载失败"); sys.exit(1)
    time.sleep(2)
    for prompt, mx in [("用一句话解释什么是梯度下降。", 120), ("写一段 150 字关于秋天的短文。", 220)]:
        t0 = time.perf_counter()
        r = requests.post(BASE + "/v1/chat/completions", timeout=900, json={
            "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": 0})
        dt = time.perf_counter() - t0
        n = r.json().get("usage", {}).get("completion_tokens", 0)
        print("  %-24s %3d token / %6.2f 秒 -> %5.1f tok/s" % (prompt[:22], n, dt, n / dt if dt else 0))
finally:
    p.terminate()
    try: p.wait(timeout=20)
    except Exception: p.kill()
    lf.close(); time.sleep(2)
