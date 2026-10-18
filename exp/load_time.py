# -*- coding: utf-8 -*-
"""测模型加载时间 —— 这决定级联到底划不划算。

3B 2.5GB + 9B 5.7GB = 8.2GB > 8GB 显存，两个装不下，所以每次升级要换模型。
如果换模型要几十秒，级联省下的 7.6 秒就被吃光了。
"""
import os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
MODELS = [("3B", r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf", 8079),
          ("9B", r"E:\aqca\work\llm\models\Qwen3.5-9B-Q4_K_M.gguf", 8078)]

for name, model, port in MODELS:
    base = "http://127.0.0.1:%d" % port
    lf = open(os.path.join(ROOT, "out", "load_%s.log" % name), "wb")
    t0 = time.perf_counter()
    p = subprocess.Popen([SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        ok = False
        while time.perf_counter() - t0 < 300:
            time.sleep(0.5)
            try:
                if requests.get(base + "/health", timeout=2).json().get("status") == "ok":
                    ok = True; break
            except Exception:
                pass
        dt = time.perf_counter() - t0
        # 第一 token 还要额外预热
        t1 = time.perf_counter()
        requests.post(base + "/v1/chat/completions", timeout=120,
                      json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0})
        warm = time.perf_counter() - t1
        print("  %-3s 加载到可用 %.1f 秒   首次推理 %.1f 秒   合计 %.1f 秒" % (name, dt, warm, dt + warm))
        sys.stdout.flush()
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
        time.sleep(3)
