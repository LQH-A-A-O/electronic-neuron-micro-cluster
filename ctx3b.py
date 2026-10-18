# -*- coding: utf-8 -*-
"""3B 的上下文上限实测：8k / 32k / 64k / 128k。"""
import os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
PORT = 8066
FILL = "这是一段用于测试上下文长度的填充文本，不携带需要记住的信息。\n"

def free():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()

for ctx in [8192, 32768, 65536, 131072]:
    lf = open(os.path.join(ROOT, "out", "ctx3b_%d.log" % ctx), "wb")
    p = subprocess.Popen([SERVER, "-m", M, "-c", str(ctx), "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        up = False
        for _ in range(240):
            time.sleep(0.5)
            try:
                if requests.get("http://127.0.0.1:%d/health" % PORT, timeout=2).json().get("status") == "ok":
                    up = True; break
            except Exception:
                pass
            if p.poll() is not None:
                break
        if not up:
            print("  -c %-7d 启动失败（显存不够）  空闲 %s MiB" % (ctx, free())); sys.stdout.flush(); break
        n = int(ctx * 0.7 * 1.5)
        body = "请阅读材料，然后只回答：读完了吗？\n\n" + FILL * (n // len(FILL))
        t0 = time.perf_counter()
        try:
            d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % PORT, timeout=900, json={
                "messages": [{"role": "user", "content": body}], "max_tokens": 32, "temperature": 0}).json()
            if "choices" in d:
                u = d.get("usage", {})
                print("  -c %-7d OK   提示 %6d token  %.0f 秒  空闲 %s MiB"
                      % (ctx, u.get("prompt_tokens", 0), time.perf_counter() - t0, free()))
            else:
                print("  -c %-7d 推理失败 %s" % (ctx, str(d)[:70]))
        except Exception as e:
            print("  -c %-7d 异常 %s" % (ctx, str(e)[:60]))
    finally:
        p.terminate()
        try: p.wait(timeout=25)
        except Exception: p.kill()
        lf.close(); time.sleep(4)
