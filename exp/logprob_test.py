# -*- coding: utf-8 -*-
"""验证 llama.cpp 是否返回 logprobs（免费置信度的前提）。"""
import json, os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8090; BASE = "http://127.0.0.1:%d" % PORT
lf = open(os.path.join(ROOT, "out", "logprob_test_srv.log"), "wb")
p = subprocess.Popen([SERVER, "-m", r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                      "-c", "4096", "-np", "1", "-fa", "on", "--jinja",
                      "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                      "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                     stdout=lf, stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        time.sleep(2)
        try:
            if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                time.sleep(2); break
        except Exception:
            pass
    for lp in (True, False):
        body = {"messages": [{"role": "user", "content": "用一句话解释梯度下降。"}],
                "max_tokens": 40, "temperature": 0, "logprobs": lp}
        if lp: body["top_logprobs"] = 1
        r = requests.post(BASE + "/v1/chat/completions", timeout=120, json=body).json()
        ch = r["choices"][0]
        has = "logprobs" in ch and ch["logprobs"] is not None
        print("logprobs=%s -> 返回字段 %s" % (lp, "有" if has else "没有"))
        if has:
            lps = ch["logprobs"].get("content") or []
            vals = [x["logprob"] for x in lps if x and x.get("logprob") is not None]
            print("   token 数 %d，logprob 示例 %s" % (len(vals), [round(v,3) for v in vals[:6]]))
            print("   平均 %.4f  最小 %.4f" % (sum(vals)/len(vals), min(vals)))
finally:
    p.terminate()
    try: p.wait(timeout=15)
    except Exception: p.kill()
    lf.close()
