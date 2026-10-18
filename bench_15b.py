# -*- coding: utf-8 -*-
"""1.5B 服务：健康检查 + 生成速度实测。"""
import json, time
import requests
BASE = "http://127.0.0.1:8090"
print("健康:", requests.get(BASE + "/health", timeout=10).json())
for prompt, mx in [("写一段 200 字关于秋天的短文。", 300),
                   ("用一句话解释什么是梯度下降。", 120)]:
    t0 = time.perf_counter()
    r = requests.post(BASE + "/v1/chat/completions", timeout=300, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mx, "temperature": 0})
    dt = time.perf_counter() - t0
    d = r.json()
    u = d.get("usage", {})
    n = u.get("completion_tokens", 0)
    print("  提示 %-22s 生成 %3d token / %.2f 秒  ->  %.1f tok/s"
          % (prompt[:20], n, dt, n / dt if dt else 0))
    print("      回复:", (d["choices"][0]["message"]["content"] or "")[:80].replace("\n", " "))
