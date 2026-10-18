# -*- coding: utf-8 -*-
import time, requests
B = "http://127.0.0.1:8092"
print("健康:", requests.get(B + "/health", timeout=10).json())
for p, mx in [("用一句话解释什么是梯度下降。", 120),
              ("写一段 150 字关于秋天的短文。", 250),
              ("用 Python 写一个二分查找函数，只给代码。", 200)]:
    try:
        t0 = time.perf_counter()
        r = requests.post(B + "/v1/chat/completions", timeout=600, json={
            "messages": [{"role": "user", "content": p}], "max_tokens": mx, "temperature": 0})
        dt = time.perf_counter() - t0
        d = r.json(); u = d.get("usage", {}); n = u.get("completion_tokens", 0)
        print("  %-26s %3d token / %6.2f 秒 -> %5.1f tok/s" % (p[:24], n, dt, n / dt if dt else 0))
    except Exception as e:
        print("  %-26s 失败 %s" % (p[:24], str(e)[:80]))
