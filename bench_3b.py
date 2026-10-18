# -*- coding: utf-8 -*-
import time, requests
B="http://127.0.0.1:8091"
print("健康:", requests.get(B+"/health",timeout=10).json())
for p,mx in [("写一段 200 字关于秋天的短文。",300),("用一句话解释什么是梯度下降。",120)]:
    t0=time.perf_counter()
    r=requests.post(B+"/v1/chat/completions",timeout=300,json={"messages":[{"role":"user","content":p}],"max_tokens":mx,"temperature":0})
    dt=time.perf_counter()-t0; u=r.json().get("usage",{}); n=u.get("completion_tokens",0)
    print("  生成 %3d token / %.2f 秒 -> %.1f tok/s" % (n, dt, n/dt if dt else 0))
