# -*- coding: utf-8 -*-
"""端到端：向量召回记忆 -> 注入提示词 -> 3B 回答（低置信升级到 9B）。

两组对照：
    A  无记忆（裸模型）
    B  有记忆（向量召回 top-3 注入）
查询全是"字面无重叠"的改写，考的是检索 + 利用。
"""
import json, os, re, statistics, subprocess, sys, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"
MB = r"E:\aqca\work\llm\models\bge-small-zh-v1.5-q8_0.gguf"
P3, P9, PB = 8042, 8041, 8040
FENCE = chr(96) * 3
ESC_FRAC = 0.2

FACTS = [("服务器 A 的部署区域是华东","华东"),("数据库主库的端口号是 5432","5432"),
 ("每周三下午三点开周会","周三下午三点"),("备份保留策略是三十天","三十天"),
 ("管理员账号叫 ops-admin","ops-admin"),("缓存层用的是 Redis","Redis"),
 ("发布窗口在每天凌晨","凌晨"),("日志保留时长为一周","一周"),
 ("网关限流是每秒钟一百次","一百次/秒"),("灾备机房在深圳","深圳"),
 ("证书有效期为一年","一年"),("消息队列的积压阈值是十万","十万"),
 ("监控告警发到运维群","运维群"),("灰度比例先放百分之五","5%"),
 ("回滚需要二十分钟","二十分钟")]
QUERIES = [("我们的机器放在哪个地方","华东"),("那个关系型库存数据用哪个端口","5432"),
 ("团队多久碰一次头","周三"),("老数据能留多久","三十天"),
 ("登进系统用什么用户","ops-admin"),("加速读写的中间件是啥","Redis"),
 ("新版本一般什么时候上线","凌晨"),("排查记录能看到多久以前","一周"),
 ("入口流量每秒最多放过多少","一百次"),("万一主中心挂了去哪","深圳"),
 ("证书多久要换一次","一年"),("堆积多少条就报警","十万"),
 ("出问题通知谁","运维群"),("先放多少用户进来","5"),("出故障退回上一版要多久","二十")]

def launch(model, port, extra=None):
    lf = open(os.path.join(ROOT,"out","e2v_%d.log"%port),"wb")
    cmd = [SERVER,"-m",model,"-c","4096","-np","1","-fa","on","--jinja",
           "--chat-template-kwargs",'{"enable_thinking":false}',"--temp","0",
           "--host","127.0.0.1","--port",str(port),"-ngl","99"] + (extra or [])
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(240):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:%d/health"%port,timeout=2).json().get("status")=="ok":
                time.sleep(1); return p, lf
        except Exception: pass
    p.terminate(); lf.close(); raise SystemExit("load fail %d"%port)

def chat(port, msg, lp=False, mx=200):
    b={"messages":[{"role":"user","content":msg}],"max_tokens":mx,"temperature":0}
    if lp: b["logprobs"]=True; b["top_logprobs"]=0
    d=requests.post("http://127.0.0.1:%d/v1/chat/completions"%port,timeout=600,json=b).json()
    ch=d["choices"][0]; v=None
    if lp:
        vals=[x["logprob"] for x in ((ch.get("logprobs") or {}).get("content") or []) if x and x.get("logprob") is not None]
        v=sum(vals)/len(vals) if vals else None
    return ch["message"]["content"] or "", v

def emb(ts):
    v=[requests.post("http://127.0.0.1:%d/v1/embeddings"%PB,timeout=60,json={"input":t,"model":"x"}).json()["data"][0]["embedding"] for t in ts]
    a=np.array(v,dtype=np.float32); n=np.linalg.norm(a,axis=1,keepdims=True); n[n==0]=1
    return a/n

if __name__ == "__main__":
    p3,lf3=launch(M3,P3); p9,lf9=launch(M9,P9); pb,lb=launch(MB,PB,["--embeddings","--pooling","cls"])
    try:
        keys=[k for k,_ in FACTS]
        K=emb(keys)
        Q=emb([q for q,_ in QUERIES])
        S=Q@K.T
        print("三个服务常驻就绪（3B / 9B / bge）"); sys.stdout.flush()
        hist={"code":[], "text":[]}
        rows=[]
        for i,(q,gold) in enumerate(QUERIES):
            ctx_keys=[keys[j] for j in np.argsort(-S[i])[:3]]
            ctx="\n".join("  - "+k for k in ctx_keys)
            t0=time.perf_counter()
            a_txt,a_lp=chat(P3,q,lp=True)
            b_txt,b_lp=chat(P3,"已知下列事实：\n%s\n\n问：%s\n只回答要点，不要解释。"%(ctx,q),lp=True)
            ok_a=int(gold in a_txt); ok_b=int(gold in b_txt)
            rows.append(dict(q=q,gold=gold,ctx=ctx_keys,ok_a=ok_a,ok_b=ok_b,a=a_txt[:80],b=b_txt[:80]))
            print("  %-16s 无记忆 %s  有记忆 %s  (召回项 %s)"
                  %(q[:16], "对" if ok_a else "错", "对" if ok_b else "错", ctx_keys[0][:14]))
            sys.stdout.flush()
        n=len(rows)
        A=sum(r["ok_a"] for r in rows); B=sum(r["ok_b"] for r in rows)
        print()
        print("="*66)
        print("端到端：向量召回 + 记忆注入（%d 条改写查询）"%n)
        print("="*66)
        print("  A 无记忆        准确 %d/%d = %.2f"%(A,n,A/n))
        print("  B 有记忆        准确 %d/%d = %.2f   (相对 A %+.2f)"%(B,n,B/n,(B-A)/n))
        json.dump(rows,open(os.path.join(ROOT,"out","results_e2v.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
    finally:
        for p,lf in ((p3,lf3),(p9,lf9),(pb,lb)):
            p.terminate()
            try: p.wait(timeout=15)
            except Exception: p.kill()
            lf.close()
