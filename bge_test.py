# -*- coding: utf-8 -*-
"""用专用检索模型 bge-small-zh-v1.5 重跑同一个测试集。"""
import json, os, subprocess, sys, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M = r"E:\aqca\work\llm\models\bge-small-zh-v1.5-q8_0.gguf"
PORT = 8044; BASE = "http://127.0.0.1:%d" % PORT
DB = [("服务器 A 的部署区域是华东","华东"),("数据库主库的端口号是 5432","5432"),
 ("每周三下午三点开周会","周三下午三点"),("备份保留策略是三十天","三十天"),
 ("管理员账号叫 ops-admin","ops-admin"),("缓存层用的是 Redis","Redis"),
 ("发布窗口在每天凌晨","凌晨"),("日志保留时长为一周","一周"),
 ("网关限流是每秒钟一百次","一百次/秒"),("灾备机房在深圳","深圳"),
 ("证书有效期为一年","一年"),("消息队列的积压阈值是十万","十万"),
 ("监控告警发到运维群","运维群"),("灰度比例先放百分之五","5%"),
 ("回滚需要二十分钟","二十分钟")]
QUERIES = [("我们的机器放在哪个地方",0),("那个关系型库存数据用哪个端口",1),
 ("团队多久碰一次头",2),("老数据能留多久",3),("登进系统用什么用户",4),
 ("加速读写的中间件是啥",5),("新版本一般什么时候上线",6),("排查记录能看到多久以前",7),
 ("入口流量每秒最多放过多少",8),("万一主中心挂了去哪",9),("证书多久要换一次",10),
 ("堆积多少条就报警",11),("出问题通知谁",12),("先放多少用户进来",13),
 ("出故障退回上一版要多久",14)]
for pool in ("cls", "mean"):
    lf = open(os.path.join(ROOT,"out","bge_%s.log"%pool),"wb")
    p = subprocess.Popen([SERVER,"-m",M,"-c","1024","-np","1","-fa","on","--embeddings",
                          "--pooling",pool,"--host","127.0.0.1","--port",str(PORT),"-ngl","99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        up=False
        for _ in range(120):
            time.sleep(1)
            try:
                if requests.get(BASE+"/health",timeout=2).json().get("status")=="ok":
                    time.sleep(1); up=True; break
            except Exception: pass
        if not up:
            print("  pooling=%-4s 启动失败" % pool); continue
        def emb(ts):
            v=[requests.post(BASE+"/v1/embeddings",timeout=60,json={"input":t,"model":"x"}).json()["data"][0]["embedding"] for t in ts]
            a=np.array(v,dtype=np.float32); return a/np.linalg.norm(a,axis=1,keepdims=True)
        K=emb([k for k,_ in DB]); Q=emb([q for q,_ in QUERIES]); S=Q@K.T
        hit=0; margins=[]
        for i,(q,gi) in enumerate(QUERIES):
            j=int(np.argmax(S[i])); hit+=int(j==gi)
            margins.append(float(S[i][gi])-max(float(x) for t,x in enumerate(S[i]) if t!=gi))
        print("  bge pooling=%-4s 召回 %2d/%d = %.2f   正确项均分 %.4f   平均领先 %+.4f"
              % (pool,hit,len(QUERIES),hit/len(QUERIES),
                 float(np.mean([S[i][gi] for i,(_,gi) in enumerate(QUERIES)])), float(np.mean(margins))))
    finally:
        p.terminate()
        try: p.wait(timeout=15)
        except Exception: p.kill()
        lf.close(); time.sleep(3)
