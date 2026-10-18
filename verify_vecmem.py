# -*- coding: utf-8 -*-
"""验证：GatedMemory + 向量检索，能不能召回改写查询。"""
import os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import encluster as E

SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M = r"E:\aqca\work\llm\models\bge-small-zh-v1.5-q8_0.gguf"
PORT = 8043; BASE = "http://127.0.0.1:%d" % PORT

FACTS = [("服务器 A 的部署区域是华东","华东"),("数据库主库的端口号是 5432","5432"),
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

if __name__ == "__main__":
    lf = open(os.path.join(ROOT,"out","vec_mem.log"),"wb")
    p = subprocess.Popen([SERVER,"-m",M,"-c","1024","-np","1","-fa","on","--embeddings",
                          "--pooling","cls","--host","127.0.0.1","--port",str(PORT),"-ngl","99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            time.sleep(1)
            try:
                if requests.get(BASE+"/health",timeout=2).json().get("status")=="ok":
                    time.sleep(1); break
            except Exception: pass
        mem = E.GatedMemory()
        for k,v in FACTS:
            mem.add(k,v)
        emb = E.Embedder(BASE)
        n = mem.build_index(emb)
        print("入库 %d 个 key，建索引 %d 条" % (len(mem.store), n))
        print()
        hit_e = hit_v = 0
        for q, gi in QUERIES:
            gold = FACTS[gi][0]
            e = mem.recall(q, 1)                      # 精确
            ok_e = any(k == q for k,_ in e)
            v = mem.recall_semantic(q, k=3)           # 向量
            ok_v = any(k == gold for k,_ in v)
            hit_e += int(ok_e); hit_v += int(ok_v)
            top = v[0][0] if v else "（无）"
            print("  %-18s 正确项=%-18s 向量top-1=%-18s %s" % (q[:18], gold[:18], top[:18], "OK" if ok_v else "MISS"))
        print()
        print("  精确匹配召回 %.2f (%d/%d)" % (hit_e/len(QUERIES), hit_e, len(QUERIES)))
        print("  向量检索召回 %.2f (%d/%d)" % (hit_v/len(QUERIES), hit_v, len(QUERIES)))
    finally:
        p.terminate()
        try: p.wait(timeout=15)
        except Exception: p.kill()
        lf.close()
