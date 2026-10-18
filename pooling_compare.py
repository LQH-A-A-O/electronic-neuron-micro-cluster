# -*- coding: utf-8 -*-
"""pooling 方式对比：mean / cls / last，看哪种能给出有区分度的相似度。"""
import json, os, re, statistics, subprocess, sys, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M15 = r"E:\aqca\work\llm\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

DB = [
 ("服务器 A 的部署区域是华东", "华东"),
 ("数据库主库的端口号是 5432", "5432"),
 ("每周三下午三点开周会", "周三下午三点"),
 ("备份保留策略是三十天", "三十天"),
 ("管理员账号叫 ops-admin", "ops-admin"),
 ("缓存层用的是 Redis", "Redis"),
 ("发布窗口在每天凌晨", "凌晨"),
 ("日志保留时长为一周", "一周"),
 ("网关限流是每秒钟一百次", "一百次/秒"),
 ("灾备机房在深圳", "深圳"),
 ("证书有效期为一年", "一年"),
 ("消息队列的积压阈值是十万", "十万"),
 ("监控告警发到运维群", "运维群"),
 ("灰度比例先放百分之五", "5%"),
 ("回滚需要二十分钟", "二十分钟"),
]
QUERIES = [
 ("我们的机器放在哪个地方", 0), ("那个关系型库存数据用哪个端口", 1),
 ("团队多久碰一次头", 2), ("老数据能留多久", 3),
 ("登进系统用什么用户", 4), ("加速读写的中间件是啥", 5),
 ("新版本一般什么时候上线", 6), ("排查记录能看到多久以前", 7),
 ("入口流量每秒最多放过多少", 8), ("万一主中心挂了去哪", 9),
 ("证书多久要换一次", 10), ("堆积多少条就报警", 11),
 ("出问题通知谁", 12), ("先放多少用户进来", 13),
 ("出故障退回上一版要多久", 14),
]

def run(pool, port):
    BASE = "http://127.0.0.1:%d" % port
    lf = open(os.path.join(ROOT, "out", "pool_%s.log" % pool), "wb")
    p = subprocess.Popen([SERVER, "-m", M15, "-c", "4096", "-np", "1", "-fa", "on", "--jinja",
                          "--embeddings", "--pooling", pool,
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            time.sleep(1)
            try:
                if requests.get(BASE + "/health", timeout=2).json().get("status") == "ok":
                    time.sleep(2); break
            except Exception:
                pass
        def emb(ts):
            v = []
            for t in ts:
                d = requests.post(BASE + "/v1/embeddings", timeout=120,
                                  json={"input": t, "model": "x"}).json()
                v.append(d["data"][0]["embedding"])
            a = np.array(v, dtype=np.float32)
            return a / np.linalg.norm(a, axis=1, keepdims=True)
        K = emb([k for k, _ in DB]); Q = emb([q for q, _ in QUERIES])
        S = Q @ K.T
        hit = 0; sep = []
        for i, (q, gi) in enumerate(QUERIES):
            j = int(np.argmax(S[i])); hit += int(j == gi)
            right = float(S[i][gi]); other = max(float(x) for t, x in enumerate(S[i]) if t != gi)
            sep.append(right - other)
        return dict(pool=pool, hit=hit, n=len(QUERIES),
                    mean_right=float(np.mean([S[i][gi] for i, (_, gi) in enumerate(QUERIES)])),
                    mean_top=float(np.mean(S.max(axis=1))),
                    margin=float(np.mean(sep)))
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(4)

if __name__ == "__main__":
    for pool, port in (("mean", 8047), ("cls", 8046), ("last", 8045)):
        r = run(pool, port)
        print("  pooling=%-5s 召回 %2d/%d = %.2f   正确项均分 %.4f   最大项均分 %.4f   平均领先 %.4f"
              % (r["pool"], r["hit"], r["n"], r["hit"] / r["n"], r["mean_right"], r["mean_top"], r["margin"]))
        sys.stdout.flush()
