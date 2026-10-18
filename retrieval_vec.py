# -*- coding: utf-8 -*-
"""真向量检索验证：精确 / n-gram / embedding 近邻，三路对比。

关键：查询集必须"语义能连、字面无重叠"，否则测不出东西（前两次都栽在这）。
"""
import json, os, re, statistics, subprocess, sys, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M15 = r"E:\aqca\work\llm\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
PORT = 8048; BASE = "http://127.0.0.1:%d" % PORT

# 神经元库：每条形如 (key, value)。key 是自然语言描述，query 与它字面几乎不重叠。
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
# 改写查询：和 key 语义相关，但字面几乎不重合
QUERIES = [
 ("我们的机器放在哪个地方", 0),
 ("那个关系型库存数据用哪个端口", 1),
 ("团队多久碰一次头", 2),
 ("老数据能留多久", 3),
 ("登进系统用什么用户", 4),
 ("加速读写的中间件是啥", 5),
 ("新版本一般什么时候上线", 6),
 ("排查记录能看到多久以前", 7),
 ("入口流量每秒最多放过多少", 8),
 ("万一主中心挂了去哪", 9),
 ("证书多久要换一次", 10),
 ("堆积多少条就报警", 11),
 ("出问题通知谁", 12),
 ("先放多少用户进来", 13),
 ("出故障退回上一版要多久", 14),
]

def grams(s, n=2):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}

def jacc(a, b):
    A, B = grams(a), grams(b)
    return len(A & B) / len(A | B) if A | B else 0.0

def emb(texts):
    out = []
    for t in texts:
        r = requests.post(BASE + "/v1/embeddings", timeout=120, json={"input": t, "model": "x"})
        d = r.json()
        if "data" not in d:
            raise RuntimeError(str(d)[:200])
        out.append(d["data"][0]["embedding"])
    return np.array(out, dtype=np.float32)

if __name__ == "__main__":
    lf = open(os.path.join(ROOT, "out", "embed_srv.log"), "wb")
    p = subprocess.Popen([SERVER, "-m", M15, "-c", "4096", "-np", "1", "-fa", "on", "--jinja",
                          "--embeddings", "--pooling", "mean",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            time.sleep(1)
            try:
                if requests.get(BASE + "/health", timeout=2).json().get("status") == "ok":
                    time.sleep(2); break
            except Exception:
                pass
        print("服务就绪（--embeddings）")
        keys = [k for k, _ in DB]
        K = emb(keys)
        Qraw = emb([q for q, _ in QUERIES])
        print("库 %d 条，查询 %d 条，向量维度 %d" % (len(keys), len(QUERIES), K.shape[1]))
        # 归一化
        Kn = K / np.linalg.norm(K, axis=1, keepdims=True)
        Qn = Qraw / np.linalg.norm(Qraw, axis=1, keepdims=True)
        S = Qn @ Kn.T
        hit_e = hit_n = hit_v = 0
        rows = []
        for i, (q, gi) in enumerate(QUERIES):
            # 精确：整个 key 是否出现在 query 里（几乎必然失败）
            e = next((j for j, k in enumerate(keys) if k in q), None)
            # n-gram
            sims = [jacc(q, k) for k in keys]
            nj = int(np.argmax(sims)); nv = max(sims)
            n = nj if nv >= 0.45 else None
            # 向量
            vj = int(np.argmax(S[i])); vv = float(S[i][vj])
            v = vj if vv >= 0.35 else None
            hit_e += int(e == gi); hit_n += int(n == gi); hit_v += int(v == gi)
            rows.append(dict(q=q, gold=gi, exact=e, ngram=n, vec=v,
                             ng_score=round(nv, 3), vec_score=round(vv, 3),
                             ok_e=int(e == gi), ok_n=int(n == gi), ok_v=int(v == gi)))
        N = len(QUERIES)
        print()
        print("=" * 66)
        print("字面无重叠的改写查询上的召回")
        print("=" * 66)
        print("  精确匹配      %.2f  (%d/%d)" % (hit_e / N, hit_e, N))
        print("  字符 n-gram   %.2f  (%d/%d)" % (hit_n / N, hit_n, N))
        print("  **向量近邻**  %.2f  (%d/%d)" % (hit_v / N, hit_v, N))
        print()
        print("  向量相似度：命中时均值 %.3f，未命中时均值 %.3f"
              % (statistics.mean([r["vec_score"] for r in rows if r["ok_v"]] or [0]),
                 statistics.mean([r["vec_score"] for r in rows if not r["ok_v"]] or [0])))
        json.dump(rows, open(os.path.join(ROOT, "out", "results_retrieval_vec.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
