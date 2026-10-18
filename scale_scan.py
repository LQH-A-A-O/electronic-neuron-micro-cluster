# -*- coding: utf-8 -*-
"""知识规模扫描：100 / 1000 / 10000 条事实，看召回率与延迟怎么退化。"""
import json, os, subprocess, sys, time
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
MB = r"E:\aqca\work\llm\models\bge-small-zh-v1.5-q8_0.gguf"
PB = 8037
rnd = __import__('random').Random(20261018)
TOPICS = ['缓存','并发','日志','加密','索引','重试','限流','快照','回滚','灰度','埋点','熔断',
          '分片','副本','幂等','超时','序列化','压缩','校验','迁移','队列','批处理','沙箱','证书']
ATTR = ['超时时间','重试次数','并发上限','内存配额','保留天数','版本号','端口','域名','负责人','告警阈值']
def gen(n):
    rows = []
    for i in range(n):
        t = TOPICS[i % len(TOPICS)]; a = ATTR[(i // len(TOPICS)) % len(ATTR)]
        # key 唯一：带上编号，避免大量重复
        v = '%s-%d 的%s是 %d' % (t, i, a, 100 + (i * 37) % 9000)
        rows.append((v, str(100 + (i * 37) % 9000)))
    return rows
def launch(model, port, extra=None):
    lf = open(os.path.join(ROOT, "out", "scale_%d.log" % port), "wb")
    cmd = [SERVER, "-m", model, "-c", "1024", "-np", "1", "-fa", "on", "--embeddings",
           "--pooling", "cls", "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"]
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(240):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                time.sleep(1); return p, lf
        except Exception: pass
    p.terminate(); lf.close(); raise SystemExit("load fail")
def emb(ts):
    v = [requests.post("http://127.0.0.1:%d/v1/embeddings" % PB, timeout=300, json={"input": t, "model": "x"}).json()["data"][0]["embedding"] for t in ts]
    a = np.array(v, dtype=np.float32); n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
if __name__ == '__main__':
    p, lf = launch(MB, PB)
    try:
        res = []
        for n in (100, 1000, 10000):
            facts = gen(n)
            keys = [k for k, _ in facts]
            t0 = time.time(); K = emb(keys); build = time.time() - t0
            # 20 条查询：取已知事实，改写成问法（保留少量关键词，模拟真实）
            qs = []
            for j in rnd.sample(range(n), min(20, n)):
                k, v = facts[j]
                part = k.split(' ')[0] + ' ' + k.split(' ')[1]
                qs.append((part + ' 是多少？', v, k))
            t1 = time.time(); Q = emb([q for q, _, _ in qs]); qtime = time.time() - t1
            S = Q @ K.T
            hit = 0; lat = []
            for i, (q, v, k) in enumerate(qs):
                ta = time.time()
                j = int(np.argmax(S[i]))
                lat.append(time.time() - ta)
                hit += int(keys[j] == k)
            res.append(dict(n=n, hit=hit, total=len(qs), build_s=round(build, 1),
                            q_emb_s=round(qtime, 2), recall_ms=round(float(np.mean(lat)) * 1000, 3)))
            print('  %5d 条事实  召回 %2d/%d = %.2f   建索引 %.1f 秒   查询编码 %.2f 秒   检索 %.3f ms'
                  % (n, hit, len(qs), hit / len(qs), build, qtime, np.mean(lat) * 1000))
            sys.stdout.flush()
        json.dump(res, open(os.path.join(ROOT, "out", "results_scale.json"), "w"), indent=1)
    finally:
        p.terminate()
        try: p.wait(timeout=15)
        except Exception: p.kill()
        lf.close()