# -*- coding: utf-8 -*-
"""注入方式对比：打"利用"瓶颈。"""
import json, os, subprocess, sys, time
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MB = r"E:\aqca\work\llm\models\bge-small-zh-v1.5-q8_0.gguf"
P3, PB = 8039, 8038
FACTS = [('服务器 A 的部署区域是华东','华东'),('数据库主库的端口号是 5432','5432'),
 ('每周三下午三点开周会','周三'),('备份保留策略是三十天','三十天'),
 ('管理员账号叫 ops-admin','ops-admin'),('缓存层用的是 Redis','Redis'),
 ('发布窗口在每天凌晨','凌晨'),('日志保留时长为一周','一周'),
 ('网关限流是每秒钟一百次','一百次'),('灾备机房在深圳','深圳'),
 ('证书有效期为一年','一年'),('消息队列的积压阈值是十万','十万'),
 ('监控告警发到运维群','运维群'),('灰度比例先放百分之五','百分之五'),('回滚需要二十分钟','二十分钟')]
QUERIES = [('我们的机器放在哪个地方','华东'),('那个关系型库存数据用哪个端口','5432'),
 ('团队多久碰一次头','周三'),('老数据能留多久','三十天'),('登进系统用什么用户','ops-admin'),
 ('加速读写的中间件是啥','Redis'),('新版本一般什么时候上线','凌晨'),('排查记录能看到多久以前','一周'),
 ('入口流量每秒最多放过多少','一百次'),('万一主中心挂了去哪','深圳'),('证书多久要换一次','一年'),
 ('堆积多少条就报警','十万'),('出问题通知谁','运维群'),('先放多少用户进来','百分之五'),
 ('出故障退回上一版要多久','二十分钟')]
def launch(model, port, extra=None):
    lf = open(os.path.join(ROOT, "out", "u_%d.log" % port), "wb")
    cmd = [SERVER, "-m", model, "-c", "4096", "-np", "1", "-fa", "on", "--jinja",
           "--chat-template-kwargs", chr(123) + chr(34) + "enable_thinking" + chr(34) + ":false" + chr(125), "--temp", "0",
           "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"] + (extra or [])
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(240):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                time.sleep(1); return p, lf
        except Exception: pass
    p.terminate(); lf.close(); raise SystemExit("load fail")
def chat(port, msg, mx=120):
    d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600,
                      json={"messages": [{"role": "user", "content": msg}], "max_tokens": mx, "temperature": 0}).json()
    return d["choices"][0]["message"]["content"] or ""
def emb(ts):
    v = [requests.post("http://127.0.0.1:%d/v1/embeddings" % PB, timeout=60, json={"input": t, "model": "x"}).json()["data"][0]["embedding"] for t in ts]
    a = np.array(v, dtype=np.float32); n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
PUNCT = ' ' + chr(9) + chr(10) + '，。、；：！？（）()[]【】「」' + chr(34) + chr(39) + '~!@#$%^&*+=|/<>-'
def norm(t):
    t = t.lower()
    for c in PUNCT: t = t.replace(c, '')
    t = t.replace('百分之', '').replace('一百', '100').replace('十', '10')
    for k, v in CN.items(): t = t.replace(k, v)
    return t
def hit(g, t):
    return bool(norm(g)) and norm(g) in norm(t)
if __name__ == '__main__':
    p3, lf3 = launch(M3, P3); pb, lb = launch(MB, PB, ['--embeddings', '--pooling', 'cls'])
    try:
        keys = [k for k, _ in FACTS]; K = emb(keys)
        Q = emb([q for q, _ in QUERIES]); S = Q @ K.T
        rows = []
        for i, (q, gold) in enumerate(QUERIES):
            order = np.argsort(-S[i]); ctx3 = [keys[j] for j in order[:3]]; ctx1 = keys[order[0]]
            a = chat(P3, q)
            b = chat(P3, '已知下列事实：' + chr(10) + chr(10).join('  - ' + k for k in ctx3) + chr(10) + chr(10) + '问：' + q + chr(10) + '只回答要点。')
            c = chat(P3, '事实：' + ctx1 + chr(10) + chr(10) + '问：' + q + chr(10) + '只允许使用上面这一条事实回答。如果它没有直接答案，就回答「不知道」。只输出答案本身。')
            d = chat(P3, '已知下列事实（编号 1..3）：' + chr(10) + chr(10).join('  %d. %s' % (j+1, k) for j, k in enumerate(ctx3)) + chr(10) + chr(10) + '问：' + q + chr(10) + '只能从上面 3 条里选一条作为依据，直接给出答案，不要解释。')
            r = dict(q=q, gold=gold, A=int(hit(gold, a)), B=int(hit(gold, b)), C=int(hit(gold, c)), D=int(hit(gold, d)),
                     a=a[:60], b=b[:60], c=c[:60], d=d[:60])
            rows.append(r)
            print('  %-16s A%d B%d C%d D%d' % (q[:16], r['A'], r['B'], r['C'], r['D'])); sys.stdout.flush()
        n = len(rows)
        print(); print('=' * 62); print('注入方式对比（%d 条改写查询）' % n); print('=' * 62)
        for k, nm in (('A','A 无记忆'),('B','B top-3 裸注入'),('C','C top-1 + 强约束'),('D','D top-3 + 约束 + 编号')):
            s = sum(r[k] for r in rows)
            print('  %-22s %2d/%d = %.2f' % (nm, s, n, s / n))
        json.dump(rows, open(os.path.join(ROOT, "out", "results_util.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        for p, lf in ((p3, lf3), (pb, lb)):
            p.terminate()
            try: p.wait(timeout=15)
            except Exception: p.kill()
            lf.close()