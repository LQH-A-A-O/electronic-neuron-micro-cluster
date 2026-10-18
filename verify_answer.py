# -*- coding: utf-8 -*-
"""答案可验证性检查：回答必须能在召回文本里找到依据，否则弃答。

三组对照（15 条改写查询）：
  A  裸注入（基线 0.73）
  B  裸注入 + 可验证性检查（找不到依据 -> 不知道）
  C  裸注入 + 可验证性检查 + 一次重试（更严格的提示）
"""
import json, os, re, subprocess, sys, time
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MB = r"E:\aqca\work\llm\models\bge-small-zh-v1.5-q8_0.gguf"
P3, PB = 8036, 8035
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
CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
KEEP = set('0123456789abcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千萬万零两')
def norm(t):
    t = t.lower()
    t = t.replace('百分之', '').replace('一百', '100').replace('十', '10')
    for k, v in CN.items(): t = t.replace(k, v)
    out = []
    for c in t:
        if c in KEEP: out.append(c)
    return ''.join(out)
def hit(g, t):
    return bool(norm(g)) and norm(g) in norm(t)
def has_support(ans, ctx):
    """回答里的数字/英文实体，是否都能在召回文本里找到。"""
    a = norm(ans); c = norm(ctx)
    if not a: return False
    # 抽取回答里的数字串和英文串
    toks = re.findall(r'[0-9]{1,6}', a) + re.findall(r'[a-z][a-z0-9-]{1,20}', a)
    if not toks: return True   # 纯中文回答，无法逐 token 校验，放过
    return all(t in c for t in toks)
def launch(model, port, extra=None):
    lf = open(os.path.join(ROOT, "out", "ver_%d.log" % port), "wb")
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
def chat(port, msg, mx=110):
    d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600,
                      json={"messages": [{"role": "user", "content": msg}], "max_tokens": mx, "temperature": 0}).json()
    return d["choices"][0]["message"]["content"] or ""
def emb(ts):
    v = [requests.post("http://127.0.0.1:%d/v1/embeddings" % PB, timeout=60, json={"input": t, "model": "x"}).json()["data"][0]["embedding"] for t in ts]
    a = np.array(v, dtype=np.float32); n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
if __name__ == '__main__':
    p3, lf3 = launch(M3, P3); pb, lb = launch(MB, PB, ['--embeddings', '--pooling', 'cls'])
    try:
        keys = [k for k, _ in FACTS]; K = emb(keys)
        Q = emb([q for q, _ in QUERIES]); S = Q @ K.T
        rows = []
        for i, (q, gold) in enumerate(QUERIES):
            order = np.argsort(-S[i]); ctx3 = [keys[j] for j in order[:3]]
            ctx = chr(10).join('  - ' + k for k in ctx3)
            base = '已知下列事实：' + chr(10) + ctx + chr(10) + chr(10) + '问：' + q + chr(10) + '只回答要点。'
            a = chat(P3, base)
            okA = int(hit(gold, a))
            # B: 可验证性检查
            b = a if has_support(a, ctx) else '不知道'
            okB = int(hit(gold, b))
            # C: 检查不过就重试一次（更严格）
            c = a
            if not has_support(a, ctx):
                c = chat(P3, '事实：' + chr(10) + ctx + chr(10) + chr(10) + '问：' + q + chr(10) + '只能从上面的事实里摘取答案，不许推理或补充。只输出答案本身，不超过 10 个字。')
                if not has_support(c, ctx): c = '不知道'
            okC = int(hit(gold, c))
            rows.append(dict(q=q, gold=gold, A=okA, B=okB, C=okC, a=a[:70], b=b[:40], c=c[:40]))
            print('  %-16s A%d B%d C%d  |%s|' % (q[:16], okA, okB, okC, a[:26])); sys.stdout.flush()
        n = len(rows)
        print(); print('=' * 64); print('可验证性检查（%d 条改写查询）' % n); print('=' * 64)
        for k, nm in (('A', 'A 裸注入'), ('B', 'B +可验证性(弃答)'), ('C', 'C +可验证性+重试')):
            s = sum(r[k] for r in rows)
            ab = sum(1 for r in rows if r[k] == 0)
            print('  %-22s 答对 %2d/%d = %.2f   错/弃答 %d' % (nm, s, n, s / n, ab))
        json.dump(rows, open(os.path.join(ROOT, "out", "results_verify.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        for p, lf in ((p3, lf3), (pb, lb)):
            p.terminate()
            try: p.wait(timeout=15)
            except Exception: p.kill()
            lf.close()