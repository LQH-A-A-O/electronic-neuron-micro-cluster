# -*- coding: utf-8 -*-
"""在 130 条评测集上跑三组对照。

  A 无记忆           基线
  B 有记忆（top-3）  记忆注入
  C 有记忆 + 可验证性  找不到依据 -> 不知道

指标：
  可答题准确率（110 条）
  不可答题的误答率（20 条，越低越好）
"""
import json, os, re, subprocess, sys, time
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MB = r"E:\aqca\work\llm\models\bge-small-zh-v1.5-q8_0.gguf"
P3, PB = 8034, 8033
CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
KEEP = set('0123456789abcdefghijklmnopqrstuvwxyz' + '一二三四五六七八九十百千万零两')
def norm(t):
    t = t.lower().replace('百分之','').replace('一百','100').replace('万','0000')
    for k, v in CN.items(): t = t.replace(k, v)
    return ''.join(c for c in t if c in KEEP)
def hit(g, t):
    return bool(norm(g)) and norm(g) in norm(t)
def supported(ans, ctx):
    a, c = norm(ans), norm(ctx)
    toks = re.findall(r'[0-9]{2,7}', a) + re.findall(r'[a-z][a-z0-9-]{1,18}', a)
    if not toks: return True
    return all(t in c for t in toks)
def launch(model, port, extra=None):
    lf = open(os.path.join(ROOT, "out", "qb_%d.log" % port), "wb")
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
def chat(port, msg, mx=80):
    d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600,
                      json={"messages": [{"role": "user", "content": msg}], "max_tokens": mx, "temperature": 0}).json()
    return d["choices"][0]["message"]["content"] or ""
def emb(ts):
    v = [requests.post("http://127.0.0.1:%d/v1/embeddings" % PB, timeout=60, json={"input": t, "model": "x"}).json()["data"][0]["embedding"] for t in ts]
    a = np.array(v, dtype=np.float32); n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
if __name__ == '__main__':
    D = json.load(open(os.path.join(ROOT, 'data', 'qa_bench.json'), encoding='utf-8'))
    facts, queries = D['facts'], D['queries']
    p3, lf3 = launch(M3, P3); pb, lb = launch(MB, PB, ['--embeddings', '--pooling', 'cls'])
    try:
        keys = [f['key'] for f in facts]; K = emb(keys)
        Q = emb([x['q'] for x in queries]); S = Q @ K.T
        rows = []
        for i, x in enumerate(queries):
            order = np.argsort(-S[i]); ctx3 = [keys[j] for j in order[:3]]
            ctx = chr(10).join('  - ' + k for k in ctx3)
            a = chat(P3, x['q'] + chr(10) + '只回答要点，不超过15个字。')
            b = chat(P3, '已知下列事实：' + chr(10) + ctx + chr(10) + chr(10) + '问：' + x['q'] + chr(10) + '只回答要点，不超过15个字。')
            c = b if supported(b, ctx) else '不知道'
            rows.append(dict(q=x['q'], gold=x['gold'], ans=x['answerable'], A=a[:60], B=b[:60], C=c[:40]))
            if (i+1) % 30 == 0:
                print('  %d/%d' % (i+1, len(queries))); sys.stdout.flush()
        def acc(key):
            ans = [r for r in rows if r['ans'] == 1]
            got = sum(1 for r in ans if r['gold'] and hit(r['gold'], r[key]))
            return got, len(ans)
        def false_ans(key):
            un = [r for r in rows if r['ans'] == 0]
            bad = sum(1 for r in un if norm(r['C' if key=='C' else key]).strip() not in ('', '不知道'))
            return bad, len(un)
        print(); print('=' * 68)
        print('130 条评测集（110 可答 / 20 不可答）')
        print('=' * 68)
        for key, nm in (('A','A 无记忆'), ('B','B 有记忆'), ('C','C 有记忆+可验证')):
            g, n = acc(key)
            lo = 1.96 * ((g/n) * (1-g/n) / n) ** 0.5 if n else 0
            bad, un = false_ans(key)
            print('  %-18s 可答准确 %3d/%d = %.3f ±%.3f   不可答时乱答 %2d/%d'
                  % (nm, g, n, g/n, lo, bad, un))
        json.dump(rows, open(os.path.join(ROOT, "out", "results_qa130.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        for p, lf in ((p3, lf3), (pb, lb)):
            p.terminate()
            try: p.wait(timeout=15)
            except Exception: p.kill()
            lf.close()