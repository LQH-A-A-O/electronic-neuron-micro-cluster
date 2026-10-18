# -*- coding: utf-8 -*-
"""神经元知识服务：HTTP + 最小前端。

    GET  /            网页（输入问题，看答案与依据）
    GET  /api/ask?q=  返回 {answer, evidence[], verified, abstained, ms}
    GET  /api/stats   库状态

依赖的后端（需先起）：
    3B     :8050
    bge    :8044  (--embeddings --pooling cls)

设计约束（来自实测）：
  - 记忆写入源必须是外部数据，**不接受模型自采样**（实测门控 precision=0）
  - 答案必须能在召回文本里找到依据，否则弃答（把乱答从 40% 压到 20%）
"""
import json, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
LLM = 'http://127.0.0.1:8050'
EMB = 'http://127.0.0.1:8044'
PORT = 8499
TOP_K = 3

CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
KEEP = set('0123456789abcdefghijklmnopqrstuvwxyz')
def norm(t):
    t = t.lower().replace('百分之','').replace('万','0000')
    for k, v in CN.items(): t = t.replace(k, v)
    return ''.join(c for c in t if c in KEEP)

class Store:
    def __init__(self, facts):
        self.facts = facts
        self.keys = [f['key'] for f in facts]
        self.vecs = self._embed(self.keys)
    def _embed(self, ts):
        v = [requests.post(EMB + '/v1/embeddings', timeout=60,
                           json={'input': t, 'model': 'x'}).json()['data'][0]['embedding'] for t in ts]
        a = np.array(v, dtype=np.float32)
        n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
        return a / n
    def search(self, q, k=TOP_K):
        qv = self._embed([q])[0]
        s = self.vecs @ qv
        idx = np.argsort(-s)[:k]
        return [(self.keys[i], float(s[i])) for i in idx]

def llm(msgs, mx=90):
    d = requests.post(LLM + '/v1/chat/completions', timeout=300,
                      json={'messages': msgs, 'max_tokens': mx, 'temperature': 0}).json()
    return d['choices'][0]['message']['content'] or ''

def supported(ans, ctx):
    """答案是否能在召回文本里找到依据。

    三条判据，任一通过即算有据：
      1. 去空白的原文子串出现在上下文里（处理「华北2」这类中文+数字短答案）
      2. 归一化后的答案整体出现在归一化上下文里
      3. 答案里的数字/英文 token 全部能在上下文里找到

    历史 bug：曾把「归一化后长度<=2 判不可信」当兜底，
    导致「华北2」这类答案 100% 被误杀（压测里 2/12 的问题永远弃答）。已修。
    """
    if not ans or not ans.strip():
        return False
    raw_a = "".join(ans.split())
    raw_c = "".join(ctx.split())
    if raw_a and raw_a in raw_c:
        return True
    a, c = norm(ans), norm(ctx)
    if a and a in c:
        return True
    toks = re.findall(r'[0-9]{2,7}', a) + re.findall(r'[a-z][a-z0-9-]{1,18}', a)
    if not toks:
        return False
    return all(t in c for t in toks)

STORE = None

def answer(q):
    t0 = time.time()
    hits = STORE.search(q)
    ctx = chr(10).join('  - ' + k for k, _ in hits)
    a = llm([{'role': 'user', 'content': '已知下列事实：' + chr(10) + ctx + chr(10) + chr(10)
             + '问：' + q + chr(10) + '只回答要点，不超过15个字。'}]).strip()
    ok = supported(a, ctx)
    return dict(answer=('不知道' if not ok else a),
                raw=a,
                verified=bool(ok),
                abstained=bool(not ok),
                evidence=[{'fact': k, 'score': round(s, 4)} for k, s in hits],
                ms=int((time.time() - t0) * 1000))

PAGE = '''<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>神经元知识服务</title><style>
body{font-family:system-ui,'Segoe UI',sans-serif;max-width:760px;margin:40px auto;padding:0 20px;background:#0d1117;color:#e6edf3}
h1{font-size:20px;font-weight:600;margin-bottom:4px}
.sub{color:#8b949e;font-size:13px;margin-bottom:24px}
.row{display:flex;gap:8px}
input{flex:1;padding:12px 14px;border-radius:8px;border:1px solid #30363d;background:#161b22;color:#e6edf3;font-size:15px;outline:none}
input:focus{border-color:#1f6feb}
button{padding:12px 20px;border-radius:8px;border:0;background:#1f6feb;color:#fff;font-size:15px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
#out{margin-top:24px}
.ans{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px;margin-bottom:14px}
.ans .v{font-size:22px;font-weight:600}
.ans .v.no{color:#f0883e}
.meta{color:#8b949e;font-size:12px;margin-top:8px}
.ev{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px}
.ev h3{font-size:13px;color:#8b949e;margin:0 0 8px;font-weight:500}
.ev div{font-size:13px;padding:5px 0;border-bottom:1px solid #21262d;color:#c9d1d9}
.ev div:last-child{border:0}
.ev span{color:#58a6ff;font-size:11px;float:right}
</style></head><body>
<h1>神经元知识服务</h1>
<div class=sub>本地 3B + 向量记忆 · 答案均附召回依据</div>
<div class=row><input id=q placeholder='问点什么，比如：订单服务的监听端口是多少？' autofocus>
<button id=b onclick='go()'>提问</button></div>
<div id=out></div>
<script>
async function go(){
  const q=document.getElementById('q').value.trim(); if(!q) return;
  const b=document.getElementById('b'); b.disabled=true; b.textContent='思考中';
  const o=document.getElementById('out'); o.innerHTML='';
  try{
    const r=await fetch('/api/ask?q='+encodeURIComponent(q));
    const d=await r.json();
    const no=d.abstained?' no':'';
    o.innerHTML='<div class=ans><div class="v'+no+'">'+d.answer+'</div>'
      +'<div class=meta>'+(d.abstained?'依据不足，已弃答':'已验证')+' · '+d.ms+' ms</div></div>'
      +'<div class=ev><h3>召回依据</h3>'+
      d.evidence.map(e=>'<div>'+e.fact+'<span>'+e.score+'</span></div>').join('')+'</div>';
  }catch(e){ o.innerHTML='<div class=ans>请求失败：'+e+'</div>'; }
  b.disabled=false; b.textContent='提问';
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
// 深链接：/?q=... 打开即自动提问
(function(){var p=new URLSearchParams(location.search).get('q');if(p){document.getElementById('q').value=p;go();}})();
</script></body></html>'''

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, body, ctype):
        b = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            return self._send(200, PAGE, 'text/html')
        if u.path == '/api/ask':
            q = (parse_qs(u.query).get('q') or [''])[0].strip()
            if not q:
                return self._send(400, json.dumps({'error': 'empty q'}), 'application/json')
            try:
                return self._send(200, json.dumps(answer(q), ensure_ascii=False), 'application/json')
            except Exception as e:
                return self._send(500, json.dumps({'error': str(e)[:200]}, ensure_ascii=False), 'application/json')
        if u.path == '/api/stats':
            return self._send(200, json.dumps({'facts': len(STORE.facts), 'llm': LLM, 'emb': EMB},
                                              ensure_ascii=False), 'application/json')
        return self._send(404, 'not found', 'text/plain')

if __name__ == '__main__':
    D = json.load(open(os.path.join(ROOT, 'data', 'qa_bench.json'), encoding='utf-8'))
    print('加载 %d 条事实并建索引...' % len(D['facts']))
    STORE = Store(D['facts'])
    print('就绪： http://127.0.0.1:%d/' % PORT)
    sys.stdout.flush()
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()