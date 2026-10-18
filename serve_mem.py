# -*- coding: utf-8 -*-
"""知识服务 v2：知识源换成 .memory/，每轮自动写 episodes。

与 v1 的区别：
  1. 知识来自 .memory/（facts.md + episodes.md），不是 qa_bench.json
  2. 每次回答后自动追加一条 episode（对话记忆）
  3. 上下文 = 最近 2 轮（原样）+ 检索 top-3（历史/事实）
  4. 启动时自检：索引过期就自动重建
"""
import json, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import memory_store as MS
LLM = 'http://127.0.0.1:8050'
PORT = 8497
RECENT = 2
TOPK = 3
# 两个事实的相似度差小于这个值就判为歧义（追问而不是猜）
AMBIG_MARGIN = 0.04
# 触发追问的最低相关度：低于这个值说明"都不沾边"，不是歧义
AMBIG_FLOOR = 0.55
# 事实注入几条。实测 1 比 3 准：多给会诱发"挑错一条"。
FACT_TOPK = 1
CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
KEEP = set('0123456789abcdefghijklmnopqrstuvwxyz')
def norm(t):
    t = t.lower().replace('百分之','').replace('万','0000')
    for k, v in CN.items(): t = t.replace(k, v)
    return ''.join(c for c in t if c in KEEP)
def llm(msgs, mx=80):
    d = requests.post(LLM + '/v1/chat/completions', timeout=300,
                      json={'messages': msgs, 'max_tokens': mx, 'temperature': 0}).json()
    return d['choices'][0]['message']['content'] or ''
def supported(ans, ctx):
    if not ans or not ans.strip():
        return False
    ra = ''.join(ans.split()); rc = ''.join(ctx.split())
    if ra and ra in rc:
        return True
    a, c = norm(ans), norm(ctx)
    if a and a in c:
        return True
    # 从**原始回答**提 token，不要从 norm 后的串提 ——
    # norm 会剥掉空格和标点，把 "P95 ... 0.1745" 粘成 "p9501745"，导致误判不可验证。
    toks = re.findall(r'[0-9]{2,7}', ans) + re.findall(r'[A-Za-z][A-Za-z0-9-]{1,18}', ans)
    if not toks:
        return False
    return all(t.lower() in c for t in toks)
# 事实与对话片段**分两个索引**。
# 实测：混在一个索引里时，episode 的相似度（0.88~0.92）远高于事实（0.5~0.6），
# 给事实加 0.08 的权重也压不住，"问过同样问题"的旧会话会把真事实挤出 top-3。
CHUNKS, VECS = [], None          # 全量（供 stats 用）
FCH, FV = [], None               # 事实索引
ECH, EV = [], None               # 片段索引
NEXT_TURN = [1]
def load_index():
    global CHUNKS, VECS, FCH, FV, ECH, EV
    CHUNKS, VECS = MS.load()
    if CHUNKS is None:
        MS.rebuild(); CHUNKS, VECS = MS.load()
    fi = [i for i, c in enumerate(CHUNKS) if c['kind'] == 'fact']
    ei = [i for i, c in enumerate(CHUNKS) if c['kind'] == 'episode']
    FCH = [CHUNKS[i] for i in fi]; FV = VECS[fi] if fi else None
    ECH = [CHUNKS[i] for i in ei]; EV = VECS[ei] if ei else None
    NEXT_TURN[0] = len(MS.parse_episodes()) + 1
def add_vec(chunk):
    global ECH, EV
    v = MS.embed([chunk['text']])
    ECH.append(chunk)
    EV = v if EV is None else np.vstack([EV, v])
def recent_episodes(n=RECENT):
    return MS.parse_episodes()[-n:]
def answer(q):
    t0 = time.time()
    qv = MS.embed([q])[0]
    # 事实：取 top-3
    hits = []
    # 事实**只注入 top-1**：
    # 实测注入 top-3 时，模型会在两条都叫"准确率"的事实里挑错一条
    # （检索对、生成错）。少给一条反而更准。
    if FV is not None and len(FCH):
        sc = FV @ qv
        for i in np.argsort(-sc)[:FACT_TOPK]:
            hits.append((FCH[i], float(sc[i])))
    # 片段：取 top-2，单独一路，不和事实抢位
    eps_recent = set()
    for e in recent_episodes():
        eps_recent.add(e['text'][:60])
    if EV is not None and len(ECH):
        sc = EV @ qv
        n_ep = 0
        for i in np.argsort(-sc):
            if ECH[i]['text'][:60] in eps_recent:
                continue
            hits.append((ECH[i], float(sc[i])))
            n_ep += 1
            if n_ep >= 2:
                break
    # ---- 歧义检测：两个事实分数太接近 -> 不猜，追问 ----
    facts_only = [h for h in hits if h[0]['kind'] == 'fact']
    if len(facts_only) >= 2:
        s0, s1 = facts_only[0][1], facts_only[1][1]
        k0 = facts_only[0][0].get('key', '')
        k1 = facts_only[1][0].get('key', '')
        # 只有"两个都够相关"才算歧义。否则是"都不沾边"，应该交给可验证性检查去弃答。
        if k0 != k1 and (s0 - s1) < AMBIG_MARGIN and s0 >= AMBIG_FLOOR:
            opts = [{'key': facts_only[0][0].get('key', ''), 'text': facts_only[0][0]['text'][:60],
                     'score': round(s0, 3)},
                    {'key': facts_only[1][0].get('key', ''), 'text': facts_only[1][0]['text'][:60],
                     'score': round(s1, 3)}]
            return dict(answer='你的问题可能指两条不同的记录，请确认是哪一条？',
                        clarify=True, options=opts, verified=False, abstained=False,
                        evidence=[{'fact': h[0]['text'][:70], 'score': round(h[1], 4),
                                   'kind': h[0]['kind']} for h in hits],
                        recent_used=len(recent_episodes()), turn=NEXT_TURN[0],
                        ms=int((time.time() - t0) * 1000))
    ctx = chr(10).join('- ' + h[0]['text'][:280].replace(chr(10), ' ') for h in hits)
    rec = recent_episodes()
    recent = chr(10).join('- ' + e['text'][:200].replace(chr(10), ' ') for e in rec)
    msg = ('相关记忆：' + chr(10) + ctx + (chr(10) + chr(10) + '最近对话：' + chr(10) + recent if recent else '')
           + chr(10) + chr(10) + '问：' + q + chr(10) + '只回答要点，不超过15字。')
    a = llm([{'role': 'user', 'content': msg}]).strip()
    ok = supported(a, ctx)
    ans = a if ok else '不知道'
    # 写入对话记忆
    turn = NEXT_TURN[0]; NEXT_TURN[0] += 1
    stamp = time.strftime('%Y-%m-%d %H:%M')
    # ⚠️ 写记忆的规则（实测教训）：
    #   模型的未验证输出**不能**当事实写进去 —— 之前把错答案写成
    #   "- 回答：intent_map_accuracy = 0.81"，之后模型读到自己的错误就照抄，
    #   越错越自信（episodes 里堆了 5 条同样的错误）。
    # 现在：验证通过才写"回答"，不通过写"未验证猜测"；"依据"只记召回的 key。
    src_key = hits[0][0].get('key', '') if hits else ''
    lines = [chr(10) + '## 轮次 %d | %s' % (turn, stamp),
             '- 用户：' + q]
    if ok:
        lines.append('- 回答：' + ans)
    else:
        lines.append('- 未验证猜测：' + (a or '') + '（依据不足，未采纳）')
    lines.append('- 依据：' + (src_key or '无'))
    blk = chr(10).join(lines) + chr(10)
    with open(MS.EPISODES, 'a', encoding='utf-8') as f:
        f.write(blk)
    add_vec(dict(key='轮次 %d' % turn, value='', at=stamp, n='',
                 text=blk.strip()[:500], kind='episode'))
    return dict(answer=ans, raw=a, verified=bool(ok), abstained=bool(not ok),
                evidence=[{'fact': h[0]['text'][:70], 'score': round(h[1], 4), 'kind': h[0]['kind']} for h in hits],
                recent_used=len(rec), turn=turn, ms=int((time.time()-t0)*1000))
PAGE = '''<!doctype html><html lang=zh><head><meta charset=utf-8><title>记忆服务</title><style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;background:#0d1117;color:#e6edf3}
h1{font-size:20px}.sub{color:#8b949e;font-size:13px;margin-bottom:20px}
.row{display:flex;gap:8px}input{flex:1;padding:12px;border-radius:8px;border:1px solid #30363d;background:#161b22;color:#e6edf3;font-size:15px}
button{padding:12px 20px;border-radius:8px;border:0;background:#238636;color:#fff;font-size:15px;cursor:pointer}
#out{margin-top:20px}.ans{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:12px}
.v{font-size:22px;font-weight:600}.v.no{color:#f0883e}.meta{color:#8b949e;font-size:12px;margin-top:6px}
.ev{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px}
.ev div{font-size:13px;padding:4px 0;border-bottom:1px solid #21262d}.ev div:last-child{border:0}
</style></head><body><h1>记忆服务 v2</h1>
<div class=sub>知识源 .memory/ · 每轮自动记入对话记忆 · 答案附依据</div>
<div class=row><input id=q placeholder='问点什么，比如：级联的 pass@1 是多少' autofocus>
<button id=b onclick='go()'>提问</button></div><div id=out></div>
<script>async function go(){const q=document.getElementById('q').value.trim();if(!q)return;
const b=document.getElementById('b');b.disabled=true;b.textContent='…';const o=document.getElementById('out');
try{const d=await(await fetch('/api/ask?q='+encodeURIComponent(q))).json();
o.innerHTML='<div class=ans><div class="v'+(d.abstained?' no':'')+'">'+d.answer+'</div>'
+'<div class=meta>'+(d.abstained?'依据不足，已弃答':'已验证')+' · '+d.ms+'ms · 第'+d.turn+'轮</div></div>'
+'<div class=ev>'+d.evidence.map(e=>'<div>['+e.kind+'] '+e.fact+'  <span style="color:#58a6ff">'+e.score+'</span></div>').join('')+'</div>';
}catch(e){o.innerHTML='<div class=ans>请求失败：'+e+'</div>';}b.disabled=false;b.textContent='提问';}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')go()});</script></body></html>'''
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _s(self, code, body, ct):
        b = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', ct + '; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            return self._s(200, PAGE, 'text/html')
        if u.path == '/api/ask':
            q = (parse_qs(u.query).get('q') or [''])[0].strip()
            if not q:
                return self._s(400, json.dumps({'error': 'empty'}), 'application/json')
            try:
                return self._s(200, json.dumps(answer(q), ensure_ascii=False), 'application/json')
            except Exception as e:
                return self._s(500, json.dumps({'error': str(e)[:200]}, ensure_ascii=False), 'application/json')
        if u.path == '/api/stats':
            return self._s(200, json.dumps({'chunks': len(CHUNKS), 'next_turn': NEXT_TURN[0]}), 'application/json')
        return self._s(404, 'not found', 'text/plain')
if __name__ == '__main__':
    MS.ensure()
    iss = __import__('memory_check').check()
    if any(c in ('index-stale', 'index-missing') for _, c, _ in iss):
        print('索引过期，重建...'); MS.rebuild()
    load_index()
    print('就绪 http://127.0.0.1:%d/  记忆 %d 条，下一轮次 %d' % (PORT, len(CHUNKS), NEXT_TURN[0]))
    sys.stdout.flush()
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()