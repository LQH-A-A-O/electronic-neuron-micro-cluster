# -*- coding: utf-8 -*-
"""`.memory/` 存储层：真源是 markdown，索引是派生。

借鉴 interactive-novel 的「状态外置」：
  - 真源：.memory/facts.md / episodes.md（人可读、可 diff、可 git）
  - 派生：.memory/index.json（向量索引，随时可 rebuild）

用法：
    python memory_store.py rebuild      从真源重建索引
    python memory_store.py add-fact     交互式加一条事实
    python memory_store.py search 问题  检索
"""
import glob, hashlib, json, os, re, sys, time
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
MEM = os.path.join(ROOT, '.memory')
EMB = 'http://127.0.0.1:8044'
FACTS = os.path.join(MEM, 'facts.md')
EPISODES = os.path.join(MEM, 'episodes.md')
INDEX = os.path.join(MEM, 'index.json')
META = os.path.join(MEM, 'meta.json')
MAXCH = 500
def ensure():
    os.makedirs(MEM, exist_ok=True)
    for p, hdr in ((FACTS, '# 稳定事实（唯一真源）' + chr(10) + chr(10)
                           + '格式：- `key` = `value`  <-- measured_at: YYYY-MM-DD, n: 样本数' + chr(10)),
                   (EPISODES, '# 对话片段（唯一真源）' + chr(10) + chr(10)
                              + '格式：## 轮次 N | 时间' + chr(10) + '- 用户：...' + chr(10) + '- 回答：...' + chr(10) + '- 事实：...' + chr(10))):
        if not os.path.exists(p):
            open(p, 'w', encoding='utf-8').write(hdr)
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16] if os.path.exists(p) else ''
def parse_facts():
    out = []
    if not os.path.exists(FACTS):
        return out
    for line in open(FACTS, encoding='utf-8'):
        m = re.match(r'\s*[-*]\s+`([^`]+)`\s*=\s*`([^`]+)`(.*)$', line)
        if m:
            k, v, tail = m.group(1), m.group(2), m.group(3)
            at = (re.search(r'measured_at:\s*([0-9-]+)', tail) or [None, ''])[1]
            n = (re.search(r'n:\s*([0-9]+)', tail) or [None, ''])[1]
            desc = (re.search(r'说明:\s*([^,]+)', tail) or [None, ''])[1].strip()
            out.append(dict(key=k, value=v, at=at, n=n, desc=desc,
                            text=('%s = %s  %s' % (k, v, desc)).strip(), kind='fact'))
    return out
def parse_episodes():
    out = []
    if not os.path.exists(EPISODES):
        return out
    txt = open(EPISODES, encoding='utf-8').read()
    for blk in re.split(r'^## ', txt, flags=re.M)[1:]:
        head = blk.split(chr(10))[0].strip()
        body = blk.strip()
        if len(body) < 10:
            continue
        out.append(dict(key=head, value='', at=head.split('|')[-1].strip() if '|' in head else '',
                        n='', text=body[:MAXCH], kind='episode'))
    return out
def embed(ts):
    v = []
    for t in ts:
        d = requests.post(EMB + '/v1/embeddings', timeout=90, json={'input': t, 'model': 'x'}).json()
        v.append(d['data'][0]['embedding'])
    a = np.array(v, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
def rebuild():
    ensure()
    chs = parse_facts() + parse_episodes()
    if not chs:
        print('真源是空的，没什么可建')
        return 0
    V = embed([c['text'] for c in chs])
    json.dump(dict(chunks=chs, vecs=V.tolist()), open(INDEX, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(dict(facts_sha=sha(FACTS), episodes_sha=sha(EPISODES),
                   built_at=time.strftime('%Y-%m-%d %H:%M:%S'), n=len(chs)),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False)
    print('索引已重建：%d 条（事实 %d / 片段 %d）'
          % (len(chs), sum(1 for c in chs if c['kind']=='fact'), sum(1 for c in chs if c['kind']=='episode')))
    return len(chs)
def load():
    if not os.path.exists(INDEX):
        return None, None
    d = json.load(open(INDEX, encoding='utf-8'))
    return d['chunks'], np.array(d['vecs'], dtype=np.float32)
def search(q, k=3):
    chs, V = load()
    if chs is None:
        return []
    qv = embed([q])[0]
    idx = np.argsort(-(V @ qv))[:k]
    return [(chs[i], float(V[i] @ qv)) for i in idx]
def add_fact(key, value, at=None, n=None):
    ensure()
    at = at or time.strftime('%Y-%m-%d')
    line = '- `%s` = `%s`  <-- measured_at: %s%s' % (key, value, at, (', n: %s' % n) if n else '')
    with open(FACTS, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))
    print('已追加到 facts.md：%s' % line)
if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'rebuild'
    if cmd == 'rebuild':
        rebuild()
    elif cmd == 'search':
        for c, s in search(' '.join(sys.argv[2:])):
            print('  %.3f  [%s] %s' % (s, c['kind'], c['text'][:90].replace(chr(10), ' ')))
    elif cmd == 'add-fact':
        add_fact(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)