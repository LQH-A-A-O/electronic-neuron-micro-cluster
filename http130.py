# -*- coding: utf-8 -*-
"""130 条基准走 HTTP 服务（验证部署链路，而不是只验模型）。"""
import json, os, re, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = 'http://127.0.0.1:8499'
CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
KEEP = set('0123456789abcdefghijklmnopqrstuvwxyz' + '一二三四五六七八九十百千万零两')
def norm(t):
    t = t.lower().replace('百分之','').replace('一百','100').replace('万','0000')
    for k, v in CN.items(): t = t.replace(k, v)
    return ''.join(c for c in t if c in KEEP)
def hit(g, t):
    return bool(norm(g)) and norm(g) in norm(t)
if __name__ == '__main__':
    D = json.load(open(os.path.join(ROOT, 'data', 'qa_bench.json'), encoding='utf-8'))
    Q = D['queries']
    rows = []
    t0 = time.time()
    for i, x in enumerate(Q):
        try:
            d = requests.get(BASE + '/api/ask', params={'q': x['q']}, timeout=120).json()
        except Exception as e:
            d = {'answer': 'ERR', 'error': str(e)[:80]}
        rows.append(dict(q=x['q'], gold=x['gold'], ans=x['answerable'],
                          got=d.get('answer',''), verified=d.get('verified'), ms=d.get('ms')))
        if (i+1) % 40 == 0:
            print('  %d/%d' % (i+1, len(Q))); sys.stdout.flush()
    el = time.time() - t0
    ans = [r for r in rows if r['ans'] == 1]
    un = [r for r in rows if r['ans'] == 0]
    ok = sum(1 for r in ans if r['gold'] and hit(r['gold'], r['got']))
    n = len(ans)
    lo = 1.96 * ((ok/n)*(1-ok/n)/n) ** 0.5
    abst_un = sum(1 for r in un if norm(r['got']).strip() in ('', '不知道'))
    ms = [r['ms'] for r in rows if r.get('ms')]
    ms.sort()
    print()
    print('=' * 68)
    print('130 条基准 · 走 HTTP（%s）' % BASE)
    print('=' * 68)
    print('  可答准确 %d/%d = %.3f ±%.3f' % (ok, n, ok/n, lo))
    print('  不可答时弃答 %d/%d = %.0f%%' % (abst_un, len(un), 100*abst_un/len(un)))
    print('  总耗时 %.0f 秒   服务端 P50 %d ms   P95 %d ms' % (el, ms[len(ms)//2] if ms else 0, ms[int(len(ms)*0.95)] if ms else 0))
    json.dump(rows, open(os.path.join(ROOT, 'out', 'results_http130.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)