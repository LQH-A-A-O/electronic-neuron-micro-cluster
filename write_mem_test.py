# -*- coding: utf-8 -*-
"""记忆能救写作吗：分节生成，有记忆 vs 无记忆。"""
import re, sys, time
import requests
LLM = 'http://127.0.0.1:8050'
OUTLINE = ['为什么需要外部记忆', '记忆层的设计', '可验证性检查', '实测数据与局限']
def gen(msgs, mx=700):
    d = requests.post(LLM + '/v1/chat/completions', timeout=900,
                      json={'messages': msgs, 'max_tokens': mx, 'temperature': 0.7}).json()
    return d['choices'][0].get('message', {}).get('content') or ''
def cn(t):
    return len(re.findall(r'[\u4e00-\u9fff]', t))
def rep8(t):
    s = re.sub(r'\s+', '', t)
    if len(s) < 24:
        return 0.0
    g = [s[i:i+8] for i in range(len(s)-8)]
    return 1.0 - len(set(g)) / len(g) if g else 0.0
def cross_rep(a, b):
    sa, sb = re.sub(r'\s+', '', a), re.sub(r'\s+', '', b)
    if len(sa) < 24 or len(sb) < 24:
        return 0.0
    ga = {sa[i:i+8] for i in range(len(sa)-8)}
    gb = {sb[i:i+8] for i in range(len(sb)-8)}
    inter = ga & gb
    return len(inter) / max(len(gb), 1)
def run(with_mem):
    parts, notes = [], []
    for i, title in enumerate(OUTLINE):
        if with_mem and notes:
            memo = chr(10).join('- 第%d节「%s」要点：%s' % (j+1, OUTLINE[j], n) for j, n in enumerate(notes))
            prompt = ('全文大纲：' + ' / '.join(OUTLINE) + chr(10)
                      + '已经写好的部分要点：' + chr(10) + memo + chr(10) + chr(10)
                      + '现在只写第 %d 节「%s」，200 字左右。**不要重复前面已说过的内容**，要有新信息。' % (i+1, title))
        else:
            prompt = ('全文大纲：' + ' / '.join(OUTLINE) + chr(10)
                      + '现在只写第 %d 节「%s」，200 字左右。' % (i+1, title))
        t = gen([{'role': 'user', 'content': prompt}])
        parts.append(t)
        notes.append(t[:40].replace(chr(10), ' '))
    return parts
if __name__ == '__main__':
    for label, wm in (('A 无记忆', False), ('B 有记忆', True)):
        t0 = time.time()
        parts = run(wm)
        total = sum(cn(p) for p in parts)
        cr = max(cross_rep(parts[i], parts[j]) for i in range(len(parts)) for j in range(i+1, len(parts)))
        rr = rep8(''.join(parts))
        cov = sum(1 for k in OUTLINE if any(k[:4] in p or k in p for p in parts))
        print('=== %s ===' % label)
        print('  总字数 %d   节数 %d   覆盖大纲 %d/%d' % (total, len(parts), cov, len(OUTLINE)))
        print('  单节内重复 %.1f%%   跨节最高重叠 %.1f%%   用时 %.0f 秒' % (100*rr, 100*cr, time.time()-t0))
        for i, p in enumerate(parts):
            print('    第%d节 %d 字：%s' % (i+1, cn(p), p[:44].replace(chr(10), ' ')))
        print()
        sys.stdout.flush()