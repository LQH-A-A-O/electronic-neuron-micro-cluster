# -*- coding: utf-8 -*-
"""重算准确率：归一化数字表达后再判分（用缓存回答，不跑模型）。"""
import json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))

PUNCT = ' \t\n，。、；：！？（）()[]【】「」"\'`~!@#$%^&*+=|\\/<>-'
CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}

def cn2num(s):
    s = s.replace('百分之','').replace('％','').replace('%','')
    s = s.replace('一百','100').replace('十','10').replace('百','00')
    for k, v in CN.items():
        s = s.replace(k, v)
    return s

def norm(t):
    t = t.lower()
    for ch in PUNCT:
        t = t.replace(ch, '')
    return cn2num(t)

def hit(gold, txt):
    g, x = norm(gold), norm(txt)
    return bool(g) and g in x

if __name__ == '__main__':
    rows = json.load(open(os.path.join(ROOT,'out','results_e2v.json'), encoding='utf-8'))
    n = len(rows)
    a0 = sum(r['ok_a'] for r in rows); b0 = sum(r['ok_b'] for r in rows)
    a1 = sum(int(hit(r['gold'], r['a'])) for r in rows)
    b1 = sum(int(hit(r['gold'], r['b'])) for r in rows)
    print('归一化前后对比（%d 条）' % n)
    print()
    print('  %-10s %-16s %s' % ('组','原判分','归一化后'))
    print('  %-10s %-16s %s' % ('A 无记忆','%d/%d = %.2f'%(a0,n,a0/n),'%d/%d = %.2f'%(a1,n,a1/n)))
    print('  %-10s %-16s %s' % ('B 有记忆','%d/%d = %.2f'%(b0,n,b0/n),'%d/%d = %.2f'%(b1,n,b1/n)))
    print()
    print('  变化：A %+d 条，B %+d 条' % (a1-a0, b1-b0))
    print()
    flips = [r for r in rows if (not r['ok_b']) and hit(r['gold'], r['b'])]
    if flips:
        print('  B 组由错变对（原来冤枉了模型）：')
        for r in flips:
            print('    %-16s 期望「%s」  实际「%s」' % (r['q'][:16], r['gold'], r['b'][:40]))
    print()
    print('  B 归一化后仍错：')
    for r in rows:
        if not hit(r['gold'], r['b']):
            print('    %-16s 期望「%-8s」 实际「%s」 召回「%s」' % (r['q'][:16], r['gold'], r['b'][:30], r['ctx'][0][:14]))