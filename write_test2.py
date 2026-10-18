# -*- coding: utf-8 -*-
"""续写拼接 + 文章推理测试。"""
import re, sys, time
import requests
LLM = 'http://127.0.0.1:8050'
ART = '某团队在 2026 年 9 月开始测试一套本地知识系统。9 月只有 3B 模型，单次延迟 0.12 秒；10 月加入了 9B 级联，延迟升到 0.70 秒，但准确率从 0.677 升到 0.738。11 月他们做了三件事：把缓存从内存换到显存、把门控阈值从 0.2 调到 0.1、加了可验证性检查。结果延迟回落到 0.18 秒，准确率保持 0.738，但弃答率从 5% 涨到 17%。12 月他们发现弃答里有 40% 是本可答对的，于是回滚了阈值调整。'
def gen(msgs, mx=2500):
    t0 = time.time()
    d = requests.post(LLM + '/v1/chat/completions', timeout=900,
                      json={'messages': msgs, 'max_tokens': mx, 'temperature': 0.7}).json()
    return (d['choices'][0].get('message', {}).get('content') or '',
            d.get('usage', {}).get('completion_tokens', 0), time.time() - t0)
def cn(t):
    return len(re.findall(r'[\u4e00-\u9fff]', t))
def rep(t, n=8):
    s = re.sub(r'\s+', '', t)
    if len(s) < n * 3:
        return 0.0
    g = [s[i:i+n] for i in range(len(s)-n)]
    return 1.0 - len(set(g)) / len(g) if g else 0.0
if __name__ == '__main__':
    print('=== A 续写拼接 ===')
    base = [{'role': 'user', 'content': '写一篇 3000 字的技术文章，主题是「本地小模型 + 外部记忆」，要有 6 个以上小标题。'}]
    total = ''
    for i in range(3):
        if i == 0:
            t, _, dt = gen(base)
        else:
            t, _, dt = gen(base + [{'role': 'assistant', 'content': total},
                                   {'role': 'user', 'content': '继续写，接着上文往下写，不要重复前面的内容。'}])
        total += t
        print('  第%d段 +%d 字  累计 %d 字  %.0f 秒' % (i+1, cn(t), cn(total), dt))
        sys.stdout.flush()
    print('  总字数 %d   整体重复率 %.1f%%' % (cn(total), 100*rep(total)))
    print()
    print('=== B 文章推理 ===')
    QS = [('这个系统最终（12月回滚后）的延迟比 9 月高还是低？', '高'),
          ('11 月弃答率上涨，最可能是哪个改动导致的？', '可验证性'),
          ('12 月为什么要回滚阈值？', '误杀')]
    ok = 0
    for q, need in QS:
        a, _, _ = gen([{'role': 'user', 'content': '材料：' + ART + chr(10) + chr(10) + '问：' + q + chr(10) + '简短回答。'}], 120)
        h = 1 if need in a else 0
        ok += h
        print('  %-28s -> %-30s [%s]' % (q[:28], a[:30].replace(chr(10), ' '), 'OK' if h else 'MISS'))
    print()
    print('  推理题 %d/3' % ok)