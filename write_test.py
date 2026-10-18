# -*- coding: utf-8 -*-
"""写作能力实测：短文 / 长文 / 文章推理。"""
import os, re, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
LLM = 'http://127.0.0.1:8050'
def gen(prompt, mx):
    t0 = time.time()
    d = requests.post(LLM + '/v1/chat/completions', timeout=900,
                      json={'messages': [{'role': 'user', 'content': prompt}],
                            'max_tokens': mx, 'temperature': 0.7}).json()
    ch = d['choices'][0].get('message', {}).get('content') or ''
    return ch, d.get('usage', {}).get('completion_tokens', 0), time.time() - t0
def cn_len(t):
    return len(re.findall(r'[\u4e00-\u9fff]', t))
def rep_rate(t, n=8):
    s = re.sub(r'\s+', '', t)
    if len(s) < n * 3:
        return 0.0
    gs = [s[i:i+n] for i in range(len(s) - n)]
    if not gs:
        return 0.0
    return 1.0 - len(set(gs)) / len(gs)
JOBS = [
  ('短文 600-800字', '写一篇 600 到 800 字的技术短文，主题是「为什么单机部署也能做知识问答」，要有 3 个小标题。', 1600),
  ('长文 3000-4000字', '写一篇 3000 到 4000 字的完整技术文章，主题是「本地小模型 + 外部记忆的架构设计」，要有 6 个以上的小标题，并在结尾给出总结。', 5000),
]
if __name__ == '__main__':
    for name, prompt, mx in JOBS:
        print('=== %s ===' % name)
        t, nt, dt = gen(prompt, mx)
        n = cn_len(t)
        heads = len(re.findall(r'^#{1,3}\s|^[一二三四五六七八九十]+、', t, re.M))
        rr = rep_rate(t)
        print('  实际中文字数 %d   生成 token %d   用时 %.0f 秒   速度 %.1f tok/s'
              % (n, nt, dt, nt / max(dt, 0.01)))
        print('  小标题 %d 个   8-gram 重复率 %.1f%%' % (heads, 100 * rr))
        print('  开头：%s' % t[:60].replace(chr(10), ' '))
        print('  结尾：%s' % t[-60:].replace(chr(10), ' '))
        print()
        sys.stdout.flush()