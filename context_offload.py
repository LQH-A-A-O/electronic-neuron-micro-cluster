# -*- coding: utf-8 -*-
"""上下文卸载实验：全量 vs 滑窗 vs 检索式记忆。

构造 20 轮对话，每轮引入一个事实；最后问一个必须回溯早期事实的问题。
三种策略：
  A 全量上下文（上界）
  B 滑窗（只留最近 5 轮）
  C 检索式记忆（每轮入库，按当前问题召回 top-3）
"""
import json, os, sys
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
EMB = 'http://127.0.0.1:8044'
LLM = 'http://127.0.0.1:8050'
# 20 轮对话：每轮一个事实（项目代号、人名、数字）
TURNS = [
  '我们的项目代号定为「青鸾」。',
  '数据库用的是 PostgreSQL 16。',
  '缓存层选了 Redis，版本 7.2。',
  '发布窗口定在每周四凌晨两点。',
  '灰度比例初始设 5%。',
  '负责人是陈工，备份是李工。',
  '日志保留 30 天。',
  '网关限流每秒 200 次。',
  '灾备机房在广州。',
  '证书有效期一年，2027 年 3 月到期。',
  '监控告警发到「运维一号群」。',
  '消息队列积压阈值 5 万条。',
  '回滚预案要求 15 分钟内完成。',
  '压测目标是 500 并发。',
  '代码仓库在 GitLab 私有组。',
  'CI 用 GitHub Actions 的镜像。',
  '每次提交要求双人 review。',
  '前端框架用 Vue 3。',
  '移动端用 Flutter。',
  '文档统一放语雀。',
]
# (问题, 必须出现的关键词, 需要回溯到第几轮)
QS = [
  ('项目代号是什么', '青鸾', 0),
  ('数据库用的是什么版本', '16', 1),
  ('灰度比例初始设多少', '5', 4),
  ('灾备机房在哪', '广州', 8),
  ('监控告警发到哪个群', '运维一号群', 10),
  ('回滚预案要求多久完成', '15', 12),
  ('压测目标是多少并发', '500', 13),
  ('移动端用什么框架', 'Flutter', 18),
]
def emb(ts):
    v = []
    for t in ts:
        d = requests.post(EMB + '/v1/embeddings', timeout=60, json={'input': t, 'model': 'x'}).json()
        v.append(d['data'][0]['embedding'])
    a = np.array(v, dtype=np.float32); n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
def llm(msg, mx=40):
    d = requests.post(LLM + '/v1/chat/completions', timeout=300,
                      json={'messages': [{'role': 'user', 'content': msg}], 'max_tokens': mx, 'temperature': 0}).json()
    return d['choices'][0]['message']['content'] or ''
if __name__ == '__main__':
    V = emb(TURNS)
    full = chr(10).join('- ' + t for t in TURNS)
    res = {'A': 0, 'B': 0, 'C': 0}
    for q, need, src in QS:
        # A 全量
        a = llm('对话记录：' + chr(10) + full + chr(10) + chr(10) + '问：' + q + chr(10) + '只答要点。')
        # B 滑窗（最近 5 轮）
        win = chr(10).join('- ' + t for t in TURNS[-5:])
        b = llm('对话记录：' + chr(10) + win + chr(10) + chr(10) + '问：' + q + chr(10) + '只答要点。')
        # C 检索式（按问题召回 top-3）
        qv = emb([q])[0]
        idx = np.argsort(-(V @ qv))[:3]
        mem = chr(10).join('- ' + TURNS[j] for j in idx)
        c = llm('相关记忆：' + chr(10) + mem + chr(10) + chr(10) + '问：' + q + chr(10) + '只答要点。')
        ha, hb, hc = int(need in a), int(need in b), int(need in c)
        res['A'] += ha; res['B'] += hb; res['C'] += hc
        print('  %-22s A%s B%s C%s  (需第%d轮)' % (q[:22], 'O' if ha else 'X', 'O' if hb else 'X',
              'O' if hc else 'X', src + 1))
        print('       A答「%s」 B答「%s」 C答「%s」' % (a[:14].replace(chr(10),' '),
              b[:14].replace(chr(10),' '), c[:14].replace(chr(10),' ')))
        sys.stdout.flush()
    n = len(QS)
    print()
    print('=' * 56)
    print('上下文卸载（%d 轮对话，%d 个回溯问题）' % (len(TURNS), n))
    print('=' * 56)
    print('  A 全量上下文   %d/%d = %.0f%%   注入 %d 轮' % (res['A'], n, 100*res['A']/n, len(TURNS)))
    print('  B 滑窗(5轮)    %d/%d = %.0f%%   注入 5 轮' % (res['B'], n, 100*res['B']/n))
    print('  C 检索式(3轮)  %d/%d = %.0f%%   注入 3 轮' % (res['C'], n, 100*res['C']/n))