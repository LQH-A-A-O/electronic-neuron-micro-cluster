# -*- coding: utf-8 -*-
"""在真实文档知识库上测：检索命中率 + 问答准确率。"""
import json, os, sys
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
EMB = 'http://127.0.0.1:8044'
LLM = 'http://127.0.0.1:8050'
# (问题, 答案里必须出现的关键词, 应该在哪个文档里)
QS = [
  ('知识问答基准的可答准确率是多少', '80', '评测-130条基准.md'),
  ('MoE 用满显存后的 tok/s 是多少', '3.4', '显存重测与第三层.md'),
  ('门控被一致幻觉骗过时的 precision 是多少', '0', '门控真实噪声-重大负结果.md'),
  ('bge 向量检索的召回率是多少', '87', '向量检索验证-结论.md'),
  ('3B 模型的 HumanEval 是多少', '0.677', '三层级联判定.md'),
  ('1.5B 到 9B 级联后的 pass@1 是多少', '0.738', '级联实测成立.md'),
  ('3B 在多少 token 上下文下实测正常', '20003', '上下文上限理论.md'),
  ('共驻留时显存占用是多少 MiB', '7481', '共驻留与换模型.md'),
  ('6 小时压测的 P95 延迟是多少', '0.181', '压测中期分析.md'),
  ('深链接的 URL 参数是什么', 'q', '前端视觉验证.md'),
]
def emb(ts):
    v = []
    for t in ts:
        d = requests.post(EMB + '/v1/embeddings', timeout=60, json={'input': t, 'model': 'x'}).json()
        v.append(d['data'][0]['embedding'])
    a = np.array(v, dtype=np.float32); n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
def llm(msg, mx=60):
    d = requests.post(LLM + '/v1/chat/completions', timeout=300,
                      json={'messages': [{'role': 'user', 'content': msg}], 'max_tokens': mx, 'temperature': 0}).json()
    return d['choices'][0]['message']['content'] or ''
if __name__ == '__main__':
    chs = json.load(open(os.path.join(ROOT, 'data', 'doc_index.json'), encoding='utf-8'))
    V = np.load(os.path.join(ROOT, 'data', 'doc_vecs.npy'))
    Q = emb([q for q, _, _ in QS])
    S = Q @ V.T
    hit3 = hit1 = ans_ok = doc_ok = 0
    for i, (q, need, doc) in enumerate(QS):
        idx = np.argsort(-S[i])[:3]
        top = [chs[j] for j in idx]
        ctx = chr(10).join('  【%s】%s' % (c['doc'], c['text'][:300].replace(chr(10), ' ')) for c in top)
        a = llm('资料：' + chr(10) + ctx + chr(10) + chr(10) + '问：' + q + chr(10) + '只回答要点，不超过15字。')
        h1 = int(need in a)
        h3 = int(any(need in c['text'] for c in top))
        d = int(any(c['doc'] == doc for c in top))
        hit1 += h1; hit3 += h3; doc_ok += d
        print('  %-32s 答「%s」 %s | 命中正确文档 %s' % (q[:32], a[:14].replace(chr(10),' '),
              'OK' if h1 else 'MISS', 'OK' if d else 'MISS'))
        sys.stdout.flush()
    n = len(QS)
    print()
    print('=' * 64)
    print('真实文档知识库（%d 个切片 / 87 份文档）' % len(chs))
    print('=' * 64)
    print('  top-1 答案含关键词 %d/%d = %.0f%%' % (hit1, n, 100*hit1/n))
    print('  top-3 里有答案   %d/%d = %.0f%%' % (hit3, n, 100*hit3/n))
    print('  top-3 命中正确文档 %d/%d = %.0f%%' % (doc_ok, n, 100*doc_ok/n))