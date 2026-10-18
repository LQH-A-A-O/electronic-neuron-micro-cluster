# -*- coding: utf-8 -*-
"""版本漂移修复：给切片加时间戳，检索时按「新」加权。"""
import json, os, sys
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
EMB = 'http://127.0.0.1:8044'
QS = [
  ('知识问答基准的可答准确率是多少', '80', '评测-130条基准.md'),
  ('MoE 用满显存后的 tok/s 是多少', '3.4', '显存重测与第三层.md'),
  ('门控被一致幻觉骗过时的 precision 是多少', '0.000', '门控真实噪声-重大负结果.md'),
  ('bge 向量检索的召回率是多少', '87', '向量检索验证-结论.md'),
  ('3B 模型的 HumanEval 是多少', '0.677', '三层级联判定.md'),
  ('1.5B 到 9B 级联后的 pass@1 是多少', '0.738', '级联实测成立.md'),
  ('3B 在多少 token 上下文下实测正常', '20003', '上下文上限理论.md'),
  ('共驻留时显存占用是多少 MiB', '7481', '共驻留与换模型.md'),
  ('6 小时压测的 P95 延迟是多少', '0.181', '压测中期分析.md'),
  ('深链接的 URL 参数是什么', '/?q=', '前端视觉验证.md'),
]
def emb(ts):
    v = []
    for t in ts:
        d = requests.post(EMB + '/v1/embeddings', timeout=60, json={'input': t, 'model': 'x'}).json()
        v.append(d['data'][0]['embedding'])
    a = np.array(v, dtype=np.float32); n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n
if __name__ == '__main__':
    chs = json.load(open(os.path.join(ROOT, 'data', 'doc_index.json'), encoding='utf-8'))
    V = np.load(os.path.join(ROOT, 'data', 'doc_vecs.npy'))
    # 给每个切片配文件 mtime（索引里没存，重新算）
    mt = {}
    for c in chs:
        p = os.path.join(ROOT, c['doc'])
        mt[c['doc']] = os.path.getmtime(p) if os.path.exists(p) else 0
    times = np.array([mt.get(c['doc'], 0) for c in chs], dtype=np.float64)
    rk = (times - times.min()) / max(times.max() - times.min(), 1)   # 0..1 新旧程度
    Q = emb([q for q, _, _ in QS])
    S = Q @ V.T
    print('%-34s %-8s %-8s %-8s' % ('问题', '纯相似', '+新旧0.05', '+新旧0.15'))
    acc = {0: 0, 0.05: 0, 0.15: 0}
    for i, (q, need, doc) in enumerate(QS):
        line = '%-34s' % q[:34]
        for lam in (0, 0.05, 0.15):
            s = S[i] + lam * rk
            top = np.argsort(-s)[:3]
            ok = int(any(need in chs[j]['text'] for j in top))
            acc[lam] += ok
            line += ' %-8s' % ('OK' if ok else 'MISS')
        print(line)
    n = len(QS)
    print()
    for lam in (0, 0.05, 0.15):
        print('  λ=%.2f  top-3 含正确答案 %d/%d = %.0f%%' % (lam, acc[lam], n, 100*acc[lam]/n))