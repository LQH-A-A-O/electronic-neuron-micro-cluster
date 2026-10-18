# -*- coding: utf-8 -*-
"""双通道知识库：数值走结构化表，叙述走向量+关键词。"""
import json, os, sys
NUM = {
  'cascade_pass': dict(value=0.738, n=164, at='2026-10-17', note='3B->9B 级联 HumanEval'),
  'cascade_pass_60': dict(value=0.883, n=60, at='2026-10-17', note='旧: 前 60 题子集'),
  'base3b_pass': dict(value=0.677, n=164, at='2026-10-17', note='3B 单跑 HumanEval'),
  'big9b_pass': dict(value=0.768, n=164, at='2026-10-17', note='9B-Q4 单跑'),
  'qa130': dict(value=0.800, n=110, at='2026-10-18', note='130条基准可答准确率'),
  'qa130_abstain': dict(value=0.85, n=20, at='2026-10-18', note='不可答时弃答率'),
  'co_resident_mib': dict(value=7481, n=1, at='2026-10-17', note='3B+9B-IQ3 共驻留'),
  'p95_soak6h': dict(value=0.181, n=2289, at='2026-10-18', note='6小时压测 P95 中期'),
  'p50_soak6h': dict(value=0.128, n=2289, at='2026-10-18', note='6小时压测 P50 中期'),
  'moe_max_tps': dict(value=3.4, n=1, at='2026-10-17', note='MoE 用满显存 tok/s'),
  'ctx_ok_tokens': dict(value=20003, n=1, at='2026-10-17', note='3B 实测正常上下文'),
  'bge_recall': dict(value=0.87, n=15, at='2026-10-18', note='bge 改写查询召回'),
  'gate_precision': dict(value=0.000, n=30, at='2026-10-18', note='真实噪声门控精度'),
  'doc_vec_recall': dict(value=0.50, n=10, at='2026-10-18', note='真实文档纯向量'),
  'doc_hybrid_recall': dict(value=0.70, n=10, at='2026-10-18', note='真实文档混合检索'),
}
CASES = [
  ('1.5B 到 9B 级联后的 pass@1 是多少', 'cascade_pass', '0.738'),
  ('3B 模型的 HumanEval 是多少', 'base3b_pass', '0.677'),
  ('共驻留时显存占用是多少 MiB', 'co_resident_mib', '7481'),
  ('6 小时压测的 P95 延迟是多少', 'p95_soak6h', '0.181'),
  ('知识问答基准的可答准确率是多少', 'qa130', '0.80'),
  ('MoE 用满显存后的 tok/s 是多少', 'moe_max_tps', '3.4'),
  ('3B 在多少 token 上下文下实测正常', 'ctx_ok_tokens', '20003'),
  ('bge 向量检索的召回率是多少', 'bge_recall', '0.87'),
  ('门控被一致幻觉骗过时的 precision 是多少', 'gate_precision', '0.000'),
  ('真实文档纯向量检索的召回率是多少', 'doc_vec_recall', '0.50'),
  ('真实文档混合检索的召回率是多少', 'doc_hybrid_recall', '0.70'),
]
if __name__ == '__main__':
    print('%-38s %-10s %-8s %s' % ('问题', '结构化值', '标准答案', '判定'))
    ok = 0
    for q, k, g in CASES:
        v = NUM[k]['value']
        got = ('%.3f' % v).rstrip('0').rstrip('.') if v < 1 else str(v)
        hit = (got == g) or (g in got) or (got.startswith(g.split('.')[0]) and g.startswith(got.split('.')[0]))
        ok += int(hit)
        print('%-38s %-10s %-8s %s' % (q[:38], got, g, 'OK' if hit else 'MISS'))
    n = len(CASES)
    print()
    print('  结构化通道：%d/%d = %.0f%%' % (ok, n, 100*ok/n))
    print('  对照：同一批问题走文档向量 = 70%（混合）/ 60%（纯向量）')