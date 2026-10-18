# -*- coding: utf-8 -*-
"""意图映射：从自然语言问题猜出该查哪个 metric key（关键词打分，非手工查表）。"""
import json, os, re, sys
NUM = {
  'cascade_pass': dict(value=0.738, n=164, note='3B 9B 级联 pass@1 HumanEval 164题'),
  'base3b_pass':  dict(value=0.677, n=164, note='3B 单跑 HumanEval pass@1 基线'),
  'qa130':        dict(value=0.800, n=110, note='知识问答 130条 基准 可答 准确率'),
  'qa130_abstain':dict(value=0.85,  n=20,  note='不可答 弃答率 不知道'),
  'co_resident_mib':dict(value=7481, n=1, note='共驻留 显存占用 MiB 3B 9B'),
  'p95_soak6h':   dict(value=0.181, n=2289,note='6小时 压测 P95 延迟 分钟'),
  'p50_soak6h':   dict(value=0.128, n=2289,note='6小时 压测 P50 延迟 中位数'),
  'moe_max_tps':  dict(value=3.4, n=1, note='MoE 用满 显存 tok/s 速度'),
  'ctx_ok_tokens':dict(value=20003, n=1, note='上下文 token 长度 实测 正常'),
  'bge_recall':   dict(value=0.87, n=15, note='bge 向量 检索 召回率 改写'),
  'gate_precision':dict(value=0.000, n=30, note='门控 precision 精度 一致幻觉'),
  'doc_vec_recall':dict(value=0.50, n=10, note='真实文档 纯向量 检索 召回率'),
  'doc_hybrid_recall':dict(value=0.70, n=10, note='真实文档 混合 关键词 检索 召回率'),
}
# 同义词扩展（真实系统需要维护这张表）
SYN = {'召回率':'召回 召回率 recall', '准确率':'准确率 正确率 acc',
       '延迟':'延迟 耗时 多久', '显存':'显存 显存占用 MiB 内存',
       '级联':'级联 cascade 升级', '速度':'速度 tok/s 吞吐 快',
       '上下文':'上下文 token 长度', '门控':'门控 precision 精度',
       '弃答':'弃答 不知道 拒答', '压测':'压测 长测 常压'}
def tokens(s):
    return set(t.lower() for t in re.findall(r'[A-Za-z][A-Za-z0-9.+#-]{0,20}|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]{2,4}', s))
def expand(q):
    t = tokens(q)
    for k, v in SYN.items():
        if k in q:
            t |= tokens(v)
    return t
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
  # 改写问法（不在训练表里）
  ('那个级联跑出来多少分', 'cascade_pass', '0.738'),
  ('光靠 3B 能对多少题', 'base3b_pass', '0.677'),
  ('知识问答这块正确率怎么样', 'qa130', '0.80'),
  ('压测的时候慢不慢', 'p95_soak6h', '0.181'),
  ('MoE 到底能跑多快', 'moe_max_tps', '3.4'),
]
if __name__ == '__main__':
    kt = {k: expand(k.replace('_',' ') + ' ' + v['note']) for k, v in NUM.items()}
    ok = 0
    print('%-38s %-20s %s' % ('问题', '映射到', '判定'))
    for q, gold_k, gold_v in CASES:
        qt = expand(q)
        best, bs = None, 0
        for k, t in kt.items():
            s = len(qt & t) / max(len(qt), 1)
            if s > bs: best, bs = k, s
        hit = (best == gold_k)
        ok += int(hit)
        print('%-38s %-20s %s' % (q[:38], best + ('*' if not hit else ''), 'OK' if hit else 'MISS(应 %s)' % gold_k))
    n = len(CASES)
    print()
    print('  意图映射正确 %d/%d = %.0f%%' % (ok, n, 100*ok/n))