# -*- coding: utf-8 -*-
"""生成文档索引 INDEX.md（按主题分类）。"""
import glob, os, re
ROOT = os.path.dirname(os.path.abspath(__file__))
RULES = [
  ('核心文档', ['README.md', '技术文档-神经元架构.md', 'LICENSE']),
  ('结论与验收', ['能落地了吗', '落地验收报告', '落地-知识服务', '部署手册', '醒来先看这个',
                  '系统规格', '最终配置', '方案定稿']),
  ('评测与基准', ['评测-130条基准', '压测', '常压长测', '长压力负载', '并发测试', '并发压测']),
  ('记忆架构', ['memory', '记忆', '门控', '幻觉回灌', '语料入库', '追问', '修复-']),
  ('检索与知识库', ['检索', '向量', '双通道', '意图映射', '真实文档', '混合检索', '规模验证', '上下文']),
  ('模型与显存', ['显存', 'GPU', 'KV', '共驻留', '9B', '第三层', '级联', '蒸馏', '自建参数库', 'NeuRAG', '三层级联']),
  ('写作能力', ['写作', '记忆对写作']),
  ('阶段报告', ['报告-']),
]
def classify(name):
    for cat, keys in RULES:
        for k in keys:
            if k in name:
                return cat
    return '其他记录'
if __name__ == '__main__':
    files = sorted(f for f in glob.glob(os.path.join(ROOT, '*.md')) if os.path.basename(f) != 'INDEX.md')
    groups = {}
    for f in files:
        b = os.path.basename(f)
        groups.setdefault(classify(b), []).append(b)
    order = [c for c, _ in RULES] + ['其他记录']
    lines = ['# 文档索引', '',
             '本项目的全部结论文档。每份都对应一组实测，负结果照实记录。', '',
             '> 建议阅读顺序：**README → 技术文档 → 能落地了吗 → 按主题深入**', '']
    for cat in order:
        if cat not in groups:
            continue
        lst = sorted(groups[cat])
        lines.append('## %s（%d 份）' % (cat, len(lst)))
        lines.append('')
        for b in lst:
            lines.append('- [%s](%s)' % (b[:-3], b))
        lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('总计 **%d 份文档**。另有 `exp/` 与根目录下的实验脚本可复现每个数字。' % len(files))
    open(os.path.join(ROOT, 'INDEX.md'), 'w', encoding='utf-8').write(chr(10).join(lines))
    print('INDEX.md 已生成：%d 份文档' % len(files))
    for cat in order:
        if cat in groups:
            print('  %-14s %d' % (cat, len(groups[cat])))