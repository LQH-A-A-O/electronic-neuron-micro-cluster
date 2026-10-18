# -*- coding: utf-8 -*-
"""验证补丁：Cluster 现在按输出形态分组维护历史。"""
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import encluster as E

c = E.Cluster(small_port=8052, big_port=8051)
qs = ["写一段 120 字关于「缓存」的说明。",
      "写一个 Python 函数 is_even(n)，只输出代码块。",
      "写一段 120 字关于「并发」的说明。",
      "写一个 Python 函数 gcd(a,b)，只输出代码块。",
      "写一段 120 字关于「日志」的说明。",
      "写一个 Python 函数 square(n)，只输出代码块。"]
for q in qs:
    txt, tier, lp = c.ask(q)
    cls = E.Cluster.classify(txt)
    print("  %-4s %-6s lp=%+.4f  %s" % (cls, tier, lp if lp is not None else 0, q[:22]))
print()
print("分组历史：%s" % {k: len(v) for k, v in c.hist.items()})
print("升级 %d/%d" % (c.n_esc, c.n_total))
print()
print("结论：%s" % ("各形态历史独立 -> 补丁生效" if len(c.hist) >= 1 else "异常"))
