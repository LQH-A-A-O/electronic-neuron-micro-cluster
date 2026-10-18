# -*- coding: utf-8 -*-
"""零 GPU 验证：精确匹配 vs 模糊检索（字符 n-gram），在"同义改写"查询上的召回率。

这是一个弱代理：n-gram 只能捕捉字面相似，抓不到真正语义。
它回答的是"模糊检索能不能救回精确匹配漏掉的查询"，不是"向量检索能提高多少"。
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.abspath(__file__))

# 神经元库：key = 实体名，value = 编号
ENTS = [("玄枢-%03d" % i, 1000 + i * 37) for i in range(20)]

def grams(s, n=2):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}

def jacc(a, b):
    A, B = grams(a), grams(b)
    return len(A & B) / len(A | B) if A | B else 0.0

# 查询：一部分用原名（精确可命中），一部分是同义/改写（精确必漏）
QUERIES = []
for name, code in ENTS[:10]:
    QUERIES.append((name, name, True))                       # 直接问原名
for name, code in ENTS[10:20]:
    QUERIES.append(("请问「%s」这个实体的编号是多少" % name, name, True))
# 改写型：只给编号反过来问、或只给部分字
for name, code in ENTS[:10]:
    dist = name.split("-")[0][0]                             # 只取首字
    QUERIES.append(("那个叫%s什么的开头实体，编号呢" % dist, name, False))

def exact(q):
    for name, _ in ENTS:
        if name in q:
            return name
    return None

def fuzzy(q, thr=0.45):
    best, bs = None, 0.0
    for name, _ in ENTS:
        s = jacc(q, name)
        if s > bs:
            best, bs = name, s
    return best if bs >= thr else None

if __name__ == "__main__":
    hit_e = hit_f = 0
    rows = []
    for q, gold, is_orig in QUERIES:
        e = exact(q); f = fuzzy(q)
        hit_e += int(e == gold); hit_f += int(f == gold)
        rows.append(dict(q=q, gold=gold, orig=is_orig, exact=e, fuzzy=f,
                         ok_e=int(e == gold), ok_f=int(f == gold)))
    n = len(rows)
    print("查询 %d 条（原名/标准问法 %d，改写型 %d）"
          % (n, sum(1 for r in rows if r["orig"]), sum(1 for r in rows if not r["orig"])))
    print()
    print("  %-8s 总体召回 %.2f" % ("精确", hit_e / n))
    print("  %-8s 总体召回 %.2f" % ("模糊", hit_f / n))
    print()
    for tag, lab in ((True, "标准问法"), (False, "改写型")):
        sub = [r for r in rows if r["orig"] == tag]
        if not sub: continue
        print("  %-6s (%d 条)  精确 %.2f   模糊 %.2f"
              % (lab, len(sub), sum(r["ok_e"] for r in sub) / len(sub),
                 sum(r["ok_f"] for r in sub) / len(sub)))
    print()
    miss = [r for r in rows if not r["ok_e"] and r["ok_f"]]
    print("  模糊救回了 %d 条精确漏掉的：%s" % (len(miss), [r["q"][:26] for r in miss[:3]]))
    json.dump(rows, open(os.path.join(ROOT, "out", "results_retrieval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
