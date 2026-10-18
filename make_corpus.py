# -*- coding: utf-8 -*-
"""造一个语料文件：300 实体 x 5 条记录，40% 污染，带 truth 列。"""
import json, os, random
ROOT = os.path.dirname(os.path.abspath(__file__))
rnd = random.Random(20261017)
NAMES = ["玄枢", "流萤", "苍梧", "砚台", "寒汀", "拾光", "白露", "沉舟", "青釉", "归墟"]
rows = []
for i in range(300):
    name = "%s-%03d" % (NAMES[i % len(NAMES)], i)
    code = rnd.randrange(1000, 9999)
    vals = [code] * 5
    for j in rnd.sample(range(5), 2):
        vals[j] = rnd.randrange(1000, 9999)
    rnd.shuffle(vals)
    for v in vals:
        rows.append(dict(key=name, value=v, truth=code))
rnd.shuffle(rows)
p = os.path.join(ROOT, "data", "corpus_demo.jsonl")
with open(p, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
nbad = sum(1 for r in rows if r["value"] != r["truth"])
print("写出 %s：%d 条，其中脏 %d 条（%.0f%%）" % (p, len(rows), nbad, 100 * nbad / len(rows)))
