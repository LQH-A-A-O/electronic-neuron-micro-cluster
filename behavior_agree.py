# -*- coding: utf-8 -*-
"""行为互证 vs 文本互证：同一批缓存样本，换一种"一致"的定义。

行为互证：从官方 test 里抽出 candidate(...) 的**输入**（不取期望值），
让 K 个解在同一批输入上跑，比较输出是否一致。
—— 这是"两个解在同样输入上给出同样结果"，不是"它们是否正确"。
"""
import ast, gzip, json, os, re, sys, tarfile, traceback
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
    n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
    he = [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]
rows = json.load(open(os.path.join(OUT, "ladder_samples.json"), encoding="utf-8"))


def extract_inputs(prob):
    """从 test 里抽出 candidate 调用的实参（不求值期望答案）。"""
    tree = ast.parse(prob["test"])
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            l = node.test.left
            if isinstance(l, ast.Call) and getattr(l.func, "id", "") == "candidate":
                out.append([ast.unparse(a) for a in l.args])
    return out


def behavior(prob, code):
    """跑一个解，返回它在这批输入上的输出元组（异常记为异常类型）。"""
    full = code if prob["prompt"].strip().split("\n")[0] in code else prob["prompt"] + code
    ns = {}
    try:
        exec(full, ns)
    except Exception:
        return ("EXEC_FAIL",)
    fn = ns.get(prob["entry_point"])
    if fn is None:
        return ("NO_FN",)
    res = []
    for args in extract_inputs(prob):
        try:
            vals = [eval(a, dict(ns)) for a in args]
            r = fn(*vals)
            res.append(repr(r))
        except Exception as e:
            res.append("EX:" + type(e).__name__)
    return tuple(res)


def agree_beh(behavs):
    if not behavs:
        return 0.0
    c = {}
    for b in behavs:
        c[b] = c.get(b, 0) + 1
    return max(c.values()) / len(behavs)


def agree_txt(samples):
    c = {}
    for s in samples:
        k = re.sub(r"\s+", "", re.sub(r"#.*", "", s))
        c[k] = c.get(k, 0) + 1
    return max(c.values()) / len(samples)


n = len(rows)
txt = [agree_txt(r["samples"]) for r in rows]
beh = []
for r in rows:
    try:
        beh.append(agree_beh([behavior(he[r["idx"]], s) for s in r["samples"]]))
    except Exception:
        beh.append(0.0)

print("同一批 %d 道题、K=3 采样，两种互证判据的对照" % n)
print()
print("%-22s %-14s %-14s %s" % ("判据", "平均互证度", "升级率(<0.34)", "全一致比例"))
for name, v in (("文本完全相等", txt), ("行为一致", beh)):
    esc = sum(1 for x in v if x < 0.34) / n
    same = sum(1 for x in v if x >= 0.999) / n
    print("%-22s %-14.3f %-14.0f%% %.0f%%" % (name, sum(v) / n, 100 * esc, 100 * same))
print()
print("说明：行为互证若把升级率显著压低，就说明文本判据在误伤")
print("     （两个解写法不同但行为一样，文本判据会判它们不一致 -> 白升级）。")
json.dump(dict(txt=txt, beh=beh), open(os.path.join(OUT, "results_behavior.json"), "w"), indent=1)
