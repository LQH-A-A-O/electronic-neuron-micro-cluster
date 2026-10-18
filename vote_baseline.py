# -*- coding: utf-8 -*-
"""补一个"不用 oracle"的公平基线：K 个采样里做多数投票（不偷看正确答案）。"""
import gzip, json, os, re, subprocess, sys, tarfile, tempfile
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
FENCE = chr(96) * 3
with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
    n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
    he = [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]
rows = json.load(open(os.path.join(OUT, "cascade_samples.json"), encoding="utf-8"))

def run_he(prob, completion):
    code = completion
    if prob["prompt"].strip().split("\n")[0] not in code:
        code = prob["prompt"] + code
    prog = code + "\n\n" + prob["test"] + "\n\ncheck(%s)\n" % prob["entry_point"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass

def norm(s):
    s = re.sub(r"#.*", "", s)
    return re.sub(r"\s+", "", s)

n = len(rows)
vote_ok = 0; vote_tie = 0
for r in rows:
    cnt = {}
    for s in r["samples"]:
        cnt.setdefault(norm(s), []).append(s)
    best = max(cnt.values(), key=len)
    if len(best) < 2: vote_tie += 1
    vote_ok += int(run_he(he[r["idx"]], best[0]))
print("不用 oracle 的公平基线（多数投票，K=3）")
print("  pass@1 = %.3f (%d/%d)" % (vote_ok / n, vote_ok, n))
print("  其中 %d/%d 道题三个采样互不相同、投票等于随便取一个" % (vote_tie, n))
print()
print("对照（全部实测）：")
print("  1.5B 单采样                     0.717   贵模型 0%%")
print("  1.5B 多数投票(K=3, 无 oracle)   %.3f   贵模型 0%%" % (vote_ok / n))
print("  1.5B best-of-3 (有 oracle 选)   0.867   <- 不可实现，只能当上界")
print("  级联(互证失败升级 MoE)          0.967   贵模型 70%%")
