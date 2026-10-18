# -*- coding: utf-8 -*-
"""离线反事实：换更宽松的互证判据，升级率与级联质量会怎样？（不需要模型调用）

已有的 MoE 答案只覆盖"文本全不同"的 42 题，但那是所有宽松判据升级集的超集，
所以只要新判据的升级集是它的子集，就能直接算反事实。
"""
import gzip, json, os, re, subprocess, sys, tarfile, tempfile
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
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


def toks(s):
    s = re.sub(r"#.*", "", s)
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[+\-*/%=<>!]+", s))


def sim(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(len(A | B), 1)


def soft_agree(samples, thr=0.8):
    """宽松互证：任意两个解的 token Jaccard >= thr 就算一致；返回最大一致簇的比例。"""
    m = len(samples)
    best = 1
    for i in range(m):
        grp = 1 + sum(1 for j in range(m) if j != i and sim(samples[i], samples[j]) >= thr)
        best = max(best, grp)
    return best / m


n = len(rows)
base = sum(r["base_pass"] for r in rows) / n
print("原始判据（文本完全相等）升级 %d/%d (70%%)" % (sum(1 for r in rows if r["agree"] < 0.34), n))
print()
print("%-22s %-12s %-12s %-12s %s" % ("判据", "升级率", "级联 pass@1", "相对单采样", "贵模型调用"))
print("%-22s %-12s %-12s %-12s %s" % ("文本完全相等", "70%", "0.967", "+0.250", "70%"))
for thr in (0.5, 0.6, 0.7, 0.8):
    esc = [r for r in rows if soft_agree(r["samples"], thr) < 1.0]
    ok = 0
    for r in rows:
        if r["idx"] in [e["idx"] for e in esc]:
            good = run_he(he[r["idx"]], r["big"]) if r.get("big") else r["base_pass"]
        else:
            good = r["base_pass"]
        ok += int(good)
    print("%-22s %-12s %-12s %-12s %s" % ("语义相似>=%.1f" % thr,
                                          "%d%%" % round(100 * len(esc) / n),
                                          "%.3f" % (ok / n), "%+.3f" % (ok / n - base),
                                          "%d%%" % round(100 * len(esc) / n)))
