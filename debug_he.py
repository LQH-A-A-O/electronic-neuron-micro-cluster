# -*- coding: utf-8 -*-
import gzip, json, os, subprocess, sys, tarfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bench")
with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
    n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
    he = [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]
p = he[0]
print("=== 题目 prompt 尾部 ===")
print(repr(p["prompt"][-260:]))
print("=== entry_point:", p["entry_point"])
print("=== test 前 200 字 ===")
print(p["test"][:200])
# 用之前跑分器的 prompt 让模型续写（直接调已在跑的服务不行，服务已关；这里只做静态检查）
print()
print("=== 关键：我们的拼接逻辑 ===")
first = p["prompt"].strip().split("\n")[0]
print("prompt 首行:", repr(first))
print("判断: 模型若只输出函数体，首行不在输出里 -> 会前置 prompt")
