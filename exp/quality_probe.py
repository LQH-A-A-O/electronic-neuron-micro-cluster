# -*- coding: utf-8 -*-
"""质量探针：代码 / 写作 / 编辑 —— 全部**可机械判分**，不靠主观感觉。

代码   8 道：写函数 -> 抽代码块 -> 真跑断言。判分 = 通过数。
改错   6 道：给定含特定错误的句子，要求改对。判分 = 输出含正确形式且不含错误形式。
写作   4 道：给定结构与要素要求，判分 = 要素齐全 + 长度 + 小标题数（各 1 分）。

用法：python exp/quality_probe.py <模型名> <gguf路径> [额外参数...]
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8099
BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3          # 三个反引号，写成 chr(96)*3 免得被外层模板字符串吃掉

CODE = [
    ("def binary_search(arr, target)", "返回 target 在有序数组 arr 中的下标，找不到返回 -1",
     "assert binary_search([1,3,5,7,9], 5) == 2\nassert binary_search([1,3,5], 4) == -1\nassert binary_search([], 1) == -1"),
    ("def fib(n)", "返回第 n 个斐波那契数（fib(0)=0, fib(1)=1）",
     "assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55"),
    ("def is_palindrome(s)", "判断字符串是否是回文（忽略大小写和空白）",
     "assert is_palindrome('A man a plan a canal Panama')\nassert not is_palindrome('hello')"),
    ("def word_count(text)", "返回字典：每个单词出现次数（按空白切分，转小写）",
     "assert word_count('a b A') == {'a':2,'b':1}"),
    ("def flatten(nested)", "把嵌套列表展平成一维",
     "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]"),
    ("def merge_intervals(iv)", "合并重叠区间，输入 [(1,3),(2,6),(8,10)]",
     "assert merge_intervals([(1,3),(2,6),(8,10)]) == [(1,6),(8,10)]"),
    ("def caesar(s, k)", "凯撒密码，只对字母位移 k 位，保留大小写",
     "assert caesar('AbC', 1) == 'BcD'\nassert caesar('xyz', 3) == 'abc'"),
    ("def rle(s)", "游程编码：'aaabb' -> 'a3b2'",
     "assert rle('aaabb') == 'a3b2'\nassert rle('abc') == 'a1b1c1'"),
]

EDIT = [
    ("这段话有错别字，只输出改正后的句子：这个方案即能省钱，又能提速。", "既", "即能"),
    ("这段话有错别字，只输出改正后的句子：他的成绩在班里名列前矛。", "前茅", "前矛"),
    ("Fix the grammar, output only the corrected sentence: She don't like apples.", "doesn't", "don't"),
    ("Fix the grammar, output only the corrected sentence: I have went to the store yesterday.", "went", "have went"),
    ("这段话有错别字，只输出改正后的句子：我们必须按步就班地推进。", "按部就班", "按步就班"),
    ("Fix the grammar, output only the corrected sentence: There is many reasons for this.", "are", "is many"),
]

WRITE = [
    ("写一段 200 到 260 字的产品介绍，必须包含「本地部署」「隐私」「离线」三个词，并包含恰好 3 个小标题（每个以 ## 开头）。",
     ["本地部署", "隐私", "离线"], (200, 260), 3),
    ("写一段 150 到 200 字的技术说明，必须包含「延迟」「吞吐」「显存」三个词，不要使用小标题。",
     ["延迟", "吞吐", "显存"], (150, 200), 0),
    ("用 120 到 180 字解释什么是向量数据库，必须包含「相似度」和「索引」两个词。",
     ["相似度", "索引"], (120, 180), None),
    ("写一段 200 到 280 字的开源项目 README 开头，必须包含「许可证」「贡献」两个词，并包含恰好 2 个以 ## 开头的小标题。",
     ["许可证", "贡献"], (200, 280), 2),
]


def gen(prompt, mx=900, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    r = requests.post(BASE + "/v1/chat/completions", timeout=900,
                      json={"messages": msgs, "max_tokens": mx, "temperature": 0})
    return r.json()["choices"][0]["message"]["content"] or ""


def extract_code(txt):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, txt, re.S)
    return "\n".join(m) if m else txt


def score_code(txt, name, asserts):
    code = extract_code(txt)
    fname = name.split("(")[0].replace("def ", "").strip()
    if fname not in code:
        return 0, "没找到函数定义"
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return 0, "执行报错 %s" % type(e).__name__
    if fname not in ns:
        return 0, "函数未定义"
    try:
        exec(asserts, ns)
        return 1, "通过"
    except AssertionError:
        return 0, "断言失败"
    except Exception as e:
        return 0, "断言报错 %s" % type(e).__name__


def score_edit(txt, good, bad):
    t = txt.strip()
    ok = (good in t and bad not in t)
    return (1 if ok else 0), (("含正确形式" if good in t else "缺正确形式") +
                              ("" if bad not in t else " / 仍含错误形式"))


def score_write(txt, words, rng, nh):
    miss = [w for w in words if w not in txt]
    n = len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", txt))
    heads = len(re.findall(r"^##\s", txt, re.M))
    s = 0; note = []
    if not miss: s += 1
    else: note.append("缺 %s" % miss)
    if rng[0] <= n <= rng[1]: s += 1
    else: note.append("长度 %d 不在 %s" % (n, rng))
    if nh is None or heads == nh: s += 1
    else: note.append("小标题 %d 应为 %d" % (heads, nh))
    return s, (" ".join(note) or "全对")


def main():
    name = sys.argv[1]; model = sys.argv[2]; extra = sys.argv[3:]
    lf = open(os.path.join(OUT, "probe_%s.log" % name), "wb")
    # Qwen3.5/3.6 默认开思考：不带这个，content 会是空的（正文全进 reasoning_content）。
    # 漏掉它 = 所有输出为空 = 探针全 0。Qwen2.5 没有这个模板参数，带上也无害。
    cmd = [SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
           "--chat-template-kwargs", '{"enable_thinking":false}',
           "--temp", "0", "--host", "127.0.0.1", "--port", str(PORT), "--alias", name] + extra
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            time.sleep(2)
            try:
                if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                    break
            except Exception:
                pass
        else:
            print("模型加载失败"); return
        time.sleep(3)
        res = {"model": name, "code": [], "edit": [], "write": []}
        t0 = time.perf_counter()
        for fn, desc, asserts in CODE:
            try:
                out = gen("用 Python 写一个函数，签名是 %s，%s。只输出一个代码块。" % (fn, desc), 500)
                s, note = score_code(out, fn, asserts)
            except Exception:
                s, note = 0, "调用失败"
            res["code"].append([fn.split("(")[0].replace("def ", ""), s, note])
            print("  [代码] %-16s %s  %s" % (fn[:16], s, note)); sys.stdout.flush()
        for prompt, good, bad in EDIT:
            try:
                out = gen(prompt, 300); s, note = score_edit(out, good, bad)
            except Exception:
                s, note = 0, "调用失败"
            res["edit"].append([prompt[:18], s, note])
            print("  [改错] %-18s %s  %s" % (prompt[:16], s, note)); sys.stdout.flush()
        for prompt, words, rng, nh in WRITE:
            try:
                out = gen(prompt, 900); s, note = score_write(out, words, rng, nh)
            except Exception:
                s, note = 0, "调用失败"
            res["write"].append([prompt[:18], s, note])
            print("  [写作] %-18s %s  %s" % (prompt[:16], s, note)); sys.stdout.flush()
        dt = time.perf_counter() - t0
        c = sum(x[1] for x in res["code"]); e = sum(x[1] for x in res["edit"])
        w = float(sum(x[1] for x in res["write"]))
        res.update(code_score=c, code_n=len(CODE), edit_score=e, edit_n=len(EDIT),
                   write_score=w, write_n=len(WRITE) * 3, secs=dt)
        tot_n = len(CODE) + len(EDIT) + len(WRITE) * 3
        res["total"] = (c + e + w) / tot_n
        print()
        print("=== %s ===" % name)
        print("  代码 %d/%d   改错 %d/%d   写作 %.0f/%d   耗时 %.0f 秒   归一化总分 %.3f"
              % (c, len(CODE), e, len(EDIT), w, len(WRITE) * 3, dt, res["total"]))
        json.dump(res, open(os.path.join(OUT, "results_probe_%s.json" % name), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)


if __name__ == "__main__":
    main()
