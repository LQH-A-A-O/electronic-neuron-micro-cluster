# -*- coding: utf-8 -*-
"""级联实验：先 1.5B 简易推理，用「多次采样互证」决定是否升级到 35B-A3B 深推理。

这是把用户的想法落地：
    「不直接激活参数，而是依靠神经元架构的对比决定激活哪些参数，
      先进行简易推理，随后进行长思维链推理。」

映射关系：mem-neuron 的核心是「多证据互证」（NeighborValidator 的邻居投票），
这里把「证据」换成**便宜模型自己采出来的多个答案** —— 它们互相印证就不升级，票数分散就升级。

平台限制：1.5B 与 35B-A3B 不能同时驻留，所以两趟跑：
    阶段1  1.5B 对每道题采 K=3 个解 -> 存盘，并算互证度
    阶段2  MoE 只对「互证失败」的题生成 -> 与阶段1 合并 -> 统一判分

    python exp/cascade.py phase1 <N>      # N 道 HumanEval 题
    python exp/cascade.py phase2

指标：最终 pass@1 与「平均每题的贵模型调用比例」。
"""
from __future__ import annotations
import gzip, json, os, re, subprocess, sys, tarfile, tempfile, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8095; BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3
K = 3
SMALL = r"E:\aqca\work\llm\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
BIG = r"E:\aqca\work\llm\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
SMALL_ARGS = ["-ngl", "99"]
BIG_ARGS = ["-ngl", "99", "-cmoe"]
CACHE = os.path.join(OUT, "cascade_samples.json")


def load_he():
    with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
        n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
        return [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]


def start(model, extra):
    lf = open(os.path.join(OUT, "cascade_srv.log"), "ab")
    p = subprocess.Popen([SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(PORT)] + extra,
                         stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(2)
        try:
            if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                time.sleep(2); return p, lf
        except Exception:
            pass
    p.terminate(); lf.close(); raise SystemExit("加载失败")


def gen(prompt, mx=450, temp=0.9):
    r = requests.post(BASE + "/v1/chat/completions", timeout=900, json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": temp})
    return r.json()["choices"][0]["message"]["content"] or ""


def strip_fence(txt):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, txt, re.S)
    if m:
        return m[0]
    return re.sub(r"^\s*" + FENCE + r"(?:python)?\s*\n", "", txt)


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


def norm_code(s):
    """粗规范化：去掉空白与注释，用来判"两个采样是不是同一个解"。"""
    s = re.sub(r"#.*", "", s)
    return re.sub(r"\s+", "", s)


def agreement(samples):
    """互证度 = 出现次数最多的那个解占的比例。1.0 = 全部一致。"""
    if not samples:
        return 0.0
    c = {}
    for s in samples:
        c[norm_code(s)] = c.get(norm_code(s), 0) + 1
    return max(c.values()) / len(samples)


def phase1(n):
    he = load_he()[:n]
    p, lf = start(SMALL, SMALL_ARGS)
    try:
        rows = []
        t0 = time.perf_counter()
        for i, pr in enumerate(he):
            prompt = pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"
            samples = []
            for _ in range(K):
                try:
                    samples.append(strip_fence(gen(prompt, 450, 0.9)))
                except Exception:
                    samples.append("")
            ag = agreement(samples)
            rows.append(dict(idx=i, samples=samples, agree=ag,
                             base_pass=int(run_he(pr, samples[0])),
                             any_pass=int(any(run_he(pr, s) for s in samples)),
                             esc=0, big=None))
            if (i + 1) % 10 == 0:
                print("  阶段1 %d/%d  平均互证度 %.3f  单采样通过 %d/%d"
                      % (i + 1, n, sum(r["agree"] for r in rows) / len(rows),
                         sum(r["base_pass"] for r in rows), i + 1))
                sys.stdout.flush()
        json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        print("阶段1 完成，用时 %.0f 秒 -> %s" % (time.perf_counter() - t0, CACHE))
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)


def phase2(thr=1.0):
    rows = json.load(open(CACHE, encoding="utf-8"))
    he = load_he()[:len(rows)]
    esc = [r for r in rows if r["agree"] < thr]
    print("要升级的题：%d / %d（互证度 < %.2f）" % (len(esc), len(rows), thr))
    p, lf = start(BIG, BIG_ARGS)
    try:
        for j, r in enumerate(esc):
            pr = he[r["idx"]]
            try:
                r["big"] = strip_fence(gen(pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。", 450, 0.0))
                r["esc"] = 1
            except Exception:
                r["big"] = None
            if (j + 1) % 5 == 0:
                print("  阶段2 %d/%d" % (j + 1, len(esc))); sys.stdout.flush()
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)
    json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)

    n = len(rows)
    base = sum(r["base_pass"] for r in rows) / n
    anyp = sum(r["any_pass"] for r in rows) / n
    casc = sum((run_he(he[r["idx"]], r["big"]) if r["big"] else r["base_pass"]) for r in rows) / n
    esc_frac = len(esc) / n
    print()
    print("=" * 76)
    print("级联结果（HumanEval 前 %d 题，K=%d 采样互证，阈值 %.2f）" % (n, K, thr))
    print("=" * 76)
    print("  只跑 1.5B（单采样）      pass@1 = %.3f   贵模型调用 0%%" % base)
    print("  只跑 1.5B（K 个里任一过） pass@1 = %.3f   贵模型调用 0%%" % anyp)
    print("  级联（互证失败才升级）    pass@1 = %.3f   贵模型调用 %.0f%%" % (casc, 100 * esc_frac))
    print("  -> 相对单采样增益 %+.4f，代价是 %.0f%% 的题要走慢模型" % (casc - base, 100 * esc_frac))
    json.dump(dict(n=n, base=base, anyp=anyp, casc=casc, esc_frac=esc_frac, thr=thr, K=K),
              open(os.path.join(OUT, "results_cascade.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "phase1":
        phase1(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    else:
        phase2(float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
