# -*- coding: utf-8 -*-
"""标准基准实测：HumanEval pass@1 + GSM8K（抽样）。

    python exp/bench_std.py <模型名> <gguf路径> [--gsm N] [额外 llama 参数...]

HumanEval 用官方判分方式：prompt + 模型续写 + 官方 test + check(entry_point)，
在子进程里真跑，10 秒超时，退出码 0 即通过。
GSM8K 抽 N 题（默认 200），比对 #### 后面的数字。
"""
from __future__ import annotations
import gzip, json, os, random, re, subprocess, sys, tarfile, tempfile, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8097
BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3


def load_he():
    with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
        n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
        raw = gzip.decompress(t.extractfile(n).read()).decode()
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def load_gsm(n):
    with tarfile.open(os.path.join(DATA, "gsm.tar.gz")) as t:
        nm = [x for x in t.getnames() if x.endswith("data/test.jsonl")][0]
        rows = [json.loads(l) for l in t.extractfile(nm).read().decode().splitlines() if l.strip()]
    random.Random(2026).shuffle(rows)
    return rows[:n]


def gen(prompt, mx, stop=None):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": 0}
    if stop:
        body["stop"] = stop
    r = requests.post(BASE + "/v1/chat/completions", timeout=600, json=body)
    return r.json()["choices"][0]["message"]["content"] or ""


def strip_fence(txt):
    # 有闭合围栏 -> 取里面；没有（被 stop 截断）-> 至少把开头的围栏标记去掉。
    # 不去掉的话，围栏标记会留在代码第一行 -> 每道题 SyntaxError -> pass@1 假 0。
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
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass


def gsm_number(txt):
    m = re.findall(r"####\s*(-?[\d,\.]+)", txt)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", txt.split("\n")[-1] if txt.strip() else "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main():
    name = sys.argv[1]; model = sys.argv[2]
    rest = sys.argv[3:]
    n_gsm = 200
    if "--gsm" in rest:
        i = rest.index("--gsm"); n_gsm = int(rest[i + 1]); rest = rest[:i] + rest[i + 2:]
    n_he = None
    if "--he" in rest:
        i = rest.index("--he"); n_he = int(rest[i + 1]); rest = rest[:i] + rest[i + 2:]
    extra = rest
    lf = open(os.path.join(OUT, "bench_%s.log" % name), "wb")
    cmd = [SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
           "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
           "--host", "127.0.0.1", "--port", str(PORT), "--alias", name] + extra
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            time.sleep(2)
            try:
                if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                    break
            except Exception:
                pass
        else:
            print("加载失败"); return
        time.sleep(3)
        # ---- HumanEval ----
        he = load_he()
        if n_he:
            he = he[:n_he]
        ok = 0
        t0 = time.perf_counter()
        for i, pr in enumerate(he):
            try:
                # stop 里千万不能放闭合围栏，否则 strip_fence 拿不到代码
                out = gen(pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。", 450,
                          stop=["\nif __name__", "\nprint(", "\ncheck("])
                c = strip_fence(out)
                good = run_he(pr, c)
            except Exception:
                good = False
            ok += int(good)
            if (i + 1) % 20 == 0:
                print("  HumanEval %d/%d 通过 %d (%.1f%%)" % (i + 1, len(he), ok, 100 * ok / (i + 1)))
                sys.stdout.flush()
        he_dt = time.perf_counter() - t0
        print("HumanEval pass@1 = %.3f (%d/%d)  用时 %.0f 秒" % (ok / len(he), ok, len(he), he_dt))
        sys.stdout.flush()
        # ---- GSM8K ----
        gs = load_gsm(n_gsm); gok = 0
        t1 = time.perf_counter()
        for i, r in enumerate(gs):
            try:
                out = gen(r["question"] + "\n\n最后一行用 答案: 数字 的格式给出最终答案。", 600)
                got = gsm_number(out) if "####" in out else (re.findall(r"答案[：:]\s*(-?[\d,\.]+)", out) or [None])[-1]
                gold = r["answer"].split("####")[-1].strip().replace(",", "")
                if got:
                    got = got.replace(",", "").rstrip(".")
                    if re.fullmatch(r"-?\d+", got) and re.fullmatch(r"-?\d+", gold):
                        gok += int(int(got) == int(gold))
                    else:
                        gok += int(float(got) == float(gold))
            except Exception:
                pass
            if (i + 1) % 25 == 0:
                print("  GSM8K %d/%d 正确 %d (%.1f%%)" % (i + 1, len(gs), gok, 100 * gok / (i + 1)))
                sys.stdout.flush()
        gs_dt = time.perf_counter() - t1
        print("GSM8K acc = %.3f (%d/%d)  用时 %.0f 秒" % (gok / len(gs), gok, len(gs), gs_dt))
        json.dump(dict(model=name, he_pass=ok, he_n=len(he), he_acc=ok / len(he), he_secs=he_dt,
                       gsm_ok=gok, gsm_n=len(gs), gsm_acc=gok / len(gs), gsm_secs=gs_dt),
                  open(os.path.join(OUT, "results_bench_%s.json" % name), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close(); time.sleep(2)


if __name__ == "__main__":
    main()
