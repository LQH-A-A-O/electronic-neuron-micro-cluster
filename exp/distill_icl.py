# -*- coding: utf-8 -*-
"""上下文级蒸馏：把 9B 的解答当 few-shot 教学样本喂给 3B，能不能把差距补上？

权重级蒸馏做不了（没有 transformers/peft，torch 是 CPU 版），
但"把大模型的知识塞进上下文"是可以做的，而且不需要训练。

三组（同一批测试题）：
    A  3B 零样本
    B  3B + 8 个 9B 解答当 few-shot        <- 蒸馏组
    C  9B 零样本（上界）
"""
import gzip, json, os, re, subprocess, sys, tarfile, tempfile, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
FENCE = chr(96) * 3
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"
N_SHOT = 8
N_TEST = 40
PORT3, PORT9 = 8070, 8069


def load_he():
    with tarfile.open(os.path.join(DATA, "he.tar.gz")) as t:
        n = [x for x in t.getnames() if x.endswith("HumanEval.jsonl.gz")][0]
        return [json.loads(l) for l in gzip.decompress(t.extractfile(n).read()).decode().splitlines() if l.strip()]


def strip_fence(t):
    m = re.findall(FENCE + r"(?:python)?\s*\n(.*?)" + FENCE, t, re.S)
    return m[0] if m else re.sub(r"^\s*" + FENCE + r"(?:python)?\s*\n", "", t)


def run_he(prob, code):
    c = code if prob["prompt"].strip().split("\n")[0] in code else prob["prompt"] + code
    prog = c + "\n\n" + prob["test"] + "\n\ncheck(%s)\n" % prob["entry_point"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass


def launch(model, port):
    lf = open(os.path.join(OUT, "distill_%d.log" % port), "wb")
    p = subprocess.Popen([SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(1)
        try:
            if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                time.sleep(2); return p, lf
        except Exception:
            pass
    p.terminate(); lf.close(); raise SystemExit("load failed %d" % port)


def gen(port, prompt, mx=450):
    r = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=900, json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": 0})
    return r.json()["choices"][0]["message"]["content"] or ""


if __name__ == "__main__":
    he = load_he()
    shots = he[:N_SHOT]
    tests = he[N_SHOT:N_SHOT + N_TEST]
    p9, lf9 = launch(M9, PORT9)
    try:
        print("收 9B 的教学样本（%d 道）..." % N_SHOT)
        teach = []
        for pr in shots:
            code = strip_fence(gen(PORT9, pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"))
            teach.append((pr["prompt"], code))
            sys.stdout.flush()
        print("  收完")
        print("收 9B 在测试集上的上界（%d 道）..." % N_TEST)
        ok9 = 0
        for i, pr in enumerate(tests):
            ok9 += int(run_he(pr, strip_fence(gen(PORT9, pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"))))
            if (i + 1) % 10 == 0:
                print("  %d/%d 9B 通过 %d" % (i + 1, N_TEST, ok9)); sys.stdout.flush()
        print("  9B 上界 = %.3f" % (ok9 / N_TEST))
    finally:
        p9.terminate()
        try: p9.wait(timeout=20)
        except Exception: p9.kill()
        lf9.close()
        time.sleep(3)
    # few-shot 前缀
    prefix = "以下是几个 Python 函数补全的示例（问题 -> 完整函数）：\n\n"
    for q, a in teach:
        prefix += "--- 问题 ---\n" + q + "\n--- 答案 ---\n" + FENCE + "python\n" + a + "\n" + FENCE + "\n\n"
    prefix += "现在请按同样风格补全下面这个函数，只输出一个代码块。\n\n"

    p3, lf3 = launch(M3, PORT3)
    try:
        okA = okB = 0
        t0 = time.perf_counter()
        for i, pr in enumerate(tests):
            q = pr["prompt"] + "\n\n只补全这个函数体，不要任何解释。"
            okA += int(run_he(pr, strip_fence(gen(PORT3, q))))
            okB += int(run_he(pr, strip_fence(gen(PORT3, prefix + q))))
            if (i + 1) % 10 == 0:
                print("  %d/%d  3B零样本 %d  3B+蒸馏 %d" % (i + 1, N_TEST, okA, okB)); sys.stdout.flush()
        print()
        print("=" * 70)
        print("上下文级蒸馏（%d 道测试题，%d 个 9B 教学样本）" % (N_TEST, N_SHOT))
        print("=" * 70)
        print("  A  3B 零样本           %.3f" % (okA / N_TEST))
        print("  B  3B + 9B 教学样本    %.3f   （相对 A %+.3f）" % (okB / N_TEST, (okB - okA) / N_TEST))
        print("  C  9B 零样本（上界）    %.3f" % (ok9 / N_TEST))
        print("  用时 %.0f 秒" % (time.perf_counter() - t0))
        json.dump(dict(a=okA / N_TEST, b=okB / N_TEST, c=ok9 / N_TEST, n=N_TEST, shot=N_SHOT),
                  open(os.path.join(OUT, "results_distill.json"), "w", encoding="utf-8"), indent=1)
    finally:
        p3.terminate()
        try: p3.wait(timeout=20)
        except Exception: p3.kill()
        lf3.close()
