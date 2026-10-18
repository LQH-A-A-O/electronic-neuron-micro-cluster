# -*- coding: utf-8 -*-
"""把 logprob 级联的 0.75 从"推算"变成"实测"。

做法：取 3B 的 mean_logprob 最低的 20% 题，让 9B 跑这 33 题，测它在这批上的真实通过率。
然后算级联的真实 pass@1 与延迟。
"""
import gzip, json, os, re, subprocess, sys, tarfile, tempfile, time
import numpy as np
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out"); DATA = os.path.join(ROOT, "data", "bench")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
PORT = 8087; BASE = "http://127.0.0.1:%d" % PORT
FENCE = chr(96) * 3
BIG = r"E:\aqca\work\llm\models\Qwen3.5-9B-Q4_K_M.gguf"

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

if __name__ == "__main__":
    he = load_he()
    rows = [r for r in json.load(open(os.path.join(OUT, "logprob_conf.json"), encoding="utf-8"))
            if r["mean_lp"] is not None]
    ml = np.array([r["mean_lp"] for r in rows])
    thr = float(np.quantile(ml, 0.20))
    esc = [r for r in rows if r["mean_lp"] <= thr]
    print("升级子集：%d/%d 题（mean_lp <= %.4f）" % (len(esc), len(rows), thr))
    print("这批在 3B 上的通过率 = %.3f" % (sum(r["ok"] for r in esc) / len(esc)))
    lf = open(os.path.join(OUT, "esc9b_srv.log"), "wb")
    p = subprocess.Popen([SERVER, "-m", BIG, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            time.sleep(2)
            try:
                if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                    time.sleep(2); break
            except Exception:
                pass
        ok9 = 0
        for j, r in enumerate(esc):
            try:
                resp = requests.post(BASE + "/v1/chat/completions", timeout=600, json={
                    "messages": [{"role": "user", "content": he[r["idx"]]["prompt"] + "\n\n只补全这个函数体，不要任何解释。"}],
                    "max_tokens": 450, "temperature": 0}).json()
                code = strip_fence(resp["choices"][0]["message"]["content"] or "")
                r["ok9"] = int(run_he(he[r["idx"]], code))
            except Exception:
                r["ok9"] = 0
            ok9 += r["ok9"]
            if (j + 1) % 10 == 0:
                print("  %d/%d  9B 通过 %d (%.1f%%)" % (j + 1, len(esc), ok9, 100 * ok9 / (j + 1)))
                sys.stdout.flush()
        json.dump(rows, open(os.path.join(OUT, "logprob_conf.json"), "w", encoding="utf-8"), ensure_ascii=False)
        n = len(rows)
        base = sum(r["ok"] for r in rows) / n
        casc = sum((r.get("ok9", 0) if r["mean_lp"] <= thr else r["ok"]) for r in rows) / n
        ef = len(esc) / n
        lat_c = 450 / 66.0 + ef * 450 / 25.0
        print()
        print("=" * 70)
        print("logprob 级联：实测（HumanEval 164 题）")
        print("=" * 70)
        print("  3B 单次            pass@1 = %.3f   延迟 ≈ %.1f 秒/题" % (base, 450 / 66.0))
        print("  升级子集上 3B       pass@1 = %.3f   (%d 题)" % (sum(r["ok"] for r in esc) / len(esc), len(esc)))
        print("  升级子集上 9B       pass@1 = %.3f" % (ok9 / len(esc)))
        print("  级联（实测）        pass@1 = %.3f   延迟 ≈ %.1f 秒/题  升级 %.0f%%" % (casc, lat_c, 100 * ef))
        print("  9B 直接跑（实测）   pass@1 = 0.768   延迟 ≈ %.1f 秒/题" % (450 / 25.0))
        json.dump(dict(base=base, casc=casc, esc_frac=ef, thr=thr,
                       esc_3b=sum(r["ok"] for r in esc) / len(esc), esc_9b=ok9 / len(esc),
                       lat_casc=lat_c), open(os.path.join(OUT, "results_esc9b.json"), "w"), indent=1)
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        lf.close()
