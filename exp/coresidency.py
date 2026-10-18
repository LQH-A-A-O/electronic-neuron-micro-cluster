# -*- coding: utf-8 -*-
"""两个模型能否同时驻留？—— 能的话级联就没有换模型的开销。

3B Q4 (~2.0 GiB) + 9B IQ3_XXS (~3.7 GiB) = 5.7 GiB + KV，理论上装得进 8 GB。
"""
import os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
A = ("3B-Q4", r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf", 8077)
B = ("9B-IQ3", r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf", 8076)

def launch(name, model, port):
    lf = open(os.path.join(ROOT, "out", "co_%s.log" % name), "wb")
    p = subprocess.Popen([SERVER, "-m", model, "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    t0 = time.perf_counter()
    for _ in range(300):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                return p, time.perf_counter() - t0, lf
        except Exception:
            pass
    return p, None, lf

if __name__ == "__main__":
    procs = []
    try:
        for nm, m, pt in (A, B):
            p, dt, lf = launch(nm, m, pt)
            procs.append((p, lf))
            print("  %-7s 加载 %s 秒" % (nm, ("%.1f" % dt) if dt else "失败"))
            sys.stdout.flush()
        v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
        print("  显存（两个都在跑）：%s" % v)
        ok = 0
        for nm, m, pt in (A, B):
            try:
                r = requests.post("http://127.0.0.1:%d/v1/chat/completions" % pt, timeout=300,
                                  json={"messages": [{"role": "user", "content": "1+1=?"}],
                                        "max_tokens": 8, "temperature": 0}).json()
                ans = (r["choices"][0]["message"]["content"] or "").strip()
                print("  %-7s 可用 -> %r" % (nm, ans[:20]))
                ok += 1
            except Exception as e:
                print("  %-7s 不可用 %s" % (nm, str(e)[:50]))
        print()
        print("结论：%s" % ("两个模型可同时驻留，级联无需换模型" if ok == 2 else "装不下或冲突"))
    finally:
        for p, lf in procs:
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
