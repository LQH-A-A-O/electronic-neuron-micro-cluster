# -*- coding: utf-8 -*-
"""MoE 的 tok/s 随"显存里放多少专家"的曲线。

-ncmoe N = 把前 N 层的专家权重留在 CPU，其余层放显存。N 越小 -> 显存里专家越多 -> 越快（如果装得下）。
"""
import json, os, re, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
MODEL = r"E:\aqca\work\llm\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
PORT = 8093; BASE = "http://127.0.0.1:%d" % PORT

def run(ncmoe):
    tag = "cmo" if ncmoe < 0 else ("n%d" % ncmoe)
    lf = open(os.path.join(OUT, "sweep_%s.log" % tag), "wb")
    args = ["-ngl", "99"] + (["-cmoe"] if ncmoe < 0 else ["-ncmoe", str(ncmoe)])
    cmd = [SERVER, "-m", MODEL, "-c", "4096", "-np", "1", "-fa", "on", "--jinja",
           "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
           "--host", "127.0.0.1", "--port", str(PORT)] + args
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    try:
        ok = False
        for _ in range(180):
            time.sleep(2)
            try:
                if requests.get(BASE + "/health", timeout=3).json().get("status") == "ok":
                    ok = True; break
            except Exception:
                pass
            if p.poll() is not None:
                break
        if not ok:
            return dict(config=tag, ok=False)
        time.sleep(2)
        t0 = time.perf_counter()
        r = requests.post(BASE + "/v1/chat/completions", timeout=900, json={
            "messages": [{"role": "user", "content": "用一句话解释什么是梯度下降。"}],
            "max_tokens": 120, "temperature": 0})
        dt = time.perf_counter() - t0
        n = r.json().get("usage", {}).get("completion_tokens", 0)
        v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        return dict(config=tag, ok=True, toks=n, secs=dt, tps=n / dt if dt else 0, vram_mb=int(v))
    finally:
        p.terminate()
        try: p.wait(timeout=30)
        except Exception: p.kill()
        lf.close(); time.sleep(3)

if __name__ == "__main__":
    res = []
    for n in [int(x) for x in sys.argv[1:]]:
        r = run(n); res.append(r)
        if r.get("ok"):
            print("  %-6s 显存 %5d MiB  %5.1f tok/s" % (r["config"], r["vram_mb"], r["tps"]))
        else:
            print("  %-6s 装不下 / 加载失败" % r["config"])
        sys.stdout.flush()
    json.dump(res, open(os.path.join(OUT, "results_moe_vram.json"), "w", encoding="utf-8"), indent=1)
