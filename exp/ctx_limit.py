# -*- coding: utf-8 -*-
"""上下文长度上限实测：从 8k 往上加，找到真的能跑得动的最大上下文。

1M 上下文对这台机器是物理不可能（KV 就要 32~64 GB），
但真实上限在哪，必须实测。
"""
import os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
MODELS = [("3B-Q4", r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf", 8068),
          ("9B-IQ3", r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf", 8067)]
CTXS = [8192, 16384, 32768, 65536]
FILL = "这是一段用于测试上下文长度的填充文本，它不携带任何需要记住的信息。\n"


def vram():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def try_ctx(name, model, port, ctx):
    lf = open(os.path.join(ROOT, "out", "ctx_%s_%d.log" % (name, ctx)), "wb")
    p = subprocess.Popen([SERVER, "-m", model, "-c", str(ctx), "-np", "1", "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    try:
        up = False
        for _ in range(240):
            time.sleep(0.5)
            try:
                if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                    up = True; break
            except Exception:
                pass
            if p.poll() is not None:
                break
        if not up:
            return dict(ctx=ctx, ok=False, why="启动失败/加载失败", free=vram())
        # 塞一个接近上限的提示（留 512 token 给生成）
        n_char = int((ctx * 0.75) * 1.5)    # 目标 ~75% 上下文；中文约 1.5 字符/token
        body = ("请阅读下面材料，然后只回答：读完了吗？\n\n" + FILL * (n_char // len(FILL)))
        t0 = time.perf_counter()
        try:
            r = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=900, json={
                "messages": [{"role": "user", "content": body}], "max_tokens": 32, "temperature": 0})
            d = r.json()
            if "choices" not in d:
                return dict(ctx=ctx, ok=False, why=str(d)[:80], free=vram())
            u = d.get("usage", {})
            return dict(ctx=ctx, ok=True, pt=u.get("prompt_tokens", 0),
                        sec=time.perf_counter() - t0, free=vram())
        except Exception as e:
            return dict(ctx=ctx, ok=False, why=("超时/异常 " + str(e)[:60]), free=vram())
    finally:
        p.terminate()
        try: p.wait(timeout=25)
        except Exception: p.kill()
        lf.close()
        time.sleep(4)


if __name__ == "__main__":
    for name, model, port in MODELS:
        print("=== %s ===" % name)
        for ctx in CTXS:
            r = try_ctx(name, model, port, ctx)
            if r["ok"]:
                print("  -c %-6d  OK   提示 %6d token  %.0f 秒   空闲显存 %s MiB"
                      % (ctx, r["pt"], r["sec"], r["free"]))
            else:
                print("  -c %-6d  失败：%s   空闲显存 %s MiB" % (ctx, r["why"], r["free"]))
                break
            sys.stdout.flush()
