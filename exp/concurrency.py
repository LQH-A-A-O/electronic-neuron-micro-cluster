# -*- coding: utf-8 -*-
"""并发压测：
A) -np 1（单槽）下并发请求会不会报错/排队多久
B) -np 2 / -np 4 能不能起得来（KV 变大，468 MiB 余量可能撑不住）
"""
import concurrent.futures as cf, os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
M9 = r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf"

def vram():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().replace("\n", " | ")

def launch(name, model, port, np_, ctx=8192):
    lf = open(os.path.join(ROOT, "out", "conc_%s.log" % name), "wb")
    p = subprocess.Popen([SERVER, "-m", model, "-c", str(ctx), "-np", str(np_), "-fa", "on", "--jinja",
                          "--chat-template-kwargs", '{"enable_thinking":false}', "--temp", "0",
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"],
                         stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(400):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                return p, lf, True
        except Exception:
            pass
        if p.poll() is not None:
            break
    return p, lf, False

def one(port, i):
    t0 = time.perf_counter()
    try:
        r = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600,
                          json={"messages": [{"role": "user", "content": "用一句话说明第 %d 个数是奇数还是偶数。%d" % (i, i)}],
                                "max_tokens": 40, "temperature": 0})
        d = r.json()
        n = d.get("usage", {}).get("completion_tokens", 0) if "choices" in d else -1
        return (n >= 0, time.perf_counter() - t0, n)
    except Exception:
        return (False, time.perf_counter() - t0, -1)

if __name__ == "__main__":
    # A) 单槽 + 8 并发
    print("=== A) -np 1 下 8 个并发请求 ===")
    procs = []
    try:
        p, lf, ok = launch("A", M3, 8073, 1); procs.append((p, lf))
        print("  启动 %s，显存 %s" % ("OK" if ok else "失败", vram()))
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(8) as ex:
            rs = list(ex.map(lambda i: one(8073, i), range(8)))
        wall = time.perf_counter() - t0
        good = [r for r in rs if r[0]]
        print("  成功 %d/8   墙钟 %.1f 秒   平均单请求 %.1f 秒   总生成 %d token"
              % (len(good), wall, sum(r[1] for r in good) / max(len(good), 1), sum(r[2] for r in good)))
        print("  -> 串行吞吐折算 %.1f tok/s（单请求约 %.1f tok/s）"
              % (sum(r[2] for r in good) / wall, sum(r[2] for r in good) / max(sum(r[1] for r in good), 1)))
    finally:
        for p, lf in procs:
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
        time.sleep(4)

    # B) 两个模型共驻留 + np 2
    print()
    print("=== B) 共驻留下把 3B 的槽数提到 2 ===")
    procs = []
    try:
        p9, lf9, ok9 = launch("B9", M9, 8072, 1); procs.append((p9, lf9))
        print("  9B-IQ3 np=1 启动 %s" % ("OK" if ok9 else "失败"))
        p3, lf3, ok3 = launch("B3", M3, 8071, 2); procs.append((p3, lf3))
        print("  3B np=2  启动 %s" % ("OK" if ok3 else "失败（很可能 OOM）"))
        print("  显存 %s" % vram())
        if ok9 and ok3:
            with cf.ThreadPoolExecutor(5) as ex:
                rs = list(ex.map(lambda i: one(8071, i), range(5)))
            print("  3B 5 并发：成功 %d/5" % sum(1 for r in rs if r[0]))
    finally:
        for p, lf in procs:
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
