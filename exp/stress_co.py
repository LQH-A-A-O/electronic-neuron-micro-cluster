# -*- coding: utf-8 -*-
"""共驻留压测：两个模型同时跑，塞长上下文，看会不会 OOM。

587 MiB 余量是这套配置上线前的最大风险。
"""
import os, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
CFG = [("3B-Q4", r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf", 8075),
       ("9B-IQ3", r"E:\aqca\work\llm\models\Qwen3.5-9B-UD-IQ3_XXS.gguf", 8074)]

def vram():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().replace("\n", " | ")

def launch(name, model, port, ctx):
    lf = open(os.path.join(ROOT, "out", "stress_%s.log" % name), "wb")
    p = subprocess.Popen([SERVER, "-m", model, "-c", str(ctx), "-np", "1", "-fa", "on", "--jinja",
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

def try_gen(port, nprompt, label):
    """塞 nprompt 个重复 token 的上下文，再让它生成。"""
    ctx = ("请仔细阅读下面的材料并记住它。\n" + ("这是一段用于压力测试的填充文本。" * (nprompt // 10))) \
          + "\n\n材料结束。请只回答：你读完了吗？"
    t0 = time.perf_counter()
    try:
        r = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=600,
                          json={"messages": [{"role": "user", "content": ctx}],
                                "max_tokens": 64, "temperature": 0})
        d = r.json()
        if "choices" in d:
            u = d.get("usage", {})
            print("    %-14s 提示 %5d token -> 生成 %d token  %.1f 秒   OK"
                  % (label, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.perf_counter() - t0))
            return True
        print("    %-14s 失败: %s" % (label, str(d)[:90]))
        return False
    except Exception as e:
        print("    %-14s 异常: %s" % (label, str(e)[:90]))
        return False

if __name__ == "__main__":
    procs = []
    ok_all = True
    try:
        for nm, m, pt in CFG:
            p, lf, ok = launch(nm, m, pt, 8192)
            procs.append((p, lf))
            print("  %-7s 启动 %s" % (nm, "OK" if ok else "失败"))
            ok_all &= ok
        print("  共驻留显存: %s (MiB 已用 / 空闲)" % vram())
        print()
        print("  压测（短 / 中 / 长 上下文，两个模型都打）:")
        for n in (200, 2000, 6000):
            print("  --- 约 %d 字上下文 ---" % n)
            for nm, m, pt in CFG:
                try_gen(pt, n, nm)
        print()
        print("  压测后显存: %s" % vram())
        print()
        print("结论：%s" % ("共驻留在 8k 上下文下可承受" if ok_all else "有模型起不来"))
    finally:
        for p, lf in procs:
            p.terminate()
            try: p.wait(timeout=20)
            except Exception: p.kill()
            lf.close()
