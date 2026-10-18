# -*- coding: utf-8 -*-
"""真实噪声下的门控精度：用模型自己的幻觉当污染源。

关键问题：门控靠「多记录互证」，但如果模型**稳定地编同一个错答案**，
多记录就会一致 -> 门控被骗 -> 脏数据进库。这是个真实风险，必须测。

做法：
  1. 对每个事实，让 3B 在无记忆下采样 K=5 次（temp=0.8）
  2. 这些回答就是「真实记录流」：有的对（模型碰巧知道），多数错（幻觉）
  3. 门控：同 key 的记录里，票数 >50% 的值收下
  4. 指标：precision（收下的值里有多少是对的）、recall（多少事实收到了值）
"""
import json, os, re, subprocess, sys, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = r"E:\aqca\work\llm\runtime\b9637-cuda12.4\llama-server.exe"
M3 = r"E:\aqca\work\llm\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
P3 = 8032
K = 5
N_FACT = 30
CN = {'零':'0','一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
KEEP = set('0123456789abcdefghijklmnopqrstuvwxyz')
def norm(t):
    t = t.lower().replace('百分之','').replace('万','0000')
    for k, v in CN.items(): t = t.replace(k, v)
    return ''.join(c for c in t if c in KEEP)
def launch(model, port):
    lf = open(os.path.join(ROOT, "out", "gate_%d.log" % port), "wb")
    cmd = [SERVER, "-m", model, "-c", "4096", "-np", "1", "-fa", "on", "--jinja",
           "--chat-template-kwargs", chr(123) + chr(34) + "enable_thinking" + chr(34) + ":false" + chr(125),
           "--temp", "0.8", "--top-p", "0.95",
           "--host", "127.0.0.1", "--port", str(port), "-ngl", "99"]
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(240):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:%d/health" % port, timeout=2).json().get("status") == "ok":
                time.sleep(1); return p, lf
        except Exception: pass
    p.terminate(); lf.close(); raise SystemExit("load fail")
def chat(port, msg, mx=40):
    d = requests.post("http://127.0.0.1:%d/v1/chat/completions" % port, timeout=300,
                      json={"messages": [{"role": "user", "content": msg}], "max_tokens": mx, "temperature": 0.8, "top_p": 0.95}).json()
    return d["choices"][0]["message"]["content"] or ""
if __name__ == '__main__':
    D = json.load(open(os.path.join(ROOT, 'data', 'qa_bench.json'), encoding='utf-8'))
    facts = D['facts'][:N_FACT]
    p3, lf3 = launch(M3, P3)
    try:
        recs = []
        for i, f in enumerate(facts):
            vals = []
            for _ in range(K):
                a = chat(P3, f['key'].split(' ')[0] + ' ' + f['key'].split(' ')[1] + ' 是多少？只回答值本身。')
                vals.append(norm(a)[:12])
            recs.append(dict(key=f['key'], truth=norm(f['value']), vals=vals))
            if (i+1) % 10 == 0:
                print('  采样 %d/%d' % (i+1, len(facts))); sys.stdout.flush()
        # 门控：票数 > 50% 的值
        kept = rejected_ambig = 0
        correct_kept = 0
        fooled = []
        for r in recs:
            cnt = {}
            for v in r['vals']:
                cnt[v] = cnt.get(v, 0) + 1
            top, c = max(cnt.items(), key=lambda x: x[1])
            if c / len(r['vals']) > 0.5:
                kept += 1
                if norm(r['truth']) and norm(r['truth']) in top:
                    correct_kept += 1
                else:
                    fooled.append((r['key'][:34], r['truth'], top[:12], c))
            else:
                rejected_ambig += 1
        n = len(recs)
        print()
        print('=' * 70)
        print('真实噪声门控（%d 条事实 x K=%d 采样，temp=0.8，无记忆）' % (n, K))
        print('=' * 70)
        print('  收下 %d 条 / 拒绝（无多数） %d 条' % (kept, rejected_ambig))
        print('  收下的里面正确的 %d 条' % correct_kept)
        print('  **门控精度 precision = %.3f**' % (correct_kept / kept if kept else 0))
        print('  **召回 recall = %.3f**' % (correct_kept / n if n else 0))
        print()
        if fooled:
            print('  🚩 被「一致幻觉」骗过的（门控收下了错值）：')
            for k, t, got, c in fooled[:8]:
                print('    %-36s 真值 %-10s 收下 %-10s (%d/%d 票)' % (k, t, got, c, K))
        else:
            print('  ✅ 没有被一致幻觉骗过')
        # 统计幻觉的一致性
        consistent = sum(1 for r in recs if len(set(r['vals'])) == 1)
        print()
        print('  模型输出的一致性：K 次全相同的 %d/%d 条 (%.0f%%)'
              % (consistent, n, 100 * consistent / n))
        json.dump(recs, open(os.path.join(ROOT, "out", "results_gate_real.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        p3.terminate()
        try: p3.wait(timeout=15)
        except Exception: p3.kill()
        lf3.close()