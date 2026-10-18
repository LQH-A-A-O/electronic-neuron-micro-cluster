# -*- coding: utf-8 -*-
"""6 小时连压测：直接打 HTTP 服务，最贴近真实使用。"""
import json, os, statistics, subprocess, sys, threading, time
import requests
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = 'http://127.0.0.1:8399'
LOG = os.path.join(ROOT, 'out', 'soak3.jsonl')
VLOG = os.path.join(ROOT, 'out', 'soak3_vram.jsonl')
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
QS = [
  '订单服务的监听端口是多少？', '支付网关的超时毫秒数是多少？', '用户中心的部署区域是哪个？',
  '消息总线的日请求上限是多少？', '搜索集群归哪个团队负责？', '推荐引擎的监听端口是多少？',
  '风控系统的超时毫秒数是多少？', '报表平台的部署区域是哪个？', '文件存储的日请求上限是多少？',
  '调度中心归哪个团队负责？', '临时服务X01的端口是多少？', '临时服务X02的超时是多少？',
]
def free():
    try:
        return int(subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.strip())
    except Exception:
        return -1
stop = threading.Event()
def vloop():
    while not stop.is_set():
        with open(VLOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(t=time.time(), free=free())) + chr(10))
        stop.wait(60)
if __name__ == '__main__':
    threading.Thread(target=vloop, daemon=True).start()
    t0 = time.time(); t_end = t0 + HOURS * 3600
    n = err = abst = 0
    print('6小时压测启动，目标 %s，时长 %.1f 小时' % (BASE, HOURS)); sys.stdout.flush()
    while time.time() < t_end:
        q = QS[n % len(QS)]
        ts = time.time()
        try:
            d = requests.get(BASE + '/api/ask', params={'q': q}, timeout=180).json()
            if 'error' in d: err += 1; continue
            if d.get('abstained'): abst += 1
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(n=n, q=q, ans=d.get('answer','')[:24],
                                          verified=d.get('verified'), ms=d.get('ms'),
                                          sec=time.time()-ts, free=free())) + chr(10))
        except Exception:
            err += 1
        n += 1
        if n % 100 == 0:
            el = time.time() - t0
            print('  %d 条  %.0f 分钟  弃答 %d  错 %d  空闲 %d MiB' % (n, el/60, abst, err, free()))
            sys.stdout.flush()
    print('完成：%d 条，弃答 %d，错 %d，%.0f 分钟' % (n, abst, err, (time.time()-t0)/60))