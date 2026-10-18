# -*- coding: utf-8 -*-
"""并发测服务：1/2/4/8 并发下，看排队与失败。"""
import concurrent.futures as cf, json, statistics as st, sys, time
import requests
BASE = 'http://127.0.0.1:8499'
QS = ['订单服务的监听端口是多少？','支付网关的超时毫秒数是多少？','用户中心的部署区域是哪个？',
      '搜索集群归哪个团队负责？','推荐引擎的监听端口是多少？','风控系统的超时毫秒数是多少？']
def one(i):
    t0 = time.time()
    try:
        d = requests.get(BASE + '/api/ask', params={'q': QS[i % len(QS)]}, timeout=180).json()
        return (True, time.time() - t0, d.get('ms', 0), d.get('answer', ''))
    except Exception as e:
        return (False, time.time() - t0, 0, str(e)[:40])
if __name__ == '__main__':
    res = []
    for c in (1, 2, 4, 8):
        t0 = time.time()
        with cf.ThreadPoolExecutor(c) as ex:
            rs = list(ex.map(one, range(c * 3)))
        wall = time.time() - t0
        ok = [r for r in rs if r[0]]
        bad = len(rs) - len(ok)
        lat = sorted(r[1] for r in ok)
        srv = sorted(r[2] for r in ok if r[2])
        print('  并发 %d  发出 %2d 条  成功 %2d 失败 %d  墙钟 %.2f 秒  P50 %.2f 秒  P95 %.2f 秒  服务端 P50 %d ms'
              % (c, len(rs), len(ok), bad, wall,
                 st.median(lat) if lat else 0, lat[int(len(lat)*0.95)] if lat else 0,
                 srv[len(srv)//2] if srv else 0))
        sys.stdout.flush()
        res.append(dict(conc=c, n=len(rs), ok=len(ok), bad=bad, wall=round(wall,2),
                        p50=round(st.median(lat),3) if lat else 0,
                        p95=round(lat[int(len(lat)*0.95)],3) if lat else 0))
    json.dump(res, open('out/results_concurrency.json','w'), indent=1)