# -*- coding: utf-8 -*-
"""并行分片下载 —— 专治 cave.cs.toronto.edu 的单连接限速（42 KB/s）。

实测：单连接 42 KB/s；16 路并发合计约 270 KB/s。所以切片并发是唯一可行路线。
"""
import os, sys, threading, time
import requests

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "cifar10.tar.gz")
N = 24
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})

def main():
    r = S.head(URL, timeout=30, allow_redirects=True)
    total = int(r.headers["Content-Length"])
    part = (total + N - 1) // N
    print("总大小 %d 字节，切 %d 片，每片 %d" % (total, N, part)); sys.stdout.flush()
    tmp = OUT + ".part"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(tmp, "wb") as f:
        f.truncate(total)

    done = [0] * N
    def worker(i):
        s, e = i * part, min((i + 1) * part - 1, total - 1)
        for attempt in range(6):
            try:
                got = s
                with S.get(URL, headers={"Range": "bytes=%d-%d" % (got, e)}, stream=True,
                           timeout=(30, 120)) as resp:
                    resp.raise_for_status()
                    with open(tmp, "r+b") as fh:
                        fh.seek(got)
                        for chunk in resp.iter_content(1 << 16):
                            fh.write(chunk); got += len(chunk); done[i] = got - s
                if got > e:
                    return
            except Exception as ex:
                time.sleep(2 + 3 * attempt)
        print("  片 %d 未完成（%d/%d）" % (i, done[i], e - s + 1))

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
    t0 = time.time()
    for t in ts: t.start()
    while any(t.is_alive() for t in ts):
        time.sleep(20)
        print("  %.0f 秒  已下 %.1f / %.1f MB  (%.0f KB/s)"
              % (time.time() - t0, sum(done) / 1e6, total / 1e6, sum(done) / 1e3 / max(time.time() - t0, 1)))
        sys.stdout.flush()
    for t in ts: t.join()
    if sum(done) >= total:
        os.replace(tmp, OUT)
        import hashlib
        h = hashlib.md5(open(OUT, "rb").read()).hexdigest()
        print("完成。md5=%s  期望 c58f30108f718f92721af3b95e74349a  -> %s"
              % (h, "一致" if h == "c58f30108f718f92721af3b95e74349a" else "不一致!"))
    else:
        print("未下完：%d / %d" % (sum(done), total))

if __name__ == "__main__":
    main()
