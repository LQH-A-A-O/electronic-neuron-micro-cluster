# -*- coding: utf-8 -*-
import threading, time, requests, psutil
B = "http://127.0.0.1:8092"
d0 = psutil.disk_io_counters(); t0 = time.perf_counter()
res = {}
def gen():
    r = requests.post(B + "/v1/chat/completions", timeout=600, json={
        "messages": [{"role": "user", "content": "写一段 300 字的技术说明。"}],
        "max_tokens": 300, "temperature": 0})
    res["u"] = r.json().get("usage", {})
th = threading.Thread(target=gen); th.start()
peak = 0
reads = []
while th.is_alive():
    vm = psutil.virtual_memory()
    di = psutil.disk_io_counters()
    reads.append(di.read_bytes)
    peak = max(peak, vm.used)
    time.sleep(0.5)
th.join()
dt = time.perf_counter() - t0
d1 = psutil.disk_io_counters()
n = res.get("u", {}).get("completion_tokens", 0)
vm = psutil.virtual_memory()
print("生成 %d token / %.1f 秒 -> %.1f tok/s" % (n, dt, n / dt if dt else 0))
print("内存: 总 %.1f GB / 已用 %.1f GB / 可用 %.1f GB" % (vm.total/1e9, vm.used/1e9, vm.available/1e9))
print("生成期间磁盘读取 %.2f GB（%.0f MB/s）" % ((d1.read_bytes-d0.read_bytes)/1e9, (d1.read_bytes-d0.read_bytes)/1e6/max(dt,1)))
