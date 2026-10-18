# -*- coding: utf-8 -*-
import os, sys
try:
    import requests
except ImportError:
    print("no requests"); raise SystemExit
H = {"User-Agent": "Mozilla/5.0"}
tests = [
    ("HuggingFace parquet", "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet"),
    ("HF api", "https://huggingface.co/api/datasets/fancyzhx/ag_news"),
    ("raw.githubusercontent", "https://raw.githubusercontent.com/ageron/handson-ml2/master/README.md"),
    ("github codeload", "https://codeload.github.com/pandas-dev/pandas/tar.gz/refs/tags/v2.2.0"),
]
for name, u in tests:
    try:
        r = requests.get(u, headers=H, stream=True, timeout=30, allow_redirects=True)
        print("%-22s %s  %s  %s" % (name, r.status_code, r.headers.get("content-length", "?"), r.headers.get("content-type", "")[:40]))
        r.close()
    except Exception as e:
        print("%-22s ERROR %s" % (name, type(e).__name__ + ": " + str(e)[:90]))
print()
# UCI 候选的体积
uci = [(2, "adult"), (321, "electricity"), (228, "sms spam"), (217, "reuters"),
       (350, "credit default"), (17, "breast cancer"), (144, "?"), (73, "?")]
for i, n in uci:
    u = "https://archive.ics.uci.edu/static/public/%d/%s.zip" % (i, n.replace(" ", "+"))
    try:
        r = requests.head(u, headers=H, timeout=25, allow_redirects=True)
        print("UCI %-4d %-16s %s  长度=%s" % (i, n, r.status_code, r.headers.get("content-length", "?")))
    except Exception as e:
        print("UCI %-4d %-16s ERROR %s" % (i, n, str(e)[:60]))
