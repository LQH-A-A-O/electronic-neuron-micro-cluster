# -*- coding: utf-8 -*-
import io, os
d = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(d, "out", "stage4_log.txt")
raw = open(src, "rb").read()
for enc in ("utf-16", "utf-8"):
    try:
        t = raw.decode(enc); break
    except Exception:
        continue
open(os.path.join(d, "out", "stage4_log_utf8.txt"), "w", encoding="utf-8").write(t)
print(t)
