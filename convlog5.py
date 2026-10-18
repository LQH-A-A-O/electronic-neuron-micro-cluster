# -*- coding: utf-8 -*-
import os
d = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(d, "out", "stage5_log.txt")
raw = open(src, "rb").read()
t = raw.decode("utf-16")
open(os.path.join(d, "out", "stage5_log_utf8.txt"), "w", encoding="utf-8").write(t)
print(t)
