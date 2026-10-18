# -*- coding: utf-8 -*-
import os, sys
d = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(d, "out", sys.argv[1]), "rb").read().decode("utf-16")
open(os.path.join(d, "out", sys.argv[1].replace(".txt", "_utf8.txt")), "w", encoding="utf-8").write(t)
print(t)
