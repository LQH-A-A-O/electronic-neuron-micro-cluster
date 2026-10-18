# -*- coding: utf-8 -*-
import json, os, glob
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rows = []
for f in sorted(glob.glob(os.path.join(OUT, "results_probe_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    rows.append((d["model"], d["code_score"], d["code_n"], d["edit_score"], d["edit_n"],
                 int(d["write_score"]), d["write_n"], d["total"], d["secs"]))
print("%-14s %-12s %-12s %-14s %-10s %s" % ("模型", "代码", "改错", "写作", "总分", "耗时"))
for r in rows:
    print("%-14s %-12s %-12s %-14s %-10.3f %.0f 秒"
          % (r[0], "%d/%d" % (r[1], r[2]), "%d/%d" % (r[3], r[4]),
             "%d/%d" % (r[5], r[6]), r[7], r[8]))
