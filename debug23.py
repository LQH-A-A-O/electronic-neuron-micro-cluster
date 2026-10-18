# -*- coding: utf-8 -*-
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp.stage23 as S

st = S.make_stream(0)
R = S.R23("audit", 0)
print("轮次  工具准确率  全局一致率  限定到本工具类的一致率  池大小  工具类数")
for t, task in enumerate(st):
    hv_x, hv_y = task.x_holdout, task.y_holdout
    tf = lambda x: int(hv_y[int(np.argmin(np.linalg.norm(hv_x - np.asarray(x), axis=1)))])
    for s0 in range(0, len(task.x_train), S.ROUND):
        R.round(task.x_train[s0:s0 + S.ROUND], task.y_train[s0:s0 + S.ROUND], tf, hv_x, hv_y)
        if R.tool is not None and len(R.audit_x) >= S.AUDIT_MIN and t == 0:
            X = np.array(R.audit_x); Y = np.array(R.audit_y)
            p = R.tool._proba(X)
            pred = np.array(R.tool.classes)[p.argmax(axis=1)]
            glob = float((pred == Y).mean())
            m = np.isin(Y, R.tool.classes)
            loc = float((pred[m] == Y[m]).mean()) if m.any() else float("nan")
            print("%-5s %-11.3f %-11.3f %-23.3f %-7d %d"
                  % ("%d.%d" % (t, s0 // S.ROUND), R.tool_hist[-1], glob, loc, len(X), len(R.tool.classes)))
