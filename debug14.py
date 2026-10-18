# -*- coding: utf-8 -*-
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.stage10 import N_OBS, N_TASK, ROUND, TOL, phi
from exp.stage11 import AuditRunner, make_stream
for mode in ("audit_revisit", "quarantine"):
    st = make_stream(0)
    R = AuditRunner(mode, 0, st)
    print("== mode", mode)
    for t in range(N_TASK):
        xo, yo, _ = st.obs[t]
        tf = lambda x, _t=t: float((phi(x) @ st.ws[_t]).ravel()[0])
        for s0 in range(0, N_OBS, ROUND):
            R.process_round(xo[s0:s0 + ROUND], yo[s0:s0 + ROUND], tf)
            R.try_compile(st.centers[t])
        per = [float((np.abs(R.predict(st.queries[j][0]) - st.queries[j][1]) < TOL).mean())
               for j in range(t + 1)]
        print("  task%d demoted=%d recovered=%d quar=%d tools=%d accepted=%d  perf=%s"
              % (t, R.demoted, R.recovered, len(R.quarantine), len(R.tools),
                 len(R.accepted_x), " ".join("%.2f" % v for v in per)))
