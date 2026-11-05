"""Official Task 4 episodes (server per-call logs): the pushed block's estimate confidence at the row where each push
that ended "ok" fired its departure test (the push node runs one row per call: call ``steps`` of the node's window),
against the block's final rest state (``tower_rl.eval.mirror.rest_state`` on the last call's estimates). Did official
pushes end on the estimator's coasted poses (confidence < 0.5), and where did those blocks come to rest?

  cd ~/Documents/tower_rl && python scripts/official/push_end_conf.py  (Python >= 3.10: the Isaac env) [results/student/server/official_1]
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.eval.mirror import footprint_frame, rest_state  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/student/server/official_1"
for var in ["task4-2_full_tower_pushdown_2", "task4-3_full_tower_pushdown_3"]:
    tab = collections.Counter()
    order = list(Layout.load(var).blocks)  # the server's block order (State.block_ids)
    for f in sorted(glob.glob(f"{RUN}/episodes/{var}/*.jsonl")):
        L = [r for r in (json.loads(x) for x in open(f)) if "pos" in r]
        if len(L) < 50:
            continue
        summ = json.load(open(f[:-6] + "_summary.json"))
        P, Q, C = (np.array([r[k] for r in L]) for k in ("pos", "quat", "conf"))
        nodes = [r.get("node", {}).get("node", "") for r in L]
        blocks = [r.get("node", {}).get("block") for r in L]
        base_c, yaw = footprint_frame(P[-1, :3], Q[-1, :3])
        oks = [e for e in summ["events"] if e[1].endswith("_push") and e[2] == "ok" and "steps" in e[3]]
        used = set()
        for t, name, status, info in oks:
            starts = [i for i in range(len(nodes)) if nodes[i] == name and (i == 0 or nodes[i - 1] != name) and i not in used]
            if not starts:
                continue
            i0 = starts[0]
            used.add(i0)
            k = i0 + int(info["steps"])
            if k >= len(L):
                continue
            j = order.index(blocks[i0])
            conf = float(C[k, j])
            rs = rest_state(P[-1, j], Q[-1, j], base_c, yaw, C[-1, j])[0]
            tab[("conf" if conf >= 0.5 else "UNCONF", rs)] += 1
    print(f"===== {var}")
    tot = collections.Counter()
    for (c, rs), n in sorted(tab.items()):
        tot[c] += n
        print(f"  {c:7s} {rs:22s} {n}")
    print("  pushes ended ok:", dict(tot))
