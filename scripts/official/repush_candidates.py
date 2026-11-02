"""Official Task 4 episodes (server per-call logs): the blocks the student's R4.1 re-push could act on. For every pushed
block whose last confident estimate rests "on/leaning on base": R4.1's preconditions on that estimate (pushed >= 0.10 m
along its push, near end >= 45 mm high), how often it was confidently seen at that pose after the last push (the
``seen_s`` gate needs two sightings within 15 s), and the episode's other final rest states (a re-push of the last
push's block can also turn the episode's TSR).

  cd ~/Documents/tower_rl && python scripts/official/repush_candidates.py [results/student/server/official_1]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.eval.mirror import corners, footprint_frame, rest_state  # noqa: E402
from tower_rl.recovery.rules import NEAR_END_Z, PUSHED_M  # noqa: E402
from tower_rl.student.spec import TaskSpec  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/student/server/official_1"
for var in ["task4-2_full_tower_pushdown_2", "task4-3_full_tower_pushdown_3"]:
    spec, order = TaskSpec.load(var), list(Layout.load(var).blocks)
    print(f"===== {var}")
    for f in sorted(glob.glob(f"{RUN}/episodes/{var}/*.jsonl")):
        L = [r for r in (json.loads(x) for x in open(f)) if "pos" in r]
        if len(L) < 50:
            continue
        P, Q, C, T = (np.array([r[k] for r in L]) for k in ("pos", "quat", "conf", "t"))
        foot = footprint_frame(P[-1, :3], Q[-1, :3])
        nodes = [r.get("node", {}).get("node", "") for r in L]
        last_push = max(i for i, n in enumerate(nodes) if n.endswith("_push"))
        txy = P[0, :3, :2].mean(0)
        final = [rest_state(P[-1, order.index(g.block)], Q[-1, order.index(g.block)], *foot,
                            C[-1, order.index(g.block)])[0] for g in spec.goals]
        for k, g in enumerate(spec.goals):
            b = order.index(g.block)
            seen = [i for i in range(len(L)) if C[i, b] >= 0.5]
            if not seen:
                continue
            j = seen[-1]
            if rest_state(P[j, b], Q[j, b], *foot)[0] != "on/leaning on base":
                continue
            d = np.asarray(g.direction, float)
            d[2] = 0
            d /= np.linalg.norm(d)
            moved = float((P[j, b] - P[0, b]) @ d)
            cs = corners(P[j, b], Q[j, b])
            al = (cs[:, :2] - txy) @ d[:2]
            near = float(cs[al <= al.min() + 0.03][:, 2].max())
            ts = T[[i for i in seen if i > last_push and np.linalg.norm(P[i, b] - P[j, b]) < 0.004]]
            pair = bool(len(ts) > 1 and np.min(np.diff(ts)) <= 15.0)
            ok = moved >= PUSHED_M and near >= NEAR_END_Z and pair
            others = [kd for i, kd in enumerate(final) if i != k]
            tsr = ok and all(kd == "flat clear" for kd in others) and k == spec.K - 1
            print(f"  {os.path.basename(f)[:8]} goal {k} ({g.block}): last seen leaning at {T[j]:.0f} s, pushed "
                  f"{1000 * moved:.0f} mm, near end {1000 * near:.0f} mm, sightings after the plan {len(ts)}"
                  f" -> {'R4.1 fires' if ok else 'no'}{' (and the episode would pass)' if tsr else ''}")
