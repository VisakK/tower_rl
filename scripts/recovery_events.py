"""The student's recovery node in a camera-loop run (graph "recovery"): every R4.1 decision with the block's state as
the rule saw it (rest estimate) and as it truly was (simulator pose, ``tower_rl.eval.mirror.rest_state``), the re-push
nodes that followed, and where the block came to rest at the end; the injected short pushes (rehearse --inject) and
what became of their blocks; the rule's abstentions.

  python scripts/recovery_events.py <run dir under results/v1_1/t4_camera> [variant 4-2|4-3]
"""
import collections
import glob
import json
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from tower_rl.eval.mirror import footprint_frame, rest_state  # noqa: E402

R = "/home/visakii/Documents/tower_rl"
arm = sys.argv[1]
v = sys.argv[2] if len(sys.argv) > 2 else "4-2"
var = {"4-2": "task4-2_full_tower_pushdown_2", "4-3": "task4-3_full_tower_pushdown_3"}[v]
g = json.load(open(f"{R}/results/bundles/student_v1_1/graphs/{var}.json"))
order = [n["block"] for n in g["nodes"] if n.get("skill") == "push"]
d, = glob.glob(f"{R}/results/v1_1/t4_camera/{arm}/{var}/*/")
eps = json.load(open(d + "episodes.json"))
pl = pickle.load(open(d + "perception_log.pkl", "rb"))
ids = pl["block_ids"]
fids = [ids.index(f) for f in g["tower"]["foundation"]]
tally = collections.Counter()
ABBR = {"flat clear": "1", "tipped": "T", "on/leaning on base": "L", "touching base": "t", "on a block": "B",
        "off the table": "O"}
for ep in eps:
    x = ep["extra"]
    L = [e for e in pl["log"] if e["seed"] == x["seed"]]
    ts = np.array([e["t"] for e in L])
    foot = footprint_frame(np.asarray(L[0]["true_pos"])[fids], np.asarray(L[0]["true_quat"])[fids])

    def true_state(t, k):
        e = L[int(np.clip(np.searchsorted(ts, t), 0, len(L) - 1))]
        i = ids.index(order[k])
        return rest_state(np.asarray(e["true_pos"][i]), np.asarray(e["true_quat"][i]), *foot)[0]

    rest = [r[0] for r in x["rest"]]
    log = x["policy_log"]
    inj = [int(m.group(1)) for t, n, s, info in log for m in [re.match(r"p(\d+)_push$", n)] if m and info.get("injected_stop")]
    rec = (x.get("recovery") or {})
    decs = rec.get("decisions", [])
    line = f"--- {x['case']:17s} TSR {int(ep['task_success'])} rest {''.join(ABBR.get(r, '?') for r in rest)}"
    print(line + (f"  injected short: {inj}" if inj else ""))
    for k in inj:
        tally["injected short pushes"] += 1
        tally[f"injected block at the end: {rest[k]}"] += 1
    for dd in decs:
        k = dd["goal"]
        tally["R4.1 decisions"] += 1
        ts_true = true_state(dd["t"], k)
        tally[f"R4.1 on a block truly {ts_true}"] += 1
        nodes = [(t, n, s, info) for t, n, s, info in log if re.match(rf"p{k}_re_", n) and t >= dd["t"] - 0.5]
        print(f"   R4.1 t {dd['t']:6.1f} goal {k}: seen {dd.get('kind')} gap {dd.get('gap_mm')} z {dd.get('z_mm')} "
              f"tilt {dd.get('tilt_deg')} | truly {ts_true} | end: {rest[k]}")
        for t, n, s, info in nodes[:2]:
            print(f"      {t:6.1f} {n:18s} {s:8s} {info}")
    for k in sorted(set(dd["goal"] for dd in decs)):
        tally[f"re-pushed block at the end: {rest[k]}"] += 1
    for a in rec.get("abstentions", []):
        tally[f"abstained: {a['why']} ({a['kind']})"] += 1
print()
for k_, n in sorted(tally.items()):
    print(f"{n:4d}  {k_}")
