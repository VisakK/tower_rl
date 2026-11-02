"""Live view of the official_v1_2 evaluation's episodes (files written since SINCE): per episode the orientation
choice, blind flag, confident blocks on the first and last call, simulated time reached, last node, call latency.

  python3 scripts/official/live_episodes.py [SINCE 'YYYY-mm-dd HH:MM:SS'] [ENDPOINT]
"""
import glob
import json
import os
import sys
import time
from datetime import datetime

SINCE = datetime.strptime(sys.argv[1] if len(sys.argv) > 1 else "2026-10-28 19:55:00", "%Y-%m-%d %H:%M:%S").timestamp()
NAME = sys.argv[2] if len(sys.argv) > 2 else "official_v1_2"
D = f"/home/visakii/Documents/tower_rl/results/student/server/{NAME}/episodes"

rows = []
for f in glob.glob(f"{D}/*/*.jsonl"):
    if os.path.getctime(f) < SINCE:
        continue
    L = []
    for x in open(f):
        try:
            L.append(json.loads(x))
        except ValueError:
            pass
    calls = [r for r in L if "total_ms" in r]
    var = os.path.basename(os.path.dirname(f))
    trial = os.path.basename(f)[:8]
    summ = f[:-6] + "_summary.json"
    s = json.load(open(summ)) if os.path.exists(summ) else None
    if not calls:
        rows.append((os.path.getctime(f), f"{var:32s} {trial} no calls yet"))
        continue
    first = calls[0]
    o = first.get("orient") or {}
    seen = [sum(c >= 0.5 for c in r["conf"]) for r in calls if r.get("conf")]
    tot = sorted(r["total_ms"] for r in calls)
    p99 = tot[min(len(tot) - 1, int(0.99 * len(tot)))]
    node = (calls[-1].get("node") or {}).get("node")
    holds = {}
    for r in calls:
        if r.get("hold"):
            holds[r["hold"]] = holds.get(r["hold"], 0) + 1
    extra = ""
    if s:
        rec = s.get("recovery") or {}
        extra = (f" | END errors {s.get('errors')} vetoed {s.get('vetoed')} recovery decisions "
                 f"{len(rec.get('decisions', []))} abstentions {len(rec.get('abstentions', []))}")
    rows.append((os.path.getctime(f),
                 f"{var:32s} {trial} calls {len(calls):4d} t {calls[-1].get('t', 0):6.1f} rotate {o.get('rotate')!s:5s} "
                 f"same {o.get('same')} rot180 {o.get('rot180')} blind {first.get('blind')} seen first/last "
                 f"{seen[0] if seen else '-'}/{seen[-1] if seen else '-'} of {len(first.get('conf') or [])} "
                 f"ms p50 {tot[len(tot) // 2]:.0f} p99 {p99:.0f} holds {holds} node {node}{extra}"))
for _, r in sorted(rows):
    print(r)
print(time.strftime("%H:%M:%S"), len(rows), "episodes")
