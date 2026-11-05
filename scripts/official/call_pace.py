"""Sample when each new call row appears in the open episode files (polled every 0.25 s) for DUR seconds, then print
the wall interval between calls against the rows the previous reply carried, and our server time per call.

  python3 scripts/official/call_pace.py [DUR_S] [ENDPOINT]   # only while episodes are running (default official_1)
"""
import json, sys, time, statistics as st
from pathlib import Path
from collections import defaultdict
D = Path("/home/visakii/Documents/tower_rl/results/student/server") / (sys.argv[2] if len(sys.argv) > 2 else "official_1") / "episodes"
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 600
seen, arrivals = {}, defaultdict(list)   # file -> line count; file -> [(t_arrival, row)]
t0 = time.time()
while time.time() - t0 < DUR:
    now = time.time()
    for f in D.glob("*/*.jsonl"):
        if now - f.stat().st_mtime > 30:
            continue
        lines = f.read_text().splitlines()
        n0 = seen.get(f)
        seen[f] = len(lines)
        if n0 is None:
            continue
        for l in lines[n0:]:
            try:
                arrivals[f].append((now, json.loads(l)))
            except ValueError:
                pass
    time.sleep(0.25)
by_rows = defaultdict(list)
srv = []
for f, a in arrivals.items():
    for (ta, ra), (tb, rb) in zip(a, a[1:]):
        by_rows[ra.get("rows", 0)].append(tb - ta)
    srv += [r["total_ms"] for _, r in a if "total_ms" in r]
print(f"{sum(len(a) for a in arrivals.values())} calls in {DUR:.0f} s over {len(arrivals)} episode file(s): "
      + ", ".join(f"{f.parent.name[:7]} {len(a)}" for f, a in arrivals.items()))
for rows in sorted(by_rows):
    x = by_rows[rows]
    print(f"  after a {rows}-row reply: n {len(x)}, interval mean {st.mean(x):.2f} s, median {st.median(x):.2f} s")
if srv:
    print(f"  our server time per call: mean {st.mean(srv):.0f} ms, max {max(srv):.0f} ms")
