"""Official Task 4 episodes, from the server's per-call logs (estimated poses), for comparison with the site's scorecard
(Plan-v1.1.MD section 1.1). Per episode:
  * rest states: each pushed block's final estimated pose, "flat clear" if it lies on the table (centre z < 25 mm, tilt
    < 20 deg) at least 2 mm outside the foundation's 150 x 150 mm footprint; else on / leaning on the base, touching the
    base, tipped, on another block, or unseen (confidence < 0.5). On official run 1 the "flat clear" share reproduced
    the site's LHPR: 4-2 44 / 60 = 73.3 % (site 73.3 %), 4-3 48 / 90 = 53 % (site 56.7 %) with an axis-aligned
    footprint (``AXIS_ALIGNED=1``); the rule now lives in ``tower_rl.eval.mirror`` (footprint in the foundation frame);
  * short: pushes the runtime judged "ok" with less than 152 mm of travel along the push, i.e. before the block's near
    end cleared the footprint (the radial rule of tower_rl/push.py::departed fires early for the side rows);
  * drift: how far (xy) each pushed block had moved from its start before its own push began;
  * moved: per push (and 2 s after), other blocks that moved > 10 mm, tower (not yet pushed) vs debris (pushed).

  cd ~/Documents/tower_rl && python3 scripts/official/task4_outcomes.py [results/student/server/official_1]
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tower_rl.eval.mirror import footprint_frame, rest_state  # noqa: E402  (pure numpy: the local mirror's rule)

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/student/server/official_1"
VARIANTS = ["task4-2_full_tower_pushdown_2", "task4-3_full_tower_pushdown_3"]
CLEAR_TRAVEL_MM = 152                         # travel along the push at which the near end is 2 mm past the footprint
AXIS_ALIGNED = os.environ.get("AXIS_ALIGNED") == "1"  # the first version's footprint (axis-aligned, no foundation yaw)


def push_windows(nodes):
    """[(node, first call, last call)] for each run of consecutive calls of one push node."""
    out, i = [], 0
    while i < len(nodes):
        if nodes[i].endswith("_push"):
            j = i
            while j + 1 < len(nodes) and nodes[j + 1] == nodes[i]:
                j += 1
            out.append((nodes[i], i, j))
            i = j + 1
        else:
            i += 1
    return out


for var in VARIANTS:
    print(f"===== {var}")
    tot = {"blocks": 0, "flat clear": 0, "ok": 0, "short": 0, "tower": 0, "debris": 0}
    for f in sorted(glob.glob(f"{RUN}/episodes/{var}/*.jsonl")):
        L = [r for r in (json.loads(l) for l in open(f)) if "pos" in r]
        if len(L) < 50:
            continue
        summ = json.load(open(f[:-6] + "_summary.json"))
        P = np.array([r["pos"] for r in L]); Q = np.array([r["quat"] for r in L])
        C = np.array([r["conf"] for r in L]); T = np.array([r["t"] for r in L])
        nodes = [r.get("node", {}).get("node", "") for r in L]
        base_c, yaw = footprint_frame(P[-1, :3], Q[-1, :3])  # the tower's foundation (pad A: the first three blocks)
        if AXIS_ALIGNED:
            base_c, yaw = P[-1, :3, :2].mean(0), 0.0
        # rest states at the last call
        kinds = [rest_state(P[-1, b], Q[-1, b], base_c, yaw, C[-1, b]) for b in range(6, P.shape[1])]
        clear = sum(k[0] == "flat clear" for k in kinds)
        # runtime verdicts: pushes judged ok, and those that ended before the block cleared the footprint
        oks = [e for e in summ["events"] if e[1].endswith("_push") and e[2] == "ok" and "along_mm" in e[3]]
        short = [e for e in oks if e[3]["along_mm"] < CLEAR_TRAVEL_MM]
        # drift before each push, and blocks moved > 10 mm during each push (+2 s)
        drift, moved, pushed = [], [], set()
        for node, a, b in push_windows(nodes):
            seg_end = min(int(np.searchsorted(T, T[b] + 2.0)), len(L) - 1)
            disp = 1000 * np.linalg.norm(P[a:seg_end + 1] - P[a], axis=2).max(0)
            seen = (C[a:seg_end + 1] >= 0.5).all(0)
            tgt = int(np.argmax(np.linalg.norm(P[b] - P[a], axis=1)))   # the block this push moved most
            drift.append(1000 * float(np.linalg.norm(P[a, tgt, :2] - P[0, tgt, :2])))
            others = [m for m in range(6, P.shape[1]) if m != tgt and disp[m] > 10 and seen[m]]
            moved.append((sum(m not in pushed for m in others), sum(m in pushed for m in others)))
            pushed.add(tgt)
        tot["blocks"] += len(kinds); tot["flat clear"] += clear; tot["ok"] += len(oks); tot["short"] += len(short)
        tot["tower"] += sum(t for t, _ in moved); tot["debris"] += sum(d for _, d in moved)
        bad = "; ".join(f"{k} (gap {g:.0f} mm, tilt {t:.0f}, z {z:.0f})" for k, g, t, z in kinds if k != "flat clear")
        print(f"{os.path.basename(f)[:8]} flat clear {clear}/{len(kinds)} | runtime ok {len(oks)}, short {len(short)} | "
              f"drift mm {' '.join(f'{d:.1f}' for d in drift)} | moved T/D {' '.join(f'{t}/{d}' for t, d in moved)}")
        if bad:
            print(f"         not clear: {bad}")
    n = max(tot["blocks"], 1)
    print(f"  TOTAL flat clear {tot['flat clear']}/{tot['blocks']} = {100 * tot['flat clear'] / n:.1f} % | runtime ok "
          f"{tot['ok']}, of which short of the footprint {tot['short']} | moved > 10 mm: tower {tot['tower']}, debris {tot['debris']}")
