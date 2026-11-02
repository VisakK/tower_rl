"""Do planned transits and parks disturb the structure? Camera-loop runs with truth poses in perception_log.pkl
(Plan-v1.1.MD section 1.3). For each run of consecutive calls of a transit or park node, the largest true displacement
(start of the run to the call after it) of a structure block:
  * Task 4: movable blocks still on the tower (centre within 0.12 m of the tower axis, z > 35 mm);
  * Task 5: the side and roof blocks;
  * Tasks 1-3: every movable block not held.
On G10 (student_v1_kp_v2, 160 episodes): Tasks 1-3 0 / 736 segments > 3 mm, Task 4 9 / 303, Task 5 3 / 165.

  cd ~/Documents/tower_rl && ~/Documents/moves/env_isaaclab/bin/python scripts/official/transit_structure.py [results/student/closed_loop/student_v1_kp_v2]
"""
import collections, glob, pickle, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/student/closed_loop/student_v1_kp_v2"
counts = collections.defaultdict(collections.Counter)
worst = collections.defaultdict(list)
for f in sorted(glob.glob(f"{ROOT}/*/*/perception_log.pkl")):
    var = f.split("/")[-3]
    d = pickle.load(open(f, "rb"))
    ids, log = d["block_ids"], d["log"]
    movable = [i for i, b in enumerate(ids) if "base" not in b]
    episodes = collections.defaultdict(list)
    for r in log:
        episodes[r["seed"]].append(r)
    for seed, L in episodes.items():
        L.sort(key=lambda r: r["t"])
        axis = L[0]["true_pos"][:3, :2].mean(0)
        runs, cur = [], None
        for i, r in enumerate(L):
            s = r.get("skill") or "none"
            if cur is None or s != cur[0]:
                cur = [s, i, i]
                runs.append(cur)
            else:
                cur[2] = i
        for skill, a, b in runs:
            if skill not in ("transit", "park"):
                continue
            b2 = min(b + 1, len(L) - 1)
            p0, p1 = L[a]["true_pos"], L[b2]["true_pos"]
            if var.startswith("task4"):
                r0 = np.linalg.norm(p0[:, :2] - axis, axis=1)
                struct = [i for i in movable if r0[i] < 0.12 and p0[i, 2] > 0.035]
            elif var.startswith("task5"):
                struct = [i for i, bid in enumerate(ids) if "side" in bid or "roof" in bid]
            else:
                held = L[a]["held"] | L[b2]["held"]
                struct = [i for i in movable if not held[i]]
            if not struct:
                continue
            m = 1000 * float(np.linalg.norm(p1[struct] - p0[struct], axis=1).max())
            fam = var[:5]
            counts[fam]["segments"] += 1
            counts[fam][">3 mm"] += m > 3
            counts[fam][">10 mm"] += m > 10
            if m > 10:
                worst[fam].append((round(m, 1), var, seed, round(L[a]["t"], 1), skill))
for fam in sorted(counts):
    c = counts[fam]
    print(f"{fam}: transit/park segments {c['segments']:5d}, structure moved > 3 mm in {c['>3 mm']:3d}, > 10 mm in {c['>10 mm']:3d}")
    for w in sorted(worst[fam], reverse=True)[:5]:
        print(f"    {w}")
