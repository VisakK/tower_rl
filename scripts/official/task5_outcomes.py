"""Official Task 5 episodes, from the server's per-call logs (estimated poses): each insert block's final offset from
its slot goal (the midpoint of its layer's two side blocks at the first call), its confidence, the largest
displacement of a structure block (sides and roof) over the episode, and the runtime's verdicts (Plan-v1.1.MD section
1.2). In 5-3 the two inserts' estimates can swap (issue 13): an upper insert reported ~32 mm low and a lower one
~32 mm high, both on the slot axis, means both slots are filled. On official run 1 a block within ~4 mm of its goal
with the structure moved <= ~8 mm matched the site's 5 / 10 (5-2) and 4 / 10 (5-3).

  cd ~/Documents/tower_rl && ~/Documents/moves/env_isaaclab/bin/python scripts/official/task5_outcomes.py [results/student/server/official_1]
"""
import glob, json, os, sys
import numpy as np
import yaml

sys.path.insert(0, ".")
from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.perception.estimator import block_order  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/student/server/official_1"

for var in ["task5-2_tower_insertion_2", "task5-3_tower_insertion_3"]:
    ids = block_order(Layout.load(var))
    transfers = yaml.safe_load(open(f"configs/layouts/{var}.yaml"))["transfers"]
    struct = [i for i, b in enumerate(ids) if "side" in b or "roof" in b]
    print(f"===== {var}")
    for f in sorted(glob.glob(f"{RUN}/episodes/{var}/*.jsonl")):
        L = [r for r in (json.loads(l) for l in open(f)) if "pos" in r]
        if len(L) < 50:
            continue
        p0, p1, c1 = np.array(L[0]["pos"]), np.array(L[-1]["pos"]), np.array(L[-1]["conf"])
        summ = json.load(open(f[:-6] + "_summary.json"))
        out = []
        for t in transfers:
            k = ids.index(t["block"])
            goal = p0[[ids.index(s) for s in t["side_blocks"]]].mean(0)   # slot centre from the perceived side blocks
            off = 1000 * (p1[k] - goal)
            out.append(f"{t['block'].replace('source_', '')}: xy {np.hypot(off[0], off[1]):5.1f} z {off[2]:6.1f} mm conf {c1[k]:.1f}")
        moved = 1000 * np.linalg.norm(p1[struct] - p0[struct], axis=1).max()
        verdicts = " ".join(f"{e[1].split('_', 1)[1]}:{e[2]}{'(' + e[3]['why'] + ')' if e[3].get('why') else ''}"
                            for e in summ["events"] if any(s in e[1] for s in ("_grasp", "_insert", "_push_in")) and "pre" not in e[1])
        print(f"{os.path.basename(f)[:8]} {' | '.join(out)} | structure max {moved:5.1f} mm | {verdicts}")
