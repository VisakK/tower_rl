#!/usr/bin/env python3
"""Router behaviour after failures (Studentv1_3_plan Step 6), transfer variants only. Per run directory (rehearse
--out_root, any depth), from each episode's node events:
- buried reworks: an acquire (``tK_to_pregrasp`` / ``tK_grasp``) of goal K after K's block was placed and after a goal
  of a higher destination layer was placed ok on top of it (the official 588e8a2a pattern);
- failed places redone at once: a failed ``tK_place`` whose next goal option is goal K again (re-grasp, re-place), not
  another goal (the official 584b1314 / 7a9650ef pattern: on to the next goal, K never redone).

  python3 scripts/rework_report.py <run dir> [<run dir> ...]
"""
import glob
import json
import re
import sys
from functools import lru_cache

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from tower_rl.graph.graph import load_graph  # noqa: E402

NODE = re.compile(r"^t(\d+)_(to_pregrasp|grasp|to_preplace|place)(_alt)?$")


@lru_cache(None)
def layers(variant: str) -> dict:
    g = load_graph(variant)
    return {n["transfer"]: n["dest_layer"] for n in g["nodes"] if n["skill"] == "place_release"}


def report(root: str):
    eps = buried = failed = redone = 0
    examples = []
    for f in sorted(glob.glob(f"{root}/**/episodes.json", recursive=True)):
        for e in json.load(open(f)):
            if not e["variant"].startswith(("task1", "task2", "task3")):
                continue
            eps += 1
            lay = layers(e["variant"])
            ev = [(t, NODE.match(n), st) for t, n, st, _ in (e.get("extra") or {}).get("policy_log") or []]
            ev = [(t, int(m.group(1)), m.group(2), st) for t, m, st in ev if m]
            placed_ok, placed_any = set(), set()
            for i, (t, k, kind, st) in enumerate(ev):
                if kind in ("to_pregrasp", "grasp") and k in placed_any and \
                        any(lay[j] > lay[k] for j in placed_ok if j != k):
                    buried += 1
                    examples.append((e["variant"][:7], (e.get("extra") or {}).get("seed"), round(t, 1), k))
                    placed_any.discard(k)  # count each rework once (its grasp follows its pre-grasp)
                if kind == "place":
                    placed_any.add(k)
                    if st == "ok":
                        placed_ok.add(k)
                    else:
                        failed += 1
                        nxt = next((x for x in ev[i + 1:] if x[2] in ("to_pregrasp", "grasp")), None)
                        redone += int(nxt is not None and nxt[1] == k)
    print(f"== {root}: {eps} transfer episodes | buried reworks {buried} | failed places {failed}, "
          f"redone at once {redone}" + (f" | e.g. {examples[:4]}" if examples else ""))


if __name__ == "__main__":
    for root in sys.argv[1:]:
        report(root)
