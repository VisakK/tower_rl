"""Points under the official mirror (Plan-v1.1 §1.5, §2.2): per variant TSR, LHPR, ODR, the fitted task score and the
points (score x weight / 18, weights 1 : 1 : 1 : 3 : 3), from rehearsal results (``<root>/<variant>/<tier dir>/
episodes.json``, e.g. ``rehearse.py --mirror``). Also per variant: episodes that reached success and lost it by the end
(run to 210 s), goal nodes that started after the first success, and for Task 4 the rest states of the pushed blocks.

  python scripts/mirror_points.py <root> [<root> ...] [--filter _mirror] [--by_case]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.eval.metrics import EpisodeOutcome, aggregate  # noqa: E402
from tower_rl.eval.mirror import points, task_score  # noqa: E402

GOAL_NODE = ("_grasp", "_place", "_push", "_insert", "_push_in", "_to_")


def load(roots, flt):
    by = defaultdict(list)
    for r in roots:
        for f in sorted(Path(r).glob("*/*/episodes.json")):
            if flt and flt not in f.parent.name:
                continue
            for d in json.loads(f.read_text()):
                by[d["variant"]].append((f.parent.name, EpisodeOutcome(**d)))
    return by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--filter", default="", help="only tier dirs whose name contains this")
    ap.add_argument("--by_case", action="store_true")
    args = ap.parse_args()
    by = load(args.roots, args.filter)
    total = 0.0
    print("| variant | n | TSR | LHPR | ODR | score | points | lost after success | goal nodes after success | dirs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v in sorted(by):
        outs = [o for _, o in by[v]]
        a = aggregate(outs)
        tsr, lhpr, odr = a["TSR"]["value"], a["LHPR"]["value"], a["ODR"]["value"]
        pts = points(v, tsr, lhpr, odr)
        total += pts
        lost = sum(1 for o in outs if o.extra.get("t_first_success") is not None and not o.task_success)
        after = 0
        for o in outs:
            t1 = o.extra.get("t_first_success")
            if t1 is None:
                continue
            after += sum(1 for t, nid, st, info in o.extra.get("policy_log", [])
                         if t > t1 + 0.05 and any(k in nid for k in GOAL_NODE) and "park" not in nid)
        dirs = ",".join(sorted({d for d, _ in by[v]}))
        print(f"| {v} | {len(outs)} | {100 * tsr:.1f} | {100 * lhpr:.1f} | {odr:.3f} | {task_score(tsr, lhpr, odr):.1f} | "
              f"{pts:.2f} | {lost} | {after} | {dirs} |")
        if v.startswith("task4"):
            kinds = Counter(r[0] for o in outs for r in o.extra.get("rest", []))
            strict = sum(bool(o.extra.get("task_success_strict")) for o in outs)
            radial = sum(bool(o.extra.get("task_success_radial")) for o in outs)
            fnd = [o.extra.get("foundation_moved_mm", 0) for o in outs]
            print(f"|   rest states | {dict(kinds)} | strict TSR {strict}/{len(outs)} | radial (v1 rule) {radial}/{len(outs)} | "
                  f"foundation moved max {max(fnd) if fnd else 0:.1f} mm | | | | | |")
        if args.by_case:
            cases = defaultdict(list)
            for o in outs:
                cases[o.extra.get("case", "?")].append(o)
            for c, os_ in sorted(cases.items()):
                b = aggregate(os_)
                print(f"|   {c} | {len(os_)} | {100 * b['TSR']['value']:.0f} | {100 * b['LHPR']['value']:.0f} | "
                      f"{b['ODR']['value']:.2f} | | | | | |")
    print(f"\ntotal points over the variants present: {total:.2f} (all ten tasks: max 100)")


if __name__ == "__main__":
    main()
