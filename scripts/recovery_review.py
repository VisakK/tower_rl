"""The recovery node's review sheet (Plan-v1.1 §5.3 step 2: Claude writes the next rules from clusters of failures).
From rehearsal results (``<root>/<variant>/<tier dir>/episodes.json``, runs with ``oracle_state_v2`` or a student with
the ``recovery`` flag): per variant, the recovery decisions by rule and outcome of the episode, the abstentions by cause,
and the failures no rule touched (for Task 4 the final rest states of the pushed blocks, for Task 5 the final operation
outcomes), each with a few example seeds and cases to replay.

  python scripts/recovery_review.py <root> [<root> ...] [--examples 3]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    for root in args.roots:
        for f in sorted(Path(root).glob("*/*/episodes.json")):
            eps = json.loads(f.read_text())
            v = f.parent.parent.name
            rules, abst, fails, ex = Counter(), Counter(), Counter(), defaultdict(list)
            n_rec = n_ok = 0
            for e in eps:
                x = e["extra"]
                rec = x.get("recovery") or {}
                tag = (x.get("seed"), x.get("case"))
                n_ok += bool(e["task_success"])
                if rec.get("decisions"):
                    n_rec += 1
                for d in rec.get("decisions", []):
                    key = (d["rule"], d.get("kind", ""), "success" if e["task_success"] else "fail")
                    rules[key] += 1
                    ex[("rule",) + key].append(tag)
                for a in rec.get("abstentions", []):
                    key = (a["why"], a["kind"])
                    abst[key] += 1
                    ex[("abst",) + key].append(tag)
                if not e["task_success"]:
                    if "rest" in x:
                        touched = {d["goal"] for d in rec.get("decisions", [])}
                        for k, r in enumerate(x["rest"]):
                            if r[0] != "flat clear":
                                key = (f"push {k}", r[0], "recovered-tried" if k in touched else "untouched")
                                fails[key] += 1
                                ex[("fail",) + key].append(tag)
                    else:
                        bad = [f"G{k}" for k, ok in enumerate(e["grasp_ok"]) if not ok] + \
                              [f"P{k}" for k, ok in enumerate(e["place_ok"]) if not ok]
                        key = (" ".join(bad) or "final arrangement",)
                        fails[key] += 1
                        ex[("fail",) + key].append(tag)
            print(f"===== {v} / {f.parent.name}: {len(eps)} episodes, success {n_ok}, with recovery decisions {n_rec}")
            for title, c, kind in (("recovery decisions (rule, kind, episode outcome)", rules, "rule"),
                                   ("abstentions (why, kind)", abst, "abst"), ("failures (where, what, recovery)", fails, "fail")):
                if not c:
                    continue
                print(f"  {title}:")
                for key, cnt in c.most_common():
                    print(f"    {cnt:3d}  {' | '.join(str(k) for k in key)}   e.g. {ex[(kind,) + key][:args.examples]}")


if __name__ == "__main__":
    main()
