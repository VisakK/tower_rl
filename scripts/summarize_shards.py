"""Summarise collected switcher shards (student plan Step 6, gate G6): decisions and progress samples, non-nominal
share per family, label options, sources, episode outcomes per variant and tier.

Usage: python scripts/summarize_shards.py results/student/data/r0 [more rounds] [--out <json>]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.student.logging import load_shards  # noqa: E402
from tower_rl.student.spec import OPTIONS  # noqa: E402

FAMILY = {"task1": "transfer", "task2": "transfer", "task3": "transfer", "task4": "pushdown", "task5": "insertion"}
SOURCES = ("oracle", "switcher", "explore", "inject")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = {}
    for root in args.roots:
        d, idx = load_shards(root)
        dec = d["kind"] == 0
        fam = np.array([FAMILY[idx[e]["variant"][:5]] for e in d["ep"]])
        r = {"episodes": len(idx), "rows": int(len(dec)), "decisions": int(dec.sum()), "progress_samples": int((~dec).sum()),
             "nonnominal_share": round(float(d["nonnominal"][dec].mean()), 4) if dec.any() else None,
             "by_family": {}, "labels": dict(Counter(OPTIONS[i] for i in d["label_opt"][dec])),
             "sources": dict(Counter(SOURCES[s] for s in d["source"][dec])), "by_cell": {}}
        for f in sorted(set(fam)):
            m = dec & (fam == f)
            r["by_family"][f] = {"decisions": int(m.sum()), "nonnominal_share": round(float(d["nonnominal"][m].mean()), 4)}
        cells = defaultdict(list)
        for x in idx:
            cells[(x["variant"], x["tier"])].append(x)
        for (v, t), xs in sorted(cells.items()):
            r["by_cell"][f"{v}/{t}"] = {"episodes": len(xs), "TSR": round(float(np.mean([x["task_success"] for x in xs])), 3),
                                        "decisions_mean": round(float(np.mean([x["decisions"] for x in xs])), 1),
                                        "masked": int(sum(x["masked"] for x in xs))}
        rep[root] = r
        print(f"== {root}: {r['episodes']} episodes, {r['decisions']} decisions, {r['progress_samples']} progress samples, "
              f"non-nominal {r['nonnominal_share']}")
        print("   by family:", json.dumps(r["by_family"]))
        print("   labels:", r["labels"], " sources:", r["sources"])
        for k, c in r["by_cell"].items():
            print(f"   {k:42s} n {c['episodes']:4d}  TSR {c['TSR']:.3f}  decisions/episode {c['decisions_mean']:5.1f}  masked {c['masked']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
