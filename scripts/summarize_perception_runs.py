"""Table of closed-loop perception rehearsals (``rehearse.py --perception``): per variant and tier the challenge metrics
(metrics.json) and the estimation errors (perception.json): all blocks, held blocks, and the target block per skill.

Usage: python -I scripts/summarize_perception_runs.py results/perception/closed_loop/<run> [...]
"""
import json
import sys
from pathlib import Path


def main():
    rows = []
    for root in sys.argv[1:]:
        for pj in sorted(Path(root).glob("*/*/perception.json")):
            m = json.loads((pj.parent / "metrics.json").read_text())["metrics"]
            p = json.loads(pj.read_text())
            cell = f"{pj.parent.parent.name} {pj.parent.name}"
            tgt = {k.split("@")[1]: v for k, v in p.items() if k.startswith("target@")}

            def f(x, k="pos_mm_p90"):
                return f"{x[k]:.2f}" if x else "-"

            rows.append(f"| {cell} | {m['TSR']['value']:.2f} ({m['TSR']['k']}/{m['TSR']['n']}) | {m['PSR']['value']:.2f} | {m['ODR']['value']:.3f} | "
                        f"{f(p.get('all'), 'pos_mm_median')} / {f(p.get('all'))} / {f(p.get('all'), 'pos_mm_p99')} | {f(p.get('held'))} | "
                        + ", ".join(f"{k} {f(v)}" for k, v in sorted(tgt.items())) + f" | {p.get('calls')} |")
    print("| variant / tier | TSR | PSR | ODR | all blocks mm (median / p90 / p99) | held p90 mm | target block p90 mm by skill | calls |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    sys.exit(main())
