#!/usr/bin/env python3
"""Studentv1_3_plan Step 6: REF vs CAND from cpu_v3_gates.sh's run directory: per variant the points (mean over the
seeds: the task score is linear in TSR, LHPR and ODR, so with equal episodes per seed this is the pooled value),
successes, LHPR, guard vetoes; totals; then the extra cells (rework_report / grip_report).

  python3 scripts/gates_compare.py results/v1_3/gates/<name>
"""
import glob
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

W = {"task1": 1, "task2": 1, "task3": 1, "task4": 3, "task5": 3}


def cells(root: Path, arm: str):
    out = defaultdict(list)
    for d in sorted(root.glob(f"{arm}_[0-9]*")):
        for f in glob.glob(f"{d}/**/episodes.json", recursive=True):
            for e in json.load(open(f)):
                out[e["variant"]].append(e)
    return out


def score(eps):
    n = len(eps)
    tsr = sum(bool(e["task_success"]) for e in eps) / n
    lh = sum((sum(e["grasp_ok"]) + sum(e["place_ok"])) / max(1, e["planned_g"] + e["planned_p"]) for e in eps) / n
    odr = sum(sum(e.get("disturbed_per_op") or []) / max(1, e["planned_g"] + e["planned_p"] + 1 - bool(e["task_success"]))
              for e in eps) / n
    s = 0.34 * 100 * tsr + 0.62 * 100 * lh + 3.7 * (1 - odr)
    return tsr, lh, odr, s


def main(root: Path):
    ref, cand = cells(root, "REF"), cells(root, "CAND")
    tot = {"REF": 0.0, "CAND": 0.0}
    print(f"{'variant':32s} {'n':>3s} | REF succ  LHPR  vetoes  points | CAND succ  LHPR  vetoes  points | diff")
    for v in sorted(set(ref) | set(cand)):
        line = f"{v:32s} {len(cand.get(v, [])):3d} |"
        pts = {}
        for name, c in (("REF", ref), ("CAND", cand)):
            eps = c.get(v, [])
            if not eps:
                line += " " * 32 + "|"
                continue
            tsr, lh, odr, s = score(eps)
            p = s * W[v[:5]] / 18
            pts[name] = p
            tot[name] += p
            vet = sum((e.get("extra") or {}).get("vetoed", 0) for e in eps)
            line += f" {sum(bool(e['task_success']) for e in eps):3d}/{len(eps):<3d} {100 * lh:5.1f} {vet:6d} {p:7.2f} |"
        if len(pts) == 2:
            line += f" {pts['CAND'] - pts['REF']:+.2f}"
        print(line)
    print(f"{'TOTAL':32s}     | REF {tot['REF']:.2f}   CAND {tot['CAND']:.2f}   diff {tot['CAND'] - tot['REF']:+.2f} "
          "(points over the variants present; same seeds)")
    here = Path(__file__).parent
    extra = [str(d) for d in sorted(root.glob("*_inject")) + sorted(root.glob("*_edge"))]
    if extra:
        subprocess.run([sys.executable, str(here / "rework_report.py"), *extra])
        subprocess.run([sys.executable, str(here / "grip_report.py"), *extra])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
