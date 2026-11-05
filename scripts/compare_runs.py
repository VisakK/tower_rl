"""Compare two rehearsal result roots cell by cell (student plan Step 3: parity G3a, A/B comparisons G3b and later).

A root holds ``<variant>/<tier>[_suffix]/{metrics.json, episodes.json}`` (rehearse.py / rehearse_many.py output).
Cells are matched by variant and base tier (D0, D1, ...); ``--suffix_a`` / ``--suffix_b`` pick among several dirs of
one base tier (e.g. ``_student-oracle_graph``; "" = the bare tier). Per cell: TSR / GSR / PSR / ODR with Wilson
intervals for both, and whether every episode's node-event sequence (policy_log: time, node id, status, info) is
identical; for each episode that differs, the first divergence.

Usage: python -I scripts/compare_runs.py <rootA> <rootB> [--suffix_a S] [--suffix_b S] [--no_info] [--out f.json]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.eval.metrics import wilson  # noqa: E402


def cells(root: Path, suffix: str | None) -> dict:
    out = defaultdict(list)
    for f in sorted(root.glob("*/*/episodes.json")):
        tier = f.parent.name
        base, _, rest = tier.partition("_")
        if suffix is not None and (("_" + rest) if rest else "") != suffix:
            continue
        out[(f.parent.parent.name, base)].append(f.parent)
    for k, v in out.items():
        if len(v) > 1:
            raise SystemExit(f"{root}: several dirs for {k}: {[p.name for p in v]}; pass --suffix_a / --suffix_b")
    return {k: v[0] for k, v in out.items()}


def norm_event(e, info: bool):
    t, nid, status, inf = e
    return (round(float(t), 2), nid, status, json.dumps(inf, sort_keys=True, default=str) if info else None)


def rates(eps: list[dict]) -> dict:
    n = len(eps)
    succ = sum(bool(e["task_success"]) for e in eps)
    g = sum(sum(e["grasp_ok"]) for e in eps), sum(e["planned_g"] for e in eps)
    p = sum(sum(e["place_ok"]) for e in eps), sum(e["planned_p"] for e in eps)
    steps = sum(e["planned_g"] + e["planned_p"] + 1 - int(bool(e["task_success"])) for e in eps)
    dist = sum(sum(e["disturbed_per_op"]) for e in eps)
    return {"n": n, "TSR": succ / n if n else 0.0, "TSR_k": succ, "TSR_ci": wilson(succ, n), "GSR": g[0] / g[1] if g[1] else 0.0,
            "PSR": p[0] / p[1] if p[1] else 0.0, "ODR": dist / steps if steps else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root_a")
    ap.add_argument("root_b")
    ap.add_argument("--suffix_a", default=None)
    ap.add_argument("--suffix_b", default=None)
    ap.add_argument("--no_info", action="store_true", help="compare events without their info dicts")
    ap.add_argument("--max_diffs", type=int, default=3, help="divergences printed per cell")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = cells(Path(args.root_a), args.suffix_a), cells(Path(args.root_b), args.suffix_b)
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit(f"no common cells (A: {sorted(A)}, B: {sorted(B)})")
    lines = ["| variant | tier | n | TSR A (95 %) | TSR B (95 %) | GSR A / B | PSR A / B | ODR A / B | events identical |",
             "|---|---|---|---|---|---|---|---|---|"]
    report, diffs_all = [], []
    tot = {"a": [0, 0], "b": [0, 0], "eps": 0, "same": 0}
    for k in keys:
        ea = json.loads((A[k] / "episodes.json").read_text())
        eb = json.loads((B[k] / "episodes.json").read_text())
        ra, rb = rates(ea), rates(eb)
        by_a = {e["extra"].get("seed"): e for e in ea}
        by_b = {e["extra"].get("seed"): e for e in eb}
        seeds = [s for s in by_a if s in by_b]
        same, diffs = 0, []
        for sd in seeds:
            la = [norm_event(e, not args.no_info) for e in by_a[sd]["extra"].get("policy_log", [])]
            lb = [norm_event(e, not args.no_info) for e in by_b[sd]["extra"].get("policy_log", [])]
            if la == lb:
                same += 1
                continue
            i = next((j for j, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
            d = {"variant": k[0], "tier": k[1], "seed": sd, "case": by_a[sd]["extra"].get("case"), "index": i,
                 "a": la[i] if i < len(la) else None, "b": lb[i] if i < len(lb) else None,
                 "task_success": [by_a[sd]["task_success"], by_b[sd]["task_success"]]}
            diffs.append(d)
        diffs_all += diffs
        tot["a"][0] += ra["TSR_k"]
        tot["a"][1] += ra["n"]
        tot["b"][0] += rb["TSR_k"]
        tot["b"][1] += rb["n"]
        tot["eps"] += len(seeds)
        tot["same"] += same
        lines.append(f"| {k[0]} | {k[1]} | {ra['n']} / {rb['n']} | {ra['TSR']:.3f} ({ra['TSR_ci'][0]:.2f}-{ra['TSR_ci'][1]:.2f}) | "
                     f"{rb['TSR']:.3f} ({rb['TSR_ci'][0]:.2f}-{rb['TSR_ci'][1]:.2f}) | {ra['GSR']:.3f} / {rb['GSR']:.3f} | "
                     f"{ra['PSR']:.3f} / {rb['PSR']:.3f} | {ra['ODR']:.3f} / {rb['ODR']:.3f} | {same} / {len(seeds)} |")
        report.append({"variant": k[0], "tier": k[1], "a": ra, "b": rb, "identical": same, "episodes": len(seeds),
                       "dir_a": str(A[k]), "dir_b": str(B[k])})
    ta, tb = tot["a"], tot["b"]
    lines.append(f"| **all** | | {ta[1]} / {tb[1]} | {ta[0] / max(ta[1], 1):.3f} ({ta[0]}) | {tb[0] / max(tb[1], 1):.3f} ({tb[0]}) | | | | "
                 f"{tot['same']} / {tot['eps']} |")
    print("\n".join(lines))
    if diffs_all:
        print(f"\nfirst divergences ({len(diffs_all)} episodes differ):")
        shown = defaultdict(int)
        for d in diffs_all:
            if shown[(d["variant"], d["tier"])] >= args.max_diffs:
                continue
            shown[(d["variant"], d["tier"])] += 1
            print(f"  {d['variant']} {d['tier']} seed {d['seed']} ({d['case']}) event {d['index']}: A {d['a']} | B {d['b']} "
                  f"| success A/B {d['task_success']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"cells": report, "divergences": diffs_all, "table": lines}, indent=1, default=str))


if __name__ == "__main__":
    main()
