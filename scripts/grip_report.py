#!/usr/bin/env python3
"""Off-centre grips and their places (Studentv1_3_plan Steps 2-3 and the gates): per run directory (rehearse
--out_root, any depth), every place node with the grip it followed, bucketed by the grip's offset along the block.

Per bucket: places, node verdict ok, the place's truth success (the operation tracker's P for that goal, latched),
release height (the block centre above its goal centre when the gripper opened: ``release_above_mm``), releases > 25 mm,
planned off-centre places; per run: TSR, LHPR, ODR-relevant disturbances.

  python3 scripts/grip_report.py <run dir> [<run dir> ...]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 1e9)]


def bucket(x):
    a = abs(x)
    return next(f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}" for lo, hi in BUCKETS if lo <= a < hi)


def report(root: str):
    rows, eps = [], []
    for f in sorted(glob.glob(f"{root}/**/episodes.json", recursive=True)):
        for e in json.load(open(f)):
            ev = (e.get("extra") or {}).get("policy_log") or []
            eps.append(e)
            grip = {}
            for t, nid, st, info in ev:
                info = info or {}
                if nid.endswith("_grasp") and st == "ok" and "grasp_offset_mm" in info:
                    grip[nid.split("_")[0]] = info["grasp_offset_mm"][0]
                if "_place" in nid and "place_xy_mm" in info:
                    k = nid.split("_")[0]
                    g = info.get("grip_along_mm", grip.get(k))
                    goal = int(k[1:]) if k[1:].isdigit() else None
                    truth = None
                    if goal is not None and goal < len(e.get("place_ok") or []):
                        truth = bool(e["place_ok"][goal])
                    rows.append({"variant": e["variant"][:7], "grip": g, "ok": st == "ok", "truth": truth,
                                 "xy": info["place_xy_mm"], "above": info.get("release_above_mm"),
                                 "offcentre": bool(info.get("offcentre"))})
    print(f"== {root}: {len(eps)} episodes, {len(rows)} places")
    if not eps:
        return
    tsr = sum(bool(e["task_success"]) for e in eps) / len(eps)
    lh = [(sum(e["grasp_ok"]) + sum(e["place_ok"])) / max(1, e["planned_g"] + e["planned_p"]) for e in eps]
    dist = sum(sum(e.get("disturbed_per_op") or []) for e in eps)
    print(f"   TSR {100 * tsr:.1f} %  LHPR {100 * sum(lh) / len(lh):.1f} %  disturbed non-target occurrences {dist}")
    by = defaultdict(list)
    for r in rows:
        by[bucket(r["grip"]) if r["grip"] is not None else "unknown"].append(r)
    print("   grip |x| mm   places  node ok  truth P  release p50 / p90 mm  releases > 25 mm  planned (off-centre)")
    for b in [f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}" for lo, hi in BUCKETS] + ["unknown"]:
        rs = by.get(b)
        if not rs:
            continue
        ab = sorted(r["above"] for r in rs if r["above"] is not None)
        tr = [r["truth"] for r in rs if r["truth"] is not None]
        p = (lambda q: ab[min(len(ab) - 1, int(q * len(ab)))]) if ab else (lambda q: float("nan"))
        print(f"   {b:>9s}  {len(rs):6d}  {sum(r['ok'] for r in rs) / len(rs):7.2f}  "
              f"{(sum(tr) / len(tr)) if tr else float('nan'):7.2f}  {p(0.5):8.1f} / {p(0.9):5.1f}  "
              f"{sum(a > 25 for a in ab):16d}  {sum(r['offcentre'] for r in rs):10d}")


if __name__ == "__main__":
    for root in sys.argv[1:]:
        report(root)
