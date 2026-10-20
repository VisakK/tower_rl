"""Summarize noise_diag.py outputs: per file and arm, the place verdicts against truth, the true placement error at the
release along the gripper's jaw and the block's length, and the grasp outcomes.

Usage: python -I scripts/summarize_noise_diag.py results/noise_diag/<a>.jsonl [<b>.jsonl ...]
"""
import json
import sys
from collections import Counter

import numpy as np


def summarize(path: str) -> None:
    recs = [json.loads(line) for line in open(path)]
    eps = [r for r in recs if r.get("episode")]
    places = [r for r in recs if not r.get("episode") and "grasp" not in r]
    grasps = Counter()
    for e in eps:
        for _, nid, st, _ in e.get("policy_log") or []:
            if "_grasp" in nid:
                grasps[st] += 1
    print(f"{path}: TSR {sum(e['success'] for e in eps)}/{len(eps)}, places {len(places)}, grasps {dict(grasps)}")
    for arm in ("back", "left"):
        rs = [r for r in places if r["arm"] == arm]
        if not rs:
            continue
        conf = Counter()
        for r in rs:
            ok_true = r["true_xy_mm"] <= 5.0 and r["true_yaw"] <= 5.0
            conf[("ok" if r["status"] == "ok" else "fail") + "/" + ("ok" if ok_true else "FAIL")] += 1
        line = f"  {arm:4s} n {len(rs):2d} | executor/truth {dict(sorted(conf.items()))}"
        rel = [r["release"] for r in rs if r.get("release")]
        if rel:
            tr = np.array([x["true_xy_mm"] for x in rel])
            norm = np.linalg.norm(tr, axis=1)
            line += f" | true at release |xy| p50 {np.median(norm):.2f} p90 {np.percentile(norm, 90):.2f} > 5 mm {np.mean(norm > 5):.2f}"
            if all("jaw_world" in x for x in rel):
                jaw = np.array([x["jaw_world"][:2] for x in rel])
                jaw /= np.linalg.norm(jaw, axis=1, keepdims=True)
                ln = np.stack([-jaw[:, 1], jaw[:, 0]], 1)
                line += f" | sd jaw {np.std((tr * jaw).sum(1)):.2f} length {np.std((tr * ln).sum(1)):.2f}"
        print(line)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
