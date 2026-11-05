"""Gate G5 (student plan Step 5): per-class position-error percentiles (p50 / p90 / p99, confident calls) and the
share of unseen calls, the camera loop (``perception_log.pkl`` of CameraPerception) vs a rehearsal with the estimator
noise model (``perception_log.pkl`` of NoiseModel), per variant / tier and pooled. Classes as in
``scripts/fit_estimator_noise.py``. A class passes when every percentile is within +-25 % of the camera loop's (or
within 0.1 mm absolute, for sub-0.4 mm values).

Usage: python scripts/compare_noise_calibration.py <camera root> <noise root> [--out <json>]
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_estimator_noise import MOVING_SPEED, episodes  # noqa: E402
from tower_rl.student.noise import CLASSES  # noqa: E402


def camera_classes(log: list[dict]) -> dict:
    """Per class: position errors (mm) of confident calls and the unseen share, CameraPerception log."""
    err, unseen = defaultdict(list), defaultdict(lambda: [0, 0])
    for seed, recs in episodes(log).items():
        prev = None
        for r in recs:
            tp = np.asarray(r["true_pos"], dtype=np.float64)
            sp = np.zeros(len(tp)) if prev is None else np.linalg.norm(tp - prev[0], axis=-1) / max(r["t"] - prev[1], 0.05)
            prev = (tp, r["t"])
            for i in range(len(tp)):
                if r["fixed"][i]:
                    c = "foundation"
                elif tp[i, 2] < 0:
                    c = "fallen"
                elif r["held"][i]:
                    c = "held"
                elif r.get("block") == i and r.get("skill"):
                    c = f"target:{r['skill']}"
                elif sp[i] > MOVING_SPEED:
                    c = "moving"
                else:
                    c = "static"
                unseen[c][1] += 1
                if r["conf"][i] >= 0.5:
                    err[c].append(r["err_mm"][i])
                else:
                    unseen[c][0] += 1
    return {c: {"err": np.array(err[c]), "unseen": unseen[c][0] / max(unseen[c][1], 1)} for c in unseen}


def noise_classes(log: list[dict]) -> dict:
    err, unseen = defaultdict(list), defaultdict(lambda: [0, 0])
    for r in log:
        for i, ci in enumerate(r["cls"]):
            c = CLASSES[int(ci)]
            unseen[c][1] += 1
            if r["conf"][i] >= 0.5:
                err[c].append(r["err_mm"][i])
            else:
                unseen[c][0] += 1
    return {c: {"err": np.array(err[c]), "unseen": unseen[c][0] / max(unseen[c][1], 1)} for c in unseen}


def pct(e):
    return [round(float(np.percentile(e, q)), 3) for q in (50, 90, 99)] if len(e) else None


def close(a, b):
    if a is None or b is None:
        return None
    return all(abs(x - y) <= max(0.25 * abs(x), 0.1) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camera_root")
    ap.add_argument("noise_root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pooled = {"camera": defaultdict(list), "noise": defaultdict(list)}
    for f in sorted(Path(args.camera_root).glob("*/*/perception_log.pkl")):
        v, tier = f.parent.parent.name, f.parent.name.split("_")[0]
        g = next(Path(args.noise_root).glob(f"{v}/{tier}_*/perception_log.pkl"), None)
        if g is None:
            continue
        for name, path, fn in (("camera", f, camera_classes), ("noise", g, noise_classes)):
            for c, d in fn(pickle.load(open(path, "rb"))["log"]).items():
                pooled[name][c].append(d)
    rep = {}
    print(f"{'class':22s} {'camera p50/p90/p99 mm':>26s} {'noise':>26s}  {'unseen cam / noise':>20s}  within 25 %")
    for c in CLASSES:
        if c not in pooled["camera"] and c not in pooled["noise"]:
            continue
        ce = np.concatenate([d["err"] for d in pooled["camera"].get(c, [])] or [np.zeros(0)])
        ne = np.concatenate([d["err"] for d in pooled["noise"].get(c, [])] or [np.zeros(0)])
        cu = float(np.mean([d["unseen"] for d in pooled["camera"].get(c, [])])) if c in pooled["camera"] else None
        nu = float(np.mean([d["unseen"] for d in pooled["noise"].get(c, [])])) if c in pooled["noise"] else None
        a, b = pct(ce), pct(ne)
        rep[c] = {"camera": a, "noise": b, "camera_n": int(len(ce)), "noise_n": int(len(ne)), "unseen_camera": cu,
                  "unseen_noise": nu, "within_25pct": close(a, b)}
        print(f"{c:22s} {str(a):>26s} {str(b):>26s}  {str(None if cu is None else round(cu, 4)):>9s} / "
              f"{str(None if nu is None else round(nu, 4)):<9s}  {rep[c]['within_25pct']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
