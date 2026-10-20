"""Step 1: operations, success semantics and grasp/release geometry over the fetched subset.

Writes results/data/subset_analysis.json (per-episode records and per-variant / per-arm summaries).
Usage: python -I scripts/analyze_subset.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tower_rl.data.episode import Episode  # noqa: E402
from tower_rl.data.grasps import operation_geometry  # noqa: E402
from tower_rl.data.semantics import episode_semantics, summarize  # noqa: E402
from tower_rl.kinematics import ArmKinematics  # noqa: E402
from tower_rl.paths import DATA_DIR, RESULTS_DIR  # noqa: E402


def stats(vals):
    a = np.array(vals, dtype=float)
    return {"mean": a.mean(0).round(4).tolist(), "std": a.std(0).round(4).tolist(), "min": a.min(0).round(4).tolist(), "max": a.max(0).round(4).tolist()}


def main():
    files = sorted((DATA_DIR / "subset").glob("*/*.npz"))
    kin = ArmKinematics(dtype=torch.float64)
    sem, geo_rows = [], []
    for f in files:
        ep = Episode.load(f)
        s = episode_semantics(ep)
        s["folder"] = f.parent.name
        sem.append(s)
        for g in operation_geometry(ep, kin):
            g["folder"] = f.parent.name
            geo_rows.append(g)
    for s in sem:
        s["variant"] = s["folder"]
    by = defaultdict(list)
    for g in geo_rows:
        by[(g["folder"], g["arm"])].append(g)
    grasp_summary = {}
    for (v, arm), rs in sorted(by.items()):
        grasp_summary[f"{v}/{arm}"] = {
            "n": len(rs),
            "pinch_in_block_meas_m": stats([r["pinch_in_block_meas"] for r in rs]),
            "approach_in_block_meas": stats([r["approach_in_block_meas"] for r in rs]),
            "approach_tilt_deg_meas": stats([r["approach_tilt_deg_meas"] for r in rs]),
            "cmd_minus_meas_pinch_mm": stats([r["cmd_minus_meas_pinch_mm"] for r in rs]),
            "place_err_mm": stats([r["place_err_mm"] for r in rs if "place_err_mm" in r]),
            "place_yaw_err_deg": stats([r["place_yaw_err_deg"] for r in rs if "place_yaw_err_deg" in r]),
            "lift_m": stats([r["lift_m"] for r in rs]),
        }
    out = {"files": len(files), "semantics_summary": summarize(sem), "grasp_summary": grasp_summary,
           "episodes": sem, "operations": geo_rows}
    path = RESULTS_DIR / "data" / "subset_analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["semantics_summary"], indent=1))
    for k, v in grasp_summary.items():
        print(k, "n", v["n"], "pinch(mm)", [round(x * 1000, 2) for x in v["pinch_in_block_meas_m"]["mean"]],
              "tilt", v["approach_tilt_deg_meas"]["mean"], "approach", v["approach_in_block_meas"]["mean"],
              "cmd-meas(mm)", v["cmd_minus_meas_pinch_mm"]["mean"], "place(mm)", v["place_err_mm"]["mean"], "max", v["place_err_mm"]["max"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
