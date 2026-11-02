"""Fit the estimator noise model (student plan Step 5) from the camera-loop perception logs.

Reads every ``perception_log.pkl`` of the network runs (kp_v2 preferred; kp_v1b / kp_v1 only for classes kp_v2 lacks;
oracle / truth reference runs are skipped). Per call and block, a class:
  foundation (fixed) | fallen (truth z < 0) | held (fingers on it) | target:<skill> (the running node's block, by
  skill) | moving (truth speed > 1 cm/s) | static
and the estimation error: position error (estimate - truth, world frame) and rotation error (rotation vector of
R_est R_true^T, with the estimate's box symmetry nearest the truth), and the confidence level (1 / 0.6 / 0.3 / 0).
Contiguous same-class runs of one block become replay traces. Push targets also get P(lost | tilt, travel) and the
lengths of their lost runs.

Writes ``configs/student/noise.yaml`` (classes, sources, percentiles, push-loss table) and
``results/student/noise/traces.npz``. Usage: python scripts/fit_estimator_noise.py [--roots ...] [--prefer kp_v2 ...]
"""
import argparse
import datetime
import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.paths import REPO_ROOT, RESULTS_DIR  # noqa: E402
from tower_rl.student.noise import (CLASSES, LOST_ALONG_BINS, LOST_TILT_BINS, MAX_TRACE, MIN_TRACE, SYM, conf_index,  # noqa: E402
                                    quat_to_mat, rotvec_of)

MOVING_SPEED = 0.01


def model_of(path: str) -> str | None:
    """The keypoint model of a closed-loop run from its directory name (``.../kp_v2/...``, ``.../teacher_kp_v2/...``);
    None for the oracle-heatmap and truth reference runs."""
    parts = Path(path).parts
    if any(x.startswith(("oracle", "truth")) for x in parts):
        return None
    for m in ("kp_v2", "kp_v1b", "kp_v1"):
        if any(x == m or x.endswith("_" + m) for x in parts):
            return m
    return None


def rot_err(Re: np.ndarray, Rt: np.ndarray) -> np.ndarray:
    """Rotation vectors ``[B, 3]`` of R_est' R_true^T, R_est' = the box symmetry of the estimate nearest the truth."""
    best = None
    for S in SYM:
        Rs = Re @ S
        tr = np.einsum("bij,bij->b", Rs, Rt)  # trace(Rs Rt^T)
        if best is None:
            best, bt = Rs.copy(), tr
        else:
            m = tr > bt
            best[m], bt = Rs[m], np.maximum(bt, tr)
    return rotvec_of(best @ np.transpose(Rt, (0, 2, 1)))


def episodes(log: list[dict]) -> dict:
    by = defaultdict(list)
    for r in log:
        by[r["seed"]].append(r)
    return {k: sorted(v, key=lambda r: r["t"]) for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[str(RESULTS_DIR / "perception" / "closed_loop"),
                                                     str(RESULTS_DIR / "student" / "closed_loop")])
    ap.add_argument("--prefer", nargs="+", default=["kp_v2", "kp_v1b", "kp_v1"])
    ap.add_argument("--min_calls", type=int, default=400, help="a class takes the preferred model with this many calls")
    ap.add_argument("--out_cfg", default=str(REPO_ROOT / "configs" / "student" / "noise.yaml"))
    ap.add_argument("--out_bank", default=str(RESULTS_DIR / "student" / "noise" / "traces.npz"))
    args = ap.parse_args()
    files = sorted({f for r in args.roots for f in glob.glob(f"{r}/**/perception_log.pkl", recursive=True)})
    files = [(f, model_of(f)) for f in files if model_of(f) in args.prefer]
    print(f"{len(files)} logs: " + ", ".join(f"{m} {sum(1 for _, x in files if x == m)}" for m in args.prefer), flush=True)
    # per model and class: traces [(L, 7) arrays], plus push-target loss samples
    traces = {m: defaultdict(list) for m in args.prefer}
    calls = {m: defaultdict(int) for m in args.prefer}
    push = {m: [] for m in args.prefer}  # (tilt deg, along m, lost)
    lost_runs = {m: [] for m in args.prefer}
    sources = defaultdict(list)
    for f, model in files:
        d = pickle.load(open(f, "rb"))
        sources[model].append(str(Path(f).relative_to(RESULTS_DIR)))
        for seed, recs in episodes(d["log"]).items():
            nb = len(recs[0]["pos"])
            prev_tp, prev_t = None, None
            seg = [None] * nb  # per block: (class, rows)
            push_start, lost_len = {}, {}
            for r in recs:
                tp = np.asarray(r["true_pos"], dtype=np.float64)
                tq = np.asarray(r["true_quat"], dtype=np.float64)
                ep = np.asarray(r["pos"], dtype=np.float64)
                eq = np.asarray(r["quat"], dtype=np.float64)
                speed = np.zeros(nb) if prev_tp is None else np.linalg.norm(tp - prev_tp, axis=-1) / max(r["t"] - prev_t, 0.05)
                prev_tp, prev_t = tp, r["t"]
                dp = ep - tp
                dr = rot_err(quat_to_mat(eq), quat_to_mat(tq))
                conf = np.asarray(r["conf"], dtype=np.float64)
                tgt, skill = r.get("block"), r.get("skill")
                for i in range(nb):
                    if r["fixed"][i]:
                        c = "foundation"
                    elif tp[i, 2] < 0:
                        c = "fallen"
                    elif r["held"][i]:
                        c = "held"
                    elif tgt == i and skill:
                        c = f"target:{skill}"
                    elif speed[i] > MOVING_SPEED:
                        c = "moving"
                    else:
                        c = "static"
                    if c not in CLASSES:
                        c = "moving" if speed[i] > MOVING_SPEED else "static"
                    row = np.r_[dp[i], dr[i], conf_index(conf[i])].astype(np.float32)
                    calls[model][c] += 1
                    if seg[i] is not None and seg[i][0] == c and len(seg[i][1]) < MAX_TRACE[c]:
                        seg[i][1].append(row)
                    else:
                        if seg[i] is not None and len(seg[i][1]) >= MIN_TRACE:
                            traces[model][seg[i][0]].append(np.stack(seg[i][1]))
                        seg[i] = (c, [row])
                    if c == "target:push":
                        p0 = push_start.setdefault((seed, i), tp[i].copy())
                        along = float(np.linalg.norm(tp[i, :2] - p0[:2]))
                        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (tq[i, 1] ** 2 + tq[i, 2] ** 2), -1, 1))))
                        lost = bool(conf[i] < 0.5)
                        push[model].append((tilt, along, lost))
                        if lost:
                            lost_len[i] = lost_len.get(i, 0) + 1
                        elif lost_len.get(i):
                            lost_runs[model].append(lost_len.pop(i))
            for i in range(nb):
                if seg[i] is not None and len(seg[i][1]) >= MIN_TRACE:
                    traces[model][seg[i][0]].append(np.stack(seg[i][1]))
            lost_runs[model] += [v for v in lost_len.values() if v]
    # per class: the first preferred model with enough calls
    bank, cfg_classes = {}, {}
    for c in CLASSES:
        chosen = next((m for m in args.prefer if calls[m][c] >= args.min_calls and traces[m][c]), None)
        if chosen is None:
            chosen = max(args.prefer, key=lambda m: calls[m][c])
        tr = traces[chosen][c]
        if not tr:
            print(f"  class {c}: no traces", flush=True)
            continue
        data = np.concatenate(tr)
        lengths = np.array([len(t) for t in tr])
        bank[f"{c}__data"] = data
        bank[f"{c}__lengths"] = lengths
        err = np.linalg.norm(data[:, :3], axis=-1) * 1000
        rot = np.degrees(np.linalg.norm(data[:, 3:6], axis=-1))
        lv = np.bincount(data[:, 6].astype(int), minlength=4) / len(data)
        seen = data[:, 6] <= 1
        cfg_classes[c] = {"model": chosen, "calls": int(calls[chosen][c]), "traces": len(tr),
                          "trace_len_median": float(np.median(lengths)),
                          "pos_mm": [round(float(np.percentile(err[seen], q)), 3) if seen.any() else None for q in (50, 90, 99)],
                          "rot_deg": [round(float(np.percentile(rot[seen], q)), 3) if seen.any() else None for q in (50, 90, 99)],
                          "conf_levels": [round(float(x), 4) for x in lv]}
        print(f"  {c:22s} {chosen:7s} calls {calls[chosen][c]:7d} traces {len(tr):5d} pos mm p50/p90/p99 "
              f"{cfg_classes[c]['pos_mm']} conf levels {cfg_classes[c]['conf_levels']}", flush=True)
    # push target losses: P(lost | tilt, along) from the preferred model with data
    pm = next((m for m in args.prefer if len(push[m]) >= 200), max(args.prefer, key=lambda m: len(push[m])))
    arr = np.array(push[pm]) if push[pm] else np.zeros((0, 3))
    table = []
    for ti in range(len(LOST_TILT_BINS) - 1):
        rowp = []
        for ai in range(len(LOST_ALONG_BINS) - 1):
            m = (arr[:, 0] >= LOST_TILT_BINS[ti]) & (arr[:, 0] < LOST_TILT_BINS[ti + 1]) & \
                (arr[:, 1] >= LOST_ALONG_BINS[ai]) & (arr[:, 1] < LOST_ALONG_BINS[ai + 1]) if len(arr) else np.zeros(0, bool)
            n = int(m.sum())
            rowp.append(round(float((arr[m, 2].sum() + 0.5) / (n + 1)), 4) if n else None)  # smoothed
        table.append(rowp)
    runs = lost_runs[pm]
    cfg = {"written": datetime.datetime.now().isoformat(timespec="minutes"), "by": "scripts/fit_estimator_noise.py",
           "source_logs": {m: sources[m] for m in args.prefer if sources[m]}, "moving_speed_m_s": MOVING_SPEED,
           "classes": cfg_classes, "bank": str(Path(args.out_bank).resolve().relative_to(REPO_ROOT)),
           "push_lost": {"model": pm, "samples": int(len(arr)), "tilt_bins_deg": list(LOST_TILT_BINS),
                         "along_bins_m": list(LOST_ALONG_BINS), "p_lost": table,
                         "run_lengths": [int(x) for x in runs], "coast_decay": 0.8, "coast_max_calls": 5}}
    Path(args.out_bank).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_bank, **bank)
    Path(args.out_cfg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_cfg).write_text("# written by scripts/fit_estimator_noise.py; do not edit by hand\n" +
                                  yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=120))
    print(f"push losses ({pm}, {len(arr)} calls): P(lost | tilt, along) = {json.dumps(table)}; lost runs {len(runs)} "
          f"(median {np.median(runs) if runs else 0})", flush=True)
    print(f"wrote {args.out_cfg} and {args.out_bank}", flush=True)


if __name__ == "__main__":
    main()
