"""Tracking accuracy of the pose estimator through whole organizer episodes (the evaluator's renderer, with real grasps,
transports, placements and arm occlusions): the four JPEG streams at 10 Hz (every third 30 Hz frame) and the
recorded joints / widths go through the estimator call by call as on the policy server; every block's estimate is
compared with the recorded pose.

Reports error statistics by block state (static; held: centre within 3 cm of a closed gripper; moving: > 1 cm/s) and,
1 s after every release, the placed block's error and its error relative to the destination foundation (the
placement-relevant quantity: shared errors cancel). ``--foundation_corrections`` (scripts/correct_dataset_foundations.py's
report) moves the recorded foundation poses to where the renderer drew them (they differ by up to ~7 mm).

Usage: python scripts/eval_perception_sequence.py --hdf5 data/episodes_full/data/<variant>/<episode>.hdf5 \
           --checkpoint <model.pt> --out results/perception/eval/seq_<name>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.perception.cameras import CAMERA_IDS  # noqa: E402
from tower_rl.perception.closed_loop import rot_err_deg  # noqa: E402
from tower_rl.perception.estimator import ARMS, PoseEstimator  # noqa: E402


def stats(e, d):
    if len(e) == 0:
        return None
    e, d = np.asarray(e), np.asarray(d)
    return {"n": int(len(e)), "pos_mm_median": round(float(np.median(e)), 3), "pos_mm_p90": round(float(np.percentile(e, 90)), 3),
            "pos_mm_p99": round(float(np.percentile(e, 99)), 3), "pos_mm_max": round(float(e.max()), 2),
            "rot_deg_median": round(float(np.median(d)), 3), "rot_deg_p90": round(float(np.percentile(d, 90)), 3),
            "frac_gt_5mm": round(float((e > 5).mean()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--foundation_corrections", default=None, help="foundation_corrections.json of the episodes' frame shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", action="store_true", help="also write every call's estimates and truth (<out>.trace.npz)")
    args = ap.parse_args()
    traces, trace_ids = {}, {}
    corr = json.loads(Path(args.foundation_corrections).read_text()) if args.foundation_corrections else {}
    est = PoseEstimator(args.checkpoint, device=args.device, source="evaluator")
    report = {}
    for path in args.hdf5:
        variant = Path(path).parent.name
        lay = Layout.load(variant)
        with h5py.File(path, "r") as f:
            ids = json.loads(f["sim"].attrs["block_ids"])  # a JSON list
            bp = f["sim/block_poses"][()]
            fc = corr.get(f"{variant}/{Path(path).stem}", {})
            for pad, c in fc.items():  # (translations; the corrections carry no yaw by default)
                for b in lay.pads[pad]["foundation_blocks"]:
                    bp[:, ids.index(b), 0] += c["dx_mm"] / 1000
                    bp[:, ids.index(b), 1] += c["dy_mm"] / 1000
            qpos = f["observations/qpos"][()]
            T = len(qpos)
            est.reset(variant)
            order = [ids.index(b) for b in est.block_ids]
            frames = list(range(0, T, args.stride))
            cats = {"static": ([], []), "held": ([], []), "moving": ([], [])}
            conf_low = 0
            t_call = []
            held_prev = [None, None]
            releases = []
            dest = [est.block_ids.index(b) for b in lay.pads["destination"]["foundation_blocks"]]
            last_est = {}
            for k, t in enumerate(frames):
                imgs = {n: cv2.imdecode(np.frombuffer(bytes(f[f"observations/images/{n}"][t]), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                        for n in CAMERA_IDS}
                q = np.stack([qpos[t, 0:7], qpos[t, 8:15]])
                w = np.array([qpos[t, 7], qpos[t, 15]])
                t0 = time.time()
                pos, quat, conf = est(imgs, q, w)
                if est.dev.type == "cuda":
                    torch.cuda.synchronize()
                t_call.append(time.time() - t0)
                gp = bp[t, order, :3]
                gq = np.concatenate([bp[t, order, 6:7], bp[t, order, 3:6]], 1)
                err = 1000 * np.linalg.norm(pos - gp, axis=-1)
                deg = rot_err_deg(geo.quat_to_mat(torch.as_tensor(quat)).numpy(), geo.quat_to_mat(torch.as_tensor(gq)).numpy())
                tp = min(t + args.stride, T - 1)
                speed = np.linalg.norm(bp[tp, order, :3] - gp, axis=-1) / max(1e-6, (tp - t) / 30.0)
                pinch = [est._pinch(q, a)[0].cpu().numpy() for a in ARMS]
                held = np.zeros(len(gp), dtype=bool)
                for i in range(2):
                    if w[i] < 0.055:
                        held |= np.linalg.norm(gp - pinch[i], axis=-1) < 0.03
                ok = conf >= 0.5
                conf_low += int((~ok).sum())
                for b in range(len(gp)):
                    c = "held" if held[b] else ("moving" if speed[b] > 0.01 else "static")
                    if ok[b]:
                        cats[c][0].append(err[b])
                        cats[c][1].append(deg[b])
                # releases: an arm's held block, 1 s (10 calls) after its fingers open
                for i in range(2):
                    hb = int(np.argmin(np.linalg.norm(gp - pinch[i], axis=-1))) if w[i] < 0.055 else None
                    if hb is not None and np.linalg.norm(gp[hb] - pinch[i]) > 0.03:
                        hb = None
                    if held_prev[i] is not None and hb is None and w[i] > 0.065:
                        releases.append({"block": held_prev[i], "call": k + 10, "t": round(t / 30.0, 1)})
                    held_prev[i] = hb if hb is not None else (held_prev[i] if w[i] <= 0.065 else None)
                last_est[k] = (pos.copy(), gp.copy())
                if args.trace:
                    traces.setdefault(variant, []).append((t / 30.0, pos.copy(), gp.copy(), conf.copy()))
                    trace_ids[variant] = list(est.block_ids)
            rel = []
            for r in releases:
                if r["call"] in last_est:
                    pe, pt = last_est[r["call"]]
                    b = r["block"]
                    fe, ft = pe[dest].mean(0), pt[dest].mean(0)
                    rel.append({"t": r["t"], "block": est.block_ids[b], "err_mm": round(1000 * float(np.linalg.norm(pe[b] - pt[b])), 2),
                                "rel_dest_mm": round(1000 * float(np.linalg.norm((pe[b] - fe) - (pt[b] - ft))), 2),
                                "rel_dest_xy_mm": round(1000 * float(np.linalg.norm((pe[b] - fe)[:2] - (pt[b] - ft)[:2])), 2)})
            report[variant] = {"episode": Path(path).stem, "foundation_correction": fc, "calls": len(frames),
                               "call_s_mean": round(float(np.mean(t_call)), 3),
                               "call_s_max": round(float(np.max(t_call)), 3),
                               "frac_conf_lt_0.5": round(conf_low / (len(frames) * len(est.block_ids)), 4),
                               **{c: stats(*v) for c, v in cats.items()}, "placements": rel}
            print(f"[seq] {variant}: " + json.dumps(report[variant]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    if args.trace:
        np.savez_compressed(Path(args.out).with_suffix(".trace.npz"), **{
            f"{v}__{k}": np.stack([c[i] for c in calls]) for v, calls in traces.items() for i, k in enumerate(("t", "pos", "true", "conf"))},
            block_ids=np.array(json.dumps(trace_ids)))


if __name__ == "__main__":
    sys.exit(main())
