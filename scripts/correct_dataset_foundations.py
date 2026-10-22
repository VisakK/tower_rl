"""Correct the foundation poses of organizer frame records (scripts/fetch_dataset_frames.py shards) to where the
evaluator's renderer drew them.

In the organizers' data the recorded foundation poses (and the goals built on the same jittered tower centres) can lie
up to ~5 mm from the drawn foundation blocks (BUILDLOG 2026-10-21: edge registration of static views; movable blocks
register to < 0.1 px). Foundations never move within an episode, so each pad (its three blocks) gets one rigid
correction (dx, dy, dyaw) per episode: the one that maximises the image gradient along the pad's projected visible
edges in both static cameras over several frames (visibility from the recorded poses, minus pixels an arm covers in the
ids maps). The records keep the recorded poses in ``pos_recorded`` / ``quat_recorded``; run
scripts/label_dataset_frames.py again afterwards (visibility bits and ids maps follow the poses).

Usage: python scripts/correct_dataset_foundations.py --root data/perception/dataset_train [--dry_run]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.perception.boxes import corners_world, raycast  # noqa: E402
from tower_rl.perception.calib import EDGES  # noqa: E402
from tower_rl.perception.cameras import Camera  # noqa: E402

STATIC = ("left_back_camera", "right_front_camera")


def grad_image(img_rgb: np.ndarray) -> np.ndarray:
    g = ndimage.gaussian_filter(img_rgb.astype(np.float64).mean(-1), 0.7)
    return np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))


def edge_points_3d(corners: np.ndarray, n: int = 30) -> np.ndarray:
    """Points ``[B * 12 * n, 3]`` along every edge of boxes with corners ``[B, 8, 3]``."""
    s = np.linspace(0.05, 0.95, n)[:, None]
    return np.concatenate([corners[b, i] * (1 - s) + corners[b, j] * s for b in range(len(corners)) for i, j in EDGES])


def visible(cam: Camera, pts: np.ndarray, depth: np.ndarray, ids: np.ndarray | None) -> np.ndarray:
    uv, z = cam.project(pts)
    ui, vi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    ok = (z > 0) & (ui >= 2) & (ui < cam.width - 2) & (vi >= 2) & (vi < cam.height - 2)
    ui, vi = np.clip(ui, 0, cam.width - 1), np.clip(vi, 0, cam.height - 1)
    ok &= depth[vi, ui] >= z - 0.003
    if ids is not None:  # not where an arm or a finger covers a block
        occ = cv2.dilate((ids == 254).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ok &= ~occ[vi, ui]
    return ok


def register_pad(frames, shard: Path, idx: list[int], r_mm: float = 8.0, yaw_deg: float = 0.0):
    """Best rigid (dx, dy, dyaw) of the pad blocks ``idx`` (record order) over ``frames``, coarse to fine (1 mm /
    0.5 deg, then 0.2 mm / 0.1 deg around the best; by default no yaw: the dataset's offsets are translations, dyaw
    <= 0.1 deg in a 1.5 deg search); returns the shift, the score at zero and at the best, and the number of views."""
    terms = []
    for rec in frames:
        pos, quat, half = np.asarray(rec["pos"]), np.asarray(rec["quat"]), np.asarray(rec["half"])
        pts = edge_points_3d(corners_world(pos, quat, half)[idx])
        for name in STATIC:
            cam = Camera.from_dict(rec["cams"][name])
            img = cv2.imread(str(shard / "img" / f"{rec['idx']:06d}_{name}.jpg"), cv2.IMREAD_COLOR)[..., ::-1]
            ids_p = shard / "ids" / f"{rec['idx']:06d}_{name}.png"
            ids = cv2.imread(str(ids_p), cv2.IMREAD_UNCHANGED) if ids_p.exists() else None
            depth = raycast(cam, pos, quat, half)[0].numpy()
            m = visible(cam, pts, depth, ids)
            if m.sum() >= 30:
                terms.append((cam, grad_image(img), pts[m]))
    if not terms:
        return None
    centre = np.mean([np.asarray(frames[0]["pos"])[i] for i in idx], 0)

    def score(dx, dy, dyaw):
        c, s = math.cos(math.radians(dyaw)), math.sin(math.radians(dyaw))
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        d = np.array([dx, dy, 0.0]) / 1000
        sc = 0.0
        for cam, g, p in terms:
            uv, _ = cam.project((p - centre) @ Rz.T + centre + d)
            sc += ndimage.map_coordinates(g, [uv[:, 1], uv[:, 0]], order=1).mean()
        return sc / len(terms)

    def grid(c, r, st, ry, sty):
        return [(score(dx, dy, dyaw), dx, dy, dyaw) for dyaw in np.arange(c[2] - ry, c[2] + ry + 1e-9, sty)
                for dx in np.arange(c[0] - r, c[0] + r + 1e-9, st) for dy in np.arange(c[1] - r, c[1] + r + 1e-9, st)]

    best = max(grid((0.0, 0.0, 0.0), r_mm, 1.0, yaw_deg, 0.5))
    best = max(grid(best[1:], 1.0, 0.2, min(yaw_deg, 0.5), 0.1))
    return {"dx_mm": round(float(best[1]), 2), "dy_mm": round(float(best[2]), 2), "dyaw_deg": round(float(best[3]), 2),
            "score_zero": round(score(0.0, 0.0, 0.0), 2), "score_best": round(float(best[0]), 2), "views": len(terms), "centre": centre}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", type=int, default=4, help="frames per episode used for the registration")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    report = {}
    for fj in sorted(Path(args.root).glob("*/*/frames.jsonl")):
        shard = fj.parent
        recs = [json.loads(line) for line in open(fj)]
        if "foundation_correction" in recs[0]:  # corrected before: keep (idempotent; new shards get theirs)
            report[f"{shard.parent.name}/{shard.name}"] = recs[0]["foundation_correction"]
            continue
        lay = Layout.load(recs[0]["variant"])
        ids = recs[0]["block_ids"]
        # frames with both grippers away from the pads (open, high) register best; take the first ones (arms at home)
        frames = sorted(recs, key=lambda r: r["t"])[: args.frames]
        out = {}
        for pad in ("source", "destination"):
            idx = [ids.index(b) for b in lay.pads[pad]["foundation_blocks"]]
            reg = register_pad(frames, shard, idx)
            if reg is None:
                continue
            centre = reg.pop("centre")
            out[pad] = reg
            if args.dry_run:
                continue
            c, s = math.cos(math.radians(reg["dyaw_deg"])), math.sin(math.radians(reg["dyaw_deg"]))
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            d = np.array([reg["dx_mm"], reg["dy_mm"], 0.0]) / 1000
            qz = np.array([math.cos(math.radians(reg["dyaw_deg"]) / 2), 0, 0, math.sin(math.radians(reg["dyaw_deg"]) / 2)])
            for rec in recs:
                rec.setdefault("pos_recorded", [list(p) for p in rec["pos"]])
                rec.setdefault("quat_recorded", [list(q) for q in rec["quat"]])
                for i in idx:
                    p0 = np.asarray(rec["pos_recorded"][i])
                    q0 = np.asarray(rec["quat_recorded"][i])  # wxyz
                    rec["pos"][i] = ((p0 - centre) @ Rz.T + centre + d).tolist()
                    w1, x1, y1, z1 = qz
                    w2, x2, y2, z2 = q0
                    rec["quat"][i] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
                rec.setdefault("foundation_correction", {})[pad] = reg
        report[f"{shard.parent.name}/{shard.name}"] = out
        print(f"[found] {shard.parent.name} {shard.name}: " + json.dumps(out), flush=True)
        if not args.dry_run:
            with open(fj, "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
    shifts = np.array([[v["dx_mm"], v["dy_mm"]] for o in report.values() for v in o.values()])
    if len(shifts):
        n = np.linalg.norm(shifts, axis=1)
        print("REPORT_JSON " + json.dumps({"pads": len(n), "shift_mm_median": round(float(np.median(n)), 2),
                                           "shift_mm_p90": round(float(np.percentile(n, 90)), 2), "shift_mm_max": round(float(n.max()), 2),
                                           "frac_gt_1mm": round(float((n > 1).mean()), 3)}), flush=True)
    (Path(args.root) / "foundation_corrections.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
