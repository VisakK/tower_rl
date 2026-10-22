"""Single-frame accuracy of the pose estimator on labelled frames (held-out renders or organizer JPEGs).

For each sampled frame the estimator starts either from a perturbed truth (``--mode prior``: what tracking provides,
default 4 mm / 3 deg yaw / 1 deg tilt) or from the variant's nominal layout with its coarse search (``--mode init``:
the first call of an episode), makes one call, and every block's estimate is compared with the recorded pose:
position error (mm), rotation error (deg, over the box's symmetries), confidence. ``--oracle`` replaces the network by
heatmaps splatted from the frame's own visible-corner / visible-edge labels (tests the fit alone).

Usage: python scripts/eval_perception.py --data data/perception/r1_val --checkpoint <model.pt> --mode prior \
           --frames 300 --out results/perception/eval/<name>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.perception.cameras import CAMERA_IDS  # noqa: E402
from tower_rl.perception.data import keypoints, load_shards, splat  # noqa: E402
from tower_rl.perception.estimator import ARMS, PoseEstimator  # noqa: E402
from tower_rl.perception.model import SIGMA  # noqa: E402

SYM = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]  # the box's proper symmetries


def rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    best = np.pi
    for S in SYM:
        c = (np.trace(Ra.T @ Rb @ S) - 1) / 2
        best = min(best, float(np.arccos(np.clip(c, -1, 1))))
    return float(np.degrees(best))


def viz(path: Path, est, shard, rec, maps, images, pos, quat):
    """Four views: the edge map in blue, corner peaks as yellow dots, estimated boxes red, recorded boxes green."""
    from PIL import Image, ImageDraw

    from tower_rl.perception.boxes import corners_world
    from tower_rl.perception.calib import EDGES
    from tower_rl.perception.cameras import Camera
    from tower_rl.perception.model import peaks

    if maps is None:
        maps = est.heatmaps(images)
    ce = corners_world(pos, quat, est.half.cpu().numpy())  # estimator order
    cg = corners_world(np.asarray(rec["pos"]), np.asarray(rec["quat"]), np.asarray(rec["half"]))  # record order
    tiles = []
    for ci, name in enumerate(CAMERA_IDS):
        img = cv2.imread(str(shard / "img" / f"{rec['idx']:06d}_{name}.jpg"), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
        e = maps[1][ci].float().cpu().numpy()
        img[..., 2] = np.maximum(img[..., 2], 255 * e)
        im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        cam = Camera.from_dict(rec["cams"][name])
        for corners, col in ((cg, (0, 255, 0)), (ce, (255, 40, 40))):
            for b in range(len(corners)):
                uv, z = cam.project(corners[b])
                if (z > 0).all():
                    for i, j in EDGES:
                        d.line([tuple(uv[i]), tuple(uv[j])], fill=col, width=1)
        for x, y, sc in peaks(maps[0][ci].float()).cpu().numpy():
            d.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 255, 0))
        tiles.append(im)
    g = Image.new("RGB", (1280, 960))
    for i, t in enumerate(tiles):
        g.paste(t, ((i % 2) * 640, (i // 2) * 480))
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--source", default=None, help="camera model: render (our shards) or evaluator (organizer JPEGs); "
                                                    "default from the shard records")
    ap.add_argument("--mode", choices=["prior", "init"], default="prior")
    ap.add_argument("--noise", nargs=3, type=float, default=[0.004, 3.0, 1.0], help="prior noise: pos m, yaw deg, tilt deg")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--max_t", type=float, default=None, help="only frames up to this episode time (init mode: t < 2)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", type=int, default=0, help="save overlays of the first N frames (edge map, corner peaks, "
                                                        "estimated wireframes; truth in green)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frames = load_shards(args.data, args.variants)
    if args.max_t is not None:
        frames = [f for f in frames if f[1].get("t", 0.0) <= args.max_t]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(frames), min(args.frames, len(frames)), replace=False)
    source = args.source or ("evaluator" if frames[0][1].get("source") == "dataset" else "render")
    est = PoseEstimator(None if args.oracle else args.checkpoint, device=args.device, source=source)
    rows, t_call = [], []
    for k in pick:
        shard, rec = frames[int(k)]
        est.reset(rec["variant"])
        order = [rec["block_ids"].index(b) for b in est.block_ids]  # organizer records list blocks in their own order
        gt_p = np.asarray(rec["pos"])[order]
        gt_R = geo.quat_to_mat(torch.as_tensor(np.asarray(rec["quat"])[order], dtype=torch.float64)).numpy()
        if args.mode == "prior":
            n = len(gt_p)
            dp = rng.normal(0, args.noise[0], (n, 3)) * np.array([1, 1, 0.5])
            rv = np.concatenate([rng.normal(0, np.radians(args.noise[2]), (n, 2)), rng.normal(0, np.radians(args.noise[1]), (n, 1))], 1)
            Rn = geo.quat_to_mat(geo.quat_from_rotvec(torch.as_tensor(rv))).numpy()
            est.pos = torch.as_tensor(gt_p + dp, dtype=torch.float32, device=est.dev)
            est.R = torch.as_tensor(Rn @ gt_R, dtype=torch.float32, device=est.dev)
            est.calls = 1  # tracking: no initial search
        images, maps = None, None
        if args.oracle:
            cm, em, mm, hm = [], [], [], []
            for name in CAMERA_IDS:
                ids_path = shard / "ids" / f"{rec['idx']:06d}_{name}.png"
                ids = cv2.imread(str(ids_path), cv2.IMREAD_UNCHANGED) if ids_path.exists() else None
                mm.append(torch.as_tensor((ids < 254).astype(np.float32)) if ids is not None else None)
                hm.append(torch.as_tensor((ids == 254).astype(np.float32)) if ids is not None else None)
                c, e = keypoints(rec, name, ids)
                c, e = torch.as_tensor(c, dtype=torch.float32)[None], torch.as_tensor(e, dtype=torch.float32)[None]
                cm.append(splat(c, torch.tensor([c.shape[1]]), 480, 640, SIGMA["corner"], 4)[0])
                em.append(splat(e, torch.tensor([e.shape[1]]), 480, 640, SIGMA["edge"], 3)[0])
            full = all(m is not None for m in mm)
            maps = (torch.stack(cm), torch.stack(em), torch.stack(mm) if full else None, torch.stack(hm) if full else None)
        else:
            images = {name: cv2.imread(str(shard / "img" / f"{rec['idx']:06d}_{name}.jpg"), cv2.IMREAD_COLOR)[..., ::-1] for name in CAMERA_IDS}
        t0 = time.time()
        pos, quat, conf = est(images, np.asarray(rec["q"]), np.asarray(rec["width"]), maps=maps)
        if est.dev.type == "cuda":
            torch.cuda.synchronize()
        t_call.append(time.time() - t0)
        R = geo.quat_to_mat(torch.as_tensor(quat)).numpy()
        # the score of the recorded poses with the same maps: tells a search failure from an ambiguous score
        mp = maps if maps is not None else est.heatmaps(images)
        from tower_rl.perception.model import peaks as _peaks
        cams_e = est._cams(np.asarray(rec["q"]))
        pk = [_peaks(mp[0][c].to(est.dev), est.cfg.corner_thr) for c in range(4)]
        gpos = torch.as_tensor(gt_p, dtype=torch.float32, device=est.dev)
        gR = torch.as_tensor(gt_R, dtype=torch.float32, device=est.dev)
        all_on = torch.ones(len(gt_p), dtype=torch.bool, device=est.dev)
        _, gt_ratio = est._score(cams_e, mp[0].to(est.dev), mp[1].to(est.dev), pk, gpos, gR, all_on)
        gt_cr = est.corner_stats[0].cpu().numpy()
        est._score(cams_e, mp[0].to(est.dev), mp[1].to(est.dev), pk, torch.as_tensor(pos, dtype=torch.float32, device=est.dev),
                   geo.quat_to_mat(torch.as_tensor(quat, dtype=torch.float32, device=est.dev)), all_on)
        cr, cv = (x.cpu().numpy() for x in est.corner_stats)
        # (est._hidden still holds this call's arm-hidden map)
        if len(t_call) <= args.viz:
            viz(Path(args.out).with_suffix("") / f"viz_{len(t_call):03d}.jpg", est, shard, rec, maps, images, pos, quat)
        npix = rec.get("npix")
        pinch = [est._pinch(np.asarray(rec["q"]), a)[0].cpu().numpy() for a in ARMS]
        for b, bid in enumerate(est.block_ids):
            cams_seen = sum(1 for name in CAMERA_IDS if npix and npix[name][order[b]] > 50) if npix else None
            held = any(rec["width"][i] < 0.055 and np.linalg.norm(gt_p[b] - pinch[i]) < 0.03 for i in range(2))
            rows.append({"variant": rec["variant"], "case": rec.get("case"), "block": bid, "fixed": bool(est.fixed[b]),
                         "err_mm": float(1000 * np.linalg.norm(pos[b] - gt_p[b])),
                         "err_xy_mm": float(1000 * np.linalg.norm(pos[b, :2] - gt_p[b, :2])),
                         "err_local_mm": (1000 * gt_R[b].T @ (pos[b] - gt_p[b])).round(2).tolist(),  # in the block's frame
                         "err_deg": rot_err_deg(R[b], gt_R[b]), "conf": float(conf[b]), "cams_seen": cams_seen, "held": held,
                         "inliers": float(est.stats["inliers"][b]), "ratio": float(est.stats["ratio"][b]),
                         "gt_ratio": float(gt_ratio[b]), "c_ratio": float(cr[b]), "c_vis": float(cv[b]), "gt_c_ratio": float(gt_cr[b])})
    errs = np.array([r["err_mm"] for r in rows])
    degs = np.array([r["err_deg"] for r in rows])
    conf = np.array([r["conf"] for r in rows])
    ok = conf >= 0.5

    def stats(m):
        if not m.any():
            return None
        e, d = errs[m], degs[m]
        return {"n": int(m.sum()), "pos_mm_median": round(float(np.median(e)), 3), "pos_mm_p90": round(float(np.percentile(e, 90)), 3),
                "pos_mm_p99": round(float(np.percentile(e, 99)), 3), "rot_deg_median": round(float(np.median(d)), 3),
                "rot_deg_p90": round(float(np.percentile(d, 90)), 3), "frac_gt_5mm": round(float((e > 5).mean()), 4)}

    seen = np.array([(r["cams_seen"] or 0) for r in rows])
    rep = {"args": vars(args), "source": source, "frames": len(pick), "blocks": len(rows),
           "call_s_mean": round(float(np.mean(t_call)), 3), "conf_ge_0.5": round(float(ok.mean()), 4),
           "all_confident": stats(ok), "all": stats(np.ones_like(ok)),
           "seen_2plus_cams": stats(ok & (seen >= 2)) if rows[0]["cams_seen"] is not None else None,
           "movable": stats(ok & ~np.array([r["fixed"] for r in rows])),
           "held": stats(ok & np.array([r["held"] for r in rows])),
           "per_variant": {v: stats(ok & np.array([r["variant"] == v for r in rows])) for v in sorted({r["variant"] for r in rows})}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    with open(Path(args.out).with_suffix(".rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: rep[k] for k in ("frames", "blocks", "call_s_mean", "conf_ge_0.5", "all_confident", "all", "seen_2plus_cams",
                                          "movable", "held")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
