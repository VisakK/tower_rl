"""Train the pose estimator's keypoint network (tower_rl.perception.model) on rendered shards and report detection
metrics on held-out shards: corner recall / precision within 3 px and the sub-pixel error of matched corners, edge
recall / precision, visible-block mask IoU. GPU, one job at a time (wrap in scripts/run_logged.sh).

Usage: python scripts/train_perception.py --train data/perception/r1 --val data/perception/r1_val \
           --iters 20000 --out results/perception/models/kp_v1
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.perception.data import KeypointDataset, load_shards, splat  # noqa: E402
from tower_rl.perception.model import SIGMA, KeypointNet, peaks, quality_focal  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--train", nargs="+", required=True, help="shard roots; <root>:<k> samples that root's frames k times as often")
ap.add_argument("--val", nargs="+", default=[])
ap.add_argument("--iters", type=int, default=20000)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--crop", nargs=2, type=int, default=[512, 384])
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--val_every", type=int, default=2000)
ap.add_argument("--val_images", type=int, default=400)
ap.add_argument("--resume", default=None)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(vars(args), indent=1))
dev = torch.device(args.device)
train_frames, frame_w = [], []
for spec in args.train:
    root, _, k = spec.partition(":")
    fr = load_shards([root])
    train_frames += fr
    frame_w += [float(k or 1)] * len(fr)
    print(f"[train] {root}: {len(fr)} frames, weight {float(k or 1)}", flush=True)
val_frames = load_shards(args.val) if args.val else []
print(f"[train] {len(train_frames)} train frames, {len(val_frames)} val frames", flush=True)
ds = KeypointDataset(train_frames, augment=True, crop=tuple(args.crop))
# views without visible blocks (a wrist camera on the table) teach little: sampled at 0.15
w = [frame_w[fi] * (1.0 if sum(train_frames[fi][1].get("npix", {}).get(cam, [1000])) > 200 else 0.15) for fi, cam in ds.items]
sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True, pin_memory=True,
                    persistent_workers=args.workers > 0)
model = KeypointNet(pretrained=args.resume is None).to(dev)
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=dev)["model"])
enc = [p for n, p in model.named_parameters() if n.split(".")[0] in ("stem", "l1", "l2", "l3", "l4")]
dec = [p for n, p in model.named_parameters() if n.split(".")[0] not in ("stem", "l1", "l2", "l3", "l4")]
opt = torch.optim.AdamW([{"params": enc, "lr": args.lr * 0.3}, {"params": dec, "lr": args.lr}], weight_decay=1e-4)
base_lr = [g["lr"] for g in opt.param_groups]
WARM = 500


def lr_at(it):
    return min(1.0, (it + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1.0, it / args.iters)))


def targets(c, nc, e, ne, h, w):
    return splat(c, nc, h, w, SIGMA["corner"], 4), splat(e, ne, h, w, SIGMA["edge"], 3)


def losses(logits, mask, tc, te):
    known = mask >= 0
    if known.any():
        lm = F.binary_cross_entropy_with_logits(logits[:, 2][known], (mask[known] == 1).float())
        lh = F.binary_cross_entropy_with_logits(logits[:, 3][known], (mask[known] == 2).float(), pos_weight=torch.tensor(5.0, device=mask.device))
    else:
        lm = lh = logits.sum() * 0
    return {"corner": quality_focal(logits[:, 0], tc), "edge": quality_focal(logits[:, 1], te), "mask": lm, "hidden": lh}


@torch.no_grad()
def evaluate(n_images: int) -> dict:
    model.eval()
    vds = KeypointDataset(val_frames, augment=False)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(vds), min(n_images, len(vds)), replace=False)
    tot = {"gt_c": 0, "hit_c": 0, "det_c": 0, "good_det_c": 0, "gt_e": 0, "hit_e": 0, "det_e": 0, "good_det_e": 0,
           "inter": 0, "union": 0, "h_inter": 0, "h_union": 0}
    errs, losses_v = [], []
    for k in pick:
        x, mask, c, nc, e, ne = vds[int(k)]
        x = x[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = model(x).float()
        h, w = x.shape[-2:]
        tc, te = targets(c[None].to(dev), torch.tensor([nc], device=dev), e[None].to(dev), torch.tensor([ne], device=dev), h, w)
        losses_v.append({k2: float(v) for k2, v in losses(lg, mask[None].to(dev), tc, te).items()})
        prob = lg.sigmoid()[0]
        det = peaks(prob[0]).cpu().numpy()
        gt = c[:nc].numpy()
        if len(gt) and len(det):
            d = np.linalg.norm(gt[:, None] - det[None, :, :2], axis=-1)
            near = d.min(1)
            tot["hit_c"] += int((near < 3).sum())
            errs += near[near < 3].tolist()
            tot["good_det_c"] += int((d.min(0) < 3).sum())
        tot["gt_c"] += len(gt)
        tot["det_c"] += len(det)
        pe = (prob[1] > 0.5).cpu().numpy()
        ge = e[:ne].numpy()
        img = np.ones((h, w), np.uint8)
        if len(ge):
            img[np.rint(ge[:, 1]).astype(int), np.rint(ge[:, 0]).astype(int)] = 0
            dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
            pmax = F.max_pool2d(prob[1][None, None], 3, 1, 1)[0, 0].cpu().numpy()
            tot["hit_e"] += int((pmax[np.rint(ge[:, 1]).astype(int), np.rint(ge[:, 0]).astype(int)] > 0.5).sum())
            tot["good_det_e"] += int((dist[pe] <= 1.5).sum())
        tot["gt_e"] += len(ge)
        tot["det_e"] += int(pe.sum())
        m = mask.numpy()
        if (m >= 0).any():
            pm = (prob[2] > 0.5).cpu().numpy()
            tot["inter"] += int((pm & (m == 1)).sum())
            tot["union"] += int((pm | (m == 1)).sum())
            ph = (prob[3] > 0.5).cpu().numpy()
            tot["h_inter"] += int((ph & (m == 2)).sum())
            tot["h_union"] += int((ph | (m == 2)).sum())
    model.train()
    errs = np.asarray(errs) if errs else np.zeros(1)
    return {"corner_recall_3px": round(tot["hit_c"] / max(1, tot["gt_c"]), 4),
            "corner_precision_3px": round(tot["good_det_c"] / max(1, tot["det_c"]), 4),
            "corner_err_px_median": round(float(np.median(errs)), 3), "corner_err_px_p90": round(float(np.percentile(errs, 90)), 3),
            "edge_recall": round(tot["hit_e"] / max(1, tot["gt_e"]), 4), "edge_precision_1.5px": round(tot["good_det_e"] / max(1, tot["det_e"]), 4),
            "mask_iou": round(tot["inter"] / max(1, tot["union"]), 4), "hidden_iou": round(tot["h_inter"] / max(1, tot["h_union"]), 4),
            "val_loss": {k: round(float(np.mean([v[k] for v in losses_v])), 4) for k in losses_v[0]}}


log = open(out / "train_log.jsonl", "a")
it, t0, run = 0, time.time(), {}
model.train()
while it < args.iters:
    for x, mask, c, nc, e, ne in loader:
        if it >= args.iters:
            break
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * lr_at(it)
        x, mask, c, nc, e, ne = (t.to(dev, non_blocking=True) for t in (x, mask, c, nc, e, ne))
        h, w = x.shape[-2:]
        tc, te = targets(c, nc, e, ne, h, w)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = model(x)
        ls = losses(lg.float(), mask, tc, te)
        loss = ls["corner"] + ls["edge"] + 0.5 * ls["mask"] + 0.5 * ls["hidden"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        for k, v in ls.items():
            run[k] = 0.98 * run.get(k, float(v)) + 0.02 * float(v)
        it += 1
        if it % 100 == 0:
            rec = {"it": it, "s": round(time.time() - t0), "lr": round(opt.param_groups[1]["lr"], 6), **{k: round(v, 4) for k, v in run.items()}}
            print("[train] " + json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if (it % args.val_every == 0 or it == args.iters) and val_frames:
            vr = {"it": it, "val": evaluate(args.val_images)}
            print("[val] " + json.dumps(vr), flush=True)
            log.write(json.dumps(vr) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "it": it, "val": vr["val"], "args": vars(args)}, out / f"model_{it}.pt")
        if it % 1000 == 0:
            torch.save({"model": model.state_dict(), "it": it, "args": vars(args)}, out / "model_last.pt")
torch.save({"model": model.state_dict(), "it": it, "args": vars(args)}, out / "model_last.pt")
print("TRAIN_DONE", flush=True)
log.close()
os._exit(0)  # persistent loader workers can keep the interpreter from exiting (kp_v1 hung after TRAIN_DONE)
