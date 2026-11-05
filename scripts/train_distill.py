"""Train the distilled Task 5 skills (student plan Step 9c) on the DART shards of ``collect_distill.py``.

Per skill (grasp_t5, insert, push_in) and arm: the rows of that skill's segments, split by episode (90 / 10,
deterministic), the observation normaliser fitted on the training rows (constant features keep std 1), ``DistillNet``
(256-256-256, ELU) trained with MSE on the six motion values + 0.1 x BCE on the gripper bit (the rare class weighted up
to balance, <= ``--w_rare_max``: the insert's single release row per segment); AdamW, cosine schedule; the epoch with the best
validation loss is kept. Export: TorchScript ``<out>/<skill>_bc_<arm>.pt`` (normaliser inside) + ``.json`` (obs_dim,
metrics), the files a bundle's ``skills/`` takes (``tower_rl.student.distill.nodes.make_distilled``).

Usage: python scripts/train_distill.py --data results/student/distill/data/c1 [more] --out results/student/distill/models/v1
       [--skills grasp_t5 insert push_in] [--arms left back] [--epochs 60] [--only_ok]
"""
import argparse
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.student.distill.core import ACT_DIM, OBS_DIM, POS_LIM, PREV_A, ROT_LIM, DistillNet  # noqa: E402
from tower_rl.student.distill.data import VERSION, load  # noqa: E402


def val_episode(ep: str) -> bool:
    return int(hashlib.sha256(ep.encode()).hexdigest()[:8], 16) % 10 == 0


def losses(net, o, a, w0: float):
    y = net(o)
    mse = ((y[:, :6] - a[:, :6]) ** 2).mean(1)
    lab = a[:, 6]
    wt = torch.where(lab < 0.5, torch.full_like(lab, w0), torch.ones_like(lab))
    bce = F.binary_cross_entropy_with_logits(y[:, 6], lab, reduction="none") * wt
    return mse, bce, y


def evaluate(net, o, a, seg, row, w0, skill):
    with torch.no_grad():
        mse, bce, y = losses(net, o, a, w0)
    pos = POS_LIM[skill] * 1000
    err = (y[:, :6] - a[:, :6]).numpy()
    closed_pred = (y[:, 6] > 0).numpy()
    lab = a[:, 6].numpy() > 0.5
    res = {"loss": float((mse + 0.1 * bce).mean()), "mse": float(mse.mean()), "bce": float(bce.mean()),
           "rms_pos_mm": float(np.sqrt((err[:, :3] ** 2).mean()) * pos),
           "rms_rot_deg": float(np.sqrt((err[:, 3:] ** 2).mean()) * math.degrees(ROT_LIM)),
           "grip_acc": float((closed_pred == lab).mean()), "rows": int(len(a))}
    # the gripper switch per segment: first row the label changes (grasp: closes; insert: releases) vs the prediction
    offs = []
    seg = np.asarray(seg)
    for s in np.unique(seg):
        m = np.nonzero(seg == s)[0]
        m = m[np.argsort(row[m])]
        lab_s, pred_s = lab[m], closed_pred[m]
        if skill == "grasp_t5":
            t_lab = np.argmax(lab_s) if lab_s.any() else None
            t_pred = np.argmax(pred_s) if pred_s.any() else None
        elif skill == "insert":
            t_lab = np.argmax(~lab_s) if (~lab_s).any() else None
            t_pred = np.argmax(~pred_s) if (~pred_s).any() else None
        else:
            continue
        if t_lab is not None:
            offs.append(None if t_pred is None else int(t_pred - t_lab))
    if offs:
        got = [x for x in offs if x is not None]
        res["switch"] = {"segments": len(offs), "never": sum(x is None for x in offs),
                         "abs_le_1": sum(abs(x) <= 1 for x in got), "early": sum(x < -1 for x in got),
                         "late": sum(x > 1 for x in got), "median": float(np.median(got)) if got else None}
    return res


def train_one(d, index, skill, arm, args, out: Path):
    seg_ok = np.array([r["skill"] == skill and r["arm"] == arm and (r["status"] == "ok" or not args.only_ok) for r in index])
    rows = np.nonzero(seg_ok[d["seg"]])[0]
    if len(rows) < 100:
        print(f"[train_distill] {skill}_{arm}: {len(rows)} rows, skipped", flush=True)
        return None
    is_val = np.array([val_episode(r["episode"]) for r in index])[d["seg"][rows]]
    tr, va = rows[~is_val], rows[is_val]
    o_tr, a_tr = torch.as_tensor(d["obs"][tr]), torch.as_tensor(d["act"][tr])
    o_va, a_va = torch.as_tensor(d["obs"][va]), torch.as_tensor(d["act"][va])
    mask = torch.ones(OBS_DIM)
    if not args.use_prev_a:
        mask[PREV_A] = 0.0
    mean = (o_tr * mask).mean(0)
    std = (o_tr * mask).std(0)
    std = torch.where(std < 1e-4, torch.ones_like(std), std)
    n1 = float((a_tr[:, 6] > 0.5).sum())
    n0 = float(len(a_tr)) - n1
    w0 = float(min(args.w_rare_max, max(1.0, n1 / max(n0, 1.0))))
    torch.manual_seed(args.seed)
    net = DistillNet(mean, std, mask=mask)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per = max(1, len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per)
    g = torch.Generator().manual_seed(args.seed)
    best, best_state, hist = math.inf, None, []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(tr), generator=g)
        tot = 0.0
        for k in range(steps_per):
            b = perm[k * args.batch:(k + 1) * args.batch]
            ob = o_tr[b]
            if args.obs_noise > 0:  # augmentation: Gaussian noise in normalised units (against memorising trajectories)
                ob = ob + torch.randn(ob.shape, generator=g) * args.obs_noise * net.std
            mse, bce, _ = losses(net, ob, a_tr[b], w0)
            loss = (mse + 0.1 * bce).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        net.eval()
        v = evaluate(net, o_va, a_va, d["seg"][va], d["row"][va], w0, skill) if len(va) else {"loss": tot / steps_per}
        hist.append({"epoch": ep, "train": tot / steps_per, "val": v["loss"]})
        if v["loss"] < best:
            best, best_state = v["loss"], {k: x.clone() for k, x in net.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[train_distill] {skill}_{arm} epoch {ep}: train {tot / steps_per:.5f} val {v['loss']:.5f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    val = evaluate(net, o_va, a_va, d["seg"][va], d["row"][va], w0, skill) if len(va) else {}
    trm = evaluate(net, o_tr, a_tr, d["seg"][tr], d["row"][tr], w0, skill)
    path = out / f"{skill}_bc_{arm}.pt"
    mod = torch.jit.script(net)
    mod.save(str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"skill": skill, "arm": arm, "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "version": VERSION, "sha256": sha,
            "rows_train": int(len(tr)), "rows_val": int(len(va)), "segments": int(seg_ok.sum()), "w_rare": w0,
            "val": val, "train": {k: trm[k] for k in ("loss", "rms_pos_mm", "rms_rot_deg", "grip_acc")}, "history": hist,
            "data": [str(x) for x in args.data], "only_ok": args.only_ok, "epochs": args.epochs, "lr": args.lr,
            "w_rare_max": args.w_rare_max, "obs_noise": args.obs_noise, "weight_decay": args.weight_decay,
            "use_prev_a": args.use_prev_a,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[train_distill] {skill}_{arm}: val {json.dumps(val)}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skills", nargs="+", default=["grasp_t5", "insert", "push_in"])
    ap.add_argument("--arms", nargs="+", default=["left", "back"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only_ok", action="store_true", help="only segments whose planned node succeeded")
    ap.add_argument("--obs_noise", type=float, default=0.0, help="training-time Gaussian noise on the observation, in "
                                                                  "units of each feature's std")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--use_prev_a", action="store_true", help="let the network see the previous action (default: masked; "
                                                              "it was copied and drifted in closed loop)")
    ap.add_argument("--w_rare_max", type=float, default=10.0, help="cap of the rare gripper class's weight (the insert's "
                                                                   "release row: higher fires the release rows early)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d, index = load(args.data)
    print(f"[train_distill] {len(d['seg'])} rows, {len(index)} segments from {args.data}", flush=True)
    report = {}
    for sk in args.skills:
        for arm in args.arms:
            m = train_one(d, index, sk, arm, args, out)
            if m is not None:
                report[f"{sk}_{arm}"] = {k: m[k] for k in ("rows_train", "rows_val", "segments", "val")}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("[train_distill] done", flush=True)


if __name__ == "__main__":
    main()
