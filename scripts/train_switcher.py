"""Train the option switcher by behaviour cloning on the collected shards (student plan Steps 7 and 8).

Loads every shard under the given round dirs (``tower_rl.student.logging.load_shards``; DAgger: the union of rounds),
splits by episode (90 / 10, deterministic), fits the feature normaliser on the training rows, trains ``SwitcherNet``
with ``configs/student/switcher.yaml`` (AdamW; CE over option / goal / arm, x5 on non-nominal decisions, + 0.5 BCE
progress), keeps the epoch with the best validation decision accuracy, and exports TorchScript + JSON.

Usage: python scripts/train_switcher.py --data results/student/data/r0 [more rounds] --out results/student/models/<name>
       [--device cuda] [--epochs N]
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.config import config_hash, load_yaml  # noqa: E402
from tower_rl.student.logging import load_shards  # noqa: E402
from tower_rl.student.switcher import Normalizer, SwitcherNet, export, loss_fn  # noqa: E402
from tower_rl.student.tokens import FEATURES  # noqa: E402

KEYS = ("blk", "blk_mask", "goal", "goal_mask", "arm", "glob", "valid_opt", "valid_goal", "valid_arm", "label_opt",
        "label_goal", "label_arm", "kind", "progress", "nonnominal")


def load(rounds: list[str]) -> tuple[dict, list]:
    parts, index = [], []
    for r in rounds:
        d, idx = load_shards(r)
        d["ep"] = d["ep"] + len(index)  # episode ids unique over rounds
        if "nonnominal" not in d:
            d["nonnominal"] = np.zeros(len(d["kind"]), dtype=np.int64)
        parts.append(d)
        index += idx
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return data, index


def check_labels(data: dict) -> np.ndarray:
    """Rows whose label is valid under their masks (decisions) or that are progress samples."""
    dec = data["kind"] == 0
    n = len(dec)
    ok = np.ones(n, dtype=bool)
    lo, lg, la = data["label_opt"], data["label_goal"], data["label_arm"]
    i = np.arange(n)
    ok &= ~dec | (lo >= 0)
    lo_c, lg_c, la_c = lo.clip(0), lg.clip(0), la.clip(0)
    ok &= ~dec | data["valid_opt"][i, lo_c]
    ok &= ~dec | (lg < 0) | data["valid_goal"][i, lo_c, lg_c]
    ok &= ~dec | (la < 0) | data["valid_arm"][i, lo_c, lg_c, la_c]
    return ok


def to_batch(data: dict, idx: np.ndarray, dev) -> dict:
    out = {}
    for k in KEYS:
        t = torch.as_tensor(data[k][idx], device=dev)
        if k in ("label_opt", "label_goal", "label_arm", "kind", "progress", "nonnominal"):
            t = t.long()
        out[k] = t
    return out


@torch.no_grad()
def evaluate(net: SwitcherNet, data: dict, idx: np.ndarray, dev, bs: int = 2048) -> dict:
    """Decision accuracy (masked argmax == label on option, goal and arm), nominal / non-nominal, and progress AUC."""
    net.eval()
    hits, nn_flags, probs, labels = [], [], [], []
    for s in range(0, len(idx), bs):
        b = to_batch(data, idx[s: s + bs], dev)
        out, prog = net.decide(b["blk"], b["blk_mask"], b["goal"], b["goal_mask"], b["arm"], b["glob"], b["valid_opt"],
                               b["valid_goal"], b["valid_arm"])
        dec = b["kind"] == 0
        lab = torch.stack([b["label_opt"], b["label_goal"], b["label_arm"]], -1)
        hits.append(((out == lab).all(-1))[dec].cpu().numpy())
        nn_flags.append(b["nonnominal"][dec].cpu().numpy())
        pm = b["progress"] >= 0
        probs.append(prog[pm].cpu().numpy())
        labels.append(b["progress"][pm].cpu().numpy())
    h, f = np.concatenate(hits), np.concatenate(nn_flags).astype(bool)
    p, y = np.concatenate(probs), np.concatenate(labels)
    return {"acc": float(h.mean()) if len(h) else 0.0, "acc_nominal": float(h[~f].mean()) if (~f).any() else None,
            "acc_nonnominal": float(h[f].mean()) if f.any() else None, "decisions": int(len(h)),
            "nonnominal": int(f.sum()), "progress_auc": auc(p, y)}


def auc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1  # ranks (ties broken arbitrarily)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--progress_frac", type=float, default=1.0, help="share of the progress-only rows used per epoch "
                                                                     "(all decision rows always; CPU training)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_yaml("student/switcher")
    assert cfg["features"] == FEATURES, (cfg["features"], FEATURES)
    tr_cfg = dict(cfg["train"])
    if args.epochs:
        tr_cfg["epochs"] = args.epochs
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data, index = load(args.data)
    ok = check_labels(data)
    print(f"rows {len(ok)} ({int((data['kind'] == 0).sum())} decisions), episodes {len(index)}, rows with invalid labels "
          f"dropped: {int((~ok).sum())}", flush=True)
    eps = np.unique(data["ep"])
    val_eps = set(rng.choice(eps, size=max(1, int(round(tr_cfg["val_frac"] * len(eps)))), replace=False).tolist())
    is_val = np.array([e in val_eps for e in data["ep"]]) & ok
    tr_idx, va_idx = np.flatnonzero(~is_val & ok), np.flatnonzero(is_val)
    stats = Normalizer.fit({k: data[k][tr_idx] for k in ("blk", "blk_mask", "goal", "goal_mask", "arm", "glob")})
    m = cfg["model"]
    dev = torch.device(args.device)
    net = SwitcherNet(m["d_model"], m["layers"], m["heads"], m.get("ff", 3 * m["d_model"]), m["dropout"], stats=stats).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, hist, t0 = None, [], time.time()
    tr_dec = tr_idx[data["kind"][tr_idx] == 0]
    tr_prog = tr_idx[data["kind"][tr_idx] != 0]
    per_epoch = len(tr_dec) + int(round(args.progress_frac * len(tr_prog)))
    steps = max(50, tr_cfg["epochs"] * max(1, -(-per_epoch // tr_cfg["batch"])))  # one-cycle needs a warm-up step
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr_cfg["lr"], total_steps=steps, pct_start=0.1)
    for ep in range(tr_cfg["epochs"]):
        net.train()
        sub = tr_prog if args.progress_frac >= 1 else rng.choice(tr_prog, int(round(args.progress_frac * len(tr_prog))), replace=False)
        perm = rng.permutation(np.concatenate([tr_dec, sub]))
        tot, n = 0.0, 0
        for s in range(0, len(perm), tr_cfg["batch"]):
            if len(perm) - s < 2:
                continue
            b = to_batch(data, perm[s: s + tr_cfg["batch"]], dev)
            loss, _ = loss_fn(net, b, tr_cfg["nonnominal_weight"], tr_cfg["progress_weight"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot, n = tot + float(loss), n + 1
        ev = evaluate(net, data, va_idx, dev)
        hist.append({"epoch": ep, "loss": tot / max(n, 1), **ev, "wall_s": round(time.time() - t0, 1)})
        print(json.dumps(hist[-1]), flush=True)
        if best is None or ev["acc"] > best[0]:
            best = (ev["acc"], ep)
            torch.save({"state": net.state_dict(), "stats": stats, "model": m, "epoch": ep}, out / "best.pt")
    ck = torch.load(out / "best.pt", map_location="cpu")
    net = SwitcherNet(m["d_model"], m["layers"], m["heads"], m.get("ff", 3 * m["d_model"]), m["dropout"], stats=stats)
    net.load_state_dict(ck["state"])
    final = {"train": evaluate(net, data, tr_idx[: 20000], torch.device("cpu")), "val": evaluate(net, data, va_idx, torch.device("cpu"))}
    meta = {"model": m, "train": tr_cfg, "data": args.data, "episodes": len(index), "rows": int(len(ok)),
            "best_epoch": int(ck["epoch"]), "val": final["val"], "config_hash": config_hash(cfg),
            "written": datetime.datetime.now().isoformat(timespec="minutes"), "val_episodes": sorted(int(e) for e in val_eps)}
    export(net, out / "switcher.pt", meta)
    (out / "train_log.json").write_text(json.dumps({"history": hist, "final": final, "meta": meta}, indent=1))
    print(f"best epoch {ck['epoch']}: val {json.dumps(final['val'])}; exported {out / 'switcher.pt'}", flush=True)


if __name__ == "__main__":
    main()
