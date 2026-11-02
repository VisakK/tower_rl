"""Offline evaluation of a switcher export on shards (student plan Step 7, gate G7a).

Per head (option, goal, arm, all three), per family and variant, nominal vs non-nominal decisions; the option
confusion matrix; progress AUC and calibration (10 bins). Uses the export's validation episodes by default
(``--split val``), or every episode of the given shards (``--split all``).

Usage: python scripts/eval_switcher_offline.py --model results/student/models/<name>/switcher.pt --data <round dirs>
       [--split val|all] [--out <json>]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.student.spec import OPTIONS  # noqa: E402
from tower_rl.student.switcher import SwitcherDecider  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_switcher import auc, check_labels, load  # noqa: E402

KEYS = ("blk", "blk_mask", "goal", "goal_mask", "arm", "glob", "valid_opt", "valid_goal", "valid_arm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "all"], default="val")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    dec = SwitcherDecider.load(args.model)
    data, index = load(args.data)
    ok = check_labels(data)
    rows = np.flatnonzero(ok)
    if args.split == "val":
        val = set(dec.meta.get("val_episodes", []))
        rows = rows[np.isin(data["ep"][rows], list(val))]
    outs, progs = [], []
    with torch.no_grad():
        for s in range(0, len(rows), 4096):
            r = rows[s: s + 4096]
            o, p = dec.mod(*(torch.as_tensor(data[k][r]) for k in KEYS))
            outs.append(o.numpy())
            progs.append(p.numpy())
    out, prog = np.concatenate(outs), np.concatenate(progs)
    d = data["kind"][rows] == 0
    lab = np.stack([data["label_opt"][rows], data["label_goal"][rows], data["label_arm"][rows]], -1)
    hit = out == lab
    nn = data["nonnominal"][rows].astype(bool)
    fam = np.array([index[e]["variant"] for e in data["ep"][rows]])

    def acc(m):
        m = m & d
        if not m.any():
            return None
        return {"n": int(m.sum()), "option": float(hit[m, 0].mean()), "goal": float(hit[m, 1].mean()),
                "arm": float(hit[m, 2].mean()), "all": float(hit[m].all(-1).mean())}

    rep = {"model": args.model, "split": args.split, "rows": int(len(rows)), "decisions": int(d.sum()),
           "all": acc(np.ones_like(d)), "nominal": acc(~nn), "nonnominal": acc(nn),
           "by_variant": {v: {"nominal": acc((fam == v) & ~nn), "nonnominal": acc((fam == v) & nn)} for v in sorted(set(fam))}}
    conf = np.zeros((len(OPTIONS), len(OPTIONS)), dtype=int)
    for a, b in zip(lab[d, 0], out[d, 0]):
        conf[a, b] += 1
    rep["option_confusion"] = {"labels_rows_pred_cols": list(OPTIONS), "matrix": conf.tolist()}
    pg = data["progress"][rows]
    m = pg >= 0
    p, y = prog[m], pg[m]
    bins = np.clip((p * 10).astype(int), 0, 9)
    rep["progress"] = {"n": int(m.sum()), "auc": auc(p, y),
                       "calibration": [{"bin": i, "n": int((bins == i).sum()), "mean_p": float(p[bins == i].mean()),
                                        "frac_true": float(y[bins == i].mean())} for i in range(10) if (bins == i).any()]}
    errs = defaultdict(int)
    for i in np.flatnonzero(d & ~hit.all(-1)):
        errs[(OPTIONS[lab[i, 0]], OPTIONS[out[i, 0]])] += 1
    rep["errors_by_label_pred"] = {f"{a}->{b}": n for (a, b), n in sorted(errs.items(), key=lambda x: -x[1])}
    print(json.dumps({k: rep[k] for k in ("rows", "decisions", "all", "nominal", "nonnominal")}, indent=1))
    print("progress AUC", rep["progress"]["auc"], "errors", rep["errors_by_label_pred"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
