"""Decision and progress shards for the switcher (student plan Step 4, Appendix C).

With ``rehearse.py --collect <dir>`` every ``StudentRun`` records, per decision, the tokens and action masks, the
labeller's decision (the state-mode oracle), the executed decision and its source, and per ``progress_every`` calls
the tokens with the per-goal truth flags (``privileged["ops"]``). After each rehearsal batch ``write_batch`` stores
``<dir>/<variant>/<tier>/b<batch seed>.npz`` and appends one row per episode to ``<dir>/<variant>/<tier>/index.jsonl``,
joined with the episode's outcome. Rows of one episode are contiguous; ``load_shards`` numbers the episodes (``ep``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tower_rl.student.tokens import FEATURES, MAX_GOALS, encode

SOURCES = ("oracle", "switcher", "explore", "inject")
TOKEN_KEYS = ("blk", "blk_mask", "goal", "goal_mask", "arm", "glob", "valid_opt", "valid_goal", "valid_arm")


def row(tokens: dict, kind: int, label=None, executed=None, source: str = "oracle", beta: float | None = None,
        t: float = 0.0, priv: dict | None = None) -> dict:
    """One shard row: a decision (kind 0) or a progress-only sample (kind 1)."""
    priv = priv or {}
    ops = priv.get("ops")
    prog = np.full(MAX_GOALS, -1, dtype=np.int8)
    if ops is not None:
        prog[:len(ops)] = np.asarray(ops, dtype=np.int8)
    lo, lg, la = encode(label) if label is not None else (-1, -1, -1)
    eo, eg, ea = encode(executed) if executed is not None else (-1, -1, -1)
    r = {k: tokens[k] for k in TOKEN_KEYS}
    r.update(kind=kind, label_opt=lo, label_goal=lg, label_arm=la, exec_opt=eo, exec_goal=eg, exec_arm=ea,
             source=SOURCES.index(source) if kind == 0 else -1, beta=-1.0 if beta is None else float(beta),
             progress=prog, latched=np.asarray(priv.get("ops_latched", (0, 0)), dtype=np.int16),
             disturbed=int(priv.get("disturbed", 0)), t=float(t))
    return r


def write_batch(root: str | Path, policy, outcomes: list, variant: str, tier: str, batch_seed: int, meta: dict) -> Path | None:
    """Store the batch's rows (``policy.runs[e].rows``) as one npz shard and index its episodes (outcomes by env) in
    ``<root>/<variant>/<tier>/index.jsonl`` (one writer per variant and tier: rehearsals run in parallel processes)."""
    root = Path(root)
    out = root / variant / tier / f"b{batch_seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, eps = [], []
    for e, o in enumerate(outcomes):
        run = policy.runs[e]
        for r in run.rows:
            r["env"] = e
        rows += run.rows
        eps.append({"variant": variant, "tier": tier, "case": o.extra.get("case"), "seed": o.extra.get("seed"),
                    "batch": batch_seed, "env": e, "features": FEATURES, **meta, "shard": str(out.relative_to(root)),
                    "rows": len(run.rows), "task_success": bool(o.task_success), "grasp_ok": list(o.grasp_ok),
                    "place_ok": list(o.place_ok), "disturbed_per_op": list(o.disturbed_per_op),
                    "duration_s": o.duration_s, "decisions": run.memory.decisions, "masked": run.memory.masked})
    if not rows:
        return None
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    arrays["features"] = np.array(FEATURES)
    np.savez_compressed(out, **arrays)
    with open(out.parent / "index.jsonl", "a") as f:
        for r in eps:
            f.write(json.dumps(r) + "\n")
    return out


def load_shards(root: str | Path, features: str = FEATURES) -> tuple[dict, list[dict]]:
    """All shards under ``root`` concatenated, with ``ep`` = the episode's row in the returned index (all
    ``index.jsonl`` files under ``root``, sorted); refuses other feature versions."""
    root = Path(root)
    index = []
    for f in sorted(root.glob("**/index.jsonl")):
        index += [json.loads(line) for line in open(f)]
    ep_of = {(r["shard"], r["env"]): i for i, r in enumerate(index)}
    for i, r in enumerate(index):
        r["ep"] = i
    parts = []
    for sh in sorted({r["shard"] for r in index}):
        z = np.load(root / sh)
        if str(z["features"]) != features:
            raise ValueError(f"{sh}: features {z['features']} != {features}")
        d = {k: z[k] for k in z.files if k != "features"}
        d["ep"] = np.array([ep_of[(sh, int(e))] for e in d.pop("env")], dtype=np.int64)
        parts.append(d)
    keys = set.intersection(*(set(p) for p in parts)) if parts else set()
    data = {k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)}
    return data, index
