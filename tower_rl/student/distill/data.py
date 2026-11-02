"""Step 9c distillation shards: one npz per rehearsal batch (``obs`` [N, 68], ``act`` [N, 7], ``seg`` [N]: the
segment's row in the batch's segment list) and ``<root>/<variant>/<tier>/index.jsonl`` with one line per segment (a
planned Task 5 node run under DART: skill, arm, node, status, DART scale, the episode's seed / case / task success)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tower_rl.student.distill.core import ACT_DIM, OBS_DIM

VERSION = "distill_v1"


def write_batch(root: str | Path, collector, policy, outcomes: list, variant: str, tier: str, batch_seed: int,
                meta: dict) -> Path | None:
    root = Path(root)
    out = root / variant / tier / f"b{batch_seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    obs, act, seg, segs = [], [], [], []
    for e, run in enumerate(policy.runs):
        ss, data = collector.take(run)
        if e >= len(outcomes):  # envs past the requested episode count
            continue
        o = outcomes[e]
        base = len(segs)
        for s in ss:
            segs.append({**s, "env": e, "seed": o.extra.get("seed"), "case": o.extra.get("case"),
                         "task_success": bool(o.task_success)})
        for sid, ob, a in data:
            obs.append(ob)
            act.append(a)
            seg.append(base + sid)
    if not obs:
        return None
    np.savez_compressed(out, obs=np.stack(obs).astype(np.float32), act=np.stack(act).astype(np.float32),
                        seg=np.asarray(seg, dtype=np.int32), version=np.array(VERSION))
    with open(out.parent / "index.jsonl", "a") as f:
        for i, s in enumerate(segs):
            f.write(json.dumps({**s, "variant": variant, "tier": tier, "batch": batch_seed, "shard": out.name,
                                "seg_in_shard": i, **meta}) + "\n")
    return out


def load(roots: list[str | Path]) -> tuple[dict, list[dict]]:
    """All shards under ``roots`` (any depth: an index's shards sit next to it): arrays ``obs``, ``act``, ``seg``
    (global segment id = row in the returned index), ``row`` (row within its segment); the index gains ``id``, ``root``
    and ``episode`` (variant, tier, batch, env)."""
    index, parts = [], []
    for root in map(Path, roots):
        for f in sorted(root.glob("**/index.jsonl")):
            lines = [json.loads(x) for x in open(f)]
            for sh in sorted({r["shard"] for r in lines}):
                z = np.load(f.parent / sh)
                if str(z["version"]) != VERSION:
                    raise ValueError(f"{f.parent / sh}: {z['version']} != {VERSION}")
                mine = [r for r in lines if r["shard"] == sh]
                ids = {r["seg_in_shard"]: len(index) + k for k, r in enumerate(mine)}
                for r in mine:
                    r.update(id=len(index), root=str(f.parent), episode=f"{r['variant']}/{r['tier']}/{r['batch']}/{r['env']}")
                    index.append(r)
                g = np.array([ids[int(s)] for s in z["seg"]], dtype=np.int64)
                parts.append({"obs": z["obs"], "act": z["act"], "seg": g})
    if not parts:
        return {"obs": np.zeros((0, OBS_DIM), np.float32), "act": np.zeros((0, ACT_DIM), np.float32),
                "seg": np.zeros(0, np.int64), "row": np.zeros(0, np.int64)}, index
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("obs", "act", "seg")}
    row = np.zeros(len(d["seg"]), dtype=np.int64)
    for i in range(1, len(row)):
        row[i] = row[i - 1] + 1 if d["seg"][i] == d["seg"][i - 1] else 0
    d["row"] = row
    return d, index
