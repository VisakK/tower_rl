"""Student plan G8, recovery criterion: the share of episodes with at least one failure injection that still succeed.

Evaluation runs with ``--inject`` do not log which options were perturbed. ``StudentRun`` draws them from
``default_rng([seed, 991])``, once per goal option (``Injector.maybe``; no other draws when exploration and the DAgger
mix are off), so the option sequence in each episode's policy log (the first node of each option) replays them.
Check printed at the end: the replayed deliver yaw injections (4-8 deg) against the place yaw the episode logged.

Usage: python scripts/inject_recovery.py results/student/eval/oracle_state_inject results/student/eval/bc_r2_inject
       [--p_inject 0.15]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.student.inject import Injector  # noqa: E402
from tower_rl.student.spec import STAGE_NODES, Decision  # noqa: E402

FIRST = {nodes[0][1]: stage for stage, nodes in STAGE_NODES.items()}  # "to_pregrasp" -> "acquire", ...


def option_of(node_id: str) -> str | None:
    """The goal option a node id starts (its first node: ``<prefix>_to_<target>``, arm-switch ids with ``_alt``)."""
    base = node_id[:-4] if node_id.endswith("_alt") else node_id
    for suffix, stage in FIRST.items():
        if base.endswith("_" + suffix):
            return stage
    return None


def replay(run: Path, cfg: dict, p: float) -> list[dict]:
    eps = []
    for f in sorted(glob.glob(str(run / "*" / "*" / "episodes.json"))):
        for e in json.load(open(f)):
            inj = Injector(cfg, p, np.random.default_rng([e["extra"]["seed"], 991]))
            hits = []
            for k, (_, name, _, _) in enumerate(e["extra"]["policy_log"]):
                o = option_of(name)
                if o is not None:
                    r = inj.maybe(Decision(o), [{"direction": [1.0, 0.0, 0.0]}])
                    if r is not None:
                        hits.append((k, name, r))
            eps.append({"cell": "/".join(Path(f).parts[-3:-1]), "success": bool(e["task_success"]), "hits": hits, "ep": e})
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--p_inject", type=float, default=0.15)
    args = ap.parse_args()
    cfg = load_yaml("student/collect")["inject"]
    for run in args.runs:
        eps = replay(Path(run), cfg, args.p_inject)
        inj = [e for e in eps if e["hits"]]
        rec = sum(e["success"] for e in inj)
        print(f"{Path(run).name}: {len(eps)} episodes, {sum(e['success'] for e in eps)} successes; injected {len(inj)}, "
              f"recovered {rec} ({rec / max(1, len(inj)):.3f})")
        chk = []
        for e in inj:
            log = e["ep"]["extra"]["policy_log"]
            for k, name, r in e["hits"]:
                if "goal_yaw_deg" in r and k + 1 < len(log) and isinstance(log[k + 1][3], dict) and "yaw_deg" in log[k + 1][3]:
                    chk.append((abs(r["goal_yaw_deg"]), log[k + 1][3]["yaw_deg"]))  # the option's place node
        if chk:
            a = np.array(chk)
            print(f"  check: {len(a)} deliver yaw injections ({a[:, 0].min():.1f}-{a[:, 0].max():.1f} deg); the place that "
                  f"followed logged yaw >= 3 deg in {int((a[:, 1] >= 3).sum())}, median {np.median(a[:, 1]):.1f} deg "
                  f"(bc_r2 without injection: median 0.7 deg, 2.4 % >= 3 deg)")


if __name__ == "__main__":
    main()
