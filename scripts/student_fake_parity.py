"""Student plan Step 3 (pre-check without Isaac): the executor and the option runtime with ``oracle_graph`` driven
side by side on the kinematic fake world (``tower_rl.student.fakeworld``), planned skills, every variant x scripted
failure scenario; rows and node events must be identical. Writes results/student/parity/fake_world.json.

Usage: python scripts/student_fake_parity.py [--variants ...] [--scenarios ...] [--seconds 210] [--workers 8]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.data.layouts import ALL_VARIANTS  # noqa: E402
from tower_rl.paths import RESULTS_DIR  # noqa: E402
from tower_rl.student.fakeworld import SCENARIOS  # noqa: E402


def one(job):
    import torch
    torch.set_num_threads(1)
    from tower_rl.graph.executor import Executor
    from tower_rl.student.fakeworld import run_side_by_side, scenario
    from tower_rl.student.oracle import Oracle
    from tower_rl.student.runtime import StudentExecutor
    v, sc, secs = job
    t0 = time.time()
    r = run_side_by_side(lambda: Executor(v), lambda: StudentExecutor(v, decider=Oracle("graph")), v, seconds=secs,
                         **scenario(sc, v))
    ev = r["events_a"]
    out = {"variant": v, "scenario": sc, "equal": r["equal"], "rows": r["rows"], "wall_s": round(time.time() - t0, 1),
           "events": len(ev), "failed_nodes": sum(e[2] != "ok" for e in ev),
           "alt_nodes": sum("_alt" in e[1] for e in ev), "event_ids": [f"{e[0]} {e[1]} {e[2]}" for e in ev]}
    if not r["equal"]:
        out["t"] = r["t"]
        out["events_b"] = [f"{e[0]} {e[1]} {e[2]}" for e in r["events_b"]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--seconds", type=float, default=210.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(RESULTS_DIR / "student" / "parity" / "fake_world.json"))
    args = ap.parse_args()
    jobs = [(v, sc, args.seconds) for v in args.variants for sc in args.scenarios]
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(one, jobs))
    for r in res:
        print(f"{r['variant']:34s} {r['scenario']:15s} {'EQUAL' if r['equal'] else 'DIFFERENT'} rows {r['rows']} events "
              f"{r['events']} failed {r['failed_nodes']} alt {r['alt_nodes']} ({r['wall_s']} s)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    bad = [(r["variant"], r["scenario"]) for r in res if not r["equal"]]
    print(f"{len(res) - len(bad)} / {len(res)} equal; different: {bad}")


if __name__ == "__main__":
    main()
