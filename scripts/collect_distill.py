"""Student plan Step 9c data: CPU rehearsals of Tasks 5-2 / 5-3 with the planned grasp, insert and push-in under DART
noise (``rehearse.py --distill_collect``), then a summary of the shards.

  python scripts/collect_distill.py --out results/student/distill/data/c1 --episodes 300 --chunks 3 --parallel 6
  python scripts/collect_distill.py --out results/student/distill/data/c1 --summary_only

One rehearse.py process per variant, tier and chunk (shards in ``<out>/k<chunk>``, reports in ``<out>/reports``, logs
in ``<out>/logs``). The planned nodes act on simulator poses (plus ``--est_noise`` if given).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T5 = ["task5-2_tower_insertion_2", "task5-3_tower_insertion_3"]


def summarize(out: Path) -> dict:
    from tower_rl.student.distill.data import load

    d, index = load([out])
    rows = Counter(int(s) for s in d["seg"])
    by = defaultdict(lambda: {"segments": 0, "rows": 0, "status": Counter(), "why": Counter()})
    for r in index:
        k = f"{r['skill']}_{r['arm']}"
        by[k]["segments"] += 1
        by[k]["rows"] += rows.get(r["id"], 0)
        by[k]["status"][r["status"]] += 1
        if r.get("info", {}).get("why"):
            by[k]["why"][r["info"]["why"]] += 1
    eps = {r["episode"]: r["task_success"] for r in index}
    res = {"episodes": len(eps), "task_success": sum(eps.values()), "rows": int(len(d["seg"])),
           "per_skill": {k: {**v, "status": dict(v["status"]), "why": dict(v["why"])} for k, v in sorted(by.items())}}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default=str(ROOT / "results/bundles/teacher_t45_final"))
    ap.add_argument("--variants", nargs="+", default=T5)
    ap.add_argument("--tiers", nargs="+", default=["D0", "D1"])
    ap.add_argument("--episodes", type=int, default=300, help="per variant and tier")
    ap.add_argument("--chunks", type=int, default=3, help="processes per variant and tier (seeds seed + 10000 k)")
    ap.add_argument("--seed", type=int, default=200000)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--dart_scale", type=float, default=1.5)
    ap.add_argument("--est_noise", default=None)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if not args.summary_only:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        per = -(-args.episodes // args.chunks)
        jobs = [(v, t, k) for v in args.variants for t in args.tiers for k in range(args.chunks)]
        pending, running, failed, t0 = list(jobs), {}, [], time.time()
        while pending or running:
            while pending and len(running) < args.parallel:
                v, t, k = job = pending.pop(0)
                n = min(per, args.episodes - k * per)
                cmd = [sys.executable, str(ROOT / "scripts/rehearse.py"), "--headless", "--device", "cpu", "--variant", v,
                       "--tiers", t, "--episodes", str(n), "--num_envs", str(args.num_envs), "--seed", str(args.seed + 10000 * k),
                       "--bundle", args.bundle, "--distill_collect", str(out / f"k{k}"), "--dart_scale", str(args.dart_scale),
                       "--out_root", str(out / "reports" / f"k{k}")]
                if args.est_noise:
                    cmd += ["--est_noise", args.est_noise]
                log = open(out / "logs" / f"{v}_{t}_k{k}.log", "w")
                running[job] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 preexec_fn=lambda: os.nice(args.nice)), log)
                print(f"[collect_distill] started {job} ({len(running)} running, {len(pending)} pending, {time.time() - t0:.0f} s)",
                      flush=True)
            time.sleep(10)
            for job, (proc, log) in list(running.items()):
                if proc.poll() is not None:
                    log.close()
                    del running[job]
                    if proc.returncode:
                        failed.append(job)
                    print(f"[collect_distill] finished {job}: exit {proc.returncode} ({time.time() - t0:.0f} s)", flush=True)
        print(f"[collect_distill] done in {time.time() - t0:.0f} s; failed: {failed}", flush=True)
    s = summarize(out)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
