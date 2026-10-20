"""Rehearse a bundle on many variants in parallel CPU processes (one ``rehearse.py`` process per variant), then
summarize (Step 7 acceptance: N >= 100 episodes per variant and tier; Step 8.4 export check with ``--bundle_configs``).

Usage: python scripts/rehearse_many.py --bundle results/bundles/<b> --episodes 100 --parallel 4 \
           --out_root results/rehearsal_final/<b> [--bundle_configs] [--perception_noise] [--rate_hz 30]
Each process: ``rehearse.py --headless --device cpu --variant <v> --tiers <tiers> ...``; logs in ``<out_root>/logs``.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tower_rl.data.layouts import VARIANTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--tiers", nargs="+", default=["D0", "D1"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--rate_hz", type=int, choices=[10, 30], default=10)
    ap.add_argument("--perception_noise", action="store_true")
    ap.add_argument("--noise", default="{}")
    ap.add_argument("--graph_set", default="{}", help="JSON merged into every graph's top level (rehearse.py --graph_set)")
    ap.add_argument("--bundle_configs", action="store_true", help="TOWER_CONFIG_DIR=<bundle>/configs (export check)")
    ap.add_argument("--nice", type=int, default=15, help="niceness of the rehearsal processes")
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()
    out = Path(args.out_root)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")  # small tensors: one thread per process
    if args.bundle_configs:
        env["TOWER_CONFIG_DIR"] = str(Path(args.bundle).resolve() / "configs")
    # the longest variants first so the pool drains evenly
    order = sorted(args.variants, key=lambda v: (not v.endswith("_3"), not v.startswith("task1"), v))
    pending, running, done = list(order), {}, []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.parallel:
            v = pending.pop(0)
            cmd = [sys.executable, str(ROOT / "scripts" / "rehearse.py"), "--headless", "--device", "cpu", "--variant", v,
                   "--tiers", *args.tiers, "--episodes", str(args.episodes), "--num_envs", str(args.num_envs),
                   "--seed", str(args.seed), "--bundle", args.bundle, "--rate_hz", str(args.rate_hz), "--out_root", str(out)]
            if args.perception_noise:
                cmd += ["--perception_noise", "--noise", args.noise]
            if args.graph_set != "{}":
                cmd += ["--graph_set", args.graph_set]
            log = open(out / "logs" / f"{v}.log", "w")
            # low CPU priority: rehearsals (CPU PhysX, many worker threads) must not slow a GPU training sharing the CPU
            running[v] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           preexec_fn=lambda: os.nice(args.nice)), log)
            print(f"[many] started {v} ({len(running)} running, {len(pending)} pending, {time.time() - t0:.0f} s)", flush=True)
        time.sleep(10)
        for v, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[v]
                done.append((v, proc.returncode))
                print(f"[many] finished {v}: exit {proc.returncode} ({time.time() - t0:.0f} s)", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_rehearsals.py"), "--root", str(out)], cwd=ROOT, env=env)
    bad = [v for v, rc in done if rc != 0]
    print(f"[many] done in {time.time() - t0:.0f} s; failed processes: {bad}", flush=True)


if __name__ == "__main__":
    main()
