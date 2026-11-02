"""Try the student on one task (or all ten) under a chosen randomization, with videos: a front end to ``rehearse.py``.

  source ~/Documents/moves/env_isaaclab/bin/activate && cd ~/Documents/tower_rl
  python scripts/try_student.py 4-2                          # Task 4-2, D1, a case drawn from the seed, 1 episode
  python scripts/try_student.py 4-2 --case slippery heavy --episodes 2
  python scripts/try_student.py 5-3 --tier D0 --live         # watch in the Isaac Sim window
  python scripts/try_student.py all --case lighting          # all ten tasks, one episode each
  python scripts/try_student.py 1-3 --teacher                # the teacher (executor) on the same episode, to compare
  python scripts/try_student.py --list                       # tasks, tiers and cases

Randomization (configs/tiers.yaml): ``--tier D0`` is nominal; ``--tier D1`` draws one of the evaluator's robustness
cases per episode (or ``--case`` forces them, one per episode, cycling); ``--tier D2`` is the training superset.
Poses: ``camera`` (default) runs the kp_v2 estimator on the four rendered cameras, as in the evaluation (GPU, one mp4 per
episode); ``sim`` gives the decision layer and skills the simulator's poses; ``noise`` adds the estimator noise model
(v1) to them (both CPU only, no rendering, no video). Results: ``<out>/<variant>/<tier>.../`` (feedback.md,
episodes.json) and ``<out>/videos/*.mp4`` (names end in _success / _fail); the full log is ``<out>/<task>.log``.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.data.layouts import ALL_VARIANTS  # noqa: E402

BUNDLE = ROOT / "results/bundles/student_v1"
TEACHER = ROOT / "results/bundles/teacher_t45_final"
NOISE_V1 = ROOT / "results/student/noise/noise_v1_0034.yaml"  # the switcher's training noise (G5: v1 matches the cameras)
SHOW = ("[rehearse]", "[video]", "REPORT_JSON")  # progress lines echoed to the terminal (the rest: the log)
PROBLEM = ("Traceback", "Error:", "error:", "SystemExit")  # ... and problems, except Kit's [Warning] lines


def variant_of(task: str) -> str:
    hits = [v for v in ALL_VARIANTS if v == task or v.startswith(f"task{task}_")]
    if len(hits) != 1:
        raise SystemExit(f"unknown task {task!r}; one of: {' '.join(v.split('_')[0][4:] for v in ALL_VARIANTS)} or all")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="e.g. 4-2 (or the full variant name), or all")
    ap.add_argument("--tier", default="D1", choices=["D0", "D1", "D2"])
    ap.add_argument("--case", nargs="+", default=None, help="force robustness case(s) of the tier (see --list)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--poses", default="camera", choices=["camera", "sim", "noise"])
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--live", action="store_true", help="open the Isaac Sim window and render every row (1 env)")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--teacher", action="store_true", help="run the teacher (executor + its graphs) instead of the student")
    ap.add_argument("--bundle", default=None, help=f"default {BUNDLE.relative_to(ROOT)} ({TEACHER.relative_to(ROOT)} "
                                                   "with --teacher)")
    ap.add_argument("--out", default=None, help="default results/student/try/<time>")
    ap.add_argument("--list", action="store_true", help="list tasks, tiers and robustness cases")
    ap.add_argument("--dry_run", action="store_true", help="print the rehearse.py command(s) only")
    args = ap.parse_args()
    tiers = load_yaml("tiers")["tiers"]
    if args.list or not args.task:
        print("tasks: " + " ".join(v.split("_")[0][4:] for v in ALL_VARIANTS) + "  (or all)")
        for t, c in tiers.items():
            print(f"{t} cases: {' '.join(c['cases'])}")
        return
    if args.case and set(args.case) - set(tiers[args.tier]["cases"]):
        raise SystemExit(f"--case {' '.join(sorted(set(args.case) - set(tiers[args.tier]['cases'])))}: not a {args.tier} "
                         f"case; {args.tier} cases: {' '.join(tiers[args.tier]['cases'])}")
    variants = ALL_VARIANTS if args.task == "all" else [variant_of(args.task)]
    bundle = Path(args.bundle) if args.bundle else (TEACHER if args.teacher else BUNDLE)
    out = Path(args.out) if args.out else ROOT / "results/student/try" / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    num_envs = 1 if args.live else args.num_envs
    summary = {}
    for v in variants:
        cmd = [sys.executable, str(ROOT / "scripts/rehearse.py"), "--device", "cpu", "--variant", v, "--tiers", args.tier,
               "--episodes", str(args.episodes), "--num_envs", str(num_envs), "--seed", str(args.seed),
               "--bundle", str(bundle), "--out_root", str(out)]
        if not args.live:
            cmd.append("--headless")
        else:
            cmd.append("--live")
        if not args.teacher:
            cmd += ["--student", str(bundle / "student/switcher.pt")]
        if args.case:
            cmd += ["--case", *args.case]
        if args.poses == "camera":
            model = bundle / "perception/kp_v2.pt"
            if not model.exists():
                model = ROOT / "results/perception/models/kp_v2/model_last.pt"
            cmd += ["--perception", str(model), "--perception_device", "cuda"]
            if not args.no_video:
                cmd += ["--video", str(out / "videos")]
        elif args.poses == "noise":
            cmd += ["--est_noise", str(NOISE_V1)]
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        log = out / f"{v.split('_')[0]}.log"
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                f.write(line)
                if line.startswith(SHOW) or ("[Warning]" not in line and any(s in line for s in PROBLEM)):
                    print(line.rstrip()[:300], flush=True)
                    if line.startswith("REPORT_JSON"):
                        summary.update(json.loads(line[len("REPORT_JSON"):]))
            proc.wait()
        print(f"[try_student] {v}: exit {proc.returncode}, log {log}", flush=True)
    if summary:
        print("[try_student] results (TSR = task success rate): "
              + "; ".join(f"{k}: TSR {m['TSR']:.2f}" for k, m in summary.items()))
        print(f"[try_student] outputs in {out}" + ("" if args.poses != "camera" or args.no_video else f" (videos: {out / 'videos'})"))


if __name__ == "__main__":
    main()
