"""Local evaluator CLI (Step 3): run an in-process policy in the station scene and report the challenge metrics.

Policies: ``hold`` (keeps the home pose), ``replay`` (a recording's commands from its own initial state; one episode
per recording in ``--replay_episodes``), ``executor`` (the privileged teacher, Step 7).
Use ``--device cpu``: CPU PhysX reproduces the evaluator (BUILDLOG 2026-10-18).

Examples:
  python scripts/evaluate_policy.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --policy hold --episodes 2
  python scripts/evaluate_policy.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --policy replay \
      --replay_episodes data/subset/task2-2_cross_tower_transfer_2/*.npz --finger_friction 1.2 1.0
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", required=True)
parser.add_argument("--policy", choices=["hold", "replay", "executor"], default="hold")
parser.add_argument("--tier", default="D0")
parser.add_argument("--episodes", type=int, default=4, help="episodes for hold/executor (num_envs per batch)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--rate_hz", type=int, choices=[10, 30], default=10, help="challenge track: rows held 0.1 s or 1/30 s")
parser.add_argument("--replay_episodes", nargs="*", default=[])
parser.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0])
parser.add_argument("--tol_mm", type=float, default=5.0)
parser.add_argument("--bundle", default=None, help="executor: teacher bundle / graph directory")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

# the executor's IK / FK / policy calls are tiny tensors: torch's default intra-op threads (one per core) cost ~6
# cores per process and made rehearsals slower (22.5 s -> 15.2 s per Task 3-2 batch with 1 thread)
torch.set_num_threads(1)
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.config import load_yaml, result_path  # noqa: E402
from tower_rl.data.episode import Episode  # noqa: E402
from tower_rl.domain import DomainSampler  # noqa: E402
from tower_rl.eval.metrics import aggregate  # noqa: E402
from tower_rl.eval.operations import OpCfg  # noqa: E402
from tower_rl.eval.runner import DemoReplay, HoldStill, run_batch  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402

phys = load_yaml("physics")
n_envs = 1 if args.policy == "replay" else args.num_envs
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(make_station_scene_cfg(args.variant, n_envs))
sim.reset()
world = StationWorld(scene, sim, args.variant, finger_friction=tuple(args.finger_friction))
op_cfg = OpCfg(tol_pos_mm=args.tol_mm)
outcomes = []
t0 = time.time()
if args.policy == "replay":
    for k, f in enumerate(args.replay_episodes):
        ep = Episode.load(f)
        outs = run_batch(world, DemoReplay(ep, args.rate_hz), None, seed=args.seed + k, max_time_s=args.max_time_s, op_cfg=op_cfg,
                         episode=ep, rate_hz=args.rate_hz)
        for o in outs:
            o.extra["recording"] = ep.name
        outcomes += outs
        o = outs[0]
        print(f"[eval] {ep.name}: G {o.grasp_ok} P {o.place_ok} success {o.task_success} disturbed {o.disturbed_per_op} "
              f"t={o.duration_s:.1f}s", flush=True)
else:
    if args.policy == "hold":
        policy = HoldStill()
    else:
        from tower_rl.graph.executor import Executor  # noqa: E402
        policy = Executor.from_bundle(args.bundle, args.variant, device=str(world.dev), rate_hz=args.rate_hz)
    sampler = DomainSampler(args.tier, device=world.dev)
    for b in range((args.episodes + n_envs - 1) // n_envs):
        outs = run_batch(world, policy, sampler, seed=args.seed + b, max_time_s=args.max_time_s, op_cfg=op_cfg, rate_hz=args.rate_hz)
        outcomes += outs[: args.episodes - b * n_envs]
        for o in outs:
            print(f"[eval] case {o.extra['case']}: G {o.grasp_ok} P {o.place_ok} success {o.task_success} "
                  f"disturbed {o.disturbed_per_op} t={o.duration_s:.1f}s error={o.error}", flush=True)
metrics = aggregate(outcomes)
cfg = {"variant": args.variant, "policy": args.policy, "tier": args.tier, "seed": args.seed, "tol_mm": args.tol_mm, "rate_hz": args.rate_hz,
       "finger_friction": args.finger_friction, "device": args.device, "bundle": args.bundle}
out = Path(args.out) if args.out else result_path(f"eval/{args.variant}", cfg, tag=args.policy)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"config": cfg, "metrics": metrics, "wall_s": round(time.time() - t0, 1),
                           "episodes": [o.to_dict() for o in outcomes]}, indent=1, default=str))
print("REPORT_JSON " + json.dumps({"out": str(out), **{k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in metrics.items()}}), flush=True)
os._exit(0)
