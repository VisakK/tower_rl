"""Step 7.5: rehearse the teacher (executor + skills) in the station scene and write feedback.md per variant/tier.

CPU physics (evaluator-faithful). Example:
  python scripts/rehearse.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --tiers D0 D1 --episodes 20 --num_envs 4
One variant per process (Isaac stages are not rebuilt in-process); loop in the shell for several.
Outputs: results/rehearsal/<variant>/<tier>/{feedback.md, metrics.json, episodes.json}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", required=True)
parser.add_argument("--tiers", nargs="+", default=["D0", "D1"])
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--bundle", default=None, help="teacher bundle (graphs/ + skills/); default: in-repo graphs, planned skills")
parser.add_argument("--rate_hz", type=int, choices=[10, 30], default=10)
parser.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0])
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--perception_noise", action="store_true", help="Step 8.1 provisional noise on block poses")
parser.add_argument("--noise", default="{}", help='JSON NoiseCfg overrides, e.g. {"bias_pos_m": 0.001, "shared_pos_m": 0.003}')
parser.add_argument("--graph_set", default="{}", help='JSON merged into the graph\'s top level, e.g. {"perception": {"jaw_to_goal": 1.0}}')
parser.add_argument("--out_root", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

# the executor's IK / FK / policy calls are tiny tensors: torch's default intra-op threads (one per core) cost ~6
# cores per process and made rehearsals slower (22.5 s -> 15.2 s per Task 3-2 batch with 1 thread)
torch.set_num_threads(1)
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.domain import DomainSampler  # noqa: E402
from tower_rl.eval.metrics import aggregate  # noqa: E402
from tower_rl.eval.perception_noise import NoiseCfg  # noqa: E402
from tower_rl.eval.runner import run_batch  # noqa: E402
from tower_rl.graph.executor import Executor  # noqa: E402
from tower_rl.graph.rehearsal import write_report  # noqa: E402
from tower_rl.paths import RESULTS_DIR  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402

phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
summary = {}
for variant in [args.variant]:
    scene = InteractiveScene(make_station_scene_cfg(variant, args.num_envs))
    sim.reset()
    world = StationWorld(scene, sim, variant, finger_friction=tuple(args.finger_friction))
    policy = Executor.from_bundle(args.bundle, variant, device=str(world.dev), rate_hz=args.rate_hz)
    policy.graph.update(json.loads(args.graph_set))
    for tier in args.tiers:
        sampler = DomainSampler(tier, device=world.dev, max_blocks=max(6, world.spec.num_movable))
        outcomes = []
        t0 = time.time()
        nb = (args.episodes + args.num_envs - 1) // args.num_envs
        for b in range(nb):
            outs = run_batch(world, policy, sampler, seed=args.seed + 100 * b, max_time_s=args.max_time_s, rate_hz=args.rate_hz,
                             noise=NoiseCfg(**json.loads(args.noise)) if args.perception_noise else None)
            outcomes += outs[: args.episodes - b * args.num_envs]
            done = aggregate(outcomes)
            print(f"[rehearse] {variant} {tier} batch {b + 1}/{nb}: TSR {done['TSR']['value']:.2f} GSR {done['GSR']['value']:.2f} "
                  f"PSR {done['PSR']['value']:.2f} ODR {done['ODR']['value']:.2f} ({time.time() - t0:.0f} s)", flush=True)
        out_dir = Path(args.out_root or RESULTS_DIR / "rehearsal") / variant / (tier + (("_noise" + ("_" + "_".join(f"{k}{v}" for k, v in sorted(json.loads(args.noise).items())) if json.loads(args.noise) else "")) if args.perception_noise else ""))
        cfg = {"variant": variant, "tier": tier, "episodes": args.episodes, "seed": args.seed, "bundle": args.bundle,
               "perception_noise": args.perception_noise, "noise": json.loads(args.noise),
               "rate_hz": args.rate_hz, "finger_friction": args.finger_friction, "device": args.device,
               "graph_set": json.loads(args.graph_set)}
        path = write_report(outcomes, out_dir, f"{variant} / {tier}", cfg)
        summary[f"{variant}/{tier}"] = {k: aggregate(outcomes)[k]["value"] for k in ("TSR", "GSR", "PSR", "LHPR", "ODR")}
        print(f"[rehearse] wrote {path}", flush=True)
print("REPORT_JSON " + json.dumps(summary), flush=True)
os._exit(0)
