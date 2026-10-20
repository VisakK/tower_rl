"""Step 3.5: how far can a placed block sit from its goal and the stack still stand? (CPU station scene)

For one variant, every env holds the complete destination stack (all transfers at their goals, measured rest
heights); the top layer is then offset by d (along the block's long axis or across it) and yaw theta, the stack runs
5 s, and an env "stands" when no block moved > 10 mm and none tilted > 10 deg. Prints the standing matrix; tol_pos /
tol_yaw are the largest offsets that keep every stack standing.

Usage: python scripts/sweep_tolerance.py --headless --device cpu --variant task2-3_cross_tower_transfer_3
"""
import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", default="task2-3_cross_tower_transfer_3")
parser.add_argument("--offsets_mm", nargs="+", type=float, default=[0, 3, 5, 7.5, 10, 15, 20])
parser.add_argument("--yaws_deg", nargs="+", type=float, default=[0, 2.5, 5, 7.5, 10])
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.paths import RESULTS_DIR  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402
from tower_rl.tower import foundation_poses  # noqa: E402

conds = list(itertools.product(["long", "width"], args.offsets_mm, args.yaws_deg))
n = len(conds)
phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(make_station_scene_cfg(args.variant, n, env_spacing=2.0))
sim.reset()
world = StationWorld(scene, sim, args.variant)
spec, dev = world.spec, world.dev
K = len(spec.transfer_arm)
top = int(spec.transfer_dest_layer.max())
poses = torch.zeros(n, len(world.block_ids), 7, device=dev)
poses[:, :6] = foundation_poses(spec, n)
mov = torch.zeros(n, spec.num_movable, 7, device=dev)
for k in range(K):
    b = int(spec.transfer_block[k])
    g = spec.goal[k]
    layer = int(spec.transfer_dest_layer[k])
    mov[:, b, :2] = g[:2]
    mov[:, b, 2] = world.rz[layer] + 0.0005
    mov[:, b, 3:7] = g[3:7]
    if layer == top:  # offset the top layer
        for e, (direction, d, yaw) in enumerate(conds):
            r = geo.quat_to_mat(g[3:7][None])[0]
            axis = r[:, 0] if direction == "long" else r[:, 1]
            mov[e, b, :3] += axis * d / 1000
            mov[e, b, 3:7] = geo.quat_mul(geo.quat_from_yaw(torch.tensor(math.radians(yaw), device=dev)), g[3:7])
poses[:, world.mov_idx] = mov
world._write_blocks(poses)
p0 = poses[:, world.mov_idx, :3].clone()
for _ in range(int(args.seconds / world.dt)):
    world._substep(world._cmd)
pos, quat = world.block_pose()
pos, quat = pos[:, world.mov_idx], quat[:, world.mov_idx]
disp = (pos - p0).norm(dim=-1).amax(-1) * 1000
tilt = torch.rad2deg(geo.tilt_angle(quat)).amax(-1)
stand = (disp < 10) & (tilt < 10)
table = {}
for e, (direction, d, yaw) in enumerate(conds):
    table.setdefault(direction, {}).setdefault(f"{d:g}mm", {})[f"{yaw:g}deg"] = {"stands": bool(stand[e]), "max_disp_mm": round(float(disp[e]), 1),
                                                                              "max_tilt_deg": round(float(tilt[e]), 1)}
for direction in ("long", "width"):
    print(f"== offset along {direction} (rows mm, columns deg): S = stands, x = moved/tilted", flush=True)
    for d in args.offsets_mm:
        row = table[direction][f"{d:g}mm"]
        print(f"  {d:5g} mm: " + " ".join("S" if row[f"{y:g}deg"]["stands"] else "x" for y in args.yaws_deg), flush=True)
out = Path(args.out) if args.out else RESULTS_DIR / "tolerance" / f"{args.variant}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"variant": args.variant, "top_layer": top, "seconds": args.seconds, "table": table}, indent=1))
print("REPORT_JSON " + json.dumps({"variant": args.variant, "standing": int(stand.sum()), "of": n}), flush=True)
os._exit(0)
