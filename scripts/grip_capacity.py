"""Grip capacity under the 3 N finger limit: replay one Task 4/5 recording open loop in N CPU station envs, each with
the grasped block's mass scaled and / or the held gripper command replaced, and report whether the block stays in the
fingers through the carry (its height vs the recording while the recording has it lifted).

Usage: python scripts/grip_capacity.py --headless --device cpu --episode data/subset/<variant>/<stem>.npz \
           [--scales 1 1.4 2] [--widths 0 0.043 0.0425] [--block <id>] [--until_s 30]
``--widths``: 0 = the recorded applied command; w > 0 replaces the left / back gripper command while it is below 46 mm.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--episode", required=True)
parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.4, 1.7, 2.0])
parser.add_argument("--widths", nargs="+", type=float, default=[0.0, 0.043, 0.0425, 0.040])
parser.add_argument("--block", default=None, help="grasped block (default: the first transfer's)")
parser.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0])
parser.add_argument("--until_s", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.data.episode import Episode  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim import station_scene as ss  # noqa: E402

ep = Episode.load(args.episode)
combos = list(itertools.product(args.scales, args.widths))
n = len(combos)
phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(ss.make_station_scene_cfg(ep.variant, n))
sim.reset()
world = ss.StationWorld(scene, sim, ep.variant, finger_friction=tuple(args.finger_friction))
world.reset_from_episode(ep)
bid = args.block or world.layout.transfers[0].block
i_b = world.block_ids.index(bid)
j_b = ep.block_ids.index(bid)
view = world.blocks.root_physx_view
ids = torch.tensor([i_b * n + e for e in range(n)])
masses, inert = view.get_masses(), view.get_inertias()
m0 = float(masses[ids[0]])
scale = torch.tensor([c[0] for c in combos])
masses[ids] = masses[ids] * scale.reshape(masses[ids].shape)
inert[ids] = inert[ids] * scale[:, None]
view.set_masses(masses, ids.to(torch.int32))
view.set_inertias(inert, ids.to(torch.int32))
cmd = torch.tensor(ep.qpos_applied, dtype=torch.float32)
rec = ep.block_poses
z0 = float(rec[0, j_b, 2])
wid = torch.tensor([c[1] for c in combos])
T = min(len(cmd), int(args.until_s * 30))
dev_max = torch.zeros(n)
lifted_rows = 0
first_drop = [None] * n
for t in range(T):
    c = cmd[t][None].repeat(n, 1)
    for g in (7, 15):
        sub = (wid > 0) & (c[:, g] < 0.046)
        c[sub, g] = wid[sub]
    world.apply_row(c, ticks=1)
    if rec[t, j_b, 2] > z0 + 0.03:  # the recording carries the block
        lifted_rows += 1
        pos, _ = world.block_pose()
        d = (pos[:, i_b, 2].cpu() - float(rec[t, j_b, 2])).abs()
        dev_max = torch.maximum(dev_max, d)
        for e in range(n):
            if first_drop[e] is None and float(d[e]) > 0.015:
                first_drop[e] = round(t / 30, 2)
out = [{"mass_scale": s, "mass_kg": round(m0 * s, 4), "hold_width": w or "recorded", "max_z_dev_mm": round(float(dev_max[e]) * 1000, 1),
        "dropped_at_s": first_drop[e]} for e, (s, w) in enumerate(combos)]
print("REPORT_JSON " + json.dumps({"episode": args.episode, "block": bid, "lifted_rows": lifted_rows, "runs": out}), flush=True)
os._exit(0)
