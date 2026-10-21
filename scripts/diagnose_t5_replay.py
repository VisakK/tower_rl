"""Task 4/5 replay diagnosis: replay one recording open loop in the CPU station scene and report, once per second, how
far each movable block of our simulation is from the recorded pose (and which demo event happened then).

Usage: python scripts/diagnose_t5_replay.py --headless --device cpu --episode data/subset/<variant>/<stem>.npz \
           [--finger_friction 1.2 1.0] [--every_s 1.0] [--out results/t45/replay/diag.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--episode", required=True)
parser.add_argument("--finger_friction", nargs=2, type=float, default=[1.2, 1.0])
parser.add_argument("--every_s", type=float, default=1.0)
parser.add_argument("--max_time_s", type=float, default=None)
parser.add_argument("--layout_override", default="{}", help='JSON merged into the layout physics, e.g. {"collider": "cuboid"}')
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.data.episode import Episode  # noqa: E402
from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim import station_scene as ss  # noqa: E402

ep = Episode.load(args.episode)
variant = ep.variant
over = json.loads(args.layout_override)
if over:  # e.g. try the cuboid collider on Task 5-2
    _load = Layout.load

    def _patched(v, path=None):
        lay = _load(v, path)
        lay.physics = {**lay.physics, **over}
        return lay
    ss.Layout.load = staticmethod(_patched)
phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(ss.make_station_scene_cfg(variant, 1))
sim.reset()
world = ss.StationWorld(scene, sim, variant, finger_friction=tuple(args.finger_friction))
world.reset_from_episode(ep)
cmd = ep.qpos_applied.astype(np.float32)
rec = ep.block_poses  # [T, B, 7] xyzw
rec_ids = ep.block_ids
mov = [b for b in world.block_ids if not world.layout.fixed[b]]
events = [(e["time"], e["event"], e.get("block_id", "")) for e in ep.events if e.get("event") not in ("phase", "placement_ready_open_loop")]
every = max(1, int(round(args.every_s * 30)))
T = len(cmd) if args.max_time_s is None else min(len(cmd), int(args.max_time_s * 30))
rows = []
first_dev = {}
for t in range(T):
    world.apply_row(torch.as_tensor(cmd[t])[None], ticks=1)
    if t % every == 0 or t == T - 1:
        pos, quat = world.block_pose()
        row = {"t": round(t / 30, 2), "events": [e[1] + ":" + e[2] for e in events if t / 30 - args.every_s < e[0] <= t / 30]}
        for b in mov:
            i = world.block_ids.index(b)
            j = rec_ids.index(b)
            d = float(np.linalg.norm(pos[0, i].cpu().numpy() - rec[t, j, :3])) * 1000
            row[b] = round(d, 1)
            if d > 3.0 and b not in first_dev:
                first_dev[b] = (round(t / 30, 2), round(d, 1), row["events"])
        rows.append(row)
        print(json.dumps(row), flush=True)
print("FIRST_DEVIATION > 3 mm: " + json.dumps(first_dev, indent=1), flush=True)
if args.out:
    Path(args.out).write_text(json.dumps({"episode": args.episode, "rows": rows, "first_dev": first_dev}, indent=1))
os._exit(0)
