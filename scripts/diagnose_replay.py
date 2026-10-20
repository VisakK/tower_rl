"""Where does an open-loop replay diverge from the recording? Per frame: target block error, arm joint error, finger
width, for every parallel replay of one episode. Writes an npz + a short JSON summary.

Usage: python scripts/diagnose_replay.py --headless --episode data/subset/<variant>/<stem>.npz --num_replays 10 --out results/calibration/diag.npz
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
parser.add_argument("--num_replays", type=int, default=10)
parser.add_argument("--finger_friction", nargs=2, type=float, default=[1.2, 1.0])
parser.add_argument("--block_friction", nargs=2, type=float, default=None)
parser.add_argument("--physx", default="{}", help="JSON dict of PhysxCfg overrides")
parser.add_argument("--block_iters", nargs=2, type=int, default=None)
parser.add_argument("--delay_substeps", nargs=2, type=int, default=[1, 1])
parser.add_argument("--settle_s", type=float, default=1.0, help="settle blocks (raised 0.5 mm) before replaying; 0 = teleport only")
parser.add_argument("--end_frame", type=int, default=None)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.data.events import operations  # noqa: E402
from tower_rl.paths import STATION_JSON  # noqa: E402
from tower_rl.scene import (CONTROL_SUBSTEPS, PHYSICS_DT, DelayedCommands, Episode, apply_friction,  # noqa: E402
                            load_station_meta, make_scene_cfg, measured_q16, set_arm_targets, settle_recorded_state, write_recorded_state)

import tower_rl.scene as _scene_mod  # noqa: E402
if args.block_iters:
    _scene_mod.BLOCK_SOLVER_ITERATIONS[:] = args.block_iters
ep = Episode.load(args.episode)
meta = load_station_meta(STATION_JSON)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=args.device,
                                                          physx=sim_utils.PhysxCfg(**json.loads(args.physx))))
scene = InteractiveScene(make_scene_cfg(meta, ep, args.num_replays))
sim.reset()
apply_friction(scene, finger=tuple(args.finger_friction), block=tuple(args.block_friction) if args.block_friction else None)
dev = sim.device
if args.settle_s > 0:
    settle_recorded_state(scene, sim, ep, 0, seconds=args.settle_s)
else:
    write_recorded_state(scene, ep, 0)
lq = torch.tensor(ep.qpos_cmd, dtype=torch.float32, device=dev)
T = args.end_frame or lq.shape[0]
ops = operations(ep)
bi = ops[0].block_index
blk = scene["blocks"]
rec_bp = torch.tensor(ep.block_poses[:, :, :3], device=dev)
rec_q = torch.tensor(ep.qpos_obs, device=dev)
berr, qerr, width = [], [], []
cmd = DelayedCommands(lq[0], tuple(args.delay_substeps))
for t in range(T - 1):
    for _ in range(CONTROL_SUBSTEPS):
        set_arm_targets(scene, cmd.push(lq[t]))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(PHYSICS_DT)
    pos = blk.data.object_link_pos_w[:, bi] - scene.env_origins
    berr.append((pos - rec_bp[t + 1, bi]).cpu().numpy())
    q = torch.stack([measured_q16(scene, e) for e in range(args.num_replays)])
    qerr.append((q - rec_q[t + 1]).cpu().numpy())
berr, qerr = np.stack(berr), np.stack(qerr)  # [T, R, 3], [T, R, 16]
np.savez_compressed(args.out, berr=berr, qerr=qerr, close=ops[0].close_frame, open=ops[0].open_frame)
o = ops[0]
summ = {"episode": ep.name, "op": o.to_dict()}
for name, f in (("pre_close", o.close_frame - 1), ("close+30", o.close_frame + 30), ("close+80", o.close_frame + 80),
                ("mid_carry", (o.close_frame + o.open_frame) // 2), ("open", o.open_frame), ("open+30", o.open_frame + 30)):
    if f < T - 1:
        summ[name] = {"block_err_mm_per_replay": (np.linalg.norm(berr[f], axis=-1) * 1000).round(1).tolist(),
                      "left_joint_err_max_rad_per_replay": np.abs(qerr[f, :, :7]).max(-1).round(4).tolist(),
                      "left_width_err_mm_per_replay": (qerr[f, :, 7] * 1000).round(2).tolist()}
Path(args.out).with_suffix(".json").write_text(json.dumps(summ, indent=1))
print("REPORT_JSON " + json.dumps(summ), flush=True)
os._exit(0)
