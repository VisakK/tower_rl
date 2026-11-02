"""Payload check for the reaching env (Reach-PPO-plan.MD P2): with zero actions, does the arm hold its commanded pose
while holding a block? Half the envs carry ``payload.kg``; the pinch error to FK(q_des) after ``--rows`` rows, per group.

  python scripts/reach_payload_check.py --headless --arm left --force 1    # the block's weight applied in the sim
  python scripts/reach_payload_check.py --headless --arm left --force 0    # compensation only (overshoots upward)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="left")
ap.add_argument("--num_envs", type=int, default=128)
ap.add_argument("--rows", type=int, default=30)
ap.add_argument("--force", type=int, default=1)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.skills.reach_env import ReachEnv, ReachEnvCfg  # noqa: E402

ov = {"arm": args.arm, "payload": {"frac": 0.5, "force": bool(args.force)}, "reset": {"uniform_frac": 0.0}}
cfg = ReachEnvCfg()
cfg.setup(num_envs=args.num_envs, device=args.device, overrides=ov)
env = ReachEnv(cfg)
env.reset()
zero = torch.zeros(env.num_envs, 7, device=env.device)
for _ in range(args.rows):
    env.step(zero)
q = env.robot.data.joint_pos[:, :7]
pb, qb = env.kin.fk(q, "pinch")
pd, qdes = env.kin.fk(env.q_des, "pinch")
pos = (pb - pd).norm(dim=-1) * 1000
rot = torch.rad2deg(geo.quat_angle(qb, qdes))
dz = (pb - pd)[:, 2] * 1000
out = {"arm": args.arm, "force": bool(args.force), "rows": args.rows}
for name, m in (("payload", env.payload > 0), ("none", env.payload == 0)):
    if m.any():
        out[name] = {"n": int(m.sum()), "pos_mm_p50": round(pos[m].median().item(), 2),
                     "pos_mm_max": round(pos[m].max().item(), 2), "dz_mm_mean": round(dz[m].mean().item(), 2),
                     "rot_deg_p50": round(rot[m].median().item(), 3)}
print("PAYLOAD_JSON", json.dumps(out), flush=True)
os._exit(0)  # app.close() hangs in Isaac Sim 5.1
