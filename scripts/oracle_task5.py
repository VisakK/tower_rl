"""Sanity check of the Task 5 skill envs with scripted privileged oracles through the same action interface.

insert: align the held block on the slot line (true poses), slide in at 1 mm per step, release at ``entry_m``, back off
2 cm along the axis, rise. push_in: move behind the outer end face at the block's height, push along the axis until the
block is centred, hold. If these cannot reach success, the env (geometry, physics, success / failure logic) is wrong.

Usage: python scripts/oracle_task5.py --headless --skill insert --arm left --num_envs 64 [--ig_override '{...}']
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--skill", choices=["insert", "push_in"], default="insert")
parser.add_argument("--arm", default="left")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--override", default="{}")
parser.add_argument("--ig_override", default="{}", help='JSON merged into the insert geometry, e.g. {"entry_lift_m": {"cuboid": 0.0}}')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

torch.set_num_threads(2)
from tower_rl import geometry as geo  # noqa: E402
from tower_rl.config import deep_update  # noqa: E402

ov = deep_update({"arm": args.arm, "action": {"ema": 1.0}}, json.loads(args.override))
if args.skill == "insert":
    from tower_rl.skills.task5_env import InsertEnv as Env, InsertEnvCfg as EnvCfg
else:
    from tower_rl.skills.task5_env import PushInEnv as Env, PushInEnvCfg as EnvCfg
from tower_rl.skills.task5_env import INSERT_SLOT  # noqa: E402

cfg = EnvCfg()
cfg.setup(num_envs=args.num_envs, device=args.device, overrides=ov)
env = Env(cfg)
env.ig = deep_update(env.ig, json.loads(args.ig_override))
env.reset()
n, dev = env.num_envs, env.device
ac, ig = env.sk["action"], env.ig
phase = torch.zeros(n, dtype=torch.long, device=dev)
k = torch.zeros(n, dtype=torch.long, device=dev)
done_once = torch.zeros(n, dtype=torch.bool, device=dev)
outcome = [""] * n
info = []
_d0 = env._get_dones


def _stash():
    out = _d0()
    env._st_term = {kk: v.clone() for kk, v in env.st.items()}
    return out


env._get_dones = _stash
for t in range(env.max_episode_length):
    s = env._common_state()
    d = env.axis
    lat = torch.stack([-d[:, 1], d[:, 0], torch.zeros(n, device=dev)], -1)
    tp, tq = s["tpos"], s["tquat"]
    tgt = env.ctrl.pos.clone()
    a = torch.zeros(n, env.cfg.action_space, device=dev)
    rotv = torch.zeros(n, 3, device=dev)
    if args.skill == "insert":
        e_lat = -((tp - env.slot_pos) * lat).sum(-1)
        e_z = env.slot_pos[:, 2] + ig["entry_lift_m"][env.sk["scene"].get("collider", "hull")] - tp[:, 2]
        e_yaw = torch.remainder(geo.yaw_from_quat(env.slot_quat) - geo.yaw_from_quat(tq) + torch.pi / 2, torch.pi) - torch.pi / 2
        aligned = (e_lat.abs() < 0.001) & (e_z.abs() < 0.001) & (e_yaw.abs() < 0.009)
        along = s["along"]
        entering = along > -0.16
        adv = torch.where(aligned | ~entering, torch.full_like(along, 0.001), torch.zeros_like(along))
        adv = torch.minimum(adv, (-ig["entry_m"] - along).clamp(min=0))
        delta = d * adv[:, None] + lat * (0.6 * e_lat).clamp(-0.002, 0.002)[:, None]
        delta[:, 2] += (0.6 * e_z).clamp(-0.002, 0.002)
        rotv[:, 2] = 0.6 * e_yaw
        at_depth = (along >= -ig["entry_m"] - 0.003)
        phase = torch.where((phase == 0) & at_depth, torch.ones_like(phase), phase)
        k = torch.where(phase >= 1, k + 1, k)
        # 1: open (6 steps); 2: back off 4 steps along -d; 3: rise
        opening = phase == 1
        phase = torch.where(opening & (k > 6), torch.full_like(phase, 2), phase)
        back = phase == 2
        phase = torch.where(back & (k > 10), torch.full_like(phase, 3), phase)
        delta = torch.where((phase == 2)[:, None], -d * 0.005, delta)
        up = torch.zeros_like(delta)
        up[:, 2] = 0.008
        delta = torch.where((phase == 3)[:, None], up, delta)
        delta = torch.where((phase == 1)[:, None], torch.zeros_like(delta), delta)
        rotv = torch.where((phase >= 1)[:, None], torch.zeros_like(rotv), rotv)
        a[:, 6] = torch.where(phase >= 1, -1.0, 1.0)
    else:
        cp = tp + d * (ig["pushin_contact_along_m"] - 0.004)
        cp[:, 2] = env.slot_pos[:, 2]
        dv = cp - tgt
        phase = torch.where((phase == 0) & (dv.norm(dim=-1) < 0.002), torch.ones_like(phase), phase)
        approach = dv * (0.008 / dv.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
        on_line = (env.slot_pos - tgt) - d * ((env.slot_pos - tgt) * d).sum(-1, keepdim=True)  # back onto the slot line
        end = env.slot_pos + d * ig["pushin_contact_along_m"]
        rem = ((end - tgt) * d).sum(-1)
        push = on_line * 0.5 + d * rem.clamp(0, 0.002)[:, None]
        delta = torch.where((phase == 0)[:, None], approach, push)
    scale = env._action_scale()[:, None]
    a[:, :3] = env.base.vec_to_base(delta) / scale
    a[:, 3:6] = env.base.vec_to_base(rotv) / ac["rot_rad"]
    _, _, term, trunc, _ = env.step(a.clamp(-1, 1))
    new = (term | trunc) & ~done_once
    for e in new.nonzero(as_tuple=True)[0].tolist():
        outcome[e] = "success" if bool(env.term["success"][e]) else "failed" if bool(env.term["failed"][e]) else "timeout"
        st = env._st_term
        info.append({"outcome": outcome[e], "step": t + 1, "along_mm": round(float(st["along"][e]) * 1000, 1),
                     "lat_mm": round(float(st["lat"][e]) * 1000, 2), "vert_mm": round(float(st["vert"][e]) * 1000, 1),
                     "moved_mm": round(float(st["moved"][e]) * 1000, 1), "tilt_deg": round(float(st["tilt"][e]) * 57.3, 1)})
    done_once |= term | trunc
    phase = torch.where(term | trunc, torch.zeros_like(phase), phase)
    k = torch.where(term | trunc, torch.zeros_like(k), k)
    if bool(done_once.all()):
        break
c = Counter(outcome)
fails = [x for x in info if x["outcome"] != "success"]
print("REPORT_JSON " + json.dumps({"skill": args.skill, "arm": args.arm, "n": n, "outcomes": dict(c),
                                   "success_rate": round(c["success"] / n, 3), "fail_examples": fails[:10],
                                   "success_examples": [x for x in info if x["outcome"] == "success"][:4]}), flush=True)
os._exit(0)
