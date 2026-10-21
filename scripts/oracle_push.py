"""Sanity check of the push env (Task 4) with a scripted privileged oracle through the same action interface: move the
pinch to the contact pose behind the target's end face (true poses), then push straight along the push direction at
``--speed`` m per 0.1 s row, keeping the pinch on the block's centre line. If this cannot reach success, the env is wrong
(geometry, physics, success / failure logic), not the learner.

Usage: python scripts/oracle_push.py --headless --arm left --num_envs 256 [--speed 0.01] [--override '{...}']
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
parser.add_argument("--arm", default="left")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--speed", type=float, default=0.01)
parser.add_argument("--zoff", type=float, default=None, help="pinch height above the block centre (default: demo prior)")
parser.add_argument("--override", default="{}")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl import push as P  # noqa: E402
from tower_rl.config import deep_update  # noqa: E402
from tower_rl.skills.push_env import PushEnv, PushEnvCfg  # noqa: E402

ov = deep_update({"arm": args.arm, "action": {"ema": 1.0}}, json.loads(args.override))
cfg = PushEnvCfg()
cfg.setup(num_envs=args.num_envs, device=args.device, overrides=ov)
env = PushEnv(cfg)
env.reset()
n, dev = env.num_envs, env.device
ac = env.sk["action"]
g = env.sk["geometry"]
phase = torch.zeros(n, dtype=torch.long, device=dev)
done_once = torch.zeros(n, dtype=torch.bool, device=dev)
outcome = [""] * n
steps = torch.zeros(n, dtype=torch.long, device=dev)
dist_at_end = torch.zeros(n, device=dev)
fail_info = []
_dones0 = env._get_dones


def _dones_stash():  # keep the terminal state: the env auto-resets before step() returns
    out = _dones0()
    env._st_term = {k: v.clone() for k, v in env.st.items()}
    env._p0_term = env.block_p0.clone()
    env._meta_term = (env.layer.clone(), env.env_stage.clone(), list(env.env_case), env.present.clone())
    return out


env._get_dones = _dones_stash
for t in range(env.max_episode_length):
    s = env._compute_state()
    zoff = env._zoff() if args.zoff is None else torch.full((n,), args.zoff, device=dev)
    cp, cq = P.push_pose(s["tpos"], env.dir, env.arm, g["contact_along_m"] - 0.006, zoff)
    tgt = env.ctrl.pos.clone()
    approach = phase == 0
    d_app = cp - tgt
    phase = torch.where(approach & (d_app.norm(dim=-1) < 0.003), torch.ones_like(phase), phase)
    # push: advance along the direction, hold the lateral line and height of the block's start
    p0 = env.block_p0[torch.arange(n, device=dev), env.target_slot]
    lat = torch.stack([-env.dir[:, 1], env.dir[:, 0], torch.zeros(n, device=dev)], -1)
    on_line = tgt - lat * ((tgt - p0) * lat).sum(-1, keepdim=True)
    on_line[:, 2] = p0[:, 2] + zoff
    desired_push = on_line + env.dir * args.speed
    desired = torch.where(phase[:, None] == 0, cp, desired_push)
    scale = env._action_scale()[:, None]
    a = torch.zeros(n, 6, device=dev)
    a[:, :3] = env.base.vec_to_base(desired - tgt) / scale
    a[:, 3:6] = env.base.vec_to_base(geo.quat_error_rotvec(env.ctrl.quat, cq)) / ac["rot_rad"]
    _, _, term, trunc, _ = env.step(a.clamp(-1, 1))
    new = (term | trunc) & ~done_once
    for e in new.nonzero(as_tuple=True)[0].tolist():
        outcome[e] = "success" if bool(env.term["success"][e]) else "failed" if bool(env.term["failed"][e]) else "timeout"
        steps[e] = t + 1
        if outcome[e] == "failed":  # culprit: the most displaced other block, same layer as the target or below
            st = env._st_term
            dn = st["disp_nt"][e]
            j = int(dn.argmax())
            tl = int(env._meta_term[0][e])
            p0 = env._p0_term[e]
            jl = int(round(float((p0[j, 2] - 0.015) / 0.029)))
            fail_info.append({"case": env._meta_term[2][e], "stage": int(env._meta_term[1][e]), "target_layer": tl,
                              "culprit_layer": jl, "culprit_disp_mm": round(float(dn[j]) * 1000, 1),
                              "target_along_mm": round(float(st["along"][e]) * 1000, 1), "target_lat_mm": round(float(st["lat"][e]) * 1000, 1),
                              "step": t + 1})
    done_once |= term | trunc
    phase = torch.where(term | trunc, torch.zeros_like(phase), phase)
    if bool(done_once.all()):
        break
c = Counter(outcome)
cases = Counter()
for e in range(n):
    cases[(env.env_case[e].split("/")[0], outcome[e])] += 1
res = {"arm": args.arm, "n": n, "outcomes": dict(c), "success_rate": round(c["success"] / n, 3),
       "mean_steps_success": round(float(steps[[o == "success" for o in outcome]].float().mean()), 1) if c["success"] else None,
       "by_tier": {f"{k[0]}:{k[1]}": v for k, v in sorted(cases.items())},
       "fail_culprit": dict(Counter(("same" if f["culprit_layer"] == f["target_layer"] else "below" if f["culprit_layer"] < f["target_layer"] else "above")
                                    for f in fail_info)),
       "fail_examples": [f for f in fail_info if f["step"] > 3][:16], "fail_steps": sorted(f["step"] for f in fail_info)}
print("REPORT_JSON " + json.dumps(res), flush=True)
os._exit(0)
