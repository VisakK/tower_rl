"""Time the parts of one skill-env step (physics, action processing, dones/rewards/obs, resets).

Usage: python scripts/profile_skill_env.py --headless --num_envs 4096 --steps 20
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--skill", default="grasp_lift")
parser.add_argument("--arm", default="left")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import torch  # noqa: E402

if args.skill == "grasp_lift":
    from tower_rl.skills.grasp_lift_env import GraspLiftEnv as Env, GraspLiftEnvCfg as EnvCfg
else:
    from tower_rl.skills.place_release_env import PlaceReleaseEnv as Env, PlaceReleaseEnvCfg as EnvCfg

cfg = EnvCfg()
cfg.setup(num_envs=args.num_envs, device=args.device, overrides={"arm": args.arm})
env = Env(cfg)
env.reset()
T = defaultdict(float)


def timed(name, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t
        return out
    return wrap


env._pre_physics_step = timed("pre_physics(action->IK)", env._pre_physics_step)
env._apply_action = timed("apply_action", env._apply_action)
env._get_dones = timed("dones(+state)", env._get_dones)
env._get_rewards = timed("rewards", env._get_rewards)
env._get_observations = timed("observations", env._get_observations)
env._reset_idx = timed("reset_idx", env._reset_idx)
env._sample_towers = timed("  reset/_sample_towers", env._sample_towers)
env._apply_dynamics = timed("  reset/_apply_dynamics", env._apply_dynamics)
env._set_arm = timed("  reset/_set_arm", env._set_arm)
env.kin.ik = timed("  reset+ctrl/kin.ik", env.kin.ik)
_n_reset = [0]
_orig_reset = env._reset_idx


def _count(ids):
    _n_reset[0] += len(ids)
    return _orig_reset(ids)


env._reset_idx = _count
env.sim.step = timed("sim.step", env.sim.step)
env.scene.update = timed("scene.update", env.scene.update)
env.scene.write_data_to_sim = timed("write_data_to_sim", env.scene.write_data_to_sim)
for k in list(T):
    T[k] = 0.0
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(args.steps):
    a = torch.rand(env.num_envs, env.cfg.action_space, device=env.device) * 2 - 1
    env.step(a)
torch.cuda.synchronize()
total = time.perf_counter() - t0
rep = {"num_envs": args.num_envs, "steps": args.steps, "total_s_per_step": round(total / args.steps, 4),
       "env_steps_per_s": round(args.num_envs * args.steps / total, 1), "resets_per_step": _n_reset[0] / args.steps,
       **{k: round(v / args.steps, 4) for k, v in sorted(T.items(), key=lambda kv: -kv[1])}}
print("REPORT_JSON " + json.dumps(rep), flush=True)
os._exit(0)
