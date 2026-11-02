"""Drive response in the reaching env, GPU vs CPU PhysX (Reach-PPO-plan.MD §7: on CPU the left reacher overshot descents
by up to 7 cm). One joint per env group: q_des ramps at ``--speed`` rad/s for ``--ramp`` rows, then holds; joint angles
logged every physics substep. Prints per joint the lag at the end of the ramp, the overshoot past the final q_des and
the residual oscillation, as RESPONSE_JSON.

  python scripts/reach_step_response.py --headless --arm left --device cpu
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="left")
ap.add_argument("--speed", type=float, default=1.0, help="rad/s")
ap.add_argument("--ramp", type=int, default=4, help="rows of ramp (10 Hz)")
ap.add_argument("--hold", type=int, default=10, help="rows after the ramp")
ap.add_argument("--sign", type=float, default=1.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from tower_rl.skills.reach_env import ReachEnv, ReachEnvCfg  # noqa: E402

n = 7
cfg = ReachEnvCfg()
cfg.setup(num_envs=n, device=args.device, overrides={"arm": args.arm, "reset": {"uniform_frac": 0.0, "posture_sigma": 0.0}})
env = ReachEnv(cfg)
env.reset()
dev = env.device
# every env at the same demo posture (index 1: the first prior), joint j moves in env j
q0 = env.postures[1].expand(n, 7).clone()
from tower_rl.sim.assets import q_full  # noqa: E402

ids = torch.arange(n, device=dev)
env.robot.write_joint_state_to_sim(q_full(q0, env.w_open), torch.zeros(n, 10, device=dev), env_ids=ids)
cmd = torch.max(torch.min(env.kin.sag_compensated_command(q0, env.payload), env.kin.limits[:, 1]), env.kin.limits[:, 0])
env.robot.set_joint_position_target(q_full(cmd, env.w_open))
env.q_des[:], env.q_cmd[:], env.q_cmd_prev[:], env.prev_q[:] = q0, cmd, cmd, q0
log = []
inner = env._apply_action


def logged():
    log.append(env.robot.data.joint_pos[:, :7].clone())
    inner()


env._apply_action = logged
zero = torch.zeros(n, 7, device=dev)
for _ in range(5):  # settle at the start pose
    env.step(zero)
log.clear()
a = torch.zeros(n, 7, device=dev)
eye = torch.eye(7, device=dev)
# the env's action map: speed v_max * (lin_frac a + (1 - lin_frac) a|a|); solve for the action giving --speed
b = env.lin_frac
u = args.speed / env.v_max
act = (-b + (b * b + 4 * (1 - b) * u) ** 0.5) / (2 * (1 - b))
for k in range(args.ramp + args.hold):
    env.step(eye * act * args.sign if k < args.ramp else zero)
traj = torch.stack(log)                                  # [substeps, n, 7]
q_final = env.q_des
ar = torch.arange(n, device=dev)
moved = traj[:, ar, ar]                                  # joint j in env j
target = q_final[ar, ar]
start = q0[ar, ar]
sub = cfg.decimation
ramp_end = args.ramp * sub
lag = (target - moved[ramp_end - 1]).abs()
over = ((moved[ramp_end:] - target) * args.sign).amax(0)  # past the final q_des, in the direction of motion
tail = moved[-3 * sub:]
resid = (tail.amax(0) - tail.amin(0))
out = {"arm": args.arm, "device": args.device, "speed": args.speed, "ramp_rows": args.ramp,
       "travel_rad": [round(v, 3) for v in (target - start).tolist()],
       "lag_at_stop_mrad": [round(v * 1000, 1) for v in lag.tolist()],
       "overshoot_mrad": [round(v * 1000, 1) for v in over.tolist()],
       "tail_pp_mrad": [round(v * 1000, 2) for v in resid.tolist()]}
print("RESPONSE_JSON", json.dumps(out), flush=True)
os._exit(0)  # app.close() hangs in Isaac Sim 5.1
