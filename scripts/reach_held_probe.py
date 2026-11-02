"""Held-block physics probe (P2d, corl_challenge/Reacher-V1.MD §11.1): a block in the fingers of the reach env
(``held_block``), first still, then moved straight up or sideways at a fixed pinch speed by the IK controller (no
policy), then held again; per block case (D1 masses and frictions), finger regime and speed. Reports the block's motion
in the pinch frame: slip per row while moving, total drift, drops.

The 10-31 station traces (Task 1-3 D1 heavy, the reacher's held transits) show the block sliding ~6 mm per row on a
0.14 m/s rise and ~2 mm in all on the planned transit (<= 0.07 m/s); this probe checks the light env reproduces that
before the reacher trains in it, on GPU and CPU PhysX.

  python scripts/reach_held_probe.py --headless --arm left [--device cpu] [--speeds 0 0.05 0.1 0.15] [--json out.json]
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="left")
ap.add_argument("--speeds", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1, 0.15, 0.25])
ap.add_argument("--dirs", nargs="+", default=["up", "side"])
ap.add_argument("--cases", nargs="+", default=None)
ap.add_argument("--regimes", nargs="+", type=int, default=[0, 1], help="0: Tasks 1-3 fingers (22 N, 40 mm), 1: Task 4/5 (3 N, 42.5 mm)")
ap.add_argument("--settle_rows", type=int, default=6)
ap.add_argument("--move_rows", type=int, default=10)
ap.add_argument("--hold_rows", type=int, default=15)
ap.add_argument("--ramp_rows", type=int, default=0, help="speed ramps up / down over this many rows (0: a velocity step)")
ap.add_argument("--quintic", action="store_true", help="the planned transit's time law instead (transit._smoothstep over "
                                                      "move_rows, peak speed = the speed): 1.875 x the mean speed")
ap.add_argument("--goals", default="results/reach/tasks/goals_*.jsonl", help="recorded transits: held starts")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from tower_rl.control import ArmController  # noqa: E402
from tower_rl.graph.interface import ARMS  # noqa: E402
from tower_rl.sim.assets import q_full  # noqa: E402
from tower_rl.skills.reach_env import ReachEnv, ReachEnvCfg  # noqa: E402
from tower_rl.skills.reach_tasks import load_goals  # noqa: E402

# mass scale, block static / dynamic friction, finger static / dynamic friction (station: wood 0.40 / 0.23, pads 2.4 / 2.0)
CASES = {"nominal": (1.0, 0.40, 0.23, 2.4, 2.0), "light": (0.5, 0.40, 0.23, 2.4, 2.0), "heavy14": (1.4, 0.40, 0.23, 2.4, 2.0),
         "heavy20": (2.0, 0.40, 0.23, 2.4, 2.0), "slip15": (1.0, 0.15, 0.07, 2.4, 2.0), "slip30": (1.0, 0.30, 0.14, 2.4, 2.0),
         "high": (1.0, 1.3, 0.8, 2.4, 2.0), "heavy18_slip20": (1.8, 0.20, 0.10, 2.4, 2.0),
         "fingers10": (1.0, 0.40, 0.23, 1.0, 0.8), "heavy20_fingers12": (2.0, 0.40, 0.23, 1.2, 1.0)}
cases = args.cases or list(CASES)
combos = [(c, v, d, rg) for c in cases for rg in args.regimes for d in args.dirs for v in args.speeds]
n = len(combos)
held_starts = [g["q"][ARMS.index(args.arm)] for f in sorted(glob.glob(args.goals))
               for g in load_goals([f], arm=args.arm) if g.get("held") and "q" in g]
assert held_starts, "no recorded held transits"
ov = {"arm": args.arm, "reset": {"uniform_frac": 0.0, "task_frac": 0.0}, "goal": {"tasks": {"frac": 0.0}},
      "held_block": {"frac": 1.0, "nominal_frac": 1.0, "regime_frac": 0.0, "start_tasks": 1.0, "settle_rows": args.settle_rows,
                     "offset": {"wide_frac": 0.0, "rot_deg": 0.0, "x_mm": [0.0, 0.0], "z_mm": [0.0, 0.0]},
                     "drop_m": 10.0, "drop_width_m": -1.0}}  # the probe records drops itself, the env never resets
cfg = ReachEnvCfg()
cfg.seed = args.seed
cfg.setup(num_envs=n, device=args.device, overrides=ov)
env = ReachEnv(cfg)
env.reset()
dev = env.device
ids = torch.arange(n, device=dev)
q0 = env._clamp_q(torch.tensor([held_starts[i % len(held_starts)] for i in range(n)], dtype=torch.float32, device=dev))
env.held[:] = True
w_init = env._spawn_held(ids, q0)
rg = env.hb["regimes"]
env.regime[:] = torch.tensor([float(c[3]) for c in combos], device=dev)
env.w_cmd[:, 0] = torch.where(env.regime > 0.5, rg["t45"]["closed_m"], rg["t13"]["closed_m"])
dyn = torch.tensor([CASES[c[0]] for c in combos], dtype=torch.float32, device=dev)
env.set_held_dynamics(ids, dyn[:, 0], dyn[:, 1:3], dyn[:, 3:5])
env.payload[:] = 0.1056
lim = env.kin.limits
q_cmd = torch.max(torch.min(env.kin.sag_compensated_command(q0, env.payload), lim[:, 1]), lim[:, 0])
env.robot.write_joint_state_to_sim(q_full(q0, w_init), torch.zeros(n, 10, device=dev), env_ids=ids)
env.robot.set_joint_position_target(q_full(q_cmd, env.w_cmd))
for buf, v in ((env.q_des, q0), (env.q_cmd, q_cmd), (env.q_cmd_prev, q_cmd), (env.prev_q, q0)):
    buf[:] = v
env.goal_dur.fill_(10 ** 6)
env.episode_length_buf.zero_()

ctrl = ArmController(args.arm, n, device=dev, kin=env.kin)
ctrl.reset(q0)
p0w, q0w = ctrl.pinch_world(q0)
speed = torch.tensor([c[1] for c in combos], device=dev)
dirv = torch.zeros(n, 3, device=dev)
up = torch.tensor([c[2] == "up" for c in combos], device=dev)
dirv[up, 2] = 1.0
side = ~up  # horizontal, along the arm's facing direction projected on the table (away from the base)
f = p0w[:, :2] - env.base.pos[:2]
dirv[side, :2] = (f / f.norm(dim=-1, keepdim=True).clamp_min(1e-9))[side]
row = {"k": 0}
dt = env.row_dt


def disp(k: int) -> torch.Tensor:
    """Pinch displacement (m) after ``k`` rows of the move (constant speed, optional linear ramps)."""
    t = min(max(k, 0), args.move_rows)
    if args.quintic:
        u = t / args.move_rows
        return speed * (args.move_rows * dt / 1.875) * (u ** 3 * (10 - 15 * u + 6 * u * u))
    if args.ramp_rows <= 0:
        return speed * t * dt
    rr = args.ramp_rows  # trapezoid in speed: ramp up over rr rows, cruise, ramp down over the last rr rows
    prof = [min(1.0, (i + 0.5) / rr, (args.move_rows - i - 0.5) / rr) for i in range(args.move_rows)]
    return speed * sum(prof[:t]) * dt


def pre(actions):
    k = row["k"] - args.settle_rows
    ctrl.set_target(p0w + dirv * disp(k + 1)[:, None], q0w)
    env.q_cmd_prev[:] = env.q_cmd
    env.q_cmd[:] = ctrl.command(env.robot.data.joint_pos[:, :7], env.payload)
    env.q_des[:] = ctrl.q_des
    env.substep = 0
    row["k"] += 1


env._pre_physics_step = pre
offs, widths, pz = [], [], []
zeros = torch.zeros(n, 7, device=dev)
for k in range(args.settle_rows + args.move_rows + args.hold_rows):
    env.step(zeros)
    q = env.robot.data.joint_pos[:, :7]
    pw, qw = env.base.to_world(*env.kin.fk(q, "pinch"))
    off, _ = env.block_in_pinch(pw, qw)
    offs.append(off.clone())
    widths.append(env.width().clone())
    pz.append(pw.clone())
O = torch.stack(offs, 1) * 1000.0  # [n, rows, 3] mm, pinch frame (z: the approach, out of the fingers)
W = torch.stack(widths, 1)
P = torch.stack(pz, 1)
s0 = args.settle_rows - 1
ref = O[:, s0]
mv = slice(args.settle_rows, args.settle_rows + args.move_rows)
step = (O[:, 1:] - O[:, :-1]).norm(dim=-1)  # row k+1 vs k
slip_move = step[:, s0:s0 + args.move_rows]
drift = (O - ref[:, None]).norm(dim=-1)
dropped = ((W < 0.044) | (drift > 25.0)).any(1)
settle_drift = (O[:, s0] - O[:, 0]).norm(dim=-1)
vz = (P[:, mv] - P[:, s0:s0 + args.move_rows]).norm(dim=-1).mean(1) / dt  # achieved mean pinch speed while moving
recs = []
for i, (c, v, d, r_) in enumerate(combos):
    recs.append({"case": c, "regime": r_, "dir": d, "v": v, "v_meas": round(float(vz[i]), 3),
                 "settle_mm": round(float(settle_drift[i]), 2), "slip_mean_mm": round(float(slip_move[i].mean()), 2),
                 "slip_max_mm": round(float(slip_move[i].max()), 2), "drift_end_mm": round(float(drift[i, -1]), 2),
                 "dz_end_mm": round(float(O[i, -1, 2] - ref[i, 2]), 2), "dropped": bool(dropped[i]),
                 "width_end_mm": round(float(W[i, -1]) * 1000, 1)})
print(f"{'case':18s} rg dir    " + " ".join(f"v={v:<5g}" for v in args.speeds) + "   (drift at the end, mm; D = dropped)")
for c in cases:
    for r_ in args.regimes:
        for d in args.dirs:
            cells = []
            for v in args.speeds:
                x = next(x for x in recs if x["case"] == c and x["regime"] == r_ and x["dir"] == d and x["v"] == v)
                cells.append(("D" if x["dropped"] else f"{x['drift_end_mm']:.1f}").rjust(7))
            print(f"{c:18s} {r_}  {d:5s} " + " ".join(cells))
if args.json:
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"args": vars(args), "recs": recs}, indent=0))
print("PROBE_JSON " + json.dumps({"device": args.device, "arm": args.arm, "n": n,
                                  "dropped": int(dropped.sum()), "settle_max_mm": round(float(settle_drift.max()), 2)}), flush=True)
os._exit(0)  # app.close() hangs in Isaac Sim 5.1
