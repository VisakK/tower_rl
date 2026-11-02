"""Check the reaching env (Reach-PPO-plan.MD) or a trained reacher: goals settled, end-of-goal errors, time to settle.

  --policy zero   : no motion (the arm must hold its pose: sag compensation)
  --policy ik     : oracle, joint velocities toward the IK solution of each goal (the env's mechanics and goal
                    reachability; an upper bound for the learned policy at the same speed limit)
  --policy <run>  : a trained run directory (latest model_*.pt) or checkpoint

  python scripts/reach_check.py --headless --arm left --policy ik --num_envs 256 --steps 600 [--level 3]
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
ap.add_argument("--policy", default="ik")
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--level", type=int, default=None, help="curriculum level to test (default: the config's start)")
ap.add_argument("--override", default="{}")
ap.add_argument("--trace", type=int, default=0, help="with --dump: also per-row q, action, errors of the first N envs")
ap.add_argument("--dump", default=None, help="per-goal records (start / end errors, end motion, limit distance) to this JSON")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.skills.reach_env import ReachEnv, ReachEnvCfg  # noqa: E402

ov = json.loads(args.override)
ov["arm"] = args.arm
if args.level is not None:
    ov.setdefault("curriculum", {})["start_level"] = args.level
    ov["curriculum"]["window"] = 10 ** 9  # stay at this level
cfg = ReachEnvCfg()
cfg.setup(num_envs=args.num_envs, device=args.device, overrides=ov)
env = ReachEnv(cfg)
obs, _ = env.reset()
policy = None
if args.policy not in ("zero", "ik"):
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from tower_rl.skills.ppo_cfg import runner_cfg
    p = Path(args.policy)
    ckpt = p if p.is_file() else max(p.glob("model_*.pt"), key=lambda f: f.stat().st_mtime)
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg(cfg.skill, "reach_check").to_dict(), log_dir=None, device=args.device)
    runner.load(str(ckpt), load_optimizer=False)  # a check needs the weights only (widened checkpoints have none)
    policy = runner.get_inference_policy(device=args.device)
    obs = wrapped.get_observations()
ends, z_min, hold, recs = [], 1.0, [], []
trace = []
start = None  # per env: the goal's error when it was drawn (pos m, rot rad)
for t in range(args.steps):
    if args.policy == "zero":
        a = torch.zeros(env.num_envs, 7, device=env.device)
    elif args.policy == "ik":
        q_goal, _, _ = env.kin.ik(env.q_des, env.goal_p, env.goal_q, iters=30, q_ref=env.q_des,
                                  limit_margin=cfg.skill["action"]["limit_margin"])
        a = ((q_goal - env.q_des) / (env.v_max * env.row_dt)).clamp(-1, 1)
        if env.amap == "square":
            a = a.sign() * a.abs().sqrt()
        elif env.amap == "blend":  # invert b a + (1 - b) a^2 = |u| for a >= 0
            b = env.lin_frac
            a = a.sign() * (-b + (b * b + 4 * (1 - b) * a.abs()).sqrt()) / (2 * (1 - b))
    else:
        with torch.inference_mode():
            a = policy(obs)
    goal_t_before = env.goal_t.clone()
    if policy is not None:
        obs, _, _, _ = wrapped.step(a)
    else:
        obs, _, _, _, _ = env.step(a)
    st = env._st
    if args.trace:
        k = args.trace
        trace.append({"q": st["q"][:k].tolist(), "a": a[:k].tolist(), "pos_mm": (st["pos_err"][:k] * 1000).tolist(),
                      "rot_deg": torch.rad2deg(st["rot_err"][:k]).tolist(), "goal_t": env.goal_t[:k].tolist(),
                      "q_des": env.q_des[:k].tolist(), "q_cmd": env.q_cmd[:k].tolist()})
    done = st["settled"] | st["goal_timeout"]
    for i in done.nonzero(as_tuple=True)[0].tolist():
        fs = int(st["first_settle"][i])  # time to settle: when it first settled (the goal may run on, goal.switch timer)
        ends.append((bool(st["settled"][i]), float(st["pos_err"][i]), float(st["rot_err"][i]),
                     float(fs if fs >= 0 else goal_t_before[i] + 1) * env.row_dt, fs >= 0))
        if args.dump and start is not None:
            near = torch.minimum(env.q_des[i] - env.lo, env.hi - env.q_des[i])
            recs.append({"settled": ends[-1][0], "pos_mm": ends[-1][1] * 1000, "rot_deg": math.degrees(ends[-1][2]),
                         "t_s": ends[-1][3], "start_pos_mm": start[0][i].item() * 1000,
                         "start_rot_deg": math.degrees(start[1][i].item()),
                         "qd_max": float(st["qd"][i].abs().max()), "limit_min_rad": float(near.min()),
                         "limit_joint": int(near.argmin()), "q": [round(v, 4) for v in st["q"][i].tolist()],
                         "z_world": float(env.base.to_world(st["pb"][i:i + 1], st["qb"][i:i + 1])[0][0, 2]),
                         "p0_w": [round(v, 4) for v in start[2][i].tolist()],  # start / goal pinch, station frame
                         "p1_w": [round(v, 4) for v in start[3][i].tolist()],
                         "path_z_min": round(float(pz_min[i]), 4)})  # lowest pinch (station z) during the goal
    if args.dump:  # the new goals' starting errors (goals drawn this row, or all at the first row)
        new = (env.goal_t == 0) if start is not None else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        pb, qb = env.kin.fk(env.robot.data.joint_pos[:, :7], "pinch")  # after any episode reset of this row
        pe = (env.goal_p - pb).norm(dim=-1)
        re = geo.quat_angle(qb, env.goal_q)
        cur = (pe, re, env.base.to_world(pb, qb)[0], env.base.to_world(env.goal_p, env.goal_q)[0])
        start = cur if start is None else tuple(torch.where(new if c.dim() == 1 else new[:, None], c, o)
                                                for c, o in zip(cur, start))
    pw, _ = env.base.to_world(st["pb"], st["qb"])
    z_min = min(z_min, float(pw[:, 2].min()))
    pz_min = torch.where(env.goal_t <= 1, pw[:, 2], torch.minimum(pz_min, pw[:, 2])) if t else pw[:, 2].clone()
    if args.policy == "zero":
        hold.append(float((st["q"] - env.q_des).abs().max()))
n = len(ends)
s = [e for e in ends if e[0]]
srt = lambda v: sorted(v)  # noqa: E731
pct = lambda v, p: srt(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")  # noqa: E731
pos = [e[1] for e in ends]
rot = [e[2] for e in ends]
rep = {"policy": args.policy, "arm": args.arm, "level": env.level, "goals": n,
       "settled_frac": round(len(s) / max(n, 1), 4), "ever_settled_frac": round(sum(e[4] for e in ends) / max(n, 1), 4),
       "end_pos_mm_p50": round(pct(pos, 0.5) * 1000, 2), "end_pos_mm_p95": round(pct(pos, 0.95) * 1000, 2),
       "end_rot_deg_p50": round(math.degrees(pct(rot, 0.5)), 2), "end_rot_deg_p95": round(math.degrees(pct(rot, 0.95)), 2),
       "within_1cm_frac": round(sum(1 for e in ends if e[1] <= 0.01) / max(n, 1), 4),
       "settle_time_s_p50": round(pct([e[3] for e in ends if e[4]], 0.5), 2), "pinch_z_min_m": round(z_min, 4)}
if hold:
    rep["hold_max_joint_err_rad"] = round(max(hold), 4)
print("REPORT_JSON " + json.dumps(rep), flush=True)
if args.dump:
    Path(args.dump).write_text(json.dumps(recs if not trace else {"goals": recs, "trace": trace}))
os._exit(0)  # app.close() hangs in Isaac Sim 5.1 (as train_skill.py)
