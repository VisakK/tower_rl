"""Sanity check of a skill env with a scripted privileged oracle through the same action interface.

grasp_lift oracle: move the controller target to the true canonical grasp pose (via the pre-grasp), close when the
pinch is within 3 mm, wait, then lift. If this cannot reach the success condition, the env (physics, reward or
termination logic) is wrong, not the learner. Reports success / failure causes and timing.

Usage: python scripts/oracle_skill.py --headless --skill grasp_lift --arm left --num_envs 256 --episodes 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--skill", default="grasp_lift")
parser.add_argument("--arm", default="left")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--override", default="{}")
parser.add_argument("--release_z", type=float, default=0.005, help="place oracle: held block height above its goal at the release")
parser.add_argument("--hold_steps", type=int, default=0, help="place oracle: steps at the release pose before opening")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402

overrides = json.loads(args.override)
overrides["arm"] = args.arm
if args.skill == "grasp_lift":
    from tower_rl.skills.grasp_lift_env import GraspLiftEnv as Env, GraspLiftEnvCfg as EnvCfg
else:
    from tower_rl.skills.place_release_env import PlaceReleaseEnv as Env, PlaceReleaseEnvCfg as EnvCfg
cfg = EnvCfg()
cfg.setup(num_envs=args.num_envs, device=args.device, overrides=overrides)
env = Env(cfg)
env.reset()
n, dev = env.num_envs, env.device
ac = env.sk["action"]
phase = torch.zeros(n, dtype=torch.long, device=dev)  # 0 approach above, 1 descend, 2 close, 3 lift
timer = torch.zeros(n, dtype=torch.long, device=dev)
totals = {"success": 0, "disturbed": 0, "dropped": 0, "timeout": 0}
finished = torch.zeros(n, dtype=torch.bool, device=dev)
drops = []
if args.skill == "place_release":
    totals = {"success": 0, "failed": 0, "disturbed": 0, "timeout": 0}
    fails = []
    z_rel = torch.zeros(n, device=dev)
    vmax = torch.zeros(n, device=dev)
    term_state = {}
    get_dones0 = env._get_dones

    def get_dones_hook():  # terminal values, before DirectRLEnv auto-resets the finished envs
        out = get_dones0()
        term_state.update({k: v.clone() for k, v in env.st.items()})
        term_state.update(released=env.released.clone(), since=env.since_release.clone(), z_release=env.z_release.clone(),
                          case=list(env.env_case))
        v = env.blocks.data.object_link_lin_vel_w[torch.arange(n, device=dev), env.slot_idx[env.target_slot]].norm(dim=-1)
        vmax.copy_(torch.where(env.released & ~finished, torch.maximum(vmax, v), vmax))
        return out

    env._get_dones = get_dones_hook
    for step in range(args.steps):
        s = env._compute_state()
        # pinch target that puts the (true) held block 5 mm above its goal: P = G * T_pb^-1
        t_pb = geo.frame_relative(s["pw"], s["qw"], s["tpos"], s["tquat"])
        g2 = geo.quat_mul(env.goal_quat, env.flip.expand(n, 4))
        use_flip = geo.quat_angle(s["tquat"], g2) < geo.quat_angle(s["tquat"], env.goal_quat)
        gq = torch.where(use_flip[:, None], g2, env.goal_quat)
        gpos = env.goal_pos.clone()
        gpos[:, 2] += args.release_z
        ip, iq = geo.frame_inverse(*t_pb)
        tp, tq = geo.frame_compose(gpos, gq, ip, iq)
        # phases: 0 align above (2 cm), 1 descend, 2 open + wait, 3 retreat
        above_p = tp.clone()
        above_p[:, 2] += 0.02
        tgt = torch.where((phase == 0)[:, None], above_p, tp)
        up = env.ctrl.pos.clone()
        up[:, 2] = env.z_release + 0.05
        tgt = torch.where((phase == 3)[:, None], up, tgt)
        dpos_b = env.base.vec_to_base(tgt - env.ctrl.pos)
        a_pos = (dpos_b / env._action_scale()[:, None]).clamp(-1, 1)
        a_rot = (env.base.vec_to_base(geo.quat_error_rotvec(env.ctrl.quat, tq)) / ac["rot_rad"]).clamp(-1, 1)
        a_rot = torch.where((phase >= 2)[:, None], torch.zeros_like(a_rot), a_rot)
        raw = torch.cat([a_pos, a_rot], -1)
        a6 = ((raw - (1 - ac["ema"]) * env.ema_actions) / ac["ema"]).clamp(-1, 1)
        perr = (s["pw"] - tp).norm(dim=-1)
        phase = torch.where((phase == 0) & ((s["pw"] - above_p).norm(dim=-1) < 0.004), 1, phase)
        phase = torch.where((phase == 1) & (perr < 0.002), 2, phase)
        timer = torch.where(phase >= 2, timer + 1, timer)  # phase 2 holds closed for --hold_steps, then opens
        phase = torch.where((phase == 2) & (timer > args.hold_steps), 4, phase)
        phase = torch.where((phase == 4) & (timer > args.hold_steps + 6), 3, phase)
        grip = torch.where((phase == 4) | (phase == 3), -1.0, 1.0)[:, None]
        z_rel = torch.where((phase == 4) & (timer == args.hold_steps + 1), s["tpos"][:, 2] - env.goal_pos[:, 2], z_rel)
        _, _, term, trunc, _ = env.step(torch.cat([a6, grip], -1))
        done = term | trunc
        s2 = term_state
        for k, m in (("success", env.term["success"]), ("failed", env.term["failed"]), ("disturbed", env.term["disturbed"])):
            totals[k] += int((m & done & ~finished).sum())
        totals["timeout"] += int((trunc & ~term & ~finished).sum())
        for e in ((env.term["failed"] | env.term["disturbed"]) & done & ~finished).nonzero(as_tuple=True)[0].tolist():
            fails.append({"dxy_mm": round(float(s2["dxy"][e]) * 1000, 1), "dz_mm": round(float(s2["dz"][e]) * 1000, 1),
                          "dyaw_deg": round(float(s2["dyaw"][e]) * 57.3, 1), "tilt_deg": round(float(s2["tilt"][e]) * 57.3, 1),
                          "tspeed": round(float(s2["tspeed"][e]), 3), "tvel": round(float(s2["tvel"][e]), 3),
                          "disp_mm": round(float(s2["disp_pl"][e].max()) * 1000, 1),
                          "released": bool(s2["released"][e]), "since": int(s2["since"][e]),
                          "rise_mm": round(float(s2["pw"][e, 2] - s2["z_release"][e]) * 1000, 1), "release_dz_mm": round(float(z_rel[e]) * 1000, 1),
                          "vmax": round(float(vmax[e]), 3), "case": s2["case"][e]})
        finished |= done
        phase = torch.where(done, 0, phase)
        timer = torch.where(done, 0, timer)
        if bool(finished.all()):
            break
    totals["unfinished"] = int((~finished).sum())
    totals["steps"] = step + 1
    print("REPORT_JSON " + json.dumps(totals), flush=True)
    for f in fails:
        print("FAIL " + json.dumps(f), flush=True)
    os._exit(0)
for step in range(args.steps):
    s = env._compute_state()
    gp, gq = s["gp"], s["gq"]
    tgt = gp.clone()
    above = gp.clone()
    above[:, 2] += 0.03
    tgt = torch.where((phase == 0)[:, None], above, tgt)
    lift = gp.clone()
    lift[:, 2] = env.z0 + 0.12
    tgt = torch.where((phase == 3)[:, None], lift, tgt)
    if args.skill != "grasp_lift":
        raise SystemExit("oracle implemented for grasp_lift only")
    # desired delta of the controller target, in the base frame, normalized to the action scale
    dpos_w = tgt - env.ctrl.pos
    dpos_b = env.base.vec_to_base(dpos_w)
    scale = env._action_scale()[:, None]
    a_pos = (dpos_b / scale).clamp(-1, 1)
    drot_w = geo.quat_error_rotvec(env.ctrl.quat, gq)
    a_rot = (env.base.vec_to_base(drot_w) / ac["rot_rad"]).clamp(-1, 1)
    # undo the EMA so the executed step equals the desired one
    ema = ac["ema"]
    raw = torch.cat([a_pos, a_rot], -1)
    a6 = ((raw - (1 - ema) * env.ema_actions) / ema).clamp(-1, 1)
    perr = (s["pw"] - gp).norm(dim=-1)
    phase = torch.where((phase == 0) & ((s["pw"] - above).norm(dim=-1) < 0.006), 1, phase)
    phase = torch.where((phase == 1) & (perr < 0.003), 2, phase)
    timer = torch.where(phase == 2, timer + 1, timer)
    phase = torch.where((phase == 2) & (timer > 12), 3, phase)
    grip = torch.where(phase >= 2, 1.0, -1.0)[:, None]
    a = torch.cat([a6, grip], -1)
    pre = {k: v.clone() for k, v in env._compute_state().items() if k in ("lift", "grasped", "tilt", "w", "f", "both")}
    _, _, term, trunc, _ = env.step(a)
    done = term | trunc
    for k, m in (("success", env.term["success"]), ("disturbed", env.term["disturbed"]), ("dropped", env.term["dropped"])):
        totals[k] += int((m & done & ~finished).sum())
    totals["timeout"] += int((trunc & ~term & ~finished).sum())
    newly_dropped = env.term["dropped"] & done & ~finished
    for e in newly_dropped.nonzero(as_tuple=True)[0].tolist()[:12]:
        drops.append({"step": step, "phase": int(phase[e]), "lift_mm": round(float(pre["lift"][e]) * 1000, 1),
                      "grasped": bool(pre["grasped"][e]), "both": bool(pre["both"][e]), "tilt_deg": round(float(pre["tilt"][e]) * 57.3, 1),
                      "width_mm": round(float(pre["w"][e]) * 1000, 1), "forces": [round(x, 2) for x in pre["f"][e].tolist()],
                      "mass": round(float(env.mass_scale[e, env.target_slot[e]]), 2), "block_fric": [round(x, 2) for x in env.block_fric[e].tolist()],
                      "finger_fric": [round(x, 2) for x in env.finger_fric[e].tolist()], "variant": int(env.env_variant[e]), "stage": int(env.env_stage[e])})
    finished |= done
    phase = torch.where(done, 0, phase)
    timer = torch.where(done, 0, timer)
    if bool(finished.all()):
        break
totals["unfinished"] = int((~finished).sum())
totals["steps"] = step + 1
print("REPORT_JSON " + json.dumps(totals), flush=True)
for d in drops:
    print("DROP " + json.dumps(d), flush=True)
os._exit(0)
