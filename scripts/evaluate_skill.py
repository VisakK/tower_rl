"""Step 5/6 acceptance: evaluate a trained skill (deterministic actor, noisy actor observations) per tier.

Runs ``--episodes`` episodes per tier in the light skill scene and reports success rate (Wilson 95 %), failure causes
(disturbed / dropped / failed / timeout), disturbance rate and episode length; D1 is broken down by robustness case.

Usage: python scripts/evaluate_skill.py --headless --skill grasp_lift --arm left --checkpoint <run dir or .pt> \
           --tier D1 --episodes 500
One tier per process (Isaac cannot rebuild the stage in-process).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--skill", choices=["grasp_lift", "place_release"], default="grasp_lift")
parser.add_argument("--arm", choices=["left", "back"], default="left")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tier", default="D0")
parser.add_argument("--episodes", type=int, default=500, help="= num_envs: one episode per env")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--override", default="{}")
parser.add_argument("--out", default=None)
parser.add_argument("--perception_noise", action="store_true",
                    help="Step 8.1: the provisional perception model on the actor's block poses (per-episode bias, jitter, "
                         "dropouts, outliers; tower_rl.eval.perception_noise.NoiseCfg) instead of the training noise")
parser.add_argument("--noise", default="{}", help="JSON NoiseCfg overrides for --perception_noise")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from tower_rl.config import result_path  # noqa: E402
from tower_rl.eval.metrics import wilson  # noqa: E402
from tower_rl.eval.perception_noise import NoiseCfg  # noqa: E402
from tower_rl.skills.export import export  # noqa: E402

if args.skill == "grasp_lift":
    from tower_rl.skills.grasp_lift_env import GraspLiftEnv as Env, GraspLiftEnvCfg as EnvCfg
else:
    from tower_rl.skills.place_release_env import PlaceReleaseEnv as Env, PlaceReleaseEnvCfg as EnvCfg

tmp = Path("/tmp") / f"tower_eval_{args.skill}_{args.arm}_{os.getpid()}.pt"
actor = torch.jit.load(str(export(args.checkpoint, tmp)), map_location=args.device).eval()
results = {}
tier = args.tier
ov = json.loads(args.override)
ov.update({"arm": args.arm, "tiers": {tier: 1.0}})
nc = NoiseCfg(**json.loads(args.noise)) if args.perception_noise else None
if nc is not None:  # per-episode bias and per-step jitter through the env's own noise terms (the shared bias: place only)
    ov["obs_noise"] = {"episode_pos_m": nc.bias_pos_m, "episode_yaw_deg": nc.bias_yaw_deg, "step_pos_m": nc.step_pos_m,
                       "step_yaw_deg": nc.step_yaw_deg, "shared_pos_m": nc.shared_pos_m, "shared_yaw_deg": nc.shared_yaw_deg}
cfg = EnvCfg()
cfg.seed = args.seed
cfg.setup(num_envs=args.episodes, device=args.device, overrides=ov)
cfg.skill["tiers"] = {tier: 1.0}  # replace the training mix (deep_update would merge it)
env = Env(cfg)
if nc is not None:  # dropouts (a block's pose freezes for dropout_s) and outliers (a 1-step jump), one draw per env step
    noisy0 = env.noisy_block_poses
    ns = {"step": None, "p": None, "q": None, "frozen": None}
    drop_steps = max(1, round(nc.dropout_s / env.step_dt))

    def noisy_with_dropouts():
        step = int(env.common_step_counter)
        if ns["step"] == step:  # the observation and the next action's step scaling see the same estimate
            return ns["p"], ns["q"]
        p, q = noisy0()
        out = torch.rand(p.shape[:2], device=p.device) < nc.outlier_p
        d = torch.randn_like(p)
        p = p + out[..., None] * d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9) * nc.outlier_m
        if ns["frozen"] is None:
            ns["frozen"] = torch.zeros(p.shape[:2], dtype=torch.long, device=p.device)
        else:
            start = (torch.rand(p.shape[:2], device=p.device) < nc.dropout_p) & (ns["frozen"] == 0)
            ns["frozen"] = torch.where(start, drop_steps, ns["frozen"])
            fr = ns["frozen"] > 0
            p = torch.where(fr[..., None], ns["p"], p)
            q = torch.where(fr[..., None], ns["q"], q)
            ns["frozen"] = (ns["frozen"] - 1).clamp(min=0)
        ns.update(step=step, p=p, q=q)
        return p, q

    env.noisy_block_poses = noisy_with_dropouts
obs, _ = env.reset()
n = env.num_envs
cases = list(env.env_case)
outcome = [None] * n
lengths = [0] * n
quality = {}  # grasp_lift: per-episode held-grasp quality (|dx| mm, |dz| mm, rotation deg) when the hold was reached
term_st = {}
get_dones0 = env._get_dones


def get_dones_hook():  # terminal state, before DirectRLEnv auto-resets the finished envs
    out = get_dones0()
    term_st.clear()
    term_st.update({k: env.st[k].clone() for k in ("q_dx", "q_dz", "q_rot") if k in env.st})
    return out


env._get_dones = get_dones_hook
done_once = torch.zeros(n, dtype=torch.bool, device=env.device)
with torch.no_grad():
    for step in range(env.max_episode_length + 5):
        obs, rew, term, trunc, extras = env.step(actor(obs["policy"]))
        d = (term | trunc) & ~done_once
        for e in d.nonzero(as_tuple=True)[0].tolist():
            cause = "timeout"
            for k in ("success", "bad_grasp", "disturbed", "dropped", "failed"):
                if k in env.term and bool(env.term[k][e]):
                    cause = k
                    break
            outcome[e] = cause
            lengths[e] = step + 1
            if cause in ("success", "bad_grasp") and "q_dx" in term_st:
                quality[e] = (abs(float(term_st["q_dx"][e])) * 1000, abs(float(term_st["q_dz"][e])) * 1000,
                              float(torch.rad2deg(term_st["q_rot"][e])))
        done_once |= d
        if bool(done_once.all()):
            break


def summarize(idx):
    m = len(idx)
    counts = {}
    for i in idx:
        counts[outcome[i]] = counts.get(outcome[i], 0) + 1
    k = counts.get("success", 0)
    lo, hi = wilson(k, m)
    out = {"episodes": m, "success_rate": round(k / max(m, 1), 4), "wilson95": [round(lo, 4), round(hi, 4)], "causes": counts,
           "disturbance_rate": round(counts.get("disturbed", 0) / max(m, 1), 4),
           "mean_steps": round(sum(lengths[i] for i in idx) / max(m, 1), 1)}
    qs = [quality[i] for i in idx if i in quality]
    if qs:  # held grasps: median / 90th percentile of |dx| (along the block), |dz| (mm) and the rotation (deg)
        a = torch.tensor(qs)
        out["grasp_quality_p50_p90"] = {name: [round(float(torch.quantile(a[:, j], 0.5)), 1), round(float(torch.quantile(a[:, j], 0.9)), 1)]
                                         for j, name in enumerate(("dx_mm", "dz_mm", "rot_deg"))}
    return out


results[tier] = summarize(list(range(n)))
by_case = {}
for i, c in enumerate(cases):
    by_case.setdefault(c, []).append(i)
if len(by_case) > 1:
    results[tier]["by_case"] = {c: summarize(idx) for c, idx in sorted(by_case.items())}
print(f"[eval_skill] {args.skill} {args.arm} {tier}: " + json.dumps(results[tier]), flush=True)
out = Path(args.out) if args.out else result_path(f"skills_eval/{args.skill}_{args.arm}", {"ckpt": args.checkpoint, "tier": tier}, tag=tier)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"checkpoint": args.checkpoint, "results": results}, indent=1))
print("REPORT_JSON " + json.dumps(results), flush=True)
os._exit(0)
