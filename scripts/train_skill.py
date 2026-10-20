"""Train a contact skill with PPO (rsl_rl) in the light skill scene (Steps 5/6).

Usage:
  python scripts/train_skill.py --headless --skill grasp_lift --arm left --num_envs 4096 --max_iterations 1500
  python scripts/train_skill.py --headless --skill grasp_lift --arm left --num_envs 64 --max_iterations 3 --smoke
Logs and checkpoints: results/skills/<skill>_<arm>/<date>_<config hash>[_tag]/ (config.json, model_*.pt, TensorBoard).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--skill", choices=["grasp_lift", "place_release"], default="grasp_lift")
parser.add_argument("--arm", choices=["left", "back"], default="left")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag", default="")
parser.add_argument("--override", default="{}", help="JSON dict merged into the skill YAML")
parser.add_argument("--resume", default=None, help="checkpoint .pt to continue from")
parser.add_argument("--smoke", action="store_true", help="tiny run: check dims, NaNs and timing")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from tower_rl.config import config_hash  # noqa: E402
from tower_rl.paths import RESULTS_DIR  # noqa: E402
from tower_rl.skills.ppo_cfg import runner_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
overrides = json.loads(args.override)
overrides["arm"] = args.arm
if args.skill == "grasp_lift":
    from tower_rl.skills.grasp_lift_env import GraspLiftEnv as Env, GraspLiftEnvCfg as EnvCfg
else:
    from tower_rl.skills.place_release_env import PlaceReleaseEnv as Env, PlaceReleaseEnvCfg as EnvCfg

cfg = EnvCfg()
cfg.seed = args.seed
cfg.setup(num_envs=args.num_envs, device=args.device, overrides=overrides)
skill = cfg.skill
experiment = f"{args.skill}_{args.arm}"
rcfg = runner_cfg(skill, experiment, args.max_iterations, seed=args.seed, device=args.device)
h = config_hash({"skill": skill, "num_envs": cfg.scene.num_envs, "seed": args.seed})
run_dir = RESULTS_DIR / "skills" / experiment / (time.strftime("%Y-%m-%d_%H%M") + f"_{h}" + (f"_{args.tag}" if args.tag else ""))
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(json.dumps({"skill": skill, "num_envs": cfg.scene.num_envs, "seed": args.seed,
                                                 "runner": rcfg.to_dict(), "resume": args.resume}, indent=1, default=str))
env = Env(cfg)
wrapped = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(wrapped, rcfg.to_dict(), log_dir=str(run_dir), device=args.device)
if args.resume:
    runner.load(args.resume)
print(f"[train] {experiment}: {cfg.scene.num_envs} envs, run dir {run_dir}", flush=True)
t0 = time.time()
runner.learn(num_learning_iterations=rcfg.max_iterations, init_at_random_ep_len=True)
print(f"[train] done in {time.time() - t0:.0f} s", flush=True)
if args.smoke:
    obs = wrapped.get_observations()
    print("REPORT_JSON " + json.dumps({"policy_obs": list(obs["policy"].shape), "critic_obs": list(obs["critic"].shape),
                                       "finite": bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()),
                                       "wall_s": round(time.time() - t0, 1)}), flush=True)
runner.save(str(run_dir / "model_final.pt"))
os._exit(0)
