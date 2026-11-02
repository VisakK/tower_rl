"""Re-initialise one joint's action head in a reacher checkpoint (rsl_rl ActorCritic): zero its row of the actor's
output layer (mean 0: the joint holds still) and set its action noise, so that joint is relearned under new rewards
while the others keep what they learned (10-30: joint 1's entrenched +-1 dither survived four resumed runs).

  python scripts/reach_reset_joint.py <in.pt> <out.pt> --joint 0 --std 0.3
Resume from <out.pt> with --resume_no_optimizer (the Adam moments of the edited weights are stale).
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--joint", type=int, action="append", required=True)
ap.add_argument("--std", type=float, default=0.3)
args = ap.parse_args()
c = torch.load(args.src, map_location="cpu", weights_only=False)
sd = c["model_state_dict"]
out = max((k for k in sd if k.startswith("actor.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
for j in args.joint:
    sd[out][j] = 0.0
    sd[out.replace("weight", "bias")][j] = 0.0
    sd["std"][j] = args.std
print(f"{args.src} -> {args.dst}: joints {args.joint} zeroed in {out}, std now {[round(v, 3) for v in sd['std'].tolist()]}")
torch.save(c, args.dst)
