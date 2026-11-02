"""Widen a 65-D push checkpoint to the 67-D observation of student plan Step 9a (seen, since_seen appended).

The actor's first layer gets zero columns for the two new inputs (the policy starts out ignoring them); the critic's
input embeds the actor observation first, so its two new columns go in after index 65; the empirical normalizers get
mean 0, variance 1 for them. The optimizer state is dropped (resume with ``train_skill.py --resume_no_optimizer``).

Usage: python scripts/widen_push_checkpoint.py <model.pt> <out.pt> [old new [new_critic]]
  old / new: any checkpoint whose actor observation (the critic's first inputs) grows from old by new appended inputs,
  e.g. 55 9 for the reacher's obs.fine_m (corl_challenge/Reach-PPO-plan.MD). new_critic (default new): the critic's
  new columns after index old, when it gains critic-only inputs there too (the reacher's held_block.obs: 65 2 22).
"""
import sys

import torch

OLD, NEW = 65, 2
NEW_CRITIC = None


def widen(sd: dict) -> dict:
    sd = dict(sd)
    w = sd["actor.0.weight"]
    assert w.shape[1] == OLD, w.shape
    sd["actor.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], NEW, dtype=w.dtype)], 1)
    nc = NEW if NEW_CRITIC is None else NEW_CRITIC
    wc = sd["critic.0.weight"]
    sd["critic.0.weight"] = torch.cat([wc[:, :OLD], torch.zeros(wc.shape[0], nc, dtype=wc.dtype), wc[:, OLD:]], 1)
    for net, at, k_new in (("actor_obs_normalizer", None, NEW), ("critic_obs_normalizer", OLD, nc)):
        for k, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            key = f"{net}.{k}"
            if key not in sd:
                continue
            x = sd[key]
            add = torch.full((x.shape[0], k_new), fill, dtype=x.dtype)
            sd[key] = torch.cat([x, add], 1) if at is None else torch.cat([x[:, :at], add, x[:, at:]], 1)
    return sd


def main():
    global OLD, NEW, NEW_CRITIC
    src, dst = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        OLD, NEW = int(sys.argv[3]), int(sys.argv[4])
    if len(sys.argv) > 5:
        NEW_CRITIC = int(sys.argv[5])
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["model_state_dict"] = widen(ck["model_state_dict"])
    ck.pop("optimizer_state_dict", None)
    ck["widened_from"] = src
    torch.save(ck, dst)
    sd = ck["model_state_dict"]
    print({k: tuple(v.shape) for k, v in sd.items() if k in ("actor.0.weight", "critic.0.weight", "actor_obs_normalizer._mean",
                                                            "critic_obs_normalizer._mean")})


if __name__ == "__main__":
    main()
