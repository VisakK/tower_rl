"""Widen a 65-D push checkpoint to the 67-D observation of student plan Step 9a (seen, since_seen appended).

The actor's first layer gets zero columns for the two new inputs (the policy starts out ignoring them); the critic's
input embeds the actor observation first, so its two new columns go in after index 65; the empirical normalizers get
mean 0, variance 1 for them. The optimizer state is dropped (resume with ``train_skill.py --resume_no_optimizer``).

Usage: python scripts/widen_push_checkpoint.py <model.pt> <out.pt>
"""
import sys

import torch

OLD, NEW = 65, 2


def widen(sd: dict) -> dict:
    sd = dict(sd)
    w = sd["actor.0.weight"]
    assert w.shape[1] == OLD, w.shape
    sd["actor.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], NEW, dtype=w.dtype)], 1)
    wc = sd["critic.0.weight"]
    sd["critic.0.weight"] = torch.cat([wc[:, :OLD], torch.zeros(wc.shape[0], NEW, dtype=wc.dtype), wc[:, OLD:]], 1)
    for net, at in (("actor_obs_normalizer", None), ("critic_obs_normalizer", OLD)):
        for k, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            key = f"{net}.{k}"
            if key not in sd:
                continue
            x = sd[key]
            add = torch.full((x.shape[0], NEW), fill, dtype=x.dtype)
            sd[key] = torch.cat([x, add], 1) if at is None else torch.cat([x[:, :at], add, x[:, at:]], 1)
    return sd


def main():
    src, dst = sys.argv[1], sys.argv[2]
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
