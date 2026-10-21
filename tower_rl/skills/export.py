"""Freeze a trained skill (Step 5 deliverable): rsl_rl checkpoint -> TorchScript actor with its observation normalizer.

The exported module maps the raw actor observation ``[N, D]`` to the deterministic action mean ``[N, 7]`` (the
executor clamps to [-1, 1]). No Isaac imports: usable in the bundle and the later policy server.

Usage: python -m tower_rl.skills.export <run dir or model_*.pt> <out.pt>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict


class FrozenActor(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module, normalizer: torch.nn.Module):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


def load_actor_critic(ckpt: Path) -> tuple[ActorCritic, dict]:
    run = ckpt.parent
    cfg = json.loads((run / "config.json").read_text())
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    n_actor = sd["actor.0.weight"].shape[1]
    n_critic = sd["critic.0.weight"].shape[1]
    n_act = sd["std"].shape[0] if "std" in sd else sd["log_std"].shape[0]
    pol = dict(cfg["runner"]["policy"])
    pol.pop("class_name", None)
    obs = TensorDict({"policy": torch.zeros(1, n_actor), "critic": torch.zeros(1, n_critic)}, batch_size=[1])
    ac = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, n_act, **pol)
    ac.load_state_dict(sd)
    ac.eval()
    return ac, cfg


def export(ckpt: str | Path, out: str | Path) -> Path:
    ckpt = Path(ckpt)
    if ckpt.is_dir():
        final = ckpt / "model_final.pt"
        ckpt = final if final.exists() else sorted(ckpt.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    ac, cfg = load_actor_critic(ckpt)
    frozen = FrozenActor(ac.actor, ac.actor_obs_normalizer).eval()
    scripted = torch.jit.script(frozen)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))
    # sanity: the scripted module equals the rsl_rl inference path
    x = torch.randn(8, ac.actor[0].in_features)
    with torch.no_grad():
        ref = ac.act_inference(TensorDict({"policy": x, "critic": torch.zeros(8, ac.critic[0].in_features)}, batch_size=[8]))
        got = torch.jit.load(str(out))(x)
    assert torch.allclose(ref, got, atol=1e-5), "exported actor differs from rsl_rl inference"
    out.with_suffix(".json").write_text(json.dumps({"checkpoint": str(ckpt), "obs_dim": int(x.shape[1]),
                                                    "skill": cfg["skill"]}, indent=1, default=str))
    return out


if __name__ == "__main__":
    print(export(sys.argv[1], sys.argv[2]))
