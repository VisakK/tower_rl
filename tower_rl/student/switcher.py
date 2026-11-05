"""The option switcher (student plan Step 7, §2.3): a small pre-LN set transformer over the global, arm, block and
goal tokens (``tower_rl.student.tokens``), with factorised heads

    p(option) . p(goal | option) . p(arm | option, goal),  a progress P(satisfied) per goal,  a value (Step 12)

and deterministic masked argmax decoding. ``SwitcherNet.decide`` is pure tensor code, exported with TorchScript
together with the feature normaliser; ``SwitcherDecider`` wraps the export as a runtime decider (CPU, ~1 ms per
decision) and refuses an export of another feature version.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tower_rl.student.tokens import ARM_DIM, BLK_DIM, FEATURES, GLOB_DIM, GOAL_DIM, decode

N_OPT = 7
TAKES_GOAL = (1, 1, 1, 1, 1, 0, 0)  # acquire deliver push_off insert push_home | park | idle
TAKES_ARM = (1, 1, 1, 1, 1, 1, 0)
TOKEN_DIMS = {"blk": BLK_DIM, "goal": GOAL_DIM, "arm": ARM_DIM, "glob": GLOB_DIM}
NEG = -1e9


class Normalizer(nn.Module):
    """Per-feature standardisation of each token type (stats over the training rows' valid tokens; std floored)."""

    def __init__(self, stats: dict | None = None):
        super().__init__()
        for k, d in TOKEN_DIMS.items():
            m = torch.zeros(d) if stats is None else torch.as_tensor(stats[k]["mean"], dtype=torch.float32)
            s = torch.ones(d) if stats is None else torch.as_tensor(stats[k]["std"], dtype=torch.float32)
            self.register_buffer(f"{k}_mean", m)
            self.register_buffer(f"{k}_std", s)

    @staticmethod
    def fit(data: dict, std_floor: float = 0.1) -> dict:
        out = {}
        for k in TOKEN_DIMS:
            x = data[k]
            if k in ("blk", "goal"):
                x = x[data[f"{k}_mask"]]
            x = x.reshape(-1, TOKEN_DIMS[k]).astype(np.float64)
            out[k] = {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), std_floor).tolist()}
        return out

    def forward(self, blk, goal, arm, glob):
        return (((blk - self.blk_mean) / self.blk_std).clamp(-8.0, 8.0), ((goal - self.goal_mean) / self.goal_std).clamp(-8.0, 8.0),
                ((arm - self.arm_mean) / self.arm_std).clamp(-8.0, 8.0), ((glob - self.glob_mean) / self.glob_std).clamp(-8.0, 8.0))


class SwitcherNet(nn.Module):
    def __init__(self, d_model: int = 128, layers: int = 3, heads: int = 4, ff: int = 384, dropout: float = 0.0,
                 stats: dict | None = None):
        super().__init__()
        d = d_model
        self.d = d
        self.norm_in = Normalizer(stats)
        self.p_blk, self.p_goal = nn.Linear(BLK_DIM, d), nn.Linear(GOAL_DIM, d)
        self.p_arm, self.p_glob = nn.Linear(ARM_DIM, d), nn.Linear(GLOB_DIM, d)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.opt_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_OPT))
        self.opt_emb = nn.Embedding(N_OPT, d)
        self.q, self.k = nn.Linear(2 * d, d), nn.Linear(d, d)
        self.null_goal = nn.Parameter(torch.zeros(d))
        self.arm_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 2))
        self.prog_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("takes_goal", torch.tensor(TAKES_GOAL, dtype=torch.bool))
        self.register_buffer("takes_arm", torch.tensor(TAKES_ARM, dtype=torch.bool))

    def encode(self, blk, blk_mask, goal, goal_mask, arm, glob):
        blk, goal, arm, glob = self.norm_in(blk, goal, arm, glob)
        te = self.type_emb
        x = torch.cat([(self.p_glob(glob) + te[0])[:, None], self.p_arm(arm) + te[1], self.p_blk(blk) + te[2],
                       self.p_goal(goal) + te[3]], 1)
        b = blk.shape[0]
        pad = torch.cat([torch.zeros(b, 3, dtype=torch.bool, device=blk.device), ~blk_mask, ~goal_mask], 1)
        h = self.ln(self.enc(x, src_key_padding_mask=pad))
        return h[:, 0], h[:, 3 + blk.shape[1]:]

    def heads(self, hg, hgoal, opt, goal):
        """Logits for a given (option, goal) context (teacher forcing in training, the chosen ones when decoding)."""
        e = self.opt_emb(opt.clamp(min=0))
        q = self.q(torch.cat([hg, e], -1))
        goal_logits = torch.einsum("bgd,bd->bg", self.k(hgoal), q) / math.sqrt(self.d)
        idx = goal.clamp(min=0)
        hsel = hgoal[torch.arange(hgoal.shape[0], device=hgoal.device), idx]
        hsel = torch.where((goal >= 0)[:, None], hsel, self.null_goal.expand_as(hsel))
        arm_logits = self.arm_head(torch.cat([hg, e, hsel], -1))
        return goal_logits, arm_logits

    def forward(self, blk, blk_mask, goal, goal_mask, arm, glob, label_opt, label_goal):
        hg, hgoal = self.encode(blk, blk_mask, goal, goal_mask, arm, glob)
        opt_logits = self.opt_head(hg)
        goal_logits, arm_logits = self.heads(hg, hgoal, label_opt, label_goal)
        prog = self.prog_head(hgoal).squeeze(-1)
        value = self.value_head(hg).squeeze(-1)
        return opt_logits, goal_logits, arm_logits, prog, value

    @torch.jit.export
    def decide(self, blk, blk_mask, goal, goal_mask, arm, glob, valid_opt, valid_goal, valid_arm):
        """Masked argmax: ``[B, 3]`` (option, goal or -1, arm or -1) and the progress probabilities ``[B, 9]``."""
        hg, hgoal = self.encode(blk, blk_mask, goal, goal_mask, arm, glob)
        b = hg.shape[0]
        ar = torch.arange(b, device=hg.device)
        o = self.opt_head(hg).masked_fill(~valid_opt, -1e9).argmax(-1)
        vg = valid_goal[ar, o]
        g0 = torch.full_like(o, -1)
        gl, _ = self.heads(hg, hgoal, o, g0)
        g = gl.masked_fill(~vg, -1e9).argmax(-1)
        g = torch.where(self.takes_goal[o] & vg.any(-1), g, g0)
        _, al = self.heads(hg, hgoal, o, g)
        va = valid_arm[ar, o, g.clamp(min=0)]
        a = al.masked_fill(~va, -1e9).argmax(-1)
        a = torch.where(self.takes_arm[o] & va.any(-1), a, torch.full_like(a, -1))
        prog = torch.sigmoid(self.prog_head(hgoal).squeeze(-1))
        return torch.stack([o, g, a], -1), prog


def masked_ce(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row cross entropy over the valid entries (``target`` must be valid)."""
    return F.cross_entropy(logits.masked_fill(~mask, NEG), target, reduction="none")


def loss_fn(net: SwitcherNet, batch: dict, nonnominal_weight: float = 5.0, progress_weight: float = 0.5) -> tuple:
    """CE(option) + CE(goal | option takes one) + CE(arm | option takes one), x``nonnominal_weight`` on non-nominal
    decisions, + ``progress_weight`` BCE(progress) over valid goals of every row."""
    lo, lg, la = batch["label_opt"], batch["label_goal"], batch["label_arm"]
    dec = batch["kind"] == 0
    opt_logits, goal_logits, arm_logits, prog, _ = net(batch["blk"], batch["blk_mask"], batch["goal"], batch["goal_mask"],
                                                       batch["arm"], batch["glob"], lo.clamp(min=0), lg)
    w = 1.0 + (nonnominal_weight - 1.0) * batch["nonnominal"].float()
    zero = opt_logits.sum() * 0
    stats = {}
    if dec.any():
        b = torch.nonzero(dec).squeeze(-1)
        lo_d, lg_d, la_d, w_d = lo[b], lg[b], la[b], w[b]
        l_opt = (masked_ce(opt_logits[b], batch["valid_opt"][b], lo_d) * w_d).sum() / w_d.sum()
        vg = batch["valid_goal"][b, lo_d]
        hg = lg_d >= 0
        l_goal = (masked_ce(goal_logits[b][hg], vg[hg], lg_d[hg]) * w_d[hg]).sum() / w_d[hg].sum().clamp(min=1e-6) \
            if hg.any() else zero
        va = batch["valid_arm"][b, lo_d, lg_d.clamp(min=0)]
        ha = la_d >= 0
        l_arm = (masked_ce(arm_logits[b][ha], va[ha], la_d[ha]) * w_d[ha]).sum() / w_d[ha].sum().clamp(min=1e-6) \
            if ha.any() else zero
        stats.update(opt=float(l_opt), goal=float(l_goal), arm=float(l_arm))
    else:
        l_opt = l_goal = l_arm = zero
    pm = batch["progress"] >= 0
    l_prog = F.binary_cross_entropy_with_logits(prog[pm], batch["progress"][pm].float()) if pm.any() else zero
    stats["progress"] = float(l_prog)
    return l_opt + l_goal + l_arm + progress_weight * l_prog, stats


class _Export(nn.Module):
    """The TorchScript entry point: forward = SwitcherNet.decide."""

    def __init__(self, net: SwitcherNet):
        super().__init__()
        self.net = net

    def forward(self, blk, blk_mask, goal, goal_mask, arm, glob, valid_opt, valid_goal, valid_arm):
        return self.net.decide(blk, blk_mask, goal, goal_mask, arm, glob, valid_opt, valid_goal, valid_arm)


def export(net: SwitcherNet, path: str | Path, meta: dict) -> Path:
    """TorchScript (``<path>``) + JSON (``<path>.json``: feature version, model config, normaliser stats, sha256)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = net.cpu().eval()
    mod = torch.jit.script(_Export(net))
    mod.save(str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    stats = {k: {"mean": getattr(net.norm_in, f"{k}_mean").tolist(), "std": getattr(net.norm_in, f"{k}_std").tolist()}
             for k in TOKEN_DIMS}
    Path(str(path) + ".json").write_text(json.dumps({"features": FEATURES, "sha256": sha, "normalizer": stats, **meta}, indent=1))
    return path


class SwitcherDecider:
    """A runtime decider from an export: tokens (the run's ``TokenTracker``) -> masked argmax -> Decision."""

    needs_tokens = True
    exact = False

    def __init__(self, module, meta: dict):
        self.mod = module
        self.meta = meta
        self.name = "switcher"
        self.last_progress = None

    @classmethod
    def load(cls, path: str | Path) -> SwitcherDecider:
        meta = json.loads(Path(str(path) + ".json").read_text())
        if meta.get("features") != FEATURES:
            raise ValueError(f"{path}: features {meta.get('features')} != {FEATURES}")
        mod = torch.jit.load(str(path), map_location="cpu").eval()
        return cls(mod, meta)

    @torch.no_grad()
    def decide(self, view):
        tk = view.run.tok.tokens(view)
        t = {k: torch.as_tensor(np.asarray(v))[None] for k, v in tk.items()}
        out, prog = self.mod(t["blk"], t["blk_mask"], t["goal"], t["goal_mask"], t["arm"], t["glob"], t["valid_opt"],
                             t["valid_goal"], t["valid_arm"])
        o, g, a = (int(x) for x in out[0])
        self.last_progress = prog[0].numpy()
        return decode(o, g, a)
