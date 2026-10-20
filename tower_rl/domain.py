"""Per-episode domain samples for a tier of ``configs/tiers.yaml`` (pure torch, seeded, no Isaac imports)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tower_rl.config import deep_update, load_yaml

NOMINAL_BLOCK_FRICTION = (0.40, 0.23)
SLIPPERY = ((0.15, 0.30), (0.07, 0.14))
HIGH_FRICTION = ((0.9, 1.3), (0.5, 0.8))


@dataclass
class DomainSample:
    case: list[str]
    assembly_xy: torch.Tensor    # [N, 2] m
    assembly_yaw: torch.Tensor   # [N] rad
    layer_xy: torch.Tensor       # [N, L, 2]
    layer_yaw: torch.Tensor      # [N, L]
    block_xy: torch.Tensor       # [N, M, 2]
    mass_scale: torch.Tensor     # [N, M]
    friction: torch.Tensor       # [N, M, 2] static, dynamic
    finger_friction: torch.Tensor  # [N, 2]
    light_scale: torch.Tensor    # [N]
    color_temp_k: torch.Tensor   # [N] (nan = nominal)
    light_azimuth: torch.Tensor  # [N] rad

    def subset(self, idx: torch.Tensor) -> DomainSample:
        i = idx.tolist()
        return DomainSample([self.case[k] for k in i], *(getattr(self, f)[idx] for f in (
            "assembly_xy", "assembly_yaw", "layer_xy", "layer_yaw", "block_xy", "mass_scale", "friction",
            "finger_friction", "light_scale", "color_temp_k", "light_azimuth")))


def _u(gen, shape, lo, hi, device):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def _pm(gen, shape, a, device):
    return _u(gen, shape, -a, a, device)


class DomainSampler:
    def __init__(self, tier: str, max_layers: int = 3, max_blocks: int = 6, device: str = "cpu",
                 robot_randomization: bool = True, cfg: dict | None = None):
        cfg = cfg or load_yaml("tiers")
        self.tier = tier
        self.cases = {name: deep_update(cfg["nominal"], over or {}) for name, over in cfg["tiers"][tier]["cases"].items()}
        self.case_names = list(self.cases)
        self.robot = cfg["robot"] if robot_randomization else None
        self.L, self.M, self.device = max_layers, max_blocks, device

    def sample(self, n: int, gen: torch.Generator | None = None, case: str | None = None) -> DomainSample:
        dev = self.device
        if case is not None:
            idx = torch.full((n,), self.case_names.index(case), dtype=torch.long)
        else:
            idx = torch.randint(len(self.case_names), (n,), generator=gen, device=dev)
        names = [self.case_names[i] for i in idx.tolist()]
        out = DomainSample(
            case=names,
            assembly_xy=torch.zeros(n, 2, device=dev), assembly_yaw=torch.zeros(n, device=dev),
            layer_xy=torch.zeros(n, self.L, 2, device=dev), layer_yaw=torch.zeros(n, self.L, device=dev),
            block_xy=torch.zeros(n, self.M, 2, device=dev), mass_scale=torch.ones(n, self.M, device=dev),
            friction=torch.tensor(NOMINAL_BLOCK_FRICTION, device=dev).repeat(n, self.M, 1),
            finger_friction=torch.tensor([2.4, 2.0], device=dev).repeat(n, 1),
            light_scale=torch.ones(n, device=dev), color_temp_k=torch.full((n,), float("nan"), device=dev),
            light_azimuth=torch.zeros(n, device=dev),
        )
        for ci, name in enumerate(self.case_names):
            sel = (idx == ci).nonzero(as_tuple=True)[0]
            k = len(sel)
            if k == 0:
                continue
            c = self.cases[name]
            out.assembly_xy[sel] = _pm(gen, (k, 2), c["assembly_xy_m"], dev)
            out.assembly_yaw[sel] = _pm(gen, (k,), math.radians(c["assembly_yaw_deg"]), dev)
            out.layer_xy[sel] = _pm(gen, (k, self.L, 2), c["layer_xy_m"], dev)
            out.layer_yaw[sel] = _pm(gen, (k, self.L), math.radians(c["layer_yaw_deg"]), dev)
            out.block_xy[sel] = _pm(gen, (k, self.M, 2), c["block_xy_m"], dev)
            if c["per_block_mass"]:
                out.mass_scale[sel] = _u(gen, (k, self.M), *c["per_block_mass"], dev)
            else:
                out.mass_scale[sel] = _u(gen, (k, 1), *c["mass_scale"], dev).expand(k, self.M)
            if c["mixed_friction"]:
                slip = torch.rand((k, self.M), generator=gen, device=dev) < 0.5
                lo_s, lo_d = SLIPPERY
                hi_s, hi_d = HIGH_FRICTION
                fs = torch.where(slip, _u(gen, (k, self.M), *lo_s, dev), _u(gen, (k, self.M), *hi_s, dev))
                fd = torch.where(slip, _u(gen, (k, self.M), *lo_d, dev), _u(gen, (k, self.M), *hi_d, dev))
                out.friction[sel] = torch.stack([fs, fd], -1)
            elif c["friction_static"]:
                fs = _u(gen, (k, 1), *c["friction_static"], dev).expand(k, self.M)
                fd = _u(gen, (k, 1), *c["friction_dynamic"], dev).expand(k, self.M)
                out.friction[sel] = torch.stack([fs, torch.minimum(fd, fs)], -1)
            out.light_scale[sel] = _u(gen, (k,), *c["light_scale"], dev)
            if c["color_temp_k"]:
                out.color_temp_k[sel] = _u(gen, (k,), *c["color_temp_k"], dev)
            out.light_azimuth[sel] = _pm(gen, (k,), math.radians(c["light_azimuth_deg"]), dev)
        if self.robot is not None:
            fs = _u(gen, (n,), *self.robot["finger_friction_static"], dev)
            out.finger_friction = torch.stack([fs, fs * _u(gen, (n,), *self.robot["finger_friction_dynamic_ratio"], dev)], -1)
        return out
