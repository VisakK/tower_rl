"""Posture priors (``configs/skills/posture_priors.yaml``): demo joint configurations used as IK seeds / null-space
references so our IK stays in the demonstrated posture families (pure python/torch)."""

from __future__ import annotations

import math

import torch

from tower_rl import geometry as geo
from tower_rl.config import load_yaml

PADS = {"A": (0.426, -0.277), "B": (0.174, -0.525)}
_P = None


def priors() -> dict:
    global _P
    if _P is None:
        _P = load_yaml("skills/posture_priors")["postures"]
    return _P


def pad_of(xy) -> str:
    x, y = float(xy[0]), float(xy[1])
    return min(PADS, key=lambda k: math.dist(PADS[k], (x, y)))


def yaw_class(yaw: float) -> int:
    """Block yaw (rad) -> 0 or 90 (deg), the block being 180-deg symmetric."""
    y = (math.degrees(yaw) + 45) % 180 - 45
    return 0 if abs(y) < 45 else 90


def prior(arm: str, phase: str, pad: str, ycls: int) -> list[float]:
    """Joint seed for ``phase`` in {grasp, release, travel}; falls back to any yaw class / pad of that phase."""
    p = priors()[arm]
    ph = p.get(phase) or p["grasp"]
    by_pad = ph.get(pad) or next((v for k, v in ph.items() if k.startswith(pad) or k.endswith(pad)), None) or next(iter(ph.values()))
    entry = by_pad.get(ycls) or by_pad.get(str(ycls)) or next(iter(by_pad.values()))
    return entry["q"]


def prior_for_pose(arm: str, phase: str, pos: torch.Tensor, block_quat: torch.Tensor) -> torch.Tensor:
    """Batched seeds ``[N, 7]`` for blocks at ``pos`` [N,3] with orientations ``block_quat`` [N,4]."""
    yaws = geo.yaw_from_quat(block_quat).tolist()
    rows = [prior(arm, phase, pad_of(p[:2]), yaw_class(y)) for p, y in zip(pos.tolist(), yaws)]
    return torch.tensor(rows, dtype=pos.dtype, device=pos.device)
