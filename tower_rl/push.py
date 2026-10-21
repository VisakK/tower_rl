"""Pusher geometry for Task 4 (push a block off the tower) and Task 5 (push an inserted block home), pure torch.

Measured in the demos (data/subset/task4-*, BUILDLOG 2026-10-20): the fingers are fully closed (width 0) and form a
tool; the pinch starts ~100 mm behind the block centre along the push direction (25 mm outside the end face; first
contact at ~86.5 mm) and 7-16 mm above the block centre (``configs/skills/push_priors.yaml`` per arm and layer); the
approach axis is tilted toward the push direction (left 45 deg, back 30 deg) with the jaw axis horizontal and
perpendicular to the push. The push direction is the block's long axis pointing away from the arm (left +x, back +y);
the layout records it per push (``Layout.pushes``).
"""

from __future__ import annotations

import math

import torch

from tower_rl import geometry as geo
from tower_rl.config import load_yaml

_CFG = None
_PRI = None


def cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_yaml("skills/push")["geometry"]
    return _CFG


def priors() -> dict:
    global _PRI
    if _PRI is None:
        _PRI = load_yaml("skills/push_priors")["push"]
    return _PRI


def pusher_quat(direction: torch.Tensor, arm: str) -> torch.Tensor:
    """Pinch orientation ``[N, 4]`` for pushing along ``direction`` ``[N, 3]`` (world, horizontal unit vectors)."""
    t = math.radians(cfg()["tilt_deg"][arm])
    d = direction.clone()
    d[..., 2] = 0
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    up = torch.zeros_like(d)
    up[..., 2] = 1.0
    app = math.sin(t) * d - math.cos(t) * up          # approach (pinch z): down, leaning toward the push
    jaw = torch.cross(d, up, dim=-1)                   # horizontal, perpendicular to the push (demo sign)
    x = torch.cross(jaw, app, dim=-1)
    return geo.mat_to_quat(torch.stack([x, jaw, app], dim=-1))


def z_offset(arm: str, layer: int) -> float:
    """Pinch height above the block centre at the push start (demo median for this arm and layer)."""
    p = priors()[arm]
    e = p.get(layer) or p.get(str(layer)) or next(iter(p.values()))
    return e["pinch_rel_push_frame_mm"][2] / 1000.0


def push_pose(block_pos: torch.Tensor, direction: torch.Tensor, arm: str, along_m: float, z_m: float | torch.Tensor):
    """Pinch pose ``along_m`` (negative: behind) from the block centre along the push and ``z_m`` above it."""
    d = direction.clone()
    d[..., 2] = 0
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    p = block_pos + d * along_m
    p = p.clone()
    p[..., 2] = block_pos[..., 2] + z_m
    return p, pusher_quat(d, arm)


def prepush_pose(block_pos, direction, arm: str, layer: int):
    return push_pose(block_pos, direction, arm, cfg()["prepush_along_m"], z_offset(arm, layer))


def contact_pose(block_pos, direction, arm: str, layer: int):
    return push_pose(block_pos, direction, arm, cfg()["contact_along_m"], z_offset(arm, layer))


def prior(arm: str, layer: int) -> list[float]:
    p = priors()[arm]
    e = p.get(layer) or p.get(str(layer)) or next(iter(p.values()))
    return e["q"]


def progress(block_pos: torch.Tensor, block_pos0: torch.Tensor, direction: torch.Tensor):
    """(along, lateral) displacement of a block from its start, in the push frame."""
    d = direction.clone()
    d[..., 2] = 0
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    lat = torch.stack([-d[..., 1], d[..., 0]], -1)
    dp = (block_pos - block_pos0)[..., :2]
    return (dp * d[..., :2]).sum(-1), (dp * lat).sum(-1)


def departed(block_pos: torch.Tensor, block_z0: torch.Tensor, tower_xy: torch.Tensor) -> torch.Tensor:
    """The block has left the tower (the evaluator-side rule of ``tower_rl.eval.operations.PushTracker``)."""
    r = (block_pos[..., :2] - tower_xy).norm(dim=-1)
    c = cfg()
    # no longer touching the tower (the generator: support clearance > 2 mm): its centre beyond the tower face by half
    # its length, or lying flat on the table next to it (a block that tipped off one end may still lean on the tower)
    return (r >= c["depart_m"]) | ((r >= 0.09) & (block_pos[..., 2] <= c["on_table_z"]))


def debris_poses(p0: torch.Tensor, q0: torch.Tensor, direction: torch.Tensor, u: torch.Tensor, db: dict):
    """Where a pushed-off block lies on the table (skill-env resets: the debris of the earlier pushes, which the last
    block of a layer lands on). Task 4 traces: flat or on its side, centre 160-175 mm out along its push. ``p0`` / ``q0``
    its pose on the tower [N, 3] / [N, 4], ``direction`` its push direction [N, 3], ``u`` uniforms [N, 4]; ``db`` the
    ``reset.debris`` config. Flat blocks start beyond the foundation (centre >= ``along_m[0]`` out)."""
    d = direction.clone()
    d[:, 2] = 0
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    lat = torch.stack([-d[:, 1], d[:, 0], torch.zeros_like(d[:, 0])], -1)
    lo, hi = db["along_m"]
    p = p0 + d * (lo + (hi - lo) * u[:, 0:1]) + lat * ((u[:, 1:2] * 2 - 1) * db["lat_m"])
    side = u[:, 3] < db["side_frac"]
    p[:, 2] = torch.where(side, torch.full_like(p[:, 2], db["z_side_m"]), torch.full_like(p[:, 2], db["z_flat_m"]))
    q = geo.quat_mul(geo.quat_from_yaw((u[:, 2] * 2 - 1) * math.radians(db["yaw_deg"])), q0)
    x_axis = torch.zeros_like(p)
    x_axis[:, 0] = 1.0
    rolled = geo.quat_mul(q, geo.quat_from_axis_angle(x_axis, torch.full_like(p[:, 0], math.pi / 2)))
    return p, geo.normalize(torch.where(side[:, None], rolled, q))
