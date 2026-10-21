"""Planned transit between skill initiation regions (Step 4): no learning, IK + via-points (pure torch).

Path for one arm from its current pinch pose to a goal pinch pose (pre-grasp or pre-place), in the station frame:
  1. rise vertically to the travel height ``z_t = max(z_start, z_goal) + rise_m`` (the start already clears the
     source tower: after grasp_lift the block hangs >= 6 cm above it; the goal is a pre-pose above its stack, so the
     straight segment at ``z_t`` clears both towers),
  2. a straight line at ``z_t`` to above the goal, the orientation slerped along it,
  3. descend vertically onto the goal.
Each segment gets a smoothstep (zero velocity and acceleration at the ends) time law whose duration respects the
speed / acceleration limits (lower when holding a block: 0.1 s rows excite the soft arm, BUILDLOG). The 10 Hz rows
come from :class:`tower_rl.control.ArmController` (IK + gravity-sag compensation), so they are exactly what the
evaluator receives. The other arm must be parked (the executor runs one operation at a time).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tower_rl import geometry as geo


@dataclass
class TransitCfg:
    rise_m: float = 0.02
    v_max: float = 0.25        # m/s, empty gripper
    v_max_held: float = 0.12   # m/s, holding a block
    a_max: float = 0.6         # m/s^2
    w_max: float = 1.0         # rad/s
    min_segment_s: float = 0.3
    settle_s: float = 0.4      # hold the goal before reporting arrival
    tol_pos_m: float = 0.03    # initiation region of the next skill
    tol_rot_deg: float = 5.0
    dt: float = 0.1            # one row


def _smoothstep(s: torch.Tensor) -> torch.Tensor:
    """Quintic 0 -> 1 with zero velocity/acceleration at both ends."""
    s = s.clamp(0, 1)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def _duration(dist: float, ang: float, v: float, a: float, w: float, min_s: float) -> float:
    # quintic peak speed = 1.875 d / T, peak acceleration = 5.77 d / T^2
    t_v = 1.875 * dist / v
    t_a = math.sqrt(5.77 * dist / a) if dist > 0 else 0.0
    t_w = 1.875 * ang / w
    return max(min_s, t_v, t_a, t_w)


def plan_waypoints(p0: torch.Tensor, q0: torch.Tensor, p1: torch.Tensor, q1: torch.Tensor, held: bool,
                   cfg: TransitCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-by-row target poses ``([T,3], [T,4])`` for one arm (single env, station frame)."""
    zt = max(float(p0[2]), float(p1[2])) + cfg.rise_m
    a = p0.clone()
    b = p0.clone()
    b[2] = zt
    c = p1.clone()
    c[2] = zt
    v = cfg.v_max_held if held else cfg.v_max
    segs = [(a, b, q0, q0), (b, c, q0, q1), (c, p1.clone(), q1, q1)]
    ps, qs = [], []
    for pa, pb, qa, qb in segs:
        dist = float((pb - pa).norm())
        ang = float(geo.quat_angle(qa, qb))
        if dist < 1e-4 and ang < 1e-3:
            continue
        T = _duration(dist, ang, v, cfg.a_max, cfg.w_max, cfg.min_segment_s)
        n = max(1, int(math.ceil(T / cfg.dt)))
        s = _smoothstep(torch.arange(1, n + 1, dtype=p0.dtype, device=p0.device) / n)
        ps.append(pa[None] + s[:, None] * (pb - pa)[None])
        qs.append(slerp(qa, qb, s))
    if not ps:
        return p1[None].clone(), q1[None].clone()
    return torch.cat(ps), torch.cat(qs)


def slerp(qa: torch.Tensor, qb: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation from ``qa`` to ``qb`` at fractions ``s [T]`` -> ``[T,4]``."""
    if float((qa * qb).sum()) < 0:
        qb = -qb
    rv = geo.rotvec_from_quat(geo.quat_mul(qb, geo.quat_conj(qa)))
    return geo.quat_mul(geo.quat_from_rotvec(s[:, None] * rv[None]), qa[None].expand(len(s), 4))


class TransitPlan:
    """Row generator for one env and arm; ``next_rows(ctrl, env, H)`` advances the controller target and returns
    sag-compensated joint rows ``[H, 7]`` (open-loop chunk; the target keeps the goal after the path ends)."""

    def __init__(self, ps: torch.Tensor, qs: torch.Tensor, held: bool, cfg: TransitCfg):
        self.ps, self.qs, self.held, self.cfg = ps, qs, held, cfg
        self.i = 0
        self.settle_rows = int(round(cfg.settle_s / cfg.dt))

    @property
    def rows_left(self) -> int:
        return max(0, len(self.ps) + self.settle_rows - self.i)

    @property
    def path_done(self) -> bool:
        return self.i >= len(self.ps)

    def next_targets(self, h: int):
        idx = torch.clamp(torch.arange(self.i, self.i + h, device=self.ps.device), max=len(self.ps) - 1)
        self.i += h
        return self.ps[idx], self.qs[idx]


def arrived(pinch_pos: torch.Tensor, pinch_quat: torch.Tensor, goal_pos: torch.Tensor, goal_quat: torch.Tensor,
            cfg: TransitCfg) -> torch.Tensor:
    return ((pinch_pos - goal_pos).norm(dim=-1) <= cfg.tol_pos_m) & \
        (geo.quat_angle(pinch_quat, goal_quat) <= math.radians(cfg.tol_rot_deg))
