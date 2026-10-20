"""Rigid-body math in torch, free of Isaac imports (usable in tests, data analysis and a future policy server).

Conventions: quaternions are ``(w, x, y, z)`` like Isaac Lab; poses are ``(pos [..., 3], quat [..., 4])``.
The dataset stores ``x y z qx qy qz qw``: convert with :func:`xyzw_to_wxyz` / :func:`pose7_to_pq`.
"""

from __future__ import annotations

import math

import torch


def as_tensor(x, dtype=torch.float32, device=None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., :3]], dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def pose7_to_pq(p7: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dataset pose ``[..., x y z qx qy qz qw]`` -> ``(pos, quat_wxyz)``."""
    return p7[..., :3], xyzw_to_wxyz(p7[..., 3:7])


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by unit quaternions ``q`` (broadcasting)."""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = normalize(q).unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` -> unit quaternions with ``w >= 0``."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # four candidate solutions, pick the numerically best per element
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)) / 2
    cands = torch.stack(
        [
            torch.stack([qw, (m21 - m12) / (4 * qw.clamp_min(1e-9)), (m02 - m20) / (4 * qw.clamp_min(1e-9)), (m10 - m01) / (4 * qw.clamp_min(1e-9))], -1),
            torch.stack([(m21 - m12) / (4 * qx.clamp_min(1e-9)), qx, (m01 + m10) / (4 * qx.clamp_min(1e-9)), (m02 + m20) / (4 * qx.clamp_min(1e-9))], -1),
            torch.stack([(m02 - m20) / (4 * qy.clamp_min(1e-9)), (m01 + m10) / (4 * qy.clamp_min(1e-9)), qy, (m12 + m21) / (4 * qy.clamp_min(1e-9))], -1),
            torch.stack([(m10 - m01) / (4 * qz.clamp_min(1e-9)), (m02 + m20) / (4 * qz.clamp_min(1e-9)), (m12 + m21) / (4 * qz.clamp_min(1e-9)), qz], -1),
        ],
        dim=-2,
    )
    best = torch.stack([qw, qx, qy, qz], -1).argmax(-1)
    q = torch.gather(cands, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = normalize(q)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = angle[..., None] / 2
    return torch.cat([torch.cos(half), normalize(axis) * torch.sin(half)], dim=-1)


def quat_from_rotvec(rv: torch.Tensor) -> torch.Tensor:
    angle = rv.norm(dim=-1)
    half = angle / 2
    # sin(half)/angle with a Taylor fallback near zero
    k = torch.where(angle > 1e-6, torch.sin(half) / angle.clamp_min(1e-12), 0.5 - angle**2 / 48)
    return torch.cat([torch.cos(half)[..., None], rv * k[..., None]], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of a unit quaternion, shortest rotation (angle in [0, pi])."""
    q = torch.where(q[..., :1] < 0, -q, q)
    s = q[..., 1:].norm(dim=-1)
    angle = 2 * torch.atan2(s, q[..., 0])
    k = torch.where(s > 1e-7, angle / s.clamp_min(1e-12), 2.0 + 0 * s)
    return q[..., 1:] * k[..., None]


def quat_error_rotvec(q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """World-frame rotation vector taking ``q_current`` to ``q_target``."""
    return rotvec_from_quat(quat_mul(q_target, quat_conj(q_current)))


def quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (rad) of the relative rotation between unit quaternions (atan2 form: exact near zero in float32)."""
    r = quat_mul(quat_conj(a), b)
    return 2 * torch.atan2(r[..., 1:].norm(dim=-1), r[..., 0].abs())


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)


def tilt_angle(q: torch.Tensor) -> torch.Tensor:
    """Angle between the body z axis and the world z axis (rad)."""
    w, x, y, z = q.unbind(-1)
    return torch.acos((1 - 2 * (x * x + y * y)).clamp(-1, 1))


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def yaw_diff_sym(a: torch.Tensor, b: torch.Tensor, period: float = math.pi) -> torch.Tensor:
    """Smallest |a - b| for yaw angles of an object with a ``period`` symmetry (a block is symmetric under pi)."""
    d = torch.remainder(a - b + period / 2, period) - period / 2
    return d.abs()


def frame_compose(p1, q1, p2, q2):
    """``T1 * T2``."""
    return p1 + quat_apply(q1, p2), quat_mul(q1, q2)


def frame_inverse(p, q):
    qi = quat_conj(q)
    return -quat_apply(qi, p), qi


def frame_relative(p1, q1, p2, q2):
    """``T1^-1 * T2`` (pose of frame 2 expressed in frame 1)."""
    pi, qi = frame_inverse(p1, q1)
    return frame_compose(pi, qi, p2, q2)


def box_corners(half_extents: torch.Tensor) -> torch.Tensor:
    """The 8 corners ``[8, 3]`` of a box centred at the origin."""
    s = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=half_extents.dtype, device=half_extents.device)
    return s * half_extents
