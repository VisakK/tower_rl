"""Task 5 actor observations as pure functions (shared by the insert / push_in envs and the executor; no Isaac imports).

``insert_actor_obs`` (base frame): pinch pose / velocity / lead (21), width, gripper request, width command (3), the insert
block in the pinch frame (9), slot centre and axis (6), the block relative to the slot (along / across / vertical, cos /
sin 2 dyaw) (5), phase one-hot (3), previous action (7), progress (1) = 55.
"""

from __future__ import annotations

import torch

from tower_rl import geometry as geo
from tower_rl.skills.obs import _axes_in, rot6d

ACTOR_DIM = 55


def slot_relative(block_pos, block_quat, slot_pos, slot_quat, axis):
    """(along, across, vertical, cos 2dyaw, sin 2dyaw) of a block relative to a slot (``axis`` = insertion direction)."""
    d = block_pos - slot_pos
    lat = torch.stack([-axis[..., 1], axis[..., 0], torch.zeros_like(axis[..., 0])], -1)
    dyaw = geo.yaw_from_quat(block_quat) - geo.yaw_from_quat(slot_quat)
    return torch.stack([(d * axis).sum(-1), (d * lat).sum(-1), d[..., 2], torch.cos(2 * dyaw), torch.sin(2 * dyaw)], -1)


def insert_actor_obs(*, pb, qb, prev_pb, prev_qb, dt, tgt_pb, tgt_qb, width, closed, width_cmd, block_pb, block_qb,
                     slot_pb, slot_qb, axis_b, phase, prev_actions, progress) -> torch.Tensor:
    vel = torch.cat([(pb - prev_pb) / dt, geo.quat_error_rotvec(prev_qb, qb) / dt], -1)
    lead = torch.cat([tgt_pb - pb, geo.quat_error_rotvec(qb, tgt_qb)], -1)
    bp, bx, bz = _axes_in(pb, qb, block_pb, block_qb)
    rel = slot_relative(block_pb, block_qb, slot_pb, slot_qb, axis_b)
    return torch.cat([pb, rot6d(qb), vel, lead, width[:, None], closed.float()[:, None] * 2 - 1, width_cmd[:, None],
                      bp, bx, bz, slot_pb, axis_b, rel, phase, prev_actions, progress[:, None]], -1)


def pad_actions(a: torch.Tensor) -> torch.Tensor:
    """Previous actions as 7 values (push_in acts with 6: no gripper channel)."""
    return a if a.shape[1] == 7 else torch.cat([a, torch.zeros(a.shape[0], 7 - a.shape[1], device=a.device)], -1)
