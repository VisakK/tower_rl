"""Actor observations of the contact skills as pure functions (shared by the training envs and the executor).

Everything here is computable at test time from joint angles, gripper widths, the executor's own commands and
(estimated) block poses; nothing privileged. All poses are in the arm's base frame.
"""

from __future__ import annotations

import torch

from tower_rl import geometry as geo
from tower_rl.grasp import cfg as grasp_cfg
from tower_rl.grasp import gripper_quat


def rot6d(q: torch.Tensor) -> torch.Tensor:
    m = geo.quat_to_mat(q)
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def grasp_target_world(block_pos: torch.Tensor, block_quat: torch.Tensor, arm: str, tilt_deg: float | None = None):
    """Canonical pinch pose that grasps the block centred (station frame)."""
    q = gripper_quat(block_quat, block_pos, arm, tilt_deg)
    off = torch.tensor(grasp_cfg()["pinch_offset_in_block_m"], device=block_pos.device, dtype=block_pos.dtype)
    return block_pos + geo.quat_apply(block_quat, off.expand_as(block_pos)), q


def neighbors(other_pb: torch.Tensor, other_qb: torch.Tensor, valid: torch.Tensor, tpb: torch.Tensor, tqb: torch.Tensor,
              k: int = 2) -> torch.Tensor:
    """``k`` nearest valid other blocks relative to the target: target-frame position, cos/sin 2*dyaw, valid flag."""
    n = tpb.shape[0]
    ar = torch.arange(n, device=tpb.device)
    if other_pb.shape[1] < k:  # fewer candidates than features (e.g. Task 2-2 at the executor): pad as invalid
        pad = k - other_pb.shape[1]
        other_pb = torch.cat([other_pb, torch.zeros(n, pad, 3, device=tpb.device, dtype=tpb.dtype)], 1)
        other_qb = torch.cat([other_qb, torch.tensor([1.0, 0, 0, 0], device=tpb.device).expand(n, pad, 4)], 1)
        valid = torch.cat([valid, torch.zeros(n, pad, dtype=torch.bool, device=tpb.device)], 1)
    rel_p, rel_q = geo.frame_relative(tpb[:, None].expand_as(other_pb), tqb[:, None].expand_as(other_qb), other_pb, other_qb)
    d = torch.where(valid, rel_p.norm(dim=-1), torch.full_like(rel_p[..., 0], 1e3))
    idx = d.argsort(-1)[:, :k]
    out = []
    for j in range(k):
        i = idx[:, j]
        ok = valid[ar, i].float()[:, None]
        yaw = geo.yaw_from_quat(rel_q[ar, i])
        out.append(torch.cat([rel_p[ar, i] * ok, torch.cos(2 * yaw)[:, None] * ok, torch.sin(2 * yaw)[:, None] * ok, ok], -1))
    return torch.cat(out, -1)


def grasp_lift_actor_obs(*, pb, qb, prev_pb, prev_qb, dt, tgt_pb, tgt_qb, width, closed, width_cmd, target_pb, target_qb,
                         grasp_pb, grasp_qb, other_pb, other_qb, other_valid, target_z_world, target_z0_world,
                         prev_actions, progress) -> torch.Tensor:
    """65-D grasp_lift actor observation (layout documented in ``grasp_lift_env``)."""
    vel = torch.cat([(pb - prev_pb) / dt, geo.quat_error_rotvec(prev_qb, qb) / dt], -1)
    lead = torch.cat([tgt_pb - pb, geo.quat_error_rotvec(qb, tgt_qb)], -1)
    rel_p, rel_q = geo.frame_relative(pb, qb, target_pb, target_qb)
    rm = geo.quat_to_mat(rel_q)
    bx = rm[..., :, 0] * torch.where(rm[..., 0:1, 0] < 0, -1.0, 1.0)  # canonical sign (block is pi-symmetric)
    bz = rm[..., :, 2]
    gerr_p, gerr_q = geo.frame_relative(pb, qb, grasp_pb, grasp_qb)
    gerr = torch.cat([gerr_p, geo.rotvec_from_quat(gerr_q)], -1)
    neigh = neighbors(other_pb, other_qb, other_valid, target_pb, target_qb)
    lifted = (target_z_world - target_z0_world) > 0.01
    phase = torch.stack([~closed, closed & ~lifted, closed & lifted], -1).float()
    return torch.cat([pb, rot6d(qb), vel, lead, width[:, None], closed.float()[:, None] * 2 - 1, width_cmd[:, None],
                      rel_p, bx, bz, target_pb, gerr, neigh, prev_actions, phase, progress[:, None]], -1)


def _axes_in(pb, qb, p, q):
    """Pose ``(p, q)`` expressed in the frame ``(pb, qb)``: position, canonical x axis (pi-symmetric), z axis."""
    rel_p, rel_q = geo.frame_relative(pb, qb, p, q)
    rm = geo.quat_to_mat(rel_q)
    bx = rm[..., :, 0] * torch.where(rm[..., 0:1, 0] < 0, -1.0, 1.0)
    return rel_p, bx, rm[..., :, 2]


def place_release_actor_obs(*, pb, qb, prev_pb, prev_qb, dt, tgt_pb, tgt_qb, width, closed, width_cmd, goal_pb, goal_qb,
                            block_pb, block_qb, other_pb, other_qb, other_valid, phase, prev_actions, progress) -> torch.Tensor:
    """73-D place_release actor observation (base frame; layout documented in ``place_release_env``)."""
    vel = torch.cat([(pb - prev_pb) / dt, geo.quat_error_rotvec(prev_qb, qb) / dt], -1)
    lead = torch.cat([tgt_pb - pb, geo.quat_error_rotvec(qb, tgt_qb)], -1)
    gp, gx, gz = _axes_in(pb, qb, goal_pb, goal_qb)
    bp, bx, bz = _axes_in(pb, qb, block_pb, block_qb)
    dyaw = geo.yaw_from_quat(block_qb) - geo.yaw_from_quat(goal_qb)
    berr = torch.cat([block_pb - goal_pb, torch.cos(2 * dyaw)[:, None], torch.sin(2 * dyaw)[:, None]], -1)
    sup = neighbors(other_pb, other_qb, other_valid, goal_pb, goal_qb)
    return torch.cat([pb, rot6d(qb), vel, lead, width[:, None], closed.float()[:, None] * 2 - 1, width_cmd[:, None],
                      gp, gx, gz, bp, bx, bz, berr, sup, goal_pb, prev_actions, phase, progress[:, None]], -1)


def push_actor_obs(*, pb, qb, prev_pb, prev_qb, dt, tgt_pb, tgt_qb, target_pb, target_qb, dir_b, along, lateral, dyaw,
                   contact_pb, contact_qb, other_pb, other_qb, other_valid, z0_world, prev_actions, progress) -> torch.Tensor:
    """65-D push actor observation (base frame; Task 4 push and Task 5 push-in, layout in ``push_env``): pinch pose /
    velocity / target lead (21), target block in the pinch frame (9), target position (3), push direction (3), target
    travel along / across the push and its yaw change since the start (3), pinch error to the contact pose (6), two
    nearest other blocks relative to the target (12), the target's start height (1), previous action (6), progress (1)."""
    vel = torch.cat([(pb - prev_pb) / dt, geo.quat_error_rotvec(prev_qb, qb) / dt], -1)
    lead = torch.cat([tgt_pb - pb, geo.quat_error_rotvec(qb, tgt_qb)], -1)
    tp, tx, tz = _axes_in(pb, qb, target_pb, target_qb)
    cp, cq = geo.frame_relative(pb, qb, contact_pb, contact_qb)
    neigh = neighbors(other_pb, other_qb, other_valid, target_pb, target_qb)
    return torch.cat([pb, rot6d(qb), vel, lead, tp, tx, tz, target_pb, dir_b, along[:, None], lateral[:, None], dyaw[:, None],
                      cp, geo.rotvec_from_quat(cq), neigh, z0_world[:, None], prev_actions, progress[:, None]], -1)
