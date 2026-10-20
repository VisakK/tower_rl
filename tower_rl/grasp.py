"""Canonical pinch poses relative to blocks (pure torch): grasp, pre-grasp, place and pre-place targets.

The demonstrations grasp with the jaw axis along the block's y axis (its 0.05 m width), the pinch point at the block
centre and the approach axis tilted about the jaw axis so the finger tips lean away from the arm base (left arm 12 deg,
back arm 5.5 deg; ``configs/skills/grasp_geometry.yaml``). Blocks are symmetric under a 180 deg yaw, so the jaw sign is
chosen to face the arm's natural wrist orientation (jaw = base->block direction turned -90 deg about z), which keeps
joint 7 away from its +-90 deg limits.
"""

from __future__ import annotations

import math

import torch

from tower_rl import geometry as geo
from tower_rl.config import load_yaml
from tower_rl.kinematics import load_model

_CFG = None


def cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_yaml("skills/grasp_geometry")
    return _CFG


def base_xy(arm: str, device="cpu") -> torch.Tensor:
    return torch.tensor(load_model()["base_world"][arm]["pos"][:2], dtype=torch.float32, device=device)


def gripper_quat(block_quat: torch.Tensor, block_pos: torch.Tensor, arm: str, tilt_deg: float | None = None) -> torch.Tensor:
    """Pinch orientation ``[N, 4]`` for grasping blocks with orientations ``block_quat`` at ``block_pos`` (station)."""
    tilt = math.radians(cfg()["approach_tilt_deg"][arm] if tilt_deg is None else tilt_deg)
    dev = block_quat.device
    r_b = geo.quat_to_mat(block_quat)
    jaw = r_b[..., :, 1].clone()
    jaw[..., 2] = 0.0
    jaw = jaw / jaw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    facing = block_pos[..., :2] - base_xy(arm, dev)
    facing = facing / facing.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    ref = torch.stack([facing[..., 1], -facing[..., 0]], -1)  # facing turned -90 deg
    sign = torch.where((jaw[..., :2] * ref).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
    jaw = jaw * sign
    # horizontal "away" direction perpendicular to the jaw
    away = torch.cat([facing, torch.zeros_like(facing[..., :1])], -1)
    away = away - (away * jaw).sum(-1, keepdim=True) * jaw
    away = away / away.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand_as(away)
    approach = math.cos(tilt) * down + math.sin(tilt) * away
    x = torch.cross(jaw, approach, dim=-1)
    return geo.mat_to_quat(torch.stack([x, jaw, approach], dim=-1))


def grasp_relation_error(block_pos: torch.Tensor, block_quat: torch.Tensor, pinch_pos: torch.Tensor, pinch_quat: torch.Tensor,
                         arm: str, tilt_deg: float | None = None):
    """Grasp quality, invariant to the block's tilt in the world (a lifted block may hang tilted): the pinch pose in the
    block frame against the canonical grasp of the same block lying flat. Returns the pinch offset from the canonical one
    in the block frame ``[N, 3]`` (x along the block, y across it, z up) and the rotation ``[N]`` (rad) between the actual
    and the canonical pinch orientations in the block frame; the fingers are interchangeable, so a pi turn about the
    approach axis is the same grasp."""
    dev, dt = block_quat.device, block_quat.dtype
    flat = geo.quat_from_yaw(geo.yaw_from_quat(block_quat))
    can = geo.quat_mul(geo.quat_conj(flat), gripper_quat(flat, block_pos, arm, tilt_deg))
    act = geo.quat_mul(geo.quat_conj(block_quat), pinch_quat)
    off = torch.tensor(cfg()["pinch_offset_in_block_m"], dtype=dt, device=dev)
    d = geo.quat_apply(geo.quat_conj(block_quat), pinch_pos - block_pos) - off
    flip = geo.quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], dtype=dt, device=dev).expand_as(act[..., :3]),
                                    torch.full_like(act[..., 0], math.pi))
    rot = torch.minimum(geo.quat_angle(act, can), geo.quat_angle(geo.quat_mul(act, flip), can))
    return d, rot


def grasp_pose(block_pos: torch.Tensor, block_quat: torch.Tensor, arm: str, tilt_deg: float | None = None):
    """Pinch target ``(pos, quat)`` that grasps the block centred."""
    off = torch.tensor(cfg()["pinch_offset_in_block_m"], dtype=block_pos.dtype, device=block_pos.device)
    q = gripper_quat(block_quat, block_pos, arm, tilt_deg)
    return block_pos + geo.quat_apply(block_quat, off.expand_as(block_pos)), q


def above(pos: torch.Tensor, quat: torch.Tensor, height: float):
    """Same orientation, ``height`` higher (world z)."""
    p = pos.clone()
    p[..., 2] += height
    return p, quat


def place_pose(goal_pos: torch.Tensor, goal_quat: torch.Tensor, arm: str, tilt_deg: float | None = None):
    """Pinch target that sets a centred-grasped block down at its goal (pinch at the goal centre)."""
    return grasp_pose(goal_pos, goal_quat, arm, tilt_deg)


def gripper_cfg():
    """The shared gripper command profile (``grasp_geometry.yaml: gripper``)."""
    from tower_rl.control import GripperCfg
    return GripperCfg(**cfg()["gripper"])
