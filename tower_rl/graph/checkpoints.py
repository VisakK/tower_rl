"""Checkpoint predicates on ``State`` (Step 7.2), shared by the executor's nodes.

Inputs are deployable only (joint angles, gripper widths, block poses, the executor's own records), so the same
predicates run on simulator truth now and on estimated poses later. Thresholds come from the graph nodes and configs.
"""

from __future__ import annotations

import math

import numpy as np

from tower_rl.graph.interface import ARMS, State

ARM_CLEAR_RADIUS_M = 0.18  # other arm's pinch beyond this (xy) from a transit target; home poses are >= 0.23 m from a pad


def holding(state: State, arm: str, held_width: tuple[float, float]) -> bool:
    """The measured opening is a block's width (fingers on a block, neither empty nor open)."""
    lo, hi = held_width
    return lo <= float(state.width[ARMS.index(arm)]) <= hi


def gripping(state: State, arm: str, block_width: float = 0.0497, tol_m: float = 0.0025) -> bool:
    """The fingers are on a block's faces (opening = the block width), not just closing through the holding range."""
    return abs(float(state.width[ARMS.index(arm)]) - block_width) <= tol_m


def lifted(state: State, block: str, z0: float, lift_m: float) -> bool:
    return float(state.block(block)[0][2]) - z0 >= lift_m


def grasped(state: State, arm: str, block: str, z0: float, lift_m: float, held_width: tuple[float, float]) -> bool:
    """Grasp checkpoint on estimable state: the block rose ``lift_m`` while the gripper holds a block."""
    return holding(state, arm, held_width) and lifted(state, block, z0, lift_m)


def yaw_err_deg(q1_wxyz, q2_wxyz) -> float:
    """Yaw difference of two block orientations, pi-symmetric (deg)."""
    def yaw(q):
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    d = (yaw(q1_wxyz) - yaw(q2_wxyz) + math.pi / 2) % math.pi - math.pi / 2
    return abs(math.degrees(d))


def placement_error(block_pos, block_quat, goal_pos, goal_quat) -> tuple[float, float, float]:
    """(xy error m, |z error| m, yaw error deg) of a block against its goal."""
    bp, gp = np.asarray(block_pos, dtype=float), np.asarray(goal_pos, dtype=float)
    return float(np.linalg.norm(bp[:2] - gp[:2])), float(abs(bp[2] - gp[2])), yaw_err_deg(block_quat, goal_quat)


def placed(block_pos, block_quat, goal_pos, goal_quat, tol_pos_m: float, tol_yaw_deg: float, tol_z_m: float) -> bool:
    dxy, dz, dyaw = placement_error(block_pos, block_quat, goal_pos, goal_quat)
    return dxy <= tol_pos_m and dyaw <= tol_yaw_deg and dz <= tol_z_m


def static(pos_now, pos_prev, dt: float, speed: float) -> bool:
    """Mean speed over the last ``dt`` below ``speed`` (displacement-based, as the place_release env)."""
    return float(np.linalg.norm(np.asarray(pos_now, dtype=float) - np.asarray(pos_prev, dtype=float))) / dt < speed


def max_disturbance_m(state: State, ids: list[str], p0: np.ndarray | None) -> float:
    """Largest displacement of the watched blocks since ``p0`` (the 10 mm no-disturbance rule)."""
    if p0 is None or not ids:
        return 0.0
    return float(np.linalg.norm(np.stack([state.block(b)[0] for b in ids]) - p0, axis=-1).max())


def arm_clear(pinch_world, target_xy, radius_m: float = ARM_CLEAR_RADIUS_M) -> bool:
    """An arm is clear of a tower when its pinch is outside the vertical cylinder of ``radius_m`` around it."""
    p = np.asarray(pinch_world, dtype=float).ravel()
    return float(np.linalg.norm(p[:2] - np.asarray(target_xy, dtype=float)[:2])) > radius_m
