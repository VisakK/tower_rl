"""The official evaluation as we understand it after official run 1 (Plan-v1.1 §1, §2.2): the rules as code, shared by
the local evaluator (simulator truth, ``tower_rl.eval.operations``), the re-scoring of the official server logs
(estimated poses, ``scripts/official/``) and the student's contracts. Pure numpy.

* Task 4 rest states. Each pushed block is judged where it comes to rest: "flat clear" when it lies on the table (centre
  z < ``TABLE_Z_M``, tilt < ``FLAT_TILT_DEG``) at least ``CLEAR_M`` outside the foundation's 150 x 150 mm footprint;
  otherwise leaning on / on the base, touching the base, tipped (on its side), on another block, off the table, or
  unseen. Applied to our final estimates this reproduces the site's Task 4-2 LHPR exactly (44 / 60) and 4-3 within 3.4
  points (official run 1, ``scripts/official/task4_outcomes.py``). The footprint is taken in the foundation's frame
  (yaw from its blocks; the shifted-assembly case turns the tower up to 10 deg).
* Points. A least-squares fit to the scorecard of official run 1 (residuals <= 0.8): task score = 0.34 TSR + 0.62
  LHPR + 3.7 (1 - ODR), TSR and LHPR in %; points per variant = score x weight / 18, weights 1 : 1 : 1 : 3 : 3 by task.
"""

from __future__ import annotations

import math

import numpy as np

HALF_L, HALF_W, HEIGHT = 0.075, 0.02488, 0.0283  # block hull half-length, half-width, height (m)
FOOT_HALF = 0.075        # half-width of the foundation footprint (three blocks side by side)
CLEAR_M = 0.002          # a pushed block must be this far outside the footprint ("support clearance > 2 mm")
TABLE_Z_M = 0.025        # centre height below which a block lies on the table (flat 14.7 mm, on its side 24.9 mm)
FLAT_TILT_DEG = 20.0
OFF_TABLE_Z_M = -0.02    # fell off the table
WEIGHTS = {"task1": 1, "task2": 1, "task3": 1, "task4": 3, "task5": 3}
FLAT = "flat clear"


def rot(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def footprint_frame(found_pos: np.ndarray, found_quat: np.ndarray) -> tuple[np.ndarray, float]:
    """(centre [2], yaw) of the foundation footprint from its blocks' poses [3, 3] / [3, 4] (the square is symmetric
    under quarter turns: the yaw is the blocks' mean long-axis direction mod 90 deg)."""
    c = np.asarray(found_pos, dtype=float)[:, :2].mean(0)
    ang = []
    for q in found_quat:
        R = rot(q)
        ang.append(math.atan2(R[1, 0], R[0, 0]))
    yaw = 0.25 * math.atan2(sum(math.sin(4 * a) for a in ang), sum(math.cos(4 * a) for a in ang))
    return c, yaw


def corners(p, q) -> np.ndarray:
    R = rot(q)
    return np.array([np.asarray(p, dtype=float) + R @ np.array([sx * HALF_L, sy * HALF_W, sz * HEIGHT / 2])
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def gap_to_footprint(p, q, centre: np.ndarray, yaw: float = 0.0) -> float:
    """Smallest horizontal distance (m) of the block's hull corners outside the footprint; 0 if any corner is over it."""
    c = corners(p, q)[:, :2] - centre
    cy, sy = math.cos(yaw), math.sin(yaw)
    lx, ly = c[:, 0] * cy + c[:, 1] * sy, -c[:, 0] * sy + c[:, 1] * cy
    dx = np.maximum(np.abs(lx) - FOOT_HALF, 0)
    dy = np.maximum(np.abs(ly) - FOOT_HALF, 0)
    return float(np.min(np.hypot(dx, dy)))


def tilt_deg(q) -> float:
    return math.degrees(math.acos(min(1.0, abs(rot(q)[2, 2]))))


def rest_state(p, q, centre: np.ndarray, yaw: float = 0.0, conf: float = 1.0) -> tuple[str, float, float, float]:
    """(kind, gap_mm, tilt_deg, z_mm) of one pushed block's pose relative to the foundation footprint."""
    gap = 1000 * gap_to_footprint(p, q, centre, yaw)
    tilt = tilt_deg(q)
    z = 1000 * float(p[2])
    if conf < 0.5:
        kind = "unseen"
    elif z < 1000 * OFF_TABLE_Z_M:
        kind = "off the table"
    elif gap < 1000 * CLEAR_M:
        kind = "on/leaning on base" if z > 1000 * TABLE_Z_M else "touching base"
    elif tilt > FLAT_TILT_DEG:
        kind = "tipped"
    elif z > 1000 * TABLE_Z_M:
        kind = "on a block"
    else:
        kind = FLAT
    return kind, gap, tilt, z


def task_score(tsr: float, lhpr: float, odr: float) -> float:
    """The fitted task score (0-100) from TSR, LHPR in [0, 1] and ODR."""
    return 0.34 * 100 * tsr + 0.62 * 100 * lhpr + 3.7 * (1 - odr)


def points(variant: str, tsr: float, lhpr: float, odr: float) -> float:
    return task_score(tsr, lhpr, odr) * WEIGHTS[variant.split("-")[0]] / 18.0
