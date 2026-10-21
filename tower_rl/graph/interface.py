"""Typed interface between the executor, the skills and the world (Step 7.1). Pure python/numpy.

``State`` has one schema for simulator truth and for estimated poses (``block_conf`` is 1.0 for simulator truth and
reserved for the pose estimator). Everything a deployed policy may read is in the top-level fields; privileged
simulator quantities live in ``privileged`` and must only be used by critics, checkpoints during training and the
local evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

ARMS = ("left", "back")


@dataclass
class State:
    t: float                    # simulated time (s)
    step: int                   # control actions executed so far
    variant: str
    instruction: str
    q: np.ndarray               # [2, 7] measured joint angles, left then back
    width: np.ndarray           # [2] measured gripper opening (m)
    block_ids: list[str]
    block_pos: np.ndarray       # [B, 3] station frame (m)
    block_quat: np.ndarray      # [B, 4] wxyz
    block_conf: np.ndarray      # [B] 1.0 = simulator truth; estimator confidence later
    privileged: dict = field(default_factory=dict)  # e.g. block_vel, finger contacts (evaluator/critic only)

    def block(self, bid: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.block_ids.index(bid)
        return self.block_pos[i], self.block_quat[i]


@dataclass
class Pose:
    pos: np.ndarray   # [3]
    quat: np.ndarray  # [4] wxyz


@dataclass
class Tol:
    pos_m: float = 0.005
    yaw_deg: float = 5.0


@dataclass
class SkillCall:
    skill: str                  # transit | grasp_lift | place_release | park
    arm: str
    block_id: str | None = None
    goal: Pose | None = None
    tolerance: Tol = field(default_factory=Tol)
    timeout_s: float = 20.0
    params: dict = field(default_factory=dict)


class Status(str, Enum):
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SkillResult:
    status: Status
    pose_error: float | None = None
    disturbance_flag: bool = False
    duration_s: float = 0.0
    info: dict = field(default_factory=dict)
