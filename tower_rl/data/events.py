"""Gripper close/open events and pick-and-place operations in recorded episodes (Step 1).

Events come from the commanded opening width (``leader/qpos[:, 7]`` left, ``[:, 15]`` back) with hysteresis
(< 0.06 m closed, > 0.09 m open). The demonstrations close to ~0.044 m (2.4 N per finger on the 0.0497 m block),
relax to ~0.046 m for ~1 s at release and then open to 0.1 m, so a close -> open pair is one operation. Each
operation is matched to the movable block that moved most while the gripper was closed and cross-checked with
``sim/active_block`` and the measured width (``observations/gripper_status``: width, force, enabled, 0 per arm).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from tower_rl.data.episode import ARMS, Episode

CLOSE_BELOW = 0.06
OPEN_ABOVE = 0.09
WIDTH_COLUMN = {"left": 7, "back": 15}
STATUS_COLUMNS = {"left": slice(0, 4), "back": slice(4, 8)}


@dataclass
class Operation:
    arm: str
    close_frame: int
    open_frame: int  # -1 when the gripper never re-opened
    block_index: int | None = None
    block_id: str | None = None
    active_block_id: str | None = None  # sim/active_block shortly after the close
    held_width_m: float | None = None  # measured opening while holding
    lift_m: float | None = None  # max rise of the block above its start while the gripper was closed
    displacement_m: float | None = None  # block centre displacement close -> open

    def to_dict(self) -> dict:
        return asdict(self)


def hysteresis_events(width: np.ndarray, close_below: float = CLOSE_BELOW, open_above: float = OPEN_ABOVE) -> list[tuple[int, str]]:
    """``[(frame, 'close'|'open'), ...]`` transitions of a width signal with hysteresis."""
    closed = bool(width[0] < close_below)
    out = []
    for t, w in enumerate(width):
        if not closed and w < close_below:
            closed = True
            out.append((t, "close"))
        elif closed and w > open_above:
            closed = False
            out.append((t, "open"))
    return out


def operations(ep: Episode, close_below: float = CLOSE_BELOW, open_above: float = OPEN_ABOVE) -> list[Operation]:
    """Close -> open operations of both arms, sorted by close time, each matched to a block."""
    poses = ep.block_poses
    movable = [i for i, fixed in enumerate(ep.block_fixed) if not fixed]
    active = ep.arrays.get("sim__active_block")
    status = ep.arrays.get("observations__gripper_status")
    ops = []
    for arm in ARMS:
        ev = hysteresis_events(ep.qpos_cmd[:, WIDTH_COLUMN[arm]], close_below, open_above)
        pending = None
        for t, kind in ev:
            if kind == "close":
                pending = t
            elif kind == "open" and pending is not None:
                ops.append(Operation(arm, pending, t))
                pending = None
        if pending is not None:
            ops.append(Operation(arm, pending, -1))
    ops.sort(key=lambda o: o.close_frame)
    T = ep.num_frames
    for op in ops:
        t0, t1 = op.close_frame, (op.open_frame if op.open_frame >= 0 else T - 1)
        seg = poses[t0: t1 + 1][:, movable, :3]
        disp = np.linalg.norm(seg[-1] - seg[0], axis=-1)
        rise = seg[..., 2].max(0) - seg[0, :, 2]
        k = int(np.argmax(disp + rise))
        op.block_index = movable[k]
        op.block_id = ep.block_ids[op.block_index]
        op.lift_m = float(rise[k])
        op.displacement_m = float(disp[k])
        if active is not None:
            a = int(active[min(t0 + 15, T - 1)])
            op.active_block_id = ep.block_ids[a] if a >= 0 else None
        if status is not None:
            mid = (t0 + t1) // 2
            op.held_width_m = float(status[mid, STATUS_COLUMNS[op.arm]][0])
    return ops


def expected_operation_count(variant: str) -> int:
    """Task1: two blocks per layer; Task2 and Task3: one block per layer."""
    task, layers = variant.split("_")[0].replace("task", "").split("-")
    return int(layers) * (2 if task == "1" else 1)


def first_lift_frame(ep: Episode, op: Operation, rise_m: float = 0.001) -> int:
    """First frame after the close at which the operation's block rose ``rise_m`` above its pose at the close."""
    z = ep.block_poses[:, op.block_index, 2]
    t1 = op.open_frame if op.open_frame >= 0 else ep.num_frames - 1
    above = np.nonzero(z[op.close_frame: t1] > z[op.close_frame] + rise_m)[0]
    return op.close_frame + int(above[0]) if len(above) else op.close_frame
