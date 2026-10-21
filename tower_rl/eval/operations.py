"""Planned operations and their outcomes from simulator state (Step 3.1). Pure python/numpy, one tracker per env.

Our reconstruction of the organizers' (unpublished) operation semantics, from the docs and the data generator:
  * Plan: per transfer k (layout order: top layer first) one G_k then one P_k.
  * G_k succeeds when the target block has risen >= ``lift_m`` (2 cm) above its initial rest height while a closed
    gripper holds it (its pinch point within ``hold_radius_m`` of the block centre).
  * P_k succeeds when the block, released (no closed gripper near it), stays within ``tol_pos_mm`` (xy) / 10 mm (z,
    against the measured rest height of the goal layer) / ``tol_yaw_deg`` of its goal and static (speed < 1 cm/s)
    for ``settle_s``; blocks of lower destination layers must still be within tolerance of their goals.
    G/P outcomes latch once achieved (the docs count later successes after an earlier failure).
  * Operation windows (for disturbances): the tracker points at the first unresolved operation in plan order; it
    advances when that operation succeeds or when a later transfer's block is grasped (skip). A non-target movable
    block whose centre moves > 10 mm from its position at the window start counts once for that window; when the
    task fails, a final window covers the time after the last operation (ODR's "+1 - task_success" step).
  * Task success: every G and P succeeded and the final arrangement is within tolerance and static for ``settle_s``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tower_rl.eval.metrics import EpisodeOutcome


@dataclass
class OpCfg:
    lift_m: float = 0.02
    hold_radius_m: float = 0.03
    tol_pos_mm: float = 5.0
    tol_z_mm: float = 10.0
    tol_yaw_deg: float = 5.0
    settle_s: float = 1.0
    static_speed: float = 0.01
    disturb_mm: float = 10.0
    # Task 4/5 (protocols "..._structure_settled20"): the final arrangement must hold for the generator's
    # final_settle_seconds (2.0 s), and the remaining tower ("structure") must stay in place: no block moved more than
    # structure_mm, tilted more than collapse_deg or dropped more than collapse_drop_m (the generator's collapse rule)
    final_settle_s: float | None = None   # None: settle_s (Tasks 1-3)
    structure_mm: float = 10.0
    collapse_deg: float = 30.0
    collapse_drop_m: float = 0.015
    depart_m: float = 0.16                # Task 4: a pushed block's centre this far (xy) from the tower axis has left it,
    on_table_z: float = 0.02              # ... or one lying flat on the table (centre z) >= 0.09 m out (not leaning on it)


def _tilt(q_wxyz: np.ndarray) -> np.ndarray:
    """Angle (rad) between each block's z axis and the world z axis; ``q`` [..., 4] wxyz."""
    x, y = q_wxyz[..., 1], q_wxyz[..., 2]
    return np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))


def structure_intact(pos: np.ndarray, quat: np.ndarray, pos0: np.ndarray, ids: list[int], cfg: OpCfg) -> bool:
    """The blocks ``ids`` are where they started: moved <= structure_mm, tilted <= collapse_deg, not dropped."""
    if not ids:
        return True
    d = np.linalg.norm(pos[ids] - pos0[ids], axis=-1) * 1000
    drop = pos0[ids, 2] - pos[ids, 2]
    return bool((d <= cfg.structure_mm).all() and (np.degrees(_tilt(quat[ids])) <= cfg.collapse_deg).all()
                and (drop <= cfg.collapse_drop_m).all())


@dataclass
class Transfer:
    block: int            # index into the tracked block list
    arm: str
    goal_pos: np.ndarray  # [3] (z = measured rest height of the goal layer)
    goal_yaw: float
    dest_layer: int


def _yaw(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_err(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


@dataclass
class OperationTracker:
    variant: str
    transfers: list[Transfer]
    movable: list[int]
    cfg: OpCfg = field(default_factory=OpCfg)
    structure: list[int] = field(default_factory=list)  # Task 5: side and roof blocks that must stay in place

    def __post_init__(self):
        k = len(self.transfers)
        self.g_ok = [False] * k
        self.p_ok = [False] * k
        self.place_since = [None] * k
        self.pointer = 0  # index into the op sequence G0, P0, G1, P1, ...
        self.window_start = None
        self.window_disturbed: list[set] = []
        self.current_disturbed: set = set()
        self.z0 = None
        self.pos0 = None
        self.final_since = None
        self.task_success = False
        self.time = 0.0
        self.events: list[tuple[float, str]] = []

    # ------------------------------------------------------------------ helpers
    def _op(self, idx: int):
        return ("G" if idx % 2 == 0 else "P", idx // 2)

    def _ok(self, idx: int) -> bool:
        kind, k = self._op(idx)
        return self.g_ok[k] if kind == "G" else self.p_ok[k]

    def _target(self) -> int | None:
        if self.pointer >= 2 * len(self.transfers):
            return None
        return self.transfers[self.pointer // 2].block

    def _open_window(self, pos: np.ndarray):
        self.window_start = pos.copy()
        self.current_disturbed = set()

    def _close_window(self):
        self.window_disturbed.append(set(self.current_disturbed))

    def _in_goal(self, k: int, pos: np.ndarray, quat: np.ndarray) -> bool:
        t = self.transfers[k]
        p = pos[t.block]
        dxy = np.linalg.norm(p[:2] - t.goal_pos[:2]) * 1000
        dz = abs(p[2] - t.goal_pos[2]) * 1000
        return dxy <= self.cfg.tol_pos_mm and dz <= self.cfg.tol_z_mm and \
            math.degrees(_yaw_err(_yaw(quat[t.block]), t.goal_yaw)) <= self.cfg.tol_yaw_deg

    # ------------------------------------------------------------------ API
    def reset(self, pos: np.ndarray):
        self.__post_init__()
        self.z0 = pos[:, 2].copy()
        self.pos0 = pos.copy()
        self._open_window(pos)

    def update(self, t: float, pos: np.ndarray, quat: np.ndarray, speed: np.ndarray, pinch: np.ndarray, closed: np.ndarray):
        """``pos`` [B,3], ``quat`` [B,4] wxyz, ``speed`` [B] (m/s), ``pinch`` [A,3] per arm, ``closed`` [A] bool."""
        self.time = t
        held = np.zeros(len(pos), dtype=bool)
        for a in range(len(pinch)):
            if closed[a]:
                d = np.linalg.norm(pos[self.movable] - pinch[a], axis=-1)
                j = int(np.argmin(d))
                if d[j] <= self.cfg.hold_radius_m:
                    held[self.movable[j]] = True
        # grasp / place outcomes (latched)
        for k, tr in enumerate(self.transfers):
            if not self.g_ok[k] and held[tr.block] and pos[tr.block, 2] >= self.z0[tr.block] + self.cfg.lift_m:
                self.g_ok[k] = True
                self.events.append((t, f"G{k} ok"))
            if not self.p_ok[k]:
                lower_ok = all(self._in_goal(j, pos, quat) for j, o in enumerate(self.transfers) if o.dest_layer < tr.dest_layer and self.p_ok[j])
                if (not held[tr.block]) and speed[tr.block] < self.cfg.static_speed and self._in_goal(k, pos, quat) and lower_ok:
                    self.place_since[k] = self.place_since[k] if self.place_since[k] is not None else t
                    if t - self.place_since[k] >= self.cfg.settle_s - 1e-6:
                        self.p_ok[k] = True
                        self.events.append((t, f"P{k} ok"))
                else:
                    self.place_since[k] = None
        # advance the operation pointer: resolved op, or a later transfer's block grasped (skip)
        n_ops = 2 * len(self.transfers)
        while self.pointer < n_ops:
            later_grasp = next((2 * k for k in range(self.pointer // 2 + 1, len(self.transfers))
                                if held[self.transfers[k].block]), None)
            if self._ok(self.pointer):
                self._close_window()
                self.pointer += 1
                self._open_window(pos)
            elif later_grasp is not None:
                while self.pointer < later_grasp:
                    self._close_window()
                    self.pointer += 1
                    self._open_window(pos)
            else:
                break
        # disturbances in the current window: movable blocks other than the target that are not being held
        tgt = self._target()
        disp = np.linalg.norm(pos - self.window_start, axis=-1) * 1000
        for b in self.movable:
            if b != tgt and not held[b] and disp[b] > self.cfg.disturb_mm:
                self.current_disturbed.add(b)
        # task success: every op succeeded and the whole arrangement has been in tolerance and static for settle_s
        # (Task 5: final_settle_s, with the tower's side / roof blocks still in place)
        ok = all(self._in_goal(k, pos, quat) for k in range(len(self.transfers))) and \
            all(speed[b] < self.cfg.static_speed for b in self.movable) and not held.any() and \
            structure_intact(pos, quat, self.pos0, self.structure, self.cfg)
        self.final_since = (self.final_since if self.final_since is not None else t) if ok else None
        settle = self.cfg.final_settle_s or self.cfg.settle_s
        self.task_success = all(self.g_ok) and all(self.p_ok) and ok and (t - self.final_since >= settle - 1e-6)
        return self.task_success

    def outcome(self, duration_s: float, error: str | None = None) -> EpisodeOutcome:
        windows = list(self.window_disturbed)
        n_ops = 2 * len(self.transfers)
        # unresolved operations: the current window belongs to the pointer op; the rest are empty windows
        if self.pointer < n_ops:
            windows.append(set(self.current_disturbed))
            windows += [set()] * (n_ops - len(windows))
            final = set()
        else:
            final = set(self.current_disturbed)
        counts = [len(w) for w in windows[:n_ops]]
        if not self.task_success:
            counts.append(len(final))
        return EpisodeOutcome(self.variant, len(self.transfers), len(self.transfers), list(self.g_ok), list(self.p_ok), counts,
                              bool(self.task_success) and error is None, duration_s, error,
                              extra={"events": self.events, "disturbed_ids": [sorted(int(b) for b in w) for w in windows[:n_ops]]
                                     + ([sorted(int(b) for b in final)] if not self.task_success else [])})


@dataclass
class Push:
    block: int            # index into the tracked block list
    arm: str
    direction: np.ndarray  # [3] world push direction (unit)


@dataclass
class PushTracker:
    """Task 4 (protocol ``ordered_pushdown_preserved_structure_settled20``), our reconstruction from the docs and the
    generator's records (``push_departed`` once the block's support clearance exceeds 2 mm; ``outcome.completed``):

      * Plan: per push k (prescribed order: top layer first, row slot 2 -> 0) one G_k then one P_k.
      * G_k ("engaged"): every earlier push has succeeded and the target has moved >= ``lift_m`` (2 cm) along its push
        direction.
      * P_k ("departed"): every earlier push has succeeded, the target no longer touches the tower: its centre is
        >= ``depart_m`` (0.16 m: tower half-width + block half-length + 1 cm) from the tower axis, or it lies flat on
        the table (centre z <= 2 cm) >= 0.09 m out, and it has been static for ``settle_s``. (A block that tipped off
        one end may lean on the tower: the generator's "support clearance > 2 mm" does not count it.) A block that
        leaves the tower out of order breaks the order.
      * Structure: blocks not yet pushed must stay in place (moved <= structure_mm, the collapse rule); once broken the
        task fails (a disturbed tower is no longer the "required tower structure").
      * Disturbances (ODR): per operation window, tower blocks other than the target (not yet departed) moving > 10 mm;
        blocks that already left the tower are debris, not counted.
      * Task success: every push succeeded in order, the structure never broke, everything static for final_settle_s.
    """
    variant: str
    pushes: list[Push]
    movable: list[int]
    tower_center: np.ndarray  # [2]
    cfg: OpCfg = field(default_factory=OpCfg)

    def __post_init__(self):
        k = len(self.pushes)
        self.g_ok = [False] * k
        self.p_ok = [False] * k
        self.depart_since = [None] * k
        self.departed = set()      # tracked indices of blocks that left the tower (in or out of order)
        self.out_of_order = False
        self.structure_broken = False
        self.pointer = 0
        self.window_start = None
        self.window_disturbed: list[set] = []
        self.current_disturbed: set = set()
        self.pos0 = None
        self.final_since = None
        self.task_success = False
        self.time = 0.0
        self.events: list[tuple[float, str]] = []

    def _op(self, idx: int):
        return ("G" if idx % 2 == 0 else "P", idx // 2)

    def _ok(self, idx: int) -> bool:
        kind, k = self._op(idx)
        return self.g_ok[k] if kind == "G" else self.p_ok[k]

    def _target(self) -> int | None:
        return None if self.pointer >= 2 * len(self.pushes) else self.pushes[self.pointer // 2].block

    def _off_tower(self, pos: np.ndarray, b: int) -> bool:
        r = float(np.linalg.norm(pos[b, :2] - self.tower_center))
        return r >= self.cfg.depart_m or (r >= 0.09 and pos[b, 2] <= self.cfg.on_table_z)

    def reset(self, pos: np.ndarray):
        self.__post_init__()
        self.pos0 = pos.copy()
        self.window_start = pos.copy()

    def update(self, t: float, pos: np.ndarray, quat: np.ndarray, speed: np.ndarray, pinch: np.ndarray, closed: np.ndarray):
        self.time = t
        order = [p.block for p in self.pushes]
        for b in order:
            if b not in self.departed and self._off_tower(pos, b):
                self.departed.add(b)
                k = order.index(b)
                if not all(self.p_ok[:k]):
                    self.out_of_order = True
                    self.events.append((t, f"push {k} departed out of order"))
        for k, p in enumerate(self.pushes):
            earlier = all(self.p_ok[:k])
            moved = float(np.dot(pos[p.block] - self.pos0[p.block], p.direction))
            if not self.g_ok[k] and earlier and moved >= self.cfg.lift_m:
                self.g_ok[k] = True
                self.events.append((t, f"G{k} ok"))
            if not self.p_ok[k]:
                if earlier and self.g_ok[k] and p.block in self.departed and speed[p.block] < self.cfg.static_speed:
                    self.depart_since[k] = self.depart_since[k] if self.depart_since[k] is not None else t
                    if t - self.depart_since[k] >= self.cfg.settle_s - 1e-6:
                        self.p_ok[k] = True
                        self.events.append((t, f"P{k} ok"))
                else:
                    self.depart_since[k] = None
        # the tower that remains (blocks not yet departed and not the current target) must stay in place
        tgt = self._target()
        remaining = [b for b in order if b not in self.departed and b != tgt]
        if not self.structure_broken and not structure_intact(pos, quat, self.pos0, remaining, self.cfg):
            self.structure_broken = True
            self.events.append((t, "structure broken"))
        n_ops = 2 * len(self.pushes)
        while self.pointer < n_ops:
            later = next((2 * k for k in range(self.pointer // 2 + 1, len(self.pushes)) if self.g_ok[k]), None)
            if self._ok(self.pointer):
                self.window_disturbed.append(set(self.current_disturbed))
                self.pointer += 1
                self.window_start = pos.copy()
                self.current_disturbed = set()
            elif later is not None:
                while self.pointer < later:
                    self.window_disturbed.append(set(self.current_disturbed))
                    self.pointer += 1
                    self.window_start = pos.copy()
                    self.current_disturbed = set()
            else:
                break
        tgt = self._target()
        disp = np.linalg.norm(pos - self.window_start, axis=-1) * 1000
        for b in order:
            if b != tgt and b not in self.departed and disp[b] > self.cfg.disturb_mm:
                self.current_disturbed.add(b)
        ok = all(self.p_ok) and not self.out_of_order and not self.structure_broken and \
            all(speed[b] < self.cfg.static_speed for b in self.movable)
        self.final_since = (self.final_since if self.final_since is not None else t) if ok else None
        settle = self.cfg.final_settle_s or self.cfg.settle_s
        self.task_success = ok and (t - self.final_since >= settle - 1e-6)
        return self.task_success

    def outcome(self, duration_s: float, error: str | None = None) -> EpisodeOutcome:
        windows = list(self.window_disturbed)
        n_ops = 2 * len(self.pushes)
        if self.pointer < n_ops:
            windows.append(set(self.current_disturbed))
            windows += [set()] * (n_ops - len(windows))
            final = set()
        else:
            final = set(self.current_disturbed)
        counts = [len(w) for w in windows[:n_ops]]
        if not self.task_success:
            counts.append(len(final))
        return EpisodeOutcome(self.variant, len(self.pushes), len(self.pushes), list(self.g_ok), list(self.p_ok), counts,
                              bool(self.task_success) and error is None, duration_s, error,
                              extra={"events": self.events, "structure_broken": self.structure_broken,
                                     "out_of_order": self.out_of_order,
                                     "disturbed_ids": [sorted(int(b) for b in w) for w in windows[:n_ops]]
                                     + ([sorted(int(b) for b in final)] if not self.task_success else [])})
