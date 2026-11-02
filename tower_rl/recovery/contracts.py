"""Contracts (Plan-v1.1 §5.2): each goal's postcondition and the structure rule of the official evaluation as we
reconstructed it (``tower_rl.eval.mirror``), as deterministic checks on a scene. The teacher's recovery node judges
them on simulator truth, the student's on its pose estimates; the code is the same. Pure numpy.

Per family, goal k's check (``Check``):
* pushdown (``push_off(k)`` ensures: block k at rest on the table, clear of the tower, not on debris): ``kind`` is
  "on tower" while the block has not left the tower (the radial test), else its rest kind (``mirror.rest_state``:
  "flat clear", "on/leaning on base", "touching base", "tipped", "on a block", "off the table", "unseen"); ok = flat
  clear. The remaining structure: blocks of later goals within 10 mm of their start (and not tilted / dropped).
* insertion (``insert(k)`` + ``push_home(k)`` ensure: the insert at its slot goal, the destination structure in place):
  the insert's offset from its slot goal (the midpoint of its layer's two side blocks at the start, the layout's goal
  height), split along the slot axis (``along_mm`` < 0: short of home, on the entry side), across and in height;
  ``kind`` "home" (within the tracker's 5 mm / 10 mm), "short" (on the slot line and in height, not deep enough: a
  push-in finishes it), "out" (elsewhere), "unseen". The structure: the side and roof blocks within 10 mm.
* transfer: within 5 mm / 5 deg / 10 mm of the goal given by ``goal_fn(k)`` (the runtime's ``dest_goal``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tower_rl.eval import mirror

STRUCTURE_MM = 10.0
COLLAPSE_DEG = 30.0
COLLAPSE_DROP_M = 0.015
INSERT_TOL_MM, INSERT_TOL_Z_MM = 5.0, 10.0  # the operation tracker's place tolerance (Task 5 official: <= ~4 mm)
SHORT_LAT_MM, SHORT_MAX_MM = 6.0, 120.0     # "short": on the slot line within this, at most this far out
PLACE_TOL_MM, PLACE_TOL_DEG, PLACE_TOL_Z_MM = 5.0, 5.0, 10.0
CONF_SEEN = 0.5


@dataclass
class Check:
    k: int
    ok: bool
    kind: str
    detail: dict = field(default_factory=dict)


def _tilt_deg(q) -> float:
    return mirror.tilt_deg(q)


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Contracts:
    """``spec``: the variant's ``TaskSpec``; ``block_ids`` the scene's block order; ``pos0`` / ``quat0`` the start
    poses (truth for the teacher, the first rest estimates for the student)."""

    def __init__(self, spec, block_ids: list[str], pos0: np.ndarray, quat0: np.ndarray, goal_fn=None):
        self.spec, self.ids = spec, list(block_ids)
        self.pos0, self.quat0 = np.asarray(pos0, dtype=float).copy(), np.asarray(quat0, dtype=float).copy()
        self.goal_fn = goal_fn
        self.family = spec.family
        self.idx = {b: i for i, b in enumerate(self.ids)}
        self.blocks = [self.idx[g.block] for g in spec.goals]
        if self.family == "pushdown":
            found = [self.idx[b] for b in spec.tower["foundation"]]
            self.foot = mirror.footprint_frame(self.pos0[found], self.quat0[found])
            self.tower_xy = self.pos0[found, :2].mean(0)
        if self.family == "insertion":
            self.structure_ids = [self.idx[b] for b in spec.tower.get("structure", [])]
            self.slots = []
            for g in spec.goals:
                sides = [self.idx[b] for b in (g.side_blocks or [])]
                c = self.pos0[sides].mean(0) if len(sides) == 2 else None
                if c is not None:
                    c = c.copy()
                    c[2] = float(g.goal_z)
                yaws = [_yaw(self.quat0[i]) for i in sides]
                ya = 0.5 * math.atan2(sum(math.sin(2 * y) for y in yaws), sum(math.cos(2 * y) for y in yaws)) if yaws else 0.0
                ax = np.array([math.cos(ya), math.sin(ya), 0.0])
                d = np.asarray(g.direction or [1.0, 0.0, 0.0], dtype=float)
                if float(ax @ d) < 0:
                    ax = -ax
                self.slots.append((c, ax))

    # ------------------------------------------------------------------ structure
    def moved_mm(self, pos: np.ndarray, ids: list[int]) -> float:
        if not ids:
            return 0.0
        return 1000 * float(np.linalg.norm(pos[ids] - self.pos0[ids], axis=-1).max())

    def structure(self, pos: np.ndarray, quat: np.ndarray, after_goal: int | None = None) -> tuple[bool, float]:
        """(intact, largest move mm): pushdown: the blocks of goals after ``after_goal`` (all when None) still in place;
        insertion: the side and roof blocks; transfer: always intact (0)."""
        if self.family == "pushdown":
            ids = [self.blocks[k] for k in range(len(self.blocks)) if after_goal is None or k > after_goal]
        elif self.family == "insertion":
            ids = self.structure_ids
        else:
            return True, 0.0
        if not ids:
            return True, 0.0
        mm = self.moved_mm(pos, ids)
        tilt = max(_tilt_deg(quat[i]) for i in ids)
        drop = float((self.pos0[ids, 2] - pos[ids, 2]).max())
        return bool(mm <= STRUCTURE_MM and tilt <= COLLAPSE_DEG and drop <= COLLAPSE_DROP_M), round(mm, 1)

    # ------------------------------------------------------------------ goals
    def check(self, k: int, pos: np.ndarray, quat: np.ndarray, conf: np.ndarray | None = None) -> Check:
        b = self.blocks[k]
        c = 1.0 if conf is None else float(conf[b])
        if self.family == "pushdown":
            r = float(np.linalg.norm(pos[b, :2] - self.tower_xy))
            if c >= CONF_SEEN and float(np.linalg.norm(pos[b] - self.pos0[b])) < 0.02:  # not pushed yet
                return Check(k, False, "on tower", {"r_mm": round(1000 * r, 1)})
            kind, gap, tilt, z = mirror.rest_state(pos[b], quat[b], *self.foot, conf=c)
            return Check(k, kind == mirror.FLAT, kind, {"gap_mm": round(gap, 1), "tilt_deg": round(tilt, 1),
                                                        "z_mm": round(z, 1), "r_mm": round(1000 * r, 1)})
        if self.family == "insertion":
            centre, ax = self.slots[k]
            if centre is None:
                return Check(k, False, "no slot", {})
            if c < CONF_SEEN:
                return Check(k, False, "unseen", {})
            off = pos[b] - centre
            lat_ax = np.array([-ax[1], ax[0], 0.0])
            along, lat, dz = 1000 * float(off @ ax), 1000 * float(off @ lat_ax), 1000 * float(off[2])
            xy = math.hypot(along, lat)
            det = {"along_mm": round(along, 1), "lat_mm": round(lat, 1), "dz_mm": round(dz, 1), "xy_mm": round(xy, 1)}
            if xy <= INSERT_TOL_MM and abs(dz) <= INSERT_TOL_Z_MM:
                return Check(k, True, "home", det)
            if abs(lat) <= SHORT_LAT_MM and abs(dz) <= INSERT_TOL_Z_MM and -SHORT_MAX_MM <= along < 0:
                return Check(k, False, "short", det)
            return Check(k, False, "out", det)
        # transfer
        if self.goal_fn is None:
            return Check(k, False, "no goal", {})
        gp, gyaw = self.goal_fn(k)
        if c < CONF_SEEN:
            return Check(k, False, "unseen", {})
        dxy = 1000 * float(np.linalg.norm(pos[b, :2] - gp[:2]))
        dz = 1000 * abs(float(pos[b, 2] - gp[2]))
        dyaw = abs((_yaw(quat[b]) - gyaw + math.pi / 2) % math.pi - math.pi / 2)
        ok = dxy <= PLACE_TOL_MM and dz <= PLACE_TOL_Z_MM and math.degrees(dyaw) <= PLACE_TOL_DEG
        return Check(k, ok, "in tolerance" if ok else "off goal", {"xy_mm": round(dxy, 1), "z_mm": round(dz, 1),
                                                                  "yaw_deg": round(math.degrees(dyaw), 2)})

    def all(self, pos, quat, conf=None) -> list[Check]:
        return [self.check(k, pos, quat, conf) for k in range(len(self.blocks))]
