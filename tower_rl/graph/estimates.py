"""Robust pose estimates from perceived block poses (Step 8.1), shared by the executor's nodes.

Perceived poses carry per-call jitter, one-call outliers and dropouts (stale poses with confidence 0), so no decision
(goal, pre-grasp target, held-block offset, place verdict) may hinge on a single sample. Pure numpy; on simulator truth
every sample agrees and the estimates equal the latest pose.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

CONF_MIN = 0.5  # samples below this confidence (a dropout's stale pose) are ignored


def yaw_of(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _flip_yaw(q: np.ndarray) -> np.ndarray:
    """``q * Rz(pi)``: the same block pose (a block is symmetric under a half turn about its z axis)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([-z, y, -x, w], -1)


def align_to(qs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Each quaternion as the representative (sign and half-turn symmetry) closest to ``ref``."""
    qs = np.asarray(qs, dtype=float).reshape(-1, 4)
    ref = np.asarray(ref, dtype=float).reshape(4)
    fl = _flip_yaw(qs)
    qs = np.where((np.abs(fl @ ref) > np.abs(qs @ ref))[:, None], fl, qs)
    return qs * np.where(qs @ ref < 0, -1.0, 1.0)[:, None]


def median_pose(ps, qs) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise median position and orientation (orientations aligned to the latest sample, renormalised)."""
    ps = np.asarray(ps, dtype=float).reshape(-1, 3)
    qs = align_to(qs, np.asarray(qs, dtype=float).reshape(-1, 4)[-1])
    q = np.median(qs, 0)
    return np.median(ps, 0), q / np.linalg.norm(q)


class StaticPose:
    """Estimate of a block that should be at rest: the median of its latest ``window`` mutually consistent samples.
    A sample farther than ``gate_m`` from the estimate (an outlier, or the block moving) is held back; ``confirm``
    consecutive held-back samples mean the block moved, and the latest of them start a new window."""

    def __init__(self, window: int = 10, gate_m: float = 0.004, confirm: int = 2):
        self.window, self.gate, self.confirm = window, gate_m, confirm
        self.ps: deque = deque(maxlen=window)
        self.qs: deque = deque(maxlen=window)
        self.pending: list = []
        self.est: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.ps)

    def add(self, pos, quat, conf: float = 1.0) -> None:
        if conf < CONF_MIN:
            return
        p, q = np.asarray(pos, dtype=float).copy(), np.asarray(quat, dtype=float).copy()
        if self.est is None or np.linalg.norm(p - self.est[0]) <= self.gate:
            self.pending = []
            self.ps.append(p)
            self.qs.append(q)
        else:
            self.pending.append((p, q))
            if len(self.pending) < self.confirm:
                return
            new = [(a, b) for a, b in self.pending if np.linalg.norm(a - p) <= self.gate]
            self.ps = deque([a for a, _ in new], maxlen=self.window)
            self.qs = deque([b for _, b in new], maxlen=self.window)
            self.pending = []
        self.est = median_pose(self.ps, self.qs)

    def pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self.est


class PoseWindow:
    """The latest confident samples of one block, for place verdicts: the median of a recent window, and a static
    test on the drift between the medians of two consecutive windows (a jitter-proof ``static``)."""

    def __init__(self, maxlen: int = 40):
        self.ps: deque = deque(maxlen=maxlen)
        self.qs: deque = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self.ps)

    def add(self, pos, quat, conf: float = 1.0) -> None:
        if conf >= CONF_MIN:
            self.ps.append(np.asarray(pos, dtype=float).copy())
            self.qs.append(np.asarray(quat, dtype=float).copy())

    def median(self, last: int, skip: int = 0) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.ps)
        lo, hi = max(0, n - skip - last), n - skip
        return median_pose(list(self.ps)[lo:hi], list(self.qs)[lo:hi])

    def drift(self, k: int) -> float | None:
        """Distance between the medians of the last ``k`` samples and the ``k`` before them (None: too few samples)."""
        if len(self.ps) < 2 * k:
            return None
        return float(np.linalg.norm(self.median(k)[0] - self.median(k, skip=k)[0]))
