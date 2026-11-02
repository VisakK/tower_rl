"""Estimator noise (student plan Step 5): makes simulator block poses look like the pose estimator's output.

Fitted by ``scripts/fit_estimator_noise.py`` from the camera-loop logs into ``configs/student/noise.yaml`` and a
trace bank (``results/student/noise/traces.npz``). Per call and block, a class (as in the fit):
  foundation | fallen (truth z < 0) | held (fingers within 3 cm, closed below 55 mm) | target:<skill> (the block of
  the policy's running node) | moving (truth speed > 1 cm/s) | static
selects a bank of recorded error traces (position error, rotation-error vector, confidence level), replayed call by
call until the trace ends or the class changes (foundation and static blocks then keep their last error: the estimator
freezes foundations and anchors resting blocks). A "not seen" call (0.3) returns the previous output, coasted for up
to 5 calls (x0.8 per call) when the block was moving; a ghost (0) keeps the previous output. Push targets take their
losses from the fitted P(lost | tilt, travel) with recorded run lengths. Outputs keep quaternion continuity (the box
symmetry nearest the previous output). The truth stays in ``privileged`` (``true_block_pos`` / ``true_block_quat``).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from tower_rl.config import load_yaml
from tower_rl.graph.interface import ARMS, State
from tower_rl.paths import REPO_ROOT

CLASSES = ("foundation", "static", "held", "moving", "fallen", "target:transit", "target:grasp_lift",
           "target:place_release", "target:push", "target:insert", "target:push_in", "target:park")
MIN_TRACE = 5
MAX_TRACE = {c: (2000 if c in ("foundation", "static") else 100) for c in CLASSES}
LEVELS = (1.0, 0.6, 0.3, 0.0)
LOST_TILT_BINS = (0.0, 10.0, 30.0, 60.0, 181.0)
LOST_ALONG_BINS = (0.0, 0.03, 0.06, 0.10, 10.0)
SYM = [np.diag(d).astype(np.float64) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
HOLD_WIDTH, HOLD_RADIUS = 0.055, 0.03


def conf_index(c: float) -> int:
    return 0 if c >= 0.95 else 1 if c >= 0.5 else 2 if c > 1e-6 else 3


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """wxyz of rotation matrices [..., 3, 3] (Shepperd)."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    out = np.zeros((len(R), 4))
    for i, m in enumerate(R):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            out[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            out[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            out[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            out[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out.reshape(*shp, 4)


def rotvec_of(R: np.ndarray) -> np.ndarray:
    """Rotation vectors [..., 3] of rotation matrices (log map)."""
    R = np.asarray(R, dtype=np.float64)
    c = np.clip((np.einsum("...ii->...", R) - 1) / 2, -1.0, 1.0)
    ang = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(ang)
    k = np.where(s[..., None] > 1e-9, ang[..., None] / (2 * np.where(s > 1e-9, s, 1.0))[..., None], 0.5)
    return v * k


def rotvec_to_mat(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    k = w / np.where(th > 1e-12, th, 1.0)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def nearest_twin(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """The box-symmetric equivalent of ``q`` (and sign) closest to ``ref`` (wxyz, [4])."""
    R = quat_to_mat(q)
    best, bd = q, -2.0
    for S in SYM:
        c = mat_to_quat(R @ S)
        d = abs(float(c @ ref))
        if d > bd:
            best, bd = (c if c @ ref >= 0 else -c), d
    return best


class _Pinch:
    """Pinch positions of both arms from joint angles (cached kinematics)."""

    def __init__(self):
        import torch

        from tower_rl.kinematics import ArmBase, ArmKinematics
        self.torch = torch
        self.kin = {}
        for a in ARMS:
            k = ArmKinematics(device="cpu")
            k.set_arm(a)
            self.kin[a] = (k, ArmBase(a, device="cpu"))

    def __call__(self, q: np.ndarray) -> np.ndarray:
        out = []
        for i, a in enumerate(ARMS):
            k, b = self.kin[a]
            p, qq = k.fk(self.torch.as_tensor(np.asarray(q[i], dtype=np.float32))[None], "pinch")
            out.append(b.to_world(p, qq)[0][0].numpy())
        return np.stack(out).astype(np.float64)


class NoiseModel:
    """The fitted model (noise.yaml + bank), shared by every env; ``make`` returns one env's ``EstimatorNoise``."""

    def __init__(self, cfg: dict, bank: dict):
        self.cfg = cfg
        self.bank = {}
        for c in CLASSES:
            if f"{c}__data" in bank:
                data, lengths = bank[f"{c}__data"], bank[f"{c}__lengths"]
                starts = np.r_[0, np.cumsum(lengths)[:-1]]
                self.bank[c] = (np.asarray(data, dtype=np.float64), starts, lengths)
        pl = cfg.get("push_lost") or {}
        self.p_lost = pl.get("p_lost")
        self.runs = np.asarray(pl.get("run_lengths") or [3], dtype=np.int64)
        self.coast_decay = pl.get("coast_decay", 0.8)
        self.coast_max = pl.get("coast_max_calls", 5)
        self.moving_speed = cfg.get("moving_speed_m_s", 0.01)
        self.pinch = None
        self.log: list[dict] = []
        self.keep_log = True  # per-call records for the calibration (off for data collection)

    @classmethod
    def load(cls, path: str | Path) -> NoiseModel:
        cfg = load_yaml(Path(path))
        return cls(cfg, dict(np.load(REPO_ROOT / cfg["bank"])))

    def fallback(self, c: str) -> str:
        if c in self.bank:
            return c
        if c.startswith("target:"):
            return "moving" if "moving" in self.bank else "static"
        return "static"

    def p_lost_at(self, tilt_deg: float, along_m: float) -> float:
        if not self.p_lost:
            return 0.0
        ti = min(max(int(np.searchsorted(LOST_TILT_BINS, tilt_deg, side="right")) - 1, 0), len(LOST_TILT_BINS) - 2)
        ai = min(max(int(np.searchsorted(LOST_ALONG_BINS, along_m, side="right")) - 1, 0), len(LOST_ALONG_BINS) - 2)
        p = self.p_lost[ti][ai]
        return 0.0 if p is None else float(p)

    def make(self, block_ids: list[str], fixed: np.ndarray, seed: int, focus=None) -> EstimatorNoise:
        if self.pinch is None:
            self.pinch = _Pinch()
        return EstimatorNoise(self, block_ids, fixed, seed, focus)

    def summary(self) -> dict:
        """Per-class error percentiles of the noisy outputs (as CameraPerception.summary, plus per class)."""
        if not self.log:
            return {}
        out = {"calls": len(self.log)}
        cls = np.concatenate([r["cls"] for r in self.log])
        err = np.concatenate([r["err_mm"] for r in self.log])
        conf = np.concatenate([r["conf"] for r in self.log])
        for c in CLASSES:
            m = (cls == CLASSES.index(c)) & (conf >= 0.5)
            if m.any():
                e = err[m]
                out[c] = {"n": int(m.sum()), "pos_mm": [round(float(np.percentile(e, q)), 3) for q in (50, 90, 99)],
                          "frac_conf_lt_0.5": round(float((conf[cls == CLASSES.index(c)] < 0.5).mean()), 4)}
        return out


class EstimatorNoise:
    """One env's noisy-estimate stream (call it on every ``State`` handed to the policy)."""

    def __init__(self, model: NoiseModel, block_ids: list[str], fixed: np.ndarray, seed: int, focus=None):
        self.m = model
        self.ids = list(block_ids)
        self.fixed = np.asarray(fixed, dtype=bool)
        self.rng = np.random.default_rng(seed)
        self.focus = focus  # callable -> (block id | None, skill | None)
        nb = len(self.ids)
        self.cls = [None] * nb
        self.trace = [None] * nb
        self.idx = np.zeros(nb, dtype=np.int64)
        self.out_p: np.ndarray | None = None
        self.out_q: np.ndarray | None = None
        self.vel = np.zeros((nb, 3))
        self.lost_k = np.zeros(nb, dtype=np.int64)
        self.lost_left = np.zeros(nb, dtype=np.int64)
        self.p_enter = [None] * nb
        self.seed = seed

    def _classes(self, s: State, tp: np.ndarray) -> list[str]:
        speed = np.asarray(s.privileged.get("block_speed", np.zeros(len(self.ids))), dtype=np.float64)
        pinch = self.m.pinch(np.asarray(s.q, dtype=np.float64))
        held = np.zeros(len(self.ids), dtype=bool)
        for a in range(2):
            if float(s.width[a]) < HOLD_WIDTH:
                held |= np.linalg.norm(tp - pinch[a], axis=-1) < HOLD_RADIUS
        tgt, skill = self.focus() if self.focus is not None else (None, None)
        out = []
        for i, b in enumerate(self.ids):
            if self.fixed[i]:
                c = "foundation"
            elif tp[i, 2] < 0:
                c = "fallen"
            elif held[i]:
                c = "held"
            elif b == tgt and skill:
                c = f"target:{skill}" if f"target:{skill}" in CLASSES else "moving"
            elif speed[i] > self.m.moving_speed:
                c = "moving"
            else:
                c = "static"
            out.append(c)
        return out, held

    def _row(self, i: int, c: str) -> np.ndarray:
        """The next error row of block i's trace for class c (a new trace on a class change or at its end)."""
        bc = self.m.fallback(c)
        tr = self.trace[i]
        if self.cls[i] != c or tr is None or (self.idx[i] >= len(tr) and c not in ("foundation", "static")):
            data, starts, lengths = self.m.bank[bc]
            j = int(self.rng.integers(len(starts)))
            self.trace[i] = tr = data[starts[j]: starts[j] + lengths[j]]
            self.idx[i] = 0
            self.cls[i] = c
        r = tr[min(self.idx[i], len(tr) - 1)]
        self.idx[i] += 1
        return r

    def __call__(self, s: State) -> State:
        tp = np.asarray(s.block_pos, dtype=np.float64)
        tq = np.asarray(s.block_quat, dtype=np.float64)
        nb = len(self.ids)
        classes, held = self._classes(s, tp)
        first = self.out_p is None
        if first:
            self.out_p, self.out_q = tp.copy(), tq.copy()
        pos, quat, conf = self.out_p.copy(), self.out_q.copy(), np.zeros(nb)
        Rt = quat_to_mat(tq)
        for i in range(nb):
            c = classes[i]
            if c != "target:push":
                self.p_enter[i] = None
            r = self._row(i, c)
            lvl = int(r[6])
            if c == "target:push":  # losses from P(lost | tilt, travel) and recorded run lengths
                if self.p_enter[i] is None:
                    self.p_enter[i] = tp[i].copy()
                along = float(np.linalg.norm(tp[i, :2] - self.p_enter[i][:2]))
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(Rt[i, 2, 2])))))
                if self.lost_left[i] > 0:
                    self.lost_left[i] -= 1
                    lvl = 2
                else:
                    p = self.m.p_lost_at(tilt, along)
                    L = float(self.m.runs.mean()) if len(self.m.runs) else 1.0
                    p_start = p / (L * (1 - p) + p) if p < 1 else 1.0
                    if self.rng.random() < p_start:
                        self.lost_left[i] = int(self.m.runs[int(self.rng.integers(len(self.m.runs)))]) - 1
                        lvl = 2
                    elif lvl >= 2:
                        lvl = 0
            if first and lvl >= 2:
                lvl = 0  # the first call searches every block: report the fit
            if lvl <= 1:
                p_new = tp[i] + r[:3]
                q_new = mat_to_quat(rotvec_to_mat(r[3:6]) @ Rt[i])
                q_new = nearest_twin(q_new, self.out_q[i])
                self.vel[i] = p_new - self.out_p[i]
                self.lost_k[i] = 0
                pos[i], quat[i] = p_new, q_new
            else:
                self.lost_k[i] += 1
                if lvl == 2 and np.linalg.norm(self.vel[i]) > 0.003 and self.lost_k[i] <= self.m.coast_max:
                    pos[i] = self.out_p[i] + self.vel[i] * self.m.coast_decay ** self.lost_k[i]
            conf[i] = LEVELS[lvl]
        self.out_p, self.out_q = pos, quat
        err = np.linalg.norm(pos - tp, axis=-1) * 1000
        tgt, skill = self.focus() if self.focus is not None else (None, None)
        if self.m.keep_log:
            self.m.log.append({"seed": self.seed, "t": round(float(s.t), 2), "err_mm": err, "conf": conf.copy(), "held": held,
                               "fixed": self.fixed.copy(), "cls": np.array([CLASSES.index(c) for c in classes]),
                               "skill": skill, "block": self.ids.index(tgt) if tgt in self.ids else None})
        priv = dict(s.privileged)
        priv["true_block_pos"], priv["true_block_quat"] = tp, tq
        return replace(s, block_pos=pos, block_quat=quat, block_conf=conf, privileged=priv)
