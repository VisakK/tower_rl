"""Provisional perception noise for rehearsal (Step 8.1): corrupt the block poses the executor sees.

Per block: a per-episode bias (pos sigma ``bias_pos_m``, yaw sigma ``bias_yaw_deg``), per-call jitter, dropouts
(with probability ``dropout_p`` per call a block's pose freezes at its last estimate for ``dropout_s``) and outliers
(probability ``outlier_p`` of an extra ``outlier_m`` jump for one call). Foundations get the same model (the estimator
sees them too). Simulator truth stays in ``State.privileged`` for the evaluator. Seeded per episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from tower_rl.graph.interface import State


@dataclass
class NoiseCfg:
    """Defaults = the build file's provisional model (independent 3 mm / 3 deg per block). A multi-view estimator's
    errors are largely shared by all blocks (calibration, viewpoint) and cancel in block-to-goal placement:
    ``shared_*`` adds a bias common to every block of the episode."""
    bias_pos_m: float = 0.003
    bias_yaw_deg: float = 3.0
    shared_pos_m: float = 0.0
    shared_yaw_deg: float = 0.0
    step_pos_m: float = 0.001
    step_yaw_deg: float = 0.5
    dropout_p: float = 0.02
    dropout_s: float = 0.5
    outlier_p: float = 0.005
    outlier_m: float = 0.01


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


class PerceptionNoise:
    def __init__(self, cfg: NoiseCfg, num_blocks: int, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        c = cfg
        self.bias_p = (self.rng.normal(0, c.bias_pos_m, (num_blocks, 3)) + self.rng.normal(0, c.shared_pos_m, (1, 3))) * np.array([1, 1, 0.5])
        self.bias_y = np.radians(self.rng.normal(0, c.bias_yaw_deg, num_blocks) + self.rng.normal(0, c.shared_yaw_deg))
        self.frozen_until = np.full(num_blocks, -1.0)
        self.last_p = None
        self.last_q = None

    def __call__(self, s: State) -> State:
        c = self.cfg
        b = len(s.block_ids)
        p = s.block_pos + self.bias_p + self.rng.normal(0, c.step_pos_m, (b, 3)) * np.array([1, 1, 0.5])
        yaw = self.bias_y + np.radians(self.rng.normal(0, c.step_yaw_deg, b))
        q = _qmul(_yaw_quat(yaw), s.block_quat)
        out_mask = self.rng.random(b) < c.outlier_p
        if out_mask.any():
            d = self.rng.normal(0, 1, (int(out_mask.sum()), 3))
            d[:, 2] *= 0.3
            p[out_mask] += d / np.linalg.norm(d, axis=-1, keepdims=True) * c.outlier_m
        start = (self.rng.random(b) < c.dropout_p) & (self.frozen_until < s.t)
        self.frozen_until[start] = s.t + c.dropout_s
        frozen = self.frozen_until >= s.t
        conf = np.ones(b)
        if self.last_p is not None and frozen.any():
            p[frozen], q[frozen] = self.last_p[frozen], self.last_q[frozen]
            conf[frozen] = 0.0
        self.last_p, self.last_q = p.copy(), q.copy()
        priv = dict(s.privileged)
        priv["true_block_pos"], priv["true_block_quat"] = s.block_pos, s.block_quat
        return replace(s, block_pos=p, block_quat=q / np.linalg.norm(q, axis=-1, keepdims=True), block_conf=conf, privileged=priv)
