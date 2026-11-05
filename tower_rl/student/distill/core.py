"""Observation, action, DART collection and network for the distilled Task 5 skills (student plan Step 9c).

Observation (68, the arm's base frame), from the quantities the planned nodes use: pinch pose / velocity / lead (21);
width, gripper command (+-1), width command (3); the node's block in the pinch frame (9); a reference pose (6: the
slot centre and axis for insert / push-in, the grasp target and the block's long axis for the grasp); the block's
offsets from that reference over the last 3 calls (15: along, across, vertical, signed yaw, pitch for insert /
push-in; the pinch from the grasp target along / across / vertical, signed yaw, lift for the grasp); node features
(7: gap below, entry distance, entry lift, rows since the node started / 300, released, rows since the release / 30,
the cross-window flag); the previous action (7).

Action per row: the pinch target delta (3, / ``POS_LIM``) and rotation delta (3, rotation vector / ``ROT_LIM``) in the
base frame, and a gripper bit (closed). Applied as the planned nodes do: ``ctrl.set_target(ctrl.pos + dp, R(dr) quat)``.

DART: before each recorded row an Ornstein-Uhlenbeck increment shifts the controller's target (the plan's sigma: 25 %
of +-4 mm, +-0.6 deg per row); the planned node then computes its (corrective) command from there, which is the label.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from tower_rl import geometry as geo
from tower_rl.graph.interface import ARMS
from tower_rl.skills.obs import _axes_in, grasp_target_world, rot6d

OBS_DIM = 68
ACT_DIM = 7
NODE_SKILL = {"grasp_lift": "grasp_t5", "insert": "insert", "push_in": "push_in"}
POS_LIM = {"grasp_t5": 0.008, "insert": 0.006, "push_in": 0.012}  # m per row (the planned nodes' largest steps)
ROT_LIM = math.radians(1.5)                                          # rad per row
ROW_DT = 0.1


def _f(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


class DistillObs:
    """Per-node observation state (offset history, previous pinch pose and action, row counters)."""

    def __init__(self, node, skill: str):
        self.skill = skill
        self.hist: deque = deque(maxlen=3)
        self.prev_pb = self.prev_qb = None
        self.prev_a = torch.zeros(1, ACT_DIM)
        self.rows = 0
        self.released_at = None

    def offsets(self, node, s):
        """(reference position [1,3], reference axis [1,3], 5 offsets) in the world frame."""
        r, arm = node.run, node.arm
        bid = node.spec["block"]
        if self.skill == "grasp_t5":
            bp, bq = r.block_est(s, bid)
            gp, gq = grasp_target_world(bp, bq, arm)
            if r.grasp_quat[arm] is not None:
                gq = r.grasp_quat[arm]
            pp, pq = r.pinch(s, arm)
            R = geo.quat_to_mat(bq)[0]
            d = (pp - gp)[0]
            signed = float(((geo.yaw_from_quat(gq) - geo.yaw_from_quat(pq) + math.pi / 2) % math.pi - math.pi / 2)[0])
            rel = [float(d @ R[:, 0]), float(d @ R[:, 1]), float(d[2]), signed, float(bp[0, 2]) - getattr(node, "z0", float(bp[0, 2]))]
            ax = R[:, 0][None]
            return gp, ax, rel
        sp, sq, d = node.slot
        bp, bq = r.block(s, bid)
        along = float(((bp - sp) * d).sum())
        lat_dir = torch.stack([-d[:, 1], d[:, 0], torch.zeros_like(d[:, 0])], -1)
        across = float(((bp - sp) * lat_dir).sum())
        vertical = float(bp[0, 2] - sp[0, 2])
        signed = float(((geo.yaw_from_quat(sq) - geo.yaw_from_quat(bq) + math.pi / 2) % math.pi - math.pi / 2)[0])
        bx = geo.quat_apply(bq, torch.tensor([[1.0, 0.0, 0.0]], dtype=bq.dtype))
        bx = bx * (1.0 if float((bx * d).sum()) >= 0 else -1.0)
        pitch = float(torch.asin(bx[0, 2].clamp(-1, 1)))
        return sp, d, [along, across, vertical, signed, pitch]

    def __call__(self, node, s, released: bool = False) -> torch.Tensor:
        r, arm = node.run, node.arm
        ctrl, base = r.ctrl[arm], r.ctrl[arm].base
        qm = r.q_meas(s, arm)
        pb, qb = ctrl.kin.fk(qm, "pinch")
        if self.prev_pb is None:
            self.prev_pb, self.prev_qb = pb, qb
        vel = torch.cat([(pb - self.prev_pb) / ROW_DT, geo.quat_error_rotvec(self.prev_qb, qb) / ROW_DT], -1)
        tpb, tqb = base.to_base(ctrl.pos, ctrl.quat)
        lead = torch.cat([tpb - pb, geo.quat_error_rotvec(qb, tqb)], -1)
        w = float(s.width[ARMS.index(arm)])
        grip = r.grip[arm]
        bpw, bqw = r.block_est(s, node.spec["block"])
        bpb, bqb = base.to_base(bpw, bqw)
        bp_rel, bx, bz = _axes_in(pb, qb, bpb, bqb)
        ref_p, ref_ax, rel = self.offsets(node, s)
        refb = base.to_base(ref_p, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=ref_p.dtype))[0]
        axb = base.vec_to_base(ref_ax.to(refb.dtype))
        self.hist.append(rel)
        while len(self.hist) < 3:
            self.hist.appendleft(rel)
        sp = node.spec
        if released and self.released_at is None:
            self.released_at = self.rows
        since_rel = 0.0 if self.released_at is None else (self.rows - self.released_at) / 30
        in_cross = 0.0
        if sp.get("gap_below") and self.skill == "push_in":
            c0, c1 = node.ig.get("cross_window_m", [0.0, 0.0])
            in_cross = float(c0 - 0.01 <= rel[0] <= c1)
        feats = [float(bool(sp.get("gap_below"))), float(sp.get("entry_m", 0.0)) * 10, float(sp.get("entry_lift_m", 0.0)) * 100,
                 min(self.rows / 300, 1.5), float(released), min(since_rel, 1.5), in_cross]
        hist = np.asarray(self.hist, dtype=np.float32).copy()
        hist[:, :3] *= 10  # metres -> decimetres
        o = torch.cat([pb, rot6d(qb), vel, lead, _f([[w * 10, 1.0 if bool(grip.closed) else -1.0, float(grip.width) * 10]]),
                       bp_rel, bx, bz, refb, axb, _f(hist.reshape(1, -1)), _f([feats]), self.prev_a], -1).float()
        self.prev_pb, self.prev_qb = pb, qb
        self.rows += 1
        assert o.shape[1] == OBS_DIM, o.shape
        return o


def action_of(node, skill: str, p0, q0, closed: bool) -> torch.Tensor:
    """The node's command this row as a normalized action (target delta in the base frame since ``p0, q0``)."""
    ctrl = node.run.ctrl[node.arm]
    base = ctrl.base
    dp = base.vec_to_base(ctrl.pos - p0) / POS_LIM[skill]
    dr = base.vec_to_base(geo.quat_error_rotvec(q0, ctrl.quat)) / ROT_LIM
    return torch.cat([dp.clamp(-1, 1), dr.clamp(-1, 1), torch.tensor([[1.0 if closed else 0.0]])], -1).float()


def apply_action(node, skill: str, a: torch.Tensor):
    """Move the arm controller's target by a normalized action (as the planned nodes move it)."""
    ctrl = node.run.ctrl[node.arm]
    base = ctrl.base
    dp = base.vec_to_world(a[:, :3].clamp(-1, 1) * POS_LIM[skill])
    dr = base.vec_to_world(a[:, 3:6].clamp(-1, 1) * ROT_LIM)
    ctrl.set_target(ctrl.pos + dp, geo.normalize(geo.quat_mul(geo.quat_from_rotvec(dr), ctrl.quat)))


class Dart:
    """Ornstein-Uhlenbeck target perturbation of one node (base frame): per-row increments ``sigma_m`` / ``sigma_rad``,
    reverting with ``theta`` (stationary offset about 1.4 sigma at 0.3)."""

    def __init__(self, rng: np.random.Generator, sigma_m: float, sigma_rad: float, theta: float = 0.3):
        self.rng, self.sigma_m, self.sigma_rad, self.theta = rng, sigma_m, sigma_rad, theta
        self.n = np.zeros(6)

    def perturb(self, node, skill: str):
        ctrl = node.run.ctrl[node.arm]
        base = ctrl.base
        new = (1 - self.theta) * self.n + self.rng.standard_normal(6) * np.repeat([self.sigma_m, self.sigma_rad], 3)
        inc, self.n = new - self.n, new
        dp = base.vec_to_world(_f([inc[:3]]).to(ctrl.pos.dtype))
        dr = base.vec_to_world(_f([inc[3:]]).to(ctrl.pos.dtype))
        ctrl.set_target(ctrl.pos + dp, geo.normalize(geo.quat_mul(geo.quat_from_rotvec(dr), ctrl.quat)))


PREV_A = slice(OBS_DIM - ACT_DIM, OBS_DIM)  # the previous action's columns


class DistillNet(nn.Module):
    """MLP (256-256-256, ELU) with the observation normalizer: obs -> 6 bounded deltas (tanh) + the gripper logit.
    ``mask`` (0 / 1 per feature) zeroes inputs the network must not use: trained with the previous action as an input,
    the networks copied it (zeroing it raised the insert's validation error 200x) and drifted in closed loop."""

    def __init__(self, mean=None, std=None, hidden=(256, 256, 256), mask=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(OBS_DIM) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(OBS_DIM) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("mask", torch.ones(OBS_DIM) if mask is None else torch.as_tensor(mask, dtype=torch.float32))
        layers, d = [], OBS_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, ACT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        y = self.mlp(((obs * self.mask - self.mean) / self.std).clamp(-10.0, 10.0))
        return torch.cat([torch.tanh(y[:, :6]), y[:, 6:7]], -1)
