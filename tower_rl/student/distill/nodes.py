"""Distilled Task 5 nodes and the DART collection hook (student plan Step 9c).

The distilled nodes subclass the planned ones: the network drives the motion and the gripper decision (grasp: when to
close; insert: when to release); the planned nodes' mechanical parts stay: attaching the block on the 14th closed row,
the payload, the grasp verdicts (lifted while holding / empty gripper), the insert's release wait, retreat and slot
verdict, the push-in's retreat and verdict, the structure checks and the timeouts. Mechanical guards (the planned
nodes' own bounds; BUILDLOG): the grasp closes only within 5 mm of the grasp target, once, and stays closed, holding
still while the fingers close; a block 1 s at the grasp target closes; no release more than 1 cm short of the entry depth, a block 1 s at the entry depth is
released, no advance past the entry depth; the push target stops 6 mm past "block centred", and the push ends once the
block is home (or past it), stalls after moving, or after 200 rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import torch

from tower_rl.graph import checkpoints as ck
from tower_rl.graph.executor import InsertNode, PlannedGraspNode, PushInNode, _structure_moved, grasp_cfg_payload, \
    record_held_offset
from tower_rl.graph.interface import ARMS, Status
from tower_rl.skills.obs import grasp_target_world
from tower_rl.student.distill.core import NODE_SKILL, Dart, DistillObs, action_of, apply_action

ATTACH_ROW = 14  # the closed row on which the planned grasp attaches the block (its phase 1 ends: 1 + 13 rows)


def _act(net, o: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return net(o).clone()


TRACE = os.environ.get("TOWER_DISTILL_TRACE")  # debugging: one JSON line per distilled row appended to this file


def _trace(node, s, a, **kw):
    if TRACE:
        with open(TRACE, "a") as f:
            f.write(json.dumps({"seed": node.run.seed, "t": round(float(s.t), 2), "node": node.spec["id"],
                                "rel": [round(float(x), 5) for x in node.obs.hist[-1]],
                                "a": [round(float(x), 4) for x in a[0]], **kw}) + "\n")


class _Distilled:
    """Marks the node's end info ``distilled`` (rehearsal reports, compare_runs)."""

    def step(self, s):
        status, out = self._step(s)
        if status is not None:
            self.info = dict(self.info or {}, distilled=True)
        return status, out


class DistilledGraspNode(_Distilled, PlannedGraspNode):
    """Descend, close and lift by the network; ok once the block rose ``lift_m`` while held."""

    def __init__(self, run, spec, s, net):
        super().__init__(run, spec, s)
        self.net, self.obs = net, DistillObs(self, "grasp_t5")
        self.closed_rows, self.near_rows = 0, 0

    def _step(self, s):
        r, arm = self.run, self.arm
        if self.timed_out(s):
            return Status.TIMEOUT, []
        a = _act(self.net, self.obs(self, s))
        _trace(self, s, a, closed_rows=self.closed_rows)
        near = False
        if self.closed_rows == 0:
            bp, bq = r.block_est(s, self.spec["block"])
            gp, _ = grasp_target_world(bp, bq, arm)
            pp, _ = r.pinch(s, arm)
            dist = float((pp - gp).norm())
            near = dist < 0.005  # the planned grasp closes within 4 mm of the target: no earlier close
            self.near_rows = self.near_rows + 1 if dist < 0.004 else 0
        # latched once closed; the network's close only near the target; 1 s at the target closes
        close = self.closed_rows > 0 or (bool(a[0, 6] > 0) and near) or self.near_rows > 10
        n = self.closed_rows + int(close)  # closed rows including this one
        if 2 <= n <= ATTACH_ROW:
            a[0, :6] = 0.0  # hold still while the fingers close (the planned phase 1)
        a[0, 6] = 1.0 if close else 0.0
        apply_action(self, "grasp_t5", a)
        self.obs.prev_a = a  # the executed action, as the training labels: gripper bit 0 / 1, not the logit
        self.closed_rows = n
        if n > ATTACH_ROW:
            r.refresh_attached(s, arm)
        elif n == ATTACH_ROW and ck.gripping(s, arm):
            r.attach(s, arm, self.spec["block"])  # fingers on the block, before it moves (as the planned grasp)
        r.grip[arm].step(torch.tensor([close], device=r.dev))
        r.command(arm, r.q_meas(s, arm), payload=grasp_cfg_payload() if n >= ATTACH_ROW else 0.0)
        if n >= ATTACH_ROW:
            if ck.grasped(s, arm, self.spec["block"], self.z0, self.spec["lift_m"], r.ex.cfg.held_width):
                bp, _ = r.block_est(s, self.spec["block"])
                self.info = {"lift_mm": round((float(bp[0, 2]) - self.z0) * 1000, 1),
                             **record_held_offset(r, s, arm, self.spec["block"])}
                return Status.OK, [r.row()]
            if float(s.width[ARMS.index(arm)]) < r.ex.cfg.held_width[0]:
                self.info = {"why": "empty gripper"}
                return Status.FAILED, [r.row()]
        return None, [r.row()]


class DistilledInsertNode(_Distilled, InsertNode):
    """The held slide and the release decision by the network; the planned release wait, retreat and verdict."""

    def __init__(self, run, spec, s, net):
        super().__init__(run, spec, s, None)
        self.net, self.obs = net, DistillObs(self, "insert")
        self.at_depth = 0

    def _step(self, s):
        if self.phase != 0:
            return InsertNode.step(self, s)
        r, arm = self.run, self.arm
        if self.timed_out(s):
            return Status.TIMEOUT, []
        moved = _structure_moved(r, s, self.ref)
        if moved > 0.010:
            self.info = {"why": "structure", "moved_mm": round(moved * 1000, 1)}
            return Status.FAILED, []
        sp, _, d = self.slot
        bp, _ = r.block(s, self.spec["block"])
        along = float(((bp - sp) * d).sum())
        entry = -self.ig["entry_m"]
        self.at_depth = self.at_depth + 1 if along > entry - 0.004 else 0
        a = _act(self.net, self.obs(self, s))
        _trace(self, s, a, along=round(along, 5), k=self.k)
        if (bool(a[0, 6] <= 0) and along > entry - 0.01) or self.at_depth > 10:
            self.phase, self.k = 1, 1  # release: the planned phase 1 (open, wait 0.6 s), retreat and verdict
            return InsertNode.step(self, s)
        if along > entry + 0.003:  # the planned slide's bound: its depth target stops at the entry depth
            a[0, :3] = a[0, :3] - torch.clamp((r.ctrl[arm].base.vec_to_base(d) * a[0, :3]).sum(), min=0.0) * r.ctrl[arm].base.vec_to_base(d)[0]
        a[0, 6] = 1.0  # held
        apply_action(self, "insert", a)
        self.obs.prev_a = a  # the executed action, as the training labels
        self.k += 1
        if self.k > 400:
            self.info = {"why": "slide stalled", "along_mm": round(along * 1000, 1)}
            return Status.FAILED, []
        r.grip[arm].step(torch.tensor([True], device=r.dev))
        r.command(arm, r.q_meas(s, arm), payload=grasp_cfg_payload())
        return None, [r.row()]


class DistilledPushInNode(_Distilled, PushInNode):
    """Descend behind the end face and push home by the network; the planned retreat and verdict once home."""

    def __init__(self, run, spec, s, net):
        super().__init__(run, spec, s, None)
        self.net, self.obs = net, DistillObs(self, "push_in")
        self.static_rows, self.prev_along_d, self.along0 = 0, None, None

    def _step(self, s):
        if self.phase >= 2:
            return PushInNode.step(self, s)
        r, arm = self.run, self.arm
        if self.timed_out(s):
            return Status.TIMEOUT, []
        sp, _, d = self.slot
        moved = _structure_moved(r, s, self.ref)
        if moved > 0.010:
            self.info = {"why": "structure", "moved_mm": round(moved * 1000, 1)}
            return Status.FAILED, []
        bp, _ = r.block(s, self.spec["block"])
        along = self.kin_along(s, sp, d, float(((bp - sp) * d).sum()))  # with "pushin_kin": the pusher's contact bound
        if self.along0 is None:
            self.along0 = along
        static = self.prev_along_d is not None and abs(along - self.prev_along_d) < 0.0005
        self.prev_along_d = along
        self.static_rows = self.static_rows + 1 if static else 0
        a = _act(self.net, self.obs(self, s))
        a[0, 6] = 1.0  # the pusher: the gripper commanded closed at the pusher width
        apply_action(self, "push_in", a)
        self.obs.prev_a = a  # the executed action, as the training labels
        ctrl = r.ctrl[arm]
        over = float(((ctrl.pos - sp) * d).sum()) - (self.ig["pushin_contact_along_m"] + 0.006)
        if over > 0:  # the planned push's bound: its target stops at "block centred" + at most 6 mm of nudges
            ctrl.set_target(ctrl.pos - d * over, ctrl.quat)
        r.command(arm, r.q_meas(s, arm))
        self.k += 1
        if along > -0.0015 or (along - self.along0 > 0.002 and self.static_rows > 15) or self.k > 200:
            self.phase, self.k = 2, 0  # home, past it, stalled or out of rows: next the planned retreat and verdict
        return None, [r.row()]


def distill_kind(node) -> str | None:
    """The distillable skill of a planned Task 5 node (planned grasp with a lift speed, planned insert, planned
    push-in), else None."""
    if type(node) is PlannedGraspNode and node.spec.get("lift_v_m_s") is not None:
        return "grasp_t5"
    if type(node) is InsertNode and node.pol is None:
        return "insert"
    if type(node) is PushInNode and node.pol is None:
        return "push_in"
    return None


def make_distilled(run, spec, s):
    """The distilled node for this spec when the bundle has its network (``skills/<skill>_bc_<arm>.pt``), else None."""
    sk = NODE_SKILL.get(spec["skill"])
    if sk is None or (sk == "grasp_t5" and spec.get("lift_v_m_s") is None):
        return None
    net = run.ex.skills.get((f"{sk}_bc", spec["arm"]))
    if net is None:
        return None
    cls = {"grasp_t5": DistilledGraspNode, "insert": DistilledInsertNode, "push_in": DistilledPushInNode}[sk]
    return cls(run, spec, s, net)


class DartCollector:
    """Executor node hook (``Executor.node_hook``) for the Step 9c data: every planned Task 5 node runs under DART
    (an Ornstein-Uhlenbeck shift of the controller target before each row; the planned node's command from there is
    the label) and its rows are kept per env as (skill, arm, segment, observation, action); ``take(run)`` hands an
    episode's rows over. ``SIGMA``: per-row increments (m, rad) per skill; each node draws a scale in [0, ``max_scale``].
    The plan's 1 mm (25 % of 4 mm) holds for the grasp and the push-in; the planned insert, which advances only while
    the block is aligned within 0.8 mm / 0.6 mm / 0.4 deg / 0.3 deg, timed out under it in 14 of 24 pilot segments (it
    waits for alignment), so its noise is 0.2 mm / 0.04 deg per row."""

    SIGMA = {"grasp_t5": (0.001, math.radians(0.15)), "insert": (0.0002, math.radians(0.04)),
             "push_in": (0.0005, math.radians(0.1))}

    def __init__(self, seed: int, theta: float = 0.3, max_scale: float = 1.5, sigma: dict | None = None):
        self.rng = np.random.default_rng([seed, 9300])
        self.theta, self.max_scale = theta, max_scale
        self.sigma = dict(self.SIGMA, **(sigma or {}))
        self.rows: dict[int, list] = {}
        self.segments: dict[int, list] = {}

    def __call__(self, run, node):
        sk = distill_kind(node)
        if sk is None:
            return node
        key = id(run)
        segs = self.segments.setdefault(key, [])
        seg = {"skill": sk, "arm": node.arm, "node": node.spec["id"], "variant": run.ex.variant, "status": None, "rows": 0,
               "scale": float(self.rng.uniform(0.0, self.max_scale))}
        segs.append(seg)
        sm, sr = self.sigma[sk]
        dart = Dart(self.rng, sm * seg["scale"], sr * seg["scale"], self.theta)
        obs = DistillObs(node, sk)
        inner = node.step
        sink = self.rows.setdefault(key, [])
        sid = len(segs) - 1
        state = {"released": False}

        def phase_of():
            ph = getattr(node, "phase", 0)
            if sk == "grasp_t5":
                return True, ph != 1                   # record every row; no shift while the fingers close
            if sk == "insert":
                rec = ph == 0 or (ph == 1 and not state["released"])  # the slide and the release row
                return rec, ph == 0
            return ph < 2, ph < 2                      # push-in: descent and push

        def step(s):
            rec, pert = phase_of()
            ph0 = getattr(node, "phase", 0)
            if pert and seg["scale"] > 0:
                dart.perturb(node, sk)
            o = obs(node, s) if rec else None
            ctrl = node.run.ctrl[node.arm]
            p0, q0 = ctrl.pos.clone(), ctrl.quat.clone()
            status, out = inner(s)
            if rec and out:
                if sk == "insert" and ph0 == 1:
                    state["released"] = True  # the release row (the first row of the planned phase 1)
                a = action_of(node, sk, p0, q0, bool(node.run.grip[node.arm].closed))
                obs.prev_a = a
                sink.append((sid, o[0].numpy().copy(), a[0].numpy().copy()))
                seg["rows"] += 1
            if status is not None:
                seg["status"] = status.value
                seg["info"] = {k: v for k, v in (node.info or {}).items() if isinstance(v, (int, float, str, bool))}
            return status, out

        node.step = step
        return node

    def reseed(self, seed: int, salt: str = ""):
        """Per rehearsal batch: the DART noise depends on the batch seed and ``salt`` (variant / tier) only, not on the
        process history."""
        self.rng = np.random.default_rng([seed, int(hashlib.sha256(salt.encode()).hexdigest()[:15], 16), 9300])

    def take(self, run) -> tuple[list, list]:
        """(segments, rows) recorded for this run (an episode), removed from the collector."""
        key = id(run)
        return self.segments.pop(key, []), self.rows.pop(key, [])
