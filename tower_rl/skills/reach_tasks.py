"""Task goals for the reacher (Reach-PPO-plan.MD P2): the teacher's own transits, recorded in rehearsals.

``TransitRecorder`` is an executor node hook (``Executor.node_hook``; ``rehearse.py --transit_log <file>``): every
transit node appends one JSON line when it is built (arm, target, held block, both arms' measured joints, the goal pinch
pose in the station frame, the graph's arrival tolerance, the planned rows, every block's pose) and one when it ends
(status, time). ``scripts/reach_task_check.py`` replays the goals on the reacher from the same start joints.

``BANDS``: the next skill's tight start band per transit target (``configs/skills/*.yaml`` ``reset``), the P2 gate.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

from tower_rl import geometry as geo
from tower_rl.graph.interface import ARMS
from tower_rl.paths import REPO_ROOT

# target -> (next skill, pos m, rot deg): grasp_lift / place_release / push tight bands; insert and push_in start bands
# are per axis (insert outside: lateral 8 mm, height 4 mm, yaw 3 deg; push_in: lateral 4 mm, rot 3 deg), the
# tightest axis here
BANDS = {"pregrasp": ("grasp_lift", 0.03, 5.0), "preplace": ("place_release", 0.02, 5.0),
         "prepush": ("push", 0.015, 5.0), "preinsert": ("insert", 0.004, 3.0), "prepushin": ("push_in", 0.004, 3.0)}


def _r(x, n=5):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


class TransitRecorder:
    """Node hook: transit goals and outcomes to a JSONL file (``tier`` is set by the caller per tier)."""

    def __init__(self, path: str | Path, trace: bool = False):
        self.trace = trace  # per row: time, this arm's finger opening, joints, pinch and the node's block position
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a")
        self.tier = None
        self.n = 0

    def __call__(self, run, node):
        if type(node).__name__ not in ("TransitNode", "LearnedTransitNode"):
            return node
        spec, arm = node.spec, node.arm
        rid = f"{run.ex.variant}/{self.tier}/{run.seed}/{self.n}"
        self.n += 1
        gp, gq = node.goal
        ctrl = run.ctrl[arm]
        rec = {"kind": "start", "id": rid, "variant": run.ex.variant, "tier": self.tier, "seed": run.seed,
               "node": spec["id"], "target": spec["target"], "arm": arm, "block": spec.get("block"),
               "held": node.plan.held, "payload_kg": node.payload,
               "q_des": _r(ctrl.q_des[0]), "goal_pos": _r(gp[0]), "goal_quat": _r(gq[0]),
               "tol_pos_m": spec["tol_pos_m"], "tol_rot_deg": spec["tol_rot_deg"],
               "planned_rows": len(node.plan.ps) + node.plan.settle_rows, "unreachable": bool(node.unreachable),
               **({"goal_adjust": node.info_adjust} if hasattr(node, "info_adjust") else {})}
        inner = node.step
        state = {"first": True}
        rows = []

        def step(s):
            if state["first"]:  # the measured joints / blocks when the transit starts (the node is built before)
                state["first"] = False
                rec.update(t0=round(float(s.t), 3), q=[_r(s.q[i]) for i in range(len(ARMS))],
                           blocks={b: _r(np.concatenate([s.block_pos[i], s.block_quat[i]])) for i, b in enumerate(s.block_ids)})
                pp, pq = run.pinch(s, arm)
                rec["start_pos_mm"] = round(float((pp - gp).norm()) * 1000, 1)
                rec["start_rot_deg"] = round(math.degrees(float(geo.quat_angle(pq, gq)[0])), 2)
                self.f.write(json.dumps(rec) + "\n")
                self.f.flush()
            if self.trace:
                ai = ARMS.index(arm)
                pp, _ = run.pinch(s, arm)
                b = spec.get("block")
                rows.append([round(float(s.t), 3), round(float(s.width[ai]), 4), _r(s.q[ai], 4), _r(pp[0], 4),
                             _r(s.block_pos[s.block_ids.index(b)], 4) if b in s.block_ids else None])
            status, out = inner(s)
            if status is not None:
                end = {"kind": "end", "id": rid, "status": str(getattr(status, "name", status)),
                       "t_s": round(float(s.t) - rec.get("t0", float(s.t)), 3), "learned": type(node).__name__ != "TransitNode"
                       and bool(getattr(node, "learned", False)), **node.info}
                if self.trace:
                    end["trace"] = rows
                self.f.write(json.dumps(end) + "\n")
                self.f.flush()
            return status, out

        node.step = step
        return node


def load_goals(paths, arm: str | None = None, completed: bool = False) -> list[dict]:
    """Start records (with their ``end`` merged in as ``end``) from recorder files, optionally one arm / finished ones."""
    recs, ends = {}, {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            d = json.loads(line)
            (recs if d["kind"] == "start" else ends)[d["id"]] = d
    out = []
    for rid, d in recs.items():
        if arm and d["arm"] != arm:
            continue
        d["end"] = ends.get(rid)
        if completed and d["end"] is None:
            continue
        out.append(d)
    return out


def load_task_pool(pattern: str, arm: str, device, with_held: bool = False):
    """The reacher's training pool (``goal.tasks`` / ``reset.task_frac``) from recorder files (``pattern``: a glob,
    relative to the repo): start joints ``[N, 7]`` and goal pinch poses ``[N, 3]``, ``[N, 4]`` (station frame);
    ``with_held``: also whether each transit held a block ``[N]``."""
    import torch

    files = sorted(glob.glob(str(REPO_ROOT / pattern)))
    recs = [g for g in load_goals(files, arm=arm) if "q" in g]
    assert recs, f"no recorded transits for {arm} in {pattern}"
    t = lambda key: torch.tensor([g[key] for g in recs], dtype=torch.float32, device=device)  # noqa: E731
    q = torch.tensor([g["q"][ARMS.index(arm)] for g in recs], dtype=torch.float32, device=device)
    if with_held:
        return q, t("goal_pos"), t("goal_quat"), t("payload_kg") > 0
    return q, t("goal_pos"), t("goal_quat")


def structure_top(blocks: dict, exclude: str | None = None) -> float:
    """Highest block top (m, station frame) among recorded / estimated block poses ``id -> [x, y, z, qw, qx, qy, qz]``
    (blocks lie flat: centre + half the 3 cm height), without ``exclude`` (a held block)."""
    return max(v[2] + 0.015 for b, v in blocks.items() if b != exclude)


class Carrot:
    """A moving goal for the reacher, batched: each row it advances toward a target pose (a via-point or the goal), position
    and rotation together, at most ``v [n]`` m/s and ``w`` rad/s, speeding up and slowing down at ``a`` m/s^2, stopping on
    the target. The reacher tracks it a little behind, so a speed limit needs no retraining (10-31: P2b's speed-limit input
    left a held block at ~0.9 m/s with a 0.15 m/s limit; its actor weight on the input stayed ~0.04 vs ~1.2 for the others).
    The planned transit's limits: 0.25 m/s empty, 0.12 held, 0.6 m/s^2, 1 rad/s (``transit.TransitCfg``)."""

    def __init__(self, p, q, v, a: float = 0.6, w: float = 1.0):
        import torch

        self.p, self.q, self.v, self.a, self.w = p.clone(), q.clone(), v, a, w
        self.f = torch.zeros_like(v)  # current speed / v

    def step(self, tp, tq, dt: float):
        """Advance one row toward ``tp [n, 3]``, ``tq [n, 4]``; returns the envs on the target."""
        import torch

        d = (tp - self.p).norm(dim=-1)
        rv = geo.quat_error_rotvec(self.q, tq)
        left = torch.maximum(d / self.v, rv.norm(dim=-1) / self.w)  # seconds to go at full speed
        # speed up at a, and slow down in time to stop on the target (left * v: the distance to go in position units)
        self.f = torch.minimum((self.f + self.a * dt / self.v).clamp(max=1.0), (2 * self.a * left / self.v).sqrt())
        s = torch.where(left > 1e-6, self.f * dt / left.clamp(min=1e-6), torch.ones_like(left)).clamp(max=1.0)
        self.p = self.p + s[:, None] * (tp - self.p)
        self.q = geo.normalize(geo.quat_mul(geo.quat_from_rotvec(s[:, None] * rv), self.q))
        on = s >= 1.0
        self.p[on], self.q[on] = tp[on], tq[on]
        return on


def via_points(p0, q0, p1, q1, top, held, margin: float = 0.03, held_extra: float = 0.03, rise: float = 0.05,
               cap: float = 0.33):
    """The reacher's via-points for transits, batched, station frame: poses ``[n, 3, 7]`` (above the start with the start
    orientation, above the goal with the goal orientation, the goal) and valid flags ``[n, 3]``; the reacher gets the
    next valid one when it is within ~1 cm / 20 deg of the current one (``scripts/reach_task_check.py --via``).

    Travel height: ``margin`` above the tallest structure ``top [n]`` (+ ``held_extra`` with a block, ``held [n]``), at
    least 2 cm above the goal, at most ``cap``. The planned transit's rule (2 cm above the higher of start and goal) put
    it at 0.40 m from the home posture, above the reacher's training goals (z <= 0.35), and the reacher parked 2.5 cm
    short of it. A start below travel height + ``rise`` first rises at least ``rise`` straight up: heading straight for
    the next via-point the reacher dipped ~1 cm in its first rows, finger tips level with a block it had just placed
    (10-31, held-out task goals: 3 cm rise 7.3 % of left pre-grasps grazed a block, 5 cm none). Without via-points its
    direct paths crossed other blocks on 17-37 % of task transits."""
    import torch

    zt = torch.maximum(top + margin + held.float() * held_extra, p1[:, 2] + 0.02).clamp(max=cap)
    low = p0[:, 2] < zt + rise
    zt = torch.where(low, torch.maximum(zt, p0[:, 2] + rise).clamp(max=cap), zt)
    a, b = p0.clone(), p1.clone()
    a[:, 2], b[:, 2] = zt, zt
    poses = torch.stack([torch.cat([a, q0], -1), torch.cat([b, q1], -1), torch.cat([p1, q1], -1)], 1)
    valid = torch.stack([low, zt - p1[:, 2] > 0.01, torch.ones_like(low)], 1)
    return poses, valid
