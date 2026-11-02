"""The recovery node (Plan-v1.1 §5.3-5.4): Claude-authored recovery rules over the contracts (``contracts.py``).

``RecoveryNode.propose(view, d)`` runs at every decision point after the base decider (the teacher's state oracle, or the
student's switcher) chose ``d``; it returns a recovery decision that replaces ``d``, or None. It reads the scene from
simulator truth (``source="truth"``: the teacher, labels) or from the runtime's rest estimates (``source="estimate"``:
the deployable student, gated by perception health). Rules, from the failure clusters of official run 1 and the local
mirror runs (each rule's precondition keeps the remaining structure out of reach, Plan-v1.1 §5.5):

* R4.1 re-push (pushdown): a block whose push ended ok but which rests leaning on the foundation's edge ("on/leaning on
  base": the official check fails it; 3 of 16 failing rest states in official Task 4-2) -> ``repush(k)``: the planned
  straight push from behind its raised end until it is confirmed clear (``PushNode`` with ``recover``). Preconditions
  (``repush_ok``; the action mask of features v2 uses the same): the block moved >= 0.10 m along its push, its near
  end stands >= ``NEAR_END_Z`` high (a block lying flat against the base cannot be reached from behind: its pre-push
  pose is inside the foundation; smoke run 2026-10-28: transit timeouts), no block of a higher layer is still on the
  tower, at most ``max_per_goal`` times, with >= ``min_left_s`` of the episode left.
* R5.1 retry the insert (insertion): an insert that failed while its arm still holds the block, the structure intact
  -> ``insert(k)`` again (the transport retry), once.
* R5.2 push home (insertion): an insert block released on the slot line short of home ("short"), the structure intact,
  its push-in not yet run or not finished -> ``push_home(k)``, at most twice (official run 1: one insert left ~50 mm
  out with the structure intact). The student needs the block confidently seen >= 15 mm short (the estimate of a
  nearly hidden insert jumps, issue 13).
* Abstain: a goal whose contract fails at the end audit with no rule above is logged once (``abstentions``: variant,
  time, goal, kind, details) for the offline review that writes the next rules (Claude, Plan-v1.1 §5.3 step 2).

Perception health (student): no recovery while the episode is BLIND, while fewer than ``health_min`` of the movable
blocks are confidently seen, or when the goal's block was not confidently seen on its last calls.

``seen_s`` (student, opt-in through graph ``recovery: {"seen_s": s}``): "seen" and "at rest" from the block's
confident sightings over the last ``seen_s`` instead of every call of the last ``static_s``. Camera loop 2026-10-28: a
block left leaning on the base edge was confidently seen on 4 of ~30 idle calls (never two in a row), and the idle
phase calls every 1.6 s (``idle_chunk`` 16), so the default gate could never pass after the last push. With
``seen_s`` it needs >= 2 confident sightings in that window, all within 4 mm of the block's rest estimate; the
unconfident (coasted / stale) calls in between are ignored rather than vetoing.

``at_end`` (opt-in, graph ``recovery: {"at_end": true}``): R4.1 only once every goal has been pushed or its block is
confidently seen off the tower (a goal the decider skipped), no block is left on the tower and the base decider idles
or parks. Camera loop 2026-10-28 (injected short pushes): the two
re-pushes made at the plan's end left their blocks flat clear; one made after layer 2 with layer 1 still on the tower
(the layer-2 block had fallen against a layer-1 block's end) came in over layer 1 at the leaning block's height and
moved two layer-1 blocks 19-22 mm ("structure"). A block left leaning breaks the push order once the next block
departs (the mirror's TSR), so a re-push at the end rescues its rest state (LHPR) and, for the last push, the TSR.
"""

from __future__ import annotations

import numpy as np

from tower_rl.graph.interface import ARMS
from tower_rl.recovery.contracts import Contracts
from tower_rl.student.spec import Decision

EPISODE_S = 210.0
SEEN = 0.5
LEAN_KINDS = ("on/leaning on base",)
PUSHED_M = 0.10      # R4.1: the block moved at least this far along its push (it was pushed out, not stuck on the tower)
NEAR_END_Z = 0.045   # ... and its end nearest the tower stands this high (the foundation's top is at 28.3 mm)


def repush_ok(spec, mem, C, k: int, pos: np.ndarray, quat: np.ndarray, conf: np.ndarray | None = None, check=None) -> bool:
    """R4.1's preconditions on goal k (pushdown), from the scene (truth or estimates) and the runtime memory."""
    from tower_rl.eval.mirror import corners
    from tower_rl.student.spec import RECOVERY_STAGE

    gm = mem.goals[k]
    if gm.last_status != "ok" or RECOVERY_STAGE.get(gm.last_option, gm.last_option) != "push_off":
        return False
    ck = check if check is not None else C.check(k, pos, quat, conf)
    if ck.kind not in LEAN_KINDS:
        return False
    g = spec.goals[k]
    i = C.blocks[k]
    d = np.asarray(g.direction, dtype=float)
    d[2] = 0
    d /= max(np.linalg.norm(d), 1e-9)
    if float((pos[i] - C.pos0[i]) @ d) < PUSHED_M:
        return False
    cs = corners(pos[i], quat[i])
    along = (cs[:, :2] - C.tower_xy) @ d[:2]
    near = cs[along <= along.min() + 0.03]
    if float(near[:, 2].max()) < NEAR_END_Z:
        return False
    layer = g.layer or 0
    for j in range(spec.K):  # no block of a higher layer still on the tower
        if j != k and (spec.goals[j].layer or 0) > layer and C.check(j, pos, quat, conf).kind == "on tower":
            return False
    return True


class RecoveryNode:
    def __init__(self, source: str = "estimate", max_per_goal: int = 2, min_left_s: float = 20.0,
                 health_min: float = 0.7, static_s: float = 1.0, seen_s: float | None = None, at_end: bool = False):
        assert source in ("truth", "estimate"), source
        self.source = source
        self.max_per_goal, self.min_left_s, self.health_min, self.static_s = max_per_goal, min_left_s, health_min, static_s
        self.seen_s = seen_s  # estimate source, opt-in: sporadic confident sightings (module doc)
        self.at_end = at_end  # opt-in: R4.1 only once the plan is done (module doc)
        self.contracts: Contracts | None = None
        self.attempts: dict[tuple[str, int], int] = {}
        self.abstentions: list[dict] = []
        self._abstained: set = set()
        self.log: list[dict] = []
        self.hist: dict[int, list] = {}   # estimate source: per block the recent (t, pos, conf), for "static" and "seen"

    # ------------------------------------------------------------------ the scene
    def _scene(self, view):
        s = view.state
        if self.source == "truth":
            pos = np.asarray(s.privileged.get("true_block_pos", s.block_pos), dtype=float)
            quat = np.asarray(s.privileged.get("true_block_quat", s.block_quat), dtype=float)
            return pos, quat, np.ones(len(pos))
        run = view.run
        pos = np.stack([run.block_rest(s, b)[0][0].cpu().numpy() for b in s.block_ids]).astype(float)
        quat = np.stack([run.block_rest(s, b)[1][0].cpu().numpy() for b in s.block_ids]).astype(float)
        conf = np.asarray(s.block_conf, dtype=float)
        if self.seen_s is not None:  # seen: a confident sighting within seen_s where the rest estimate now is
            conf = np.array([1.0 if c < SEEN and self._sightings(view, i, pos[i]) else c for i, c in enumerate(conf)])
        return pos, quat, conf

    def observe(self, view):
        """Every runtime call (estimate source): the recent perceived poses per block."""
        s = view.state
        keep = max(3.0, self.seen_s or 0.0)
        for i in range(len(s.block_ids)):
            h = self.hist.setdefault(i, [])
            h.append((float(s.t), np.asarray(s.block_pos[i], dtype=float).copy(), float(s.block_conf[i])))
            while h and s.t - h[0][0] > keep:
                h.pop(0)

    def _sightings(self, view, i: int, near=None) -> list:
        """``seen_s``: block i's confident sightings over the last ``seen_s`` (within 4 mm of ``near``, if given)."""
        s = view.state
        h = [x for x in self.hist.get(i, []) if s.t - x[0] <= self.seen_s + 1e-6 and x[2] >= SEEN]
        return h if near is None else [x for x in h if float(np.linalg.norm(x[1] - near)) < 0.004]

    def _static_seen(self, view, i: int) -> bool:
        """Block i is at rest and confidently seen over the last ``static_s`` (truth: its speed; ``seen_s``: its
        confident sightings over that window, >= 2, all within 4 mm of its rest estimate)."""
        s = view.state
        if self.source == "truth":
            sp = s.privileged.get("block_speed")
            return sp is None or float(np.asarray(sp)[i]) < 0.01
        if self.seen_s is not None:
            h = self._sightings(view, i)
            if len(h) < 2:
                return False
            p = np.stack([x[1] for x in h])
            rest = view.run.block_rest(s, s.block_ids[i])[0][0].cpu().numpy()
            return float(np.linalg.norm(p - rest, axis=-1).max()) < 0.004
        h = [x for x in self.hist.get(i, []) if s.t - x[0] <= self.static_s + 1e-6]
        if len(h) < 2 or s.t - h[0][0] < self.static_s * 0.8 or any(c < SEEN for _, _, c in h):
            return False
        p = np.stack([x[1] for x in h])
        return float(np.linalg.norm(p - p[0], axis=-1).max()) < 0.004

    def _health(self, view) -> bool:
        if self.source == "truth":
            return True
        if getattr(view.run, "blind", False):
            return False
        s = view.state
        mov = [s.block_ids.index(b) for b in view.spec.movable]
        return bool(np.mean(np.asarray(s.block_conf)[mov] >= SEEN) >= self.health_min) if mov else True

    def _contracts(self, view, pos, quat) -> Contracts:
        if self.contracts is None:  # the start poses: the first call's scene
            run = view.run
            goal_fn = None
            if view.spec.family == "transfer":
                def goal_fn(k, run=run, view=view):
                    from tower_rl.graph.estimates import yaw_of
                    node = view.spec.goals[k].stages["deliver"][-1]
                    gp, gq = run.dest_goal(view.state, node)
                    return gp[0].cpu().numpy(), yaw_of(gq[0].cpu().numpy())
            self.contracts = Contracts(view.spec, view.state.block_ids, pos, quat, goal_fn)
        return self.contracts

    def start(self, view):
        """The episode's first decision: fix the start poses."""
        pos, quat, _ = self._scene(view)
        self._contracts(view, pos, quat)

    def summary(self) -> dict:
        return {"source": self.source, "decisions": self.log, "abstentions": self.abstentions}

    # ------------------------------------------------------------------ the rules
    def propose(self, view, d: Decision) -> Decision | None:
        spec, s = view.spec, view.state
        if spec.family == "transfer" or EPISODE_S - s.t < self.min_left_s or not self._health(view):
            return None
        pos, quat, conf = self._scene(view)
        C = self._contracts(view, pos, quat)
        mem = view.memory
        if spec.family == "pushdown":
            return self._pushdown(view, d, C, pos, quat, conf, mem)
        return self._insertion(view, d, C, pos, quat, conf, mem)

    def _left(self, kind: str, k: int) -> bool:
        return self.attempts.get((kind, k), 0) < self.max_per_goal

    def _take(self, kind: str, k: int, d: Decision, why: str, view, detail: dict) -> Decision:
        d = Decision(d.option, d.goal, d.arm, recovery=True)
        self.attempts[(kind, k)] = self.attempts.get((kind, k), 0) + 1
        self.log.append({"t": round(view.state.t, 2), "rule": why, "goal": k, "decision": [d.option, d.goal, d.arm],
                         **detail})
        return d

    def _pushdown(self, view, d, C: Contracts, pos, quat, conf, mem) -> Decision | None:
        spec = view.spec
        pushed = [k for k in range(spec.K) if sum(mem.goals[k].attempts.values()) > 0]
        if not pushed:
            return None
        checks = {k: C.check(k, pos, quat, conf) for k in range(spec.K)}
        on_tower = [k for k in range(spec.K) if checks[k].kind == "on tower"]
        end = d.option in ("idle", "park") and not any(view.holding(a) for a in ARMS) and not on_tower
        if self.at_end and not (end and all(k in pushed or checks[k].kind not in ("on tower", "unseen")
                                            for k in range(spec.K))):
            return None  # at_end: every goal pushed or its block seen off the tower, none on it, the decider idling
        for k in pushed:
            ck = checks[k]
            if ck.kind not in LEAN_KINDS:
                continue
            if not repush_ok(spec, mem, C, k, pos, quat, conf, ck):
                self._abstain(view, k, ck, "R4.1 preconditions")
                continue
            i = view.state.block_ids.index(spec.goals[k].block)
            if not self._static_seen(view, i) or not self._left("repush", k):
                continue
            if any(view.holding(a) for a in ARMS):
                continue
            arm = spec.goals[k].arm
            return self._take("repush", k, Decision("repush", k, arm), "R4.1", view,
                              {"kind": ck.kind, **ck.detail})
        if end:
            for k in range(spec.K):
                ck = checks[k]
                if not ck.ok:
                    self._abstain(view, k, ck, "end audit: no rule")
        return None

    def _insertion(self, view, d, C: Contracts, pos, quat, conf, mem) -> Decision | None:
        spec, s = view.spec, view.state
        intact, moved = C.structure(pos, quat)
        if not intact:
            for k in range(spec.K):
                self._abstain(view, k, C.check(k, pos, quat, conf), f"structure moved {moved} mm (prevention's job)")
            return None
        for k in range(spec.K):
            g, gm = spec.goals[k], mem.goals[k]
            if gm.last_option is None:
                continue  # not started: the base decider's
            arm = g.arm
            # R5.1: the insert failed and the arm still holds the block
            if gm.last_option == "insert" and gm.last_status in ("failed", "timeout") and view.holds_goal(arm, k):
                if self._left("insert", k):
                    return self._take("insert", k, Decision("insert", k, arm), "R5.1", view, {"why": gm.last_why})
                continue
            if any(view.holding(a) for a in ARMS):
                continue
            ck = C.check(k, pos, quat, conf)
            if ck.kind != "short":
                if not ck.ok and gm.last_option == "push_home" and d.option in ("idle", "park"):
                    self._abstain(view, k, ck, "end audit: no rule")
                continue
            i = s.block_ids.index(g.block)
            if not self._static_seen(view, i):
                continue
            if self.source == "estimate" and ck.detail["along_mm"] > -15.0:
                continue  # the estimate of a nearly hidden insert is unreliable within a few mm (issue 13)
            if gm.last_option in ("insert", "push_home") and self._left("push_home", k):
                if gm.last_option == "insert" and gm.last_status == "ok" and d.option == "push_home" and d.goal == k:
                    continue  # the base decider pushes it home anyway
                return self._take("push_home", k, Decision("push_home", k, arm), "R5.2", view,
                                  {"kind": ck.kind, **ck.detail})
        return None

    def _abstain(self, view, k: int, ck, why: str):
        key = (k, ck.kind, why)
        if key in self._abstained:
            return
        self._abstained.add(key)
        self.abstentions.append({"t": round(view.state.t, 2), "variant": view.spec.variant, "goal": k, "kind": ck.kind,
                                 "why": why, **ck.detail})
