"""Option runtime (student plan Step 2): the decider's options run through the executor's node classes.

``StudentRun`` is a thin ``EnvRun`` subclass. It keeps the executor's controllers, grippers, rest estimates, goals,
held-block attachment and node classes; instead of a graph pointer it holds the running option's node queue. When an
option ends (its last node ok, or any node failed), the decider (``tower_rl.student.oracle.Oracle`` or the learned
switcher) chooses the next (option, goal, arm) in the same ``act`` call, as the executor advances its pointer, and the
runtime applies the mechanical rules (safety filters, not decisions):

* an option that does not deliver / insert the block an arm holds first opens that arm's gripper;
* before a transit into a tower the other arm occupies, the other arm is parked (``EnvRun._clearance_park``: an
  inserted node, the option continues after it), unless it has not moved since its last park;
* masks: deliver / insert only for the goal whose block that arm holds, only the family's options, goal < K (an
  invalid decision becomes idle and is counted);
* idle holds the current command and asks the decider again after ``idle_requery_s`` of sim time;
* rows are clipped to the joint and gripper limits; a non-finite chunk becomes a hold chunk.

Node events ``(t, node id, status, info)`` are logged exactly as the executor logs them, so a decider with the graph's
semantics reproduces the executor's event sequence (gate G3a).
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from tower_rl.config import load_yaml
from tower_rl.graph.executor import (EnvRun, Executor, LearnedGraspNode, LearnedPlaceNode, PlannedGraspNode,
                                     PlannedPlaceNode, PushInNode, PushNode, InsertNode, make_node)
from tower_rl.graph.interface import ARMS, State, Status
from tower_rl.student.inject import Injector
from tower_rl.student.logging import row as collect_row
from tower_rl.student.oracle import Oracle
from tower_rl.student.spec import OPTIONS, TRANSPORT, WHY, Decision, TaskSpec
from tower_rl.student.tokens import TokenTracker, action_masks, encode

GOAL_OPTIONS = ("acquire", "deliver", "push_off", "insert", "push_home")
IDLE_SPEC = {"id": "idle", "skill": "idle", "arm": None, "timeout_s": math.inf}
CONTACT_NODES = (PlannedGraspNode, LearnedGraspNode, PlannedPlaceNode, LearnedPlaceNode, PushNode, InsertNode, PushInNode)
GRIPPER_LIMITS = (0.0, 0.1)
CLIP_TOL = 1e-6  # rows beyond the limits by less than this are float rounding, kept as they are


def why_category(status: Status, info: dict | None) -> str:
    """The failure category of a node's end (goal-token history feature)."""
    if status == Status.OK:
        return "ok"
    if status == Status.TIMEOUT:
        return "timeout"
    info = info or {}
    w = str(info.get("why", ""))
    if w in ("dropped", "empty gripper"):
        return "dropped"
    for key in ("unreachable", "structure", "shoved", "stalled", "lost", "timeout"):
        if key in w:
            return key
    if not w and any(k in info for k in ("place_xy_mm", "along_mm", "lat_mm")):
        return "misplaced"  # a place / insert / push-in verdict outside tolerance
    return "other"


# ---------------------------------------------------------------------- runtime memory
@dataclass
class GoalMemory:
    tries: int = 0                 # failed options on this goal (any arm)
    attempts: dict = field(default_factory=lambda: {a: 0 for a in ARMS})  # options started, per arm
    last_option: str | None = None
    last_status: str | None = None  # ok | failed | timeout
    last_why: str = "none"          # WHY
    alt: str = "none"               # the non-prescribed arm on this goal: none | queued (started) | done (ended)


@dataclass
class ArmMemory:
    last_option: str | None = None
    moved_since_park: bool = False
    t_end: float | None = None      # sim time the arm's last option ended
    holding_goal: int | None = None  # goal whose block this arm acquired (valid while the gripper holds a block)


class Memory:
    """The runtime's own record of its options and their outcomes (deployable: no simulator truth)."""

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.goals = [GoalMemory() for _ in spec.goals]
        self.arms = {a: ArmMemory() for a in ARMS}
        self.option_log: list[dict] = []
        self.decisions = 0
        self.masked = 0     # decisions the masks turned into idle
        self.clipped = 0    # action values the limits changed
        self.books: dict = {}  # the oracles' bookkeeping, by mode (never a token)

    def start(self, d: Decision, t: float, source: str):
        self.decisions += 1
        if d.option in GOAL_OPTIONS:
            g = self.goals[d.goal]
            g.attempts[d.arm] += 1
            if d.arm != self.spec.goals[d.goal].arm and g.alt == "none":
                g.alt = "queued"
        if d.arm is not None and d.option != "idle":
            self.arms[d.arm].last_option = d.option
            if d.option != "park":
                self.arms[d.arm].moved_since_park = True
        self.option_log.append({"decision": d, "t_start": round(t, 2), "source": source})

    def end(self, d: Decision, status: Status, info: dict, t: float):
        if d.option in GOAL_OPTIONS:
            g = self.goals[d.goal]
            g.last_option, g.last_status, g.last_why = d.option, status.value, why_category(status, info)
            if status != Status.OK:
                g.tries += 1
            if d.arm != self.spec.goals[d.goal].arm and (status != Status.OK or d.option in TRANSPORT):
                g.alt = "done"
            if d.option == "acquire" and status == Status.OK:
                self.arms[d.arm].holding_goal = d.goal
        if d.arm is not None and d.option != "idle":
            self.arms[d.arm].t_end = t
            if d.option == "park":
                self.parked(d.arm)
        if self.option_log:
            self.option_log[-1].update(t_end=round(t, 2), status=status.value, info=info)

    def parked(self, arm: str):
        self.arms[arm].moved_since_park = False
        self.arms[arm].holding_goal = None


@dataclass
class View:
    """What a decider sees: the run (kinematic helpers), this call's state, the spec and the memory. Deciders other
    than the state-mode oracle must not read ``state.privileged``."""
    run: StudentRun
    state: State

    @property
    def spec(self) -> TaskSpec:
        return self.run.spec

    @property
    def memory(self) -> Memory:
        return self.run.memory

    def holding(self, arm: str) -> bool:
        return self.run.holding(arm, self.state)

    def holds_goal(self, arm: str, k: int) -> bool:
        return self.holding(arm) and self.memory.arms[arm].holding_goal == k


# ---------------------------------------------------------------------- the per-env runtime
class IdleNode:
    """Hold the current command; end after ``requery_s`` of sim time (the runtime then asks the decider again)."""

    inserted = False

    def __init__(self, run: StudentRun, s: State, requery_s: float):
        self.run, self.spec, self.arm, self.info = run, IDLE_SPEC, None, {}
        self.t0, self.requery_s = s.t, requery_s

    def step(self, s: State):
        if s.t - self.t0 >= self.requery_s - 1e-9:
            return Status.OK, []
        return None, [self.run.row()]


class StudentRun(EnvRun):
    def __init__(self, ex: StudentExecutor, seed: int):
        super().__init__(ex, seed)
        self.spec: TaskSpec = ex.spec
        self.decider = ex.decider
        self.labeler = ex.labeler      # an oracle labelling every decision (data collection), or None
        self.queue: list[dict] = []    # node specs of the running option; the head is the running / next node
        self.option: Decision | None = None
        self.memory = Memory(self.spec)
        self.requery_s = ex.idle_requery_s
        self.exact = bool(getattr(self.decider, "exact", False))  # reproduce the executor exactly (gate G3a)
        lim = self.ctrl["left"].kin.limits.cpu().numpy()
        lo = np.concatenate([lim[:, 0], [GRIPPER_LIMITS[0]]])
        hi = np.concatenate([lim[:, 1], [GRIPPER_LIMITS[1]]])
        self.lo, self.hi = np.concatenate([lo, lo]).astype(np.float32), np.concatenate([hi, hi]).astype(np.float32)
        self.collect = ex.collect      # record decision / progress rows (tower_rl.student.logging)
        self.rows: list[dict] = []
        self.calls = 0
        need_tokens = self.collect or getattr(self.decider, "needs_tokens", False)
        self.tok = TokenTracker(self) if need_tokens else None
        self.perturbed = False  # an explored or injected option happened in this episode
        self.option_prev, self.idle_streak = None, 0
        self.rng = np.random.default_rng([seed, 991])  # exploration, injection, the DAgger mix: deterministic per episode
        # opt-in (graph "keep_delivered"): a goal whose deliver ended ok is not acquired again while its block stays
        # where it was released (rest estimate within 5 mm). bc_r0 re-grasped correctly placed first blocks of Task 1-3
        # (place yaw ~4.3 deg, near the 5 deg tolerance: the state oracle re-acquires only when truth says outside, so
        # the tokens cannot tell those labels apart) and re-placed them 10-14 mm off (G7b: 1-3 D0 18 / 24 vs 24 / 24)
        self.keep_delivered = bool(self.g.get("keep_delivered", False))
        self.delivered: dict[int, np.ndarray] = {}
        self.injector = Injector(load_yaml("student/collect")["inject"], ex.p_inject, self.rng) if ex.p_inject > 0 else None

    def holding(self, arm: str, s: State) -> bool:
        """The executor's holding test: the gripper is commanded closed and the measured opening is a block's width."""
        lo, hi = self.ex.cfg.held_width
        return bool(self.grip[arm].closed) and lo <= float(s.width[ARMS.index(arm)]) <= hi

    # ------------------------------------------------------------------ main loop
    def act(self, s: State) -> np.ndarray:
        if not self.ready:
            self._cmd = {}
            for a in ARMS:
                qm = self.q_meas(s, a)
                self.ctrl[a].reset(qm)
                self.command(a)
                self.grip[a].reset(width=float(s.width[ARMS.index(a)]) if s.width[ARMS.index(a)] > 0.09 else 0.1)
            self.ready = True
        self.observe(s)
        self.calls += 1
        if self.tok is not None:
            self.tok.observe(s)
            if self.collect and self.calls % self.ex.progress_every == 0:
                r = collect_row(self.tok.tokens(View(self, s)), 1, t=s.t, priv=s.privileged)
                r["nonnominal"] = int(self.perturbed or any(g.tries > 0 for g in self.memory.goals))
                self.rows.append(r)
        rows: list[np.ndarray] = []
        for _ in range(64):  # nodes that finish without emitting rows must not stall the loop
            if self.node is None:
                if not self.queue:
                    self._decide(s)
                self.node = self._make(s)
            status, out = self.node.step(s)
            if out and self.ex.cfg.rate_hz == 30 and isinstance(self.node, CONTACT_NODES):
                out = self._expand(out)  # 10 Hz decisions -> 3 interpolated 30 Hz rows each
            if out and not isinstance(self.node, IdleNode):
                self._last_row = out[-1]
            rows += out
            if status is not None:
                self._node_done(status, s)
            if rows:
                break
        return self._safe(np.stack(rows) if rows else self.row()[None], s)

    def _make(self, s: State):
        spec = self.queue[0]
        if spec["skill"] == "idle":
            return IdleNode(self, s, self.requery_s)
        if not self.option.alt:  # the executor runs its arm-switch nodes without the arm_clear check
            other = next((a for a in ARMS if a != spec["arm"]), None)
            # an arm already parked (not moved since) cannot get further out of the way: re-parking it is a livelock
            # (executor: a block dropped within ARM_CLEAR_RADIUS_M of the other arm's home pinch re-parks it until the
            # episode ends; found on the fake world). Exact-executor deciders (oracle_graph) keep that behaviour.
            if self.exact or spec["skill"] != "transit" or self.memory.arms[other].moved_since_park:
                park = self._clearance_park(spec, s)
                if park is not None:
                    return park
        return make_node(self, spec, s)

    def _node_done(self, status: Status, s: State):
        node, self.node = self.node, None
        spec = node.spec
        if spec["skill"] != "idle":
            self.events.append((round(s.t, 2), spec["id"], status.value, node.info))
        if getattr(node, "inserted", False):  # arm_clear park: the option continues with the node it preceded
            self.memory.parked(spec["arm"])
            return
        self.queue.pop(0)
        if status != Status.OK:
            self.queue.clear()  # a failed node aborts the rest of its option
            if spec["skill"] not in ("park", "idle") and not self.holding(spec["arm"], s):
                self._open(spec["arm"])  # empty-handed after a failure: open (a held block stays held)
        elif self.queue:
            return
        self._option_end(status, s, spec, node.info)
        self._decide(s)

    def _option_end(self, status: Status, s: State, spec: dict, info: dict):
        d = self.option
        self.memory.end(d, status, info, s.t)
        if self.keep_delivered and d is not None and d.option == "deliver":
            if status == Status.OK:
                self.delivered[d.goal] = self.block_rest(s, self.spec.goals[d.goal].block)[0][0].cpu().numpy().copy()
            else:
                self.delivered.pop(d.goal, None)
        view = View(self, s)
        for o in self._books():
            o.on_option_end(view, d, status, spec, info)
        self.option = None

    def _books(self) -> list:
        """The deciders that keep bookkeeping across options (the oracle, also when it only labels)."""
        out = []
        for o in (self.decider, self.labeler):
            if o is not None and hasattr(o, "on_option_end") and all(o is not p for p in out):
                out.append(o)
        return out

    def _decide(self, s: State):
        view = View(self, s)
        tokens = self.tok.tokens(view) if self.collect else None  # before the decision changes the memory
        d = self.decider.decide(view)
        source = "oracle" if isinstance(self.decider, Oracle) else "switcher"
        label = d if self.labeler is None or self.labeler is self.decider else self.labeler.decide(view)
        if self.ex.beta is not None and self.labeler is not None and self.labeler is not self.decider \
                and self.rng.random() < self.ex.beta:
            d, source = label, "oracle"  # DAgger mixture: the expert acts with probability beta
        if self.ex.explore_eps > 0 and self.rng.random() < self.ex.explore_eps:
            alt = self._alternative(d, tokens if tokens is not None else action_masks(view))
            if alt is not None:
                d, source = alt, "explore"
        if not self.valid(d, view):
            self.memory.masked += 1
            d = Decision("idle")
        nodes = self.spec.option_nodes(d) or [dict(IDLE_SPEC)]
        if self.injector is not None and d.option in GOAL_OPTIONS and self.injector.maybe(d, nodes) is not None \
                and source != "explore":
            source = "inject"  # the expert's decision, executed with a perturbation (data collection)
        if source in ("explore", "inject"):
            self.perturbed = True
        self.idle_streak = self.idle_streak + 1 if (d.option == "idle" and self.option_prev == "idle") else 0
        self.option_prev = d.option
        if self.collect and self.idle_streak % 10 == 0:  # a long idle (after the end) is logged once per 10 re-queries
            r = collect_row(tokens, 0, label, d, source, self.ex.beta, s.t, s.privileged)
            # non-nominal: after a failed option, or after exploration / injection in this episode (weighted x5 in BC)
            r["nonnominal"] = int(self.perturbed or any(g.tries > 0 for g in self.memory.goals))
            self.rows.append(r)
        for a in ARMS:  # an option that does not deliver / insert the block an arm holds first opens that arm
            if self.holding(a, s) and not (d.option in TRANSPORT and d.arm == a and d.goal == self.memory.arms[a].holding_goal):
                self._open(a)
        self.option = d
        self.memory.start(d, s.t, source)
        self.queue = nodes

    def _alternative(self, d: Decision, masks: dict) -> Decision | None:
        """A uniformly random valid decision other than ``d`` (exploration: another goal, the other arm, park, idle)."""
        cands = []
        for oi, o in enumerate(OPTIONS):
            if not masks["valid_opt"][oi]:
                continue
            if o == "idle":
                cands.append(Decision("idle"))
            elif o == "park":
                cands += [Decision("park", None, a) for ai, a in enumerate(ARMS) if masks["valid_arm"][oi, 0, ai]]
            else:
                cands += [Decision(o, k, a) for k in range(self.spec.K) for ai, a in enumerate(ARMS)
                          if masks["valid_arm"][oi, k, ai]]
        cands = [c for c in cands if encode(c) != encode(d)]
        return cands[int(self.rng.integers(len(cands)))] if cands else None

    def redo_blocked(self, k: int, s: State) -> bool:
        """``keep_delivered``: goal k's deliver ended ok and its block rests within 5 mm of where it was released."""
        p = self.delivered.get(k)
        if p is None:
            return False
        now = self.block_rest(s, self.spec.goals[k].block)[0][0].cpu().numpy()
        return float(np.linalg.norm(now - p)) <= 0.005

    def valid(self, d: Decision, view: View) -> bool:
        """The action masks (plan §2.2)."""
        if d.option == "idle":
            return True
        if self.keep_delivered and d.option == "acquire" and d.goal is not None and self.redo_blocked(d.goal, view.state):
            return False
        if d.option not in self.spec.options or d.arm is None:
            return False
        if d.option == "park":
            return True
        if d.goal is None or not 0 <= d.goal < self.spec.K:
            return False
        if d.option in TRANSPORT:
            return view.holds_goal(d.arm, d.goal)
        return True

    def _open(self, arm: str):
        self.grip[arm].reset(closed=False, width=float(self.grip[arm].width))
        self.detach(arm)
        self.memory.arms[arm].holding_goal = None

    def _safe(self, rows: np.ndarray, s: State) -> np.ndarray:
        """Rows clipped to the joint / gripper limits; a non-finite chunk becomes a hold chunk (measured joints)."""
        if not np.isfinite(rows).all():
            hold = np.concatenate([s.q[0], [float(self.grip["left"].width)], s.q[1], [float(self.grip["back"].width)]])
            return np.nan_to_num(hold.astype(np.float32), nan=0.1)[None]
        if self.exact:  # the executor's rows as they are: its parks command up to ~0.03 rad past a joint limit
            return rows  # from extreme postures (sag compensation is not clamped there); the server clips to the track
        # only real violations: Isaac reports an open gripper at 0.10000002 m and the executor commands that back, so
        # clipping at float precision changed rows by 2e-8 and the physics amplified it (the shadow parity found it)
        bad = (rows < self.lo - CLIP_TOL) | (rows > self.hi + CLIP_TOL)
        if not bad.any():
            return rows
        self.memory.clipped += int(bad.sum())
        return np.where(bad, np.clip(rows, self.lo, self.hi), rows).astype(np.float32)

    # ------------------------------------------------------------------ for the perception loop and logs
    def focus_block(self) -> str | None:
        """The block the running option acts on (the estimator's focus); None for parks and idle (the arm-switch
        sequence's parks keep their goal's block, as the executor's pointer stays on the failed place)."""
        d = self.option
        if d is None or d.goal is None or (d.option in ("park", "idle") and not d.alt):
            return None
        return self.spec.goals[d.goal].block

    def node_info(self) -> dict:
        if not self.queue or self.queue[0]["skill"] == "idle":
            return {}
        nd = self.queue[0]
        return {"skill": nd.get("skill"), "node": nd.get("id"), "block": nd.get("block")}


class StudentExecutor(Executor):
    """The runner's ``Policy``: one ``StudentRun`` per env. ``graph`` is the spec's graph dict (``--graph_set``
    updates it before ``reset``, which builds the spec)."""

    def __init__(self, variant: str, graph: dict | None = None, skills: dict | None = None, cfg=None, decider=None,
                 labeler=None, idle_requery_s: float | None = None, collect: bool = False, progress_every: int = 5,
                 explore_eps: float = 0.0, beta: float | None = None, p_inject: float = 0.0):
        super().__init__(variant, graph, skills, cfg)
        self.decider = decider
        self.labeler = labeler
        if collect and labeler is None:  # labels come from the state-mode oracle
            self.labeler = decider if isinstance(decider, Oracle) and decider.mode == "state" else Oracle("state")
        self.collect, self.progress_every = collect, progress_every
        self.explore_eps, self.beta, self.p_inject = explore_eps, beta, p_inject
        self.idle_requery_s = idle_requery_s if idle_requery_s is not None else \
            load_yaml("student/switcher")["decide"]["idle_requery_s"]
        self.spec: TaskSpec | None = None

    @classmethod
    def from_bundle(cls, bundle: str | None, variant: str, device: str = "cpu", rate_hz: int = 10, decider=None,
                    labeler=None, **kw) -> StudentExecutor:
        ex = Executor.from_bundle(bundle, variant, device=device, rate_hz=rate_hz)
        return cls(variant, ex.graph, ex.skills, ex.cfg, decider, labeler, **kw)

    def reset(self, n_envs: int, variant: str, seeds: list[int]):
        assert variant == self.variant
        self.spec = TaskSpec.from_graph(self.graph)
        self.runs = [StudentRun(self, s) for s in seeds[:n_envs]]

    def focus_block(self, e: int) -> str | None:
        return self.runs[e].focus_block() if e < len(self.runs) else None

    def node_info(self, e: int) -> dict:
        return self.runs[e].node_info() if e < len(self.runs) else {}

    def memory(self, e: int) -> Memory:
        return self.runs[e].memory


def make_decider(name: str):
    """``oracle_graph`` / ``oracle_state`` (``tower_rl.student.oracle``), or a switcher export (``*.pt``, Step 7)."""
    if name in ("oracle_graph", "oracle_state"):
        return Oracle(name.split("_", 1)[1])
    from tower_rl.student.switcher import SwitcherDecider  # Step 7
    return SwitcherDecider.load(name)


class ShadowPolicy:
    """Gate G3a without PhysX history: the ``primary`` policy's rows drive the simulation while the ``shadow`` gets the
    same states; per env, the first call where their chunks differ is recorded, and at the end whether their node events
    are equal. (Separate processes drift apart after any code change, BUILDLOG session 1, so event equality across two
    runs cannot isolate the decision layer.)"""

    def __init__(self, primary, shadow, dump: str | None = None):
        self.primary, self.shadow = primary, shadow
        self.graphs = [primary.graph, shadow.graph]
        self.dump = dump  # dir: pickle the states up to each env's first inexact call (debugging)

    def reset(self, n_envs: int, variant: str, seeds: list[int]):
        self.primary.reset(n_envs, variant, seeds)
        self.shadow.reset(n_envs, variant, seeds)
        self.first = [None] * n_envs   # first call whose chunks differ beyond ``tol`` (a logic difference)
        self.calls = [0] * n_envs
        self.max_abs = [0.0] * n_envs  # largest row difference over the episode (float noise is ~1e-8)
        self.inexact = [0] * n_envs    # calls whose chunks are not bit-identical
        self.history = [[] for _ in range(n_envs)]

    tol = 1e-4

    def act(self, states: list[State], env_ids: list[int]) -> list[np.ndarray]:
        a = self.primary.act(states, env_ids)
        b = self.shadow.act(states, env_ids)
        for e, s, x, y in zip(env_ids, states, a, b):
            self.calls[e] += 1
            same_shape = x.shape == y.shape
            d = float(np.abs(x - y).max()) if same_shape else float("inf")
            if d > 0:
                self.inexact[e] += 1
                self.max_abs[e] = max(self.max_abs[e], d)
                if self.dump is not None and self.inexact[e] == 1:  # debugging: the first inexact call's state
                    import pickle
                    Path(self.dump).mkdir(parents=True, exist_ok=True)
                    with open(Path(self.dump) / f"env{e}_call{self.calls[e]}.pkl", "wb") as f:
                        pickle.dump({"state": s, "states_so_far": self.history[e], "a": x, "b": y}, f)
            if self.dump is not None:
                self.history[e].append(s)
            if self.first[e] is None and d > self.tol:
                ev = self.primary.runs[e].events
                self.first[e] = {"t": round(float(s.t), 2), "call": self.calls[e], "shape": [list(x.shape), list(y.shape)],
                                 "max_abs": d, "last_event": ev[-1][1] if ev else None}
        return a

    def episode_log(self, e: int) -> list:
        return self.primary.episode_log(e)

    def episode_extra(self, e: int) -> dict:
        return {"shadow": {"calls": self.calls[e], "first_mismatch": self.first[e], "max_abs": self.max_abs[e],
                           "inexact_calls": self.inexact[e],
                           "events_equal": self.primary.runs[e].events == self.shadow.runs[e].events}}

    def focus_block(self, e: int):
        return self.primary.focus_block(e)

    def node_info(self, e: int) -> dict:
        return self.primary.node_info(e)

    @property
    def runs(self):
        return self.primary.runs
