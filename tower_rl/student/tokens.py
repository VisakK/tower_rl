"""Switcher tokens (student plan Step 4, Appendix B): State + runtime memory + task spec -> fixed-size float32 arrays.

Deployable inputs only: perceived block poses and confidences (the estimator, or EstimatorNoise in training), joint
angles, gripper widths, the runtime's own commands and memory, kinematics and the task spec. ``state.privileged`` is
never read (tests/test_student_tokens.py). Block orientations enter only through half-turn-symmetric quantities
(cos / sin 2 psi, tilt), so q and q * Rz(pi) give the same tokens.

Tokens per decision: ``blk`` [16, 38] (+ mask), ``goal`` [9, 88] (+ mask), ``arm`` [2, 27], ``glob`` [16], and the
action masks ``valid_opt`` [7], ``valid_goal`` [7, 9], ``valid_arm`` [7, 9, 2] (an option without a goal keeps its arm
mask in goal slot 0). ``TokenTracker.observe`` must see every call's state (histories: start poses, speeds, sightings).

Features v2 (Plan-v1.1 §5.4, the recovery node distilled into the switcher): the v1 tokens plus, per goal, the
contracts (``tower_rl.recovery.contracts``) judged on the runtime's rest estimates (the kind one-hot over
``CONTRACT_KINDS``, gap / z / tilt / along / lateral / height-or-distance), whether its final stage ended ok, how far
its block moved since and how long ago, and its recovery options so far (``goal`` [9, 106]); globally the perception
health (share of movable blocks seen), BLIND, every goal done, and the structure's largest move (``glob`` [20]). The
option set is ``OPTIONS_V2`` (+ ``repush``): masks [8, ...].
"""

from __future__ import annotations

import math

import numpy as np

from tower_rl import geometry as geo
from tower_rl.data.layouts import ALL_VARIANTS, Layout
from tower_rl.graph.estimates import yaw_of
from tower_rl.graph.interface import ARMS, State
from tower_rl.student.spec import FAMILIES, GOAL_TYPES, OPTIONS, RECOVERY_OPTIONS, RECOVERY_STAGE, TRANSPORT, WHY

FEATURES = "v1"
MAX_BLOCKS, MAX_GOALS = 16, 9
BLK_DIM, GOAL_DIM, ARM_DIM, GLOB_DIM = 38, 88, 27, 16
OPTIONS_V2 = OPTIONS + RECOVERY_OPTIONS
OPTION_SETS = {"v1": OPTIONS, "v2": OPTIONS_V2}
CONTRACT_KINDS = ("ok", "pending", "over base", "tipped", "on block", "out", "short", "unseen")
KIND_OF = {"flat clear": "ok", "home": "ok", "in tolerance": "ok", "on tower": "pending", "on/leaning on base": "over base",
           "touching base": "over base", "tipped": "tipped", "on a block": "on block", "off the table": "out",
           "out": "out", "off goal": "out", "no slot": "out", "no goal": "out", "short": "short", "unseen": "unseen"}
GOAL_EXTRA_DIM = len(CONTRACT_KINDS) + 6 + 3 + 1
GLOB_EXTRA_DIM = 4
DIMS = {"v1": {"blk": BLK_DIM, "goal": GOAL_DIM, "arm": ARM_DIM, "glob": GLOB_DIM},
        "v2": {"blk": BLK_DIM, "goal": GOAL_DIM + GOAL_EXTRA_DIM, "arm": ARM_DIM, "glob": GLOB_DIM + GLOB_EXTRA_DIM}}
FINAL_STAGE = {"transfer": "deliver", "pushdown": "push_off", "insertion": "push_home"}
ROLES = ("source_foundation", "destination_foundation", "block", "side", "roof", "insert")
CONF_LEVELS = (0.95, 0.5, 1e-6)  # >= 1, 0.6, 0.3, 0 (estimator levels; truth = 1)
STATUS = ("none", "ok", "failed", "timeout")
ALT = ("none", "queued", "done")
HELD_WIDTH = (0.044, 0.056)
HELD_RADIUS_M = 0.03
HOME_TOL_RAD = 0.05
EPISODE_S = 210.0
REACH_MOVE_M = 0.003  # reachability is recomputed when a block's rest pose moved this much


def onehot(i: int | None, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if i is not None and 0 <= i < n:
        v[i] = 1.0
    return v


def conf_level(c: float) -> int:
    return next((i for i, th in enumerate(CONF_LEVELS) if c >= th), 3)


def tilt_of(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


class TokenTracker:
    """Per-episode histories behind the tokens, and the tokenizer itself (``features`` "v1" or "v2")."""

    def __init__(self, run, features: str = FEATURES):
        assert features in DIMS, features
        self.features = features
        self.contracts = None   # v2: the contracts on the first call's rest estimates
        self._last_state = None
        self.done_at: dict[int, tuple[float, np.ndarray]] = {}  # v2: goal -> (time, rest position) its final stage ended ok
        self.run = run
        self.spec = run.spec
        lay = Layout.load(self.spec.variant)
        self.src_c = np.asarray(lay.pads["source"]["center"], dtype=np.float64)
        self.dst_c = np.asarray(lay.pads["destination"]["center"], dtype=np.float64)
        self.dst_yaw = math.radians(lay.pads["destination"]["foundation_yaw_deg"])
        fnd = self.spec.foundation
        self.role = {}
        for b in lay.blocks:
            r = lay.role(b)
            if b in fnd.get("source", []):
                r = "source_foundation"
            elif b in fnd["destination"]:
                r = "destination_foundation"
            self.role[b] = ROLES.index(r) if r in ROLES else ROLES.index("block")
        self.p0: np.ndarray | None = None
        self.hist: list[tuple[float, np.ndarray]] = []   # (t, positions) of the last ~1.5 s
        self.unseen: np.ndarray | None = None            # calls since the confidence was last >= 0.5
        self.zero: np.ndarray | None = None              # consecutive calls with confidence 0
        self.reach: dict = {}                            # (arm, block) -> (rest position, reachable)
        self.calls = 0

    # ------------------------------------------------------------------ histories
    def observe(self, s: State):
        pos = np.asarray(s.block_pos, dtype=np.float64)
        conf = np.asarray(s.block_conf, dtype=np.float64)
        if self.p0 is None:
            self.p0 = pos.copy()
            self.unseen = np.zeros(len(pos))
            self.zero = np.zeros(len(pos))
        self.unseen = np.where(conf >= 0.5, 0.0, self.unseen + 1)
        self.zero = np.where(conf <= 0.0, self.zero + 1, 0.0)
        self.hist.append((float(s.t), pos.copy()))
        while len(self.hist) > 2 and self.hist[1][0] <= s.t - 1.0:
            self.hist.pop(0)
        self.calls += 1

    def _speed(self, s: State) -> np.ndarray:
        if len(self.hist) < 2:
            return np.zeros(len(s.block_ids))
        t0, p0 = self.hist[0]
        t1, p1 = self.hist[-1]
        return np.linalg.norm(p1 - p0, axis=-1) / max(t1 - t0, 0.1)

    def reachable(self, arm: str, bid: str, s: State, still: bool) -> bool:
        """``EnvRun._pregrasp_reachable`` (~0.1 s of IK), cached: recomputed only once a block that moved
        ``REACH_MOVE_M`` from its cached rest pose is ``still`` again (a held or moving block keeps its last value)."""
        p = self.run.block_rest(s, bid)[0][0].cpu().numpy()
        hit = self.reach.get((arm, bid))
        if hit is None or (still and np.linalg.norm(hit[0] - p) > REACH_MOVE_M):
            hit = self.reach[(arm, bid)] = (p, bool(self.run._pregrasp_reachable(arm, bid, s)))
        return hit[1]

    # ------------------------------------------------------------------ goal geometry (the runtime's own targets)
    def goal_pose(self, k: int, s: State):
        """Goal k's target pose (pos [3], yaw) from perception, as the nodes compute it; None for push goals."""
        g = self.spec.goals[k]
        if self.spec.family == "pushdown":
            return None
        node = g.stages["deliver" if self.spec.family == "transfer" else "insert"][-1]
        if self.spec.family == "insertion":
            p, q, _ = self.run.slot_pose(s, node)
        else:
            p, q = self.run.dest_goal(s, node)
        return p[0].cpu().numpy().astype(np.float64), float(geo.yaw_from_quat(q)[0])

    # ------------------------------------------------------------------ tokens
    def tokens(self, view) -> dict[str, np.ndarray]:
        s, run, spec, mem = view.state, view.run, view.spec, view.memory
        if self.p0 is None:
            self.observe(s)
        ids = list(s.block_ids)
        nb = len(ids)
        assert nb <= MAX_BLOCKS and spec.K <= MAX_GOALS
        pos = np.asarray(s.block_pos, dtype=np.float64)
        quat = np.asarray(s.block_quat, dtype=np.float64)
        conf = np.asarray(s.block_conf, dtype=np.float64)
        speed = self._speed(s)
        pinch = {a: run.pinch(s, a) for a in ARMS}
        pp = {a: pinch[a][0][0].cpu().numpy().astype(np.float64) for a in ARMS}
        width = {a: float(s.width[ARMS.index(a)]) for a in ARMS}
        closed = {a: bool(run.grip[a].closed) for a in ARMS}
        held_by = np.zeros(nb, dtype=np.int64)  # 0 none, 1 left, 2 back
        for ai, a in enumerate(ARMS):
            if closed[a] and HELD_WIDTH[0] <= width[a] <= HELD_WIDTH[1]:
                d = np.linalg.norm(pos - pp[a], axis=-1)
                for i in np.flatnonzero(d <= HELD_RADIUS_M):
                    if held_by[i] == 0:
                        held_by[i] = ai + 1
        goals = {k: self.goal_pose(k, s) for k in range(spec.K)}
        bound = {g.block: g.k for g in spec.goals}
        tower_xy = None
        if spec.family == "pushdown":
            tower_xy = np.mean([run.block_rest(s, b)[0][0].cpu().numpy() for b in spec.tower["foundation"]], 0)[:2]
        blk = np.zeros((MAX_BLOCKS, BLK_DIM), dtype=np.float32)
        for i, b in enumerate(ids):
            p, yaw = pos[i], yaw_of(quat[i])
            f = [onehot(self.role.get(b, 2), 6),
                 np.r_[(p - self.src_c) * 10, (p - self.dst_c) * 10],
                 [math.cos(2 * yaw), math.sin(2 * yaw)],
                 [tilt_of(quat[i])],
                 onehot(conf_level(conf[i]), 4),
                 [min(1.0, self.unseen[i] / 10)],
                 onehot(int(held_by[i]), 3),
                 [np.linalg.norm(p - self.p0[i]) * 10, speed[i] * 10]]
            k = bound.get(b)
            goal_off = np.zeros(5)
            push = np.zeros(3)
            slot = np.zeros(3)
            if k is not None and goals[k] is not None:
                gp, gyaw = goals[k]
                dpsi = yaw - gyaw
                goal_off = np.r_[(p - gp) * 100, math.cos(2 * dpsi), math.sin(2 * dpsi)]
            if k is not None and spec.family == "pushdown":
                d = np.asarray(spec.goals[k].direction, dtype=np.float64)
                lat = np.array([-d[1], d[0], 0.0])
                mv = p - self.p0[i]
                push = np.r_[float(mv @ d) * 10, float(mv @ lat) * 10, np.linalg.norm(p[:2] - tower_xy) * 10]
            if k is not None and spec.family == "insertion":
                node = spec.goals[k].stages["insert"][-1]
                sp, _, sd = run.slot_pose(s, node)
                sp, sd = sp[0].cpu().numpy(), sd[0].cpu().numpy()
                lat = np.array([-sd[1], sd[0], 0.0])
                slot = np.r_[float((p - sp) @ sd), float((p - sp) @ lat), p[2] - sp[2]] * 10
            fallen = float(p[2] < 0.0 or self.zero[i] >= 20)
            f += [goal_off, push, slot, [float(k is not None)], [fallen]]
            blk[i] = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in f])
        blk_mask = np.zeros(MAX_BLOCKS, dtype=bool)
        blk_mask[:nb] = True

        goal = np.zeros((MAX_GOALS, GOAL_DIM), dtype=np.float32)
        for g in spec.goals:
            gm = mem.goals[g.k]
            i = ids.index(g.block)
            geo_f = np.zeros(5)
            if goals[g.k] is not None:
                gp, gyaw = goals[g.k]
                geo_f = np.r_[(gp - self.dst_c) * 10, math.cos(2 * (gyaw - self.dst_yaw)), math.sin(2 * (gyaw - self.dst_yaw))]
            elif spec.family == "pushdown":
                geo_f = np.r_[(self.p0[i] - self.dst_c) * 10, 0.0, 0.0]
            hist = [[gm.attempts[a] / 3 for a in ARMS], [gm.tries / 3],
                    onehot(OPTIONS.index(RECOVERY_STAGE.get(gm.last_option, gm.last_option)) if gm.last_option else None, 7),
                    onehot(STATUS.index(gm.last_status) if gm.last_status in STATUS else 0, 4),
                    onehot(WHY.index(gm.last_why) if gm.last_why in WHY else WHY.index("other"), 11),
                    onehot(ALT.index(gm.alt), 3)]
            reach = np.zeros(2)
            if "acquire" in g.stages:
                still = held_by[i] == 0 and speed[i] < 0.01
                reach = np.array([float(self.reachable(a, g.block, s, still)) for a in ARMS])
            f = [onehot(GOAL_TYPES.index(g.type), 3), [g.k / spec.K], onehot(g.k, 9), onehot(ARMS.index(g.arm), 2),
                 blk[i], geo_f] + hist + [reach]
            goal[g.k] = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in f])
        goal_mask = np.zeros(MAX_GOALS, dtype=bool)
        goal_mask[:spec.K] = True

        arm = np.zeros((2, ARM_DIM), dtype=np.float32)
        home = run.home
        for ai, a in enumerate(ARMS):
            am = mem.arms[a]
            R = geo.quat_to_mat(pinch[a][1])[0].cpu().numpy()
            q = np.asarray(s.q[ai], dtype=np.float64)
            at_home = float(np.abs(q - np.asarray(home[a][:7])).max() < HOME_TOL_RAD)
            since = 1.0 if am.t_end is None else min(1.0, (s.t - am.t_end) / 10)
            holding = float(run.holding(a, s))
            f = [onehot(ai, 2), np.r_[(pp[a] - self.src_c) * 10, (pp[a] - self.dst_c) * 10], R[:, :2].T.ravel(),
                 [width[a] * 10], [float(closed[a])], [holding], [at_home], [float(am.moved_since_park)],
                 onehot(OPTIONS.index(RECOVERY_STAGE.get(am.last_option, am.last_option)) if am.last_option else None, 7), [since]]
            arm[ai] = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in f])

        glob = np.concatenate([onehot(ALL_VARIANTS.index(spec.variant), 10), onehot(FAMILIES.index(spec.family), 3),
                               [(EPISODE_S - s.t) / EPISODE_S], [spec.K / MAX_GOALS], [mem.decisions / 50]]).astype(np.float32)
        if self.features == "v2":
            goal, glob = self._v2(view, goal, glob, conf)
        out = {"blk": blk, "blk_mask": blk_mask, "goal": goal, "goal_mask": goal_mask, "arm": arm, "glob": glob}
        out.update(action_masks(view, self.features))
        return out

    def _v2(self, view, goal: np.ndarray, glob: np.ndarray, conf: np.ndarray):
        """Features v2: the contracts on the rest estimates, done / moved since, recovery options; health globally."""
        from tower_rl.recovery.contracts import Contracts

        s, run, spec, mem = view.state, view.run, view.spec, view.memory
        rest = [run.block_rest(s, b) for b in s.block_ids]
        pos = np.stack([p[0].cpu().numpy() for p, _ in rest]).astype(np.float64)
        quat = np.stack([q[0].cpu().numpy() for _, q in rest]).astype(np.float64)
        if self.contracts is None:
            goal_fn = None
            if spec.family == "transfer":
                def goal_fn(k, tk=self, s=s):
                    gp, gyaw = tk.goal_pose(k, tk._last_state or s)
                    return gp, gyaw
            self.contracts = Contracts(spec, s.block_ids, pos, quat, goal_fn)
        self._last_state = s
        C = self.contracts
        out = np.zeros((MAX_GOALS, DIMS["v2"]["goal"]), dtype=np.float32)
        out[:, :GOAL_DIM] = goal
        n_rec = {}
        for r in mem.option_log:
            d = r.get("decision")
            if d is not None and d.option in RECOVERY_OPTIONS and d.goal is not None:
                n_rec[d.goal] = n_rec.get(d.goal, 0) + 1
        all_done = True
        for g in spec.goals:
            gm = mem.goals[g.k]
            ck = C.check(g.k, pos, quat, conf)
            kind = KIND_OF.get(ck.kind, "out")
            if spec.family == "transfer" and not ck.ok:
                i = s.block_ids.index(g.block)
                if float(np.linalg.norm(pos[i] - C.pos0[i])) < 0.02:
                    kind = "pending"
            det = ck.detail
            cont = [det.get("gap_mm", 0.0) / 50, det.get("z_mm", 0.0) / 50, det.get("tilt_deg", 0.0) / 90,
                    det.get("along_mm", 0.0) / 100, det.get("lat_mm", 0.0) / 20,
                    det.get("dz_mm", det.get("xy_mm", 0.0)) / 20]
            done = gm.last_status == "ok" and RECOVERY_STAGE.get(gm.last_option, gm.last_option) == FINAL_STAGE[spec.family]
            all_done &= bool(done)
            i = s.block_ids.index(g.block)
            if done and g.k not in self.done_at:
                self.done_at[g.k] = (float(s.t), pos[i].copy())
            elif not done:
                self.done_at.pop(g.k, None)
            moved = age = 0.0
            if g.k in self.done_at:
                t0, p0 = self.done_at[g.k]
                moved = min(5.0, float(np.linalg.norm(pos[i] - p0)) * 1000 / 20)
                age = min(1.0, (s.t - t0) / 30)
            x = np.concatenate([onehot(CONTRACT_KINDS.index(kind), len(CONTRACT_KINDS)),
                                np.clip(np.asarray(cont, dtype=np.float32), -5, 5), [float(done), moved, age],
                                [n_rec.get(g.k, 0) / 2]]).astype(np.float32)
            out[g.k, GOAL_DIM:] = x
        mov = [s.block_ids.index(b) for b in spec.movable]
        health = float(np.mean(conf[mov] >= 0.5)) if mov else 1.0
        _, moved_mm = C.structure(pos, quat)
        g2 = np.concatenate([glob, [health, float(getattr(run, "blind", False)), float(all_done), min(3.0, moved_mm / 10)]])
        return out, g2.astype(np.float32)


def action_masks(view, features: str = FEATURES) -> dict[str, np.ndarray]:
    """The runtime's masks (``StudentRun.valid``) as arrays: options of the family; goals < K; deliver / insert only for
    the goal whose block the arm holds; park any arm; idle always (its goal and arm rows stay empty). v2: a recovery
    option only for a goal whose stage it repeats has run, with no arm holding a block."""
    spec = view.spec
    opts = OPTION_SETS[features]
    vo = np.zeros(len(opts), dtype=bool)
    vg = np.zeros((len(opts), MAX_GOALS), dtype=bool)
    va = np.zeros((len(opts), MAX_GOALS, 2), dtype=bool)
    for oi, o in enumerate(opts):
        if o not in spec.options:
            continue
        if o == "idle":
            vo[oi] = True
        elif o == "park":
            vo[oi] = True
            va[oi, 0, :] = True
        else:
            rec_ok = None
            if o in RECOVERY_OPTIONS:  # the recovery node's preconditions, on the rest estimates (tower_rl.recovery.rules)
                rec_ok = _recovery_ok(view, o)
            for k in range(spec.K):
                for ai, a in enumerate(ARMS):
                    ok = view.holds_goal(a, k) if o in TRANSPORT else True
                    if rec_ok is not None:
                        ok = bool(rec_ok[k]) and a == spec.goals[k].arm
                    va[oi, k, ai] = ok
                if o == "acquire" and getattr(view.run, "keep_delivered", False) and view.run.redo_blocked(k, view.state):
                    va[oi, k] = False  # opt-in: a delivered goal whose block has not moved (StudentRun.redo_blocked)
                vg[oi, k] = va[oi, k].any()
            vo[oi] = vg[oi].any()
    return {"valid_opt": vo, "valid_goal": vg, "valid_arm": va}


def _recovery_ok(view, option: str) -> list[bool]:
    """Per goal: the recovery option's preconditions hold (features v2 masks; ``repush``: ``rules.repush_ok`` on the
    rest estimates, nothing held)."""
    from tower_rl.recovery.rules import repush_ok

    spec, run, s = view.spec, view.run, view.state
    if option != "repush" or spec.family != "pushdown" or any(view.holding(a) for a in ARMS):
        return [False] * spec.K
    if spec.graph.get("router_repush", True) is False:
        return [False] * spec.K  # v1.3: Task 4's re-push stays with the recovery node (graph "recovery"), not the router
    tok = getattr(run, "tok", None)
    C = getattr(tok, "contracts", None) if tok is not None else None
    if C is None:
        return [False] * spec.K
    rest = [run.block_rest(s, b) for b in s.block_ids]
    pos = np.stack([p[0].cpu().numpy() for p, _ in rest]).astype(np.float64)
    quat = np.stack([q[0].cpu().numpy() for _, q in rest]).astype(np.float64)
    conf = np.asarray(s.block_conf, dtype=np.float64)
    return [repush_ok(spec, view.memory, C, k, pos, quat, conf) for k in range(spec.K)]


def encode(d, features: str = FEATURES) -> tuple[int, int, int]:
    """A Decision as (option, goal, arm) indices; -1 where the option takes no goal / arm (features v1: a recovery option
    as the stage it reuses, v1 has no recovery head)."""
    oi = OPTIONS_V2.index(d.option) if features == "v2" else OPTIONS.index(RECOVERY_STAGE.get(d.option, d.option))
    gi = d.goal if d.option not in ("park", "idle") and d.goal is not None else -1
    ai = ARMS.index(d.arm) if d.arm is not None and d.option != "idle" else -1
    return oi, gi, ai


def decode(oi: int, gi: int, ai: int, features: str = FEATURES):
    """(option, goal, arm) indices -> Decision; an option missing its arm (or goal) decodes to idle."""
    from tower_rl.student.spec import Decision
    o = OPTION_SETS[features][oi]
    if o == "idle" or ai < 0 or (o != "park" and gi < 0):
        return Decision("idle")
    if o == "park":
        return Decision("park", None, ARMS[ai])
    return Decision(o, int(gi), ARMS[ai])
