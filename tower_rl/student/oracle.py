"""The oracle (student plan §2.5, Step 3): the graph's decisions as a function of the scene and the runtime memory.

The executor's decisions hang on its program counter, so they cannot label the states a student reaches after
deviating from the graph. The oracle restates them per goal, from the outcomes of the options the runtime ran
(whoever chose them), in two modes.

Graph mode: exact executor semantics (``EnvRun._advance``). Per goal a stage machine: an ok option advances the stage,
the last stage closes the goal. A failure follows the failed node's ``on_fail``: ``retry`` (default) restarts the goal
at its first stage (the runtime opens the gripper), or retries the transport stage while the arm still holds the
block after a transport failure; ``skip``, a block that fell off the table, or more than ``retries`` failed tries close
the goal. A near-miss place that its own arm cannot re-grasp but the other arm can queues the arm-switch sequence
(park, acquire and deliver with the other arm, park; once per goal; the goal closes when it has run). The graph's
parks: before a goal whose arm differs from the previous goal's, and once at the end; then idle. Reads nothing
privileged, so it is also a deployable decision layer (the fallback).

State mode: the training expert (labels; reads simulator truth, so never deployable). Graph mode with these changes:
1. goal satisfaction comes from truth (``state.privileged["ops"]``, the evaluator's instantaneous P condition);
2. a goal truth says is satisfied is closed and never reworked, even when its node verdict failed;
3. a closed goal that truth no longer finds satisfied is reopened (transfer family only; its block on top of the
   destination stack and on the table level or above; >= ``reopen_min_left_s`` of the episode left; once per goal);
4. a re-grasp (after a failure or a reopen) uses the prescribed arm; when that arm cannot reach the pre-grasp
   (transfer family), the goal is given up. This replaces graph mode's arm-switch sequence: in every past teacher
   rehearsal the other arm's grasp timed out in 206 of 217 attempts and 205 of 211 such episodes failed, after
   ~30-60 s and often collateral damage;
5. parks: before an option of one arm, the other arm is parked if it moved since its last park; at the end, any arm
   that moved since its park (graph mode parks by the graph's structure, even an arm already parked);
6. an arm that holds an open goal's block delivers / inserts it when that goal is next, or (transfer family) when every
   goal of a lower layer is satisfied (labels for states the switcher reaches off the graph).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tower_rl.graph.interface import ARMS, Status
from tower_rl.student.spec import TRANSPORT, Decision

FELL_OFF_Z = -0.02  # a block below the table top: a re-grasp cannot reach it (executor rule)
EPISODE_S = 210.0
# state oracle v2 (recovery): a satisfied goal is reopened only once its block stood > 10 mm from where it was when the
# goal closed for 2 s (the student's finished guard, StudentRun._vetoed): labels the student can see (the v1 rule
# reopened on any tolerance violation, which the estimates cannot tell apart at 5 mm / 5 deg: G7b)
REOPEN_MOVED_M, REOPEN_MOVED_S = 0.010, 2.0
UNGRASPABLE_TILT_DEG = 15.0  # state mode: a block lying tilted (on its side, leaning) is not re-grasped (grasp times out)


@dataclass
class Book:
    """The oracle's bookkeeping in the runtime memory (never a token)."""
    stage: list[int]                 # per goal: index of its next stage
    closed: list[bool]               # per goal: done (succeeded or given up)
    tries: list[int]                 # per goal: failed tries as the executor counts them
    alt_done: set = field(default_factory=set)        # goals that had their arm-switch sequence
    plan: list[Decision] = field(default_factory=list)  # the queued arm-switch sequence
    park_done: set = field(default_factory=set)       # goals whose preceding graph park has run
    end_done: bool = False
    arm: list = field(default_factory=list)           # state mode: per goal, the arm of its current attempt
    satisfied: set = field(default_factory=set)       # state mode: goals closed as satisfied (truth or verdict)
    reopened: set = field(default_factory=set)        # state mode: goals reopened once
    closed_pos: dict = field(default_factory=dict)    # v2: goal -> its block's true position when it closed satisfied
    moved_since: dict = field(default_factory=dict)   # v2: goal -> time its block first stood > 10 mm from there


class Oracle:
    def __init__(self, mode: str = "graph", reopen_min_left_s: float = 30.0, exact: bool | None = None,
                 recovery: bool = False):
        """``recovery`` (state mode): state oracle v2, the teacher's recovery node (``tower_rl.recovery.rules`` on
        simulator truth, Plan-v1.1 §5.3) runs after the state oracle's decision at every decision point."""
        assert mode in ("graph", "state"), mode
        assert not recovery or mode == "state", "the recovery node reads truth: state mode only"
        self.mode = mode
        self.recovery = recovery
        self.name = f"oracle_{mode}" + ("_v2" if recovery else "")
        self.reopen_min_left_s = reopen_min_left_s
        # graph mode asks the runtime to reproduce the executor exactly, its arm_clear livelock included (gate G3a)
        self.exact = (mode == "graph") if exact is None else exact

    def recovery_node(self, view):
        """The episode's recovery node (per run, in the memory's books: the oracle is shared by envs)."""
        node = view.memory.books.get("recovery")
        if node is None:
            from tower_rl.recovery.rules import RecoveryNode

            node = view.memory.books["recovery"] = RecoveryNode("truth")
            node.start(view)
        return node

    def book(self, view) -> Book:
        b = view.memory.books.get(self.mode)
        if b is None:
            K = view.spec.K
            b = view.memory.books[self.mode] = Book([0] * K, [False] * K, [0] * K, arm=[g.arm for g in view.spec.goals])
        return b

    # ------------------------------------------------------------------ decisions
    def decide(self, view) -> Decision:
        if self.mode == "state":
            d = self._decide_state(view)
            if self.recovery:
                node = self.recovery_node(view)
                rd = node.propose(view, d)
                if rd is not None:
                    return rd
            return d
        spec, b = view.spec, self.book(view)
        if b.plan:
            return b.plan[0]
        k = next((g.k for g in spec.goals if not b.closed[g.k]), None)
        if k is None:
            if spec.end_park is not None and not b.end_done:
                return Decision("park", None, spec.end_park["arm"], park="end")
            return Decision("idle")
        p = spec.parks_before.get(k)
        if p is not None and k not in b.park_done:
            return Decision("park", k, p["arm"], park="graph")
        return Decision(spec.stages[b.stage[k]], k, spec.goals[k].arm)

    def _decide_state(self, view) -> Decision:
        spec, b, mem = view.spec, self.book(view), view.memory
        self._refresh(view, b)
        for a in ARMS:  # 6: an arm holding an open goal's block delivers it when that goal may go next
            j = mem.arms[a].holding_goal
            if j is not None and not b.closed[j] and view.holds_goal(a, j) and self._deliverable(view, b, j):
                other = next(x for x in ARMS if x != a)
                if mem.arms[other].moved_since_park:
                    return Decision("park", j, other, park="graph")
                return Decision(next(st for st in spec.stages if st in TRANSPORT), j, a)
        k = next((g.k for g in spec.goals if not b.closed[g.k]), None)
        if k is None:
            a = next((x for x in ARMS if mem.arms[x].moved_since_park), None)
            return Decision("park", None, a, park="end") if a is not None else Decision("idle")
        arm = b.arm[k]
        other = next(x for x in ARMS if x != arm)
        if mem.arms[other].moved_since_park:
            return Decision("park", k, other, park="graph")
        st = spec.stages[b.stage[k]]
        if st in TRANSPORT and not view.holds_goal(arm, k):
            st = spec.stages[0]  # the block is not in hand (dropped, or released by an earlier option): acquire it
        return Decision(st, k, arm)

    def _truth(self, view) -> tuple[list | None, np.ndarray]:
        """Per goal: its own satisfied-now condition from simulator truth (``ops_own``), and the true block positions."""
        s = view.state
        pos = s.privileged.get("true_block_pos", s.block_pos)
        return s.privileged.get("ops_own", s.privileged.get("ops")), np.asarray(pos)

    def _deliverable(self, view, b: Book, j: int) -> bool:
        """A held block of open goal j goes to its goal if j is next, or (transfer) no open goal lies on a lower layer."""
        spec = view.spec
        first = next((g.k for g in spec.goals if not b.closed[g.k]), None)
        if j == first:
            return True
        if spec.family != "transfer":
            return False
        lay = spec.goals[j].dest_layer
        return not any(not b.closed[g.k] for g in spec.goals if g.dest_layer is not None and g.dest_layer < lay)

    def _refresh(self, view, b: Book):
        """Changes 1-3: close goals truth finds satisfied; reopen closed goals it no longer does."""
        spec, s = view.spec, view.state
        ops, pos = self._truth(view)
        if ops is None:
            return
        for g in spec.goals:
            if not b.closed[g.k] and ops[g.k] and not any(view.holds_goal(a, g.k) for a in ARMS):
                b.closed[g.k] = True
                b.satisfied.add(g.k)
        if self.recovery:  # v2: where each satisfied goal's block stands since its goal closed (the reopen rule below)
            for k in b.satisfied:
                i = s.block_ids.index(spec.goals[k].block)
                p0 = b.closed_pos.setdefault(k, pos[i].copy())
                if float(np.linalg.norm(pos[i] - p0)) > REOPEN_MOVED_M:
                    b.moved_since.setdefault(k, s.t)
                else:
                    b.moved_since.pop(k, None)
        if spec.family != "transfer" or EPISODE_S - s.t < self.reopen_min_left_s:
            return
        if any(view.holding(a) for a in ARMS):
            return  # finish what is in hand first: reopening now would make the held block's arm drop it
        for g in spec.goals:
            moved = not self.recovery or s.t - b.moved_since.get(g.k, math.inf) >= REOPEN_MOVED_S - 1e-9
            if b.closed[g.k] and g.k in b.satisfied and not ops[g.k] and g.k not in b.reopened and moved and \
                    self._accessible(view, pos, g.block) and self._tilt_deg(view, pos, g.block) <= UNGRASPABLE_TILT_DEG \
                    and self._can_regrasp(view, g):
                b.closed[g.k] = False
                b.satisfied.discard(g.k)
                b.reopened.add(g.k)
                b.closed_pos.pop(g.k, None)
                b.moved_since.pop(g.k, None)
                b.stage[g.k], b.tries[g.k] = 0, 0

    def _accessible(self, view, pos: np.ndarray, bid: str) -> bool:
        """The block lies at table level or above and no movable block rests on it."""
        s = view.state
        i = s.block_ids.index(bid)
        if pos[i, 2] < FELL_OFF_Z:
            return False
        for b in view.spec.movable:
            j = s.block_ids.index(b)
            if j != i and pos[j, 2] > pos[i, 2] + 0.015 and np.linalg.norm(pos[j, :2] - pos[i, :2]) < 0.09:
                return False
        return True

    def _can_regrasp(self, view, g) -> bool:
        """Change 4: the prescribed arm reaches the block's pre-grasp (transfer family; the other families keep the
        graph's restart)."""
        return view.spec.family != "transfer" or bool(view.run._pregrasp_reachable(g.arm, g.block, view.state))

    # ------------------------------------------------------------------ bookkeeping
    def on_option_end(self, view, d: Decision, status: Status, failed: dict, info: dict):
        """After every option (whoever chose it): ``failed`` is the option's last node spec (the failed one)."""
        if self.mode == "state":
            return self._end_state(view, d, status, failed, info)
        spec, b, s = view.spec, self.book(view), view.state
        if d.alt:  # an item of the arm-switch sequence
            if b.plan and b.plan[0] == d:
                b.plan.pop(0)
            if status != Status.OK and d.option != "park":
                b.plan = [x for x in b.plan if x.option == "park"]  # give up, but park the arm
            if not b.plan:
                b.closed[d.goal] = True
            return
        if d.option == "idle":
            return
        if d.option == "park":
            if d.park == "end":
                b.end_done = True
            elif d.park == "graph" and d.goal is not None:
                b.park_done.add(d.goal)
            return
        k = d.goal
        if status == Status.OK:
            b.stage[k] = spec.stages.index(d.option) + 1
            if b.stage[k] >= len(spec.stages):
                b.closed[k] = True
            return
        policy = failed.get("on_fail", "retry")
        if policy == "skip_if_close":
            err = info.get("place_xy_mm")
            policy = "skip" if err is not None and err / 1000.0 <= failed.get("skip_if_close_m", 0.015) else "retry"
        if policy == "retry" and failed.get("block") and float(s.block(failed["block"])[0][2]) < FELL_OFF_Z:
            policy = "skip"
        if failed["skill"] == "place_release" and self._arm_switch(view, b, k, d.arm, failed, info):
            return
        b.tries[k] += 1
        if policy == "skip" or b.tries[k] > spec.retries:
            b.closed[k] = True
        elif view.holding(d.arm) and failed["skill"] in ("transit", "place_release", "insert") and \
                failed.get("target") in (None, "preplace", "preinsert"):
            b.stage[k] = next(i for i, st in enumerate(spec.stages) if st in TRANSPORT)  # retry the transport
        else:
            b.stage[k] = 0  # restart the goal (the runtime opens the gripper)

    def _end_state(self, view, d: Decision, status: Status, failed: dict, info: dict):
        spec, b = view.spec, self.book(view)
        if d.option in ("park", "idle") or d.goal is None or b.closed[d.goal]:
            return  # parks are in the runtime memory; options on a closed goal (switcher) change nothing here
        k = d.goal
        ops, pos = self._truth(view)
        if ops is not None and ops[k] and not any(view.holds_goal(a, k) for a in ARMS):
            b.closed[k] = True  # change 2: satisfied, whatever the node verdict said
            b.satisfied.add(k)
            return
        if status == Status.OK:
            b.stage[k] = spec.stages.index(d.option) + 1
            b.arm[k] = d.arm
            if b.stage[k] >= len(spec.stages):
                b.closed[k] = True
                b.satisfied.add(k)  # the verdict says done: reopened if truth disagrees (change 3)
            return
        policy = failed.get("on_fail", "retry")
        if policy == "skip_if_close":
            err = info.get("place_xy_mm")
            policy = "skip" if err is not None and err / 1000.0 <= failed.get("skip_if_close_m", 0.015) else "retry"
        if policy == "retry" and failed.get("block"):
            i = view.state.block_ids.index(failed["block"])
            if float(pos[i, 2]) < FELL_OFF_Z:
                policy = "skip"
        b.tries[k] += 1
        if policy == "skip" or b.tries[k] > spec.retries:
            b.closed[k] = True
        elif view.holding(d.arm) and failed["skill"] in ("transit", "place_release", "insert") and \
                failed.get("target") in (None, "preplace", "preinsert"):
            b.stage[k] = next(i for i, st in enumerate(spec.stages) if st in TRANSPORT)
            b.arm[k] = d.arm
        elif "acquire" in spec.stages and self._tilt_deg(view, pos, failed["block"]) > UNGRASPABLE_TILT_DEG:
            b.closed[k] = True  # change 7: a tilted block is not re-grasped (the grasp skill times out, 15 s a try)
        elif not self._can_regrasp(view, spec.goals[k]):
            b.closed[k] = True  # change 4: the prescribed arm cannot reach the re-grasp
        else:
            b.stage[k] = 0
            b.arm[k] = spec.goals[k].arm

    @staticmethod
    def _tilt_deg(view, pos: np.ndarray, bid: str) -> float:
        s = view.state
        q = np.asarray(s.privileged.get("true_block_quat", s.block_quat))[s.block_ids.index(bid)]
        return float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))))

    def _arm_switch(self, view, b: Book, k: int, arm: str, failed: dict, info: dict) -> bool:
        """``EnvRun._schedule_arm_switch``: a place released within ``arm_switch_m`` of its goal that its own arm cannot
        re-grasp (pre-grasp out of reach) but the other arm can is re-grasped and re-placed by the other arm, once."""
        err = info.get("place_xy_mm")
        if k in b.alt_done or err is None or err / 1000.0 > view.spec.arm_switch_m:
            return False
        other = next(a for a in ARMS if a != arm)
        run, s = view.run, view.state
        if run._pregrasp_reachable(arm, failed["block"], s) or not run._pregrasp_reachable(other, failed["block"], s):
            return False
        b.plan = [Decision("park", k, arm, park="alt", alt=True), Decision("acquire", k, other, alt=True),
                  Decision("deliver", k, other, alt=True), Decision("park", k, other, park="alt", alt=True)]
        b.alt_done.add(k)
        return True
