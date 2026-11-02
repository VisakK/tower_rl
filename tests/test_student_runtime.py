"""Student option runtime (plan Step 2), pure python: a scripted decider and stub nodes drive a Task 2-2 spec through
acquire -> deliver -> park; the mechanical rules (open the gripper of an arm whose block the next option does not
deliver, masks, arm_clear parks), the idle re-query, clipping, and the oracle_graph parity on the fake world."""

import numpy as np
import pytest
import torch

from tower_rl.config import load_yaml
from tower_rl.data.layouts import Layout
from tower_rl.graph.interface import State, Status
from tower_rl.student import runtime as rt
from tower_rl.student.runtime import IdleNode, StudentExecutor, why_category
from tower_rl.student.spec import Decision

V = "task2-2_cross_tower_transfer_2"


def _state(t: float = 0.0, width=(0.1, 0.1)) -> State:
    lay = Layout.load(V)
    ids = list(lay.blocks)
    home = load_yaml("scene")["home_qpos"]
    return State(t=t, step=int(round(t * 10)), variant=V, instruction="", q=np.array([home["left"][:7], home["back"][:7]]),
                 width=np.array(width, dtype=float), block_ids=ids, block_pos=np.stack([lay.blocks[b][:3] for b in ids]),
                 block_quat=np.stack([lay.blocks[b][3:7] for b in ids]), block_conf=np.ones(len(ids)))


class StubNode:
    """Emits ``rows`` rows (one per step) then ends with ``status``; ``close`` sets the arm's gripper closed."""

    plan: dict = {}

    def __init__(self, run, spec, s):
        self.run, self.spec, self.arm, self.info = run, spec, spec["arm"], {}
        self.n, self.status, close = StubNode.plan.get(spec["id"], (2, Status.OK, None))
        if close is not None:
            run.grip[self.arm].reset(closed=close, width=0.04 if close else 0.1)
        self.k = 0

    def step(self, s):
        if self.k >= self.n:
            return self.status, []
        self.k += 1
        return None, [self.run.row()]


class Scripted:
    """Returns its decisions in order (then idle), records the views it saw."""

    name = "scripted"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, view):
        self.calls.append(round(view.state.t, 2))
        return self.decisions.pop(0) if self.decisions else Decision("idle")


def _run(decider, steps, widths=None, plan=None, monkeypatch=None):
    StubNode.plan = plan or {}
    monkeypatch.setattr(rt, "make_node", lambda run, spec, s: StubNode(run, spec, s))
    ex = StudentExecutor(V, decider=decider)
    ex.reset(1, V, [0])
    run = ex.runs[0]
    run._clearance_park = lambda spec, s: None
    t, out = 0.0, []
    for i in range(steps):
        w = widths(t) if widths else (0.1, 0.1)
        chunk = ex.act([_state(t, w)], [0])[0]
        out.append(chunk)
        t = round(t + 0.1 * len(chunk), 6)
    return ex, run, out


def test_acquire_deliver_park_then_idle(monkeypatch):
    d = [Decision("acquire", 0, "left"), Decision("deliver", 0, "left"), Decision("park", 1, "left")]
    # the arm holds the block (closed, block-width opening) between the grasp and the place's release
    plan = {"t0_grasp": (2, Status.OK, True), "t0_place": (2, Status.OK, False)}
    widths = lambda t: (0.0497, 0.1) if 0.35 < t < 0.75 else (0.1, 0.1)  # noqa: E731
    dec = Scripted(d)
    ex, run, out = _run(dec, 70, widths, plan, monkeypatch)
    ids = [e[1] for e in run.events]
    assert ids == ["t0_to_pregrasp", "t0_grasp", "t0_to_preplace", "t0_place", "park_left_1"]
    assert all(e[2] == "ok" for e in run.events)
    assert all(c.shape[1] == 16 and len(c) == 1 for c in out)  # one row per stub step, hold rows when idle
    m = run.memory
    assert m.goals[0].attempts == {"left": 2, "back": 0} and m.goals[0].last_option == "deliver" and m.goals[0].tries == 0
    assert m.arms["left"].moved_since_park is False and m.arms["left"].last_option == "park"
    assert [r["decision"].option for r in m.option_log][:4] == ["acquire", "deliver", "park", "idle"]
    assert m.masked == 0
    # the decider was asked when each option ended (same call), then every 2 s of idle
    idle_calls = dec.calls[3:]
    assert len(idle_calls) >= 3 and all(abs(b - a - 2.0) < 0.11 for a, b in zip(idle_calls, idle_calls[1:]))


def test_failure_aborts_option_and_opens_empty_gripper(monkeypatch):
    d = [Decision("acquire", 0, "left"), Decision("acquire", 0, "left")]
    plan = {"t0_to_pregrasp": (1, Status.FAILED, True)}  # fails with the gripper closed on nothing
    ex, run, _ = _run(Scripted(d), 6, None, plan, monkeypatch)
    assert [e[1] for e in run.events][:2] == ["t0_to_pregrasp", "t0_to_pregrasp"]  # t0_grasp never ran
    assert not bool(run.grip["left"].closed)  # empty-handed after a failure: opened
    assert run.memory.goals[0].tries == 2 and run.memory.goals[0].last_why == "other"


def test_unheld_option_opens_a_holding_gripper(monkeypatch):
    """A failed deliver with the block still held keeps the grip for a retried deliver, but the next option that does
    not deliver it (here a park) opens the gripper first."""
    d = [Decision("acquire", 0, "left"), Decision("deliver", 0, "left"), Decision("deliver", 0, "left"),
         Decision("park", 1, "left")]
    plan = {"t0_grasp": (1, Status.OK, True), "t0_to_preplace": (1, Status.FAILED, None)}
    holding = lambda t: (0.0497, 0.1)  # noqa: E731
    seen = []
    ex, run, _ = _run(Scripted(d), 12, holding, plan, monkeypatch)
    ev = [e[1] for e in run.events]
    assert ev[:4] == ["t0_to_pregrasp", "t0_grasp", "t0_to_preplace", "t0_to_preplace"]
    assert run.memory.masked == 0  # the retried deliver was valid (the arm holds goal 0's block)
    assert ev[4] == "park_left_1" and not bool(run.grip["left"].closed)


def test_masks(monkeypatch):
    """deliver without the block, an option of another family, a goal >= K: each becomes idle."""
    d = [Decision("deliver", 0, "left"), Decision("push_off", 0, "left"), Decision("acquire", 5, "back")]
    ex, run, _ = _run(Scripted(d), 3, None, None, monkeypatch)
    assert run.memory.masked == 1 and run.option.option == "idle" and run.events == []
    view = rt.View(run, _state())
    assert not run.valid(Decision("push_off", 0, "left"), view)
    assert not run.valid(Decision("acquire", 5, "back"), view)
    assert run.valid(Decision("acquire", 1, "back"), view) and run.valid(Decision("park", None, "back"), view)
    assert run.valid(Decision("idle"), view) and not run.valid(Decision("acquire", 0, None), view)


def test_inserted_park_keeps_the_option(monkeypatch):
    """An arm_clear park runs before the transit it guards; the option continues with that transit."""
    d = [Decision("acquire", 0, "left")]
    StubNode.plan = {}
    monkeypatch.setattr(rt, "make_node", lambda run, spec, s: StubNode(run, spec, s))
    ex = StudentExecutor(V, decider=Scripted(d))
    ex.reset(1, V, [0])
    run = ex.runs[0]
    fired = []

    def clearance(spec, s):
        if spec["skill"] != "transit" or fired:
            return None
        fired.append(spec["id"])
        node = StubNode(run, {"id": "auto_park_back", "skill": "park", "arm": "back", "transfer": 0, "timeout_s": 10.0}, s)
        node.inserted = True
        return node

    run._clearance_park = clearance
    run.memory.arms["back"].moved_since_park = True  # the other arm is out somewhere
    t = 0.0
    for _ in range(8):
        t = round(t + 0.1 * len(ex.act([_state(t)], [0])[0]), 6)
    assert [e[1] for e in run.events][:3] == ["auto_park_back", "t0_to_pregrasp", "t0_grasp"]
    assert fired == ["t0_to_pregrasp"]
    assert run.memory.arms["back"].moved_since_park is False


@pytest.mark.parametrize("exact", [False, True])
def test_no_repark_of_a_parked_arm(monkeypatch, exact):
    """arm_clear never re-parks an arm that has not moved since its park (the executor does, forever, when a dropped
    block lies near that arm's home: kept only for exact-executor deciders)."""
    d = Scripted([Decision("acquire", 0, "left")])
    d.exact = exact
    StubNode.plan = {}
    monkeypatch.setattr(rt, "make_node", lambda run, spec, s: StubNode(run, spec, s))
    ex = StudentExecutor(V, decider=d)
    ex.reset(1, V, [0])
    run = ex.runs[0]

    def clearance(spec, s):
        if spec["skill"] != "transit":
            return None
        node = StubNode(run, {"id": "auto_park_back", "skill": "park", "arm": "back", "transfer": 0, "timeout_s": 10.0}, s)
        node.inserted = True
        return node

    run._clearance_park = clearance
    t = 0.0
    for _ in range(12):
        t = round(t + 0.1 * len(ex.act([_state(t)], [0])[0]), 6)
    ids = [e[1] for e in run.events]
    if exact:
        assert ids[:4] == ["auto_park_back"] * 4  # the executor's livelock
    else:
        assert ids[:2] == ["t0_to_pregrasp", "t0_grasp"]


def test_focus_and_node_info(monkeypatch):
    d = [Decision("acquire", 0, "left"), Decision("park", 1, "left"),
         Decision("park", 0, "back", park="alt", alt=True)]
    ex, run, _ = _run(Scripted(d), 1, None, {"t0_to_pregrasp": (5, Status.OK, None)}, monkeypatch)
    block = run.spec.goals[0].block
    assert ex.focus_block(0) == block and ex.node_info(0) == {"skill": "transit", "node": "t0_to_pregrasp", "block": block}
    run.option = Decision("park", 1, "left")
    assert ex.focus_block(0) is None
    run.option = Decision("park", 0, "back", park="alt", alt=True)  # the arm-switch parks keep the goal's block
    assert ex.focus_block(0) == block
    assert ex.focus_block(3) is None and ex.node_info(3) == {}


def test_clip_and_non_finite(monkeypatch):
    ex, run, _ = _run(Scripted([]), 1, None, None, monkeypatch)
    s = _state()
    rows = np.tile(run.row(), (2, 1))
    assert np.array_equal(run._safe(rows, s), rows)  # in limits: unchanged
    rows[0, 0] = 100.0
    out = run._safe(rows, s)
    assert out[0, 0] == run.hi[0] and run.memory.clipped == 1
    rows[1, 3] = np.nan
    out = run._safe(rows, s)
    assert out.shape == (1, 16) and np.isfinite(out).all() and np.allclose(out[0, :7], s.q[0])


def test_idle_node():
    class R:
        def row(self):
            return np.zeros(16, dtype=np.float32)
    n = IdleNode(R(), _state(1.0), 2.0)
    assert n.step(_state(2.9))[0] is None and n.step(_state(3.0))[0] == Status.OK
    # idle_chunk 16 (student_v1_1): 16 rows, then the 4 that reach the re-query time (2 s, as with one-row holds)
    n = IdleNode(R(), _state(1.0), 2.0, chunk=16)
    st, rows = n.step(_state(1.0))
    assert st is None and len(rows) == 16
    st, rows = n.step(_state(1.0 + 16 * 0.1))
    assert st is None and len(rows) == 4
    assert n.step(_state(1.0 + 20 * 0.1))[0] == Status.OK


def test_why_categories():
    assert why_category(Status.OK, {}) == "ok"
    assert why_category(Status.TIMEOUT, {}) == "timeout"
    assert why_category(Status.FAILED, {"why": "empty gripper"}) == "dropped"
    assert why_category(Status.FAILED, {"why": "pregrasp unreachable"}) == "unreachable"
    assert why_category(Status.FAILED, {"why": "structure", "moved_mm": 12}) == "structure"
    assert why_category(Status.FAILED, {"place_xy_mm": 7.0, "yaw_deg": 0.3}) == "misplaced"
    assert why_category(Status.FAILED, {"why": "slide stalled"}) == "stalled"
    assert why_category(Status.FAILED, {}) == "other"


@pytest.mark.parametrize("scenario", ["nominal", "misplace", "armswitch"])
def test_oracle_graph_matches_the_executor_on_the_fake_world(scenario):
    """Rows and node events identical to the executor's (planned skills; the full matrix:
    scripts/student_fake_parity.py)."""
    torch.set_num_threads(1)
    from tower_rl.graph.executor import Executor
    from tower_rl.student.fakeworld import run_side_by_side, scenario as scen
    from tower_rl.student.oracle import Oracle
    v = "task2-3_cross_tower_transfer_3"
    r = run_side_by_side(lambda: Executor(v), lambda: StudentExecutor(v, decider=Oracle("graph")), v, seconds=100.0,
                         **scen(scenario, v))
    assert r["equal"], (r.get("t"), r["events_a"][-3:], r["events_b"][-3:])
    assert len(r["events_a"]) >= 10
    if scenario == "armswitch":
        assert any(e[1].endswith("_alt") for e in r["events_a"])


def test_keep_delivered_blocks_a_redo_until_the_block_moves():
    """Graph flag ``keep_delivered``: after goal 0's deliver ends ok, acquiring goal 0 again is masked while its block
    rests where it was released; once the block is displaced (> 5 mm) it is allowed again."""
    from tower_rl.graph.graph import load_graph
    from tower_rl.student.fakeworld import FakeWorld
    from tower_rl.student.oracle import Oracle
    from tower_rl.student.spec import OPTIONS
    from tower_rl.student.tokens import action_masks

    v = "task2-2_cross_tower_transfer_2"
    ex = StudentExecutor(v, graph=dict(load_graph(v), keep_delivered=True), decider=Oracle("graph", exact=False))
    ex.reset(1, v, [0])
    w = FakeWorld(v)
    run = ex.runs[0]
    while w.t < 60.0 and 0 not in run.delivered:
        for r in ex.act([w.state()], [0])[0]:
            w.apply(r)
    assert 0 in run.delivered
    s = w.state()
    acq = OPTIONS.index("acquire")
    assert run.redo_blocked(0, s) and not action_masks(rt.View(run, s))["valid_goal"][acq, 0]
    assert not run.valid(Decision("acquire", 0, run.spec.goals[0].arm), rt.View(run, s))
    i = w.ids.index(run.spec.goals[0].block)
    w.pos[i, 0] += 0.012  # displaced: the rest estimate restarts after two calls and settles on the new pose
    for _ in range(12):
        run.observe(w.state())
    s = w.state()
    assert not run.redo_blocked(0, s) and action_masks(rt.View(run, s))["valid_goal"][acq, 0]


def test_blind_takes_place_verdicts_as_ok_and_runs_the_graph(monkeypatch):
    """student_v1_1 blind guard: the transfer family switches to the graph's plan, a failed place verdict counts as ok
    (no re-grasp of a block whose verdict compared against nominal poses)."""
    plan = {"t0_grasp": (1, Status.OK, True), "t0_place": (1, Status.FAILED, False)}
    StubNode.plan = plan
    monkeypatch.setattr(rt, "make_node", lambda run, spec, s: StubNode(run, spec, s))
    ex = StudentExecutor(V, decider=Scripted([Decision("acquire", 1, "back")]))
    ex.reset(1, V, [0])
    run = ex.runs[0]
    run._clearance_park = lambda spec, s: None
    run.set_blind()
    assert run.blind and isinstance(run.decider, rt.Oracle) and run.decider.mode == "graph" and not run.decider.exact
    orig = StubNode.step

    def step(self, s):  # the stub place fails with a place verdict, as LearnedPlaceNode does
        st, rows = orig(self, s)
        if st == Status.FAILED and self.spec["skill"] == "place_release":
            self.info = {"place_xy_mm": 7.5, "yaw_deg": 0.4}
        return st, rows

    monkeypatch.setattr(StubNode, "step", step)
    t = 0.0
    for _ in range(12):
        w = (0.0497, 0.1) if 0.15 < t < 0.45 else (0.1, 0.1)
        t = round(t + 0.1 * len(ex.act([_state(t, w)], [0])[0]), 6)
    ev = [(e[1], e[2]) for e in run.events]
    assert ev[:4] == [("t0_to_pregrasp", "ok"), ("t0_grasp", "ok"), ("t0_to_preplace", "ok"), ("t0_place", "ok")]
    assert run.events[3][3].get("blind_verdict") == "failed"
    assert ev[4][0].startswith("park") or ev[4][0] == "t1_to_pregrasp"  # the plan goes on (no re-acquire of goal 0)
    assert run.memory.goals[0].tries == 0


def test_finished_guard_vetoes_rework_of_a_finished_goal(monkeypatch):
    """student_v1_1 finished guard: once every goal ended ok, a goal option is vetoed unless its block stayed moved
    > 10 mm for 2 s."""
    d = [Decision("acquire", 0, "left"), Decision("deliver", 0, "left"), Decision("acquire", 1, "back"),
         Decision("deliver", 1, "back"), Decision("acquire", 0, "left"), Decision("acquire", 0, "left"),
         Decision("acquire", 0, "left")]
    plan = {"t0_grasp": (1, Status.OK, True), "t0_place": (1, Status.OK, False),
            "t1_grasp": (1, Status.OK, True), "t1_place": (1, Status.OK, False)}
    StubNode.plan = plan
    monkeypatch.setattr(rt, "make_node", lambda run, spec, s: StubNode(run, spec, s))
    ex = StudentExecutor(V, decider=Scripted(d))
    ex.reset(1, V, [0])
    ex.graph["finished_guard"] = True
    ex.runs[0].finished_guard = True
    run = ex.runs[0]
    run._clearance_park = lambda spec, s: None
    holds = {0: (0.15, 0.45), 1: (0.65, 0.95)}

    def widths(t):
        if holds[0][0] < t < holds[0][1]:
            return (0.0497, 0.1)
        if holds[1][0] < t < holds[1][1]:
            return (0.1, 0.0497)
        return (0.1, 0.1)

    t = 0.0
    for _ in range(40):
        t = round(t + 0.1 * len(ex.act([_state(t, widths(t))], [0])[0]), 6)
    assert sorted(run.goal_done) == [0, 1]
    assert run.memory.vetoed >= 1  # the re-acquire of goal 0 (its block never moved) became idle
    assert all(e[1] not in ("t0_to_pregrasp",) for e in run.events[4:])
