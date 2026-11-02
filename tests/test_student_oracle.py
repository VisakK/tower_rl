"""Student oracle (plan Step 3): graph-mode failure edges and the state-mode changes, on fabricated views (no Isaac).
Graph-mode parity with the executor on whole episodes: tests/test_student_runtime.py and
scripts/student_fake_parity.py (fake world), scripts/compare_runs.py (Isaac rehearsals, gate G3a)."""

import dataclasses

import numpy as np

from tower_rl.config import load_yaml
from tower_rl.data.layouts import Layout
from tower_rl.graph.interface import State, Status
from tower_rl.student.oracle import Oracle
from tower_rl.student.runtime import StudentExecutor, View
from tower_rl.student.spec import Decision


def _setup(variant: str, mode: str):
    oracle = Oracle(mode)
    ex = StudentExecutor(variant, decider=oracle)
    ex.reset(1, variant, [0])
    return oracle, ex.runs[0]


def _state(variant: str, t: float = 0.0, ops=None, width=(0.1, 0.1)) -> State:
    lay = Layout.load(variant)
    ids = list(lay.blocks)
    home = load_yaml("scene")["home_qpos"]
    priv = {} if ops is None else {"ops": list(ops), "ops_own": list(ops)}
    return State(t=t, step=0, variant=variant, instruction="", q=np.array([home["left"][:7], home["back"][:7]]),
                 width=np.array(width, dtype=float), block_ids=ids, block_pos=np.stack([lay.blocks[b][:3] for b in ids]),
                 block_quat=np.stack([lay.blocks[b][3:7] for b in ids]), block_conf=np.ones(len(ids)), privileged=priv)


def _end(oracle, run, s, d, status, failed_id=None, info=None):
    """Record an option end as the runtime does (memory first, then the oracle)."""
    spec = run.spec
    nodes = spec.option_nodes(d)
    failed = next(n for n in nodes if n["id"].startswith(failed_id)) if failed_id else nodes[-1]
    run.memory.start(d, s.t, "test")
    run.memory.end(d, status, info or {}, s.t)
    oracle.on_option_end(View(run, s), d, status, failed, info or {})


def test_graph_mode_sequence_and_skips():
    v = "task5-2_tower_insertion_2"
    o, run = _setup(v, "graph")
    s = _state(v)
    g = run.spec.goals[0]
    assert o.decide(View(run, s)) == Decision("acquire", 0, g.arm)
    _end(o, run, s, Decision("acquire", 0, g.arm), Status.OK)
    assert o.decide(View(run, s)) == Decision("insert", 0, g.arm)
    _end(o, run, s, Decision("insert", 0, g.arm), Status.FAILED, "i0_insert", {"why": "structure"})  # on_fail skip
    assert o.book(View(run, s)).closed == [True]
    assert o.decide(View(run, s)) == Decision("park", None, run.spec.end_park["arm"], park="end")
    _end(o, run, s, Decision("park", None, run.spec.end_park["arm"], park="end"), Status.OK)
    assert o.decide(View(run, s)) == Decision("idle")


def test_graph_mode_retries_then_skip_and_structural_park():
    v = "task2-2_cross_tower_transfer_2"
    o, run = _setup(v, "graph")
    s = _state(v)
    a0 = run.spec.goals[0].arm
    for i in range(3):  # retries 2: three failed grasps close goal 0
        assert o.decide(View(run, s)) == Decision("acquire", 0, a0)
        _end(o, run, s, Decision("acquire", 0, a0), Status.FAILED, "t0_grasp", {"why": "empty gripper"})
    b = o.book(View(run, s))
    assert b.closed[0] and b.tries[0] == 3
    # the graph parks the first arm before goal 1 (other arm), once
    d = o.decide(View(run, s))
    assert d == Decision("park", 1, a0, park="graph") and run.spec.option_nodes(d)[0]["id"] == f"park_{a0}_1"
    _end(o, run, s, d, Status.OK)
    assert o.decide(View(run, s)) == Decision("acquire", 1, run.spec.goals[1].arm)


def test_state_mode_parks_by_motion_and_closes_satisfied_goals():
    v = "task2-2_cross_tower_transfer_2"
    o, run = _setup(v, "state")
    a0, a1 = run.spec.goals[0].arm, run.spec.goals[1].arm
    s = _state(v, ops=[False, False])
    assert o.decide(View(run, s)) == Decision("acquire", 0, a0)  # nothing moved: no park first
    _end(o, run, s, Decision("acquire", 0, a0), Status.OK)
    # the deliver's verdict failed, but truth says goal 0 is satisfied: closed, not reworked (change 2)
    s = _state(v, 30.0, ops=[True, False])
    _end(o, run, s, Decision("deliver", 0, a0), Status.FAILED, "t0_place", {"place_xy_mm": 5.4})
    b = o.book(View(run, s))
    assert b.closed[0] and 0 in b.satisfied and b.tries[0] == 0
    # goal 1 uses the other arm, which moved: park it first (change 5), then acquire
    d = o.decide(View(run, s))
    assert d.option == "park" and d.arm == a0
    _end(o, run, s, d, Status.OK)
    assert o.decide(View(run, s)) == Decision("acquire", 1, a1)


def test_state_mode_reopens_a_disturbed_goal_once():
    v = "task2-2_cross_tower_transfer_2"
    o, run = _setup(v, "state")
    a0 = run.spec.goals[0].arm
    s = _state(v, 10.0, ops=[True, False])
    b = o.book(View(run, s))
    b.closed[0] = True
    b.satisfied.add(0)
    b.stage[0] = 2
    # disturbed at t = 100 s: reopened, re-grasped by the prescribed arm (it reaches the pre-grasp)
    run._pregrasp_reachable = lambda arm, bid, st: True
    s = _state(v, 100.0, ops=[False, False])
    assert o.decide(View(run, s)) == Decision("acquire", 0, a0) and 0 in b.reopened
    # placed again, disturbed again: no second reopen
    b.closed[0] = True
    b.satisfied.add(0)
    d = o.decide(View(run, _state(v, 120.0, ops=[False, False])))
    assert d.goal == 1 or d.option == "park"


def test_state_mode_no_reopen_late_or_buried():
    v = "task2-2_cross_tower_transfer_2"
    o, run = _setup(v, "state")
    run._pregrasp_reachable = lambda arm, bid, st: True
    b = o.book(View(run, _state(v)))
    b.closed[0] = True
    b.satisfied.add(0)
    assert o.decide(View(run, _state(v, 185.0, ops=[False, False]))).goal != 0  # < 30 s left
    s = _state(v, 100.0, ops=[False, False])
    g0, g1 = run.spec.goals[0].block, run.spec.goals[1].block
    pos = s.block_pos.copy()
    pos[s.block_ids.index(g1)] = pos[s.block_ids.index(g0)] + np.array([0.0, 0.0, 0.03])  # goal 1's block on top
    s = dataclasses.replace(s, block_pos=pos)
    assert o.decide(View(run, s)).goal != 0 and 0 not in b.reopened


def test_state_mode_gives_up_what_the_prescribed_arm_cannot_regrasp():
    """Change 4: a failed place whose re-grasp the prescribed arm cannot reach closes the goal (no other-arm attempt);
    a reachable one is retried by the prescribed arm."""
    v = "task2-3_cross_tower_transfer_3"
    o, run = _setup(v, "state")
    g = run.spec.goals[0]
    other = next(a for a in ("left", "back") if a != g.arm)
    s = _state(v, 20.0, ops=[False] * run.spec.K)
    run._pregrasp_reachable = lambda arm, bid, st: arm == other
    _end(o, run, s, Decision("acquire", 0, g.arm), Status.OK)
    _end(o, run, s, Decision("deliver", 0, g.arm), Status.FAILED, "t0_place", {"place_xy_mm": 9.0})
    b = o.book(View(run, s))
    assert b.closed[0] and 0 not in b.satisfied
    d = o.decide(View(run, s))
    assert d.goal != 0
    o2, run2 = _setup(v, "state")
    run2._pregrasp_reachable = lambda arm, bid, st: True
    _end(o2, run2, s, Decision("acquire", 0, g.arm), Status.OK)
    _end(o2, run2, s, Decision("deliver", 0, g.arm), Status.FAILED, "t0_place", {"place_xy_mm": 9.0})
    assert o2.decide(View(run2, s)) == Decision("acquire", 0, g.arm)


def test_state_mode_delivers_a_held_block():
    v = "task2-3_cross_tower_transfer_3"
    o, run = _setup(v, "state")
    g = run.spec.goals[0]
    i = ("left", "back").index(g.arm)
    w = [0.1, 0.1]
    w[i] = 0.0497
    s = _state(v, 20.0, ops=[False] * run.spec.K, width=w)
    run.grip[g.arm].reset(closed=True, width=0.04)
    run.memory.arms[g.arm].holding_goal = 0
    run.memory.arms[g.arm].moved_since_park = True
    assert o.decide(View(run, s)) == Decision("deliver", 0, g.arm)  # change 6


def test_state_mode_defers_reopen_while_holding():
    """A disturbed lower goal is not reopened while an arm holds the next goal's block: the held block is delivered
    first (reopening at once dropped it: smoke run, Task 1-2 D1)."""
    v = "task1-2_tower_transfer_2"
    o, run = _setup(v, "state")
    run._pregrasp_reachable = lambda arm, bid, st: True
    K = run.spec.K
    b = o.book(View(run, _state(v)))
    for k in range(3):
        b.closed[k] = True
        b.satisfied.add(k)
    g3 = run.spec.goals[3]
    width = [0.1, 0.1]
    width[("left", "back").index(g3.arm)] = 0.0497
    run.grip[g3.arm].reset(closed=True, width=0.04)
    run.memory.arms[g3.arm].holding_goal = 3
    s = _state(v, 70.0, ops=[False, True, True, False], width=width)  # goal 0 disturbed
    assert o.decide(View(run, s)) == Decision("deliver", 3, g3.arm) and 0 not in b.reopened


def test_state_mode_skip_buried_is_opt_in():
    """v1.3 (graph "oracle_skip_buried"): a failed goal whose block now has a block resting on it is given up, later
    (``_decide_state``) or at the failure itself (``_end_state``); an accessible one is still retried at once; without
    the flag the old behaviour (restart, also when buried) stands."""
    v = "task2-3_cross_tower_transfer_3"

    def buried(s, run):
        g0, g1 = run.spec.goals[0].block, run.spec.goals[1].block
        pos = s.block_pos.copy()
        pos[s.block_ids.index(g1)] = pos[s.block_ids.index(g0)] + np.array([0.0, 0.0, 0.03])
        return dataclasses.replace(s, block_pos=pos)

    for flag in (False, True):
        o, run = _setup(v, "state")
        run._pregrasp_reachable = lambda arm, bid, st: True
        if flag:
            run.spec.graph = dict(run.spec.graph, oracle_skip_buried=True)
        g = run.spec.goals[0]
        s = _state(v, 20.0, ops=[False] * run.spec.K)
        _end(o, run, s, Decision("acquire", 0, g.arm), Status.OK)
        _end(o, run, s, Decision("deliver", 0, g.arm), Status.FAILED, "t0_place", {"place_xy_mm": 9.0})
        assert o.decide(View(run, s)) == Decision("acquire", 0, g.arm)  # accessible: retried at once either way
        sb = buried(_state(v, 60.0, ops=[False] * run.spec.K), run)
        d = o.decide(View(run, sb))
        closed = o.book(View(run, sb)).closed[0]
        if flag:
            assert d.goal != 0 and closed
        else:
            assert d == Decision("acquire", 0, g.arm) and not closed
    o, run = _setup(v, "state")
    run._pregrasp_reachable = lambda arm, bid, st: True
    run.spec.graph = dict(run.spec.graph, oracle_skip_buried=True)
    g = run.spec.goals[0]
    sb = buried(_state(v, 20.0, ops=[False] * run.spec.K), run)
    _end(o, run, sb, Decision("acquire", 0, g.arm), Status.OK)
    _end(o, run, sb, Decision("deliver", 0, g.arm), Status.FAILED, "t0_place", {"place_xy_mm": 9.0})
    assert o.book(View(run, sb)).closed[0]
