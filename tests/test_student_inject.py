"""Failure injection (student plan Step 6) on the fake world: each perturbation produces its failure through the
normal node checks; off by default (no ``inject`` key in any graph node)."""

import numpy as np
import pytest
import torch

from tower_rl.graph.graph import load_graph
from tower_rl.data.layouts import ALL_VARIANTS
from tower_rl.student.fakeworld import FakeWorld
from tower_rl.student.inject import Injector
from tower_rl.student.oracle import Oracle
from tower_rl.student.runtime import StudentExecutor
from tower_rl.student.spec import Decision

torch.set_num_threads(1)
V = "task2-2_cross_tower_transfer_2"


class Forced(Injector):
    """Injects ``inj`` into the first option of type ``option``."""

    def __init__(self, option, inj):
        self.option, self.inj, self.done = option, inj, False

    def maybe(self, d, nodes):
        if self.done or d.option != self.option:
            return None
        self.done = True
        for n in nodes:
            n["inject"] = dict(self.inj)
        return self.inj


def _episode(option, inj, seconds=60.0):
    ex = StudentExecutor(V, decider=Oracle("graph", exact=False))
    ex.reset(1, V, [0])
    run = ex.runs[0]
    run.injector = Forced(option, inj)
    w = FakeWorld(V)
    q = []
    while w.t < seconds:
        if not q:
            q = list(ex.act([w.state()], [0])[0])
        w.apply(q.pop(0))
    return run, w


def test_no_graph_carries_inject():
    for v in ALL_VARIANTS:
        assert not any("inject" in n for n in load_graph(v)["nodes"])


def test_drop_during_lift():
    run, _ = _episode("acquire", {"drop_at_lift_m": 0.03})
    ev = [(e[1], e[2], e[3].get("why")) for e in run.events]
    assert ("t0_grasp", "failed", "empty gripper") in ev
    assert run.memory.option_log[0]["source"] == "inject"
    assert any(e[1] == "t0_place" and e[2] == "ok" for e in run.events)  # recovered by the retry


def test_drop_in_held_transit():
    run, _ = _episode("deliver", {"drop_after_rows": 8})
    ev = [(e[1], e[2], e[3].get("why")) for e in run.events]
    assert ("t0_to_preplace", "failed", "dropped") in ev


def test_goal_shift_misplaces_the_block():
    run, w = _episode("deliver", {"goal_shift": [0.012, 0.0, 0.0]}, seconds=30.0)
    place = next(e for e in run.events if e[1] == "t0_place")
    assert place[2] == "ok"  # placed where the shifted goal said
    g = run.spec.goals[0]
    i = w.ids.index(g.block)
    gp, _ = run.dest_goal(w.state(), g.stages["deliver"][-1])
    assert abs(float(w.pos[i, 0] - gp[0, 0]) - 0.012) < 0.004  # 12 mm off the true goal


def test_shifted_pregrasp_and_yaw_reach_the_nodes():
    for option, inj in (("acquire", {"target_shift": [0.02, 0.0, 0.0]}), ("deliver", {"goal_yaw_deg": 6.0})):
        run, _ = _episode(option, inj, seconds=12.0)
        assert run.memory.option_log[0 if option == "acquire" else 1]["source"] == "inject"


def test_injector_sampling():
    from tower_rl.config import load_yaml
    cfg = load_yaml("student/collect")["inject"]
    inj = Injector(cfg, 1.0, np.random.default_rng(0))
    kinds = set()
    for _ in range(200):
        for d, nodes in ((Decision("acquire", 0, "left"), [{}]), (Decision("deliver", 0, "left"), [{}]),
                         (Decision("push_off", 0, "left"), [{"direction": [1.0, 0.0, 0.0]}]),
                         (Decision("insert", 0, "left"), [{"direction": [0.0, 1.0, 0.0]}])):
            kinds |= set(inj.maybe(d, nodes))
    assert kinds == {"target_shift", "drop_at_lift_m", "goal_shift", "goal_yaw_deg", "drop_after_rows", "slot_shift"}
    assert Injector(cfg, 0.0, np.random.default_rng(0)).maybe(Decision("acquire", 0, "left"), [{}]) is None
    assert inj.maybe(Decision("park", None, "left"), [{}]) is None


def test_grasp_along_and_goal_filter():
    """v1.3: an off-centre grip on purpose (``grasp_along``, both signs) and the per-option goal filter."""
    cfg = {"acquire": {"lateral_m": [0.01, 0.03], "drop_at_lift_m": [0.025, 0.05], "grasp_along_m": [0.004, 0.008],
                       "grasp_along_frac": 1.0, "goals": [1]},
           "deliver": {"goal_shift_m": [0.005, 0.015], "goal_yaw_deg": [4, 8], "drop_after_rows": [5, 25], "goals": []}}
    inj = Injector(cfg, 1.0, np.random.default_rng(0))
    assert inj.maybe(Decision("acquire", 0, "left"), [{"id": "a"}]) is None
    assert inj.maybe(Decision("deliver", 1, "left"), [{"id": "b"}]) is None
    signs = set()
    for _ in range(20):
        nodes = [{"id": "t1_to_pregrasp"}, {"id": "t1_grasp"}]
        out = inj.maybe(Decision("acquire", 1, "left"), nodes)
        assert set(out) == {"grasp_along"} and 0.004 <= abs(out["grasp_along"]) <= 0.008
        assert all(n["inject"] == out for n in nodes)
        signs.add(out["grasp_along"] > 0)
    assert signs == {True, False}


def test_old_configs_draw_as_before():
    """Without the v1.3 keys an acquire injection consumes the same random draws as before (old seeds reproduce)."""
    import math

    from tower_rl.config import load_yaml

    cfg = load_yaml("student/collect")["inject"]
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    inj = Injector(cfg, 0.5, a)
    for k in range(40):
        got = inj.maybe(Decision("acquire", k % 3, "left"), [{"id": "x"}])
        if b.random() >= 0.5:  # the pre-v1.3 sequence of draws
            want = None
        elif b.random() < 0.5:
            ang = b.uniform(0, 2 * math.pi)
            r = b.uniform(*cfg["acquire"]["lateral_m"])
            want = {"target_shift": [r * math.cos(ang), r * math.sin(ang), 0.0]}
        else:
            want = {"drop_at_lift_m": float(b.uniform(*cfg["acquire"]["drop_at_lift_m"]))}
        assert got == want
