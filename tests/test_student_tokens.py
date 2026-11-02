"""Switcher tokens (plan Step 4): shapes and masks on every variant, no reads of ``privileged``, half-turn symmetry."""

import dataclasses

import numpy as np
import pytest
import torch

from tower_rl.data.layouts import ALL_VARIANTS
from tower_rl.student.fakeworld import FakeWorld
from tower_rl.student.oracle import Oracle
from tower_rl.student.runtime import StudentExecutor, View
from tower_rl.student.spec import OPTIONS, Decision
from tower_rl.student.tokens import (ARM_DIM, BLK_DIM, GLOB_DIM, GOAL_DIM, MAX_BLOCKS, MAX_GOALS, action_masks, decode,
                                     encode)

torch.set_num_threads(1)


class Forbidden(dict):
    """A privileged dict that fails on any read."""

    def _no(self, *a, **k):
        raise AssertionError("tokens read State.privileged")

    __getitem__ = get = __contains__ = keys = items = values = __iter__ = __len__ = _no


def _run(variant):
    ex = StudentExecutor(variant, decider=Oracle("graph"), collect=True)
    ex.reset(1, variant, [0])
    return ex.runs[0]


def _flip(q: np.ndarray) -> np.ndarray:
    """q * Rz(pi) (wxyz): the same box."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([-z, y, -x, w], -1)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_shapes_masks_and_no_privileged(variant):
    run = _run(variant)
    w = FakeWorld(variant)
    s = dataclasses.replace(w.state(), privileged=Forbidden())
    run.observe(s)
    run.tok.observe(s)
    run._cmd = {}
    for a in ("left", "back"):
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
    tk = run.tok.tokens(View(run, s))
    assert tk["blk"].shape == (MAX_BLOCKS, BLK_DIM) and tk["goal"].shape == (MAX_GOALS, GOAL_DIM)
    assert tk["arm"].shape == (2, ARM_DIM) and tk["glob"].shape == (GLOB_DIM,)
    assert all(tk[k].dtype == np.float32 for k in ("blk", "goal", "arm", "glob"))
    assert tk["blk_mask"].sum() == len(s.block_ids) and tk["goal_mask"].sum() == run.spec.K
    assert np.isfinite(tk["blk"]).all() and np.isfinite(tk["goal"]).all() and np.isfinite(tk["arm"]).all()
    assert not tk["blk"][~tk["blk_mask"]].any() and not tk["goal"][~tk["goal_mask"]].any()
    # nothing held: transport options masked, the family's other options valid for every goal and arm
    vo = dict(zip(OPTIONS, tk["valid_opt"]))
    assert vo["idle"] and vo["park"] and not vo["deliver"] and not vo["insert"]
    first = run.spec.stages[0]
    assert vo[first] and tk["valid_goal"][OPTIONS.index(first), :run.spec.K].all()
    assert not tk["valid_goal"][OPTIONS.index(first), run.spec.K:].any()
    other = {"transfer": "push_off", "pushdown": "acquire", "insertion": "deliver"}[run.spec.family]
    assert not vo[other]
    # bound flag on the goal blocks only; the goal token carries its block's features
    for g in run.spec.goals:
        i = s.block_ids.index(g.block)
        assert tk["blk"][i, 36] == 1.0
        assert np.array_equal(tk["goal"][g.k, 15:53], tk["blk"][i])


@pytest.mark.parametrize("variant", ["task2-3_cross_tower_transfer_3", "task4-2_full_tower_pushdown_2",
                                     "task5-3_tower_insertion_3"])
def test_half_turn_symmetry(variant):
    out = []
    for flip in (False, True):
        run = _run(variant)
        w = FakeWorld(variant)
        s = w.state()
        if flip:
            s = dataclasses.replace(s, block_quat=_flip(s.block_quat))
        run.act(s)
        out.append(run.tok.tokens(View(run, s)))
    for k in ("blk", "goal", "arm", "glob"):
        assert np.allclose(out[0][k], out[1][k], atol=1e-5), k


def test_transport_mask_follows_the_held_block():
    v = "task2-2_cross_tower_transfer_2"
    run = _run(v)
    w = FakeWorld(v)
    g = run.spec.goals[0]
    width = [0.1, 0.1]
    width[("left", "back").index(g.arm)] = 0.0497
    s = dataclasses.replace(w.state(), width=np.array(width, dtype=np.float32))
    run.act(s)
    run.grip[g.arm].reset(closed=True, width=0.04)
    run.memory.arms[g.arm].holding_goal = 0
    m = action_masks(View(run, s))
    oi = OPTIONS.index("deliver")
    assert m["valid_opt"][oi] and m["valid_goal"][oi].tolist() == [True] + [False] * (MAX_GOALS - 1)
    assert m["valid_arm"][oi, 0].tolist() == [g.arm == "left", g.arm == "back"]


def test_encode_decode():
    for d in (Decision("acquire", 2, "back"), Decision("deliver", 0, "left"), Decision("idle"),
              Decision("park", None, "left")):
        assert decode(*encode(d)) == d
    assert encode(Decision("park", 3, "back", park="graph")) == (OPTIONS.index("park"), -1, 1)


@pytest.mark.parametrize("variant", ["task4-2_full_tower_pushdown_2", "task5-3_tower_insertion_3", "task2-3_cross_tower_transfer_3"])
def test_features_v2_shapes_masks_and_network(variant):
    """Features v2 (the recovery node in the switcher): the contract tokens on top of v1, the repush option in the
    masks, a v2 network's decision decodes to a valid option; still no reads of privileged."""
    from tower_rl.student.spec import RECOVERY_OPTIONS
    from tower_rl.student.switcher import SwitcherNet
    from tower_rl.student.tokens import DIMS, OPTIONS_V2

    ex = StudentExecutor(variant, decider=Oracle("graph"), collect=True, features="v2")
    ex.reset(1, variant, [0])
    run = ex.runs[0]
    w = FakeWorld(variant)
    s = dataclasses.replace(w.state(), privileged=Forbidden())
    run.observe(s)
    run.tok.observe(s)
    run._cmd = {}
    for a in ("left", "back"):
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
    tk = run.tok.tokens(View(run, s))
    assert tk["goal"].shape == (MAX_GOALS, DIMS["v2"]["goal"]) and tk["glob"].shape == (DIMS["v2"]["glob"],)
    assert tk["valid_opt"].shape == (len(OPTIONS_V2),) and np.isfinite(tk["goal"]).all()
    kinds = tk["goal"][: run.spec.K, GOAL_DIM:GOAL_DIM + 8]
    assert (kinds.sum(-1) == 1).all()  # one contract kind per goal
    ri = OPTIONS_V2.index(RECOVERY_OPTIONS[0])
    assert not tk["valid_opt"][ri]  # nothing pushed yet: no re-push
    net = SwitcherNet(64, 1, 4, 128, 0.0, features="v2")
    t = {k: torch.as_tensor(np.asarray(v))[None] for k, v in tk.items()}
    out, prog = net.decide(t["blk"], t["blk_mask"], t["goal"], t["goal_mask"], t["arm"], t["glob"], t["valid_opt"],
                           t["valid_goal"], t["valid_arm"])
    o, g, a = (int(x) for x in out[0])
    d = decode(o, g, a, "v2")
    assert d.option in run.spec.options and encode(d, "v2")[0] == o
    assert encode(Decision("repush", 1, "left"), "v2")[0] == ri and encode(Decision("repush", 1, "left"))[0] == OPTIONS.index("push_off")


def test_router_repush_flag_masks_the_routers_repush(monkeypatch):
    """v1.3 graph "router_repush": false masks the router's repush option whatever R4.1's preconditions say (Task 4's
    re-push stays with the recovery node, graph "recovery")."""
    import tower_rl.recovery.rules as rules
    from tower_rl.student.tokens import _recovery_ok

    v = "task4-2_full_tower_pushdown_2"
    ex = StudentExecutor(v, decider=Oracle("graph"), collect=True, features="v2")
    ex.reset(1, v, [0])
    run = ex.runs[0]
    s = FakeWorld(v).state()
    run.observe(s)
    run.tok.observe(s)
    run._cmd = {}
    for a in ("left", "back"):
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
    run.tok.tokens(View(run, s))
    monkeypatch.setattr(rules, "repush_ok", lambda *a, **k: True)
    assert all(_recovery_ok(View(run, s), "repush"))
    run.spec.graph = dict(run.spec.graph, router_repush=False)
    assert not any(_recovery_ok(View(run, s), "repush"))
