"""Checkpoint predicates and the executor's arm_clear safety net (no simulator)."""

import math

import numpy as np
import torch

from tower_rl.graph import checkpoints as ck
from tower_rl.graph.executor import Executor, ParkNode
from tests.test_graph import _state


def _yaw_q(deg):
    h = math.radians(deg) / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def test_placement_error_and_static():
    dxy, dz, dyaw = ck.placement_error([0.103, 0.0, 0.05], _yaw_q(181.0), [0.1, 0.004, 0.044], _yaw_q(0.0))
    assert abs(dxy - 0.005) < 1e-9 and abs(dz - 0.006) < 1e-9 and abs(dyaw - 1.0) < 1e-6  # yaw is pi-symmetric
    assert ck.placed([0.1, 0.0, 0.044], _yaw_q(4.0), [0.1, 0.0, 0.044], _yaw_q(0.0), 0.005, 5.0, 0.01)
    assert not ck.placed([0.1, 0.0, 0.044], _yaw_q(6.0), [0.1, 0.0, 0.044], _yaw_q(0.0), 0.005, 5.0, 0.01)
    assert ck.static([0.0, 0.0, 0.0], [0.0005, 0.0, 0.0], 0.1, 0.01) and not ck.static([0.0, 0.0, 0.0], [0.002, 0.0, 0.0], 0.1, 0.01)
    assert ck.arm_clear([0.056, -0.291, 0.385], [0.174, -0.525]) and not ck.arm_clear([0.18, -0.5, 0.1], [0.174, -0.525])


def test_arm_clear_inserts_park():
    """Before the back arm's first transit, a left arm hovering over the destination pad is parked first; the graph
    pointer does not move."""
    v = "task1-3_tower_transfer_3"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    s = _state(v, 0.0, 0)
    spec = next(n for n in run.g["nodes"] if n["skill"] == "transit" and n["arm"] == "back" and n["target"] == "preplace")
    run._cmd = {}
    for a in ("left", "back"):
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
    pinch0 = run.pinch
    goal_xy = run.dest_goal(s, spec)[0][0, :2]
    over = torch.tensor([[float(goal_xy[0]) + 0.02, float(goal_xy[1]), 0.15]])
    run.pinch = lambda st, arm: (over, None) if arm == "left" else pinch0(st, arm)
    node = run._clearance_park(spec, s)
    assert isinstance(node, ParkNode) and node.arm == "left" and node.inserted
    run.pinch = pinch0  # home posture: clear
    assert run._clearance_park(spec, s) is None
