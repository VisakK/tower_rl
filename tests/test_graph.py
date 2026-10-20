"""Graphs, transit plans and executor determinism (no simulator)."""

import math

import numpy as np
import pytest
import torch

from tower_rl import geometry as geo
from tower_rl.data.layouts import VARIANTS, Layout
from tower_rl.graph.executor import Executor
from tower_rl.graph.graph import build_graph, validate
from tower_rl.graph.interface import State
from tower_rl.skills.transit import TransitCfg, plan_waypoints
from tower_rl.config import load_yaml


@pytest.mark.parametrize("variant", VARIANTS)
def test_graph_structure(variant):
    g = build_graph(variant)
    validate(g)
    lay = Layout.load(variant)
    places = [n for n in g["nodes"] if n["skill"] == "place_release"]
    assert [n["block"] for n in places] == [t.block for t in lay.transfers]
    # an arm switch is always preceded by parking the previous arm; the graph ends parked
    prev = None
    for i, n in enumerate(g["nodes"]):
        if n["skill"] != "park" and prev is not None and n["arm"] != prev:
            assert g["nodes"][i - 1]["skill"] == "park" and g["nodes"][i - 1]["arm"] == prev
        if n["skill"] != "park":
            prev = n["arm"]
    assert g["nodes"][-1]["skill"] == "park"


def test_transit_plan_limits():
    cfg = TransitCfg()
    p0 = torch.tensor([0.42, -0.28, 0.17])
    p1 = torch.tensor([0.17, -0.52, 0.16])
    q0 = geo.quat_from_yaw(torch.tensor(0.0))
    q1 = geo.quat_from_yaw(torch.tensor(-math.pi / 2))
    for held in (False, True):
        ps, qs = plan_waypoints(p0, q0, p1, q1, held, cfg)
        assert torch.allclose(ps[-1], p1, atol=1e-6) and float(geo.quat_angle(qs[-1], q1)) < 1e-4
        v = (ps[1:] - ps[:-1]).norm(dim=-1) / cfg.dt
        vmax = cfg.v_max_held if held else cfg.v_max
        assert float(v.max()) <= vmax * 1.05
        a = (v[1:] - v[:-1]).abs() / cfg.dt
        assert float(a.max()) <= cfg.a_max * 1.6  # finite-difference estimate of the quintic profile
        assert float(ps[:, 2].max()) <= max(float(p0[2]), float(p1[2])) + cfg.rise_m + 1e-6
    # holding a block is slower
    assert len(plan_waypoints(p0, q0, p1, q1, True, cfg)[0]) > len(plan_waypoints(p0, q0, p1, q1, False, cfg)[0])


def _state(variant: str, t: float, step: int) -> State:
    lay = Layout.load(variant)
    ids = list(lay.blocks)
    home = load_yaml("scene")["home_qpos"]
    pos = np.stack([lay.blocks[b][:3] for b in ids])
    quat = np.stack([lay.blocks[b][3:7] for b in ids])
    return State(t=t, step=step, variant=variant, instruction="", q=np.array([home["left"][:7], home["back"][:7]]),
                 width=np.array([0.1, 0.1]), block_ids=ids, block_pos=pos, block_quat=quat, block_conf=np.ones(len(ids)))


def test_executor_deterministic():
    v = "task2-2_cross_tower_transfer_2"
    outs = []
    for _ in range(2):
        ex = Executor(v)
        ex.reset(1, v, [123])
        rows = []
        t = 0.0
        for k in range(5):
            chunk = ex.act([_state(v, t, k)], [0])[0]
            rows.append(chunk)
            t += 0.1 * len(chunk)
        outs.append(np.concatenate(rows))
    assert outs[0].shape == outs[1].shape and outs[0].shape[1] == 16
    assert np.array_equal(outs[0], outs[1])
    assert 1 <= len(outs[0]) <= 5 * 16 and np.isfinite(outs[0]).all()


class _FailedNode:
    def __init__(self, info):
        self.info = info


@pytest.mark.parametrize("variant", VARIANTS)
def test_failure_edges(variant):
    """skip -> the next node of a later transfer (the park that clears this arm when the arm switches, else the next
    transfer's first node); retry -> this transfer's first node; a failed park -> the next node."""
    from tower_rl.graph.interface import Status

    ex = Executor(variant)
    ex.reset(1, variant, [0])
    run = ex.runs[0]
    nodes = run.g["nodes"]
    s = _state(variant, 0.0, 0)
    for i, n in enumerate(nodes):
        if n["skill"] != "place_release":
            continue
        n["on_fail"], n["skip_if_close_m"] = "skip_if_close", 0.015  # the policy under test (graphs default to retry)
        k = n["transfer"]
        nxt = nodes[i + 1]
        # near miss -> skip: never jump past a park that follows this transfer
        run.i, run.tries, run.node = i, {}, _FailedNode({"place_xy_mm": 8.0})
        run._advance(Status.FAILED, s)
        assert run.i == i + 1 and nodes[run.i]["transfer"] > k
        if nxt["skill"] == "park":
            assert nxt["arm"] == n["arm"]
        # far miss -> retry from this transfer's first node
        run.i, run.tries, run.node = i, {}, _FailedNode({"place_xy_mm": 40.0})
        run._advance(Status.FAILED, s)
        assert nodes[run.i]["transfer"] == k and nodes[run.i].get("first_of_transfer")
    for i, n in enumerate(nodes[:-1]):
        if n["skill"] == "park":
            run.i, run.tries, run.node = i, {}, _FailedNode({})
            run._advance(Status.TIMEOUT, s)
            assert run.i == i + 1


def test_lost_block_skips():
    """A grasp that fails because its block fell off the table skips the transfer instead of retrying it."""
    from tower_rl.graph.interface import Status

    v = "task1-3_tower_transfer_3"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    nodes = run.g["nodes"]
    i = next(j for j, n in enumerate(nodes) if n["skill"] == "grasp_lift" and n["transfer"] == 1)
    s = _state(v, 0.0, 0)
    run.i, run.tries, run.node = i, {}, _FailedNode({})
    run._advance(Status.TIMEOUT, s)
    assert nodes[run.i]["transfer"] == 1 and nodes[run.i].get("first_of_transfer")  # on the table: retry
    s.block_pos[s.block_ids.index(nodes[i]["block"]), 2] = -0.5
    run.i, run.tries, run.node = i, {}, _FailedNode({})
    run._advance(Status.TIMEOUT, s)
    assert run.i == i + 3 and nodes[run.i]["transfer"] == 2  # past t1_to_preplace / t1_place: the park of transfer 2


def test_retry_keeps_held_block():
    """A transport failure with the block still in the closed gripper retries from the pre-place transit; any other
    failure restarts the transfer with the gripper open."""
    from tower_rl.graph.interface import Status

    v = "task2-3_cross_tower_transfer_3"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    nodes = run.g["nodes"]
    arm = nodes[0]["arm"]
    ai = ["left", "back"].index(arm)
    place = next(j for j, n in enumerate(nodes) if n["skill"] == "place_release" and n["transfer"] == 0)
    pre = next(j for j, n in enumerate(nodes) if n["transfer"] == 0 and n.get("target") == "preplace")
    s = _state(v, 0.0, 0)
    s.width[ai] = 0.0497
    run.grip[arm].reset(closed=True, width=0.04)
    run.i, run.tries, run.node = place, {}, _FailedNode({})
    run._advance(Status.TIMEOUT, s)  # place timed out before releasing: still holding
    assert run.i == pre and bool(run.grip[arm].closed)
    s.width[ai] = 0.1
    run.grip[arm].reset(closed=False, width=0.1)
    run.i, run.tries, run.node = place, {}, _FailedNode({"place_xy_mm": 40.0})
    run._advance(Status.FAILED, s)  # released far off the goal: restart the transfer
    assert nodes[run.i].get("first_of_transfer") and nodes[run.i]["transfer"] == 0 and not bool(run.grip[arm].closed)


def test_reachable_pose_and_unreachable_pregrasp():
    """A reachable via-point is kept; an unreachable one is replaced by the FK of the IK solution (reachable by
    construction); a pre-grasp far out of reach fails the transit at once."""
    from tower_rl import geometry as geo
    from tower_rl.graph.executor import TransitNode, reachable_pose
    from tower_rl.graph.interface import Status

    v = "task2-2_cross_tower_transfer_2"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    s = _state(v, 0.0, 0)
    run._cmd = {}
    for a in ("left", "back"):
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
    ctrl = run.ctrl["left"]
    p0, q0 = ctrl.pos.clone(), ctrl.quat.clone()  # the current pinch pose: reachable
    (p1, q1), adj = reachable_pose(ctrl, p0, q0, ctrl.q_ref)
    assert not adj and torch.equal(p1, p0)
    far = p0 + torch.tensor([[1.5, 0.0, 0.0]])  # 1.5 m away: out of reach
    (p2, q2), adj = reachable_pose(ctrl, far, q0, ctrl.q_ref)
    assert adj["goal_adjust_mm"] > 500
    (p3, _), adj3 = reachable_pose(ctrl, p2, q2, ctrl.q_ref)
    assert not adj3 or adj3["goal_adjust_mm"] < 1.0  # the adjusted pose itself is reachable
    # a pre-grasp transit to a block moved far out of the left arm's reach fails immediately
    spec = next(n for n in run.g["nodes"] if n["skill"] == "transit" and n["arm"] == "left" and n["target"] == "pregrasp")
    s.block_pos[s.block_ids.index(spec["block"])] = np.array([1.2, -0.3, 0.05])
    node = TransitNode(run, spec, s)
    status, rows = node.step(s)
    assert status == Status.FAILED and rows == [] and node.info["why"] == "pregrasp unreachable"


def test_attached_block_follows_the_gripper():
    """An attached block's estimate is the kinematic prediction (it ignores a stale perceived pose); detaching returns
    to perception."""
    v = "task2-2_cross_tower_transfer_2"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    s = _state(v, 0.0, 0)
    bid = run.g["movable"][0]
    p_true = s.block(bid)[0].copy()
    run.attach(s, "left", bid)
    s.block_pos[s.block_ids.index(bid)] += np.array([0.0, 0.0, 0.05])  # a stale / wrong perceived pose
    p_est = run.block_est(s, bid)[0][0].numpy()
    assert np.allclose(p_est, p_true, atol=1e-5)  # the arm has not moved: the prediction stays put
    run.detach("left")
    assert np.allclose(run.block_est(s, bid)[0][0].numpy(), p_true + np.array([0.0, 0.0, 0.05]), atol=1e-6)


def test_attached_offset_tracks_fresh_perception_only():
    """refresh_attached follows a confident measurement (a slipping block), ignores a stale one (confidence 0) and a
    one-call outlier (beyond the 5 mm gate)."""
    v = "task2-2_cross_tower_transfer_2"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    s = _state(v, 0.0, 0)
    bid = run.g["movable"][0]
    i = s.block_ids.index(bid)
    run.attach(s, "left", bid)
    p0 = run.block_est(s, bid)[0][0].numpy().copy()
    s.block_pos[i] += np.array([0.0, 0.0, -0.004])  # the block slid 4 mm down in the fingers
    s.block_conf[i] = 0.0
    run.refresh_attached(s, "left")
    assert np.allclose(run.block_est(s, bid)[0][0].numpy(), p0, atol=1e-6)  # stale: ignored
    s.block_conf[i] = 1.0
    for _ in range(30):
        run.refresh_attached(s, "left")
    assert np.allclose(run.block_est(s, bid)[0][0].numpy(), s.block_pos[i], atol=2e-4)  # tracked
    s.block_pos[i] += np.array([0.03, 0.0, 0.0])  # a 3 cm outlier
    before = run.block_est(s, bid)[0][0].numpy().copy()
    run.refresh_attached(s, "left")
    assert np.allclose(run.block_est(s, bid)[0][0].numpy(), before, atol=1e-6)


def test_arm_switch_replace_schedule():
    """A near-miss place whose own arm cannot re-grasp queues park / other arm's transfer nodes / park, once; when the
    queue has run the graph continues after the transfer; a far miss or a reachable re-grasp takes the normal path."""
    from tower_rl.graph.interface import Status

    v = "task2-3_cross_tower_transfer_3"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    nodes = run.g["nodes"]
    s = _state(v, 0.0, 0)
    i = next(j for j, n in enumerate(nodes) if n["skill"] == "place_release" and n["arm"] == "left")
    k = nodes[i]["transfer"]
    reach = {"left": False, "back": True}
    run._pregrasp_reachable = lambda arm, bid, st: reach[arm]
    run.i, run.tries, run.node = i, {}, _FailedNode({"place_xy_mm": 5.3})
    run._advance(Status.FAILED, s)
    q = run.alt_queue
    assert [n["skill"] for n in q] == ["park", "transit", "grasp_lift", "transit", "place_release", "park"]
    assert q[0]["arm"] == "left" and all(n["arm"] == "back" for n in q[1:]) and run.i == i
    for _ in range(len(q)):  # every inserted node succeeds
        class _N(_FailedNode):
            alt = True
        run.node = _N({})
        run._advance(Status.OK, s)
    assert not run.alt_queue and nodes[run.i]["transfer"] > k
    # once per transfer; a far miss or a reachable re-grasp takes the normal retry
    run.i, run.node = i, _FailedNode({"place_xy_mm": 5.3})
    run._advance(Status.FAILED, s)
    assert not run.alt_queue and nodes[run.i]["transfer"] == k and nodes[run.i].get("first_of_transfer")
    run.alt_done.clear()
    run.i, run.tries, run.node = i, {}, _FailedNode({"place_xy_mm": 40.0})
    run._advance(Status.FAILED, s)
    assert not run.alt_queue
    reach["left"] = True
    run.i, run.tries, run.node = i, {}, _FailedNode({"place_xy_mm": 5.3})
    run._advance(Status.FAILED, s)
    assert not run.alt_queue
