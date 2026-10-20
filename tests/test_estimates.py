"""Robust pose estimates (Step 8.1) and the executor decisions built on them (no simulator)."""

import math

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.graph.estimates import PoseWindow, StaticPose, align_to, median_pose, yaw_of
from tower_rl.graph.executor import Executor, LearnedPlaceNode, SkillPolicy
from tower_rl.graph.interface import ARMS, Status

from test_graph import _state


def _q(yaw_deg: float) -> np.ndarray:
    a = math.radians(yaw_deg)
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def test_median_pose_half_turn_and_sign():
    """Orientation samples given as different representatives of one block pose (sign, half turn) agree."""
    q = _q(30.0)
    flipped = geo.quat_mul(torch.tensor(q), torch.tensor(_q(180.0))).numpy()
    qs = np.stack([q, -q, flipped, -flipped, _q(31.0)])
    _, qm = median_pose(np.zeros((5, 3)), qs)
    d = (math.degrees(yaw_of(qm)) - 30.0 + 90.0) % 180.0 - 90.0
    assert abs(d) < 1.0
    assert np.allclose(np.abs(align_to(qs, q) @ q), 1.0, atol=1e-3)


def test_static_pose_jitter_outlier_dropout_motion():
    rng = np.random.default_rng(0)
    est = StaticPose(window=10, gate_m=0.004, confirm=2)
    p0 = np.array([0.4, -0.3, 0.1])
    for _ in range(10):
        est.add(p0 + rng.normal(0, 0.001, 3), _q(0.0))
    assert np.linalg.norm(est.pose()[0] - p0) < 0.001  # jitter averages out
    est.add(p0 + np.array([0.01, 0.0, 0.0]), _q(0.0))  # one-call outlier: held back
    assert np.linalg.norm(est.pose()[0] - p0) < 0.001
    est.add(p0 + np.array([0.0, 0.05, 0.0]), _q(0.0), conf=0.0)  # dropout's stale pose: ignored
    assert np.linalg.norm(est.pose()[0] - p0) < 0.001
    est.add(p0, _q(0.0))  # a consistent sample clears the held-back outlier
    p1 = p0 + np.array([0.0, 0.03, 0.0])  # the block moved: the estimate follows after two calls
    est.add(p1, _q(0.0))
    assert np.linalg.norm(est.pose()[0] - p0) < 0.001
    est.add(p1, _q(0.0))
    assert np.allclose(est.pose()[0], p1)
    exact = StaticPose()  # simulator truth: the latest pose
    for k in range(5):
        exact.add(p0 + k * np.array([0.0, 0.0, 0.0001]), _q(0.0))
    assert np.linalg.norm(exact.pose()[0] - (p0 + [0.0, 0.0, 0.0004])) < 2.1e-4


def test_pose_window_drift_and_median():
    w = PoseWindow()
    assert w.drift(5) is None
    for k in range(10):  # sliding 1 mm per sample
        w.add(np.array([0.001 * k, 0.0, 0.0]), _q(0.0))
    assert abs(w.drift(5) - 0.005) < 1e-9
    w2 = PoseWindow()
    rng = np.random.default_rng(1)
    for k in range(10):  # at rest with 1 mm jitter and one 1 cm outlier
        w2.add(rng.normal(0, 0.001, 3) + (np.array([0.01, 0, 0]) if k == 7 else 0), _q(0.0))
    assert w2.drift(5) < 0.0025 and np.linalg.norm(w2.median(10)[0]) < 0.0015


def test_dest_goal_ignores_outliers_and_cancels_orientation_bias():
    """Goals come from the foundations' rest estimates (a one-call 1 cm outlier does not move them); the goal
    orientation follows a yaw error common to the perceived blocks, the in-pad positions do not."""
    v = "task1-3_tower_transfer_3"  # goals 50 mm off the pad centre
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    s = _state(v, 0.0, 0)
    node = next(n for n in run.g["nodes"] if n["skill"] == "place_release" and abs(n["goal_in_pad"]["pos"][0]) > 0.04)
    g_true = [x[0].numpy() for x in run.dest_goal(s, node, raw=True)]
    for _ in range(5):
        run.observe(s)
    f0 = s.block_ids.index(run.g["foundation"]["destination"][0])
    s.block_pos[f0] += np.array([0.01, 0.0, 0.0])
    run.observe(s)  # one outlier call
    g = [x[0].numpy() for x in run.dest_goal(s, node)]
    assert np.linalg.norm(g[0] - g_true[0]) < 1e-6 and np.allclose(np.abs(g[1] @ g_true[1]), 1.0, atol=1e-6)
    # a 3 deg yaw error on every perceived block: goal orientation +3 deg, goal position unchanged
    s2 = _state(v, 0.0, 0)
    rot = torch.tensor(_q(3.0), dtype=torch.float32)
    s2.block_quat = geo.quat_mul(rot.expand(len(s2.block_ids), 4), torch.tensor(s2.block_quat, dtype=torch.float32)).numpy()
    g2 = [x[0].numpy() for x in run.dest_goal(s2, node, raw=True)]
    assert np.linalg.norm(g2[0] - g_true[0]) < 1e-5
    dy = (math.degrees(yaw_of(g2[1]) - yaw_of(g_true[1])) + 90.0) % 180.0 - 90.0
    assert abs(dy - 3.0) < 0.05


class _ReleasePolicy:
    """Stub place actor: no motion, gripper open (releases on its first call)."""

    def __call__(self, obs):
        a = torch.zeros(obs.shape[0], 7)
        a[:, 6] = -1.0
        return a


def _place_verdict(offset_xy_m: np.ndarray, jitter_m: float, seed: int) -> tuple[Status, dict, int]:
    v = "task3-3_vertical_stacking_3"
    ex = Executor(v, skills={("place_release", "back"): SkillPolicy(_ReleasePolicy())})
    ex.reset(1, v, [0])
    run = ex.runs[0]
    spec = next(n for n in run.g["nodes"] if n["skill"] == "place_release")
    s = _state(v, 0.0, 0)
    run._cmd = {}
    for a in ARMS:  # as EnvRun.act on its first call
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
        run.grip[a].reset(width=0.1)
    gp, gq = (x[0].numpy() for x in run.dest_goal(s, spec, raw=True))
    node = LearnedPlaceNode(run, spec, s, ex.skills[("place_release", "back")])
    i = s.block_ids.index(spec["block"])
    rng = np.random.default_rng(seed)
    for k in range(60):
        s.block_pos[i] = gp + np.r_[offset_xy_m, 0.0] + rng.normal(0, jitter_m, 3) * np.array([1, 1, 0.5])
        s.block_quat[i] = gq
        st, _ = node.step(s)
        if st is not None:
            return st, node.info, k
    raise AssertionError("no verdict")


def test_place_verdict_uses_a_window_of_perceived_poses():
    """A block at rest 3 mm off its goal, perceived with 2 mm jitter (single poses beyond 5 mm a fifth of the time):
    judged placed. One 8 mm off: judged failed. Without a retreat the verdict comes 3 x settle_steps after the
    release."""
    for seed in range(5):
        st, info, k = _place_verdict(np.array([0.003, 0.0]), 0.002, seed)
        assert st == Status.OK and info["place_xy_mm"] < 5.0, (seed, info)
    st, info, k = _place_verdict(np.array([0.008, 0.0]), 0.001, 0)
    assert st == Status.FAILED and 6.0 < info["place_xy_mm"] < 10.0
    assert k == 30  # released on the first call; no retreat -> 3 x 10 rows later


def test_refresh_attached_confirms_a_slip_but_not_an_outlier():
    v = "task2-2_cross_tower_transfer_2"
    ex = Executor(v)
    ex.reset(1, v, [0])
    run = ex.runs[0]
    s = _state(v, 0.0, 0)
    bid = run.g["movable"][0]
    i = s.block_ids.index(bid)
    run.attach(s, "left", bid)
    p0 = run.block_est(s, bid)[0][0].numpy().copy()
    s.block_pos[i] += np.array([0.012, 0.0, 0.0])  # 12 mm: one call is an outlier ...
    run.refresh_attached(s, "left")
    run.refresh_attached(s, "left")
    assert np.allclose(run.block_est(s, bid)[0][0].numpy(), p0, atol=1e-6)
    run.refresh_attached(s, "left")  # ... three consistent calls are a slip
    assert np.allclose(run.block_est(s, bid)[0][0].numpy(), s.block_pos[i], atol=1e-5)


class _HoldPolicy:
    """Stub place actor: no motion, gripper closed; records the goal and block positions it observes (base frame)."""

    def __init__(self):
        self.goals, self.blocks = [], []

    def __call__(self, obs):
        self.goals.append(obs[0, -14:-11].clone())  # goal_pb: before prev_actions (7), phase (3), progress (1)
        self.blocks.append(obs[0, 33:36].clone())   # block position in the pinch frame (after pb, rot6d, vel, lead, gripper, goal frame)
        a = torch.zeros(obs.shape[0], 7)
        a[:, 6] = 1.0
        return a


def test_jaw_to_goal_moves_the_jaw_offset_to_the_goal():
    """With graph perception.jaw_to_goal = 1 the place policy sees the held block centred between the fingers and the
    goal moved by minus the block's offset along the jaw (same block-to-goal error); the offset along the block and
    the node's goal for the place check are unchanged."""
    v = "task3-3_vertical_stacking_3"
    seen = {}
    for w in (0.0, 1.0):
        pol = _HoldPolicy()
        ex = Executor(v, skills={("place_release", "back"): SkillPolicy(pol)})
        ex.graph["perception"] = {"jaw_to_goal": w}
        ex.reset(1, v, [0])
        run = ex.runs[0]
        spec = next(n for n in run.g["nodes"] if n["skill"] == "place_release")
        s = _state(v, 0.0, 0)
        run._cmd = {}
        for a in ARMS:
            run.ctrl[a].reset(run.q_meas(s, a))
            run.command(a)
            run.grip[a].reset(width=0.1)
        pp, pq = run.pinch(s, "back")
        off = torch.tensor([[0.004, 0.003, -0.002]])  # perceived in-hand offset (pinch frame; y = jaw)
        i = s.block_ids.index(spec["block"])
        s.block_pos[i] = (pp + geo.quat_apply(pq, off))[0].numpy()
        s.block_quat[i] = pq[0].numpy()
        run.attach(s, "back", spec["block"], rest=False)
        node = LearnedPlaceNode(run, spec, s, ex.skills[("place_release", "back")])
        node.step(s)
        seen[w] = (pol.goals[0], pol.blocks[0], node.goal[0].clone())
    base = run.ctrl["back"].base
    shift = base.vec_to_world((seen[1.0][0] - seen[0.0][0])[None])[0]
    jaw_off = torch.tensor([[0.0, 0.003, 0.0]])
    assert torch.allclose(shift, -geo.quat_apply(pq, jaw_off)[0], atol=2e-5), shift
    assert torch.allclose(seen[0.0][1], off[0], atol=2e-5)
    assert torch.allclose(seen[1.0][1], off[0] - jaw_off[0], atol=2e-5)
    assert torch.allclose(seen[1.0][2], seen[0.0][2])  # the node's (perceived) goal for the place check is unchanged


def test_jaw_to_goal_per_arm():
    """``jaw_to_goal`` may give one weight per arm (per place policy): an arm without an entry gets 0."""
    v = "task3-3_vertical_stacking_3"
    goals = {}
    for setting in ({"back": 1.0}, {"left": 1.0}):
        pol = _HoldPolicy()
        ex = Executor(v, skills={("place_release", "back"): SkillPolicy(pol)})
        ex.graph["perception"] = {"jaw_to_goal": setting}
        ex.reset(1, v, [0])
        run = ex.runs[0]
        spec = next(n for n in run.g["nodes"] if n["skill"] == "place_release")
        s = _state(v, 0.0, 0)
        run._cmd = {}
        for a in ARMS:
            run.ctrl[a].reset(run.q_meas(s, a))
            run.command(a)
            run.grip[a].reset(width=0.1)
        pp, pq = run.pinch(s, "back")
        i = s.block_ids.index(spec["block"])
        s.block_pos[i] = (pp + geo.quat_apply(pq, torch.tensor([[0.0, 0.003, 0.0]])))[0].numpy()
        s.block_quat[i] = pq[0].numpy()
        run.attach(s, "back", spec["block"], rest=False)
        LearnedPlaceNode(run, spec, s, ex.skills[("place_release", "back")]).step(s)
        goals[next(iter(setting))] = pol.goals[0]
    assert float((goals["back"] - goals["left"]).norm()) > 0.0025  # back arm compensated, left setting ignored
