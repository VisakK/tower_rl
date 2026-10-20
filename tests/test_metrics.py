"""Step 3 acceptance: metric formulas on worked examples (build file) and an operation tracker on synthetic traces."""

import pytest

from tower_rl.eval.metrics import EpisodeOutcome, aggregate, wilson


def test_build_file_worked_example():
    # G=3, P=3, two grasps and one place succeeded, one disturbance in op 2, no task success
    o = EpisodeOutcome("task2-3", 3, 3, [True, True, False], [True, False, False], [0, 1, 0, 0, 0, 0, 0], False)
    m = aggregate([o])
    assert m["GSR"]["value"] == pytest.approx(0.6667, abs=1e-4)
    assert m["PSR"]["value"] == pytest.approx(0.3333, abs=1e-4)
    assert m["LHPR"]["value"] == pytest.approx(0.5)
    assert o.counted_steps == 7
    assert m["ODR"]["value"] == pytest.approx(1 / 7, abs=1e-4)
    assert m["TSR"]["value"] == 0.0


def test_hold_still_scores_zero():
    o = EpisodeOutcome("task1-2", 4, 4, [False] * 4, [False] * 4, [0] * 9, False)
    m = aggregate([o])
    assert m["TSR"]["value"] == m["GSR"]["value"] == m["PSR"]["value"] == m["LHPR"]["value"] == m["ODR"]["value"] == 0.0


def test_counts_summed_before_division_and_success_steps():
    a = EpisodeOutcome("v", 2, 2, [True, True], [True, True], [0, 0, 0, 0], True)  # 4 steps (success: no final step)
    b = EpisodeOutcome("v", 2, 2, [True, False], [False, False], [2, 0, 0, 0, 1], False)  # 5 steps
    m = aggregate([a, b])
    assert a.counted_steps == 4 and b.counted_steps == 5
    assert m["ODR"]["value"] == pytest.approx(3 / 9, abs=1e-4)
    assert m["GSR"]["value"] == pytest.approx(3 / 4)
    assert m["TSR"]["value"] == pytest.approx(0.5)


def test_wilson_interval():
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    assert wilson(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------- operation tracker on synthetic traces
import numpy as np  # noqa: E402

from tower_rl.eval.operations import OpCfg, OperationTracker, Transfer  # noqa: E402

Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def _scene():
    # blocks: 0 foundation, 1 target of transfer 0, 2 target of transfer 1, 3 bystander
    pos = np.array([[0.4, -0.3, 0.015], [0.4, -0.3, 0.073], [0.4, -0.3, 0.044], [0.6, -0.3, 0.044]])
    tr = [Transfer(1, "left", np.array([0.17, -0.52, 0.044]), 0.0, 1), Transfer(2, "back", np.array([0.17, -0.52, 0.073]), 0.0, 2)]
    return pos, tr


def _run(tracker, frames):
    t = 0.0
    for pos, pinch, closed in frames:
        tracker.update(t, pos, np.repeat(Q0[None], len(pos), 0), np.zeros(len(pos)), pinch, closed)
        t += 0.1
    return t


def _carry(pos, b, goal, arm, n=20):
    """frames: grasp + lift block b with arm, carry to goal, release and settle 1.5 s."""
    frames = []
    pinch = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    p = pos.copy()
    for i in range(n):  # lift and move
        a = (i + 1) / n
        p = p.copy()
        p[b] = (1 - a) * pos[b] + a * (goal + [0, 0, 0.05])
        if i < 3:
            p[b, 2] = pos[b, 2] + 0.03 * (i + 1) / 3
        pin = pinch.copy()
        pin[arm] = p[b]
        frames.append((p, pin, np.array([arm == 0, arm == 1])))
    p = p.copy()
    p[b] = goal
    for _ in range(15):  # released and static
        frames.append((p, pinch, np.array([False, False])))
    return frames, p


def test_tracker_full_success():
    pos, tr = _scene()
    trk = OperationTracker("t", tr, movable=[1, 2, 3])
    trk.reset(pos)
    f0, p = _carry(pos, 1, tr[0].goal_pos, 0)
    f1, p = _carry(p, 2, tr[1].goal_pos, 1)
    _run(trk, f0 + f1)
    o = trk.outcome(10.0)
    assert o.grasp_ok == [True, True] and o.place_ok == [True, True] and o.task_success
    assert o.disturbed_per_op == [0, 0, 0, 0] and o.counted_steps == 4


def test_tracker_disturbance_and_skip():
    pos, tr = _scene()
    trk = OperationTracker("t", tr, movable=[1, 2, 3])
    trk.reset(pos)
    bump = pos.copy()
    bump[3, 0] += 0.02  # bystander knocked 20 mm while G0 is pending
    frames = [(bump, np.array([[0, 0, 1.0], [0, 0, 1.0]]), np.array([False, False]))]
    f1, _ = _carry(bump, 2, tr[1].goal_pos, 1)  # transfer 0 skipped, transfer 1 executed
    _run(trk, frames + f1)
    o = trk.outcome(10.0)
    assert o.grasp_ok == [False, True] and o.place_ok == [False, True] and not o.task_success
    assert o.disturbed_per_op[0] == 1  # counted in G0's window
    assert sum(o.disturbed_per_op) == 1 and len(o.disturbed_per_op) == 5


def test_tracker_release_required_for_place():
    pos, tr = _scene()
    trk = OperationTracker("t", tr, movable=[1, 2, 3], cfg=OpCfg())
    trk.reset(pos)
    f0, p = _carry(pos, 1, tr[0].goal_pos, 0)
    held = [(fp, np.array([fp[1], [0, 0, 1.0]]), np.array([True, False])) for fp, _, _ in f0[-15:]]  # never released
    _run(trk, f0[:-15] + held)
    o = trk.outcome(5.0)
    assert o.grasp_ok[0] and not o.place_ok[0]
