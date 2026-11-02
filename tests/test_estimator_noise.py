"""Estimator noise model (student plan Step 5): math helpers, classes, replay, stale / coasted unseen calls,
quaternion continuity, truth kept in privileged, determinism. Synthetic banks; no logs or Isaac needed."""

import dataclasses

import numpy as np
import pytest

from tower_rl.data.layouts import Layout
from tower_rl.student.fakeworld import FakeWorld
from tower_rl.student.noise import (CLASSES, NoiseModel, mat_to_quat, nearest_twin, quat_to_mat, rotvec_of,
                                    rotvec_to_mat)

V = "task2-2_cross_tower_transfer_2"


def _model(rows: dict, p_lost=None) -> NoiseModel:
    """A bank with one trace per class: ``rows[c]`` = list of (dp [3], drot [3], level)."""
    bank = {}
    for c in CLASSES:
        tr = rows.get(c, [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)] * 5)
        data = np.array([np.r_[dp, dr, lv] for dp, dr, lv in tr], dtype=np.float32)
        bank[f"{c}__data"], bank[f"{c}__lengths"] = data, np.array([len(data)])
    cfg = {"moving_speed_m_s": 0.01, "push_lost": {"p_lost": p_lost, "run_lengths": [2], "coast_decay": 0.8,
                                                  "coast_max_calls": 5}}
    return NoiseModel(cfg, bank)


def _fixed():
    lay = Layout.load(V)
    return np.array([bool(lay.fixed[b]) for b in lay.blocks])


def test_rotation_helpers():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(20, 3))
    w *= (rng.uniform(0.01, 3.0, size=(20, 1)) / np.linalg.norm(w, axis=-1, keepdims=True))
    R = rotvec_to_mat(w)
    assert np.allclose(rotvec_of(R), w, atol=1e-6)
    q = mat_to_quat(R)
    assert np.allclose(quat_to_mat(q), R, atol=1e-9)
    q0 = mat_to_quat(rotvec_to_mat(np.array([0.0, 0.0, 0.3])))
    flipped = mat_to_quat(quat_to_mat(q0) @ np.diag([-1.0, -1.0, 1.0]))  # the same box turned half about its z
    assert np.allclose(nearest_twin(flipped, q0), q0, atol=1e-9)


def test_classes_replay_and_truth():
    m = _model({"static": [((0.001, 0.0, 0.0), (0.0, 0.0, 0.0), 0)] * 3,
                "foundation": [((0.0, 0.002, 0.0), (0.0, 0.0, 0.0), 0)] * 3,
                "target:grasp_lift": [((0.0, 0.0, 0.003), (0.0, 0.0, 0.0), 1)] * 3})
    w = FakeWorld(V)
    ids = w.ids
    tgt = Layout.load(V).transfers[0].block
    en = m.make(ids, _fixed(), 7, focus=lambda: (tgt, "grasp_lift"))
    s = w.state()
    out = en(s)
    d = (out.block_pos - s.block_pos) * 1000
    f = _fixed()
    assert np.allclose(d[f], [0, 2, 0], atol=1e-6)  # foundations: their trace
    i = ids.index(tgt)
    assert np.allclose(d[i], [0, 0, 3], atol=1e-6) and out.block_conf[i] == 0.6  # the target: target:grasp_lift
    others = [j for j in range(len(ids)) if not f[j] and j != i]
    assert np.allclose(d[others], [1, 0, 0], atol=1e-6) and (out.block_conf[others] == 1.0).all()
    assert np.array_equal(out.privileged["true_block_pos"], s.block_pos)
    # foundation / static traces end: the last error is kept (no jump), as the estimator freezes / anchors
    for _ in range(5):
        out = en(w.state())
    assert np.allclose((out.block_pos - s.block_pos)[f] * 1000, [0, 2, 0], atol=1e-6)


def test_unseen_calls_are_stale_or_coasted():
    lost = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)] * 2 + [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2)] * 4
    m = _model({"moving": lost})
    w = FakeWorld(V)
    i = w.ids.index(Layout.load(V).transfers[0].block)
    en = m.make(w.ids, _fixed(), 3)
    speed = np.zeros(len(w.ids))
    speed[i] = 0.1
    outs = []
    for k in range(6):
        w.pos[i, 0] += 0.01  # moving 1 cm per call
        s = dataclasses.replace(w.state(), privileged={"block_speed": speed})
        outs.append(en(s))
    p = [o.block_pos[i, 0] for o in outs]
    assert outs[1].block_conf[i] == 1.0 and outs[2].block_conf[i] == 0.3
    assert p[2] == pytest.approx(p[1] + 0.01 * 0.8) and p[3] == pytest.approx(p[2] + 0.01 * 0.8 ** 2)  # coasting
    v = 0.01
    m2 = _model({"static": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)] + [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 3)] * 4})
    en2 = m2.make(w.ids, _fixed(), 3)
    a = en2(w.state())
    w.pos[i, 1] += v
    b = en2(w.state())
    assert b.block_conf[i] == 0.0 and np.array_equal(b.block_pos[i], a.block_pos[i])  # a ghost keeps its pose


def test_quaternion_continuity_and_determinism():
    m = _model({"static": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.002), 0)] * 50})
    runs = []
    for _ in range(2):
        w = FakeWorld(V)
        en = m.make(w.ids, _fixed(), 11)
        qs = []
        for k in range(4):
            s = w.state()
            if k % 2:  # the simulator reports the half-turned twin on odd calls
                q = s.block_quat
                s = dataclasses.replace(s, block_quat=np.stack([-q[:, 3], q[:, 2], -q[:, 1], q[:, 0]], -1))
            qs.append(en(s).block_quat.copy())
        runs.append(np.stack(qs))
        for a, b in zip(qs, qs[1:]):
            assert (np.abs((a * b).sum(-1)) > 0.999).all()
            assert ((a * b).sum(-1) > 0).all()  # continuous: no sign or twin flips between calls
    assert np.array_equal(runs[0], runs[1])


def test_push_target_losses():
    m = _model({"target:push": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)] * 50},
               p_lost=[[0.0] * 4, [0.0] * 4, [1.0] * 4, [1.0] * 4])
    w = FakeWorld(V)
    i = w.ids.index(Layout.load(V).transfers[0].block)
    en = m.make(w.ids, _fixed(), 5, focus=lambda: (w.ids[i], "push"))
    assert en(w.state()).block_conf[i] == 1.0  # level: never lost
    tilt = np.array([np.cos(np.radians(20)), np.sin(np.radians(20)), 0.0, 0.0])  # 40 deg about x
    w.quat[i] = tilt
    out = [en(w.state()).block_conf[i] for _ in range(3)]
    assert out[0] == 0.3  # tipped 40 deg: P(lost) = 1
