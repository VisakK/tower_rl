"""Policy-server adapter (student plan Step 11.2), with a stub estimator over the fake world (no GPU, no network)."""

import numpy as np
import torch

from tower_rl.data.layouts import Layout
from tower_rl.perception.cameras import CAMERA_IDS
from tower_rl.perception.estimator import block_order
from tower_rl.student.adapter import StudentAdapter
from tower_rl.student.fakeworld import FakeWorld
from tower_rl.student.oracle import Oracle

torch.set_num_threads(1)
V = "task2-2_cross_tower_transfer_2"


class StubEstimator:
    """Returns the fake world's poses in the estimator's block order; records the focus of each call."""

    def __init__(self, world: FakeWorld, fail_at: int | None = None):
        self.w, self.fail_at, self.focus_log, self.n = world, fail_at, [], 0

    def reset(self, variant):
        lay = Layout.load(variant)
        self.block_ids = block_order(lay)
        self.fixed = torch.tensor([bool(lay.fixed[b]) for b in self.block_ids])

    def __call__(self, images, q, widths, focus=None):
        self.n += 1
        if self.fail_at == self.n:
            raise RuntimeError("estimator failure")
        self.focus_log.append(focus)
        idx = [self.w.ids.index(b) for b in self.block_ids]
        return self.w.pos[idx].copy(), self.w.quat[idx].copy(), np.ones(len(idx))


def _obs(w: FakeWorld, blank=False):
    img = np.zeros((480, 640, 3), dtype=np.uint8) if blank else np.full((480, 640, 3), 7, dtype=np.uint8)
    return {"images": {c: img for c in CAMERA_IDS}, "joint_positions": w.q.astype(np.float32),
            "gripper_widths_m": w.width.astype(np.float32), "task_variant": V, "instruction": "", "sim_time_s": w.t,
            "step_index": w.step}


def _adapter(w, **kw):
    est = StubEstimator(w, **kw)
    a = StudentAdapter(None, Oracle("graph", exact=False), lambda: est, log=lambda *x: None)
    return a, est


def test_blank_images_hold_then_the_runtime_acts():
    w = FakeWorld(V)
    a, est = _adapter(w)
    c = a.infer(_obs(w, blank=True))
    assert c.dtype == np.dtype("<f4") and c.shape == (1, 16)
    assert np.allclose(c[0, :7], w.q[0]) and np.allclose(c[0, 8:15], w.q[1])
    c = a.infer(_obs(w))
    assert c.shape[1] == 16 and 1 <= c.shape[0] <= 16 and np.isfinite(c).all()
    for row in c:  # run the chunk; the next call sees the runtime's option and focuses on its block
        w.apply(row)
    a.infer(_obs(w))
    g0 = a.policy.spec.goals[0].block
    assert est.focus_log[-1] == [est.block_ids.index(g0)]


def test_errors_hold_and_reset_clears():
    w = FakeWorld(V)
    a, est = _adapter(w, fail_at=2)
    a.infer(_obs(w))
    c = a.infer(_obs(w))  # the estimator raises: a hold chunk, logged
    assert a.errors == 1 and c.shape == (1, 16) and np.allclose(c[0, :7], w.q[0])
    a.reset()
    assert a.policy is None and a.variant is None and a.calls == 0


def test_bounds_and_variant_check():
    w = FakeWorld(V)
    a, _ = _adapter(w)
    lo = np.full(16, -0.5, dtype=np.float32)
    hi = np.full(16, 0.5, dtype=np.float32)
    a.bounds = (lo, hi)
    c = a.infer(_obs(w))
    assert (c >= -0.5).all() and (c <= 0.5).all()
    o = _obs(w)
    o["task_variant"] = "task3-2_vertical_stacking_2"
    c = a.infer(o)
    assert a.errors == 1 and c.shape == (1, 16)


class ImageStub(StubEstimator):
    """Records the fixed camera image each call receives."""

    def __call__(self, images, q, widths, focus=None):
        self.seen_images = images
        return super().__call__(images, q, widths, focus)


def test_orientation_choice_turns_the_fixed_cameras():
    """student_v1_1: a variant whose fixed cameras arrive rotated 180 deg (vs the reference) is turned back for the
    estimator, wrist cameras untouched; the recorder keeps the delivered images; the choice is logged."""
    from tower_rl.perception.orient import FIXED, small_gray

    rng = np.random.default_rng(0)
    base = (rng.random((60, 80, 3)) * 255).astype(np.uint8).repeat(8, 0).repeat(8, 1)  # [480, 640, 3], structured
    refs = {f"{V}/{c}": small_gray(base) for c in FIXED}
    w = FakeWorld(V)
    est = ImageStub(w)
    recs = []
    a = StudentAdapter(None, Oracle("graph", exact=False), lambda: est, log=lambda *x: None, orient_refs=refs,
                       recorder=lambda rec, imgs: recs.append((rec, imgs)))
    o = _obs(w)
    delivered = np.ascontiguousarray(base[::-1, ::-1])
    o["images"] = {c: (delivered if c in FIXED else base) for c in CAMERA_IDS}
    a.infer(o)
    assert a.orient_info["rotate"] and a.orient_info["how"] == "correlation"
    assert recs[0][0]["orient"]["rotate"] is True
    assert np.array_equal(est.seen_images["left_back_camera"], base)
    assert np.array_equal(est.seen_images["left_wrist_camera"], base)
    assert np.array_equal(recs[0][1]["left_back_camera"], delivered)  # logged as delivered
    a.reset()
    o["images"] = {c: base for c in CAMERA_IDS}  # delivered in the reference orientation: kept
    a.infer(o)
    assert not a.orient_info["rotate"] and np.array_equal(est.seen_images["left_back_camera"], base)


def test_blind_episode_is_marked_and_switches_the_runtime():
    """No movable block confidently seen on the first calls: three unconfident holds, then the runtime acts BLIND
    (graph flag blind_guard: the graph's plan)."""
    w = FakeWorld(V)

    class Unseen(StubEstimator):
        def __call__(self, images, q, widths, focus=None):
            p, q4, c = super().__call__(images, q, widths, focus)
            c = np.where(self.fixed.numpy(), 1.0, 0.3)
            return p, q4, c

    est = Unseen(w)
    a = StudentAdapter(None, Oracle("graph", exact=False), lambda: est, log=lambda *x: None)
    a.prepare(V)
    a.policy.spec.graph["blind_guard"] = True
    for _ in range(3):
        c = a.infer(_obs(w))
        assert c.shape == (1, 16)
    assert a.blind is None
    a.infer(_obs(w))
    assert a.blind is True and a.policy.runs[0].blind
    w2 = FakeWorld(V)
    a2, _ = _adapter(w2)
    a2.infer(_obs(w2))
    assert a2.blind is False


def test_per_variant_decider_is_chosen_at_init():
    """v1.3: ``VariantDeciders`` (bundle graphs with a "switcher" key) gives each variant's runtime its own decider."""
    from tower_rl.student.switcher import VariantDeciders

    special, default = Oracle("graph", exact=False), Oracle("graph", exact=False)
    a = StudentAdapter(None, VariantDeciders(default, {V: special}), lambda: StubEstimator(FakeWorld(V)),
                       log=lambda *x: None)
    a.prepare(V)
    assert a.policy.runs[0].decider is special
    v2 = "task1-2_tower_transfer_2"
    b = StudentAdapter(None, VariantDeciders(default, {V: special}), lambda: StubEstimator(FakeWorld(v2)),
                       log=lambda *x: None)
    b.prepare(v2)
    assert b.policy.runs[0].decider is default
