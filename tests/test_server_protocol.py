"""The tower-infer-v1 server (student plan Step 11): the kit's protocol with per-session model state, over real
WebSockets on loopback (no GPU, no network): metadata, reset / infer echoes and seed counters, a second reset on the
same connection, independent sessions, the reconnect contract (a duplicate request gets the committed response without
a second model call), rejections, and the student adapter end to end on the fake world."""

import asyncio
import threading
import time

import numpy as np
import pytest
import torch
from websockets.asyncio.server import serve
from websockets.sync.client import connect

from tower_rl.data.layouts import Layout
from tower_rl.perception.cameras import CAMERA_IDS
from tower_rl.perception.estimator import block_order
from tower_rl.server.handler import handler_factory
from tower_rl.server.student import EventLog, SessionModel
from tower_rl.server.tower_protocol import action_spec, observation_spec, pack, sample_seed, unpack
from tower_rl.student.adapter import StudentAdapter, clip32
from tower_rl.student.fakeworld import FakeWorld
from tower_rl.student.oracle import Oracle

torch.set_num_threads(1)
V = "task2-2_cross_tower_transfer_2"
TRACK = {"version": "test-track", "policy_track_version": "test-track", "observation_schema": "tower-challenge-observation-v1",
         "action_schema": "tower-challenge-joint16-v1", "cameras": {c: [480, 640] for c in CAMERA_IDS},
         "action_dt_s": 0.1, "max_chunk_actions": 16, "action_lower": [-6.2832] * 7 + [0] + [-6.2832] * 7 + [0],
         "action_upper": [6.2832] * 7 + [0.10000000149011612] + [6.2832] * 7 + [0.10000000149011612],
         "inference_timeout_s": 180, "episode_wall_timeout_s": 7200}
HEADERS = {"X-Tower-Task-Variant": V, "X-Tower-Track-Version": "test-track", "X-Tower-Evaluation-Version": "test-eval"}
PID = "tower-student-test"
IDENT = "sha256:" + "ab" * 32


class FakeModel:
    """Echoes the measured joints in chunks of 1-3 rows; counts starts and calls; optionally slow."""

    def __init__(self, session_id, variant, delay=0.0):
        self.session_id, self.variant, self.delay = session_id, variant, delay
        self.starts, self.calls = [], 0

    def start(self, trial_id, seed):
        self.starts.append((trial_id, seed))
        self.calls = 0

    def infer(self, observation, index):
        time.sleep(self.delay)
        self.calls += 1
        q = np.asarray(observation["joint_positions"], dtype="<f4")
        w = np.asarray(observation["gripper_widths_m"], dtype="<f4")
        row = np.concatenate([q[0], w[:1], q[1], w[1:]])
        return np.repeat(row[None], 1 + self.calls % 3, axis=0).astype("<f4")


class ServerThread:
    def __init__(self, make_model, **kw):
        self.models = []

        def mk(sid, variant):
            m = make_model(sid, variant)
            self.models.append(m)
            return m

        self.handle = handler_factory(TRACK, PID, IDENT, mk, **kw)
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self.ready.wait(10)

    def _run(self):
        asyncio.set_event_loop(self.loop)

        async def main():
            async with serve(self.handle, "127.0.0.1", 0, max_size=64 * 1024 * 1024, compression=None,
                             ping_timeout=None) as srv:
                self.port = srv.sockets[0].getsockname()[1]
                self.stop = self.loop.create_future()
                self.ready.set()
                await self.stop

        self.loop.run_until_complete(main())

    def connect(self, headers=HEADERS):
        ws = connect(f"ws://127.0.0.1:{self.port}", additional_headers=headers, compression=None,
                     max_size=64 * 1024 * 1024)
        return ws, unpack(ws.recv(timeout=10))

    def close(self):
        self.loop.call_soon_threadsafe(self.stop.set_result, None)
        self.thread.join(10)


@pytest.fixture
def server():
    s = ServerThread(lambda sid, v: FakeModel(sid, v))
    yield s
    s.close()


def ident(sid, trial, rid):
    return {"protocol": "tower-infer-v1", "session_id": sid, "trial_id": trial, "request_id": rid, "policy_id": PID}


def obs(q=None, width=None, blank=True, variant=V):
    img = np.zeros((480, 640, 3), dtype=np.uint8) if blank else np.full((480, 640, 3), 7, dtype=np.uint8)
    q = np.zeros((2, 7), dtype="<f4") if q is None else np.asarray(q, dtype="<f4")
    width = np.full(2, 0.1, dtype="<f4") if width is None else np.asarray(width, dtype="<f4")
    return {"schema": "tower-challenge-observation-v1", "step_index": 0, "sim_time_s": 0.0, "task_variant": variant,
            "instruction": "Use both arms cooperatively to transfer the cross-shaped Jenga tower to the designated target position.",
            "joint_positions": q, "gripper_widths_m": width, "images": {c: img for c in CAMERA_IDS}}


def reset(ws, sid, trial, rid, seed=7):
    ws.send(pack({**ident(sid, trial, rid), "type": "reset", "seed": seed}))
    return unpack(ws.recv(timeout=10))


def infer(ws, sid, trial, rid, o=None):
    ws.send(pack({**ident(sid, trial, rid), "type": "infer", "observation": o or obs()}))
    return unpack(ws.recv(timeout=30))


def test_metadata_reset_infer_and_seeds(server):
    ws, meta = server.connect()
    assert meta["type"] == "metadata" and meta["protocol"] == "tower-infer-v1" and meta["ready"] is True
    assert meta["policy_id"] == PID and meta["artifact_identity"] == IDENT and meta["seed_mode"] == "deterministic"
    assert meta["observation_spec"] == observation_spec(TRACK) and meta["action_spec"] == action_spec(TRACK)
    assert meta["reconnect"]["retention_s"] >= 60 + TRACK["inference_timeout_s"]
    ack = reset(ws, "s1", "t1", "r1", seed=123)
    assert ack["type"] == "reset_ack" and ack["base_seed"] == 123 and ack["next_sample_index"] == 0
    assert {k: ack[k] for k in ("session_id", "trial_id", "request_id", "policy_id")} == \
        {"session_id": "s1", "trial_id": "t1", "request_id": "r1", "policy_id": PID}
    q = np.arange(14, dtype="<f4").reshape(2, 7) / 10
    for i in range(3):
        res = infer(ws, "s1", "t1", f"i{i}", obs(q=q))
        assert res["type"] == "infer_result" and res["request_id"] == f"i{i}" and res["trial_id"] == "t1"
        assert res["sample_index"] == i and res["sample_seed"] == sample_seed(123, i) and res["base_seed"] == 123
        assert res["dt_s"] == 0.1 and res["plugin"] == meta["plugin"] and res["plugin_version"] == meta["plugin_version"]
        a = res["actions"]
        assert a.dtype == np.dtype("<f4") and a.ndim == 2 and a.shape[1] == 16 and 1 <= a.shape[0] <= 16
        assert np.allclose(a[0, :7], q[0]) and np.allclose(a[0, 8:15], q[1])
    assert server.models[0].starts == [("t1", 123)] and server.models[0].calls == 3
    ws.close()


def test_second_reset_on_the_same_connection_clears_the_episode(server):
    ws, _ = server.connect()
    reset(ws, "s1", "t1", "r1")
    infer(ws, "s1", "t1", "i1")
    ack = reset(ws, "s1", "t2", "r2", seed=7)  # the same seed again: still a fresh episode
    assert ack["type"] == "reset_ack"
    res = infer(ws, "s1", "t2", "i2")
    assert res["sample_index"] == 0 and res["trial_id"] == "t2"
    m = server.models[0]
    assert m.starts == [("t1", 7), ("t2", 7)] and m.calls == 1
    res = infer(ws, "s1", "t1", "i3")  # the old trial is over
    assert res["type"] == "error" and res["error"]["code"] == "reset_required"
    ws.close()


def test_sessions_are_independent(server):
    a, _ = server.connect()
    b, _ = server.connect()
    reset(a, "sa", "ta", "ra", seed=1)
    reset(b, "sb", "tb", "rb", seed=2)
    for i in range(3):
        ra = infer(a, "sa", "ta", f"a{i}")
        if i < 2:
            rb = infer(b, "sb", "tb", f"b{i}")
            assert rb["sample_seed"] == sample_seed(2, i)
        assert ra["sample_index"] == i and ra["sample_seed"] == sample_seed(1, i)
    ma, mb = server.models
    assert (ma.session_id, ma.calls, mb.session_id, mb.calls) == ("sa", 3, "sb", 2)
    a.close()
    b.close()


def test_reconnect_returns_the_committed_response_without_a_second_call():
    s = ServerThread(lambda sid, v: FakeModel(sid, v, delay=0.5))
    try:
        ws, _ = s.connect()
        reset(ws, "s1", "t1", "r1")
        raw = pack({**ident("s1", "t1", "i1"), "type": "infer", "observation": obs()})
        ws.send(raw)
        ws.close()  # the response is lost with the socket
        ws2, meta = s.connect()
        assert meta["type"] == "metadata"
        ws2.send(raw)  # the worker resends the exact bytes
        res = unpack(ws2.recv(timeout=10))
        assert res["type"] == "infer_result" and res["sample_index"] == 0 and res["request_id"] == "i1"
        assert s.models[0].calls == 1
        ws2.send(raw)  # an identical duplicate again: the same committed bytes
        assert unpack(ws2.recv(timeout=10))["sample_index"] == 0 and s.models[0].calls == 1
        res = infer(ws2, "s1", "t1", "i2")
        assert res["sample_index"] == 1 and s.models[0].calls == 2
        ws2.close()
    finally:
        s.close()


def test_rejections(server):
    ws, _ = server.connect()
    res = infer(ws, "ghost", "t0", "i0")  # an infer for an unknown session is never a new session
    assert res["type"] == "error" and res["error"]["code"] == "session_expired"
    ws, _ = server.connect()
    reset(ws, "s1", "t1", "r1")
    infer(ws, "s1", "t1", "i1")
    o = obs(q=np.ones((2, 7)))
    ws.send(pack({**ident("s1", "t1", "i1"), "type": "infer", "observation": o}))  # a reused id with other bytes
    res = unpack(ws.recv(timeout=10))
    assert res["type"] == "error" and res["error"]["code"] == "request_payload_mismatch"
    ws, _ = server.connect()
    ws.send(pack({**ident("s2", "t1", "r1"), "type": "reset", "seed": 1, "policy_id": "someone-else"}))
    assert unpack(ws.recv(timeout=10))["error"]["code"] == "identity_mismatch"
    ws, _ = server.connect()
    reset(ws, "s3", "t1", "r1")
    o = obs()
    o["instruction"] = " "
    assert infer(ws, "s3", "t1", "i1", o)["error"]["code"] == "invalid_instruction"


def test_clip32_stays_inside_float64_bounds():
    lo = np.asarray(TRACK["action_lower"], dtype=np.float64)
    hi = np.asarray(TRACK["action_upper"], dtype=np.float64)
    c = np.zeros((2, 16))
    c[0, 0], c[0, 7], c[1, 7], c[1, 15] = 7.0, 0.10000002, -0.001, 0.05
    out, moved = clip32(c, lo, hi)
    assert out.dtype == np.dtype("<f4") and (out.astype(np.float64) <= hi).all() and (out.astype(np.float64) >= lo).all()
    assert moved == 2  # 7.0 rad and -1 mm; the open gripper's 2e-8 is not counted
    assert out[0, 7] == np.float32(0.1)


class StubEstimator:
    """The fake world's poses in the estimator's block order (no network)."""

    def __init__(self, world):
        self.w = world

    def reset(self, variant):
        lay = Layout.load(variant)
        self.block_ids = block_order(lay)
        self.fixed = torch.tensor([bool(lay.fixed[b]) for b in self.block_ids])

    def __call__(self, images, q, widths, focus=None):
        idx = [self.w.ids.index(b) for b in self.block_ids]
        return self.w.pos[idx].copy(), self.w.quat[idx].copy(), np.ones(len(idx))


class FakeModels:
    def __init__(self, world):
        self.world = world
        self.bounds = (np.asarray(TRACK["action_lower"]), np.asarray(TRACK["action_upper"]))

    def adapter(self, recorder=None, log=print):
        return StudentAdapter(None, Oracle("graph", exact=False), lambda: StubEstimator(self.world), bounds=self.bounds,
                              log=log, recorder=recorder)


def test_student_adapter_end_to_end(tmp_path):
    w = FakeWorld(V)
    s = ServerThread(lambda sid, v: SessionModel(FakeModels(w), sid, v, tmp_path, EventLog(None, echo=None), save_images_every=2))
    try:
        ws, _ = s.connect()
        reset(ws, "s1", "t1", "r1")
        assert s.models[0].adapter.policy is not None  # built at the reset, from the variant header
        hold = infer(ws, "s1", "t1", "i0", obs(q=w.q, width=w.width, blank=True))["actions"]
        assert hold.shape == (1, 16) and np.allclose(hold[0, :7], w.q[0]) and np.allclose(hold[0, 8:15], w.q[1])
        lo, hi = s.models[0].adapter.bounds
        for i in range(1, 6):
            o = obs(q=w.q, width=w.width, blank=False)
            o["sim_time_s"], o["step_index"] = w.t, w.step
            a = infer(ws, "s1", "t1", f"i{i}", o)["actions"]
            assert a.dtype == np.dtype("<f4") and 1 <= a.shape[0] <= 16 and np.isfinite(a).all()
            assert (a.astype(np.float64) >= lo).all() and (a.astype(np.float64) <= hi).all()
            for row in a:
                w.apply(row)
        ws.close()
        time.sleep(0.3)  # the disconnect writes the episode summary
        d = tmp_path / "episodes" / V
        rows = (d / "t1.jsonl").read_text().splitlines()
        assert len(rows) == 6 and '"hold": "blank"' in rows[0]
        assert (d / "t1_summary.json").is_file() and len(list((d / "t1_images").glob("*.jpg"))) == 4 * 4  # calls 1, 2, 4, 6
    finally:
        s.close()
