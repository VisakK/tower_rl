"""Student plan Step 9c plumbing on the fake world (no Isaac): the DART hook records the planned Task 5 nodes, and the
distilled nodes driven by those clean labels (a replay network) reproduce the planned run."""

import numpy as np
import torch

from tower_rl.graph.executor import Executor
from tower_rl.graph.interface import ARMS
from tower_rl.student.distill.core import ACT_DIM, OBS_DIM, DistillNet
from tower_rl.student.distill.nodes import DartCollector
from tower_rl.student.fakeworld import FakeWorld

torch.set_num_threads(1)
V = "task5-2_tower_insertion_2"


def _pusher(ex):
    """Fake-world edit: the push-in arm's pinch shoves the inserted block along the slot axis (contact 86 mm behind its
    centre), which the kinematic world cannot do by itself."""
    def edit(w):
        run = ex.runs[0]
        node = run.node
        if node is None or node.spec["skill"] != "push_in" or getattr(node, "phase", 0) >= 2:
            return
        sp, _, d = (x[0].numpy().astype(np.float64) for x in node.slot)
        i = w.ids.index(node.spec["block"])
        pp, _ = w.pinch(node.arm)
        gap = float((pp - w.pos[i]) @ d) + 0.086
        if gap > 0 and abs(pp[2] - w.pos[i][2]) < 0.02:
            w.pos[i] = w.pos[i] + d * gap
    return edit


def _run(ex, seconds=90.0):
    ex.reset(1, V, [0])
    w = FakeWorld(V)
    w.edit = _pusher(ex)
    rows = []
    while w.t < seconds - 1e-6 and ex.runs[0].i < len(ex.runs[0].g["nodes"]):
        for r in ex.act([w.state()], [0])[0]:
            rows.append(np.asarray(r))
            w.apply(r)
    return rows, list(ex.runs[0].events)


class Replay:
    """A 'network' returning recorded actions in order (gripper bit -> logit +-5); keeps the observations it gets."""

    def __init__(self, acts):
        self.acts, self.i, self.seen = acts, 0, []

    def __call__(self, obs):
        assert obs.shape == (1, OBS_DIM)
        self.seen.append(obs[0].numpy().copy())
        a = torch.tensor(self.acts[min(self.i, len(self.acts) - 1)], dtype=torch.float32)[None].clone()
        a[0, 6] = 5.0 if a[0, 6] > 0.5 else -5.0
        self.i += 1
        return a


def test_dart_hook_records_and_replay_reproduces_the_planned_run():
    ex = Executor(V)
    col = DartCollector(seed=0, max_scale=0.0)
    ex.node_hook = col
    rows_a, ev_a = _run(ex)
    assert [e[1] for e in ev_a][:6] == ["i0_to_pregrasp", "i0_grasp", "i0_to_preinsert", "i0_insert", "i0_to_prepushin", "i0_push_in"]
    assert all(e[2] == "ok" for e in ev_a[:6]), ev_a
    segs, data = col.take(ex.runs[0])
    assert [s["skill"] for s in segs] == ["grasp_t5", "insert", "push_in"] and all(s["status"] == "ok" for s in segs)
    obs = np.stack([o for _, o, _ in data])
    act = np.stack([a for _, _, a in data])
    assert obs.shape[1] == OBS_DIM and act.shape[1] == ACT_DIM and np.isfinite(obs).all()
    assert (np.abs(act[:, :6]) < 0.999).all(), "a planned step reached the action limit"  # no clipping in nominal rows
    by = {k: [a for sid, _, a in data if sid == k] for k in range(3)}
    g = np.stack(by[0])
    assert g[0, 6] == 0 and g[-1, 6] == 1  # open on the descent, closed at the lift
    ins = np.stack(by[1])
    assert (ins[:-1, 6] == 1).all() and ins[-1, 6] == 0  # held slide, then the release row
    # distilled nodes driven by the clean labels reproduce the run
    ex2 = Executor(V, graph=dict(ex.graph, distilled=True))
    ex2.skills = {("grasp_t5_bc", "left"): Replay(by[0]), ("insert_bc", "left"): Replay(by[1]), ("push_in_bc", "left"): Replay(by[2])}
    rows_b, ev_b = _run(ex2)
    # the distilled nodes see the observations the networks were trained on (incl. the previous action as executed)
    for k, key in enumerate((("grasp_t5_bc", "left"), ("insert_bc", "left"), ("push_in_bc", "left"))):
        rec = np.stack([o for sid, o, _ in data if sid == k])
        got = np.stack(ex2.skills[key].seen)
        n = min(len(rec), len(got))
        assert abs(len(rec) - len(got)) <= 1, (key, len(rec), len(got))
        err = np.abs(rec[:n] - got[:n]).max(0)
        assert (err < 2e-3).all(), (key, np.nonzero(err >= 2e-3)[0].tolist(), err.max())
    assert [(e[1], e[2]) for e in ev_b] == [(e[1], e[2]) for e in ev_a]
    assert all(ev_b[k][3].get("distilled") for k in (1, 3))
    n = min(len(rows_a), len(rows_b))
    assert abs(len(rows_a) - len(rows_b)) <= 1 and np.abs(np.stack(rows_a[:n]) - np.stack(rows_b[:n])).max() < 2e-3


def test_dart_perturbs_and_labels_correct():
    ex = Executor(V)
    col = DartCollector(seed=3, max_scale=1.5)
    ex.node_hook = col
    _, ev = _run(ex)
    segs, data = col.take(ex.runs[0])
    assert segs and segs[0]["skill"] == "grasp_t5" and segs[0]["scale"] > 0
    g = np.stack([a for sid, _, a in data if sid == 0])
    assert np.abs(g[:, :3]).max() > 0.01  # corrective labels beyond the nominal descent


def test_distill_net_shapes():
    net = DistillNet()
    y = net(torch.zeros(4, OBS_DIM))
    assert y.shape == (4, ACT_DIM) and (y[:, :6].abs() <= 1).all()
    ts = torch.jit.script(net)
    assert torch.allclose(ts(torch.ones(2, OBS_DIM)), net(torch.ones(2, OBS_DIM)))


class _Out:
    def __init__(self, seed):
        self.extra, self.task_success = {"seed": seed, "case": f"c{seed}"}, True


def test_shards_round_trip_and_training(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from tower_rl.student.distill.data import load, write_batch

    for b in range(2):
        ex = Executor(V)
        col = DartCollector(seed=b, max_scale=1.0)
        ex.node_hook = col
        _run(ex)
        write_batch(tmp_path / f"k{b}", col, ex, [_Out(b)], V, "D0", 1000 + b, {"dart_scale": 1.0})
    d, index = load([tmp_path])
    assert len(index) == 6 and d["obs"].shape == (len(d["seg"]), OBS_DIM) and d["act"].shape[1] == ACT_DIM
    assert sorted({r["episode"] for r in index}) == [f"{V}/D0/1000/0", f"{V}/D0/1001/0"]
    for r in index:  # each segment's rows are contiguous and numbered from 0
        rows = d["row"][d["seg"] == r["id"]]
        assert rows.tolist() == list(range(len(rows))) and len(rows) == r["rows"]
    spec = importlib.util.spec_from_file_location("train_distill", Path(__file__).resolve().parents[1] / "scripts/train_distill.py")
    td = importlib.util.module_from_spec(spec)
    sys.modules["train_distill"] = td
    spec.loader.exec_module(td)
    args = SimpleNamespace(epochs=2, batch=64, lr=1e-3, seed=0, only_ok=False, data=[str(tmp_path)], w_rare_max=10.0,
                           obs_noise=0.05, weight_decay=1e-4, use_prev_a=False)
    td.val_episode = lambda ep: ep.endswith("1001/0")
    m = td.train_one(d, index, "insert", "left", args, tmp_path)
    assert m is not None and m["rows_val"] > 0 and (tmp_path / "insert_bc_left.pt").exists()
    from tower_rl.graph.executor import SkillPolicy

    pol = SkillPolicy.load(tmp_path / "insert_bc_left.pt")
    assert pol.obs_dim == OBS_DIM and pol(torch.zeros(1, OBS_DIM)).shape == (1, ACT_DIM)


def test_pushin_kin_ignores_a_drifting_estimate():
    """``pushin_kin``: the inserted block's estimate drifts 18 mm back along the slot axis and 7 deg in yaw during the
    push (as kp_v2 does when the block disappears into the tower); the push still ends with the block home and the
    verdict ok, while without the flag the nudges overshoot / the verdict fails."""
    import math

    from tower_rl import geometry as geo

    def run(flag):
        ex = Executor(V, graph=dict(Executor(V).graph, pushin_kin=flag))
        ex.reset(1, V, [0])
        w = FakeWorld(V)
        w.edit = _pusher(ex)
        drift = {"on": False}

        def state():
            st = w.state()
            node = ex.runs[0].node
            if node is not None and node.spec["skill"] == "push_in" and getattr(node, "phase", 0) >= 1:
                drift["on"] = True
            if drift["on"]:
                i = st.block_ids.index("source_l1_insert")
                d = node.slot[2][0].numpy() if node is not None and node.spec["skill"] == "push_in" else np.array([1.0, 0, 0])
                st.block_pos[i] = st.block_pos[i] - 0.018 * d
                st.block_quat[i] = geo.quat_mul(geo.quat_from_yaw(torch.tensor([math.radians(7.0)], dtype=torch.float64)),
                                                torch.as_tensor(st.block_quat[i])[None])[0].numpy()
                st.block_conf[i] = 0.6
            return st

        while w.t < 90.0 and ex.runs[0].i < len(ex.runs[0].g["nodes"]):
            for r in ex.act([state()], [0])[0]:
                w.apply(r)
        return {e[1]: e for e in ex.runs[0].events}
    ev_on = run(True)
    assert ev_on["i0_push_in"][2] == "ok", ev_on["i0_push_in"]
    assert abs(ev_on["i0_push_in"][3]["kin_along_mm"]) <= 4.0
    ev_off = run(False)
    assert ev_off["i0_push_in"][2] == "failed", ev_off["i0_push_in"]
