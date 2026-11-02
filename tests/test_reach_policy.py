"""The deployable reacher (``tower_rl.skills.reach_policy``) and the moving goal (``reach_tasks.Carrot``), no Isaac."""

import math

import torch

from tower_rl import geometry as geo
from tower_rl.config import deep_update, load_yaml
from tower_rl.skills.reach_policy import Reacher
from tower_rl.skills.reach_tasks import Carrot, via_points


def _skill(**over):
    return deep_update(load_yaml("skills/reach"), over)


def _obs(over, n=2):
    rch = Reacher(None, "left", _skill(**over), n=n)
    q = torch.zeros(n, 7)
    p, qq = rch.kin.fk(q, "pinch")
    return rch.obs(q, p, qq, width=torch.full((n,), 0.0497), regime=1.0, goal_v=torch.arange(6.0).expand(n, 6))


def test_obs_layout_per_option():
    # base 55 + fine 9 (the default config), + speed 1, + opening / regime 2, + goal velocity 6
    assert _obs({}).shape == (2, 64)
    assert _obs({"speed": {"obs": True}}).shape == (2, 65)
    held = {"frac": 0.6, "obs": True}
    assert _obs({"speed": {"obs": True}, "held_block": held}).shape == (2, 67)
    o = _obs({"speed": {"obs": True}, "held_block": held, "track": {"obs": True}})
    assert o.shape == (2, 73)
    assert torch.allclose(o[:, 65], torch.tensor((0.0497 - 0.05) / 0.05)) and bool((o[:, 66] == 1.0).all())
    assert torch.equal(o[0, 67:], torch.arange(6.0))
    # held_block.obs without a block in the scene (frac 0) adds nothing, as in the env
    assert _obs({"held_block": {"frac": 0.0, "obs": True}}).shape == (2, 64)


def test_goal_velocity_to_base_frame():
    rch = Reacher(lambda x: torch.zeros(x.shape[0], 7), "back", _skill(track={"obs": True}), n=1)
    seen = {}
    rch.policy = lambda x: (seen.setdefault("obs", x), torch.zeros(1, 7))[1]
    q = torch.zeros(1, 7)
    rch.reset(q)
    pw, qw = rch.base.to_world(*rch.kin.fk(q, "pinch"))
    v_w = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.5]])
    rch.step(q, pw, qw, goal_v_w=v_w)
    vb = seen["obs"][0, -6:]
    assert torch.allclose(rch.base.vec_to_world(vb[:3][None])[0], v_w[0, :3], atol=1e-6)
    assert torch.allclose(rch.base.vec_to_world(vb[3:][None])[0], v_w[0, 3:], atol=1e-6)


def test_carrot_speed_limit_and_stop():
    p0, q0 = torch.zeros(3, 3), torch.tensor([[1.0, 0, 0, 0]]).repeat(3, 1)
    tp = torch.tensor([[0.3, 0, 0], [0, 0.05, 0], [0, 0, 0]])
    tq = geo.quat_from_yaw(torch.tensor([0.0, 0.0, 0.5]))
    v = torch.tensor([0.15, 0.4, 0.2])
    car = Carrot(p0, q0, v, torch.tensor([0.6, 1.0, 0.6]), 1.0)
    prev, on = car.p.clone(), torch.zeros(3, dtype=torch.bool)
    for _ in range(200):
        on = car.step(tp, tq, 0.1)
        assert bool(((car.p - prev).norm(dim=-1) <= v * 0.1 + 1e-6).all())  # never faster than its limit
        prev = car.p.clone()
    assert bool(on.all()) and torch.allclose(car.p, tp) and float(geo.quat_angle(car.q, tq).max()) < 1e-5


def test_via_points_rise_and_travel_height():
    p0 = torch.tensor([[0.4, -0.3, 0.12]])
    p1 = torch.tensor([[0.2, -0.5, 0.10]])
    q = torch.tensor([[1.0, 0, 0, 0]])
    poses, valid = via_points(p0, q, p1, q, torch.tensor([0.10]), torch.tensor([True]))
    zt = 0.10 + 0.03 + 0.03  # tallest block + margin + held extra
    assert valid.tolist() == [[True, True, True]]
    assert math.isclose(float(poses[0, 0, 2]), max(zt, 0.12 + 0.05), abs_tol=1e-6)
    assert torch.allclose(poses[0, 2, :3], p1[0])


def test_chunk_rows_act_on_prediction():
    # chunk-trained models (Reacher-V2.MD §7.1): a row without a measurement acts on q_des less the offset measured at
    # the chunk's first row, and sees the rows since that measurement / k_norm (the last input)
    assert _obs({"chunk": {"obs": True}}).shape == (2, 65)
    rch = Reacher(None, "left", _skill(chunk={"obs": True}), n=1)
    seen = []
    rch.policy = lambda x: (seen.append(x.clone()), torch.full((1, 7), 0.5))[1]
    q0 = torch.full((1, 7), 0.1)
    rch.reset(q0)
    pw, qw = rch.base.to_world(*rch.kin.fk(q0, "pinch"))
    qm = q0 + 0.01  # the arm measured 0.01 rad past q_des
    rch.step(qm, pw, qw)
    off = rch.clamp(q0) - qm
    assert rch.k == 0 and torch.allclose(rch.off, off) and float(seen[-1][0, -1]) == 0.0
    qd1 = rch.q_des.clone()
    rch.step(None, pw, qw)
    q_seen = seen[-1][0, 9:16] * rch.q_half + rch.q_mid  # the (q - q_mid) / q_half inputs
    assert rch.k == 1 and torch.allclose(q_seen, (qd1 - off)[0], atol=1e-6)
    assert math.isclose(float(seen[-1][0, -1]), 1 / 16, rel_tol=1e-6)
    assert torch.allclose(rch.prev_q, qd1 - off)  # the next row's velocity input is from this row's view
    rch.step(qm + 0.02, pw, qw)  # the next call's measurement: a new offset, k back to 0
    assert rch.k == 0 and float(seen[-1][0, -1]) == 0.0


def test_via_stall_rule():
    # executor reach.via_stall_s (10-31): a via-point is done within via_tol, or within its position part once the carrot
    # has sat on it via_stall_s; never past the position tolerance, never for the goal itself
    from tower_rl.graph.executor import REACH_DEFAULTS, LearnedTransitNode

    node = object.__new__(LearnedTransitNode)
    node.rc = {**REACH_DEFAULTS, "via_stall_s": 2.0}
    node.cur, node.slot_rows = 1, 0
    p, q = torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q20 = geo.quat_from_axis_angle(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([math.radians(20.2)]))
    near = p + 0.004  # 6.9 mm away
    assert node._via_reached(p, q, near, q)
    assert not node._via_reached(p, q20, near, q)  # 20.2 deg > via_tol's 20
    node.slot_rows = 20  # 2 s on the via-point
    assert node._via_reached(p, q20, near, q)
    assert not node._via_reached(p + 0.02, q20, p, q)  # 3.5 cm off
    node.rc["via_stall_s"] = 0.0  # off: the orientation test alone
    assert not node._via_reached(p, q20, near, q)
    node.cur = 2
    assert not node._via_reached(p, q, p, q)  # the goal is not a via-point


def test_reach_policy_bundle_paths(tmp_path, monkeypatch):
    # executor.reach_policy (student_v1_4): a relative reach.policy path is the bundle's file when the bundle has it, else
    # relative to the working directory (rehearse.py --graph_set @configs/reach/<name>.json on any bundle); the student
    # runtime (the server's executor) keeps the bundle dir
    import json
    from types import SimpleNamespace

    from tower_rl.graph.executor import Executor, reach_policy
    from tower_rl.graph.graph import load_graph
    from tower_rl.student.runtime import StudentExecutor

    def export(d, tag):
        d.mkdir(parents=True)
        torch.jit.script(torch.nn.Linear(4, 7)).save(str(d / "r.pt"))
        (d / "r.json").write_text(json.dumps({"skill": {"tag": tag}}))

    v, b = "task1-3_tower_transfer_3", tmp_path / "bundle"
    export(b / "reach", "bundle")
    export(tmp_path / "cwd" / "reach", "cwd")
    export(tmp_path / "cwd" / "elsewhere", "cwd only")
    g = load_graph(v)
    g["reach"] = {"mode": "learned", "policy": {"left": "reach/r.pt", "back": "elsewhere/r.pt"}}
    (b / "graphs").mkdir()
    (b / "graphs" / f"{v}.json").write_text(json.dumps(g))
    monkeypatch.chdir(tmp_path / "cwd")
    for ex in (Executor.from_bundle(str(b), v), StudentExecutor.from_bundle(str(b), v)):
        run = SimpleNamespace(ex=ex, g=ex.graph)
        assert reach_policy(run, "left")[1] == {"tag": "bundle"}  # the bundle's copy, not the working directory's
        assert reach_policy(run, "back")[1] == {"tag": "cwd only"}  # not in the bundle
    ex = Executor(v, g)  # no bundle
    assert reach_policy(SimpleNamespace(ex=ex, g=g), "left")[1] == {"tag": "cwd"}
