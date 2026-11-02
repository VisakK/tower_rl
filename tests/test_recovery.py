"""The recovery node (Plan-v1.1 §5): contracts on hand-made scenes, the recovery option's node specs, and the rules'
decisions on stub views (pure python)."""

import math

import numpy as np

from tower_rl.data.layouts import Layout
from tower_rl.eval import mirror
from tower_rl.graph.interface import ARMS, State
from tower_rl.recovery.contracts import Contracts
from tower_rl.recovery.rules import RecoveryNode
from tower_rl.student.spec import Decision, TaskSpec

V4 = "task4-2_full_tower_pushdown_2"
V5 = "task5-2_tower_insertion_2"


def _scene(variant):
    lay = Layout.load(variant)
    ids = list(lay.blocks)
    pos = np.stack([lay.blocks[b][:3] for b in ids]).astype(float)
    quat = np.stack([lay.blocks[b][3:7] for b in ids]).astype(float)
    return ids, pos, quat


def _quat_axis(axis, ang):
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * a]


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def test_pushdown_contracts_on_rest_states():
    spec = TaskSpec.load(V4)
    ids, pos, quat = _scene(V4)
    C = Contracts(spec, ids, pos, quat)
    k = spec.K - 1  # the last push: source_l1_0, left arm, +x
    b = ids.index(spec.goals[k].block)
    assert C.check(k, pos, quat).kind == "on tower"
    d = np.asarray(spec.goals[k].direction, dtype=float)
    p = pos.copy()
    p[b] = pos[b] + d * 0.17
    p[b, 2] = 0.0147
    ck = C.check(k, p, quat)
    assert ck.ok and ck.kind == mirror.FLAT and ck.detail["gap_mm"] > 2
    # leaning on the foundation edge: centre ~0.145 m out, z 31 mm, tilted 13 deg about the jaw axis (down at the far end)
    p[b] = pos[b] + d * 0.135
    p[b, 2] = 0.031
    q = quat.copy()
    lat = np.cross([0, 0, 1.0], d)
    q[b] = _qmul(_quat_axis(lat, math.radians(13)), quat[b])
    ck = C.check(k, p, q)
    assert not ck.ok and ck.kind in ("on/leaning on base", "touching base"), ck
    assert C.structure(p, q, after_goal=k) == (True, 0.0)


def test_insertion_contracts():
    spec = TaskSpec.load(V5)
    ids, pos, quat = _scene(V5)
    C = Contracts(spec, ids, pos, quat)
    g = spec.goals[0]
    b = ids.index(g.block)
    centre, ax = C.slots[0]
    p = pos.copy()
    p[b] = centre
    assert C.check(0, p, quat).kind == "home"
    p[b] = centre - ax * 0.03
    ck = C.check(0, p, quat)
    assert ck.kind == "short" and ck.detail["along_mm"] < -25
    p[b] = centre + np.array([-ax[1], ax[0], 0.0]) * 0.05
    assert C.check(0, p, quat).kind == "out"
    p2 = p.copy()
    s0 = C.structure_ids[0]
    p2[s0] = pos[s0] + [0.02, 0, 0]
    assert C.structure(p2, quat)[0] is False


def test_repush_option_nodes():
    spec = TaskSpec.load(V4)
    assert "repush" in spec.options
    nodes = spec.option_nodes(Decision("repush", 5, "left"))
    assert [n["id"] for n in nodes] == ["p5_re_to_prepush", "p5_re_push"]
    assert all(n["recover"] and n["arm"] == "left" and n["block"] == spec.goals[5].block for n in nodes)
    assert "repush" not in TaskSpec.load(V5).options


class _Mem:
    def __init__(self, K, started):
        from tower_rl.student.runtime import GoalMemory
        self.goals = [GoalMemory() for _ in range(K)]
        for k in started:  # pushed, the verdict ok
            self.goals[k].attempts["left"] = 1
            self.goals[k].last_option, self.goals[k].last_status = "push_off", "ok"


class _Run:
    blind = False


class _View:
    def __init__(self, spec, s, mem):
        self.spec, self.state, self.memory, self.run = spec, s, mem, _Run()

    def holding(self, arm):
        return False

    def holds_goal(self, arm, k):
        return False


def test_rules_repush_the_last_layer_and_abstain_above():
    spec = TaskSpec.load(V4)
    ids, pos, quat = _scene(V4)

    def state(p, q, t):
        return State(t=t, step=0, variant=V4, instruction="", q=np.zeros((2, 7)), width=np.full(2, 0.1), block_ids=ids,
                     block_pos=p, block_quat=q, block_conf=np.ones(len(ids)),
                     privileged={"true_block_pos": p, "true_block_quat": q, "block_speed": np.zeros(len(ids))})

    node = RecoveryNode("truth")
    node.start(_View(spec, state(pos, quat, 0.0), _Mem(spec.K, [])))
    # every block pushed flat and clear, the last one leaning on the foundation: re-push it
    p, q = pos.copy(), quat.copy()
    for g in spec.goals:
        b = ids.index(g.block)
        p[b] = pos[b] + np.asarray(g.direction) * 0.18
        p[b, 2] = 0.0147
    k = spec.K - 1
    b = ids.index(spec.goals[k].block)
    d = np.asarray(spec.goals[k].direction, dtype=float)
    p[b] = pos[b] + d * 0.135
    p[b, 2] = 0.031
    q[b] = _qmul(_quat_axis(np.cross([0, 0, 1.0], d), math.radians(13)), quat[b])
    v = _View(spec, state(p, q, 60.0), _Mem(spec.K, range(spec.K)))
    rd = node.propose(v, Decision("idle"))
    assert rd == Decision("repush", k, spec.goals[k].arm, recovery=True)
    # a bottom-layer block leaning while a block of the layer above is still on the tower (out of order): no re-push
    node2 = RecoveryNode("truth")
    node2.start(_View(spec, state(pos, quat, 0.0), _Mem(spec.K, [])))
    p2, q2 = pos.copy(), quat.copy()
    k3 = 3  # source_l1_2 (layer 1); goals 0-2 are layer 2
    b3 = ids.index(spec.goals[k3].block)
    d3 = np.asarray(spec.goals[k3].direction, dtype=float)
    p2[b3] = pos[b3] + d3 * 0.135
    p2[b3, 2] = 0.031
    q2[b3] = _qmul(_quat_axis(np.cross([0, 0, 1.0], d3), math.radians(13)), quat[b3])
    v2 = _View(spec, state(p2, q2, 20.0), _Mem(spec.K, [k3]))
    assert node2.propose(v2, Decision("push_off", 4, "left")) is None
    assert any(a["why"] == "R4.1 preconditions" for a in node2.abstentions)
    # a block lying flat against the base ("touching base") is not re-pushed (its pre-push pose is inside the base)
    node3 = RecoveryNode("truth")
    node3.start(_View(spec, state(pos, quat, 0.0), _Mem(spec.K, [])))
    p3 = p.copy()
    p3[b] = pos[b] + d * 0.15
    p3[b, 2] = 0.0147
    v3 = _View(spec, state(p3, quat, 60.0), _Mem(spec.K, range(spec.K)))
    assert node3.propose(v3, Decision("idle")) is None


def _repush(flag, along=0.155, z=0.031, tilt_deg=13.0, k=5, rows=300, coast=None):
    """The recovery node's re-push on a static scene (no physics: the block stays where it is): goal k's block out from
    the foundation, tilted with its near end up, every other block pushed flat and clear. ``coast`` (m along the push):
    from the node's start the target's perceived pose is that far off, unconfident (the estimator's coasted pose).
    Returns the node, its status and the commanded pinch's travel along the push past where the node started, per row."""
    import copy

    import torch

    from tower_rl import push as push_geo
    from tower_rl.config import load_yaml
    from tower_rl.graph.executor import Executor, make_node
    from tower_rl.graph.graph import load_graph

    g = copy.deepcopy(load_graph(V4))
    if flag is not None:
        g["repush_v2"] = flag
    ex = Executor(V4, graph=g)
    ex.reset(1, V4, [0])
    run = ex.runs[0]
    spec = TaskSpec.load(V4)
    ids, pos, quat = _scene(V4)
    p, q = pos.copy(), quat.copy()
    for gl in spec.goals:
        b = ids.index(gl.block)
        p[b] = pos[b] + np.asarray(gl.direction) * 0.18
        p[b, 2] = 0.0147
    gk = spec.goals[k]
    b = ids.index(gk.block)
    d = np.asarray(gk.direction, dtype=float)
    p[b] = pos[b] + d * along
    p[b, 2] = z
    q[b] = _qmul(_quat_axis(np.cross([0, 0, 1.0], d), math.radians(tilt_deg)), quat[b])
    home = load_yaml("scene")["home_qpos"]

    def state(t):
        pp, cf = p, np.ones(len(ids))
        if coast is not None and t > 0:
            pp = p.copy()
            pp[b] = p[b] + d * coast
            cf[b] = 0.3
        return State(t=t, step=0, variant=V4, instruction="", q=np.array([home["left"][:7], home["back"][:7]]),
                     width=np.full(2, 0.1), block_ids=ids, block_pos=pp, block_quat=q, block_conf=cf)

    s = state(0.0)
    run._cmd = {}
    for a in ARMS:
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
    run.ready = True
    run.observe(s)
    nspec = spec.option_nodes(Decision("repush", k, gk.arm))[1]
    assert nspec["recover"]
    dt = torch.tensor(d, dtype=torch.float32)[None]
    ctrl = run.ctrl[gk.arm]
    pre, pq = push_geo.prepush_pose(torch.tensor(p[b], dtype=torch.float32)[None], dt, gk.arm, nspec["layer"])
    ctrl.set_target(pre, pq)  # where the "_re_to_prepush" transit leaves the pinch
    node = make_node(run, nspec, s)
    status, travel = None, []
    for i in range(rows):
        s = state(0.1 * (i + 1))
        run.observe(s)
        status, _ = node.step(s)
        travel.append(float(((ctrl.pos - pre) * dt).sum()))
        if status is not None:
            break
    return node, status, np.array(travel)


def test_repush_v2_pushes_its_stroke_and_confirms_flat():
    """Camera loop 2026-10-28: a re-push starts on a block that is already radially out, and the per-row estimate put it
    3.5 mm clear while it still leaned (raised): v1.1's re-push confirmed it at once without pushing. ``repush_v2``
    pushes ``min_push_m`` past contact first and confirms only "flat clear" (bounded by ``max_total_m``)."""
    from tower_rl.graph.executor import REPUSH_V2
    from tower_rl.graph.interface import Status

    node, status, travel = _repush(None)  # v1.1: confirmed "clear" on the gap alone, no stroke
    assert status == Status.OK and node.info["along_mm"] == 0.0 and node.info["gap_mm"] >= 2.0
    assert len(travel) <= 12 and travel.max() < 0.005
    node, status, travel = _repush(True)  # still raised: pushed, then finished up to the cap, never confirmed
    assert status == Status.OK and node.info.get("unconfirmed")
    stroke = float(((node.run.ctrl["left"].pos - node.stroke0) * node.dir).sum())
    assert REPUSH_V2["min_push_m"] <= stroke <= REPUSH_V2["max_total_m"] + 0.01
    node, status, travel = _repush({"max_total_m": 0.1})  # student_v1_2's cap (graph dict form)
    stroke = float(((node.run.ctrl["left"].pos - node.stroke0) * node.dir).sum())
    assert status == Status.OK and node.info.get("unconfirmed") and stroke <= 0.1 + 0.01
    node, status, travel = _repush(True, along=0.18, z=0.0147, tilt_deg=0.0)  # flat clear: the minimum stroke, confirmed
    assert status == Status.OK and not node.info.get("unconfirmed")
    stroke = float(((node.run.ctrl["left"].pos - node.stroke0) * node.dir).sum())
    assert REPUSH_V2["min_push_m"] <= stroke < REPUSH_V2["min_push_m"] + 0.02


def test_repush_v2_aims_at_the_last_confident_pose():
    """Camera loop 2026-10-28: after p5's push the estimator coasted the leaning block 145 mm back towards the tower
    (unconfident) and the re-push went to the contact pose behind that pose. With ``repush_v2`` an unseen row's pose is
    the last confident one (here the rest estimate), so the stroke starts behind the block itself."""
    import torch

    from tower_rl import push as push_geo

    node, status, travel = _repush(True, coast=-0.145, rows=400)
    assert node.stroke0 is not None
    bp = node.seen_pose[0]
    cp, _ = push_geo.push_pose(bp, node.dir, node.arm, node.pg["contact_along_m"] - 0.006, node.zoff)
    assert abs(float(((node.stroke0 - cp) * node.dir).sum())) < 0.01
    assert float(((bp - node.p0) * node.dir).sum()) == 0.0  # never replaced by the coasted pose
    assert float(torch.as_tensor(node.info.get("radial_along_mm", 0.0))) > -1.0


class _RunRest:
    """A runtime stub for the estimate source: rest estimates fixed to the given poses."""

    blind = False

    def __init__(self, pos, quat):
        self.pos, self.quat = pos, quat

    def block_rest(self, s, b):
        import torch

        i = s.block_ids.index(b)
        return (torch.tensor(self.pos[i], dtype=torch.float32)[None], torch.tensor(self.quat[i], dtype=torch.float32)[None])


def test_rules_seen_s_acts_on_sporadic_sightings():
    """Camera loop 2026-10-28: a block left leaning on the base edge was confidently seen on 4 of ~30 idle calls (1.6 s
    apart, never two in a row); the default gate never passes, ``seen_s`` acts on consistent sporadic sightings and
    still refuses a block whose sightings disagree (it moved)."""
    spec = TaskSpec.load(V4)
    ids, pos, quat = _scene(V4)
    p, q = pos.copy(), quat.copy()
    for g in spec.goals:
        b = ids.index(g.block)
        p[b] = pos[b] + np.asarray(g.direction) * 0.18
        p[b, 2] = 0.0147
    k = spec.K - 1
    b = ids.index(spec.goals[k].block)
    d = np.asarray(spec.goals[k].direction, dtype=float)
    p[b] = pos[b] + d * 0.135
    p[b, 2] = 0.031
    q[b] = _qmul(_quat_axis(np.cross([0, 0, 1.0], d), math.radians(13)), quat[b])

    def view(t, conf_b, pb, rest, mem):
        cf = np.ones(len(ids))
        cf[b] = conf_b
        pp = p.copy()
        pp[b] = pb
        s = State(t=t, step=0, variant=V4, instruction="", q=np.zeros((2, 7)), width=np.full(2, 0.1), block_ids=ids,
                  block_pos=pp, block_quat=q, block_conf=cf)
        v = _View(spec, s, mem)
        v.run = _RunRest(rest, q)
        return v

    def run(node, sightings, moved=0.0):
        """Idle calls every 1.6 s from 40 s; the block confidently seen at the given call indices (the later ones
        ``moved`` m along its push), coasted 0.145 m back otherwise. Returns the last call's proposal."""
        mem = _Mem(spec.K, range(spec.K))
        node.start(view(0.0, 1.0, pos[b], pos, _Mem(spec.K, [])))
        rest = p.copy()
        out = None
        for n in range(12):
            t = 40.0 + 1.6 * n
            seen = n in sightings
            pb = p[b] + d * (moved if (seen and n == max(sightings)) else 0.0) if seen else p[b] - d * 0.145
            if seen and n == max(sightings):
                rest = p.copy()
                rest[b] = pb
            v = view(t, 1.0 if seen else 0.3, pb, rest, mem)
            node.observe(v)
            if n == 11:
                out = node.propose(v, Decision("idle"))
        return out

    assert run(RecoveryNode("estimate"), [2, 9]) is None  # v1.1's gate: two calls in a row, the decision call seen
    rd = run(RecoveryNode("estimate", seen_s=15.0), [2, 9])
    assert rd == Decision("repush", k, spec.goals[k].arm, recovery=True)
    assert run(RecoveryNode("estimate", seen_s=15.0), [9]) is None  # one sighting only
    assert run(RecoveryNode("estimate", seen_s=15.0), [2, 9], moved=0.01) is None  # the sightings disagree: it moved


def test_rules_at_end_waits_for_an_empty_tower():
    """Camera loop 2026-10-28 (heavy, injected short push): R4.1 re-pushed a layer-2 block that had fallen against a
    layer-1 block's end while layer 1 was still on the tower, and the re-push moved two layer-1 blocks 19-22 mm.
    ``at_end`` re-pushes only once every goal was pushed, nothing is left on the tower and the base decider idles."""
    spec = TaskSpec.load(V4)
    ids, pos, quat = _scene(V4)

    def state(p, q, t):
        return State(t=t, step=0, variant=V4, instruction="", q=np.zeros((2, 7)), width=np.full(2, 0.1), block_ids=ids,
                     block_pos=p, block_quat=q, block_conf=np.ones(len(ids)),
                     privileged={"true_block_pos": p, "true_block_quat": q, "block_speed": np.zeros(len(ids))})

    k0 = 0  # source_l2_2: layer 2, the back arm, +y
    assert spec.goals[k0].layer == 2
    p, q = pos.copy(), quat.copy()
    for k in (1, 2):  # the rest of layer 2 pushed flat and clear; layer 1 (goals 3-5) still on the tower
        b = ids.index(spec.goals[k].block)
        p[b] = pos[b] + np.asarray(spec.goals[k].direction) * 0.18
        p[b, 2] = 0.0147
    b0 = ids.index(spec.goals[k0].block)
    d0 = np.asarray(spec.goals[k0].direction, dtype=float)
    p[b0] = pos[b0] + d0 * 0.135
    p[b0, 2] = 0.031
    q[b0] = _qmul(_quat_axis(np.cross([0, 0, 1.0], d0), math.radians(13)), quat[b0])
    for at_end, want in ((False, Decision("repush", k0, spec.goals[k0].arm, recovery=True)), (True, None)):
        node = RecoveryNode("truth", at_end=at_end)
        node.start(_View(spec, state(pos, quat, 0.0), _Mem(spec.K, [])))
        v = _View(spec, state(p, q, 22.0), _Mem(spec.K, [0, 1, 2]))
        assert node.propose(v, Decision("push_off", 3, spec.goals[3].arm)) == want, at_end
    # the plan done (layer 1 pushed flat and clear too), the base decider idling: re-push
    p2 = p.copy()
    for k in (3, 4, 5):
        b = ids.index(spec.goals[k].block)
        p2[b] = pos[b] + np.asarray(spec.goals[k].direction) * 0.18
        p2[b, 2] = 0.0147
    node = RecoveryNode("truth", at_end=True)
    node.start(_View(spec, state(pos, quat, 0.0), _Mem(spec.K, [])))
    v = _View(spec, state(p2, q, 60.0), _Mem(spec.K, range(spec.K)))
    assert node.propose(v, Decision("push_off", 3, spec.goals[3].arm)) is None  # not idling: the decider's turn
    assert node.propose(v, Decision("idle")) == Decision("repush", k0, spec.goals[k0].arm, recovery=True)
    # camera loop (irregular_layers): the decider skipped goal 3, its block already off the tower: still the end
    node = RecoveryNode("truth", at_end=True)
    node.start(_View(spec, state(pos, quat, 0.0), _Mem(spec.K, [])))
    v = _View(spec, state(p2, q, 60.0), _Mem(spec.K, [0, 1, 2, 4, 5]))
    assert node.propose(v, Decision("idle")) == Decision("repush", k0, spec.goals[k0].arm, recovery=True)
    # ... but not while that block still stands on the tower
    p3 = p2.copy()
    b3 = ids.index(spec.goals[3].block)
    p3[b3] = pos[b3]
    node = RecoveryNode("truth", at_end=True)
    node.start(_View(spec, state(pos, quat, 0.0), _Mem(spec.K, [])))
    v = _View(spec, state(p3, q, 60.0), _Mem(spec.K, [0, 1, 2, 4, 5]))
    assert node.propose(v, Decision("idle")) is None
