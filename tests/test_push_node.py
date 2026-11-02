"""The Task 4 push node's departure test on perceived poses (graph ``depart_seen``): pure python, a static scene and a
stand-in for the learned push (67-D, seen inputs) that keeps the arm still."""

import copy

import numpy as np
import torch

from tower_rl.config import load_yaml
from tower_rl.data.layouts import Layout
from tower_rl.graph.executor import Executor, PushNode
from tower_rl.graph.graph import load_graph
from tower_rl.graph.interface import ARMS, State, Status
from tower_rl.student.spec import Decision, TaskSpec

V4 = "task4-2_full_tower_pushdown_2"


class _Still:
    """A 67-D push actor (trained with the estimator's losses) that commands no motion."""

    obs_dim = 67

    def __call__(self, obs):
        return torch.zeros(obs.shape[0], 7)


def _push(flags: dict, rows: list[tuple[float, float]], k: int = 5):
    """Goal k's push node on a static scene (every other goal's block pushed flat and clear); per row, the target's
    perceived pose ``along`` its push from where it rests on the tower, with confidence ``conf``. Returns the node and
    the row index and status it ended with (None: still running after the last row)."""
    g = copy.deepcopy(load_graph(V4))
    g.update(flags)
    ex = Executor(V4, graph=g)
    ex.reset(1, V4, [0])
    run = ex.runs[0]
    spec = TaskSpec.load(V4)
    lay = Layout.load(V4)
    ids = list(lay.blocks)
    pos = np.stack([lay.blocks[b][:3] for b in ids]).astype(float)
    quat = np.stack([lay.blocks[b][3:7] for b in ids]).astype(float)
    p = pos.copy()
    for gl in spec.goals:
        if gl is not spec.goals[k]:
            b = ids.index(gl.block)
            p[b] = pos[b] + np.asarray(gl.direction) * 0.18
            p[b, 2] = 0.0147
    gk = spec.goals[k]
    b = ids.index(gk.block)
    d = np.asarray(gk.direction, dtype=float)
    home = load_yaml("scene")["home_qpos"]

    def state(t, along, conf):
        pp = p.copy()
        pp[b] = pos[b] + d * along
        cf = np.ones(len(ids))
        cf[b] = conf
        return State(t=t, step=0, variant=V4, instruction="", q=np.array([home["left"][:7], home["back"][:7]]),
                     width=np.full(2, 0.1), block_ids=ids, block_pos=pp, block_quat=quat, block_conf=cf)

    s = state(0.0, 0.0, 1.0)
    run._cmd = {}
    for a in ARMS:
        run.ctrl[a].reset(run.q_meas(s, a))
        run.command(a)
    run.ready = True
    run.observe(s)
    nspec = spec.option_nodes(Decision("push_off", k, gk.arm))[1]
    assert nspec["skill"] == "push"
    node = PushNode(run, nspec, s, policy=_Still())
    for i, (along, conf) in enumerate(rows):
        s = state(0.1 * (i + 1), along, conf)
        run.observe(s)
        status, _ = node.step(s)
        if status is not None:
            return node, i, status
    return node, None, None


def test_depart_seen_ignores_coasted_departures():
    """Camera loop 2026-10-28: the estimator coasted a lost block past the departure radius while it had stopped leaning
    on the base edge, and the push ended there. With ``depart_seen`` only a confident row ends it."""
    from tower_rl import push as push_geo

    out = 0.20  # perceived along the push: departed (r >= depart_m)
    assert push_geo.cfg()["depart_m"] <= 0.16
    coasted = [(out, 0.3)] * 10
    node, i, status = _push({}, coasted)  # v1.1: ended on the coasted pose (then the planned retreat)
    assert status == Status.OK and i <= 6
    node, i, status = _push({"depart_seen": True}, coasted)
    assert status is None and node.retreat is None
    node, i, status = _push({"depart_seen": True}, coasted + [(out, 1.0)] * 8)  # seen out: departed
    assert status == Status.OK and 10 <= i <= 16
    # seen well out (not departed), then lost for long: still ends (lost_departed)
    node, i, status = _push({"depart_seen": True}, [(0.08, 1.0)] * 3 + [(out, 0.3)] * 30)
    assert status == Status.OK and node.info.get("why") == "lost_departed" and i >= 3 + 15


def test_depart_seen_off_keeps_v1_behaviour():
    """Without the flag the node is unchanged: a confident departed row ends it, a coasted one too."""
    node, i, status = _push({}, [(0.20, 1.0)] * 10)
    assert status == Status.OK and i <= 6
    node, i, status = _push({"depart_seen": False}, [(0.20, 0.3)] * 10)
    assert status == Status.OK and i <= 6
