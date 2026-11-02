"""A kinematic stand-in for the station scene (tests only): deterministic ``State`` streams for running the executor
and the student runtime side by side without Isaac.

The arms settle on their commands' sag equilibrium within a row; a gripper commanded to close with its pinch within
``grab_m`` of a block's centre grabs it (measured opening = the block width, the block follows the pinch); opening
releases it where it is, dropped onto the highest support below it (other blocks or the table) with an optional
scripted placement error. No contact dynamics: pushes move a block only by the scripted ``push`` callback.
"""

from __future__ import annotations

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.config import load_yaml
from tower_rl.control import ArmController, ArmControllerCfg
from tower_rl.data.layouts import Layout
from tower_rl.graph.interface import ARMS, State

BLOCK_W = 0.0497
BLOCK_H = 0.0283


class FakeWorld:
    def __init__(self, variant: str, release_offset=None, grab_m: float = 0.012, payload_kg: float = 0.1056):
        lay = Layout.load(variant)
        self.variant = variant
        self.ids = list(lay.blocks)
        self.pos = np.stack([np.asarray(lay.blocks[b][:3], dtype=np.float64) for b in self.ids])
        self.quat = np.stack([np.asarray(lay.blocks[b][3:7], dtype=np.float64) for b in self.ids])
        self.fixed = np.array([bool(lay.fixed[b]) for b in self.ids])
        home = load_yaml("scene")["home_qpos"]
        self.q = np.array([home["left"][:7], home["back"][:7]], dtype=np.float64)
        self.width = np.array([0.1, 0.1])
        self.t, self.step = 0.0, 0
        self.ctrl = {a: ArmController(a, 1, "cpu", ArmControllerCfg()) for a in ARMS}
        self.held: dict = {a: None for a in ARMS}  # (block index, pos in pinch frame, quat in pinch frame)
        self.release_offset = release_offset  # callable(block id, n-th release) -> xy offset (m) added at release
        self.releases = 0
        self.grab_m = grab_m
        self.payload = payload_kg
        self.edit = None  # optional callable(world) applied after every row (scripted disturbances)

    def pinch(self, arm: str):
        q = torch.as_tensor(self.q[ARMS.index(arm)], dtype=torch.float32)[None]
        p, qq = self.ctrl[arm].pinch_world(q)
        return p[0].numpy().astype(np.float64), qq[0].numpy().astype(np.float64)

    def state(self) -> State:
        return State(t=round(self.t, 6), step=self.step, variant=self.variant, instruction="", q=self.q.astype(np.float32).copy(),
                     width=self.width.astype(np.float32).copy(), block_ids=list(self.ids), block_pos=self.pos.copy(),
                     block_quat=self.quat.copy(), block_conf=np.ones(len(self.ids)))

    def _support_z(self, i: int) -> float:
        """Rest height for block i dropped at its xy: on the highest block whose footprint overlaps, else the table."""
        z = BLOCK_H / 2
        for j in range(len(self.ids)):
            if j == i or any(h is not None and h[0] == j for h in self.held.values()):
                continue
            if np.linalg.norm(self.pos[j, :2] - self.pos[i, :2]) < 0.07 and self.pos[j, 2] < self.pos[i, 2]:
                z = max(z, self.pos[j, 2] + BLOCK_H)
        return z

    def apply(self, row: np.ndarray):
        row = np.asarray(row, dtype=np.float64)
        for k, arm in enumerate(ARMS):
            cmd_q, cmd_w = row[8 * k: 8 * k + 7], row[8 * k + 7]
            kin = self.ctrl[arm].kin
            pay = self.payload if self.held[arm] is not None else 0.0
            self.q[k] = kin.sag_equilibrium(torch.as_tensor(cmd_q, dtype=torch.float32)[None], pay)[0].numpy()
            pp, pq = self.pinch(arm)
            h = self.held[arm]
            if h is None and cmd_w < 0.06:
                d = np.linalg.norm(self.pos - pp, axis=-1)
                d[self.fixed] = np.inf
                for other in ARMS:
                    if self.held[other] is not None:
                        d[self.held[other][0]] = np.inf
                i = int(np.argmin(d))
                if d[i] <= self.grab_m:
                    rp, rq = geo.frame_relative(torch.as_tensor(pp)[None], torch.as_tensor(pq)[None],
                                                torch.as_tensor(self.pos[i])[None], torch.as_tensor(self.quat[i])[None])
                    self.held[arm] = (i, rp, rq)
                    h = self.held[arm]
            elif h is not None and cmd_w >= 0.06:
                i = h[0]
                self.held[arm] = None
                h = None
                if self.release_offset is not None:
                    self.pos[i, :2] += np.asarray(self.release_offset(self.ids[i], self.releases), dtype=np.float64)
                self.releases += 1
                self.pos[i, 2] = self._support_z(i)
            if h is not None:
                bp, bq = geo.frame_compose(torch.as_tensor(pp)[None], torch.as_tensor(pq)[None], h[1], h[2])
                self.pos[h[0]], self.quat[h[0]] = bp[0].numpy(), bq[0].numpy()
                self.width[k] = BLOCK_W
            else:
                self.width[k] = max(0.0, min(0.1, cmd_w))
        self.t += 0.1
        self.step += 1
        if self.edit is not None:
            self.edit(self)


def run_side_by_side(make_a, make_b, variant: str, seconds: float = 210.0, world_kw: dict | None = None,
                     edit=None, setup=None) -> dict:
    """Drive two policies on two identical fake worlds; stop at the first row or event difference. ``make_*`` return
    a runner ``Policy`` (reset + act); ``setup(policy)`` may patch it after reset. Returns the comparison."""
    pols, worlds, queues = [], [], []
    for mk in (make_a, make_b):
        p = mk()
        p.reset(1, variant, [0])
        if setup is not None:
            setup(p)
        w = FakeWorld(variant, **(world_kw or {}))
        w.edit = edit
        pols.append(p)
        worlds.append(w)
        queues.append([])
    rows_n = 0
    while worlds[0].t < seconds - 1e-6:
        for p, w, q in zip(pols, worlds, queues):
            if not q:
                q.extend(list(p.act([w.state()], [0])[0]))
        ra, rb = queues[0].pop(0), queues[1].pop(0)
        ev = [p.runs[0].events for p in pols]
        if not np.array_equal(ra, rb) or len(ev[0]) != len(ev[1]) or ev[0] != ev[1]:
            return {"equal": False, "t": worlds[0].t, "rows": rows_n, "row_a": ra, "row_b": rb, "events_a": ev[0], "events_b": ev[1]}
        for w, r in zip(worlds, (ra, rb)):
            w.apply(r)
        rows_n += 1
    return {"equal": True, "rows": rows_n, "events_a": pols[0].runs[0].events, "events_b": pols[1].runs[0].events,
            "worlds": worlds, "policies": pols}


# ---------------------------------------------------------------------- scripted scenarios (failure paths)
SCENARIOS = ("nominal", "misplace", "misplace_some", "armswitch", "armswitch_fail", "felloff", "nograsp", "late_drop")


def _own_arm_unreachable(p):
    """Patch a policy's run so a goal block's pre-grasp is reachable only by the other arm (arm-switch trigger)."""
    run = p.runs[0]
    own = {n["block"]: n["arm"] for n in run.g["nodes"] if n.get("block")}
    run._pregrasp_reachable = lambda arm, bid, s: arm != own[bid]


def scenario(name: str, variant: str) -> dict:
    """``run_side_by_side`` keyword arguments for a scripted scenario: misplacements (release offsets), an arm switch
    (near miss + own arm cannot reach the pre-grasp), the first goal's block falling off the table, grasps that never
    catch, and a held block dropped in transit."""
    from tower_rl.graph.graph import load_graph
    first = next(n["block"] for n in load_graph(variant)["nodes"] if n.get("block"))
    kw: dict = {}
    if name == "misplace":
        kw["world_kw"] = {"release_offset": lambda b, n: (0.008, 0.0)}
    elif name == "misplace_some":
        kw["world_kw"] = {"release_offset": lambda b, n: (0.009, 0.003) if n in (0, 3) else (0.0, 0.0)}
    elif name == "armswitch":
        kw["world_kw"] = {"release_offset": lambda b, n: (0.007, 0.0) if n == 0 else (0.0, 0.0)}
        kw["setup"] = _own_arm_unreachable
    elif name == "armswitch_fail":
        kw["world_kw"] = {"release_offset": lambda b, n: (0.007, 0.0) if n in (0, 1) else (0.0, 0.0)}
        kw["setup"] = _own_arm_unreachable
    elif name == "felloff":
        def edit(w):
            if abs(w.t - 7.5) < 0.05:
                i = w.ids.index(first)
                for a in w.held:
                    if w.held[a] is not None and w.held[a][0] == i:
                        w.held[a] = None
                w.pos[i] = [1.5, 0.5, -0.5]
        kw["edit"] = edit
    elif name == "nograsp":
        kw["world_kw"] = {"grab_m": 0.0}
    elif name == "late_drop":
        def edit(w):
            if abs(w.t - 14.0) < 0.05:
                for a in w.held:
                    if w.held[a] is not None:
                        i = w.held[a][0]
                        w.held[a] = None
                        w.pos[i, 2] = BLOCK_H / 2
        kw["edit"] = edit
    elif name != "nominal":
        raise ValueError(name)
    return kw
