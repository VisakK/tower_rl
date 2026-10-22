"""Deterministic graph executor (Step 7.4): interprets a variant's graph JSON on ``State`` and emits [H, 16] rows.

* Reads only deployable inputs: joint angles, gripper widths, time and block poses (simulator truth now, estimated
  poses later), plus its own command history. Goals are computed from the *observed* destination foundation blocks and
  the graph's in-pad goal offsets (from our layout config), never from organizer descriptors.
* Skills: ``transit``/``park`` are planned (``tower_rl.skills.transit``); ``grasp_lift``/``place_release`` use a
  learned policy when one is loaded for the arm, else a planned fallback with the same checkpoints (the build file's
  fallback for place_release; for grasp_lift only for pipeline tests, since the rules require the learned skill).
* Chunks: transit/park up to ``graph.chunk.transit`` rows (open loop), contact skills one row (closed loop).
* Perception noise (Step 8.1): targets computed once (pre-grasps, goals) use per-block rest-pose estimates, the place
  check a window of perceived poses, held blocks the kinematic prediction (``tower_rl.graph.estimates``); the skills'
  observations stay per call, as in training.
* Checkpoints are predicates on state (``tower_rl.graph.checkpoints``); edges: ok -> next, failure -> the node's
  ``on_fail`` (default: retry the transfer up to ``graph.retries`` times, then skip to the next transfer);
  ``arm_clear`` is enforced before every transit by parking the other arm if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl import push as push_geo
from tower_rl.config import load_yaml
from tower_rl.control import ArmController, ArmControllerCfg, GripperController
from tower_rl.graph import checkpoints as ck
from tower_rl.graph.estimates import CONF_MIN, PoseWindow, StaticPose, yaw_of
from tower_rl.graph.graph import load_graph
from tower_rl.graph.interface import ARMS, State, Status
from tower_rl.grasp import above
from tower_rl.grasp import cfg as grasp_cfg
from tower_rl.grasp import grasp_relation_error, gripper_cfg
from tower_rl.postures import prior_for_pose
from tower_rl.skills.obs import grasp_lift_actor_obs, grasp_target_world, place_release_actor_obs, push_actor_obs
from tower_rl.skills.task5_obs import insert_actor_obs, pad_actions
from tower_rl.skills.transit import TransitCfg, TransitPlan, arrived, plan_waypoints
from tower_rl.tower import rest_heights

ROW_DT = 0.1
MAX_PREGRASP_ADJUST_MM = 30.0   # pre-grasp via-points further out of reach fail at once (see TransitNode)
MAX_PREGRASP_ADJUST_DEG = 10.0
STATIC_DRIFT_M = 0.0025  # place check: a released block is at rest when its windowed median moves less in 0.5 s


@dataclass
class ExecCfg:
    device: str = "cpu"
    rate_hz: int = 10                     # challenge track: 10 (rows held 0.1 s) or 30 (rows held 1/30 s)
    grasp_skill_steps: int = 150
    held_width: tuple = (0.044, 0.056)   # measured opening when a block is in the closed gripper
    log: list = field(default_factory=list)


class SkillPolicy:
    """A frozen actor: observation -> action (deterministic mean), TorchScript or a callable."""

    def __init__(self, fn, device="cpu"):
        self.fn = fn
        self.device = device

    @classmethod
    def load(cls, path: str | Path, device="cpu"):
        return cls(torch.jit.load(str(path), map_location=device).eval(), device)

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fn(obs.to(self.device))


class Executor:
    def __init__(self, variant: str, graph: dict | None = None, skills: dict | None = None, cfg: ExecCfg | None = None):
        self.variant = variant
        self.graph = graph or load_graph(variant)
        self.skills = skills or {}  # (skill, arm) -> SkillPolicy
        self.cfg = cfg or ExecCfg()
        self.runs: list[EnvRun] = []

    @classmethod
    def from_bundle(cls, bundle: str | None, variant: str, device: str = "cpu", rate_hz: int = 10) -> Executor:
        skills = {}
        graph = None
        if bundle:
            b = Path(bundle)
            gp = b / "graphs" / f"{variant}.json"
            graph = load_graph(variant, gp) if gp.exists() else None
            for f in (b / "skills").glob("*.pt"):  # <skill>_<arm>.pt (TorchScript actor with normalizer)
                skill, arm = f.stem.rsplit("_", 1)
                skills[(skill, arm)] = SkillPolicy.load(f, device)
        return cls(variant, graph, skills, ExecCfg(device=device, rate_hz=rate_hz))

    def reset(self, n_envs: int, variant: str, seeds: list[int]):
        assert variant == self.variant
        self.runs = [EnvRun(self, s) for s in seeds[:n_envs]]

    def act(self, states: list[State], env_ids: list[int]) -> list[np.ndarray]:
        return [self.runs[e].act(s) for s, e in zip(states, env_ids)]

    def episode_log(self, env: int) -> list:
        """Node events ``(t, node id, status, info)`` of one env's episode (for rehearsal feedback)."""
        return list(self.runs[env].events)


# ---------------------------------------------------------------------- per-env interpreter
class EnvRun:
    def __init__(self, ex: Executor, seed: int):
        self.ex = ex
        self.g = ex.graph
        self.dev = ex.cfg.device
        self.seed = seed
        self.i = 0
        self.node = None
        self.tries = {}
        self.ctrl = {a: ArmController(a, 1, self.dev, ArmControllerCfg()) for a in ARMS}
        gcfg = gripper_cfg()
        if "gripper_closed_width" in self.g:  # Task 4/5 graphs: 3 N finger limit (0.040 m would saturate both fingers)
            gcfg.closed_width = float(self.g["gripper_closed_width"])
        self.grip = {a: GripperController(1, self.dev, gcfg) for a in ARMS}
        self.held_offset = {a: None for a in ARMS}  # held block pose in the pinch frame (pos, quat)
        self.attached = {a: None for a in ARMS}     # (block id, pose in the pinch frame) of a block moving with a gripper
        self.grasp_quat = {a: None for a in ARMS}      # feasible gripper orientations chosen by the transits
        self.release_target = {a: None for a in ARMS}  # release pinch pose chosen by the pre-place transit
        self.alt_queue: list[dict] = []  # inserted node specs (arm-switch re-place), run before the graph continues
        self.alt_done = set()            # transfers that already had their arm-switch attempt
        self.rest: dict[str, StaticPose] = {}  # per block: robust rest-pose estimate from every perceived pose
        self.fids = set(self.g["foundation"]["destination"]) | set(self.g["foundation"].get("source", []))
        self.slip = {a: [] for a in ARMS}       # held-back measurements of an attached block (refresh_attached)
        self.ready = False
        self.home = load_yaml("scene")["home_qpos"]
        self.rz = rest_heights(self.dev)
        self.events = []

    # ------------------------------------------------------------------ helpers
    def t(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    def q_meas(self, s: State, arm: str) -> torch.Tensor:
        return self.t(s.q[ARMS.index(arm)])[None]

    def pinch(self, s: State, arm: str):
        return self.ctrl[arm].pinch_world(self.q_meas(s, arm))

    def row(self) -> np.ndarray:
        """Current 16-D command (each arm: sag-compensated joints of its controller + gripper width)."""
        out = []
        for a in ARMS:
            out += [self._cmd[a].cpu().numpy().ravel(), self.grip[a].width.cpu().numpy().ravel()]
        return np.concatenate(out).astype(np.float32)

    def command(self, arm: str, q_meas=None, payload=0.0):
        self._cmd[arm] = self.ctrl[arm].command(q_meas, payload_kg=payload)[0]

    def block(self, s: State, bid: str):
        p, q = s.block(bid)
        return self.t(p)[None], self.t(q)[None]

    def observe(self, s: State):
        """Feed every perceived block pose to that block's rest-pose estimate (once per call, before any node runs;
        foundations never move and get a longer window)."""
        for i, b in enumerate(s.block_ids):
            est = self.rest.get(b)
            if est is None:
                est = self.rest[b] = StaticPose(window=30 if b in self.fids else 10)
            est.add(s.block_pos[i], s.block_quat[i], float(s.block_conf[i]))

    def block_rest(self, s: State, bid: str):
        """Robust pose of a block at rest (median of its recent consistent perceived poses: immune to jitter, one-call
        outliers and dropouts); for targets computed once (pre-grasp, goals). A block that moves restarts its estimate
        after two calls."""
        est = self.rest.get(bid)
        if est is None or est.pose() is None:
            return self.block(s, bid)
        p, q = est.pose()
        return self.t(p)[None], self.t(q)[None]

    def block_est(self, s: State, bid: str):
        """Pose estimate of block ``bid``: while it is attached to a gripper (held, from before the lift until the
        release) the kinematic prediction pinch * offset, else perception. A held block moves rigidly with the gripper
        while its perceived pose is the least reliable one (occluded by the fingers); in rehearsal, perceived poses that
        freeze for 0.5 s took Task 2-3 D1 from TSR 1.0 to 0.375 through stale held-block poses."""
        for a in ARMS:
            att = self.attached[a]
            if att is not None and att[0] == bid:
                pp, pq = self.pinch(s, a)
                return geo.frame_compose(pp, pq, *att[1])
        return self.block(s, bid)

    def attach(self, s: State, arm: str, bid: str, rest: bool = True):
        """Fix the block's pose in the pinch frame (call once the fingers are on it, before it moves): from its rest
        estimate (``block_rest``), or with ``rest=False`` from the current perceived pose if that is confident."""
        if not rest and float(s.block_conf[s.block_ids.index(bid)]) < CONF_MIN:
            return
        pp, pq = self.pinch(s, arm)
        bp, bq = self.block_rest(s, bid) if rest else self.block(s, bid)
        self.attached[arm] = (bid, geo.frame_relative(pp, pq, bp, bq))
        self.slip[arm] = []

    def detach(self, arm: str):
        self.attached[arm] = None
        self.slip[arm] = []

    def refresh_attached(self, s: State, arm: str, gain: float = 0.3, gate_m: float = 0.005, confirm: int = 3):
        """Track a block shifting in the fingers: move the attached offset toward each fresh measurement (confidence
        >= 0.5, i.e. not a dropout's stale pose) within ``gate_m`` of the prediction. A measurement beyond the gate is
        held back (a one-call outlier); ``confirm`` consecutive ones that agree with each other are a slip and replace
        the offset."""
        att = self.attached[arm]
        if att is None:
            return
        bid, (op, oq) = att
        if float(s.block_conf[s.block_ids.index(bid)]) < CONF_MIN:
            return
        pp, pq = self.pinch(s, arm)
        mp, mq = geo.frame_relative(pp, pq, *self.block(s, bid))
        mq = _align_quat(mq, oq)
        if float((mp - op).norm()) <= gate_m:
            self.slip[arm] = []
            self.attached[arm] = (bid, (op + gain * (mp - op), geo.normalize(oq + gain * (mq - oq))))
            return
        sl = self.slip[arm]
        if sl and float((mp - sl[-1][0]).norm()) > gate_m:
            sl.clear()
        sl.append((mp, mq))
        if len(sl) >= confirm:
            self.attached[arm] = (bid, (torch.stack([a for a, _ in sl]).mean(0),
                                        geo.normalize(torch.stack([_align_quat(b, sl[-1][1]) for _, b in sl]).mean(0))))
            sl.clear()

    def dest_goal(self, s: State, node: dict, raw: bool = False):
        """Goal pose of a transfer from the observed destination foundation and the in-pad offset. The foundation poses
        are their rest estimates (``block_rest``; ``raw``: the poses in ``s``, e.g. simulator truth for diagnostics)."""
        fids = self.g["foundation"]["destination"]
        poses = [self.block(s, b) if raw else self.block_rest(s, b) for b in fids]
        pts = torch.stack([p[0] for p, _ in poses])
        center = pts.mean(0)
        nominal_yaw = math.radians(self.g["foundation"]["destination_yaw_deg"])
        # the blocks' perceived yaws relative to the pad (circular mean, half-turn symmetric): the goal *orientation*
        # comes from these, i.e. from the same perception as the held block's yaw, so an orientation error common to
        # the blocks cancels in the place and in the place check
        rel = [(yaw_of(q[0].cpu().numpy()) - nominal_yaw + math.pi / 2) % math.pi - math.pi / 2 for _, q in poses]
        orient = 0.5 * math.atan2(sum(math.sin(2 * a) for a in rel), sum(math.cos(2 * a) for a in rel))
        # the pad yaw that places the in-pad offsets comes from the three foundation centres (the line block 0 ->
        # block 2 points at pad yaw + 90 deg in every layout, 100 mm long): immune to orientation errors (a shared 3 deg
        # yaw bias otherwise moved goals 50 mm off the pad centre by ~2.6 mm); the blocks' yaw is the fallback
        d = pts[2, :2] - pts[0, :2]
        line = (math.atan2(float(d[1]), float(d[0])) - math.pi / 2 - nominal_yaw + math.pi) % (2 * math.pi) - math.pi
        if float(d.norm()) < 0.05 or abs(line) > math.radians(15):
            line = orient
        if abs(orient - line) > math.radians(15):  # implausible perceived orientations: trust the centres
            orient = line
        gi = node["goal_in_pad"]
        c, sn = math.cos(nominal_yaw + line), math.sin(nominal_yaw + line)
        x, y = gi["pos"][0], gi["pos"][1]
        z = torch.tensor(node["goal_z"], device=self.dev) if "goal_z" in node else self.rz[node["dest_layer"]]
        pos = torch.stack([center[0] + c * x - sn * y, center[1] + sn * x + c * y, z])
        q = geo.quat_from_yaw(torch.tensor(nominal_yaw + orient + math.radians(gi["yaw_deg"]), device=self.dev))
        return pos[None], q[None]

    def slot_pose(self, s: State, node: dict):
        """Task 5 slot: centre = midpoint of the two side blocks of the goal layer (rest estimates), axis = their long
        axis, height = the layout's goal height; ``(pos [1,3], quat [1,4], direction [1,3])`` with the insertion
        direction (graph) flipped onto the side blocks' axis. Falls back to the destination-foundation goal."""
        sides = node.get("side_blocks") or []
        d = torch.tensor([node["direction"]], dtype=torch.float32, device=self.dev)
        if len(sides) != 2:
            gp, gq = self.dest_goal(s, node)
            return gp, gq, d
        ps = [self.block_rest(s, b) for b in sides]
        c = (ps[0][0] + ps[1][0]) / 2
        c[0, 2] = node["goal_z"]
        yaws = [float(geo.yaw_from_quat(q)[0]) for _, q in ps]
        ya = 0.5 * math.atan2(sum(math.sin(2 * y) for y in yaws), sum(math.cos(2 * y) for y in yaws))
        ax = torch.tensor([[math.cos(ya), math.sin(ya), 0.0]], device=self.dev)
        if float((ax * d).sum()) < 0:
            ax = -ax
        return c, geo.quat_from_yaw(torch.tensor([ya], device=self.dev)), ax

    # ------------------------------------------------------------------ main loop
    def act(self, s: State) -> np.ndarray:
        if not self.ready:
            self._cmd = {}
            for a in ARMS:
                qm = self.q_meas(s, a)
                self.ctrl[a].reset(qm)
                self.command(a)
                self.grip[a].reset(width=float(s.width[ARMS.index(a)]) if s.width[ARMS.index(a)] > 0.09 else 0.1)
            self.ready = True
        self.observe(s)
        rows: list[np.ndarray] = []
        for _ in range(64):  # nodes that finish without emitting rows must not stall the loop
            if self.i >= len(self.g["nodes"]):
                return self.row()[None]
            if self.node is None:
                if self.alt_queue:
                    self.node = make_node(self, self.alt_queue[0], s)
                    self.node.alt = True
                else:
                    spec = self.g["nodes"][self.i]
                    self.node = self._clearance_park(spec, s) or make_node(self, spec, s)
            status, out = self.node.step(s)
            if out and self.ex.cfg.rate_hz == 30 and isinstance(self.node, (PlannedGraspNode, LearnedGraspNode, PlannedPlaceNode, LearnedPlaceNode, PushNode, InsertNode, PushInNode)):
                out = self._expand(out)  # 10 Hz decisions -> 3 interpolated 30 Hz rows each
            if out:
                self._last_row = out[-1]
            rows += out
            if status is not None:
                self._advance(status, s)
            if rows:
                break
        return np.stack(rows) if rows else self.row()[None]

    def _expand(self, out: list[np.ndarray]) -> list[np.ndarray]:
        prev = getattr(self, "_last_row", None)
        res = []
        for r in out:
            p = r if prev is None else prev
            res += [p + (r - p) * (k / 3.0) for k in (1, 2, 3)]
            prev = r
        return res

    def _clearance_park(self, spec: dict, s: State):
        """``arm_clear`` checkpoint: before an arm's transit to a tower, the other arm's pinch must be outside that
        tower's cylinder; otherwise park the other arm first (an inserted node: the graph pointer stays). The graphs
        park before every arm switch, so this only fires after an unforeseen failure path."""
        if spec["skill"] != "transit":
            return None
        other = next(a for a in ARMS if a != spec["arm"])
        on_block = spec["target"] in ("pregrasp", "prepush", "prepushin")
        if spec["target"] == "preinsert":
            txy = self.slot_pose(s, spec)[0][0, :2].cpu().numpy()
        else:
            txy = (self.block_rest(s, spec["block"]) if on_block else self.dest_goal(s, spec))[0][0, :2].cpu().numpy()
        if ck.arm_clear(self.pinch(s, other)[0][0].cpu().numpy(), txy):
            return None
        node = ParkNode(self, {"id": f"auto_park_{other}", "skill": "park", "arm": other, "transfer": spec["transfer"],
                               "timeout_s": 10.0}, s)
        node.inserted = True
        return node

    def _schedule_arm_switch(self, node: dict, info: dict, s: State) -> bool:
        """Arm-switch re-place: a near miss (released within ``arm_switch_m`` of its goal) that its own arm cannot
        re-grasp (pre-grasp out of reach, e.g. the left arm at pad B's upper layers) is re-grasped and re-placed by the
        other arm when that arm reaches the block. Queues park / transit / grasp / transit / place / park nodes copied
        from the transfer with the arm swapped; at most once per transfer."""
        k, arm = node["transfer"], node["arm"]
        err = info.get("place_xy_mm")
        if k in self.alt_done or err is None or err / 1000.0 > self.g.get("arm_switch_m", 0.015):
            return False
        other = next(a for a in ARMS if a != arm)
        if self._pregrasp_reachable(arm, node["block"], s) or not self._pregrasp_reachable(other, node["block"], s):
            return False
        nodes = [n for n in self.g["nodes"] if n["transfer"] == k and n["skill"] != "park"]
        alt = [{"id": f"t{k}_alt_park_{arm}", "skill": "park", "arm": arm, "transfer": k, "timeout_s": 10.0}]
        for n in nodes:
            alt.append(dict(n, id=n["id"] + "_alt", arm=other, first_of_transfer=False))
        alt.append({"id": f"t{k}_alt_park_{other}", "skill": "park", "arm": other, "transfer": k, "timeout_s": 10.0})
        self.alt_done.add(k)
        self.alt_queue = alt
        self.grip[arm].reset(closed=False, width=float(self.grip[arm].width))
        self.detach(arm)
        return True

    def _pregrasp_reachable(self, arm: str, bid: str, s: State) -> bool:
        """Whether ``arm``'s pre-grasp via-point for ``bid`` (as its transit would compute it) is within reach."""
        ctrl = self.ctrl[arm]
        bp, bq = self.block_rest(s, bid)
        gp, gq = grasp_target_world(bp, bq, arm)
        seed = prior_for_pose(arm, "grasp", bp, bq)
        gq = feasible_orientation(ctrl, gp, gq, seed)
        _, adj = reachable_pose(ctrl, *above(gp, gq, grasp_cfg()["pregrasp_height_m"]), seed)
        return not adj or (adj["goal_adjust_mm"] <= MAX_PREGRASP_ADJUST_MM and adj["goal_adjust_deg"] <= MAX_PREGRASP_ADJUST_DEG)

    def _advance(self, status: Status, s: State):
        """Edges: ok -> next node; failure -> the node's ``on_fail`` policy:
        ``retry`` (default: up to ``graph.retries`` times, then skip), ``skip`` (next transfer), ``skip_if_close`` (skip
        when the block rests within ``skip_if_close_m`` of its goal; otherwise retry). A retry with the block still in
        the gripper repeats the transport from the pre-place transit; any other retry opens the gripper and restarts
        the transfer. A skip opens the gripper and continues at the next node of a later transfer (possibly the park
        that clears this arm). A park that fails just continues; a transfer whose block fell off the table is skipped."""
        if getattr(self.node, "inserted", False):  # an inserted node (arm_clear park): the graph pointer stays
            self.events.append((round(s.t, 2), self.node.spec["id"], status.value, self.node.info))
            self.node = None
            return
        if getattr(self.node, "alt", False):  # arm-switch re-place: next inserted node, then after the transfer
            spec = self.alt_queue.pop(0)
            self.events.append((round(s.t, 2), spec["id"], status.value, self.node.info))
            self.node = None
            if status != Status.OK and spec["skill"] != "park":
                self.grip[spec["arm"]].reset(closed=False, width=float(self.grip[spec["arm"]].width))
                self.detach(spec["arm"])
                self.alt_queue = [n for n in self.alt_queue if n["skill"] == "park"]  # give up, but park the arm
            if not self.alt_queue:
                k = self.g["nodes"][self.i]["transfer"]
                nxt = next((j for j, n in enumerate(self.g["nodes"]) if j > self.i and n["transfer"] > k), None)
                self.i = nxt if nxt is not None else len(self.g["nodes"])
            return
        node = self.g["nodes"][self.i]
        info = self.node.info if self.node else {}
        self.events.append((round(s.t, 2), node["id"], status.value, info))
        self.node = None
        if status == Status.OK:
            self.i += 1
            return
        if node["skill"] == "park":  # a park that timed out still leaves the arm where it got to: continue
            self.i += 1
            return
        k = node["transfer"]
        policy = node.get("on_fail", "retry")
        if policy == "skip_if_close":
            err = info.get("place_xy_mm")
            policy = "skip" if err is not None and err / 1000.0 <= node.get("skip_if_close_m", 0.015) else "retry"
        if policy == "retry" and node.get("block") and float(s.block(node["block"])[0][2]) < -0.02:
            policy = "skip"  # the block fell off the table: a re-grasp cannot reach it
        if node["skill"] == "place_release" and self._schedule_arm_switch(node, info, s):
            return
        self.tries[k] = self.tries.get(k, 0) + 1
        nodes = self.g["nodes"]
        arm = node["arm"]
        lo, hi = self.ex.cfg.held_width
        holding = bool(self.grip[arm].closed) and lo <= float(s.width[ARMS.index(arm)]) <= hi
        if policy == "skip" or self.tries[k] > self.g["retries"]:
            # skip: open this arm's gripper (release whatever it holds) and continue after this transfer's nodes; the
            # next node may be the park that clears this arm for the other one (it belongs to the next transfer but
            # precedes its first node), so jump to the next node of a later transfer, not to that transfer's first node
            self.grip[arm].reset(closed=False, width=float(self.grip[arm].width))
            self.detach(arm)
            nxt = next((j for j, n in enumerate(nodes) if j > self.i and n["transfer"] > k), None)
            self.i = nxt if nxt is not None else len(nodes)
        elif holding and node["skill"] in ("transit", "place_release", "insert") and node.get("target") in (None, "preplace", "preinsert"):
            # the transport failed with the block still in the gripper: retry from the pre-place / pre-insert transit
            self.i = next(j for j, n in enumerate(nodes) if n["transfer"] == k and n.get("target") in ("preplace", "preinsert"))
        else:  # restart the transfer from its pre-grasp transit with an open gripper
            self.grip[arm].reset(closed=False, width=float(self.grip[arm].width))
            self.detach(arm)
            self.i = next(j for j, n in enumerate(nodes) if n["transfer"] == k and n.get("first_of_transfer"))


def _align_quat(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``q`` as the representative of the same block orientation (sign; half turn about the block's z axis, under which
    a block is symmetric) closest to ``ref``."""
    fl = torch.stack([-q[..., 3], q[..., 2], -q[..., 1], q[..., 0]], -1)  # q * Rz(pi)
    q = torch.where((fl * ref).sum(-1, keepdim=True).abs() > (q * ref).sum(-1, keepdim=True).abs(), fl, q)
    return torch.where((q * ref).sum(-1, keepdim=True) < 0, -q, q)


def make_node(run: EnvRun, spec: dict, s: State):
    sk = spec["skill"]
    if sk == "transit":
        return TransitNode(run, spec, s)
    if sk == "park":
        return ParkNode(run, spec, s)
    if sk == "grasp_lift":
        # Task 5 graphs: planned grasp (``grasp_planned``; rehearse --graph_set '{"grasp_planned": false}' for the learned
        # one): the Tasks 1-3 grasp policy leaves the block 2-3 deg rotated in the fingers and the planned insert, which
        # needs < 0.3 deg pitch / 0.4 deg yaw at the slot entrance, then timed out 7 / 8 (Task 5-2 D0)
        planned = spec.get("planned", run.g.get("grasp_planned", False))
        pol = None if planned else run.ex.skills.get(("grasp_lift", spec["arm"]))
        return LearnedGraspNode(run, spec, s, pol) if pol is not None else PlannedGraspNode(run, spec, s)
    if sk == "place_release":
        pol = run.ex.skills.get(("place_release", spec["arm"]))
        return LearnedPlaceNode(run, spec, s, pol) if pol is not None else PlannedPlaceNode(run, spec, s)
    if sk == "push":
        pol = run.ex.skills.get(("push", spec["arm"]))
        return PushNode(run, spec, s, pol)
    if sk == "insert":
        return InsertNode(run, spec, s, run.ex.skills.get(("insert", spec["arm"])))
    if sk == "push_in":
        return PushInNode(run, spec, s, run.ex.skills.get(("push_in", spec["arm"])))
    raise ValueError(sk)


class NodeBase:
    def __init__(self, run: EnvRun, spec: dict, s: State):
        self.run, self.spec, self.arm = run, spec, spec["arm"]
        self.t0 = s.t
        self.info = {}

    def timed_out(self, s: State) -> bool:
        return s.t - self.t0 > self.spec["timeout_s"]


class TransitNode(NodeBase):
    """Plan once from the controller's current target to the pre-grasp / pre-place pose, then emit the rows."""

    def __init__(self, run, spec, s):
        super().__init__(run, spec, s)
        r, arm = run, self.arm
        ctrl = r.ctrl[arm]
        held = bool(spec.get("held"))
        self.payload = grasp_cfg_payload() if held else 0.0
        if spec["target"] == "pregrasp":
            bp, bq = r.block_rest(s, spec["block"])
            gp, gq = grasp_target_world(bp, bq, arm)
            ctrl.q_ref[:] = prior_for_pose(arm, "grasp", bp, bq)
            gq = feasible_orientation(ctrl, gp, gq, ctrl.q_ref)
            r.grasp_quat[arm] = gq
            self.goal = above(gp, gq, spec["height_m"])
        elif spec["target"] == "prepush":  # Task 4: fingers closed into the pusher tool, pinch behind the end face
            bp, _ = r.block_rest(s, spec["block"])
            d = torch.tensor([spec["direction"]], dtype=torch.float32, device=r.dev)
            ctrl.q_ref[:] = torch.tensor(push_geo.prior(arm, spec["layer"]), device=r.dev)[None]
            self.goal = push_geo.prepush_pose(bp, d, arm, spec["layer"])
            r.grip[arm].reset(width=push_geo.cfg()["tool_width_m"], closed=True)
        elif spec["target"] == "preinsert":  # Task 5: the held block on the slot axis, outside_m out, at slot height
            sp, sq, d = r.slot_pose(s, spec)
            ctrl.q_ref[:] = prior_for_pose(arm, "release", sp, sq)
            self.goal = place_pinch(r, arm, sp - d * spec["outside_m"], sq)
        elif spec["target"] == "prepushin":  # Task 5: pusher (fingers at the pusher width) behind the block's outer face
            bp, _ = r.block_rest(s, spec["block"])
            _, sq, d = r.slot_pose(s, spec)
            ig = load_yaml("skills/insert")["geometry"]
            ctrl.q_ref[:] = prior_for_pose(arm, "release", bp, sq)
            gp, gq = grasp_target_world(bp, sq, arm)  # the grasp orientation family (demo: same tilt as the grasp)
            p = bp + d * ig["prepushin_along_m"]
            p[0, 2] = bp[0, 2] + ig["prepushin_up_m"]
            self.goal = (p, gq)
            r.grip[arm].reset(width=ig["pusher_width_m"], closed=True)
        else:
            gp, gq = r.dest_goal(s, spec)
            place = next((n for n in r.g["nodes"] if n["transfer"] == spec["transfer"] and n["skill"] == "place_release"), {})
            pp, pq = release_pose(r, arm, gp, gq, place.get("release_clearance_m", grasp_cfg()["release_clearance_m"]))
            r.release_target[arm] = (pp, pq)
            self.goal = above(pp, pq, spec["height_m"])
        self.goal, adj = reachable_pose(ctrl, *self.goal, ctrl.q_ref)
        if adj:
            self.info_adjust = adj
        # a pre-grasp far out of reach (e.g. re-grasping a near-miss block from the destination stack at the arm's
        # workspace edge) would start the grasp skill far outside its training distribution next to the stack: give up
        self.unreachable = spec["target"] == "pregrasp" and bool(adj) and (
            adj["goal_adjust_mm"] > MAX_PREGRASP_ADJUST_MM or adj["goal_adjust_deg"] > MAX_PREGRASP_ADJUST_DEG)
        cfg = TransitCfg(tol_pos_m=spec["tol_pos_m"], tol_rot_deg=spec["tol_rot_deg"], dt=1.0 / r.ex.cfg.rate_hz,
                         **{k: spec[k] for k in ("v_max", "v_max_held", "a_max") if k in spec})  # graph-tunable speeds
        ps, qs = plan_waypoints(ctrl.pos[0], ctrl.quat[0], self.goal[0][0], self.goal[1][0], held, cfg)
        # contact monitor: blocks this transit must not move (all movable blocks except the held one), on rest estimates
        self.watch = [b for b in r.g["movable"] if not (held and b == spec["block"])]
        self.p_watch0 = self._watched(s) if self.watch else None
        self.plan = TransitPlan(ps, qs, held, cfg)
        self.cfg = cfg
        self.wait = 0

    def _watched(self, s: State) -> np.ndarray:
        return np.stack([self.run.block_rest(s, b)[0][0].cpu().numpy() for b in self.watch])

    def step(self, s: State):
        r, arm = self.run, self.arm
        if self.timed_out(s):
            return Status.TIMEOUT, []
        if self.unreachable:
            self.info = {"why": "pregrasp unreachable", **self.info_adjust}
            return Status.FAILED, []
        if self.plan.held and float(s.width[ARMS.index(arm)]) < r.ex.cfg.held_width[0]:
            # the fingers closed past the block width: the block slipped out (heavy / slippery blocks); fail now
            # instead of carrying an empty gripper to the place
            self.info = {"why": "dropped"}
            return Status.FAILED, []
        if self.plan.held:
            r.refresh_attached(s, arm)
        if not self.plan.path_done:
            h = min(r.g["chunk"]["transit"], len(self.plan.ps) - self.plan.i)
            ps, qs = self.plan.next_targets(h)
            rows = []
            for k in range(h):
                r.ctrl[arm].set_target(ps[k:k + 1], qs[k:k + 1])
                r.command(arm, payload=self.payload)
                rows.append(r.row())
            return None, rows
        pp, pq = r.pinch(s, arm)
        if bool(arrived(pp, pq, self.goal[0], self.goal[1], self.cfg)[0]) and self.wait >= self.plan.settle_rows:
            disp = float(np.linalg.norm(self._watched(s) - self.p_watch0, axis=-1).max()) if self.watch else 0.0
            self.info = {"pos_err_mm": round(float((pp - self.goal[0]).norm()) * 1000, 1),
                         "rot_err_deg": round(math.degrees(float(geo.quat_angle(pq, self.goal[1])[0])), 2),
                         "max_block_disp_mm": round(disp * 1000, 2), "rows": len(self.plan.ps),
                         **getattr(self, "info_adjust", {})}
            return Status.OK, []
        self.wait += 1
        r.command(arm, payload=self.payload)
        return None, [r.row()]


def grasp_cfg_payload() -> float:
    return ArmControllerCfg().payload_mass


def feasible_orientation(ctrl: ArmController, pos: torch.Tensor, quat: torch.Tensor, seed: torch.Tensor,
                         clear_rad: float = 0.06) -> torch.Tensor:
    """Keep the canonical gripper orientation if the arm reaches it with every joint >= ``clear_rad`` inside its limits;
    otherwise let IK choose the tilt about the jaw axis (near the workspace edge, e.g. the left arm on pad B, the wrist
    pitch saturates at the canonical 12 deg tilt) and return the achieved orientation."""
    kin, base = ctrl.kin, ctrl.base
    pb, qb = base.to_base(pos, quat)
    lim = kin.limits
    q, pe, re = kin.ik(seed, pb, qb, iters=80, q_ref=seed)
    margin = torch.minimum(q - lim[:, 0], lim[:, 1] - q).min()
    if float(pe.max()) < 5e-4 and float(re.max()) < math.radians(0.3) and float(margin) >= clear_rad:
        return quat
    jaw_b = geo.quat_to_mat(qb)[:, :, 1]
    q2, pe2, _ = kin.ik(seed, pb, qb, iters=120, q_ref=seed, free_axis=jaw_b, free_weight=0.0, limit_margin=clear_rad)
    _, qa_b = kin.fk(q2, "pinch")
    return base.to_world(pb, qa_b)[1]


def reachable_pose(ctrl: ArmController, pos: torch.Tensor, quat: torch.Tensor, seed: torch.Tensor,
                   tol_pos_m: float = 1e-3, tol_rot_deg: float = 0.5):
    """The via-point itself if IK reaches it within the controller's joint-limit margin, else the closest reachable
    pose (FK of the IK solution). A via-point a joint limit makes unreachable (e.g. a pre-place pose for a block held
    off-centre and rotated, where the wrist j7 saturates) would otherwise never satisfy the transit's arrival check;
    the contact skill that follows corrects the remaining error on its way to the (reachable) contact pose.
    Returns ``((pos, quat), info)`` with ``info`` = the adjustment, empty when none was needed."""
    kin, base = ctrl.kin, ctrl.base
    pb, qb = base.to_base(pos, quat)
    q, pe, re = kin.ik(seed, pb, qb, iters=150, q_ref=seed, limit_margin=ctrl.cfg.limit_margin)
    if float(pe.max()) <= tol_pos_m and math.degrees(float(re.max())) <= tol_rot_deg:
        return (pos, quat), {}
    pb2, qb2 = kin.fk(q, "pinch")
    p2, q2 = base.to_world(pb2, qb2)
    return (p2, q2), {"goal_adjust_mm": round(float((p2 - pos).norm()) * 1000, 1),
                      "goal_adjust_deg": round(math.degrees(float(geo.quat_angle(q2, quat)[0])), 2)}


def place_pinch(run: EnvRun, arm: str, goal_pos, goal_quat):
    """Pinch pose that puts the held block at the goal, using the block-in-pinch offset measured after the grasp."""
    gp, gq = grasp_target_world(goal_pos, goal_quat, arm)
    off = run.held_offset[arm]
    if off is None:
        return gp, gq
    # pinch = goal_block * offset^-1 where offset = block pose in the pinch frame; keep the canonical gripper yaw family
    op, oq = off
    ip, iq = geo.frame_inverse(op, oq)
    # block goal with the yaw (pi-symmetric) closest to the held block's current yaw in the pinch frame
    pq_c = geo.quat_mul(gq, oq)  # where the block would point with the canonical pinch
    flip = geo.quat_angle(pq_c, goal_quat) > math.pi / 2
    goal_q = torch.where(flip[:, None], geo.quat_mul(goal_quat, geo.quat_from_yaw(torch.tensor([math.pi], device=gq.device))), goal_quat)
    return geo.frame_compose(goal_pos, goal_q, ip, iq)


def release_pose(run: EnvRun, arm: str, goal_pos, goal_quat, clearance: float):
    """Pinch pose that holds the block's centre ``clearance`` above its goal (block flat at the goal yaw when the arm can
    reach it). Starts from the exact pose for the measured block-in-pinch offset; if that saturates a joint, IK picks
    the tilt about the jaw axis and the pinch *position* is re-solved so the held block's centre stays on the goal
    (the block is then released slightly tilted about its width axis and lands flat)."""
    ctrl = run.ctrl[arm]
    up = torch.tensor([0.0, 0.0, clearance], device=goal_pos.device)
    pp, pq = place_pinch(run, arm, goal_pos, goal_quat)
    pp = pp + up
    ctrl.q_ref[:] = prior_for_pose(arm, "release", goal_pos, goal_quat)
    fq = feasible_orientation(ctrl, pp, pq, ctrl.q_ref)
    off = run.held_offset[arm]
    if off is not None and float(geo.quat_angle(fq, pq)[0]) > 1e-4:
        pp = goal_pos + up - geo.quat_apply(fq, off[0])  # keep the block centre on the goal
    return pp, fq


class ParkNode(NodeBase):
    """Joint-space move to the home posture (gripper open), ~3 s, then idle."""

    def __init__(self, run, spec, s):
        super().__init__(run, spec, s)
        ctrl = run.ctrl[self.arm]
        self.q0 = ctrl.q_des.clone()
        self.q1 = torch.tensor(run.home[self.arm][:7], device=run.dev)[None]
        dq = float((self.q1 - self.q0).abs().max())
        self.n = max(5, int(math.ceil(dq / (0.4 / run.ex.cfg.rate_hz))))  # <= 0.4 rad/s
        self.k = 0
        run.grip[self.arm].reset(closed=False, width=float(run.grip[self.arm].width))
        run.detach(self.arm)

    def step(self, s):
        r = self.run
        if self.k >= self.n:
            return Status.OK, []
        rows = []
        for _ in range(min(r.g["chunk"]["transit"], self.n - self.k)):
            self.k += 1
            a = 0.5 - 0.5 * math.cos(math.pi * self.k / self.n)
            ctrl = r.ctrl[self.arm]
            ctrl.q_des[:] = self.q0 + a * (self.q1 - self.q0)
            p, q = ctrl.pinch_world(ctrl.q_des)
            ctrl.set_target(p, q)
            r._cmd[self.arm] = ctrl.kin.sag_compensated_command(ctrl.q_des)[0]
            r.grip[self.arm].step(torch.zeros(1, dtype=torch.bool, device=r.dev))
            rows.append(r.row())
        return None, rows


class PlannedGraspNode(NodeBase):
    """Scripted grasp (pipeline tests only): descend onto the canonical grasp pose, close, lift slowly."""

    def __init__(self, run, spec, s):
        super().__init__(run, spec, s)
        self.phase = 0
        self.k = 0
        self.vz = 0.0
        bp, bq = run.block_rest(s, spec["block"])
        self.z0 = float(bp[0, 2])

    def step(self, s):
        r, arm = self.run, self.arm
        if self.timed_out(s):
            return Status.TIMEOUT, []
        ctrl = r.ctrl[arm]
        bp, bq = r.block_est(s, self.spec["block"])
        gp, gq = grasp_target_world(bp, bq, arm)
        if r.grasp_quat[arm] is not None:
            gq = r.grasp_quat[arm]
        w = float(s.width[ARMS.index(arm)])
        if self.phase == 2:
            r.refresh_attached(s, arm)
        if self.phase == 0:  # descend 6 mm per row
            d = gp - ctrl.pos
            step = d * min(1.0, 0.006 / max(1e-6, float(d.norm())))
            ctrl.set_target(ctrl.pos + step, gq)
            pp, _ = r.pinch(s, arm)
            if float((pp - gp).norm()) < 0.004 and float(d.norm()) < 0.002:
                self.phase, self.k = 1, 0
        elif self.phase == 1:  # close
            self.k += 1
            if self.k > 12:
                self.phase, self.k = 2, 0
                if ck.gripping(s, arm):
                    r.attach(s, arm, self.spec["block"])  # fingers on the block, before it moves
        else:  # lift 1 cm per row, or (Task 5 graphs) ramp up to lift_v_m_s: the 3 N fingers let a heavy block slip
            tgt = ctrl.pos.clone()  # out of a 1 cm step (D1 heavy: "empty gripper" 9 / 16); the demos lift <= 5 cm/s
            if "lift_v_m_s" in self.spec:
                self.vz = min(self.spec["lift_v_m_s"], self.vz + self.spec["lift_a_m_s2"] * ROW_DT)
            tgt[0, 2] = min(float(ctrl.pos[0, 2]) + (self.vz * ROW_DT if "lift_v_m_s" in self.spec else 0.01),
                            self.z0 + self.spec["lift_m"] + 0.02)
            ctrl.set_target(tgt, ctrl.quat)
        closed = self.phase >= 1
        r.grip[arm].step(torch.tensor([closed], device=r.dev))
        r.command(arm, r.q_meas(s, arm), payload=grasp_cfg_payload() if self.phase == 2 else 0.0)
        if self.phase == 2:
            if ck.grasped(s, arm, self.spec["block"], self.z0, self.spec["lift_m"], r.ex.cfg.held_width):
                self.info = {"lift_mm": round((float(bp[0, 2]) - self.z0) * 1000, 1),
                             **record_held_offset(r, s, arm, self.spec["block"])}
                return Status.OK, [r.row()]
            if w < r.ex.cfg.held_width[0]:
                self.info = {"why": "empty gripper"}
                return Status.FAILED, [r.row()]
        return None, [r.row()]


class LearnedGraspNode(NodeBase):
    """The trained grasp_lift actor, fed the same observation as in training (``tower_rl.skills.obs``)."""

    def __init__(self, run, spec, s, policy: SkillPolicy):
        super().__init__(run, spec, s)
        self.pol = policy
        sk = load_yaml("skills/grasp_lift")
        self.ac = sk["action"]
        self.ema = torch.zeros(1, 6, device=run.dev)
        self.prev_a = torch.zeros(1, 7, device=run.dev)
        ctrl = run.ctrl[self.arm]
        self.prev_pb, self.prev_qb = ctrl.kin.fk(run.q_meas(s, self.arm), "pinch")
        bp, bq = run.block_rest(s, spec["block"])
        self.z0 = float(bp[0, 2])
        self.steps = 0
        self.ever_lifted = False
        self.max_steps = run.ex.cfg.grasp_skill_steps

    def _others(self, s: State):
        """All movable blocks other than the target (foundations excluded), base frame."""
        r = self.run
        ids = [b for b in r.g["movable"] if b != self.spec["block"]]
        ps = torch.stack([r.block(s, b)[0][0] for b in ids])[None]
        qs = torch.stack([r.block(s, b)[1][0] for b in ids])[None]
        return ids, ps, qs

    def step(self, s):
        r, arm = self.run, self.arm
        ctrl = r.ctrl[arm]
        base = ctrl.base
        if self.timed_out(s) or self.steps >= self.max_steps:
            return Status.TIMEOUT, []
        qm = r.q_meas(s, arm)
        pb, qb = ctrl.kin.fk(qm, "pinch")
        holding = bool(r.grip[arm].closed) and ck.holding(s, arm, r.ex.cfg.held_width)
        if not holding:
            r.detach(arm)  # opened (or closed on nothing): the block is on its own again
        elif r.attached[arm] is None:
            # attach once the fingers are on the block's faces (not while they close through the holding range: the
            # gripper may still be moving relative to the block): before the lift from the block's rest estimate (an
            # earlier rule, "one perceived pose within 3 mm of the node's first one", left half the transfers
            # unattached under perception noise), after it from the current perceived pose
            if ck.gripping(s, arm):
                r.attach(s, arm, self.spec["block"], rest=not self.ever_lifted)
        else:  # the grip may still settle (fingers sliding on the faces); refresh_attached confirms larger shifts
            r.refresh_attached(s, arm)
        bp, bq = r.block_est(s, self.spec["block"])
        tpb, tqb = base.to_base(bp, bq)
        gpw, gqw = grasp_target_world(bp, bq, arm)
        gpb, gqb = base.to_base(gpw, gqw)
        _, ops, oqs = self._others(s)
        near = (ops[0, :, :2] - bp[0, :2]).norm(dim=-1) < 0.2  # blocks of the same tower (as in training)
        opb, oqb = base.to_base(ops, oqs)
        tgt_pb, tgt_qb = base.to_base(ctrl.pos, ctrl.quat)
        w = torch.tensor([float(s.width[ARMS.index(arm)])], device=r.dev)
        obs = grasp_lift_actor_obs(
            pb=pb, qb=qb, prev_pb=self.prev_pb, prev_qb=self.prev_qb, dt=ROW_DT, tgt_pb=tgt_pb, tgt_qb=tgt_qb, width=w,
            closed=r.grip[arm].closed, width_cmd=r.grip[arm].width, target_pb=tpb, target_qb=tqb, grasp_pb=gpb, grasp_qb=gqb,
            other_pb=opb, other_qb=oqb, other_valid=near[None],
            target_z_world=bp[:, 2], target_z0_world=torch.tensor([self.z0], device=r.dev), prev_actions=self.prev_a,
            progress=torch.tensor([self.steps / self.max_steps], device=r.dev))
        self.prev_pb, self.prev_qb = pb, qb
        a = self.pol(obs).clamp(-1, 1)
        self.ema = self.ac["ema"] * a[:, :6] + (1 - self.ac["ema"]) * self.ema
        d = float((ctrl.pos - gpw).norm())
        sc = self.ac["pos_near_m"] + (self.ac["pos_far_m"] - self.ac["pos_near_m"]) * \
            min(1.0, max(0.0, (d - self.ac["near_radius_m"]) / (self.ac["far_radius_m"] - self.ac["near_radius_m"])))
        ctrl.apply_delta(base.vec_to_world(self.ema[:, :3] * sc), base.vec_to_world(self.ema[:, 3:6] * self.ac["rot_rad"]),
                         q_meas=qm, max_lead=self.ac["max_lead_m"])
        close = a[:, 6] > self.ac.get("grip_threshold", 0.0)
        lifted_now = float(bp[0, 2]) - self.z0
        payload = grasp_cfg_payload() if (bool(close) and self.ever_lifted) else 0.0
        r.command(arm, qm, payload=payload)
        r.grip[arm].step(close)
        self.prev_a = a
        self.steps += 1
        held = bool(close) and ck.holding(s, arm, r.ex.cfg.held_width)
        if held and lifted_now >= 0.02:
            self.ever_lifted = True
        if held and lifted_now >= self.spec["lift_m"]:
            self.info = {"lift_mm": round(lifted_now * 1000, 1), "steps": self.steps,
                         **record_held_offset(r, s, arm, self.spec["block"])}
            return Status.OK, [r.row()]
        if self.ever_lifted and not held and lifted_now < 0.01:
            self.info = {"why": "dropped"}
            return Status.FAILED, [r.row()]
        return None, [r.row()]


def record_held_offset(r: EnvRun, s: State, arm: str, bid: str) -> dict:
    """Store the held block's pose in the pinch frame (used by the place); returns the grasp quality for the node's
    info (``tower_rl.grasp.grasp_relation_error``): pinch offset from the canonical grasp in the block frame (mm; x
    along the block, z up) and the rotation from the canonical gripper orientation (deg)."""
    pp, pq = r.pinch(s, arm)
    bp, bq = r.block_est(s, bid)
    att = r.attached[arm]
    r.held_offset[arm] = att[1] if att is not None and att[0] == bid else geo.frame_relative(pp, pq, bp, bq)
    d, rot = grasp_relation_error(bp, bq, pp, pq, arm)
    return {"grasp_offset_mm": [round(float(v) * 1000, 1) for v in d[0]], "grasp_rot_deg": round(math.degrees(float(rot[0])), 1)}


class PlannedPlaceNode(NodeBase):
    """Place fallback (build file Step 6): slow descent to the release pose, open, wait, retreat, settle check."""

    def __init__(self, run, spec, s):
        super().__init__(run, spec, s)
        gp, gq = run.dest_goal(s, spec)
        self.goal = (gp, gq)
        rt = run.release_target[self.arm]
        self.release = rt if rt is not None else release_pose(run, self.arm, gp, gq, spec["release_clearance_m"])
        self.phase = 0
        self.k = 0
        self.post = PoseWindow()  # perceived poses of the block since the gripper opened

    def step(self, s):
        r, arm = self.run, self.arm
        if self.timed_out(s):
            return Status.TIMEOUT, []
        ctrl = r.ctrl[arm]
        qm = r.q_meas(s, arm)
        close = True
        payload = grasp_cfg_payload()
        if self.phase in (1, 2, 3):
            i = s.block_ids.index(self.spec["block"])
            self.post.add(s.block_pos[i], s.block_quat[i], float(s.block_conf[i]))
        if self.phase == 0:  # descend at <= 1.5 cm/s (3 mm per row near the end)
            d = self.release[0] - ctrl.pos
            dist = float(d.norm())
            v = 0.003 if dist < 0.02 else 0.008
            ctrl.set_target(ctrl.pos + d * min(1.0, v / max(dist, 1e-6)), self.release[1])
            pp, _ = r.pinch(s, arm)
            if dist < 0.001 and float((pp - self.release[0]).norm()) < 0.004:
                self.phase, self.k = 4, 0
            self.k += 1
            if self.k > 120:
                self.phase, self.k = 4, 0
        elif self.phase == 4:  # hold closed at the release pose (``pre_open_hold_s``) so the arm and block settle
            self.k += 1
            if self.k > round(self.spec.get("pre_open_hold_s", 0.0) / ROW_DT):
                self.phase, self.k = 1, 0
                return self.step(s)  # open in this same row (no extra row when the hold is 0)
        elif self.phase == 1:  # open and wait 0.6 s
            close = False
            payload = 0.0
            r.detach(arm)
            self.k += 1
            if self.k > 6:
                self.phase, self.k = 2, 0
        elif self.phase == 2:  # retreat 5 cm up
            close = False
            payload = 0.0
            tgt = ctrl.pos.clone()
            tgt[0, 2] = min(float(ctrl.pos[0, 2]) + 0.01, float(self.release[0][0, 2]) + 0.05)
            ctrl.set_target(tgt, ctrl.quat)
            self.k += 1
            if self.k > 8:
                self.phase, self.k = 3, 0
        else:  # settle and check
            close = False
            payload = 0.0
            self.k += 1
            if self.k * ROW_DT >= self.spec["settle_s"]:
                # the median of the last second's perceived poses (a single noisy pose misjudges good places)
                bp, bq = self.post.median(round(1.0 / ROW_DT)) if len(self.post) else s.block(self.spec["block"])
                dxy, _, dyaw = ck.placement_error(bp, bq, *(x[0].cpu().numpy() for x in self.goal))
                r.held_offset[arm] = None
                self.info = {"place_xy_mm": round(dxy * 1000, 2), "yaw_deg": round(dyaw, 2)}
                ok = dxy <= self.spec["tol_pos_m"] and dyaw <= self.spec["tol_yaw_deg"]
                return (Status.OK if ok else Status.FAILED), []
        r.grip[arm].step(torch.tensor([close], device=r.dev))
        r.command(arm, qm, payload=payload)
        return None, [r.row()]


class LearnedPlaceNode(NodeBase):
    """The trained place_release actor (same observation and action processing as its training env)."""

    def __init__(self, run, spec, s, policy: SkillPolicy):
        super().__init__(run, spec, s)
        self.pol = policy
        sk = load_yaml("skills/place_release")
        self.ac, self.pl = sk["action"], sk["place"]
        self.max_steps = sk["episode_steps"]
        self.ema = torch.zeros(1, 6, device=run.dev)
        self.prev_a = torch.zeros(1, 7, device=run.dev)
        self.prev_a[0, 6] = 1.0
        ctrl = run.ctrl[self.arm]
        self.prev_pb, self.prev_qb = ctrl.kin.fk(run.q_meas(s, self.arm), "pinch")
        self.goal = run.dest_goal(s, spec)
        self.released = False
        self.z_release = 0.0
        self.since = 0
        self.post = PoseWindow()  # perceived poses of the block since the release
        self.steps = 0

    def step(self, s):
        r, arm = self.run, self.arm
        ctrl, base = r.ctrl[arm], r.ctrl[arm].base
        if self.timed_out(s) or self.steps >= self.max_steps:
            return Status.TIMEOUT, []
        qm = r.q_meas(s, arm)
        pb, qb = ctrl.kin.fk(qm, "pinch")
        pw, pq_w = base.to_world(pb, qb)
        gp, gq = self.goal
        r.refresh_attached(s, arm)
        bp, bq = r.block_est(s, self.spec["block"])  # attached (kinematic prediction) until the release
        retreated = self.released and float(pw[0, 2]) >= self.z_release + self.pl["retreat_m"]
        # place check (the training env's settle check on perceived poses): once released, open and retreated,
        # ``settle_steps`` rows after the release and at rest -> decided by the tolerance on the median of the last
        # ``settle_steps`` perceived poses; at the latest 3 x settle_steps after the release. "At rest" = the medians of
        # the last two half-windows within STATIC_DRIFT_M. (Single poses, with a per-row displacement test, judged 11 of
        # 30 good Task 3-3 places failed under perception noise; each false failure re-grasps a placed block.)
        if self.released:
            i = s.block_ids.index(self.spec["block"])
            self.post.add(s.block_pos[i], s.block_quat[i], float(s.block_conf[i]))
            sp = self.pl["settle_steps"]
            drift = self.post.drift(sp // 2)
            rest = drift is not None and drift < STATIC_DRIFT_M
            if (self.since >= sp and retreated and not bool(r.grip[arm].closed) and rest) or self.since >= 3 * sp:
                p, q = self.post.median(sp) if len(self.post) else (x[0].cpu().numpy() for x in (bp, bq))
                dxy, dz, dyaw = ck.placement_error(p, q, *(x[0].cpu().numpy() for x in (gp, gq)))
                in_tol = dxy <= self.spec["tol_pos_m"] and dyaw <= self.spec["tol_yaw_deg"] and dz <= self.pl["tol_z_m"]
                r.held_offset[arm] = None
                self.info = {"place_xy_mm": round(dxy * 1000, 2), "yaw_deg": round(dyaw, 2), "steps": self.steps}
                return (Status.OK if in_tol else Status.FAILED), []
        att = r.attached[arm]
        w_jaw = r.g.get("perception", {}).get("jaw_to_goal", 0.0)
        w_jaw = w_jaw.get(arm, 0.0) if isinstance(w_jaw, dict) else w_jaw  # one value, or one per arm (per policy)
        if w_jaw and att is not None and att[0] == self.spec["block"]:
            # graph "perception": {"jaw_to_goal": w}: the held block's perceived offset along the jaw axis is perception
            # error (the fingers centre the block), and the place policies, trained so, ignore it and put the pinch over
            # the goal: an error the held block shares with the foundations then stays in the placement. Move the share
            # w of that offset from the block to the goal (same block-to-goal error, block centred in the fingers):
            # w = 1 cancels errors common to all blocks, w = 0 is best for independent per-block errors. (Moving the
            # whole in-hand offset instead made Task 2-3 places worse: the policies use the offset along the block.)
            jaw = geo.quat_apply(pq_w, torch.tensor([[0.0, 1.0, 0.0]], device=r.dev))
            shift = w_jaw * att[1][0][:, 1:2] * jaw
            gp, bp = gp - shift, bp - shift
        gpb, gqb = base.to_base(gp, gq)
        bpb, bqb = base.to_base(bp, bq)
        ids = [b for b in r.g["movable"] if b != self.spec["block"]]
        ops = torch.stack([r.block(s, b)[0][0] for b in ids])[None]
        oqs = torch.stack([r.block(s, b)[1][0] for b in ids])[None]
        near = (ops[0, :, :2] - gp[0, :2]).norm(dim=-1) < 0.2
        opb, oqb = base.to_base(ops, oqs)
        tgt_pb, tgt_qb = base.to_base(ctrl.pos, ctrl.quat)
        phase = torch.tensor([[not self.released, self.released and not retreated, retreated]], dtype=torch.float32, device=r.dev)
        w = torch.tensor([float(s.width[ARMS.index(arm)])], device=r.dev)
        obs = place_release_actor_obs(
            pb=pb, qb=qb, prev_pb=self.prev_pb, prev_qb=self.prev_qb, dt=ROW_DT, tgt_pb=tgt_pb, tgt_qb=tgt_qb, width=w,
            closed=r.grip[arm].closed, width_cmd=r.grip[arm].width, goal_pb=gpb, goal_qb=gqb, block_pb=bpb, block_qb=bqb,
            other_pb=opb, other_qb=oqb, other_valid=near[None], phase=phase, prev_actions=self.prev_a,
            progress=torch.tensor([self.steps / self.max_steps], device=r.dev))
        self.prev_pb, self.prev_qb = pb, qb
        a = self.pol(obs).clamp(-1, 1)
        self.ema = self.ac["ema"] * a[:, :6] + (1 - self.ac["ema"]) * self.ema
        d = float((bp - gp).norm())
        sc = self.ac["pos_near_m"] + (self.ac["pos_far_m"] - self.ac["pos_near_m"]) * \
            min(1.0, max(0.0, (d - self.ac["near_radius_m"]) / (self.ac["far_radius_m"] - self.ac["near_radius_m"])))
        ctrl.apply_delta(base.vec_to_world(self.ema[:, :3] * sc), base.vec_to_world(self.ema[:, 3:6] * self.ac["rot_rad"]),
                         q_meas=qm, max_lead=self.ac["max_lead_m"])
        close = a[:, 6] > self.ac.get("grip_threshold", 0.0)
        if not self.released and not bool(close):
            self.released, self.z_release = True, float(pw[0, 2])
            r.detach(arm)
        # payload from the gripper state before this row, as in training (PlaceReleaseEnv._payload_kg)
        r.command(arm, qm, payload=grasp_cfg_payload() if bool(r.grip[arm].closed) else 0.0)
        r.grip[arm].step(close)
        self.prev_a = a
        self.steps += 1
        if self.released:
            self.since += 1
        return None, [r.row()]


class PushNode(NodeBase):
    """Task 4 push: the learned push actor (``push_env``: same observation and action processing) or, without a
    policy, a planned straight push (pipeline tests). The node ends once the target has left the tower (perceived pose,
    the evaluator-side rule) with a planned retreat upward; it fails when the rest of the tower moved (rest estimates
    beyond ``structure_mm``: the remaining tower is the "required structure"), on a sideways shove, or at the timeout.
    The fingers stay closed into the pusher tool throughout."""

    def __init__(self, run, spec, s, policy: SkillPolicy | None = None):
        super().__init__(run, spec, s)
        self.pol = policy
        sk = load_yaml("skills/push")
        self.ac, self.pg = sk["action"], sk["geometry"]
        self.lat_fail = sk["push"]["lateral_fail_m"]
        self.max_steps = sk["episode_steps"]
        r = run
        self.dir = torch.tensor([spec["direction"]], dtype=torch.float32, device=r.dev)
        self.layer = spec["layer"]
        self.zoff = push_geo.z_offset(self.arm, self.layer)
        bp, bq = r.block_rest(s, spec["block"])
        self.p0, self.yaw0, self.z0 = bp.clone(), geo.yaw_from_quat(bq).clone(), float(bp[0, 2])
        self.tower_xy = torch.stack([r.block_rest(s, b)[0][0, :2] for b in r.g["tower"]["foundation"]]).mean(0)[None]
        # the rest of the tower: movable blocks still on it (within 0.12 m of its axis), the target excluded
        self.watch = [b for b in r.g["movable"] if b != spec["block"]
                      and float((r.block_rest(s, b)[0][0, :2] - self.tower_xy[0]).norm()) < 0.12]
        self.watch0 = {b: r.block_rest(s, b)[0][0].clone() for b in self.watch}
        self.ema = torch.zeros(1, 6, device=r.dev)
        self.prev_a = torch.zeros(1, 6, device=r.dev)
        self.prev_pb, self.prev_qb = r.ctrl[self.arm].kin.fk(r.q_meas(s, self.arm), "pinch")
        self.steps = 0
        self.retreat = None  # rows left of the planned retreat after the departure
        self.phase = 0       # planned push: 0 to the contact pose, 1 pushing
        self.handover, self.static_rows, self.prev_bp = False, 0, None  # learned -> planned hand-over (stalled block)
        self.lost_rows, self.last_along = 0, 0.0  # rows the target has not been seen (perceived confidence < CONF_MIN)
        r.grip[self.arm].reset(width=self.pg["tool_width_m"], closed=True)

    def _moved(self, s: State) -> float:
        return max((float((self.run.block_rest(s, b)[0][0] - p).norm()) for b, p in self.watch0.items()), default=0.0)

    def step(self, s):
        r, arm = self.run, self.arm
        ctrl, base = r.ctrl[arm], r.ctrl[arm].base
        qm = r.q_meas(s, arm)
        if self.retreat is not None:  # planned retreat: straight up, ~1 cm per row
            if self.retreat <= 0:
                return Status.OK, []
            tgt = ctrl.pos.clone()
            tgt[0, 2] += self.pg["retreat_m"] / 5
            ctrl.set_target(tgt, ctrl.quat)
            self.retreat -= 1
            r.command(arm, qm)
            return None, [r.row()]
        if self.timed_out(s) or self.steps >= self.max_steps:
            self.info = {"why": "timeout", "moved_mm": round(self._moved(s) * 1000, 1)}
            return Status.TIMEOUT, []
        bp, bq = r.block(s, self.spec["block"])
        along, lat = push_geo.progress(bp, self.p0, self.dir)
        if float(s.block_conf[s.block_ids.index(self.spec["block"])]) < CONF_MIN:
            # the target is not seen this row (tipping over the tower edge, behind the pusher): hold still rather than
            # act on a stale pose (the actor, seeing a block that stopped, pushed harder: perception closed loop, Task
            # 4-3); lost for long after it was last seen well out along the push, it has left the tower
            self.lost_rows += 1
            if self.lost_rows > self.pg.get("lost_hold_rows", 15) and self.last_along >= self.pg.get("lost_departed_along_m", 0.06):
                self.info = {"along_mm": round(self.last_along * 1000, 1), "why": "lost_departed", "steps": self.steps}
                self.retreat = 5
                return self.step(s)
            if self.lost_rows <= self.pg.get("lost_hold_rows", 15):
                r.command(arm, qm)
                self.steps += 1
                return None, [r.row()]
        else:
            self.lost_rows, self.last_along = 0, float(along[0])
        moved = self._moved(s)
        if moved > self.spec["structure_mm"] / 1000:
            self.info = {"why": "structure", "moved_mm": round(moved * 1000, 1), "along_mm": round(float(along[0]) * 1000, 1)}
            return Status.FAILED, []
        if bool(push_geo.departed(bp, torch.tensor([self.z0], device=r.dev), self.tower_xy)[0]):
            self.info = {"along_mm": round(float(along[0]) * 1000, 1), "lat_mm": round(float(lat[0]) * 1000, 1),
                         "moved_mm": round(moved * 1000, 1), "steps": self.steps}
            self.retreat = 5
            return self.step(s)
        if abs(float(lat[0])) > self.lat_fail:
            self.info = {"why": "shoved", "lat_mm": round(float(lat[0]) * 1000, 1)}
            return Status.FAILED, []
        cpw, cqw = push_geo.push_pose(bp, self.dir, arm, self.pg["contact_along_m"], self.zoff)
        if self.pol is not None and not self.handover and "handover_along_m" in self.pg:
            # the learned push never saw debris (training parks the pushed blocks away): the last block of a layer came
            # to rest on the earlier blocks' debris while still leaning on the tower (not departed) and the actor then
            # wandered off and hit the layer below (4-2 / 4-3 D0 trace). A block that stalls part-way out is finished
            # by the planned straight push instead
            # (not a block balanced at the tower edge or tipping over it: the actor finishes those itself, and a
            # planned re-approach behind a tipping block descended into the tower: 4-3 trace)
            speed = 0.0 if self.prev_bp is None else float((bp - self.prev_bp).norm()) / ROW_DT
            self.static_rows = self.static_rows + 1 if speed < 0.01 else 0
            level = float(geo.tilt_angle(bq)[0]) < math.radians(self.pg["handover_max_tilt_deg"])
            if float(along[0]) >= self.pg["handover_along_m"] and self.static_rows >= 3 and level:
                self.handover, self.phase = True, 0
                self.info["handover_along_mm"] = round(float(along[0]) * 1000, 1)
        self.prev_bp = bp.clone()
        if self.pol is None or self.handover:  # planned: to 6 mm behind the contact pose, then straight along the push
            if self.phase == 0:  # at 8 mm per row
                cp, _ = push_geo.push_pose(bp, self.dir, arm, self.pg["contact_along_m"] - 0.006, self.zoff)
                if self.handover and float((ctrl.pos[:, :2] - cp[:, :2]).norm()) > 0.01:
                    # back over the block first (the actor may have gone past it), then down behind its end face
                    cp = cp.clone()
                    cp[0, 2] = max(float(cp[0, 2]) + 0.05, float(ctrl.pos[0, 2]))
                d = cp - ctrl.pos
                dn = float(d.norm())
                ctrl.set_target(ctrl.pos + d * min(1.0, 0.01 / max(dn, 1e-6)), cqw)
                at_contact = float((ctrl.pos - push_geo.push_pose(bp, self.dir, arm, self.pg["contact_along_m"] - 0.006,
                                                                  self.zoff)[0]).norm()) < 0.003
                self.phase = 1 if (dn < 0.003 and (at_contact or not self.handover)) else 0
            else:
                lat_dir = torch.stack([-self.dir[:, 1], self.dir[:, 0], torch.zeros_like(self.dir[:, 0])], -1)
                tgt = ctrl.pos - lat_dir * ((ctrl.pos - self.p0) * lat_dir).sum(-1, keepdim=True)
                tgt[:, 2] = self.p0[:, 2] + self.zoff
                ctrl.set_target(tgt + self.dir * 0.008, cqw)
        else:
            pb, qb = ctrl.kin.fk(qm, "pinch")
            tgt_pb, tgt_qb = base.to_base(ctrl.pos, ctrl.quat)
            tpb, tqb = base.to_base(bp, bq)
            cpb, cqb = base.to_base(cpw, cqw)
            ids = [b for b in r.g["movable"] if b != self.spec["block"]]
            ops = torch.stack([r.block(s, b)[0][0] for b in ids])[None]
            oqs = torch.stack([r.block(s, b)[1][0] for b in ids])[None]
            valid = torch.tensor([[b in self.watch for b in ids]], device=r.dev)  # blocks on the tower (as in training)
            opb, oqb = base.to_base(ops, oqs)
            obs = push_actor_obs(
                pb=pb, qb=qb, prev_pb=self.prev_pb, prev_qb=self.prev_qb, dt=ROW_DT, tgt_pb=tgt_pb, tgt_qb=tgt_qb,
                target_pb=tpb, target_qb=tqb, dir_b=base.vec_to_base(self.dir), along=along, lateral=lat,
                dyaw=geo.wrap_angle(geo.yaw_from_quat(bq) - self.yaw0), contact_pb=cpb, contact_qb=cqb, other_pb=opb, other_qb=oqb,
                other_valid=valid, z0_world=torch.tensor([self.z0], device=r.dev), prev_actions=self.prev_a,
                progress=torch.tensor([self.steps / self.max_steps], device=r.dev))
            self.prev_pb, self.prev_qb = pb, qb
            a = self.pol(obs).clamp(-1, 1)
            self.ema = self.ac["ema"] * a[:, :6] + (1 - self.ac["ema"]) * self.ema
            d = float((ctrl.pos - cpw).norm())
            sc = self.ac["pos_near_m"] + (self.ac["pos_far_m"] - self.ac["pos_near_m"]) * \
                min(1.0, max(0.0, (d - self.ac["near_radius_m"]) / (self.ac["far_radius_m"] - self.ac["near_radius_m"])))
            ctrl.apply_delta(base.vec_to_world(self.ema[:, :3] * sc), base.vec_to_world(self.ema[:, 3:6] * self.ac["rot_rad"]),
                             q_meas=qm, max_lead=self.ac["max_lead_m"])
            self.prev_a = a[:, :6]
        r.command(arm, qm)
        self.steps += 1
        return None, [r.row()]


def _structure_moved(run: EnvRun, s: State, ref: dict) -> float:
    return max((float((run.block_rest(s, b)[0][0] - p).norm()) for b, p in ref.items()), default=0.0)


class InsertNode(NodeBase):
    """Task 5 held entry (planned; the learned ``insert`` skill plugs in here once trained): slide the held block along
    the slot axis from the pre-insert pose to ``entry_m`` from the slot centre at ``insert_speed_m_s``, every row
    re-aiming the pinch so the held block (attached estimate) sits on the slot line at the slot orientation; then open,
    wait, retreat straight up. Fails if the tower's side / roof blocks moved > 10 mm or the block is not in the slot."""

    def __init__(self, run, spec, s, policy: SkillPolicy | None = None):
        super().__init__(run, spec, s)
        self.pol = policy
        self.ig = load_yaml("skills/insert")["geometry"]
        self.slot = run.slot_pose(s, spec)
        self.phase, self.k = 0, 0
        self.s_along = None
        self.ref = {b: run.block_rest(s, b)[0][0].clone() for b in run.g["tower"]["structure"]}

    def _learned_step(self, s, sp, sq, d, moved):
        """The trained insert actor (``task5_env.InsertEnv``: same observation and action processing): 7-D delta pose +
        release logit; the node ends once the block is released, the pinch retreated and the block rested 0.5 s."""
        r, arm = self.run, self.arm
        ctrl, base = r.ctrl[arm], r.ctrl[arm].base
        qm = r.q_meas(s, arm)
        if not hasattr(self, "ema"):
            sk = load_yaml("skills/insert")
            self.ac, self.ins, self.max_steps = sk["action"], sk["insert"], sk["episode_steps"]
            self.ema = torch.zeros(1, 6, device=r.dev)
            self.prev_a = torch.zeros(1, 7, device=r.dev)
            self.prev_a[0, 6] = 1.0
            self.prev_pb, self.prev_qb = ctrl.kin.fk(qm, "pinch")
            self.released, self.z_release, self.steps, self.settled, self.prev_bp = False, 0.0, 0, 0, None
        pb, qb = ctrl.kin.fk(qm, "pinch")
        pw = base.to_world(pb, qb)[0]
        bp, bq = r.block(s, self.spec["block"])
        retreated = self.released and float(pw[0, 2]) >= self.z_release + self.ins["retreat_m"]
        if self.released:
            static = self.prev_bp is not None and float((bp - self.prev_bp).norm()) / ROW_DT < 0.01
            self.settled = self.settled + 1 if (retreated and static) else 0
            if self.settled >= self.ins["settle_steps"]:
                along = float(((bp - sp) * d).sum())
                lat_dir = torch.stack([-d[:, 1], d[:, 0], torch.zeros_like(d[:, 0])], -1)
                lat = float(((bp - sp) * lat_dir).sum())
                r.held_offset[arm] = None
                self.info = {"along_mm": round(along * 1000, 1), "lat_mm": round(lat * 1000, 1), "moved_mm": round(moved * 1000, 1),
                             "steps": self.steps}
                ok = self.ins["along_min_m"] - 0.01 <= along <= self.ins["along_max_m"] and abs(lat) <= 0.004
                return (Status.OK if ok else Status.FAILED), []
        self.prev_bp = bp.clone()
        if self.steps >= self.max_steps:
            return Status.TIMEOUT, []
        tgt_pb, tgt_qb = base.to_base(ctrl.pos, ctrl.quat)
        bpb, bqb = base.to_base(bp, bq)
        sp_b, sq_b = base.to_base(sp, sq)
        phase = torch.tensor([[not self.released, self.released and not retreated, retreated]], dtype=torch.float32, device=r.dev)
        w = torch.tensor([float(s.width[ARMS.index(arm)])], device=r.dev)
        obs = insert_actor_obs(pb=pb, qb=qb, prev_pb=self.prev_pb, prev_qb=self.prev_qb, dt=ROW_DT, tgt_pb=tgt_pb, tgt_qb=tgt_qb,
                               width=w, closed=r.grip[arm].closed, width_cmd=r.grip[arm].width, block_pb=bpb, block_qb=bqb,
                               slot_pb=sp_b, slot_qb=sq_b, axis_b=base.vec_to_base(d), phase=phase, prev_actions=self.prev_a,
                               progress=torch.tensor([self.steps / self.max_steps], device=r.dev))
        self.prev_pb, self.prev_qb = pb, qb
        a = self.pol(obs).clamp(-1, 1)
        self.ema = self.ac["ema"] * a[:, :6] + (1 - self.ac["ema"]) * self.ema
        along = float(((bp - sp) * d).sum())
        sc = self.ac["pos_near_m"] if along > -0.18 else self.ac["pos_far_m"]
        ctrl.apply_delta(base.vec_to_world(self.ema[:, :3] * sc), base.vec_to_world(self.ema[:, 3:6] * self.ac["rot_rad"]),
                         q_meas=qm, max_lead=self.ac["max_lead_m"])
        close = a[:, 6] > self.ac.get("grip_threshold", 0.0)
        if not self.released and self.ins.get("release_min_along_m") is not None and along < self.ins["release_min_along_m"]:
            close = torch.ones_like(close)  # as InsertEnv._grip_mask: no release before the block is in the slot
        if not self.released and not bool(close):
            self.released, self.z_release = True, float(pw[0, 2])
            r.detach(arm)
        r.command(arm, qm, payload=grasp_cfg_payload() if bool(r.grip[arm].closed) else 0.0)
        r.grip[arm].step(close)
        self.prev_a = a
        self.steps += 1
        return None, [r.row()]

    def step(self, s):
        r, arm = self.run, self.arm
        ctrl = r.ctrl[arm]
        qm = r.q_meas(s, arm)
        if self.timed_out(s):
            return Status.TIMEOUT, []
        sp, sq, d = self.slot
        moved = _structure_moved(r, s, self.ref)
        if moved > 0.010:
            self.info = {"why": "structure", "moved_mm": round(moved * 1000, 1)}
            return Status.FAILED, []
        if self.pol is not None:
            return self._learned_step(s, sp, sq, d, moved)
        close, payload = True, grasp_cfg_payload()
        if self.phase == 0:  # held slide along the slot line, closed loop on the perceived block
            # the held offset recorded after the grasp goes stale (the block hangs / slips in the 3 N fingers: 7.6 mm in
            # a Task 5-2 trace), so every row the pinch target moves by the perceived block's error to the slot line:
            # across / height / yaw first (aligned within 1 mm, 0.5 deg before the leading end reaches the tower face),
            # then the depth at insert_speed
            bp, bq = r.block(s, self.spec["block"])
            along = float(((bp - sp) * d).sum())
            lat_dir = torch.stack([-d[:, 1], d[:, 0], torch.zeros_like(d[:, 0])], -1)
            e_lat = -float(((bp - sp) * lat_dir).sum())
            e_z = float(sp[0, 2] + self.spec.get("entry_lift_m", 0.0) - bp[0, 2])
            signed = float(((geo.yaw_from_quat(sq) - geo.yaw_from_quat(bq) + math.pi / 2) % math.pi - math.pi / 2)[0])
            # pitch: the held block's long axis must be level (2 mm vertical clearance over a 150 mm block; a 1 deg pitch
            # in the 3 N grasp put the leading edge 1.3 mm low and it pushed the lower layer's side block: 5-3 trace)
            bx = geo.quat_apply(bq, torch.tensor([[1.0, 0.0, 0.0]], device=r.dev))
            bx = bx * (1.0 if float((bx * d).sum()) >= 0 else -1.0)  # long axis pointing into the slot
            pitch = float(torch.asin(bx[0, 2].clamp(-1, 1)))         # > 0: leading end high
            aligned = abs(e_lat) < 0.0008 and abs(e_z) < 0.0006 and abs(signed) < math.radians(0.4) and abs(pitch) < math.radians(0.3)
            if self.s_along is None:
                self.s_along = along
            entering = along > -0.16  # leading end within 1 cm of the tower face
            if aligned or not entering:
                self.s_along = min(self.s_along + self.ig["insert_speed_m_s"] * ROW_DT, -self.ig["entry_m"])
            else:  # misaligned at the entrance: hold the depth (do not chase a target the block lags behind)
                self.s_along = min(self.s_along, along)
            step_along = max(-0.004, min(0.004, self.s_along - along))
            corr = d * step_along + lat_dir * max(-0.003, min(0.003, 0.6 * e_lat))
            corr[0, 2] += max(-0.003, min(0.003, 0.6 * e_z))
            rot = geo.quat_mul(geo.quat_from_axis_angle(lat_dir, torch.tensor([0.6 * pitch], device=r.dev)),
                               geo.quat_from_yaw(torch.tensor([0.6 * signed], device=r.dev)))
            ctrl.set_target(ctrl.pos + corr, geo.quat_mul(rot, ctrl.quat))
            if self.s_along >= -self.ig["entry_m"] - 1e-9 and abs(along + self.ig["entry_m"]) < 0.004:
                self.phase, self.k = 1, 0
            self.k += 1
            if self.k > 400:
                self.info = {"why": "slide stalled", "along_mm": round(along * 1000, 1), "lat_mm": round(e_lat * 1000, 1),
                             "z_mm": round(e_z * 1000, 1)}
                return Status.FAILED, []
        elif self.phase == 1:  # open and wait 0.6 s
            close, payload = False, 0.0
            r.detach(arm)
            self.k += 1
            if self.k > 6:
                self.phase, self.k = 2, 0
        else:  # retreat: back off 2 cm along the slot axis first (rising at once dragged the roof: the open fingers
            # sit 3 mm from its face), then up; then judge the block's position in the slot
            close, payload = False, 0.0
            tgt = ctrl.pos.clone()
            if self.k < 4:
                tgt = tgt - d * 0.005
            else:
                tgt[0, 2] += self.ig["release_retreat_m"] / 4
            ctrl.set_target(tgt, ctrl.quat)
            self.k += 1
            if self.k > 8:
                bp, _ = r.block(s, self.spec["block"])
                along = float(((bp - sp) * d).sum())
                lat = float(torch.linalg.norm((bp - sp - d * along)[0, :2]))
                r.held_offset[arm] = None
                self.info = {"along_mm": round(along * 1000, 1), "lat_mm": round(lat * 1000, 1), "moved_mm": round(moved * 1000, 1)}
                ok = abs(along + self.ig["entry_m"]) < 0.015 and lat < 0.004
                return (Status.OK if ok else Status.FAILED), []
        r.grip[arm].step(torch.tensor([close], device=r.dev))
        r.command(arm, qm, payload=payload)
        return None, [r.row()]


class PushInNode(NodeBase):
    """Task 5 push home: the learned push_in actor when loaded (``task5_env.PushInEnv``: same observation and action
    processing) until the block is home, else a planned push: from the pre-push-in pose descend behind the block's outer
    end face to its centre height, push along the slot axis at ``pushin_speed_m_s`` until the block centre reaches the
    slot centre. Both end with a planned retreat (back along the axis, then up) and the judgement: OK when the block is
    within ``tol_pos_m`` of the slot centre (5 deg) and the tower's side / roof blocks stayed within 10 mm."""

    def _learned_step(self, s, sp, sq, d, bp, bq, along, moved):
        r, arm = self.run, self.arm
        ctrl, base = r.ctrl[arm], r.ctrl[arm].base
        qm = r.q_meas(s, arm)
        if not hasattr(self, "ema"):
            sk = load_yaml("skills/push_in")
            self.ac, self.pi, self.max_steps = sk["action"], sk["push_in"], sk["episode_steps"]
            self.ema = torch.zeros(1, 6, device=r.dev)
            self.prev_a = torch.zeros(1, 6, device=r.dev)
            self.prev_pb, self.prev_qb = ctrl.kin.fk(qm, "pinch")
            self.steps, self.settled, self.prev_bp = 0, 0, None
        static = self.prev_bp is not None and float((bp - self.prev_bp).norm()) / ROW_DT < 0.01
        self.prev_bp = bp.clone()
        self.settled = self.settled + 1 if (abs(along) <= self.pi["along_tol_m"] and static) else 0
        if self.settled >= self.pi["settle_steps"]:
            self.phase, self.k = 2, 0  # home: the planned retreat (back off along the axis, then up), then the judgement
            return self.step(s)
        if self.steps >= self.max_steps or along > self.pi["overshoot_m"]:
            self.phase, self.k = 2, 0
            return self.step(s)
        pb, qb = ctrl.kin.fk(qm, "pinch")
        tgt_pb, tgt_qb = base.to_base(ctrl.pos, ctrl.quat)
        bpb, bqb = base.to_base(bp, bq)
        sp_b, sq_b = base.to_base(sp, sq)
        w = torch.tensor([float(s.width[ARMS.index(arm)])], device=r.dev)
        # "closed" as in PushInEnv: the pusher phase reports the gripper open (its command is the pusher width); the
        # executor's pusher state (closed at the pusher width) fed +1, a value the actor never saw, and it lunged into
        # the tower (5-2 / 5-3 with push_in left v1: 0 / 64)
        obs = insert_actor_obs(pb=pb, qb=qb, prev_pb=self.prev_pb, prev_qb=self.prev_qb, dt=ROW_DT, tgt_pb=tgt_pb, tgt_qb=tgt_qb,
                               width=w, closed=torch.zeros(1, dtype=torch.bool, device=r.dev), width_cmd=r.grip[arm].width,
                               block_pb=bpb, block_qb=bqb, slot_pb=sp_b, slot_qb=sq_b, axis_b=base.vec_to_base(d),
                               phase=torch.tensor([[0.0, 1.0, 0.0]], device=r.dev), prev_actions=pad_actions(self.prev_a),
                               progress=torch.tensor([self.steps / self.max_steps], device=r.dev))
        self.prev_pb, self.prev_qb = pb, qb
        a = self.pol(obs).clamp(-1, 1)
        self.ema = self.ac["ema"] * a[:, :6] + (1 - self.ac["ema"]) * self.ema
        cp = bp + d * self.ig["pushin_contact_along_m"]
        dd = float((ctrl.pos - cp).norm())
        sc = self.ac["pos_near_m"] + (self.ac["pos_far_m"] - self.ac["pos_near_m"]) * \
            min(1.0, max(0.0, (dd - self.ac["near_radius_m"]) / (self.ac["far_radius_m"] - self.ac["near_radius_m"])))
        ctrl.apply_delta(base.vec_to_world(self.ema[:, :3] * sc), base.vec_to_world(self.ema[:, 3:6] * self.ac["rot_rad"]),
                         q_meas=qm, max_lead=self.ac["max_lead_m"])
        r.command(arm, qm)
        self.prev_a = a[:, :6]
        self.steps += 1
        return None, [r.row()]




    def __init__(self, run, spec, s, policy: SkillPolicy | None = None):
        super().__init__(run, spec, s)
        self.pol = policy
        self.ig = load_yaml("skills/insert")["geometry"]
        self.slot = run.slot_pose(s, spec)
        self.phase, self.k = 0, 0
        self.ref = {b: run.block_rest(s, b)[0][0].clone() for b in run.g["tower"]["structure"]}
        self.width = self.ig["pusher_width_m"]
        run.grip[self.arm].reset(width=self.width, closed=True)
        self.hold, self.extra, self.prev_along = 0, 0.0, None

    def step(self, s):
        r, arm = self.run, self.arm
        ctrl = r.ctrl[arm]
        qm = r.q_meas(s, arm)
        if self.timed_out(s):
            return Status.TIMEOUT, []
        sp, sq, d = self.slot
        moved = _structure_moved(r, s, self.ref)
        if moved > 0.010:
            self.info = {"why": "structure", "moved_mm": round(moved * 1000, 1)}
            return Status.FAILED, []
        bp, bq = r.block(s, self.spec["block"])
        along = float(((bp - sp) * d).sum())
        line = sp + d * along  # the block's projection on the slot axis
        if self.pol is not None and self.phase < 2:  # the trained push_in actor until the block is home, then the retreat
            return self._learned_step(s, sp, sq, d, bp, bq, along, moved)
        if self.phase == 0:  # down behind the outer end face (6 mm short of contact), on the slot line
            c = line + d * (self.ig["pushin_contact_along_m"] - 0.006)
            c[0, 2] = sp[0, 2]
            dv = c - ctrl.pos
            dn = float(dv.norm())
            ctrl.set_target(ctrl.pos + dv * min(1.0, 0.01 / max(dn, 1e-6)), ctrl.quat)
            if dn < 0.002:
                self.phase, self.k = 1, 0
        elif self.phase == 1:  # push: the pinch advances along the axis toward "block centred", 2 mm per row
            end = sp + d * self.ig["pushin_contact_along_m"]
            tgt = ctrl.pos - (ctrl.pos - sp - d * float(((ctrl.pos - sp) * d).sum()))  # back onto the slot line
            # hold the measured pinch at the block's centre height: the arm sags ~3 mm under the push force, and a low
            # push let the insert's leading edge catch the far side block of the empty gap below (5-3 trace, 13 mm)
            pz = float(r.pinch(s, arm)[0][0, 2])
            zt = float(sp[0, 2]) + self.ig.get("pushin_z_offset_m", 0.0)  # pinch height relative to the slot centre
            if self.spec.get("gap_below") and "cross_window_m" in self.ig:
                # crossing the open gap below (5-3 layer 2): the insert's bottom is exactly level with the far side
                # block's top (62.000 mm both; PhysX box-box then blocks it on that block's edge about half the time).
                # Sliding the pusher down the end face while it crosses drags the tail down by friction (the push force
                # rises once the nose is blocked): the nose tips up a hair and rides over the edge
                c0, c1 = self.ig["cross_window_m"]
                z0, z1 = self.ig["cross_z_m"]
                if along >= c0 - 0.01:
                    zt += z0 + (z1 - z0) * min(1.0, max(0.0, (along - c0) / (c1 - c0)))
            self.z_cmd = (zt if not hasattr(self, "z_cmd") else self.z_cmd) + 0.5 * (zt - pz)
            self.z_cmd = min(max(self.z_cmd, zt - 0.002), zt + 0.008)
            tgt[0, 2] = self.z_cmd
            rem = float(((end - tgt) * d).sum()) + self.extra
            # slow down over the last centimetres (pushin_decel_per_s x the block's remaining distance): at 4 cm/s the
            # 10 Hz steps kick the light block ~8 mm ahead every other row and it overshot the centre 5-12 mm (5-3 layer 1)
            v = min(self.ig["pushin_speed_m_s"], max(self.ig.get("pushin_min_speed_m_s", 0.0),
                                                      self.ig.get("pushin_decel_per_s", math.inf) * max(0.0, -along)))
            if self.spec.get("gap_below") and "cross_speed_m_s" in self.ig and \
                    self.ig["cross_window_m"][0] - 0.01 <= along <= self.ig["cross_window_m"][1]:
                v = min(v, self.ig["cross_speed_m_s"])  # optional slower crossing (experiment)
            tgt = tgt + d * max(0.0, min(v * ROW_DT, rem))
            ctrl.set_target(tgt, ctrl.quat)
            self.k += 1
            # the target reached "block centred": the arm lags it 4-10 mm at 4 cm/s (Task 5-3 layer 1 stopped 4.6-10.7 mm
            # short), so hold, then nudge on 1 mm per row (<= 6 mm) while the block is static and still short
            static = self.prev_along is not None and abs(along - self.prev_along) < 0.0005
            self.prev_along = along
            if rem <= 1e-4:
                self.hold += 1
                if self.hold > 3 and static and along < -0.0015:
                    self.extra = min(self.extra + 0.001, 0.006)
            if abs(along) < 0.0015 or self.hold > 15 or self.k > 150:
                self.phase, self.k = 2, 0
        else:  # back off along the axis, then up; judge
            tgt = ctrl.pos.clone()
            if self.k < 5:
                tgt = tgt - d * (self.ig["pushin_retreat_m"] / 5)
            else:
                tgt[0, 2] += 0.01
            ctrl.set_target(tgt, ctrl.quat)
            self.k += 1
            if self.k > 9:
                dxy, _, dyaw = ck.placement_error(*(x[0].cpu().numpy() for x in (bp, bq, sp, sq)))
                self.info = {"place_xy_mm": round(dxy * 1000, 2), "yaw_deg": round(dyaw, 2), "moved_mm": round(moved * 1000, 1)}
                ok = dxy <= self.spec["tol_pos_m"] and dyaw <= 5.0
                return (Status.OK if ok else Status.FAILED), []
        r.command(arm, qm)
        return None, [r.row()]
