"""Step 8.1 diagnosis: where does perception noise turn into placement error? Runs rehearsal batches with the
provisional perception noise and, for every place node, compares the executor's estimates with simulator truth.

Per place (one JSON line): the executor's verdict and perceived error; the true error at the decision; at the release
the belief error of the held block (executor estimate - truth), the perception error of the held block (measurement -
truth), the goal error (perceived goal - true goal) and the policy's own alignment error (estimate - perceived goal).
true error at release = alignment - belief + goal, so a belief that differs from the perception error is a bias that
does not cancel against the goal's.
  python scripts/noise_diag.py --headless --device cpu --variant task3-3_vertical_stacking_3 \
      --bundle results/bundles/teacher_v5 --noise '{"bias_pos_m": 0.001, "bias_yaw_deg": 1.0, "shared_pos_m": 0.003, \
      "shared_yaw_deg": 3.0}' --out results/noise_diag/t33.jsonl
Diagnostics only: truth is read from ``State.privileged`` here, never by the executor.
"""
import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", required=True)
parser.add_argument("--tier", default="D1")
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=6000)
parser.add_argument("--bundle", default=None)
parser.add_argument("--rate_hz", type=int, choices=[10, 30], default=10)
parser.add_argument("--noise", default="{}", help="JSON NoiseCfg overrides")
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--probe_bias", default=None, help='JSON {"movable": [dx, dy, dz], "foundation": [dx, dy, dz]} (m): fixed perception '
                    'biases instead of the random ones (use with --noise of zeros), e.g. to see how each place policy uses the held '
                    'block\'s perceived offset')
parser.add_argument("--graph_set", default="{}", help='JSON merged into every graph\'s top level, e.g. {"perception": {"jaw_to_goal": 1.0}}')
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
from isaaclab.scene import InteractiveScene  # noqa: E402

import tower_rl.eval.runner as runner_mod  # noqa: E402
from tower_rl import geometry as geo  # noqa: E402
from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.domain import DomainSampler  # noqa: E402
from tower_rl.eval.metrics import aggregate  # noqa: E402
from tower_rl.eval.perception_noise import NoiseCfg, PerceptionNoise  # noqa: E402
from tower_rl.graph import executor as exm  # noqa: E402
from tower_rl.graph.executor import Executor  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402

NOISERS: list[PerceptionNoise] = []
PROBE = None if args.probe_bias is None else {k: np.asarray(v, dtype=float) for k, v in json.loads(args.probe_bias).items()}
BLOCK_IDS: list[str] = []


class RegisteredNoise(PerceptionNoise):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if PROBE is not None:
            self.bias_p = np.array([PROBE["foundation" if "base" in b else "movable"] for b in BLOCK_IDS], dtype=float)
            self.bias_y = np.zeros(len(BLOCK_IDS))
        NOISERS.append(self)


runner_mod.PerceptionNoise = RegisteredNoise


def truth(s):
    p = s.privileged.get("true_block_pos")
    return s if p is None else replace(s, block_pos=p, block_quat=s.privileged["true_block_quat"])


def yaw_deg(q) -> float:
    return math.degrees(float(geo.yaw_from_quat(torch.as_tensor(np.asarray(q), dtype=torch.float32).reshape(1, 4))[0]))


def dyaw(a, b) -> float:
    """pi-symmetric yaw difference a - b (deg)."""
    return (a - b + 90.0) % 180.0 - 90.0


def pose_np(p, q):
    return np.asarray(p, dtype=float).reshape(-1)[:3], np.asarray(q, dtype=float).reshape(-1)[:4]


def t2np(x):
    return x[0].detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


EX: Executor | None = None
OUT = None
BATCH = {"seed": 0}


def env_of(run) -> int:
    return EX.runs.index(run)


def wrap_place(cls):
    orig = cls.step

    def step(self, s):
        r, bid = self.run, self.spec["block"]
        released_before = getattr(self, "released", None)
        if released_before is None:  # PlannedPlaceNode: released when phase leaves 0 / 4
            released_before = getattr(self, "phase", 0) not in (0, 4)
        est_before = [t2np(x) for x in r.block_est(s, bid)]
        att = r.attached[self.arm]
        est_att = t2np(att[1][0]) if att is not None and att[0] == bid else None
        st, rows = orig(self, s)
        released_now = getattr(self, "released", None)
        if released_now is None:
            released_now = getattr(self, "phase", 0) not in (0, 4)
        tr = truth(s)
        d = self.__dict__.setdefault("_diag", {})
        if released_now and not released_before:
            gp, gq = (t2np(x) for x in self.goal)
            tgp, tgq = (t2np(x) for x in r.dest_goal(tr, self.spec, raw=True))
            mp, mq = pose_np(*s.block(bid))
            tp, tq = pose_np(*tr.block(bid))
            ep, eq = est_before
            e = env_of(r)
            nz = NOISERS[e] if NOISERS else None
            i = s.block_ids.index(bid)
            fids = r.g["foundation"]["destination"]
            pq_now = t2np(r.pinch(s, self.arm)[1])
            jaw = geo.quat_apply(torch.as_tensor(pq_now, dtype=torch.float32)[None], torch.tensor([[0.0, 1.0, 0.0]]))[0].numpy()
            d["release"] = {
                "t": round(s.t, 2),
                "jaw_world": jaw.round(4).tolist(),
                "att_offset_mm": None if est_att is None else (est_att * 1000).round(2).tolist(),
                "align_xy_mm": ((ep[:2] - gp[:2]) * 1000).round(2).tolist(), "align_yaw": round(dyaw(yaw_deg(eq), yaw_deg(gq)), 2),
                "belief_xy_mm": ((ep[:2] - tp[:2]) * 1000).round(2).tolist(), "belief_yaw": round(dyaw(yaw_deg(eq), yaw_deg(tq)), 2),
                "meas_xy_mm": ((mp[:2] - tp[:2]) * 1000).round(2).tolist(), "meas_yaw": round(dyaw(yaw_deg(mq), yaw_deg(tq)), 2),
                "goal_xy_mm": ((gp[:2] - tgp[:2]) * 1000).round(2).tolist(), "goal_yaw": round(dyaw(yaw_deg(gq), yaw_deg(tgq)), 2),
                "true_xy_mm": ((tp[:2] - tgp[:2]) * 1000).round(2).tolist(), "true_yaw": round(dyaw(yaw_deg(tq), yaw_deg(tgq)), 2),
                "z_above_goal_mm": round((tp[2] - tgp[2]) * 1000, 1),
                "bias_block_xy_mm": (nz.bias_p[i][:2] * 1000).round(2).tolist() if nz else None,
                "bias_block_yaw": round(math.degrees(nz.bias_y[i]), 2) if nz else None,
                "bias_found_xy_mm": (np.mean([nz.bias_p[s.block_ids.index(f)][:2] for f in fids], 0) * 1000).round(2).tolist() if nz else None,
                "bias_found_yaw": round(float(np.mean([math.degrees(nz.bias_y[s.block_ids.index(f)]) for f in fids])), 2) if nz else None,
                "held_offset_mm": None if r.held_offset[self.arm] is None else (t2np(r.held_offset[self.arm][0]) * 1000).round(2).tolist(),
            }
        if st is not None:
            gp, gq = (t2np(x) for x in self.goal)
            tgp, tgq = (t2np(x) for x in r.dest_goal(tr, self.spec, raw=True))
            mp, mq = pose_np(*s.block(bid))
            tp, tq = pose_np(*tr.block(bid))
            rec = {"batch_seed": BATCH["seed"], "env": env_of(r), "node": self.spec["id"], "arm": self.arm, "t": round(s.t, 2),
                   "status": st.value, "info": self.info,
                   "perceived_xy_mm": round(float(np.linalg.norm(mp[:2] - gp[:2])) * 1000, 2),
                   "perceived_yaw": round(abs(dyaw(yaw_deg(mq), yaw_deg(gq))), 2),
                   "true_xy_mm": round(float(np.linalg.norm(tp[:2] - tgp[:2])) * 1000, 2),
                   "true_yaw": round(abs(dyaw(yaw_deg(tq), yaw_deg(tgq))), 2),
                   "true_z_mm": round(float(tp[2] - tgp[2]) * 1000, 1),
                   "release": d.get("release")}
            OUT.write(json.dumps(rec) + "\n")
            OUT.flush()
        return st, rows

    cls.step = step


wrap_place(exm.LearnedPlaceNode)
wrap_place(exm.PlannedPlaceNode)


def wrap_grasp(cls):
    """At a successful grasp: the held block's perceived offset in the pinch frame (the attached estimate) against its
    true offset, and the pinch's position against the perceived and the true block centre at the node start."""
    orig = cls.step

    def step(self, s):
        r, bid = self.run, self.spec["block"]
        if not hasattr(self, "_diag_start"):
            tr0 = truth(s)
            self._diag_start = (pose_np(*s.block(bid)), pose_np(*tr0.block(bid)))
        st, rows = orig(self, s)
        if st is not None and st.value == "ok":
            tr = truth(s)
            pp, pq = (t2np(x) for x in r.pinch(s, self.arm))
            att = r.attached[self.arm]
            tp, tq = pose_np(*tr.block(bid))
            tp_t = torch.as_tensor(tp, dtype=torch.float32)[None]
            true_off = t2np(geo.frame_relative(torch.as_tensor(pp)[None].float(), torch.as_tensor(pq)[None].float(),
                                                tp_t, torch.as_tensor(tq, dtype=torch.float32)[None])[0])
            OUT.write(json.dumps({"batch_seed": BATCH["seed"], "env": env_of(r), "grasp": self.spec["id"], "arm": self.arm,
                                  "t": round(s.t, 2),
                                  "perceived_offset_mm": None if att is None else (t2np(att[1][0]) * 1000).round(2).tolist(),
                                  "true_offset_mm": (true_off * 1000).round(2).tolist()}) + "\n")
        return st, rows

    cls.step = step


wrap_grasp(exm.LearnedGraspNode)

phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(make_station_scene_cfg(args.variant, args.num_envs))
sim.reset()
world = StationWorld(scene, sim, args.variant)
BLOCK_IDS[:] = list(world.block_ids)
EX = Executor.from_bundle(args.bundle, args.variant, device=str(world.dev), rate_hz=args.rate_hz)
EX.graph.update(json.loads(args.graph_set))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
OUT = open(args.out, "w")
noise = NoiseCfg(**json.loads(args.noise))
outcomes = []
nb = (args.episodes + args.num_envs - 1) // args.num_envs
for b in range(nb):
    NOISERS.clear()
    BATCH["seed"] = args.seed + 100 * b
    outs = runner_mod.run_batch(world, EX, DomainSampler(args.tier, device=world.dev), seed=BATCH["seed"],
                                max_time_s=args.max_time_s, rate_hz=args.rate_hz, noise=noise)
    outcomes += outs[: args.episodes - b * args.num_envs]
    for e, o in enumerate(outs):
        OUT.write(json.dumps({"batch_seed": BATCH["seed"], "env": e, "episode": True, "seed": o.extra["seed"],
                              "case": o.extra["case"], "success": o.task_success, "place_ok": o.place_ok,
                              "events": o.extra["events"], "policy_log": o.extra.get("policy_log")}) + "\n")
    m = aggregate(outcomes)
    print(f"[noise_diag] batch {b + 1}/{nb}: TSR {m['TSR']['value']:.2f} PSR {m['PSR']['value']:.2f}", flush=True)
OUT.close()
os._exit(0)
