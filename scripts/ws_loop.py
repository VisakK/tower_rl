"""Student plan Step 11, gate G11a: the station scene drives the policy server through a real WebSocket client, as the
evaluation worker does, and the local evaluator scores the episodes.

Per control call: the four cameras are rendered (as in the camera loop, ``tower_rl.perception.closed_loop``), the static
ones turned to the delivered orientation, and an ``infer`` request goes to the env's own server session with the measured
joints and widths (float32), ``sim_time_s`` and ``step_index``; the returned chunk is executed row by row before the next
call (``run_batch``). One session per env and episode, opened with the worker's headers and a ``reset`` per episode. The
server must run with ``--camera_source render`` (our Isaac renders lack the evaluator's principal-point offsets).

  python scripts/serve_student.py --track data/challenge/track.json --camera_source render --name ws_loop &
  python scripts/ws_loop.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --tiers D0 D1 \
      --episodes 4 --num_envs 2 --seed 7000 --track data/challenge/track.json --out_root results/student/ws_loop/g11a

With seed 7000 and 2 envs, the episodes are those of the Step 10 camera loop (``closed_loop/student_v1_kp_v2``); compare
with ``scripts/compare_runs.py``. Outputs per variant and tier as ``rehearse.py`` (feedback.md, episodes.json,
metrics.json) plus ``ws_loop.json`` (client-side call latency).
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--url", default="ws://127.0.0.1:9001")
parser.add_argument("--track", required=True)
parser.add_argument("--variant", required=True)
parser.add_argument("--tiers", nargs="+", default=["D0", "D1"])
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=7000)
parser.add_argument("--case", nargs="+", default=None, help="force the tier's robustness case(s), as rehearse.py --case")
parser.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0])
parser.add_argument("--finger_effort", type=float, default=None)
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--out_root", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if "fsWatcherEnabled" not in (args.kit_args or ""):  # as rehearse.py: Kit's file watchers exhaust inotify
    args.kit_args = ((args.kit_args or "") + " --/app/extensions/fsWatcherEnabled=0").strip()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
from isaaclab.scene import InteractiveScene  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.domain import DomainSampler  # noqa: E402
from tower_rl.eval.metrics import aggregate  # noqa: E402
from tower_rl.eval.runner import run_batch  # noqa: E402
from tower_rl.graph.rehearsal import write_report  # noqa: E402
from tower_rl.perception.cameras import CAMERA_IDS, deliver  # noqa: E402
from tower_rl.perception.closed_loop import CAM_KEY  # noqa: E402
from tower_rl.scene import EVAL_RENDER_CFG  # noqa: E402
from tower_rl.server.tower_protocol import pack, unpack  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402

track = json.loads(Path(args.track).read_text())
INSTRUCTION = {v["id"]: v["instruction"] for v in track["variants"]}


class RemoteObserve:
    """The four rendered cameras of the requesting envs, in the delivered orientation (as ``CameraPerception``),
    handed to ``RemotePolicy``; the states pass through unchanged (the server estimates the poses)."""

    def __init__(self, subframes: int = 4):
        self.subframes = subframes
        self.images: dict[int, dict] = {}
        self.lights = None

    def reset(self, variant: str, seeds: list[int]):
        self.images = {}

    def set_domain(self, world, dom):
        """The episodes' lighting (robustness case "lighting") on the rendered station lights."""
        if self.lights is None:
            import omni.usd

            from tower_rl.sim.lights import StationLights

            self.lights = StationLights(omni.usd.get_context().get_stage(), world.scene.env_origins.cpu().numpy())
        self.lights.apply(dom)

    def __call__(self, world, envs, states):
        for _ in range(self.subframes):
            world.sim.render()
        rgb = {}
        for name, key in CAM_KEY.items():
            cam = world.scene[key]
            cam.update(0.0, force_recompute=True)
            rgb[name] = cam.data.output["rgb"][..., :3].cpu().numpy()
        for e in envs:
            self.images[e] = {n: np.ascontiguousarray(deliver(n, rgb[n][e])) for n in CAMERA_IDS}
        return states


class RemotePolicy:
    """The runner's ``Policy``: one server session per env and episode; ``act`` sends the envs' requests at once."""

    def __init__(self, url: str, variant: str, observe: RemoteObserve):
        self.url, self.variant, self.observe = url, variant, observe
        self.headers = {"X-Tower-Task-Variant": variant,
                        "X-Tower-Track-Version": track.get("policy_track_version") or track["version"],
                        "X-Tower-Evaluation-Version": track["version"]}
        self.socks, self.sess = [], []
        self.latency: list[float] = []

    def reset(self, n_envs: int, variant: str, seeds: list[int]):
        for ws in self.socks:
            ws.close()
        self.socks, self.sess = [], []
        for e in range(n_envs):
            ws = connect(self.url, additional_headers=self.headers, compression=None, max_size=64 * 1024 * 1024, open_timeout=30)
            meta = unpack(ws.recv(timeout=10))
            assert meta["type"] == "metadata" and meta["ready"], meta
            s = {"session_id": str(uuid.uuid4()), "trial_id": str(uuid.uuid4()), "policy_id": meta["policy_id"],
                 "seed": int(seeds[e]) % 2**32, "index": 0}
            ws.send(pack({"protocol": "tower-infer-v1", "type": "reset", "request_id": str(uuid.uuid4()), "seed": s["seed"],
                          **{k: s[k] for k in ("session_id", "trial_id", "policy_id")}}))
            ack = unpack(ws.recv(timeout=track["inference_timeout_s"]))
            assert ack["type"] == "reset_ack", ack
            self.socks.append(ws)
            self.sess.append(s)

    def _one(self, e: int, st, out: dict):
        try:
            s, ws = self.sess[e], self.socks[e]
            obs = {"schema": track["observation_schema"], "step_index": int(st.step), "sim_time_s": float(st.t),
                   "task_variant": self.variant, "instruction": INSTRUCTION[self.variant],
                   "joint_positions": np.asarray(st.q, dtype="<f4").reshape(2, 7),
                   "gripper_widths_m": np.asarray(st.width, dtype="<f4").reshape(2), "images": self.observe.images[e]}
            t0 = time.perf_counter()
            ws.send(pack({"protocol": "tower-infer-v1", "type": "infer", "request_id": str(uuid.uuid4()), "observation": obs,
                          **{k: s[k] for k in ("session_id", "trial_id", "policy_id")}}))
            res = unpack(ws.recv(timeout=track["inference_timeout_s"]))
            self.latency.append(time.perf_counter() - t0)
            if res.get("type") != "infer_result" or res.get("sample_index") != s["index"]:
                raise RuntimeError(f"server reply {res.get('type')}: {res.get('error') or res.get('sample_index')}")
            s["index"] += 1
            out[e] = np.asarray(res["actions"], dtype=np.float32)
        except Exception as ex:  # noqa: BLE001 (reported by act: the runner records a policy error)
            out[e] = ex

    def act(self, states, env_ids):
        out: dict = {}
        th = [threading.Thread(target=self._one, args=(e, st, out)) for e, st in zip(env_ids, states)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for e in env_ids:
            if isinstance(out[e], Exception):
                raise out[e]
        return [out[e] for e in env_ids]

    def focus_block(self, e):
        return None

    def node_info(self, e):
        return {}

    def close(self):
        for ws in self.socks:
            ws.close()


phys = load_yaml("physics")
cfg = sim_cfg(phys, device=args.device)
cfg.render = sim_utils.RenderCfg(**EVAL_RENDER_CFG)
sim = sim_utils.SimulationContext(cfg)
variant = args.variant
scene = InteractiveScene(make_station_scene_cfg(variant, args.num_envs, cameras=True, finger_effort=args.finger_effort))
sim.reset()
world = StationWorld(scene, sim, variant, finger_friction=tuple(args.finger_friction))
observe = RemoteObserve()
policy = RemotePolicy(args.url, variant, observe)
summary = {}
for tier in args.tiers:
    sampler = DomainSampler(tier, device=world.dev, max_blocks=max(6, world.spec.num_movable))
    outcomes, t0 = [], time.time()
    policy.latency = []
    nb = (args.episodes + args.num_envs - 1) // args.num_envs
    for b in range(nb):
        case = [args.case[(b * args.num_envs + e) % len(args.case)] for e in range(args.num_envs)] if args.case else None
        outs = run_batch(world, policy, sampler, seed=args.seed + 100 * b, max_time_s=args.max_time_s, rate_hz=10,
                         observe=observe, case=case)
        outcomes += outs[: args.episodes - b * args.num_envs]
        done = aggregate(outcomes)
        print(f"[ws_loop] {variant} {tier} batch {b + 1}/{nb}: TSR {done['TSR']['value']:.2f} GSR {done['GSR']['value']:.2f} "
              f"PSR {done['PSR']['value']:.2f} ODR {done['ODR']['value']:.2f} ({time.time() - t0:.0f} s)", flush=True)
    out_dir = Path(args.out_root) / variant / (tier + "_ws_loop" + ("_case-" + "-".join(args.case) if args.case else ""))
    conf = {"variant": variant, "tier": tier, "episodes": args.episodes, "seed": args.seed, "url": args.url,
            "track": args.track, "finger_friction": args.finger_friction, "finger_effort": args.finger_effort, "case": args.case}
    path = write_report(outcomes, out_dir, f"{variant} / {tier} (ws_loop)", conf)
    lat = 1000 * np.asarray(policy.latency) if policy.latency else np.zeros(1)
    rep = {"calls": len(policy.latency), "latency_ms": {"mean": round(float(lat.mean()), 1), "p50": round(float(np.percentile(lat, 50)), 1),
                                                       "p99": round(float(np.percentile(lat, 99)), 1), "max": round(float(lat.max()), 1)},
           "wall_s": round(time.time() - t0, 1)}
    (out_dir / "ws_loop.json").write_text(json.dumps(rep, indent=1))
    summary[f"{variant}/{tier}"] = {**{k: aggregate(outcomes)[k]["value"] for k in ("TSR", "GSR", "PSR", "LHPR", "ODR")}, **rep}
    print(f"[ws_loop] wrote {path} " + json.dumps(rep), flush=True)
policy.close()
print("REPORT_JSON " + json.dumps(summary), flush=True)
os._exit(0)
