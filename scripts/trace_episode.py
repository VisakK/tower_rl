"""Run the teacher on one batch and record one env: a per-call trace (executor node, pinch, controller target, joints,
gripper, every movable block; ``--out``), a video of the four evaluator cameras (``--video``), or a live view in the
Isaac Sim window (``--live``, without ``--headless``).

To diagnose a failure signature from feedback.md the batch must match the rehearsal's (same variant, tier, seed,
num_envs, bundle and, for batch b > 0, the b earlier batches via --prior_batches) so the env reproduces; CPU physics.
Example (feedback.md example (seed 1000003, ...) -> rehearsal --seed 1000, batch 0, env 3):
  python scripts/trace_episode.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --tier D1 \
      --seed 1000 --env 3 --bundle results/bundles/ablate_gl200_graphs_v2 --out results/traces/t22_1000003.jsonl
Watch the teacher solve a task (one env; video frames every 0.1 s row, real time at 10 fps):
  python scripts/trace_episode.py --headless --device cpu --variant task2-3_cross_tower_transfer_3 --tier D0 \
      --num_envs 1 --seed 1 --bundle results/bundles/teacher_v6 --video results/videos/teacher_t23.mp4 --cam_scale 1.0
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", required=True)
parser.add_argument("--tier", default="D1")
parser.add_argument("--seed", type=int, default=1000, help="batch seed (rehearse.py --seed + 100 * batch)")
parser.add_argument("--prior_batches", type=int, default=0,
                    help="replay the rehearsal's earlier batches first (seeds seed - 100 * k): PhysX keeps state across "
                         "batch resets, so batch b > 0 of a rehearsal reproduces only after the same b batches")
parser.add_argument("--env", type=int, default=0, help="env index within the batch")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--bundle", default=None)
parser.add_argument("--rate_hz", type=int, choices=[10, 30], default=10)
parser.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0])
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--out", default=None, help="per-call trace (jsonl) of the env")
parser.add_argument("--fine", nargs=2, type=float, default=None, metavar=("T0", "T1"),
                    help="also record every 120 Hz substep in [T0, T1] s (blocks with velocities, all 10 DOFs per arm)")
parser.add_argument("--video", default=None, help="mp4 of the env's four evaluator cameras (high, low / left wrist, back "
                                                   "wrist), one frame per --video_stride rows (0.1 s each on the 10 Hz "
                                                   "track), 10 fps; RTX rendering: run while the GPU is otherwise idle. "
                                                   "Cameras perturb the simulation history, so a rendered rerun need not "
                                                   "repeat a rehearsal's failure exactly")
parser.add_argument("--video_stride", type=int, default=1, help="rows per video frame (1 = real time; 10 = 10x time-lapse)")
parser.add_argument("--cam_scale", type=float, default=0.5, help="camera resolution as a fraction of 640 x 480")
parser.add_argument("--live", action="store_true", help="render the stage after every row: run without --headless to "
                                                        "watch in the Isaac Sim window (GPU rendering at every row)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

# the executor's IK / FK / policy calls are tiny tensors: torch's default intra-op threads (one per core) cost ~6
# cores per process and made rehearsals slower (22.5 s -> 15.2 s per Task 3-2 batch with 1 thread)
torch.set_num_threads(1)

from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.domain import DomainSampler  # noqa: E402
from tower_rl.eval.runner import run_batch  # noqa: E402
from tower_rl.graph.executor import Executor  # noqa: E402
from tower_rl.graph.interface import ARMS  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402


class TracingExecutor:
    """Wraps an Executor; records the requesting state and the node of env ``env`` on every act call."""

    def __init__(self, ex: Executor, env: int, out):
        self.ex, self.env, self.out = ex, env, out

    def reset(self, n_envs, variant, seeds):
        self.ex.reset(n_envs, variant, seeds)

    def episode_log(self, env):
        return self.ex.episode_log(env)

    def act(self, states, env_ids):
        rows = self.ex.act(states, env_ids)
        for s, e, r in zip(states, env_ids, rows):
            if e != self.env:
                continue
            run = self.ex.runs[e]
            node = run.node
            spec = run.g["nodes"][min(run.i, len(run.g["nodes"]) - 1)]
            rec = {"t": round(s.t, 3), "node": spec["id"], "cls": type(node).__name__ if node else None,
                   "phase": getattr(node, "phase", None), "k": getattr(node, "k", None), "rows": len(r),
                   "q": [list(map(float, s.q[i])) for i in range(len(ARMS))], "width": list(map(float, s.width)),
                   "row_last": list(map(float, r[-1]))}
            for a in ARMS:
                pp, pq = run.pinch(s, a)
                rec[f"pinch_{a}"] = [float(x) for x in pp[0]] + [float(x) for x in pq[0]]
                rec[f"target_{a}"] = [float(x) for x in run.ctrl[a].pos[0]] + [float(x) for x in run.ctrl[a].quat[0]]
                rec[f"closed_{a}"] = bool(run.grip[a].closed[0])
            rec["blocks"] = {b: [float(x) for x in s.block(b)[0]] + [float(x) for x in s.block(b)[1]] for b in run.g["movable"]}
            self.out.write(json.dumps(rec) + "\n")
        return rows


phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(make_station_scene_cfg(args.variant, args.num_envs, cameras=bool(args.video), cam_scale=args.cam_scale))
sim.reset()
world = StationWorld(scene, sim, args.variant, finger_friction=tuple(args.finger_friction))
ex = Executor.from_bundle(args.bundle, args.variant, device=str(world.dev), rate_hz=args.rate_hz)
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
video = None
if args.video:
    import imageio
    import numpy as np
    from PIL import Image, ImageDraw

    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    video = imageio.get_writer(args.video, fps=10, macro_block_size=8)
if args.live and not args.headless:  # an overview of the station in the GUI viewport
    sim.set_camera_view(eye=[1.45, 0.55, 0.95], target=[0.33, -0.38, 0.05])
if args.fine:
    fine_out = open(Path(args.out or "trace.jsonl").with_suffix(".fine.jsonl"), "w")
    substep0 = world._substep
    tick = {"t": None, "k": 0}

    def substep_logged(q16):
        substep0(q16)
        if tick["t"] != world.time:  # world.time advances once per tick (after its substeps)
            tick["t"], tick["k"] = world.time, 0
        tick["k"] += 1
        t = world.time + tick["k"] * world.dt
        if not args.fine[0] <= t <= args.fine[1]:
            return
        e = args.env
        pos, quat = world.block_pose()
        vel = world.blocks.data.object_link_lin_vel_w
        avel = world.blocks.data.object_link_ang_vel_w
        rec = {"t": round(t, 5), "cmd": [float(x) for x in q16[e]]}
        for a in ARMS:
            rec[f"dof_{a}"] = [float(x) for x in world.scene[a].data.joint_pos[e]]
            rec[f"dofvel_{a}"] = [float(x) for x in world.scene[a].data.joint_vel[e]]
        rec["blocks"] = {b: [float(x) for x in pos[e, i]] + [float(x) for x in quat[e, i]] + [float(x) for x in vel[e, i]]
                         + [float(x) for x in avel[e, i]] for i, b in enumerate(world.block_ids)}
        fine_out.write(json.dumps(rec) + "\n")

    world._substep = substep_logged
for k in range(args.prior_batches, 0, -1):  # the same simulator history as the rehearsal process
    run_batch(world, ex, DomainSampler(args.tier, device=world.dev), seed=args.seed - 100 * k, max_time_s=args.max_time_s,
              rate_hz=args.rate_hz)
if video is not None or args.live:  # render after every row of the traced batch (not of the prior batches)
    apply_row0 = world.apply_row
    rows_done = [0]

    def frame():
        e = args.env
        cams = []
        for key in ("cam_high", "cam_low", "cam_left_wrist", "cam_back_wrist"):
            cam = scene[key]
            cam.update(0.0, force_recompute=True)
            img = cam.data.output["rgb"][e, ..., :3].cpu().numpy()
            cams.append(np.rot90(img, 2) if key in ("cam_high", "cam_low") else img)  # static cams: upright
        img = Image.fromarray(np.concatenate([np.concatenate(cams[:2], 1), np.concatenate(cams[2:], 1)], 0))
        run = ex.runs[e]
        node = run.g["nodes"][min(run.i, len(run.g["nodes"]) - 1)]["id"] if run.i < len(run.g["nodes"]) else "done"
        last = run.events[-1] if run.events else None
        txt = f"t {world.time:6.1f} s   {node}" + (f"   last: {last[1]} {last[2]}" if last else "")
        ImageDraw.Draw(img).text((6, 6), txt, fill=(255, 255, 0))
        video.append_data(np.asarray(img))

    def apply_row_viewed(q16, on_tick=None, ticks=None):
        apply_row0(q16, on_tick=on_tick, ticks=ticks)
        rows_done[0] += 1
        if args.live:
            sim.render()
        if video is not None and rows_done[0] % args.video_stride == 0:
            if not args.live:
                sim.render()
            frame()

    world.apply_row = apply_row_viewed
with open(args.out or os.devnull, "w") as f:
    outs = run_batch(world, TracingExecutor(ex, args.env, f), DomainSampler(args.tier, device=world.dev), seed=args.seed,
                     max_time_s=args.max_time_s, rate_hz=args.rate_hz)
if video is not None:
    video.close()
o = outs[args.env]
print("TRACE_RESULT " + json.dumps({"seed": o.extra["seed"], "case": o.extra["case"], "success": o.task_success,
                                    "grasp_ok": o.grasp_ok, "place_ok": o.place_ok, "events": o.extra.get("policy_log")}), flush=True)
os._exit(0)
