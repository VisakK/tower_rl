"""Step 7.5: rehearse the teacher (executor + skills) in the station scene and write feedback.md per variant/tier.

CPU physics (evaluator-faithful). Example:
  python scripts/rehearse.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --tiers D0 D1 --episodes 20 --num_envs 4
One variant per process (Isaac stages are not rebuilt in-process); loop in the shell for several.
Outputs: results/rehearsal/<variant>/<tier>/{feedback.md, metrics.json, episodes.json}
Watch the pose estimator drive the teacher (one mp4 per episode: the four cameras with the estimated blocks):
  python scripts/rehearse.py --headless --device cpu --variant task2-3_cross_tower_transfer_3 --tiers D0 --episodes 1 \
      --num_envs 1 --seed 7000 --bundle results/bundles/teacher_t45_final \
      --perception results/perception/models/kp_v2/model_last.pt --video results/perception/videos
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", required=True)
parser.add_argument("--tiers", nargs="+", default=["D0", "D1"])
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--bundle", default=None, help="teacher bundle (graphs/ + skills/); default: in-repo graphs, planned skills")
parser.add_argument("--rate_hz", type=int, choices=[10, 30], default=10)
parser.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0])
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--perception_noise", action="store_true", help="Step 8.1 provisional noise on block poses")
parser.add_argument("--noise", default="{}", help='JSON NoiseCfg overrides, e.g. {"bias_pos_m": 0.001, "shared_pos_m": 0.003}')
parser.add_argument("--graph_set", default="{}", help='JSON merged into the graph\'s top level, e.g. {"perception": {"jaw_to_goal": 1.0}}')
parser.add_argument("--out_root", default=None)
parser.add_argument("--perception", default=None, help="keypoint-network checkpoint: the executor sees block poses estimated "
                                                       "from the four rendered evaluator cameras (tower_rl.perception); "
                                                       "'oracle': heatmaps from the true state (no rendering, CPU)")
parser.add_argument("--perception_device", default="cuda")
parser.add_argument("--video", default=None, help="with --perception <model.pt>: one mp4 per episode in this directory, a "
                                                  "frame per estimator call (the four camera images, every block's "
                                                  "estimated pose, the truth where it is > 2 mm off; "
                                                  "tower_rl.perception.video); names end in _success / _fail")
parser.add_argument("--video_maps", action="store_true", help="videos also show the network's edge map (blue) and "
                                                             "corner peaks (yellow)")
parser.add_argument("--live", action="store_true", help="render the stage after every row: run without --headless to "
                                                        "watch in the Isaac Sim window (use --num_envs 1)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cameras = bool(args.perception) and args.perception not in ("oracle", "truth")
if args.video and not cameras:
    parser.error("--video needs --perception <checkpoint> (the rendered cameras)")
if cameras:
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
from tower_rl.eval.metrics import aggregate  # noqa: E402
from tower_rl.eval.perception_noise import NoiseCfg  # noqa: E402
from tower_rl.eval.runner import run_batch  # noqa: E402
from tower_rl.graph.executor import Executor  # noqa: E402
from tower_rl.graph.rehearsal import write_report  # noqa: E402
from tower_rl.paths import RESULTS_DIR  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402

phys = load_yaml("physics")
cfg = sim_cfg(phys, device=args.device)
if cameras:
    from tower_rl.scene import EVAL_RENDER_CFG

    cfg.render = sim_utils.RenderCfg(**EVAL_RENDER_CFG)
sim = sim_utils.SimulationContext(cfg)
summary = {}
for variant in [args.variant]:
    scene = InteractiveScene(make_station_scene_cfg(variant, args.num_envs, cameras=cameras))
    sim.reset()
    world = StationWorld(scene, sim, variant, finger_friction=tuple(args.finger_friction))
    policy = Executor.from_bundle(args.bundle, variant, device=str(world.dev), rate_hz=args.rate_hz)
    policy.graph.update(json.loads(args.graph_set))
    observe = None
    if args.perception:
        from tower_rl.perception.closed_loop import CameraPerception

        observe = CameraPerception(args.perception, args.num_envs, device=args.perception_device)
        observe.policy = policy
        if args.video:
            from tower_rl.perception.video import EpisodeVideos

            observe.video = EpisodeVideos(args.video, maps=args.video_maps)
    if args.live:  # env 0 from inside the curtain enclosure (play_episode.py's view), rendered after every row
        if not args.headless:
            o = scene.env_origins[0].cpu().numpy()
            sim.set_camera_view(eye=(o + [0.70, 0.00, 1.10]).tolist(), target=(o + [0.30, -0.40, 0.05]).tolist())
        apply_row0 = world.apply_row

        def apply_row_viewed(q16, on_tick=None, ticks=None):
            apply_row0(q16, on_tick=on_tick, ticks=ticks)
            sim.render()

        world.apply_row = apply_row_viewed
    for tier in args.tiers:
        if observe is not None:
            observe.log = []
            if observe.video is not None:
                observe.video.tier = tier
        sampler = DomainSampler(tier, device=world.dev, max_blocks=max(6, world.spec.num_movable))
        outcomes = []
        t0 = time.time()
        nb = (args.episodes + args.num_envs - 1) // args.num_envs
        for b in range(nb):
            outs = run_batch(world, policy, sampler, seed=args.seed + 100 * b, max_time_s=args.max_time_s, rate_hz=args.rate_hz,
                             noise=NoiseCfg(**json.loads(args.noise)) if args.perception_noise else None, observe=observe)
            outcomes += outs[: args.episodes - b * args.num_envs]
            if observe is not None and observe.video is not None:
                observe.video.close(outs)
            done = aggregate(outcomes)
            print(f"[rehearse] {variant} {tier} batch {b + 1}/{nb}: TSR {done['TSR']['value']:.2f} GSR {done['GSR']['value']:.2f} "
                  f"PSR {done['PSR']['value']:.2f} ODR {done['ODR']['value']:.2f} ({time.time() - t0:.0f} s)", flush=True)
        out_dir = Path(args.out_root or RESULTS_DIR / "rehearsal") / variant / (tier + (("_noise" + ("_" + "_".join(f"{k}{v}" for k, v in sorted(json.loads(args.noise).items())) if json.loads(args.noise) else "")) if args.perception_noise else "")
                                                                                 + ((f"_perception_{args.perception}" if args.perception in ("oracle", "truth") else "_perception") if args.perception else ""))
        cfg = {"variant": variant, "tier": tier, "episodes": args.episodes, "seed": args.seed, "bundle": args.bundle,
               "perception_noise": args.perception_noise, "noise": json.loads(args.noise),
               "rate_hz": args.rate_hz, "finger_friction": args.finger_friction, "device": args.device,
               "graph_set": json.loads(args.graph_set), "perception": args.perception}
        path = write_report(outcomes, out_dir, f"{variant} / {tier}", cfg)
        if observe is not None:
            (out_dir / "perception.json").write_text(json.dumps(observe.summary(), indent=1))
            observe.save(out_dir / "perception_log.pkl")
            print("[rehearse] perception " + json.dumps(observe.summary()), flush=True)
        summary[f"{variant}/{tier}"] = {k: aggregate(outcomes)[k]["value"] for k in ("TSR", "GSR", "PSR", "LHPR", "ODR")}
        print(f"[rehearse] wrote {path}", flush=True)
print("REPORT_JSON " + json.dumps(summary), flush=True)
os._exit(0)
