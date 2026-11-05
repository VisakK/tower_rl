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
parser.add_argument("--finger_effort", type=float, default=None, help="finger effort limit (N) for every arm; default: the "
                                                                    "layout's (Tasks 4 / 5: 3 N), else the robot model's 22 N")
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--perception_noise", action="store_true", help="Step 8.1 provisional noise on block poses")
parser.add_argument("--noise", default="{}", help='JSON NoiseCfg overrides, e.g. {"bias_pos_m": 0.001, "shared_pos_m": 0.003}')
parser.add_argument("--est_noise", default=None, help="estimator-like noise on the block poses (student plan Step 5): the "
                                                      "fitted model, e.g. configs/student/noise.yaml; writes perception.json "
                                                      "and perception_log.pkl per tier; dirs get the suffix _estnoise")
parser.add_argument("--graph_set", default="{}", help='JSON merged into the graph\'s top level, e.g. {"perception": {"jaw_to_goal": 1.0}}')
parser.add_argument("--out_root", default=None)
parser.add_argument("--student", default=None, help="run the student option runtime (tower_rl.student) with this decider "
                                                    "instead of the executor: oracle_graph | oracle_state | <switcher.pt>; "
                                                    "output dirs get the suffix _student-<name>")
parser.add_argument("--shadow", action="store_true", help="the executor drives the simulation and the policy (--student, "
                                                           "or a second executor as a control) gets the same states; reports "
                                                           "row differences and event equality per episode (parity gate G3a); "
                                                           "dirs get _shadow")
parser.add_argument("--collect", default=None, help="with --student: write switcher training shards under this dir "
                                                    "(<dir>/<variant>/<tier>/b<seed>.npz + <dir>/index.jsonl; labels from "
                                                    "the state-mode oracle; tower_rl.student.logging)")
parser.add_argument("--explore", type=float, default=0.0, help="with --student: probability of a random valid alternative "
                                                             "decision at each decision point (data collection)")
parser.add_argument("--inject", type=float, default=0.0, help="with --student: probability per goal option of a failure "
                                                             "injection (configs/student/collect.yaml inject; data collection)")
parser.add_argument("--beta", type=float, default=None, help="with --student <switcher.pt> --collect: the state-mode oracle "
                                                           "acts with this probability per decision (DAgger)")
parser.add_argument("--features", default="v1", help="with --student <oracle> --collect: the token version of the rows "
                                                     "(v2: the recovery node's contracts, labels from oracle_state_v2)")
parser.add_argument("--inject_set", default="{}", help='JSON merged into configs/student/collect.yaml inject per option, '
                                                       'e.g. {"push_off": {"stop_r_m": [0.125, 0.15], "short_frac": 0.35}}')
parser.add_argument("--distill_collect", default=None, help="student plan Step 9c: run the planned Task 5 nodes under DART "
                                                            "noise and write distillation shards under this dir "
                                                            "(tower_rl.student.distill)")
parser.add_argument("--dart_scale", type=float, default=1.5, help="with --distill_collect: each node's DART sigma is the "
                                                                "skill's base (DartCollector.SIGMA) times U(0, this)")
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
parser.add_argument("--mirror", action="store_true", help="the official mirror (Plan-v1.1 §2.2): every episode runs to "
                                                          "--max_time_s and is judged on its final arrangement; Task 4 "
                                                          "pushes judged on their rest states (tower_rl.eval.mirror); "
                                                          "dirs get _mirror")
parser.add_argument("--official_orient", action="store_true", help="with --perception <model.pt>: deliver the fixed "
                                                                    "cameras of Tasks 1-3 unrotated, as the official "
                                                                    "evaluator does, and let the adapter's orientation "
                                                                    "choice (tower_rl.perception.orient) turn them back")
parser.add_argument("--case", nargs="+", default=None, help="force the tier's robustness case(s) (configs/tiers.yaml; D1: "
                                                            "nominal lightweight heavy uneven_weights slippery high_friction "
                                                            "shifted_assembly irregular_layers lighting combined): episode j "
                                                            "of a tier gets the j-th, cycling; default: drawn from the seed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# many Kit processes at once exhaust the user's inotify watches (65536; GNOME's file indexer and VS Code hold ~58k) and a
# Kit started without a free watch aborts in omniclient's file thread (a camera-loop process, 2026-10-22): rehearsals do
# not watch the extension folders (a non-persistent /app setting; 384 -> 0 watches per process)
if "fsWatcherEnabled" not in (args.kit_args or ""):
    args.kit_args = ((args.kit_args or "") + " --/app/extensions/fsWatcherEnabled=0").strip()
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
from tower_rl.eval.operations import OpCfg  # noqa: E402
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
    scene = InteractiveScene(make_station_scene_cfg(variant, args.num_envs, cameras=cameras, finger_effort=args.finger_effort))
    sim.reset()
    world = StationWorld(scene, sim, variant, finger_friction=tuple(args.finger_friction))
    if args.student:
        from tower_rl.student.runtime import StudentExecutor, make_decider

        policy = StudentExecutor.from_bundle(args.bundle, variant, device=str(world.dev), rate_hz=args.rate_hz,
                                             decider=make_decider(args.student), collect=bool(args.collect),
                                             explore_eps=args.explore, beta=args.beta, p_inject=args.inject,
                                             features=args.features, inject_set=json.loads(args.inject_set))
    else:
        policy = Executor.from_bundle(args.bundle, variant, device=str(world.dev), rate_hz=args.rate_hz)
    policy.graph.update(json.loads(args.graph_set))
    if args.shadow:
        from tower_rl.student.runtime import ShadowPolicy

        primary = Executor.from_bundle(args.bundle, variant, device=str(world.dev), rate_hz=args.rate_hz)
        primary.graph.update(json.loads(args.graph_set))
        policy = ShadowPolicy(primary, policy, dump=os.environ.get("TOWER_SHADOW_DUMP"))
    collector = None
    if args.distill_collect:
        assert not args.shadow, "--distill_collect records the policy that drives the simulation (no --shadow)"
        from tower_rl.student.distill.nodes import DartCollector

        collector = DartCollector(seed=args.seed, max_scale=args.dart_scale)
        policy.node_hook = collector
    noise_model = None
    if args.est_noise:
        from tower_rl.student.noise import NoiseModel

        noise_model = NoiseModel.load(args.est_noise)
        noise_model.keep_log = not args.collect  # the per-call log is for calibration runs
    observe = None
    if args.perception:
        from tower_rl.perception.closed_loop import CameraPerception

        observe = CameraPerception(args.perception, args.num_envs, device=args.perception_device)
        observe.policy = policy
        if args.official_orient:
            from tower_rl.paths import RESULTS_DIR as _R
            from tower_rl.perception.orient import load_refs

            observe.set_official_orient(load_refs(Path(args.bundle or "") / "perception" / "orient_refs.npz")
                                        or load_refs(_R / "perception" / "orient_refs.npz"))
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
            observe.orient_log = []
            if observe.video is not None:
                observe.video.tier = tier
        sampler = DomainSampler(tier, device=world.dev, max_blocks=max(6, world.spec.num_movable))
        if args.case and set(args.case) - set(sampler.case_names):
            raise SystemExit(f"--case {sorted(set(args.case) - set(sampler.case_names))}: not a {tier} case "
                             f"(the {tier} cases: {' '.join(sampler.case_names)})")
        outcomes = []
        t0 = time.time()
        nb = (args.episodes + args.num_envs - 1) // args.num_envs
        for b in range(nb):
            if collector is not None:
                collector.reseed(args.seed + 100 * b, f"{variant}/{tier}")
            case = [args.case[(b * args.num_envs + e) % len(args.case)] for e in range(args.num_envs)] if args.case else None
            outs = run_batch(world, policy, sampler, seed=args.seed + 100 * b, max_time_s=args.max_time_s, rate_hz=args.rate_hz,
                             noise=noise_model or (NoiseCfg(**json.loads(args.noise)) if args.perception_noise else None),
                             observe=observe, case=case, full_episode=args.mirror,
                             op_cfg=OpCfg(push_rule="rest") if args.mirror else None)
            if args.collect:
                from tower_rl.student.logging import write_batch

                write_batch(args.collect, policy, outs[: args.episodes - b * args.num_envs], variant, tier, args.seed + 100 * b,
                            {"round": Path(args.collect).name, "decider": args.student, "bundle": args.bundle,
                             "beta": args.beta, "explore": args.explore, "inject": args.inject, "est_noise": args.est_noise,
                             "graph_set": json.loads(args.graph_set), "inject_set": json.loads(args.inject_set),
                             "mirror": args.mirror, "features": args.features})
            if collector is not None:
                from tower_rl.student.distill.data import write_batch as write_distill

                write_distill(args.distill_collect, collector, policy, outs[: args.episodes - b * args.num_envs], variant, tier,
                              args.seed + 100 * b, {"bundle": args.bundle, "dart_scale": args.dart_scale, "est_noise": args.est_noise,
                                                    "student": args.student, "graph_set": json.loads(args.graph_set)})
            outcomes += outs[: args.episodes - b * args.num_envs]
            if observe is not None and observe.video is not None:
                observe.video.close(outs)
            done = aggregate(outcomes)
            print(f"[rehearse] {variant} {tier} batch {b + 1}/{nb}: TSR {done['TSR']['value']:.2f} GSR {done['GSR']['value']:.2f} "
                  f"PSR {done['PSR']['value']:.2f} ODR {done['ODR']['value']:.2f} ({time.time() - t0:.0f} s)", flush=True)
        out_dir = Path(args.out_root or RESULTS_DIR / "rehearsal") / variant / (tier + (("_noise" + ("_" + "_".join(f"{k}{v}" for k, v in sorted(json.loads(args.noise).items())) if json.loads(args.noise) else "")) if args.perception_noise else "")
                                                                                 + ((f"_perception_{args.perception}" if args.perception in ("oracle", "truth") else "_perception") if args.perception else "")
                                                                                 + ("_estnoise" if args.est_noise else "")
                                                                                 + (f"_student-{Path(args.student).stem}" if args.student else "")
                                                                                 + ("_shadow" if args.shadow else "")
                                                                                 + ("_mirror" if args.mirror else "")
                                                                                 + ("_orient" if args.official_orient else "")
                                                                                 + ("_case-" + "-".join(args.case) if args.case else ""))
        cfg = {"variant": variant, "tier": tier, "episodes": args.episodes, "seed": args.seed, "bundle": args.bundle,
               "perception_noise": args.perception_noise, "noise": json.loads(args.noise),
               "rate_hz": args.rate_hz, "finger_friction": args.finger_friction, "finger_effort": args.finger_effort,
               "device": args.device,
               "graph_set": json.loads(args.graph_set), "perception": args.perception, "student": args.student,
               "collect": args.collect, "explore": args.explore, "beta": args.beta, "est_noise": args.est_noise, "inject": args.inject, "shadow": args.shadow,
               "distill_collect": args.distill_collect, "dart_scale": args.dart_scale if args.distill_collect else None,
               "case": args.case, "mirror": args.mirror, "official_orient": args.official_orient}
        path = write_report(outcomes, out_dir, f"{variant} / {tier}", cfg)
        if noise_model is not None and noise_model.keep_log:
            import pickle

            (out_dir / "perception.json").write_text(json.dumps(noise_model.summary(), indent=1))
            with open(out_dir / "perception_log.pkl", "wb") as f:
                pickle.dump({"block_ids": list(world.block_ids), "log": noise_model.log}, f)
            noise_model.log = []
        if observe is not None:
            (out_dir / "perception.json").write_text(json.dumps(observe.summary(), indent=1))
            observe.save(out_dir / "perception_log.pkl")
            print("[rehearse] perception " + json.dumps(observe.summary()), flush=True)
        summary[f"{variant}/{tier}"] = {k: aggregate(outcomes)[k]["value"] for k in ("TSR", "GSR", "PSR", "LHPR", "ODR")}
        if args.shadow:
            sh = [o.extra.get("shadow", {}) for o in outcomes]
            summary[f"{variant}/{tier}"]["shadow"] = {
                "episodes": len(sh), "rows_within_tol": sum(x.get("first_mismatch") is None for x in sh),
                "events_equal": sum(bool(x.get("events_equal")) for x in sh), "calls": sum(x.get("calls", 0) for x in sh),
                "inexact_calls": sum(x.get("inexact_calls", 0) for x in sh), "max_abs": max(x.get("max_abs", 0.0) for x in sh),
                "mismatches": [x["first_mismatch"] for x in sh if x.get("first_mismatch")][:5]}
            print("[rehearse] shadow " + json.dumps(summary[f"{variant}/{tier}"]["shadow"]), flush=True)
        print(f"[rehearse] wrote {path}", flush=True)
print("REPORT_JSON " + json.dumps(summary), flush=True)
os._exit(0)
