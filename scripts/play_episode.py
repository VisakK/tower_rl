"""Visualize the TOWER scene and play a recorded episode open-loop in Isaac Sim.

Modes:
  replay   : physics replay of the recorded joint commands (leader/qpos at 30 Hz), as the evaluator executes actions
  recorded : the recorded robot and block states themselves (kinematic playback, ground truth)
  compare  : replay in env 0 and the recording in env 1, side by side (--open_view implied)
  hold     : the scene at --start_frame with the arms holding their pose and the blocks free

Examples (repo root, Isaac Lab venv active):
  python scripts/play_episode.py --episode t23_ep1                    # GUI window, real time
  python scripts/play_episode.py --episode t23_ep1 --mode compare
  python scripts/play_episode.py --episode t13_ep --headless --video results/videos/t13_replay.mp4
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

from tower_rl.paths import STATION_JSON, resolve_episode  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episode", default="t23_ep1", help="episode name in data/episodes (e.g. t13_ep) or npz path")
parser.add_argument("--attrs", default=None, help="attrs json of the episode (default: <npz stem>_attrs.json)")
parser.add_argument("--mode", choices=["replay", "recorded", "compare", "hold"], default="replay")
parser.add_argument("--station_json", default=str(STATION_JSON))
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--end_frame", type=int, default=None, help="last frame to play (default: end of episode)")
parser.add_argument("--speed", type=float, default=1.0, help="GUI playback speed vs real time; 0 = as fast as possible")
parser.add_argument("--loop", action="store_true", help="restart at --start_frame when the episode ends")
parser.add_argument("--max_frames", type=int, default=None, help="stop after this many 30 Hz control steps")
parser.add_argument("--finger_friction", nargs=2, type=float, default=[1.2, 1.0], help="finger static/dynamic friction")
parser.add_argument("--block_friction", nargs=2, type=float, default=None, help="override block static/dynamic friction")
parser.add_argument("--delay_substeps", nargs=2, type=int, default=[1, 1], help="command delay (left, back) in 120 Hz substeps")
parser.add_argument("--open_view", action="store_true", help="hide the curtains and station frame (implied by compare mode)")
parser.add_argument("--dome_light", type=float, default=None, help="dome light intensity (default: 800 with --open_view, else none)")
parser.add_argument("--eye", nargs=3, type=float, default=[0.70, 0.00, 1.10], help="viewer camera position (env frame, m)")
parser.add_argument("--target", nargs=3, type=float, default=[0.30, -0.40, 0.05], help="viewer camera look-at point (env frame, m)")
parser.add_argument("--video", default=None, help="write an mp4 of the viewer camera (enables cameras)")
parser.add_argument("--video_size", nargs=2, type=int, default=[1280, 720])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from pxr import UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

from tower_rl.scene import (  # noqa: E402
    CONTROL_DT,
    CONTROL_SUBSTEPS,
    EVAL_RENDER_CFG,
    OCCLUDERS,
    PHYSICS_DT,
    DelayedCommands,
    Episode,
    apply_friction,
    load_station_meta,
    make_scene_cfg,
    measured_q16,
    set_arm_targets,
    write_recorded_state,
)

gui = not args.headless
meta = load_station_meta(args.station_json)
episode = Episode.load(*resolve_episode(args.episode, args.attrs))
compare = args.mode == "compare"

extras = {}
if args.video:
    extras["viewer_cam"] = CameraCfg(
        prim_path="/World/ViewerCam", update_period=0.0, width=args.video_size[0], height=args.video_size[1], data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=24.0, clipping_range=(0.01, 30.0)),
    )
open_view = args.open_view or compare
dome_light = args.dome_light if args.dome_light is not None else (800.0 if open_view else 0.0)
if dome_light > 0:
    extras["dome"] = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=dome_light))

render_cfg = sim_utils.RenderCfg(**EVAL_RENDER_CFG) if (gui or args.video) else sim_utils.RenderCfg()
sim = sim_utils.SimulationContext(
    sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=args.device, physx=sim_utils.PhysxCfg(), render=render_cfg)
)
scene = InteractiveScene(make_scene_cfg(meta, episode, num_envs=2 if compare else 1, extras=extras))
if open_view:
    for pattern in OCCLUDERS:
        for prim in sim_utils.find_matching_prims(pattern):
            UsdGeom.Imageable(prim).MakeInvisible()
sim.reset()
apply_friction(scene, finger=args.finger_friction, block=args.block_friction)
dev = sim.device

# viewer: inside the curtain enclosure for one env, from the front above both worktops for the comparison
origins = scene.env_origins
if compare:
    target = origins.mean(0) + torch.tensor([0.48, -0.33, 0.05], device=dev)  # worktop center
    eye = target + torch.tensor([0.0, 1.8, 3.3], device=dev)
else:
    eye = origins[0] + torch.tensor(args.eye, device=dev)
    target = origins[0] + torch.tensor(args.target, device=dev)
sim.set_camera_view(eye.tolist(), target.tolist())
if args.video:
    scene["viewer_cam"].set_world_poses_from_view(eye[None], target[None])

lq = torch.tensor(episode.arrays["leader__qpos"], dtype=torch.float32, device=dev)
phases, active = episode.arrays["sim__task_phase"], episode.arrays["sim__active_block"]
end = min(args.end_frame if args.end_frame is not None else episode.num_frames - 2, episode.num_frames - 2)
replay_ids = {"replay": None, "compare": torch.tensor([0], device=dev)}.get(args.mode, False)  # None = all envs
recorded_ids = {"recorded": None, "compare": torch.tensor([1], device=dev)}.get(args.mode, False)
max_frames = args.max_frames or (150 if (args.mode == "hold" and not gui) else None)


def start_episode(t0: int) -> DelayedCommands:
    write_recorded_state(scene, episode, t0)
    return DelayedCommands(lq[t0], tuple(args.delay_substeps))


writer = None
if args.video:
    import imageio

    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.video, fps=round(1.0 / CONTROL_DT), codec="libx264", quality=8)

print(f"[play] {episode.task} | {episode.instruction}", flush=True)
sides = ""
if compare:  # the viewer looks along -y, so larger x is further left in the image
    left_env = 0 if origins[0, 0] > origins[1, 0] else 1
    sides = f" | replay (env 0) on the {'left' if left_env == 0 else 'right'}, recording (env 1) on the {'right' if left_env == 0 else 'left'}"
print(f"[play] mode={args.mode} frames {args.start_frame}..{end} ({(end - args.start_frame) * CONTROL_DT:.0f} s){sides}", flush=True)
t = args.start_frame
commands = start_episode(t)
finished = args.mode == "hold"
steps = 0
try:
    while simulation_app.is_running() and not sim.is_stopped():
        tic = time.time()
        for k in range(CONTROL_SUBSTEPS):
            if not finished:
                if replay_ids is not False:
                    set_arm_targets(scene, commands.push(lq[t]), env_ids=replay_ids)
                if recorded_ids is not False:
                    # the state recorded after this control interval, aligned with the replayed command t
                    write_recorded_state(scene, episode, t + 1, env_ids=recorded_ids)
            scene.write_data_to_sim()
            sim.step(render=k == CONTROL_SUBSTEPS - 1)
            scene.update(PHYSICS_DT)
        if writer is not None:
            writer.append_data(scene["viewer_cam"].data.output["rgb"][0, ..., :3].cpu().numpy())
        if not finished:
            if t % 30 == 0:
                blk = episode.block_ids[active[t]] if active[t] >= 0 else "-"
                msg = f"[play] t={t * CONTROL_DT:6.1f}s frame {t:5d}/{end}  phase={episode.phase_names[phases[t]]:9s} block={blk}"
                if compare:  # divergence of the physics replay (env 0) from the recording (env 1)
                    dq = (measured_q16(scene, 0) - measured_q16(scene, 1)).abs()
                    bpos = scene["blocks"].data.object_link_pos_w - scene.env_origins[:, None, :]
                    msg += f"  | replay-recorded: arm joints max {dq[[*range(7), *range(8, 15)]].max():.4f} rad, blocks max {1000 * (bpos[0] - bpos[1]).norm(dim=-1).max():.1f} mm"
                print(msg, flush=True)
            t += 1
            if t > end:
                if args.loop:
                    t = args.start_frame
                    commands = start_episode(t)
                elif gui:
                    finished = True
                    print("[play] episode finished: holding the last commands, close the window to exit", flush=True)
                else:
                    break
        steps += 1
        if max_frames is not None and steps >= max_frames:
            break
        if gui and args.speed > 0:
            time.sleep(max(0.0, CONTROL_DT / args.speed - (time.time() - tic)))
finally:
    if writer is not None:
        writer.close()
        print(f"[play] wrote {args.video} ({steps} frames)", flush=True)
# Kit can hang in SimulationApp.close()
os._exit(0)
