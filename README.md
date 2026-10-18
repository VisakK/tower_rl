# tower_rl

Isaac Lab recreation of the [TOWER CoRL 2026 challenge](https://challenge.tower-benchmark.com/docs/) evaluator
(two AgileX Nero arms, Jenga towers) as the base for a reinforcement-learning approach.

## Setup

All commands run from the repository root with the Isaac Lab virtual environment active:

```bash
source ~/Documents/moves/env_isaaclab/bin/activate   # Isaac Sim 5.1.0 (pip) + Isaac Lab 2.3.0
cd ~/Documents/tower_rl
```

`data/`, `assets_m/` and `results/` are not in git. On a fresh clone, download the assets, the dataset metadata and
three example episodes (non-image arrays only, ~5 s each), then build the meter-unit wrappers:

```bash
python -I -c "
from huggingface_hub import snapshot_download
snapshot_download('tower-benchmark/TOWER-Assets', repo_type='dataset', local_dir='data/TOWER-Assets',
                  revision='538d8b1fb582a4c3fc8c5a2fb2bf3a82931e9a4c')
snapshot_download('tower-benchmark/TOWER-SimData', repo_type='dataset', local_dir='data/SimData-meta',
                  revision='1a59248e44866b1593e4ac8b851aa519c013220f',
                  allow_patterns=['README.md', 'schema/*', 'MANIFEST.jsonl', 'CHECKSUMS.sha256', 'CITATION.cff', 'LICENSE', 'previews/*'])
"
python -I scripts/peek_hdf5_remote.py data/task2-3_cross_tower_transfer_3/episode_000001_20260910T024604Z_2a92f2b7.hdf5 data/episodes/t23_ep1.npz
python -I scripts/peek_hdf5_remote.py data/task1-3_tower_transfer_3/episode_001022_20260909T153526Z_254a6af4.hdf5 data/episodes/t13_ep.npz
python -I scripts/peek_hdf5_remote.py data/task3-3_vertical_stacking_3/episode_000001_20260910T023237Z_72e36c99.hdf5 data/episodes/t33_ep.npz
python -I scripts/make_meter_assets.py data/TOWER-Assets assets_m station.usdz
```

The wrappers in `assets_m/` reference `data/TOWER-Assets` by absolute path: rerun the last command after moving the
repository.

## Visualize the scene

`scripts/play_episode.py` builds the scene for one recorded episode and opens the Isaac Sim window (startup takes about
a minute). Playback runs in real time; the terminal prints the task phase every simulated second.

```bash
# open-loop replay: the recorded 30 Hz joint commands drive the simulated arms, blocks are fully simulated
python scripts/play_episode.py --episode t23_ep1

# the recorded states themselves (ground truth, kinematic)
python scripts/play_episode.py --episode t23_ep1 --mode recorded

# replay (env 0, left) next to the recording (env 1, right); the terminal also prints their divergence
python scripts/play_episode.py --episode t23_ep1 --mode compare

# just the scene at a frame: arms hold their pose, blocks are free (they creep slightly, as in the evaluator)
python scripts/play_episode.py --episode t33_ep --mode hold --start_frame 0
```

Episodes in `data/episodes/`: `t23_ep1` (Task2-3 cross tower, 94 s), `t13_ep` (Task1-3 tower transfer, 178 s),
`t33_ep` (Task3-3 vertical stacking, 87 s). Useful options:

| Option | Effect |
|---|---|
| `--start_frame N`, `--end_frame N` | play a segment (30 frames per second) |
| `--speed 2` | 2x real time (`0` = as fast as possible) |
| `--loop` | restart at `--start_frame` when the episode ends |
| `--open_view` | hide the curtains and station frame; adds a dome light (`--dome_light 0` to remove) |
| `--eye x y z --target x y z` | initial viewpoint in the station frame (meters) |
| `--finger_friction S D`, `--delay_substeps L B` | replay physics (defaults 1.2 1.0 and 1 1 best match the data) |

In the window: right-drag looks around, right-drag + `W A S D` flies, `Alt` + left-drag orbits, the scroll wheel
zooms, `F` frames the prim selected in the Stage panel. Pause/Play in the toolbar pauses the playback; when the
episode ends the arms hold their last command until the window is closed.

Record a video without a window:

```bash
python scripts/play_episode.py --episode t13_ep --headless --video results/videos/t13_replay.mp4
python scripts/play_episode.py --episode t23_ep1 --mode compare --start_frame 250 --max_frames 900 \
    --headless --video results/videos/t23_compare.mp4
```

To look at another episode, pick its path in `data/SimData-meta/MANIFEST.jsonl`, fetch it, and pass its name:

```bash
python -I scripts/peek_hdf5_remote.py data/task3-2_vertical_stacking_2/<episode file>.hdf5 data/episodes/t32_ep.npz
python scripts/play_episode.py --episode t32_ep
```

## Validate

`scripts/validate_isaaclab_scene.py` checks the scene against recorded data (`--help` lists the modes):

```bash
# kinematics: recorded end-effector pose == link7 in base_link (offset std ~0.0001 mm)
python scripts/validate_isaaclab_scene.py --headless --mode fk --num_envs 1 --episode t23_ep1 --out /tmp/fk.json

# open-loop replay with block-trajectory errors against the recording
python scripts/validate_isaaclab_scene.py --headless --mode replay --num_envs 1 --episode t33_ep \
    --finger_friction 1.2 1.0 --delay_substeps 1 1 --out /tmp/replay_t33.json

# physics throughput, and rendering throughput of the four evaluator cameras
python scripts/validate_isaaclab_scene.py --headless --mode bench --num_envs 1024 --episode t23_ep1 --out /tmp/bench.json
python scripts/validate_isaaclab_scene.py --headless --enable_cameras --mode render_bench --num_envs 16 --cam_scale 0.25 \
    --episode t23_ep1 --out /tmp/rbench.json
```

## Development

### Layout

| Path | Contents |
|---|---|
| `tower_rl/paths.py` | Repository paths and episode lookup; no Isaac imports, usable before the app starts |
| `tower_rl/scene.py` | Scene config (station, arms with evaluator actuators, blocks, evaluator cameras) and playback helpers |
| `scripts/play_episode.py` | Visualization and open-loop playback |
| `scripts/validate_isaaclab_scene.py` | Validation and benchmarks (`inspect`, `fk`, `replay`, `bench`, `render`, `render_bench`) |
| `scripts/make_meter_assets.py` | Builds the meter-unit station and block wrappers in `assets_m/` |
| `scripts/smoke_native_isaacsim.py` | Opens a published task layer natively (mm stage) in plain Isaac Sim |
| `scripts/peek_hdf5_remote.py`, `fetch_frames_remote.py`, `extract_meta_remote.py` | Read TOWER-SimData over HTTP range requests (one episode's arrays, single JPEG frames, metadata of many episodes) |
| `scripts/inspect_station.py`, `inspect_xforms.py` | Static USD inspection with plain `pxr` |
| `data/`, `assets_m/`, `results/` | Downloads, generated wrappers, run outputs (git-ignored) |

Scripts put the repository root on `sys.path`, so nothing needs to be pip-installed. Isaac Lab modules (and therefore
`tower_rl.scene`) can only be imported after `AppLauncher` has started the app; follow the pattern at the top of
`scripts/play_episode.py` in new scripts.

### Conventions

- Meters, radians, seconds. Scene frame = station frame: tabletop surface at z = 0, worktop spans x 0..0.96,
  y -0.66..0; robot roots at left (-0.1165, -0.288, 0) and back (0.381, -0.774, 0).
- Quaternions: the dataset stores `x y z qx qy qz qw`, Isaac Lab uses `(w, x, y, z)`.
- Arm order is always `left` (station `StandA_Robot`) then `back` (`StandB_Robot`). A recorded arm state is 8 values
  (7 joints + gripper opening width in m); `q16` stacks both arms. Each simulated arm has 10 DOFs: 7 joints, a passive
  `gripper` joint and two fingers driven to +-width/2 (`tower_rl.scene.arm_q_full`).
- Timing: 120 Hz physics, 30 Hz commands held for 4 substeps; the challenge executes 10 Hz or 30 Hz action chunks.
- Drive parameters come from the episode metadata (`Episode.readback`, identical in all 331 Task1-3 episodes) and are
  stored in millimetre-stage units: `actuators_from_readback` multiplies angular quantities by 1e-6 and linear forces
  and velocities by 1e-3.
- Files under `data/` are downloads: run scripts that parse them directly with `python -I`.

### Checking changes

After changing the scene, assets or physics, rerun the two validation commands above and compare with these
reference values (RTX A5000, deterministic on the same machine):

- `fk`: `link7` offset std <= 0.0001 mm for both arms over 61 frames.
- `replay` on `t33_ep`: final block errors 1.04 / 0.24 / 0.02 mm (`source_l1/l2/l3_center`), distances to goal
  4.01 / 4.79 / 3.38 mm.

### Kit notes

- `SimulationApp.close()` can hang or segfault after the work is done; scripts write their outputs and call
  `os._exit(0)`. Flush prints (`flush=True`) or they are lost.
- One Isaac Sim app per process. The first launch after a driver or extension change rebuilds shader caches and is
  slow.
- Cameras need `--enable_cameras` (`--video` sets it). Headless runs without cameras cannot apply RTX settings, which
  is why the render settings are only passed when rendering.
- Memory: 1024 physics-only envs run at ~116k physics steps/s; with the four evaluator cameras, 64 envs at 160x120
  fit and 128 run out of the 62 GB host RAM with the full-detail station.

## Findings that the environment must respect

- The station USDZ is in millimetres; robots and blocks inside carry a x1000 scale. Isaac Lab disables Kit's unit
  correction, so the wrapper applies an explicit 0.001 scale.
- Five prims use `!resetXformStack!` (both robot stands, `InstalledRobots`, curtains, lighting). Without the wrapper's
  replacement matrices, robot colliders are cooked 1000x too large and every env's robots sit at the world origin.
- Gripper prismatic limits are authored in world millimetres; the wrapper converts them to meters.
- The finger friction material lives outside the station's default prim, so references drop it. Replays match the
  data with finger friction ~1.2/1.0 (`--finger_friction`); the default 0.5 makes blocks slip.
- The block collision hull has two stray vertices, so towers rock and creep 1-3 mm; the evaluator data shows the same.
- Evaluator dynamics: physics 120 Hz, 30 Hz zero-order-hold commands, delays left 5 ms / back 10 ms (one substep fits
  best), finger PD 800 / 20 with a 22 N effort limit.
- `observations/eef` is the `link7` frame in the `base_link` frame. Cameras: `left_back_camera` = station
  `low_camera`, `right_front_camera` = `high_camera` (stored rotated 180 deg), wrist cameras on `gripper_base`.
- The challenge-task USDA files are runtime-generated stubs; layouts and goals come from the SimData metadata
  (`sim/block_poses`, `target_mapping`).
- Data revisions: TOWER-Assets `538d8b1`, TOWER-SimData `1a59248` (same files as the docs-pinned `5ad6adc9`).
