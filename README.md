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

## Agentic teacher (privileged)

Built from `~/Documents/corl_challenge/TOWER_BUILD_INSTRUCTIONS.md`; progress, acceptance numbers and every deviation
are in `BUILDLOG.md`. The teacher is a typed graph of skill calls per variant (`tower_rl/graph/graphs/*.json`)
interpreted by a deterministic executor: planned transits and parks (IK + gravity-sag compensation), and two
contact skills, `grasp_lift` and `place_release`, trained with PPO on privileged critics but camera-estimable actor
inputs. Until a skill is trained for an arm, the executor uses a planned fallback with the same checkpoints.

```bash
# Step 1: data subset (non-image arrays over HTTP range requests), layouts, events, grasp geometry
python -I scripts/fetch_subset.py --per_variant 10
python -I scripts/build_layouts.py            # configs/layouts/<variant>.yaml + results/layouts/*.png
python -I scripts/analyze_subset.py           # results/data/subset_analysis.json
python -I scripts/build_posture_priors.py     # configs/skills/posture_priors.yaml
python scripts/extract_robot_model.py         # configs/robot/nero.json (chain, inertials, evaluator drives)

# Step 2: scenes (light skill scene on GPU; station scene on CPU = evaluator-faithful)
python -m pytest --isaac tests/test_scenes.py
python scripts/replay_demo.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --num_replays 2 \
    --out results/calibration/replay_task2-2_cpu.json

# Step 3: local evaluator (metrics, operation tracker, chunked runner)
python scripts/evaluate_policy.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --policy replay \
    --rate_hz 30 --replay_episodes data/subset/task2-2_cross_tower_transfer_2/*.npz --finger_friction 1.2 1.0
python scripts/sweep_tolerance.py --headless --device cpu --variant task2-3_cross_tower_transfer_3

# Steps 5-6: skills (GPU, light scene); oracle sanity checks; deterministic evaluation per tier
python scripts/oracle_skill.py --headless --skill grasp_lift --arm left --num_envs 256
python scripts/train_skill.py --headless --skill grasp_lift --arm left --num_envs 4096 --max_iterations 1500
python scripts/evaluate_skill.py --headless --skill grasp_lift --arm left --checkpoint <run dir> --tier D1 --episodes 500
#   [--perception_noise [--noise '{"bias_pos_m": 0.003}']]   # Step 8.1: provisional perception model on the actor

# Step 7: rehearsal of the whole teacher (CPU station scene) -> results/rehearsal/<variant>/<tier>/feedback.md
python scripts/rehearse.py --headless --device cpu --variant task1-3_tower_transfer_3 --tiers D0 D1 --episodes 8 \
    [--bundle bundle] [--perception_noise]
python scripts/freeze_bundle.py --skill grasp_lift:left=<run dir> ...   # bundle/ + bundle.tar.gz + sha256
# all variants in parallel processes (Step 7 acceptance N >= 100; Step 8: --perception_noise, --bundle_configs
# = configs from the bundle (export check), --rate_hz 30); summary with Step 7 pass/fail per cell
python scripts/rehearse_many.py --bundle <bundle> --episodes 100 --parallel 4 --out_root results/rehearsal_final/<b>
# diagnose a failure signature from feedback.md (example (seed 1000003, ...) = rehearsal seed 1000, batch 0, env 3);
# rehearsals are deterministic for a fixed code + bundle, so the env reproduces exactly
python scripts/trace_episode.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --tier D1 \
    --seed 1000 --env 3 --bundle <bundle> --out results/traces/t22_1000003.jsonl [--fine 17.3 18.0] [--video f.mp4]
# watch the teacher solve a task: mp4 of the four evaluator cameras, real time (one frame per 0.1 s row), or live in
# the Isaac Sim window (--live, without --headless; RTX rendering every row)
python scripts/trace_episode.py --headless --device cpu --variant task2-3_cross_tower_transfer_3 --tier D0 \
    --num_envs 1 --seed 1 --bundle results/bundles/teacher_v6 --video results/videos/teacher_t23.mp4 --cam_scale 1.0
python scripts/drop_test.py --headless --device cpu --n 64      # block-release statistics vs height (CPU or GPU)
python scripts/release_test.py --headless --device cpu          # the planned place's release, many times
# Step 8.1: where perception noise turns into placement error (per place: executor estimates vs simulator truth;
# --probe_bias for fixed biases, --graph_set to try graph settings), then a per-arm summary
python scripts/noise_diag.py --headless --device cpu --variant task3-3_vertical_stacking_3 --bundle <bundle> \
    --noise '{"bias_pos_m": 0.001, "bias_yaw_deg": 1.0, "shared_pos_m": 0.003, "shared_yaw_deg": 3.0}' \
    --out results/noise_diag/t33.jsonl
python -I scripts/summarize_noise_diag.py results/noise_diag/t33.jsonl
```

Grasp quality: `grasp_lift` succeeds only for a near-canonical grasp (pinch within 10 mm along the block and 5 mm
vertically of the centred grasp, gripper within 6 deg of the canonical orientation), measured in the block frame by
`tower_rl.grasp.grasp_relation_error` (a lifted block hangs tilted with the gripper, so world-referenced comparisons
are wrong); `place_release` trains with held offsets up to that gate. Transits aim at the nearest reachable pose of
their via-point (`reachable_pose`), so a via-point a joint limit makes unreachable cannot stall the arrival check.

Held blocks: from the moment a gripper holds a block (before it moves) until it is released, the executor estimates
that block's pose as pinch x offset (`EnvRun.block_est`), refreshed only by confident, consistent measurements, so a
perception dropout of the moving block (the one the fingers occlude) cannot mislead the skills or the release pose.

Perceived poses (`tower_rl.graph.estimates`): every call feeds each block's pose to a rest-pose estimate (median of
its recent mutually consistent poses; a one-call outlier is held back, two consecutive far poses mean the block moved,
dropouts with confidence < 0.5 are ignored). Targets computed once (pre-grasps, goals from the destination foundation,
`arm_clear`, reachability) use these estimates; the skills' observations stay per call, as in training. A gripped block
is attached from its rest estimate; the place check decides on the median of a window of perceived poses once the
block is at rest (window medians, not per-call displacement, which perception jitter breaks). Goal orientation comes
from the foundation blocks' perceived yaws (an orientation error common to all blocks cancels against the held
block's), goal positions from the line through their centres. Graph `"perception": {"jaw_to_goal": w}` moves the share
w of the held block's perceived offset along the jaw (perception error: the fingers centre the block, and the place
policies ignore that offset) from the block to the place policy's goal: w = 1 cancels errors shared by all blocks,
w = 0 suits independent per-block errors; graphs v5 use 0.5 until the pose estimator's errors are measured.

CPU scripts set `torch.set_num_threads(1)`: torch's default threads on the executor's tiny tensors cost ~6 cores per
process and made rehearsals slower.

Executor failure handling (graph `on_fail`, default `retry`): a failure restarts the transfer with the gripper open,
up to `graph.retries` times, then skips to the next node of a later transfer (which may be the park that clears the
arm); a transport failure while the block is still in the gripper retries from the pre-place transit; a held transit
fails as soon as the gripper closes past the block width; a transfer whose block fell off the table is skipped; a
pre-grasp via-point far out of reach fails at once; a near-miss place (<= `arm_switch_m`, 15 mm) that its own arm
cannot re-grasp is re-grasped and re-placed by the other arm (inserted park / transit / grasp / transit / place /
park nodes). Before every transit the other arm must be clear of the target tower (`arm_clear`), else it is parked.

Pure-python tests (no Isaac): `python -m pytest` (kinematics vs the dataset, layouts, reachability of every via-point,
metrics, operation tracker, graphs, transit limits, executor determinism).

### Tasks 4 and 5 (push-down, insertion)

Same graph + executor, two more graph families (`tower_rl/graph/graph.py`; BUILDLOG "session 3"). **Push-down** (Task
4): per block, in the prescribed order, transit to `prepush` (fingers closed into the pusher tool) -> `push` (PPO, per
arm: `tower_rl/skills/push_env.py`). A learned push whose block stalls part-way out but level (debris under the last
block of a layer) is finished by the planned straight push (`handover_*` in `configs/skills/push.yaml`). **Insertion**
(Task 5): per insert, highest gap first, transit to the pre-grasp -> `grasp_lift` (planned in Task 5: graph
`grasp_planned`) -> held transit to `preinsert` (block on the slot axis, 17 cm out) -> `insert` (held entry to 9 cm,
release, back off) -> transit to `prepushin` -> `push_in` (to the slot centre). `insert` / `push_in` have PPO envs
(`tower_rl/skills/task5_env.py`), but the planned closed-loop versions in the executor are the default: on the CPU
evaluator they beat the GPU-trained ones (BUILDLOG). Tasks 4 / 5 run with a 3 N finger effort limit: hold at 42.5 mm
(the full 3 N), lift gently (`lift_v_m_s`); the push-in crosses an open gap below its slot with the pusher sliding down
the end face (`cross_window_m`, `cross_z_m`).

```bash
python scripts/build_layouts.py --t45                      # configs/layouts/task4-* / task5-*.yaml
python -m tower_rl.graph.graph task4-2_full_tower_pushdown_2 task5-3_tower_insertion_3   # graphs
python scripts/oracle_push.py --headless --arm left        # scripted push through the skill env (sanity)
python scripts/train_skill.py --headless --skill push --arm back --max_iterations 300 \
    [--override '{"reset": {"debris": {"frac": 0.7}}}'] [--resume <model.pt>]   # debris: earlier blocks on the table
python scripts/evaluate_skill.py --headless --skill push --arm back --checkpoint <run dir> --tier D1 --episodes 500
python scripts/oracle_task5.py --headless --skill insert --arm left --num_envs 128 [--ig_override '{...}']
python scripts/train_skill.py --headless --skill push_in --arm left --max_iterations 250
python scripts/grip_capacity.py --headless --device cpu --episode data/subset/task5-2_tower_insertion_2/<stem>.npz
python scripts/rehearse_many.py --bundle results/bundles/teacher_t45_final --variants task4-2_full_tower_pushdown_2 \
    task4-3_full_tower_pushdown_3 --tiers D0 D1 --episodes 20 --out_root results/t45/<name>
```

Rehearsals of Tasks 4 / 5 report a second success rate, "TSR collapse-only": our tracker fails a task once a block of
the remaining tower is > 10 mm from its episode-start pose, while the generator's records only show its collapse rule
(30 deg tilt / 15 mm drop); which one the leaderboard uses is not published.

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
| `tower_rl/geometry.py`, `kinematics.py`, `control.py` | Torch rigid-body math, Nero FK / Jacobian / DLS IK / gravity model, 10 Hz arm + gripper controllers (no Isaac imports) |
| `tower_rl/data/` | Episode files, remote reads, gripper events, layouts, success semantics, demo grasp geometry |
| `tower_rl/{tower,domain,grasp,postures}.py` | Tower sampling per stage and tier, domain samples, canonical gripper poses, IK posture priors |
| `tower_rl/sim/` | Isaac Lab asset configs, light skill scene, layout-driven station scene + `StationWorld` runtime |
| `tower_rl/eval/` | Metrics (TSR/GSR/PSR/LHPR/ODR), operation tracker, chunked runner, perception-noise model |
| `tower_rl/skills/` | Skill envs (`grasp_lift`, `place_release`), shared actor observations, transit planner, PPO config, TorchScript export |
| `tower_rl/graph/` | `State`/`SkillCall` interface, graph builder + JSONs, executor, rehearsal reports |
| `configs/` | Physics, scene, tiers, robot model, layouts, skill YAMLs (every scoring-relevant tunable) |
| `BUILDLOG.md` | Dated build log: acceptance numbers and deviations from the build instructions |

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
- **The evaluator runs CPU PhysX.** Open-loop replays on CPU reproduce the recordings (final block poses within 0.9 mm);
  GPU PhysX makes the keel-hulled blocks creep up to 15 mm (top cross blocks). Train on GPU, rehearse / evaluate on
  CPU (`--device cpu`).
- The massless passive `gripper_link` gets PhysX's default 1 kg; with the soft evaluator gains the back elbow sags
  0.085 rad at rest. All controllers command `q_des - tau_g(q_des) / kp` (`ArmKinematics.sag_compensated_command`).
- The pinch point is `gripper_base + (0, 0, 0.178)`; demos grasp centred, jaw across the block's width, approach
  tilted 12 deg (left) / 5.5 deg (back) about the jaw axis; via-points at 6 cm (pre-grasp) and 5 cm (pre-place).
- 10 Hz rows are held 0.1 s: replaying the demos at 10 Hz drops blocks in transport (fine at 30 Hz).
