# BUILDLOG: agentic teacher (TOWER challenge)

Build instructions: `~/Documents/corl_challenge/TOWER_BUILD_INSTRUCTIONS.md` (the "build file"). One entry per session:
what was built, acceptance numbers, deviations from the build file with reasons.

## Status (updated 2026-10-21 08:40)

| Step | State | Evidence |
|---|---|---|
| 1 data, layouts | passed | 60 episodes, layouts within 3.6 mm of the published descriptor |
| 2 scenes | passed | CPU replay 10/10 episodes, blocks within 0.89 mm; CPU PhysX = evaluator |
| 3 local evaluator | passed | worked example; demo replay TSR 0.9 (30 Hz); tolerance sweep |
| 4 transit | passed | arrival 0.5-4.5 mm; nearest-reachable via-points; held transits 0.08 m/s (edit 2) |
| 5 grasp_lift | **passed, both arms** (quality-gated) | D0 / D1: left 98.6 / 98.2 % (v4q), noise-tuned left v5n 100 / 98.8 %, back v3n 99.2 / 100 % |
| 6 place_release | passed without obs noise (99-100 %); below 95 % D0 with the build file's noisy obs (deviation 13) | left 92.6 %, back 87.2 % D0 noisy |
| 7 graph, executor, rehearsal, edit loop | **passed**: all 12 cells at N = 100 for both bundles | `teacher_v4` 1186 / 1200 (precise grasps), `teacher_v6` 1182 / 1200 (noise-tuned grasps, new executor); edits: 1 reverted, 2 and 3 kept |
| 8 hardening | export, timing, both arms passed; **executor robust to perception noise** (realistic model TSR 0.29-0.88, was 0.00-0.75); build-file model (independent 3 mm per block) limited by perception (TSR 0.08-0.33); open: left place under shared errors, 30 Hz Task 1-3 | "Step 8.1 round 2" |
| Pose estimator (session 4) | **closed loop on rendered cameras**: Tasks 1-3 48 / 48, Task 5-2 8 / 8, 5-3 7 / 8, Task 4-2 D0 4 / 4 / D1 1 / 4, 4-3 D0 2 / 4 / D1 0 / 4 (truth 4-3 D0 4 / 4); organizer frames (kp_v2) all blocks p90 0.70 mm, held p90 0.24 mm; open: Task 4 push actor vs perception | `tower_rl/perception/`, `results/perception/` (deploy kp_v2); "session 4" |
| Tasks 4 / 5 (session 3) | **integrated**: graph families pushdown / insertion, executor nodes, PPO push (both arms) in the bundle; Task 5 runs planned grasp / insert / push-in (learned push_in left trained: worse on CPU) | rehearsal, CPU, `teacher_t45_final` (= v2 skills): 4-2 D0 / D1 0.97 / 0.88, 4-3 0.90 / 0.72 (40 each); 5-2 1.00 / 0.75, 5-3 1.00 / 0.58 (24 each; 5-3 D1 0.81 on two other seed sets) |

Bundles: `results/bundles/teacher_t45_final` for Tasks 4 / 5 (teacher_v6 skills + learned pushes left v3 / back v1c; graphs and configs of 2026-10-20 03:52; see session 3). `teacher_v6` to run on estimated poses (graphs v5, noise-tuned grasps), `teacher_v4` for privileged use
(DAgger labels on simulator truth: tighter in-hand rotations). Submit on the 10 Hz track.

## 2026-10-18 (session 1)

### Ground truth re-checked against the live docs (rules v1.3, 2026-10-15)

- Robustness cases (docs "Task prompts"): nominal; light mass x0.4-0.7; heavy x1.4-2.0; uneven per-block x0.4-2.0;
  slippery friction 0.15-0.30 static / 0.07-0.14 dynamic; high friction 0.9-1.3 / 0.5-0.8; shifted assembly <= 15 mm per
  axis and <= 10 deg yaw; irregular layers <= 8 mm per axis and <= 5 deg; lighting x0.25-2.0, 3000-8500 K, azimuth
  <= 60 deg; combined (mass x0.5-1.8, mixed friction, shift <= 10 mm / 8 deg, light x0.4-1.6). **Fixed**: foundation
  blocks, table, robot mounts, robot dynamics, camera calibration, destination anchors. 10 episodes per task
  (1 nominal + 9 randomized), 210 s. These replace the build file's D1 guesses (see Tiers below).
- Metrics page: TSR / GSR / PSR / LHPR / ODR as in the build file; the docs do not publish grasp/place tolerances.
- Actions: float32 [H,16], 1 <= H <= 16, absolute rad + gripper width (m); 10 Hz = 3 x 30 Hz ticks per row.

### Step 1: data subset, events, layouts, semantics: PASSED

Built: `tower_rl/data/{episode,remote,events,layouts,semantics,grasps}.py`, `scripts/fetch_subset.py`,
`scripts/build_layouts.py`, `scripts/analyze_subset.py`, `configs/layouts/<variant>.yaml` (6), `tests/test_data.py`.

- VERIFY remote partial reads: works (h5py over HfFileSystem range requests). 10 episodes per variant (60) fetched in
  ~5 min with 6 processes, ~1 MB each, JPEG streams never downloaded (`data/subset/`).
- `pytest tests/test_data.py tests/test_kinematics.py`: 75 passed. All 60 files parse; operation counts match
  (Task1 2xL, Task2/3 L) in every episode; demos follow the plan order and arm assignment exactly; every operation
  holds a 0.0497 m block (measured width 0.045-0.055) and lifts it > 4 cm.
- Task 2-3 vs the published cfg08 descriptor: source centre 3.4 mm, destination centre 3.6 mm, layer yaws < 0.1 deg,
  plan identical (l3->1 back, l2->2 left, l1->3 back). Top-down plots in `results/layouts/` checked by eye.
- Semantics (60 episodes): `sim/success` and `sim/terminal` both turn true only on the last frame, 9.2-9.8 s after the
  last release (retreat + settle + park); no collapses; every demo succeeds. `task_progress` can briefly drop while a
  block is being placed on the previous one (recomputed from state).
- Layout facts: two pad positions, A ~ (0.426, -0.277) and B ~ (0.174, -0.525), 0.356 m apart. Tasks 1/2 move A -> B,
  Task 3 moves B -> A. Task 3 uses the back arm only; Tasks 1/2 alternate arms per layer as listed in the YAMLs.
  Block centres vary only by the generator's 3 mm tower jitter (std 1.5-2.3 mm). Movable blocks rest ~1 mm (per layer)
  lower than the nominal 0.015 + 0.03 k because the collision hull is 28.3 mm tall.
- Demo grasp geometry (`results/data/subset_analysis.json`, 200 operations): pinch point within 1 mm of the block
  centre (+0.2..1.8 mm in z), jaw axis = block y (the 0.05 m width), approach tilted along block x by 9-13 deg (left
  arm) / ~5 deg (back arm). Placement errors 1-4 mm, yaw < 1 deg. Demo durations: Task1-3 175-182 s (tight against
  the 210 s budget), others 58-117 s.

Deviations:
1. Package is `tower_rl/` (existing repo), not `tower/`; layout otherwise follows the build file.
2. The files **do** contain block poses (`sim/block_poses` [T,B,7]) and the transfer plan (`target_mapping`), so
   layouts come from the recorded poses of all 331 episodes (`data/meta_tasks123.jsonl`), not from gripper events
   plus a TCP offset. Gripper events are still implemented and cross-checked against `sim/active_block`.
3. Placement tolerance: the data generator uses 5 mm / 5 deg (`placement_m`, `placement_degrees`); we adopt 5 mm as the
   primary `tol_pos` (build file: 10 mm) and report 10 mm alongside until the sweep (Step 3.5).

### Robot model findings (affect every controller)

Built: `scripts/extract_robot_model.py` -> `configs/robot/nero.json`, `tower_rl/{geometry,kinematics}.py`,
`scripts/probe_arm_statics.py`, `tests/test_kinematics.py`, `scripts/run_logged.sh` (GPU telemetry per run).

- Torch FK (from the USD joint chain) reproduces the recorded `observations/eef` (link7 in base_link) to 0.011 mm /
  0.002 deg on every frame; the pinch point is `gripper_base + (0, 0, 0.178)` (the generator's "flange" is
  gripper_base) and matches the generator's measured pinch centres to 0.001 mm.
- **Phantom mass**: the passive `gripper` joint's body (`gripper_link`) has mass 0 and no collider, so PhysX gives it
  1 kg. It sits at its 0.1 m slider limit (gripper pointing down) and dominates the wrist/elbow gravity load. With it,
  the torch gravity model matches PhysX's generalized gravity forces to 0.1-2 % per joint.
- **Gravity sag**: the evaluator's PD gains are soft (back elbow kp 68 N m/rad). Measured sag at rest: back j4
  0.085 rad (identical in the dataset and our sim), i.e. 2-4 cm at the pinch; the demos command the pinch 3.5 cm above
  the block for the back arm. The static model predicts the sag to <= 0.2 mrad (left) / ~7 mrad (back j4; joint
  friction) at demo poses. Every controller therefore commands `q_cmd = q_des - tau_g(q_des)/kp` and closes the loop
  on measured joint angles.
- Evaluator-matched dynamics stay as recorded: finger kp 800 / kd 20, finger effort limit 22 N (build file: 3 N),
  arm effort 100 N m, command delays left 5 ms / back 10 ms.

### Step 2: scenes: PASSED (with the CPU-physics finding below)

Built: `scripts/make_meter_assets.py` (+ standalone arm wrappers `assets_m/nero_<arm>_station_m.usda`),
`scripts/dump_stage_tree.py` -> `docs/stage_tree.txt`, `tower_rl/sim/{assets,skill_scene,station_scene}.py`,
`tower_rl/{control,grasp,postures,tower,domain,config}.py`, `scripts/{smoke_scenes,replay_demo,diagnose_replay,
camera_overlay,build_posture_priors}.py`, `configs/{physics,scene,tiers}.yaml`, `configs/skills/{grasp_geometry,
posture_priors}.yaml`, `tests/{test_scenes,test_reachability}.py`.

- 2.1: stage dump in `docs/stage_tree.txt`. Gripper pad material 2.4 / 2.0 and block "Wood" 0.40 / 0.23, both with the
  default *average* combine mode; articulation roots on `root_joint` (32 / 1 solver iterations); no physics scene
  authored in the station (the evaluator configures it at runtime).
- 2.2 light skill scene: the arm wrapper references the installed robot prim, so the arm, its drives, the phantom
  1 kg gripper body and the convex block hull are exactly the evaluator's. Smoke (`pytest --isaac tests/test_scenes.py`,
  3 passed): arm root at the station base pose, PhysX vs torch FK 0.002 mm, sag-compensated hold of the pre-grasp
  pose 0.2 mm (left) / 2.3 mm (back), tower creep <= 2.5 mm, kinematic foundations 0 drift, no NaN in 300 ticks.
  Contact sensors: one per finger (PhysX needs one filter match per sensor body).
- 2.3 (c): `results/calibration/camera_overlay_t23_f0.jpg`. Renders (published-station camera poses) vs demo JPEGs
  (Sep-9 station): static cameras offset 12-17 px, wrist cameras 4-7 px (a camera re-calibration between the data
  station and the published one; ~1 deg). Mapping confirmed: left_back = low_camera, right_front = high_camera, both
  stored rotated 180 deg. Perception must be trained on renders of the published station, not only on demo JPEGs.
- 2.3 (b) demo replay: **on GPU PhysX it fails, on CPU PhysX it passes.** Task 2-2, 10 episodes:
  - GPU TGS (Isaac Lab default): 7/10 episodes >= 8/10 replays. Diagnosis (`scripts/diagnose_replay.py`): the top
    cross block slides 8-19 mm in the first 0.1-0.2 s and creeps on (the keel-pointed hull rocks); in the recording it
    moves 1.1 mm. Not fixed by a 1 s settle, higher block friction, 64 solver iterations, stabilization or a smaller
    friction correlation distance. GPU PGS stops the creep but doubles the arm tracking error and breaks other grasps.
  - **CPU TGS: 10/10 episodes, 100 % of operations, final block poses within 0.89 mm (mean) of the recording**, arm
    tracking 0.43 mrad RMS, deterministic, and 6x faster than GPU at this env count (8 s per 60 s episode).
  - Conclusion: the evaluator runs CPU PhysX. GPU and CPU agree on the articulation (same tracking error); GPU differs
    in block contacts. Training uses GPU TGS (throughput); rehearsal, skill acceptance and the local evaluator use CPU.
- Settling: writing recorded (tilted) block poses directly can leave hulls interpenetrating; all resets now place
  movable blocks 0.5 mm high and settle 1 s (the generator also settles 1 s).

Deviations:
4. Physics timing: build file "control at 30 Hz (3 physics steps)" is inconsistent with dt = 1/120; we use 120 Hz
   physics, 30 Hz ticks of 4 substeps, 10 Hz rows = 3 ticks (as the docs specify).
5. Solver iterations stay at the evaluator's (robot 32/1, blocks 16/4), not Factory's 64/192: the goal is fidelity to
   the evaluator, and the CPU replay test confirms it.
6. Light scene blocks use the evaluator's convex block asset instead of boxes (fidelity; GPU handles it).
7. Via-point heights from the demos instead of the build file: pre-grasp 6 cm (10 cm is infeasible: joint 7 hits
   +-90 deg with the demos' wrist orientation), pre-place 5 cm, release 5 mm above the goal; transport peak ~0.17 m.
   The canonical gripper orientation (jaw along block y, 12 deg / 5.5 deg tilt away from the base) matches the demos to
   ~1 deg; IK is seeded with demo posture priors (`configs/skills/posture_priors.yaml`); all via-points of all six
   variants are reachable (`tests/test_reachability.py`).

### Step 3: local evaluator: mostly done

Built: `tower_rl/eval/{metrics,operations,runner}.py`, `tower_rl/graph/interface.py` (State, SkillCall, SkillResult),
`scripts/evaluate_policy.py`, `tests/test_metrics.py` (7 passed).

- Metrics: worked example GSR 0.667 / PSR 0.333 / LHPR 0.5 / ODR 1/7 / TSR 0 reproduced; counts summed before division;
  Wilson 95 % intervals. Operation semantics (our reconstruction, documented in `operations.py`): G = block raised
  >= 2 cm while a closed gripper holds it; P = released, within 5 mm / 5 deg (z 10 mm against the measured rest
  height) and static for 1 s with lower layers intact; outcomes latch; windows follow the plan pointer, which skips
  forward when a later block is grasped; ODR's final window only when the task fails.
- Runner: chunk semantics [H,16] with 1 <= H <= 16, rows held 0.1 s (10 Hz track) or 1/30 s (30 Hz track), policy
  errors end the episode; CPU station scene; tracker at 30 Hz.
- Hold-still: all metrics 0. Demo replay of the 10 Task 2-2 recordings from their own initial states,
  **30 Hz track: TSR 0.9, GSR 1.0, PSR 0.95, LHPR 0.975, ODR 0** (acceptance >= 0.6).
- **10 Hz track finding**: replaying the demos with every third command held 0.1 s drops the block in transport
  (GSR 1.0, PSR 0): 0.1 s steps excite the softly damped arm and shake a 2.4 N grasp loose. Our transit and skills must
  be smooth at 10 Hz (the skill env trains with exactly this hold), or the submission should use the 30 Hz track
  (ranked separately). **HUMAN: which track to submit (10 Hz or 30 Hz).** The executor supports both.
- Open: `scripts/sweep_tolerance.py` (placement-stability sweep) not yet run; tol_pos stays 5 mm (generator value).

### Steps 4-7: transit, skill envs, graph, executor, rehearsal: first versions running

Built: `tower_rl/skills/{transit,base_env,grasp_lift_env,place_release_env,obs,ppo_cfg,export}.py`,
`tower_rl/graph/{graph,executor,rehearsal}.py` + `tower_rl/graph/graphs/<variant>.json`, `scripts/{train_skill,
oracle_skill,profile_skill_env,rehearse}.py`, `configs/skills/{grasp_lift,place_release}.yaml`.

- Transit (planned): rise to max(start, goal) + 2 cm, straight line with slerped orientation, vertical descent; quintic
  time law per segment (v 0.25 m/s empty / 0.12 m/s holding, a 0.6 m/s^2); rows from the shared controller (IK +
  sag compensation). Arrival errors in rehearsal 0.5-4.5 mm (initiation region 3 cm / 5 deg).
- Executor: deterministic per-env interpreter of the graph JSON; transit/park chunks up to 16 rows, contact skills one
  row; goals from the *observed* destination foundation + in-pad offsets; held-block offset measured after the grasp
  and compensated at the place; a feasible-orientation step lets IK choose the tilt about the jaw axis when the
  canonical tilt saturates a joint (left arm on pad B: wrist pitch j6 at -0.71 rad); retry the transfer twice, then skip.
- Pipeline test with *planned* grasp and place fallbacks (not rules-compliant for grasp; used to validate everything
  around the learned skills), CPU station scene, 10 Hz track: Task 2-2 TSR 1.0 (D0 8/8, D1 8/8), Task 1-2 TSR 1.0
  (D0 8/8, D1 8/8), ODR 0, Task 2-2 in 41 s. Task 1-3 D0 failed on transfer 4 (left arm, destination layer 3 on pad B:
  placements 5.5-5.8 mm at the j6 limit) -> fixed with the feasible-orientation step (re-test pending).
- Skill envs (light scene, GPU): grasp_lift (65-D actor, 141-D critic) and place_release (73-D / 150-D); actor inputs are
  camera-estimable only and shared with the executor through `tower_rl/skills/obs.py`. Scripted privileged oracles
  reach 82 % (grasp_lift) and 71 % (place_release) success, so both tasks are solvable as defined.
- Throughput: 12.8k env-steps/s at 4096 envs (physics 0.19 s per policy step = 12 substeps; resets were 0.32 s per
  step until the PhysX property tensors were cached and the IK reworked: one chain pass per iteration, matrix rotation
  error, convergence check every 10 iterations). GPU under training: 160-170 W, plateau 86 C with software thermal
  slowdown (clocks 1.65-1.8 GHz); telemetry in `results/**/*.gpu.csv`.
- Training run v1 (grasp_lift, left, 4096 envs, 1500 iterations ~3.3 h) in progress.

Deviations:
8. Skill reset clearance: start pinch >= 3.5 cm above the grasp point (2 cm put the open finger tips inside blocks and
   PhysX ejected the tower).
9. Gripper closed command 0.040 m (3.9 N per finger with kp 800) instead of the demos' 0.0437 (2.4 N): margin for the
   2x heavy and slippery cases; widths are rate-limited like the demos (0.09 / 0.03 / 0.16 m/s).
10. Env cfg classes must be `@configclass`-decorated or the parent dataclass __init__ silently restores the parent's
    defaults (the place env briefly loaded the grasp YAML).

### 2026-10-19 00:15: pipeline baseline on all variants, tolerance sweep, grasp_lift v1 -> v2

- **Planned-pipeline rehearsal** (executor with *planned* grasp/place fallbacks, privileged sim state, CPU station scene,
  10 Hz track, 8 episodes per cell; `results/rehearsal/<variant>/<tier>/feedback.md`):

  | variant | D0 TSR | D1 TSR | D1 ODR |
  |---|---|---|---|
  | task1-2 | 1.00 | 1.00 | 0.00 |
  | task1-3 | 1.00 | 0.875 | 0.09 |
  | task2-2 | 1.00 | 1.00 | 0.00 |
  | task2-3 | 1.00 | 1.00 | 0.00 |
  | task3-2 | 1.00 | 1.00 | 0.00 |
  | task3-3 | 1.00 | 1.00 | 0.00 |

  The only failure: Task 1-3 D1 "combined" case, first place misses, then re-grasping the misplaced block from the
  destination stack is unreachable (transit timeouts) -> retry-after-failed-place is the weakest edge (candidate for
  the edit loop). Executor wall time 10.8 ms per call (max 139 ms, CPU), ~1100 calls per Task1-3 episode; Task 2-2
  finishes in 41 s, Task 1-3 in ~130 s (budget 210 s).
  These numbers validate transit, graph, executor, goals-from-observations and checkpoints; they are *not* the
  teacher's score (the rules require the learned grasp/place, being trained now).
- **Step 3.5 tolerance sweep** (`scripts/sweep_tolerance.py`, CPU, 5 s): complete destination stacks of Task 1-3,
  2-3 and 3-3 with the top layer offset 0-20 mm (along or across the block) and 0-10 deg all stand (70/70 each):
  with 2x-size blocks of 106 g, physical stability does not bind. tol_pos / tol_yaw stay at the generator's
  5 mm / 5 deg (the evaluator's unpublished scoring tolerance is the binding one).
- **grasp_lift v1 stalled** (0.3-1.5 % success at iteration 116, 85 % timeouts): exploring with a rate-limited binary
  gripper that needs ~7 consecutive "close" requests (and opens faster than it closes) almost never produces a grasp
  when every episode starts 6 cm above the block. **v2**: reset mixture (40 % around the pre-grasp, 30 % mid-height,
  30 % at the grasp pose with open fingers straddling the block), faster shared gripper profile (0.3 / 0.1 / 0.3 m/s,
  executor uses the same), +0.5 for requesting close while the keypoints are within 1 cm. v2 reaches 21 % success at
  iteration 47 (87 % of steps grasped).
- Perception-noise model for Step 8.1 (`tower_rl/eval/perception_noise.py`, `rehearse.py --perception_noise`):
  per-episode bias 3 mm / 3 deg, jitter, 2 % dropouts of 0.5 s, 0.5 % 1 cm outliers on all block poses.
- **Perception requirement (Step 8.1 probe, planned pipeline, Task 2-3 D1)**: build-file noise (independent 3 mm / 3 deg
  bias per block) -> TSR 0.00, PSR 0.42 (GSR 1.0): the true placement error combines the held block's and the
  foundation's estimate errors (~4 mm sigma per axis) against a 5 mm tolerance; the executor's own checks pass on the
  noisy estimates. Mostly shared errors (1 mm / 1 deg independent + 3 mm / 2 deg shared) -> TSR 0.75, PSR 0.875.
  => The deferred pose estimator needs ~1 mm *relative* (block-to-block) accuracy; shared offsets are tolerable.
- **Reward bug (v2 hovered)**: hovering at the lift height earned ~7.25 per step (~725 discounted) vs <= 470 for
  finishing; success_per_remaining_step 3.0 -> 8.0 (> max shaping) for grasp_lift and 6.0 for place_release; v2
  resumed from its iteration-100 checkpoint with the fix (v3 tag).
- **grasp_lift left v3** (v2 + success-bonus fix, resumed from v2 iteration 100): training-mix success 88.7 % at
  iteration 115 and 92.2 % at 131 (disturbed 3.0 %, dropped 4.9 %, timeouts 0.2 %, 3.3 s per episode). Training mix:
  D0 25 % / D1 25 % / D2 50 %.
- Integration check: a weak v2 checkpoint, frozen to TorchScript (`scripts/freeze_bundle.py`) and run by the executor
  in the CPU station scene (evaluator physics, GPU-trained policy), grasped and lifted in ~4 s; Task 2-2 D0 2/2 TSR.
  Fixed on the way: `neighbors()` pads to two candidates (Task 2-2 has a single other block at the executor).
- Oracles with the new gripper profile: place_release left 73 %, back 54 % (the oracle's convergence threshold times
  out on the back arm's larger sag); grasp_lift back 98 %.
- **Step 5 acceptance (left arm), GPU light scene** (`scripts/evaluate_skill.py`, v3 iteration-200 checkpoint,
  deterministic actor, noisy actor observations, 500 episodes per tier):
  D0 96.6 % (Wilson 94.6-97.9 %), disturbance rate 0.028, 2.9 s per grasp;
  D1 96.4 % (94.4-97.7 %), disturbance 0.026; every robustness case >= 93 % (slippery 93.4 %, lightweight 95.9 %,
  heavy 97.1 %, combined 97.9 %, high friction 100 %). Targets (>= 95 % D0, >= 85 % D1, disturbance < 0.1) met.
  (A first evaluation mixed the tiers: `--tier` deep-merged into the training mix; fixed to replace it.)
- **CPU PhysX cannot move kinematic bodies** through the tensor API after creation (neither `set_transforms` nor
  `set_kinematic_targets`; verified with a minimal scene). Consequences: the light skill scene on CPU keeps its
  placeholder foundations (towers fall) -> skills are evaluated on CPU through full-task rehearsals in the station
  scene, whose foundations spawn at their nominal layout poses (evaluator: fixed foundations, consistent). Episode
  replays on CPU keep the first episode's foundation poses (<= 6 mm from later episodes'; replay acceptance unaffected).
- **Learned left grasp in the CPU station scene** (bundle `results/bundles/grasp_left_v3_200`: learned left grasp,
  planned back grasp and places, graphs v1): Task 2-2 D0 1.00 / D1 0.875, Task 2-3 D0 1.00 / D1 1.00 (8 episodes
  each); the learned grasp succeeds on every first attempt, median 2.4-2.6 s (planned 3.1 s). Task 1-3 D0 0.25:
  transfer 4 (left arm, destination layer 3 at its pad-B workspace edge) placed 6.5-7.7 mm off in 8/8, then re-grasp
  retries from the destination time out. Cause: when the canonical tilt is infeasible the release orientation was
  replaced but the position kept, so the held block's centre swung by |offset| x dtilt (the learned grasp holds the
  block with its own offset). **Fix**: `release_pose()` starts from the exact pinch for the measured held offset, lets
  IK choose the tilt only if needed, then re-solves the pinch position so the block centre stays on the goal; the
  pre-place transit and the place share that target.
- **Edit loop iteration 1** (`tower_rl/graph/edits/2026-10-19_1/`: prompt, diff, before metrics): all place nodes
  `on_fail: skip_if_close` (15 mm), executor support for `on_fail` (retry | skip | skip_if_close). Rationale: a
  near-miss place triggered a re-grasp from the destination stack (outside the arms' demonstrated workspace) whose
  timeouts consumed the budget. **Reverted** after the A/B below (graphs v3 = v1 content).
- **grasp_lift back** initialised from the left policy (iteration 300): 5 % -> 71 % training-mix success within 7
  iterations; 98.2 % at iteration 500 (disturbed 0.9 %, dropped 0.9 %), stopped there.
- **Step 5 acceptance (back arm), GPU light scene** (iteration-500 checkpoint, 500 episodes per tier): D0 100 %
  (Wilson 99.2-100 %), disturbance 0.0; D1 99.6 % (98.6-99.9 %), disturbance 0.004; 2.0 s per grasp. Both arms meet
  Step 5.
- **Learned left grasp, iteration 300** (bundle `teacher_v2_gl300`, graphs v2, release-pose fix): Task 1-3 D0 1.0 /
  D1 0.875, Task 2-2 D0 1.0 / D1 1.0 (8 episodes each); the Task 1-3 D0 0.25 of the iteration-200 bundle is fixed.

### Executor fixes found by the edit-1 diagnosis (2026-10-19)

New tool `scripts/trace_episode.py`: re-runs one rehearsal batch (deterministic on CPU) and records one env per call
(node, pinch, controller target, joints, gripper, blocks) and optionally every 120 Hz substep in a time window.
- **Skip jumped over the arm-switch park** (bug): `_advance` jumped to the next transfer's `first_of_transfer` node,
  but the park that clears the failing arm belongs to the next transfer and precedes it. After a skipped left-arm
  place the left arm hovered over pad B and the back arm's held block ran into it (pre-place transit timeouts). Same
  path after exhausted retries. Fixed (skip -> next node of a later transfer; a timed-out park continues), tests
  `test_failure_edges`.
- **Planned grasp drops heavy + slippery blocks**: in the "combined" case the block slid 14 mm in the fingers during
  the planned 10 cm/s lift and fell out in transit, off the table; the executor carried an empty gripper to the place
  and retried the unreachable block for 30 s. Now a held transit fails as soon as the measured opening drops below
  the block width, and a transfer whose block is below the tabletop is skipped (`test_lost_block_skips`). (The
  learned grasp holds that block.)
- **Retry with the block still in hand** restarted at the pre-grasp transit with a closed gripper. Now a transport
  failure while holding retries from the pre-place transit; every other failure opens the gripper and restarts the
  transfer (`test_retry_keeps_held_block`).
- `LearnedPlaceNode` mirrors the training env exactly (settle check timing one row later than before, payload from the
  gripper state before the row, the new settle semantics below). The pre-place transit takes the release clearance
  from its place node's `release_clearance_m` (graph field) instead of the global config.
- **Edit 1 A/B under the fixed executor** (D1, 8 episodes, same seeds): Task 1-3 planned TSR 0.875 (v1, retry) vs 0.75
  (v2, skip_if_close); Task 2-2 learned left grasp TSR 1.0 vs 0.875. Retrying a near-miss place re-grasps the block
  from the destination stack and re-places it (1.4 mm and 0.5 mm in the two affected episodes); skipping loses the P
  and the task. Edit reverted (`tower_rl/graph/edits/2026-10-19_1/notes.md`).
- **Pop on release** (planned place, ~2-4 % of places): Task 2-2 seed 1000003, per-substep trace: the block, released
  5 mm above its goal with 1.9 deg tilt and 0.2-0.3 rad/s residual rotation from the grasp, lands on its low end,
  pivots at ~3.7 rad/s, penetrates the layer below by 1.6 mm and is ejected at 0.58 m/s, landing 24 mm away (the
  retreating finger dragged it back to 8.9 mm). `scripts/drop_test.py` (free drops of 0-8 mm with +-2 deg tilt and
  residual velocities onto the destination foundation, 64 per height, CPU and GPU): no pops, all in tolerance, all
  static after 1.5 s; so the opening fingers take part. Open item for the planned place; the learned place chooses
  its own release.
- **Release harness** (`scripts/release_test.py`, CPU station scene): the executor's own `PlannedPlaceNode` from the
  pre-place height with the block teleported into the closed gripper (canonical +-3 mm / 2 deg, or 15 / 25 mm
  off-centre along the block), clearance 2 or 5 mm, optional `pre_open_hold_s` (new node field, default 0):
  512 places on Task 2-2 transfer 0, all within tolerance (max 2.5 mm), peak upward block speed after the opening
  <= 0.005 m/s, i.e. no pop. A release from rest in the hand (256 more) and free drops do not pop either. The
  rehearsal pops therefore need the full pipeline context (the block shifting in the grasp during the held transit,
  arrival dynamics); left open, as the learned place replaces the planned one.
- Pause before opening, full planned pipeline (Task 2-2, 64 episodes each, seed 7000): `pre_open_hold_s` 0 vs 0.3 s
  gives identical TSR (D0 1.0 / D1 0.906 both) and 8/199 vs 12/206 failed place attempts -> no effect, stays 0.
  The planned pipeline's D1 failures are heavy / uneven-weight / combined blocks (drops in the held transit, landing
  6-32 mm off), not pops.
- **Learned left place on CPU** (probe bundle `probe_place_left_v1_400`: place-left v1 it. 400 + both grasps, planned
  back place; 8 episodes per cell): Task 2-2 D0/D1 TSR 1.0, Task 1-3 D0/D1 TSR 1.0; all 128 places succeed (the
  80 learned left-arm ones in 1.8-1.9 s each, errors 1.4-3.9 mm; the planned place needs ~3.5 s). The GPU-trained
  place transfers to the evaluator's CPU physics. Task 2-3 D0/D1 TSR 1.0; Task 1-2 D0 1.0 / D1 0.75: (a) every G/P
  succeeded but the final arrangement never held 1 s in tolerance: transfer 2 was placed at 4.93 mm (tolerance 5) and
  moved out when transfer 3 landed on it; (b) a 5.37 mm near-miss of transfer 2 (left arm, pad B, layer 2) was
  retried: the re-grasp's pre-grasp needed a 49 mm via-point adjustment (destination stack at the left arm's
  workspace edge, never in its grasp training), the grasp timed out and knocked two blocks, the next transfer's
  pre-grasp transits timed out. Changes: place reward + `success_precision` 30 x (1 - xy error / 5 mm) at success
  (from the v2wide fine-tune on), so placements keep margin; pre-grasp via-points more than 30 mm / 10 deg out of
  reach fail at once (`MAX_PREGRASP_ADJUST_*`) instead of starting the grasp skill there (first attempts are always
  reachable by test, so this only cuts hopeless re-grasps short).

### Step 6: place_release (2026-10-19)

- **Settle check aligned with the evaluator's P** before training: the env checked "every block static" at exactly
  1.0 s after the release while the fingers were still retreating; a scripted oracle placing within ~1 mm failed 5 %
  of its episodes that way, plus 12 % on GPU velocity jitter of blocks resting on other dynamic blocks (1-2 cm/s
  reported while not moving; CPU and the free-drop scene show none). Now: success = the target block released (gripper
  open), within 5 mm / 5 deg (z 10 mm) and static for 1.0 s *continuously* (as `tower_rl.eval.operations`), placed
  blocks undisturbed and the pinch risen 3 cm; static = displacement over the last 0.1 s step < 1 cm/s; failure = at
  rest off the goal >= 1 s after the release, or no success within 3 s. Critic + settled counter (151-D).
- Oracle (left, 256 envs, true poses): success 73 % -> 78 % (continuous check) -> 90 % (displacement-based static;
  release 5 mm above the goal) / 91 % (2 mm); the rest are
  the oracle's own convergence timeouts (8 %) and 1-1.5 % blocks lost from the hand before the release.
- **Training**: left arm from scratch, 4096 envs, `results/skills/place_release_left/2026-10-19_0133_cd7db0fad9_v1`:
  0 % success until iteration ~20 (random releases from 1-6 cm up), then 1.5 % (it. 24), 25 % (it. 50), 37 % (it. 70),
  84 % (it. 200), plateau ~90 % from it. 300 (training mix incl. 50 % D2, noisy observations; resting error 2.9 mm);
  stopped at it. 400 to continue with the wide held offsets (v2wide).
- **Grasp quality gate, baselines** (D1, 500 episodes, gate as above): left v3 it. 300 81.8 % (bad grasps 16 %, mostly
  the vertical offset; |dx| p50/p90 2.7/5.7 mm, |dz| 2.5/5.7 mm, rotation 1.7/3.6 deg); back it. 500 20.8 % (79 % bad:
  |dx| 6.6/20.4 mm, |dz| 9.6/16.1 mm, 4.8/8.1 deg). Fine-tunes from those checkpoints with the gate: left v4q
  60 % -> 76 % training-mix success within 36 iterations (bad grasps 36 -> 20 %).
- **Fine-tune v2wide** (from v1 it. 400; half the episodes hold the block with offsets up to the grasp gate; precision
  bonus 30): training-mix success stays ~89-90 % and the resting error ~3.0 mm from the first iterations on: the
  policy handles the wide held offsets at once, and the precision bonus cannot buy precision the observations do not
  carry: the actor sees the held block and the goal with independent 2 mm per-episode biases (2.8 mm relative per
  axis), which alone puts ~20 % of perfectly aligned placements beyond 5 mm (the build file's own place observation,
  goal sigma 2 mm, caps it near 92 %). The acceptance evaluation "with the noisy actor observations" therefore bounds
  place success by perception noise; it is reported together with the same evaluation without observation noise
  (the policy's own precision; the CPU executor places at ~2 mm with exact poses).
- **Left place v1 vs v2wide on CPU** (head-to-head, new grasps, planned back place, Task 1-3 and 2-3, 24 episodes
  per tier, seed 3000): v1 TSR 1.0 / 0.83 (1-3 D0 / D1), 0.96 / 0.92 (2-3); first-attempt left placements p50 2.2,
  p90 3.3 mm, 9 / 232 beyond 5 mm. v2wide 1.0 / 0.62 (ODR 0.10), 1.0 / 1.0; p50 2.8, p90 4.8 mm, 16 / 228 beyond 5 mm
  (better on a layer's first block, worse next to an already placed block). **v1 (it. 400) is the final left place.**
  The remaining Task 1-3 D1 failures are heavy / uneven-weight blocks: drops in the held transit, placements 6-17 mm
  off, and arm-switch re-grasps that fail (the back grasp times out or drops a block it must take from the
  destination stack: re-grasps from a stack are outside both grasp policies' training). Candidates for the next
  round: slower held transits, more grip force (changes an actor input, needs retraining), re-grasp scenarios in the
  grasp training.
- **Edit iteration 2 (kept)**: held transits 0.12 -> 0.08 m/s in every graph (v4; the executor now reads `v_max`,
  `v_max_held`, `a_max` from transit nodes). Task 1-3 D1, 48 episodes, same seeds and skills: TSR 0.833 (Wilson
  0.70-0.91) -> **1.000** (0.93-1.00), PSR 0.93 -> 1.0, ODR 0.067 -> 0; the six drops, sixteen place misses, grasp
  timeouts and failed re-grasps all but vanish (one miss, re-placed); +4 s per episode. On the 10 Hz track each row is
  a step of the joint targets; at 12 mm per row the soft arm shakes the held block, which shifts in the fingers
  (misses) or falls out (drops). (`tower_rl/graph/edits/2026-10-19_2/`.)
- **Back place**: initialised from the left place (as the back grasp was from the left grasp) it did not transfer:
  0 % success after 16 iterations, failures rising 70 -> 92 %, blocks released ~6 cm from their goals (the place
  actor's inputs include the goal and pinch in the arm's base frame, and the back arm's goals sit elsewhere in its
  frame). Stopped; trained from scratch (`*_v1_scratch`, 600 iterations, wide held offsets and precision bonus from
  the start).
- `tower_rl/graph/checkpoints.py` (Step 7.2, was inline in the nodes): `holding`, `lifted`, `grasped`,
  `placement_error` / `placed`, `static` (displacement-based), `max_disturbance_m`, `arm_clear`; the nodes call them.
  `arm_clear` is enforced: before an arm's transit to a tower, the other arm's pinch must be > 0.18 m (xy) from the
  target (home poses are >= 0.23 m from both pads), otherwise the executor inserts a park of the other arm first
  (`test_arm_clear_inserts_park`). The graphs already park before every arm switch; this guards unforeseen paths like
  the skip bug above.

- **Both learned grasps, planned places** (bundle `teacher_v3_gl300_gb500`: left it. 300, back it. 500, graphs v3,
  8 episodes per cell, CPU): TSR 1.0 on Task 1-2, 2-2, 2-3, 3-2 (D0 and D1), Task 1-3 D0 1.0 / D1 0.875, but
  **Task 3-3 D0 0.125 / D1 0.25** (PSR 0.42 / 0.50; 21 held pre-place transit timeouts). Trace: the back grasp holds the
  block 25.6 mm off-centre along it, 6.8 mm high and rotated 8.2 deg from the canonical grasp (block-frame metric
  below); with that held offset the release pose stays reachable but the pre-place pose 5 cm above it needs the wrist
  j7 beyond its limit (a pure +-25 mm offset along the block is reachable on every transfer except Task 3-3's last
  below -15 mm); the arm stops 19.6 mm / 5.3 deg short, just outside the transit's 5 deg arrival tolerance. Cause:
  nothing in grasp_lift's success asks for a centred grasp, and success_per_remaining_step (8 per step) makes any extra
  centring step expensive. The left grasp is off-centre too (one trace: 15 mm along the block, 1.6 deg).
  Grasp quality must be measured in the block frame (`tower_rl.grasp.grasp_relation_error`): a lifted block hangs
  tilted with the gripper (20 deg in one trace), and a comparison against the world-referenced canonical orientation
  reported that tilt as 15-20 deg of "grasp error".
  Fixes: (1) executor: transits go to the nearest reachable pose of their via-point (`reachable_pose`: IK within the
  controller's joint margins, FK of the solution when the via-point is out of reach; the contact skill corrects the
  rest on its way to the reachable contact pose); (2) grasp_lift quality gate: success needs the pinch within 10 mm
  along the block / 5 mm vertically of the canonical grasp and the gripper within 6 deg of the canonical orientation
  (fingers interchangeable); a lifted bad grasp ends the episode with -5; (3) place_release reset: half of the episodes
  hold the block with offsets as wide as that gate (uniform +-10 mm along, +-5 mm vertical, +-6 deg about the jaw axis,
  +-2 deg roll). The grasp nodes now log the grasp offset and rotation (block-frame metric) at every grasp.
  With fix (1) alone (same bundle, same seeds): Task 3-3 D0 TSR 0.125 -> 0.875, D1 0.25 -> 1.0 (GSR 1.0, ODR 0); the
  adjusted pre-place via-points sit up to 33 mm from the ideal ones and placements land at 3-6.6 mm (two misses,
  retried), so the grasp gate (2) and the wider place training (3) still matter.

- Step 8.2/8.3 instrumentation: the runner tracks each episode's closest arm-arm approach (joint 3-7 origins, flange,
  gripper base and pinch of both arms; the station scene reports no arm-arm contacts) and the feedback prints it;
  Task 1-2 D0 (arms alternate, 4 episodes): min 0.156 m (links are ~8 cm thick). Executor wall time with learned
  grasps and `reachable_pose`: mean 26 ms, max 296 ms per call (CPU; IK solves at transit creation).
- `scripts/rehearse_many.py`: parallel per-variant rehearsals of one bundle (N per tier, optional
  `TOWER_CONFIG_DIR=<bundle>/configs` for the Step 8.4 export check, perception noise, 30 Hz track) + summary.
- CPU budget: a CPU rehearsal process used 6-7 cores, slowing GPU training sharing the machine (8.7 -> 12.6 s per
  iteration with three rehearsals). Not PhysX (`/persistent/physics/numThreads` changed nothing) but torch's
  default intra-op threads on the executor's tiny tensors: `torch.set_num_threads(1)` in the CPU scripts gives 1.3
  cores per process and *faster* rehearsals (Task 3-2 batch 22.5 -> 15.2 s); rehearse_many also runs its processes
  at nice 15.

### Baseline before the final skills (2026-10-19 03:55)

Probe bundle (learned grasps left v3 it. 300 / back it. 500, learned left place v1 it. 400, planned back place) with
the current executor (attach/refresh, nearest-reachable via-points, fail-fast pre-grasps), all variants, 8 episodes
per cell, `rehearse_many.py` (5 min with 4 single-threaded processes): D0 TSR 1.0 on all six; D1 TSR 1.0 (2-2, 3-2,
3-3), 0.875 (1-3, 2-3), 0.75 (1-2); GSR 1.0 and ODR 0 everywhere. All four D1 failures are learned left places at
the tolerance edge on pad B: one at 4.86 mm pushed out when the next layer landed, three at 5.4-5.8 mm whose
re-grasps are out of the left arm's reach (failed at once; transfer skipped, no cascade). Already above the Step 7
targets, but with a planned back place.

### Quality-gated grasps in the pipeline (2026-10-19 04:30)

Fine-tunes with the grasp quality gate: left v4q (it. 300 -> 600) 94.7 % training-mix success, bad grasps 2.3 %;
back v2q (it. 500 -> 800) 10 % -> 97.9 %, bad grasps 1.5 % (it learned to grip at the block's centre height instead of
~8 mm high and ~20 mm toward one end). CPU rehearsal, all variants, 8 episodes per cell (both new grasps, place-left
v1, planned back place): D0 TSR 1.0 on all six, D1 1.0 except Task 1-3 / 2-3 0.875 (place misses), GSR 1.0, ODR 0.
Measured grasp quality on CPU (block-frame metric): left |dx| p50 1.4 mm / max 5.0, |dz| 2.0 / 4.7, rotation 0.5 /
5.2 deg; back 1.8 / 4.1 mm, 1.3 / 3.2 mm, 1.2 / 3.3 deg.
Bug found on the way: the executor attached the block as soon as the opening entered the holding range (44-56 mm)
while the fingers were still closing and the left policy still descending: attached offsets 12 mm off vertically,
which the 1 cm refresh gate then protected (Task 2-2 D0 TSR 0.75). Now it attaches once the fingers are on the
block's faces (opening within 2.5 mm of the block width) and the grasp node refreshes with a 2.5 cm gate until the
block is 2 cm up.

- **Arm-switch re-place** (executor): a place that misses by <= `arm_switch_m` (graph field, default 15 mm) whose own
  arm cannot re-grasp (pre-grasp out of reach) but the other arm can is re-grasped and re-placed by the other arm with
  its learned skills (inserted nodes: park, transit, grasp, transit, place, park; at most once per transfer; the graph
  continues after the transfer; on failure the arm is parked and the transfer skipped). Unit test
  `test_arm_switch_replace_schedule`. It did not fire in the 8-episode reruns below (no near misses there).
- Determinism: repeated CPU rehearsals with the same code, bundle and seeds are identical (also with different
  `OMP_NUM_THREADS`, PhysX worker counts, and through `rehearse_many.py`); but any code change shifts trajectories
  (Task 2-3 seed 1000000: the first grasp's offset -1.6 vs -2.0 mm), enough to flip a tolerance-edge place, so
  8-episode A/B comparisons are noise: the two D1 near misses above did not recur after an unrelated executor change.
  A thread test had written `/persistent/physics/numThreads = 2` into Kit's user config (a global setting of the
  Isaac install); removed again (results identical either way).
  Reproducibility is per process history: PhysX keeps some state across `StationWorld.reset` (contacts / solver
  warm starts), so batch b of a rehearsal depends on the b batches simulated before it in that process (0/28 episodes
  of single-batch reruns matched their original batches). `trace_episode.py --prior_batches b` replays the earlier
  batches of the same tier; a D1 batch of a "D0 D1" rehearsal also has all D0 batches in its history.
- Failure videos (Step 7.5): `trace_episode.py --video` renders the env's high + low evaluator cameras (upright,
  half resolution, node and time overlaid) into an mp4, one frame per executor call (contact skills in real time,
  transits as time-lapse). First clip `results/videos/t23_D0_5100001.mp4` (Task 2-3 D0, 60 s; GPU 109 W / 74 C at
  most). Cameras perturb the simulation history, so the rendered rerun did not repeat that episode's failure; the
  remaining failure videos need a rendered rehearsal that records failures as they happen (not done, to keep the GPU
  free of long renders given the earlier hard reset).
- Arm-switch A/B on the seven Task 2-3 D0 batches with left near misses (fresh-process reruns, 28 episodes): TSR
  26/28 with and 26/28 without it, PSR 81/84 vs 82/84. Neutral while the back grasp cannot re-grasp from a stack (7 of
  8 attempts in the N = 100 run timed out); kept, as it becomes useful once the grasps train on re-grasps.

### Step 7 acceptance: all-learned teacher, N = 100 per variant and tier (2026-10-19 07:40): PASSED

Bundle `results/bundles/teacher_v4` (sha256 99f7caa9e5e44bfb332f2a0a6db765f47534ca49207fbb8fb00b8d1df6708583):
grasp left v4q (gated), grasp back v2q (gated), place left v1 it. 400, place back from scratch it. 600, graphs v4,
no planned fallbacks. CPU station scene (evaluator physics), privileged state, 10 Hz track, seed 5000,
`rehearse_many.py` (4 single-threaded processes, 54 min; `results/rehearsal_final/teacher_v4/summary.md`):

| variant | D0 TSR (Wilson 95 %) | D1 TSR | GSR | PSR (D0 / D1) | ODR | mean s |
|---|---|---|---|---|---|---|
| task1-2 | 1.00 (0.96-1.00) | 1.00 | 1.00 | 1.00 / 1.00 | 0.00 | 82 |
| task1-3 | 1.00 (0.96-1.00) | 1.00 | 1.00 | 1.00 / 1.00 | 0.00 | 128 |
| task2-2 | 0.97 (0.92-0.99) | 1.00 | 1.00 | 0.98 / 1.00 | 0.00 | 45-50 |
| task2-3 | 0.92 (0.85-0.96) | 0.97 | 1.00 | 0.96 / 0.98 | <= 0.01 | 72-78 |
| task3-2 | 1.00 (0.96-1.00) | 1.00 | 1.00 | 1.00 / 1.00 | 0.00 | 37 |
| task3-3 | 1.00 (0.96-1.00) | 1.00 | 1.00 | 1.00 / 1.00 | 0.00 | 56 |

Targets (TSR >= 0.8 / 0.6 on D0, >= 0.6 / 0.4 on D1 for 2- / 3-layer variants, ODR < 0.1): all 12 cells pass;
1186 / 1200 episodes succeed. The 14 failures are placements at the tolerance edge: Task 2-3 D0 is the left arm's
layer-2 place on pad B (8 / 100 at 5.1-6.9 mm) followed by an arm-switch re-grasp the back grasp cannot do.

### Step 8 with the final bundle (2026-10-19 08:30)

- **8.4 export check: passed.** Fresh processes with `TOWER_CONFIG_DIR=teacher_v4/configs` (configs from the bundle
  only), D1, 24 episodes per variant, new seeds (7000): TSR 1.00 on five variants, 0.96 on Task 2-3 (N = 100 run:
  0.97); ODR 0.
- **8.3 timing**: executor <= 0.3 s per call on CPU (mean ~26 ms), well under one second per 10 Hz step.
- **8.2 both arms**: closest arm-arm approach >= 0.156 m in every rehearsed episode inspected (corrected in "Step 8.1
  round 2": the N = 100 runs have rare closer approaches); `arm_clear` parks the other arm before any transit into its
  tower; GSR 1.0 in all 1200 acceptance episodes.
- **30 Hz track** (24 episodes per cell, seed 8000): all variants 0.96-1.00 except **Task 1-3: D0 0.42, D1 0.33**
  (ODR 0.33, episodes 176-183 s against 128 s at 10 Hz). The executor's 30 Hz path (transits planned at 1/30 s, 10 Hz
  skill decisions expanded into 3 interpolated rows) was never tuned with learned skills: **submit on the 10 Hz track**
  (validated: all 12 cells pass at N = 100) until the 30 Hz path is diagnosed.
- **8.1 noise in rehearsal** (D1, 24 episodes, "realistic" model: independent 1 mm / 1 deg + shared 3 mm / 3 deg bias,
  jitter, 2 % dropouts, 0.5 % outliers): **TSR 0.00-0.75** (Task 3-2 0.75, 3-3 0.46, 2-2 0.42, 2-3 0.33, 1-2 0.12,
  1-3 0.00; ODR up to 0.36). The executor judges 59 % of the places failed (1.2 % without noise), 55 / 202 of them
  > 7 mm off, and the re-grasps that follow disturb the stacks. Changes and hypotheses so far: the destination pad's
  yaw now comes from the line through the three foundation centres instead of one block's perceived yaw (immune to
  per-block orientation errors; noise-free goals unchanged): Task 1-3 0.00 -> 0.08, Task 2-3 0.33 -> 0.33, so the
  goal was not the main loss. Open hypotheses for the next round, to be checked with traces under noise: (a) the
  attached held-block offset is measured in the gripper frame, so a shared world-frame perception bias rotates with
  the gripper during transport and no longer cancels against the goal's (refresh should undo it; unverified);
  (b) single noisy samples decide the place check (1 mm jitter, outliers) and trigger re-grasps of well-placed
  blocks; (c) the learned back place is the least noise-robust skill (skill eval D1 + 8.1 noise: 60 %).
- **Bundle `teacher_v5`** (sha256 bd7f94611d5a92f69204c77646e6ab0287e076f7f0e5c97e162493463debead9) = teacher_v4 with
  the noise-tuned back grasp v3n: noise-free rehearsal (24 per cell) 1.00 everywhere except Task 2-3 0.96 / 0.96;
  the candidate for the next round's noise work.

### Step 8.1 round 2: the executor under perception noise (2026-10-19 10:20)

Built: `tower_rl/graph/estimates.py` (`StaticPose`, `PoseWindow`), executor changes below, `scripts/noise_diag.py`
(per place: the executor's estimates against simulator truth; `--probe_bias` fixed biases, `--graph_set` graph
overrides), `scripts/summarize_noise_diag.py`, `rehearse.py` / `rehearse_many.py --graph_set`, `tests/test_estimates.py`
(8 tests; whole suite 119 passed, 3 skipped), place env option `obs_noise.shared_pos_m` / `shared_yaw_deg`.

**Diagnosis** (teacher_v5, D1, seed 6000, 8 episodes per variant, "realistic" noise = independent 1 mm / 1 deg + shared
3 mm / 3 deg + jitter + dropouts + outliers), which settles this morning's open hypotheses:
- (b) confirmed: single noisy poses decided the place check, and the per-row displacement test for "static" never
  held under 1 mm jitter, so the verdict fell on one sample: in Task 3-3, 11 of 30 places in tolerance were judged
  failed, and each false failure re-grasps a placed block from the destination stack.
- Only 14 / 30 (Task 3-3) and 11 / 29 (Task 2-3) held blocks were attached at the release: the attach rule ("one
  perceived pose within 3 mm of the node's first one") compared two noisy samples. The rest travelled and were placed
  on raw perception, dropouts included.
- (a) minor: attached estimates stay within 1-2.7 mm of the fresh measurements (the refresh keeps up).
- New, the main residual: along the gripper's jaw axis the true placement error equals the goal's perception error
  (correlation 0.96, Task 3-3). The place policies ignore the held block's perceived offset along the jaw (in
  training the fingers centre the block, so that offset is pure observation noise) and put the pinch over the
  perceived goal, so an error that the held block shares with the foundation does not cancel along the jaw. A probe
  with a fixed +3 mm bias on the held blocks only confirmed it for both arms (true error ~0 along the jaw, ~-3 mm along
  the block, where the policies do use the perceived offset).

**Executor changes** (README "Perceived poses"): per-block rest-pose estimates (median of the recent consistent poses,
one-call outliers held back, 2 consecutive far poses = the block moved, confidence < 0.5 ignored; window 30 for
foundations, 10 otherwise) for every target computed once (pre-grasps, goals, `arm_clear`, reachability, disturbance
monitor); goal orientation from the foundation blocks' perceived yaws (circular mean; cancels a common orientation
error), in-pad positions from the line through their centres; attach at the first gripping step from the rest
estimate; `refresh_attached` gate 5 mm plus 3 agreeing measurements beyond it to accept a slip (was 1 cm / 2.5 cm, which
let half of the 1 cm outliers through); place check on the median of the last 10 perceived poses, once released, open,
retreated, >= 10 rows after the release and at rest (window medians within 2.5 mm), at the latest 30 rows after it;
optional `perception.jaw_to_goal` (graph edit 3, below).

| same seeds, 8 episodes | Task 3-3 TSR | places > 5 mm at release | false "failed" | Task 2-3 TSR | back / left > 5 mm |
|---|---|---|---|---|---|
| teacher_v5, old executor | 7 / 8 | 0.10 | 11 / 30 | 5 / 8 | 0.29 / 0.25 |
| new executor | 6 / 8 | 0.15 | 1 / 27 | 5 / 8 | 0.07 / 0.30 |
| + `jaw_to_goal` 1 | 7 / 8 | 0.04 | 1 / 25 | 5 / 8 | 0.00 / 0.33 |
| + whole in-hand offset to the goal (rejected) | 8 / 8 | 0.00 | 1 / 23 | 4 / 8 | 0.15 / 0.50 |

Noise-free regression of the new executor (teacher_v5, 24 per cell, seed 5000): identical to the old one (1.00
everywhere, Task 2-3 D0 0.96, the known left-arm near miss on pad B). Executor time per call unchanged (mean ~30 ms,
max <= 0.34 s).

**Graph edit 3** (`graph/edits/2026-10-19_3/`, graphs v5): `"perception": {"jaw_to_goal": 0.5}`: half of the held
block's perceived jaw offset moves from the block to the place policy's goal. w = 1 cancels errors common to all
blocks (realistic model: sd of the jaw error 3.0 -> 1.2 mm); w = 0 is best for independent per-block errors (build-file
model: 1.4 mm at w = 0, 3.4 mm at w = 1, where the 3 mm error along the block fails 2 / 8 episodes either way). 0.5
until the pose estimator's error covariance is measured (optimum: Cov(goal error, held jaw error) / Var(held jaw
error)). Noise-free it is a no-op (the true jaw offset is ~0.1 mm).

**Left grasp v5n** (from v4q, 250 iterations with the build-file noise model like the back grasp v3n): D0 **100 %**,
D1 98.8 %, D1 + 8.1 noise **98.8 %** (v4q: 98.6 / 98.2 / 94.6 %).

**Bundle `teacher_v6`** (sha256 97904996e8699ea0cc9a426c33a5588367b7a4735dad1a42ee12f4d7f98950ac): grasps left v5n /
back v3n, places left v1 it. 400 / back scratch it. 600, graphs v5.

- **Step 7 acceptance, noise-free, N = 100 per cell, configs from the bundle (also the 8.4 export check): all 12
  cells pass, 1182 / 1200** (teacher_v4: 1186). Per cell D0 / D1: Task 1-2 1.00 / 0.99, 1-3 0.99 / **0.94** (v4 1.00),
  2-2 0.98 / 0.97, 2-3 **1.00** (v4 0.92) / 0.97, 3-2 1.00 / 0.98, 3-3 1.00 / 1.00; ODR <= 0.02.
  The noise-tuned grasps leave about twice the in-hand rotation (Task 1-3, rotation from the canonical grasp p50 / p90 /
  max: left 0.5 / 1.3 / 2.6 deg -> 1.0 / 2.2 / 4.2, back 0.9 / 1.6 / 2.1 -> 1.6 / 2.2 / 6.7; trained with 3 deg yaw
  noise, they cannot see smaller rotations), and the places do not fully remove it: 0.7 % of the left arm's first
  places end beyond 5 deg (v4: 0), and the re-grasps of those blocks from the destination stack cascade (Task 1-3 D1).
  For privileged use (DAgger labels on simulator truth) teacher_v4 stays the most precise bundle; teacher_v6 is the one
  to run on estimated poses.
- **Realistic perception noise** (D1, 24 episodes, seed 6000), against this morning's teacher_v4 / old executor:

| variant | TSR (Wilson 95 %) | before | GSR | PSR | LHPR | ODR (before) | mean s |
|---|---|---|---|---|---|---|---|
| task1-2 | 0.46 (0.28-0.65) | 0.12 | 0.97 | 0.80 | 0.89 | 0.07 (0.31) | 152 |
| task1-3 | 0.29 (0.15-0.49) | 0.00 | 0.98 | 0.71 | 0.84 | 0.05 (0.36) | 187 |
| task2-2 | 0.67 (0.47-0.82) | 0.42 | 0.98 | 0.81 | 0.90 | 0.02 (0.09) | 100 |
| task2-3 | 0.79 (0.60-0.91) | 0.33 | 0.99 | 0.90 | 0.94 | 0.03 (0.14) | 98 |
| task3-2 | 0.83 (0.64-0.93) | 0.75 | 1.00 | 0.92 | 0.96 | 0.00 (0.05) | 66 |
| task3-3 | 0.88 (0.69-0.96) | 0.46 | 1.00 | 0.94 | 0.97 | 0.00 (0.04) | 76 |

  The remaining losses are the left arm's places: in Task 1-3 13 / 33 attempts of its layer-1 place t1 failed, the back
  arm's places 48 / 48 succeeded. Single-component probes (Task 2-3) isolate it: jitter alone or a shared 3 deg yaw
  error alone cost nothing; a shared 3 mm position error leaks into the left place along the block (true error about
  -1.3 x the error's component along the block; the back place cancels it). The place policies were trained with
  independent goal and block errors only, so they rely partly on absolute positions; the left one more.
- **Build-file noise model** (independent 3 mm / 3 deg per block + jitter, dropouts, outliers; D1, 24 episodes):
  TSR 0.08-0.33 (Task 1-2 0.08, 1-3 0.12, 2-2 0.33, 2-3 0.25, 3-2 0.29, 3-3 0.21), PSR 0.51-0.68, GSR 0.94-1.00, ODR
  <= 0.09. Independent 3 mm errors between a block and its goal put about a third of the places outside 5 mm, and no
  executor or policy can see them: the requirement on the pose estimator stays ~1 mm relative accuracy between a held
  block and the destination foundation, or errors that are largely common to both.
- **Arm-arm distance (correction of 8.2 above)**: over the 2400 episodes of both N = 100 acceptance runs, 13 brought
  the arms within 10 cm of each other (closest 2.6 cm; `teacher_v4` 3, `teacher_v6` 10), all in failed episodes and 10
  of them on the arm-switch recovery path; the median per cell stays ~0.16 m.
- Arm-switch re-place: its re-grasp succeeds in 7 of 138 attempts over all rehearsals so far, and it can disturb the
  stack; switching it off under the realistic noise gave Task 1-3 0.25 (vs 0.29) and Task 2-3 0.71 (vs 0.79) with no
  shorter episodes, so it stays.
- **Left place with a shared training error (v3s, rejected)**: v1 it. 400 fine-tuned 300 iterations with
  `obs_noise.shared_pos_m` 3 mm / `shared_yaw_deg` 3 deg on top of the independent 2 mm / 2 deg (new, off-by-default
  place env option), `jaw_to_goal` 0 for the left arm (per-arm weights are now allowed). Realistic noise (D1, 24,
  seed 6000): Task 1-2 0.75 (teacher_v6 0.46), 2-2 0.79 (0.67), 2-3 0.83 (0.79), but **Task 1-3 0.00** (0.29), and
  noise-free Task 1-3 D1 **0.04** (1.00): it misses the goals 50 mm off the pad centre (Task 1's t1 / t4 / t5 places,
  6-15 mm). Skill env: 96.6 / 97.4 % D0 / D1 without obs noise (v1: 99.6 / 99.0), 84 / 87 % with the training noise
  (v1: 92.6 / 92.2). The left place's sensitivity to shared position errors stays open: the next attempt should keep the
  wide offsets and the off-centre goals in the fine-tune and be judged on Task 1-3 first, ideally against the pose
  estimator's measured error model.

### Final skills: Step 5 / 6 acceptance and Step 8.1 skill noise (2026-10-19 06:50)

GPU light scene, deterministic actors, 500 episodes per row (`results/skills_eval/final_*.json`); "noisy" = the training
observation noise (build file: acceptance "with the noisy actor observations"), "+ 8.1 noise" = the build file's
provisional perception model (independent 3 mm / 3 deg bias, jitter, 2 % dropouts of 0.5 s, 1 cm outliers).

| skill (checkpoint) | D0 | D1 | D1 + 8.1 noise | D0 / D1 without obs noise | disturbance |
|---|---|---|---|---|---|
| grasp left (v4q, gated) | **98.6 %** (97.1-99.3) | **98.2 %** (96.6-99.1) | 94.6 % (-3.6) | | 0.008 / 0.006 |
| grasp back (v2q, gated) | **99.2 %** (98.0-99.7) | **99.6 %** (98.6-99.9) | 84.2 % (-15.4; 74 drops) | | 0.0 / 0.002 |
| place left (v1 it. 400) | 92.6 % (90.0-94.6) | **92.2 %** (89.5-94.2) | 75.4 % (-17) | 99.6 % / 99.0 % | 0.002 / 0.002 |
| place back (scratch it. 600) | 87.2 % (84.0-89.9) | 84.8 % (81.4-87.7) | 60.0 % (-25) | 99.8 % / 100 % | 0.0 / 0.0 |

Grasp quality (gated grasps, D1): left |dx| p50/p90 2.0/4.9 mm, |dz| 1.2/2.6 mm, 0.9/2.0 deg; back 2.4/6.1 mm,
1.1/2.5 mm, 1.7/3.1 deg.
- **Step 5 passed for both arms** (>= 95 % D0, >= 85 % D1, disturbance < 0.1), now with the quality gate.
- **Step 6**: without observation noise both places are at 99-100 % on D0 and D1; with the training noise they reach
  92 % (left) and 85-87 % (back), below the 95 % D0 target. That gap is the observation noise (2.8 mm relative
  block-to-goal noise against a 5 mm tolerance, see above), not the policies' precision: on CPU with exact poses the
  executor places at p50 2.2 mm (left). Recorded as a deviation (13).
- **Step 8.1**: the left grasp is robust (-3.6 points); the back grasp drops blocks when its target pose freezes or
  jumps during the lift (-15 points; in the executor the held-block tracking replaces that pose once the fingers are
  on the block); the places lose 17-25 points, dominated by the 3 mm independent bias (no policy can place a
  misperceived block). Per the build file the back grasp should be retrained with these noises (dropouts and
  outliers in training): next round.
- **Step 8.1 retraining, back grasp** (`*_v3n_noise`, from v2q, 250 iterations with the provisional model in the
  skill env: bias 3 mm / 3 deg, jitter, dropouts 2 % x 0.5 s, outliers 0.5 % x 1 cm; `obs_noise.dropout_p` /
  `outlier_p` are new, off-by-default skill-env options): D0 99.2 %, D1 100 %, **D1 + 8.1 noise 99.2 %** (was 84.2 %).
  The left grasp's -3.6 points are within the build file's 5-point rule.

### Step 8.1 preview: perception noise with learned skills (2026-10-19)

Probe bundle (learned grasps both arms, learned left place v1, planned back place), Task 2-3 D1, 8 episodes, seed 1000,
one noise component at a time (`rehearse.py --perception_noise --noise ...`):

| noise on all perceived block poses | TSR | PSR | ODR |
|---|---|---|---|
| none | 1.0 | 1.0 | 0 |
| shared bias 3 mm / 3 deg (all blocks of the episode) | 0.75 | 0.92 | 0 |
| jitter 1 mm / 0.5 deg per call | 1.0 | 1.0 | 0 |
| **dropouts** 2 % per call, pose frozen 0.5 s | **0.375** | 0.58 | 0.28 |
| outliers 0.5 %, 1 cm, one call | 1.0 | 1.0 | 0 |
| independent bias 1 mm / 1 deg per block | 1.0 | 1.0 | 0 |
| build-file model (independent 3 mm / 3 deg + all of the above) | 0.25 | 0.42 | 0.11 |

Dropouts hurt through the *held* block: a stale pose of a block that moves with the gripper misleads the grasp
policy during the lift, gets recorded as the held offset (release pose centimetres off) and misleads the place policy.
**Fix (executor)**: a block is *attached* to the gripper once it is held and still rests where it was perceived
(before it moves); while attached its pose is the kinematic prediction pinch x offset (`EnvRun.block_est`), detached
when the gripper opens (place release, skip/retry, park). Dropouts: TSR 0.375 -> 0.5 (place only) -> **0.875**
(attach/detach everywhere; GSR 1.0, PSR 0.96). While attached, the offset follows fresh measurements
(`refresh_attached`: confidence >= 0.5, within 1 cm of the prediction, gain 0.3 per call) so a block slipping in
the fingers is still tracked; the noise model marks a dropout's frozen pose with confidence 0 (as an estimator that
lost a detection would) and outliers fail the 1 cm gate. The remaining large loss is the independent per-block bias: 3 mm
per block moves placements by more than the 5 mm tolerance often (build-file model with attach: TSR 0.375, PSR 0.46)
and no policy can undo a misperceived block. Requirement for the perception module: ~1 mm relative accuracy between
a block and the destination foundation; shared errors (calibration, viewpoint) are mostly tolerated.

Deviations:
11. Place settle check: the build file's (iii) "all destination blocks static" at the end of a fixed 1.0 s after the
    release is replaced by the evaluator's P semantics (target block released, in tolerance and static for 1.0 s
    continuously; placed blocks < 10 mm displacement), with "static" measured from displacement per 0.1 s step (GPU
    TGS velocity jitter of blocks resting on dynamic blocks; see above). The retreat requirement is 3 cm (build file
    5 cm: the executor's next transit rises first anyway). Release from 1-6 cm above the release pose instead of "3 cm
    / 5 deg around the pre-place pose" (the pre-place pose is 5 cm above it, so this covers it), no adaptive curriculum
    (the policy found the release without it).
12. No support-polygon term in the place reward: the goals are fixed by the layout and the 5 mm / 5 deg tolerance
    keeps every stack's centre of mass far inside its support (tolerance sweep: 70/70 stacks with 0-20 mm offsets
    stand), so the term would be zero in every reachable state.
13. Step 6 acceptance with noisy observations not met (place left 92.6 % / back 87.2 % on D0 vs >= 95 %): the
    observation noise of the build file's own place design (goal sigma 2 mm, plus the held block's noise) caps any
    policy near 90 % at a 5 mm tolerance; without observation noise the places score 99-100 %, and the CPU rehearsals
    (exact poses) are the operative test (Step 7). Not retrained against the cap.

## 2026-10-19/20 (session 3): Tasks 4 and 5 in the agentic teacher

Goal (user, 2026-10-19 22:00): add Task 4 (full tower push-down) and Task 5 (tower insertion) to the teacher with PPO
contact skills executed in sequence by the graph: `push` (Task 4), and `insert` + `push_in` (Task 5, after the existing
grasp). Code backed up before any change: `results/backups/code_2026-10-19_2225_pre_task45.tgz` (`tower_rl/{eval,graph,
sim,skills}` were untracked in git at that point). All Tasks 1-3 behaviour is kept bit-identical (defaults; the
`transfer` family code paths are unchanged); the 119 + 46 new pure-python tests pass.

### Ground truth (re-checked 2026-10-19 evening)

Tasks 4 / 5 are official: SimData PR #1 merged 2026-10-13 (commit 1a59248e; the docs pin its PR head 5ad6adc9, same 531
files), docs "Task prompts" / "Getting started" (all-ten selection 100 episodes, "Task1-3 · six tasks" 60), leaderboard
Space (Pushdown / Insertion tabs since 2026-10-13; "Overall scoring formula coming soon"). Stale: the workshop
challenge page (3 families, 331 episodes) and GitHub `yqi19/corl_challenge` (2026-10-04).

### Data and layouts

- `data/meta_tasks45.jsonl`: metadata of all 200 Task 4/5 episodes (`scripts/extract_meta_remote.py`, now also block
  sizes / masses, the generator's finger effort, compact events); `data/subset/task{4,5}-*`: 10 episodes per variant
  (arrays now include `sim/controller_qpos`, `sim/gripper_joint_positions_m`, `sim/gripper_requested_width_m`).
- `configs/layouts/task4-*.yaml`, `task5-*.yaml` (`scripts/build_layouts.py --t45`; Tasks 1-3 YAMLs untouched):
  family, Task 4 `pushes` (block, arm, world direction, demo push stats), Task 5 `transfers` (slot centre goal, side
  blocks, final rest height, demo entry geometry), per-block `dims_m` / `mass_kg` / `role`, `physics` (collider, finger
  effort).
- Task 4: full layers of three (alternating yaw) on pad A, 2 or 3 layers; push every block off, top layer first, row slot
  2 -> 0, prescribed order; arms: 4-2 L2 back, L1 left; 4-3 L3 left, L2 back, L1 left; push along the block's long axis
  away from the arm (left +x, back +y), closed fingers (width 0) as a tool, approach tilted toward the push (left 45 deg,
  back 30 deg), pinch ~100 mm behind the block centre and 7-16 mm above it at the start (first contact at -86.5 mm),
  0.15 / 0.075 m/s; demos 116-122 s (4-2), 171-176 s (4-3).
- Task 5: insert blocks on a stack on pad B; the destination tower on pad A: foundation, interior layers of two 32 mm
  side blocks (0.11264 kg) with an empty middle slot (2 mm lateral, 2 mm vertical clearance), a 3-block roof. Inserts
  highest gap first; 5-2: one insert (left arm), 5-3: layer 2 (back) then layer 1 (left). Demo recipe: centred grasp,
  block to 170 mm out on the slot axis, held entry to 90 mm (~0.5-0.8 mm per 0.1 s), release, fingers at 0.02 m behind
  the outer end face, push home (~2.2 mm per 0.1 s; residual <= 1.5 mm). Task 5-3 runs on exact cuboid colliders
  (`station_cuboid_collision.usda`).

### Findings that matter beyond Tasks 4 / 5

1. **3 N finger effort.** Every Task 4/5 demo ran with a 3 N finger effort limit (`generator_gripper_effort_application:
   applied 3.0, source profile 22.0`), and every organizer scene descriptor says 3.0 (the Task 2-3 cfg08 descriptor
   too). The Tasks 1-3 data was generated at 22 N. Open question for the Tasks 1-3 teacher (its closed command 0.040 m
   would ask ~3.9 N per finger).
2. **The generator's applied gripper width differs from `leader/qpos` while gripping**: `sim/controller_qpos` holds
   ~43.8 mm (2.4 N per finger at kp 800, below the 3 N limit) while `leader/qpos` says 0. Commanding 0 (or 0.040) at 3 N
   saturates both fingers: nothing centres the block between them and it floats sideways with the fingers (a replayed
   Task 5-2 grasp drifted 21 mm along the jaw axis and rammed the tower). Replays now use the applied channel
   (`Episode.qpos_applied`); the Task 4/5 graphs command 0.0438 m while holding (`gripper_closed_width`).
3. Task 5-3's demos used the published station's finger friction 2.4 / 2.0 (the Sep-9 station's 1.2 / 1.0 made the
   replayed push-in tip the insert and drag the roof).
4. USD prim names cannot contain `+` / `-` (Task 5 block ids): prim paths are sanitized (`station_scene.prim_name`).
5. Skill scenes with 15 blocks per env exceed PhysX's 64K-material limit at 4096 envs (one Wood material per block):
   3072 envs. On the CPU device, pose writes to kinematic bodies do not stick (the foundations jumped back to their
   spawn poses): skill envs train and are checked on the GPU only.

### Station scene and local evaluator

- Per-block assets: the cuboid wrapper `assets_m/tower_block_cuboid_m.usda` (the organizers' Task 5 overlay: hull
  collision off, exact box collider, same Wood material), 32 mm side blocks (scale + mass), finger effort per variant.
- Domain sampling: Task 4 full layers get no per-block jitter (packed blocks would interpenetrate; the generator jitters
  none; layers move rigidly); Task 5's destination tower stays at its layout pose (docs: Task 5 pose cases vary only the
  source blocks); heights from the layout for both.
- Operation semantics (`tower_rl/eval/operations.py`): `PushTracker` (Task 4: G = moved >= 2 cm along its direction in
  order, P = left the tower (centre >= 0.16 m from the axis, or flat on the table >= 0.09 m out) and static 1 s, the
  remaining tower must stay within 10 mm / the collapse rule, disturbances on tower blocks only, 2 s final settle);
  Task 5 uses the transfer tracker (slot goal, 5 mm / 5 deg) with the tower's side / roof blocks as structure and the
  2 s final settle.
- **Demo replay (CPU, 30 Hz, finger friction 2.4 / 2.0, applied gripper channel): Task 4-2 4/4, 4-3 4/4, 5-2 4/4 (TSR
  1.0, ODR 0); 5-3: grasp and held entry track the recording within 0.3 mm, the open-loop push-in is chaotic (one
  process reproduced the recording within 4 mm, the same episode in another process dragged the upper tower).**

### Task 4: push skill and executor

- `tower_rl/push.py` (pusher geometry shared by env and executor), `configs/skills/push.yaml`, `push_priors.yaml`
  (demo postures as IK seeds), `tower_rl/skills/push_env.py` (65-D actor, 159-D critic; 6-D delta-pose action, fingers
  closed), `scripts/oracle_push.py` (scripted straight push: 74-82 % left, 70 % back on GPU).
- Training: v1 collapsed to inaction (a whole push was worth ~6 vs failure penalties); v2 rebalanced (progress 150/m,
  fail 2); v3 adds the stricter departure rule and "mid-push" resets (block already slid 0-6 cm). Left v3: success
  0.30 (it. 39) -> 0.65 (79) -> 0.82 (111) -> 0.85 (135), broken 0.10.
- Executor: transit target `prepush` (fingers closed into the tool), `PushNode` (learned, or a planned straight push
  for pipeline tests): ends when the block left the tower, fails on structure > 10 mm / sideways shove / timeout;
  graphs `task4-*.json`.
- **Departure rule fix** (trace of Task 4-3): "centre >= 0.15 m or fell 2 cm" left layer-3 blocks leaning on the tower;
  the back arm's layer-2 push then dragged them and the layer. Now: centre >= 0.16 m, or flat on the table >= 0.09 m out.
- Rehearsal with the planned push (CPU): 4-2 D0 TSR 1.00 (4/4, ODR 0); 4-3 D0 0.75 (36/36 pushes), D1 0.25.

### Task 5: executor, skill envs

- Graph per insert: transit(pregrasp) -> grasp_lift -> transit(held, `preinsert`) -> insert -> transit(`prepushin`) ->
  push_in, park between arms; the slot pose from the side blocks' rest estimates (midpoint, their axis); insert /
  push_in skip on failure (a re-grasp from inside the tower wrecks it).
- Planned (scripted, closed-loop) insert and push_in, fixed from traces: (1) aim from the perceived block, not the
  stale held offset (the block hangs 6-9 mm low in the 3 N fingers); (2) level the block's pitch; (3) hold the depth
  while misaligned at the entrance; (4) per-variant entry lift (5-3 cuboid +1 mm as the demos; 5-2 hull 0); (5) back
  off 2 cm along the axis before rising after the release (rising at once dragged the roof); (6) servo the pusher height
  (it sagged 3 mm and the insert caught the far side block of the empty gap below).
- Skill envs (`tower_rl/skills/task5_env.py`): `InsertEnv` (7-D action incl. release; 55-D actor) and `PushInEnv`
  (6-D action); scene: 4 side slots (32 mm), roof, insert, an insert already in place (5-3 stage 1); cuboid colliders,
  3 N fingers. Oracles (`scripts/oracle_task5.py`, GPU): push_in 87.5 %; insert 48 % left / 31 % back (failures 11 mm
  into the slot that the CPU station scene does not show: GPU contact behaviour, see deviation 14).

Deviations:
14. Task 5 skills train on GPU physics, which treats the 1-2 mm insertion clearances differently from the CPU
    evaluator (the scripted insert succeeds on CPU, 31-48 % on GPU): the learned insert is judged in the CPU rehearsal,
    and the planned closed-loop insert stays as the fallback.

### Night of 2026-10-20 (00:30-): learned push, bundle, Task 5 fixes

- **Push left v3** (`results/skills/push_left/2026-10-19_2338_4169273dc7_v3`, 300 it., 51 min): skill eval 500 episodes
  D0 0.984 (0.969-0.992), D1 0.966 (0.946-0.979; per case: combined 0.91, high_friction 0.93, shifted 0.93,
  irregular 0.95, lightweight 0.96, others 0.98-1.0); ~12 steps per push.
- **Bundle `results/bundles/teacher_t45_v1`**: the teacher_v6 skills (grasp_lift left v5n / back v3n, place_release
  left v1 it. 400 / back scratch; re-exported: identical weights and outputs, only TorchScript serialization ids differ)
  + push left v3; push back, insert, push_in planned. Tasks 1-3 graphs and configs identical to teacher_v6 (one
  comment line in place_release.yaml differs).
- **Task 4 rehearsal (CPU, 20 episodes each, seed 7000), learned left push + planned back push**: 4-2 D0 1.00 (ODR 0),
  D1 0.65 (ODR 0.16); 4-3 D0 0.90 (ODR 0), D1 0.65 (ODR 0.09). D1 failures: almost all high_friction / combined: the
  planned back push of layer 2 drags the tower (structure > 10 mm; 4-2: 31 of 36 failed nodes) and some learned left
  layer-1 pushes "shove" in high friction. Push back is training (v1 300 it. + 250 continuation).
- **Heavy blocks fell out of the 3 N grasp** (Task 5-2 D1: "empty gripper" 9 / 16 grasp attempts, heavy and
  uneven_weights): the trace shows a centred grasp (1-2 mm) whose fingers slide up the block on the first lift row. The
  grip itself is strong enough: replaying a demo with the insert block's mass x1.0-2.0 and the held command 43.8 / 43 /
  42.5 / 40 mm held every block (`scripts/grip_capacity.py`; the station's gripper material has no combine mode authored:
  average). The difference is the lift: the planned grasp stepped 1 cm per 0.1 s row, the demos lift smoothly (peak
  46 mm/s, ~0.1 m/s^2). The Task 5 graphs now ramp the planned lift (`lift_v_m_s` 0.04, `lift_a_m_s2` 0.1; Tasks 1-3
  unchanged): Task 5-2 D1 grasp success 0.70 -> 1.00, TSR 0.40 -> 0.60 (10 episodes, seed 2000).
- **The learned Tasks 1-3 grasp in Task 5**: grasps 8 / 8 at pad B, but leaves the block 2-3 deg rotated in the fingers
  and the planned insert (gates: pitch < 0.3 deg, yaw < 0.4 deg at the entrance) then timed out 7 / 8 (5-2 D0, TSR 0).
  Task 5 graphs use the planned grasp (`grasp_planned: true`; `--graph_set '{"grasp_planned": false}'` for the
  learned one, e.g. with a learned insert, which trains with grasp rotations up to 4 deg).
- **Demo push-in speed was misread**: all 18 push-ins of the subset run at 36-40 mm/s (peaks 42-48); the held entry at
  8.8 mm/s (5-2) and 4.4 mm/s (5-3). At our 20 mm/s the 5-3 layer-2 insert, after crossing the empty layer-1 gap, butts
  into the near top edge of the far layer-1 side block and shoves it and the upper tower ~14 mm (trace); in the demos
  the leading end rides over that edge with <= 1 mm disturbance. Planned push-in 0.02 -> 0.04 m/s, push_in skill
  action limit near the face 3 -> 6 mm per step. 5-3 D0 (8 episodes, seed 1000): TSR 0.17 -> 0.50, first push-in 6 / 8.
- Push-in end: the planned push ended when its target reached "block centred"; the arm lags 4-10 mm at 4 cm/s, so it
  now holds and nudges on (1 mm per row, <= 6 mm) while the block is static and short. (The 5-3 layer-1 push-in still
  ends 5-12 mm short: under investigation.)
- Rehearsal, all planned skills with the fixes (8 episodes, seed 1000): 5-2 D0 0.875, D1 0.50; 5-3 D0 0.375, D1 0.25.
- **Push-in end, revisited** (trace 5-3 seed 1000000, layer 1): not arm lag. At 4 mm per 10 Hz row the pusher kicks the
  light block ~8 mm ahead every other row (yaw swinging +-2 deg) and it overshot the centre by ~5 mm. The planned
  push-in now slows with the block's remaining distance (`pushin_decel_per_s` 1.0: 40 mm/s from 40 mm out, >= 8 mm/s)
  and keeps the hold-and-nudge. 8 episodes each, seed 1000: 5-2 D0 1.00, D1 0.875; 5-3 D0 0.75, D1 0.125.
- **5-3 layer-2 gap crossing is a tie-break**: at the moment the insert's leading end meets the far layer-1 side block,
  the insert's bottom and that block's top are both at 62.000 mm (trace, 1 um resolution). PhysX box-box contact then
  picks a horizontal normal on that edge about half the time and the push shoves the side block and the upper tower
  (> 10 mm). The demos crossed with <= 1 mm (the generator's physics broke the tie upward). Speed / height variants
  (16 episodes each, i0 push-in OK): 40 mm/s 8 / 16; 60 mm/s 3 / 16; pusher -4 mm 5 / 16; +4 mm 4 / 16 (offsets also
  hurt the layer-1 push-in). The prompt fixes the order ("highest gap first"), so the layer-2 insert must cross the open
  gap. Being tried: in the crossing window the pusher slides down the end face (+2 -> -3 mm) so that, once the nose is
  blocked and the push force rises, friction drags the tail down (needs > ~0.55 N) and the nose tips up over the edge
  (`gap_below` push-in nodes; `cross_window_m`, `cross_z_m`).
- **Crossing window works**: 5-3 with the descending pusher (2 seeds x 8 episodes per tier): D0 16 / 16 (layer-2 push-in
  16 / 16), D1 11 / 16 (layer-2 push-in 14 / 16; other D1 failures: layer-1 insert "structure" under shifted_assembly /
  uneven_weights, one drop).
- **Full 3 N hold**: a combined-case block (mass x1.8, slippery) slid 14 mm down the fingers during the gentle lift and
  dropped in the held transit: the hold command (finger kp 800 N/m) gave 2.5 N per finger. `held_close_width_m`
  0.0438 -> 0.0425 (3.75 mm per finger = the 3 N limit; below 42.5 both fingers saturate and lose centring). 5-2 (8
  episodes, seed 1000): D0 1.00, D1 0.875 (the combined block now holds; its insert timed out: rotated 4 deg in the
  grasp). The Task 5 skill envs read the same value (insert / push_in trainings start after this change).
- **Push back** v1 (300 it.; success 0.00 until it. ~60, 0.15 at 119, 0.81 at 300) + continuation v1c (250 it., resumed
  from v1's final checkpoint; `results/skills/push_back/2026-10-20_0110_92e764d0fc_v1c`): training success 0.85, broken
  0.15 (mix D0 / D1 / D2). Skill eval 500 episodes: D0 0.996 (0.986-0.999), D1 0.946 (0.923-0.963; high_friction 0.85,
  combined 0.87, shifted 0.87, irregular 0.95, uneven 0.94, others 0.98-1.0); ~17 steps per push.
- **Bundle `results/bundles/teacher_t45_v2`** = v1 + push back v1c (both pushes learned; insert / push_in planned;
  Task 5 graphs with the planned grasp, gentle lift, 3 N hold, crossing window).
- Crossing variants (5-3, seed 5000, 16 per tier): pusher +2 -> -3 mm (baseline) D0 15 / D1 11; slower crossing
  (20 mm/s in the window) 8 / 9; steeper descent +3 -> -6 mm 16 / 13 (adopted).
- **Task 4 with both pushes learned (teacher_t45_v2, 20 episodes, seed 7000)**: 4-2 D0 0.80, D1 0.70; 4-3 D0 0.70,
  D1 0.45 (v1, learned left + planned back: 1.00 / 0.65 / 0.90 / 0.65). Trace (4-2 D0): the learned back push slid
  the last block of layer 2 ~106 mm out, where it came to rest on the two blocks pushed before it (debris on the table)
  while still leaning on the tower edge (not departed); the actor, which never saw debris (training parks pushed
  blocks away and masks non-tower blocks from its observation), lifted over it, wandered ~30 cm and hit layer 1.
  Executor fix: a learned push whose block stalls part-way (>= 9 cm out, static 3 rows, level < 15 deg, not
  departed) is finished by the planned straight push (back over the block, then down behind its end face). A first
  trigger (>= 6 cm, 2 rows, any tilt) also fired on top-layer blocks balanced at the edge or tipping over it; the
  re-approach behind a tipping block descended into the tower (4-3 D0 0.05): narrowed.
- Training fix (queued): `reset.debris` in the push env puts earlier pushed blocks on the table where they land
  (155-200 mm out along their push, flat or on a side, never two in one spot; `tower_rl.push.debris_poses`,
  `tests/test_push_debris.py`); fine-tunes push left v4d (150 it.) / back v2d (200 it.) from v3 / v1c with
  `{"reset": {"debris": {"frac": 0.7}}}`.
- Scoring note: most remaining Task 4 D1 failures with v2 are our tracker's "structure broken" (a remaining tower block
  > 10 mm from its *episode-start* pose: cumulative drift over several pushes), while every push succeeded. The
  generator's outcome records only a collapse rule (30 deg tilt / 15 mm drop for 0.25 s; `outcome.collapse`), so the
  official rule may be looser. The runner now also reports "TSR collapse-only" (a shadow tracker without the 10 mm
  rule) for Tasks 4 / 5; the headline TSR keeps the stricter rule.
- **Task 4, teacher_t45_v2 + narrowed hand-over (20 episodes, seed 7000)**: 4-2 D0 1.00 (ODR 0), D1 0.85 (ODR 0.04;
  collapse-only 0.90); 4-3 D0 0.95 (ODR 0), D1 0.70 (ODR 0.10; collapse-only 0.75). Remaining D1 failures: high
  friction (layer-2 back pushes drag the tower, layer-1 left pushes shoved sideways) and heavy / combined.
- **push_in left v1** (`results/skills/push_in_left/2026-10-20_0149_286c3e5e57_v1`, 250 it., GPU, demo-speed action
  limit): training success 0.91. First CPU rehearsal with it (bundle `teacher_t45_v3` = v2 + push_in left): 0 / 64,
  the actor dived away from the block and lunged into the tower. Cause: an observation mismatch, not physics. PushInEnv
  never marks the gripper "closed" (its command is just the pusher width), the executor reset it closed at the pusher
  width, so the actor saw +1 in a feature that was always -1 in training. Fixed in the executor's learned push-in
  step. (A GPU "smoke" run of the debris env became a full training: `--smoke` needs `--num_envs` / `--max_iterations`
  given explicitly; killed at it. 55 after confirming the env runs.)
- With the fix, push_in left v1 on CPU (seed 5000, 16 per tier): 5-2 D0 12 / 16, D1 10 / 16; 5-3 D0 10 / 16, D1
  11 / 16. Planned push-in on the same seeds: 5-2 16 / 14, 5-3 16 / 13. The learned push-in's failures are final
  position errors of 5-16 mm (GPU-trained contact dynamics vs the CPU evaluator: the light block slides further).
  The planned push-in stays the default (bundles keep the learned actor available for comparison).
- **Final Task 5 evaluation (planned skills, bundle teacher_t45_v2, seed 8000, 24 episodes per cell)**: 5-2 D0 24 / 24
  (Wilson 0.86-1.00), D1 18 / 24 (0.55-0.88); 5-3 D0 24 / 24, D1 14 / 24 (0.39-0.76; seeds 5000 / 6000 gave 13 / 16
  each). Collapse-only TSR identical. D1 failures: the 5-3 layer-2 gap crossing in friction cases (slippery x3,
  high_friction x2, combined x2: structure 11-16 mm); inserts wrecking the tower after a tilted lift (below);
  high-friction / heavy layer-1 inserts at the 10 mm limit.
- **Open: left arm on a rotated pad-B stack** (shifted_assembly / irregular / combined; trace 5-2 D1 8000001): the
  canonical grasp of a block yawed +5.6 deg needs joint 7 beyond its limit (1.54 rad); the grasp itself is centred,
  but during the lift IK cannot hold the orientation (position error 50 mm, rotation 13 deg) and the block arrives at
  the slot tilted ~22 deg; the insert then hits the tower (200+ mm). A jaw flip (fingers are interchangeable) would
  move j7 by pi at the slot as well, so it must be chosen over the whole transfer; or the lift / held transit should
  free the yaw about the approach axis while keeping the block level. ~4-5 of 48 D1 episodes.
- **push left v4d** (debris fine-tune of v3, 150 it., `results/skills/push_left/2026-10-20_0235_82ff9dde78_v4d`):
  training success 0.91, broken 0.06 with debris (v3 without: 0.91 / 0.06). Bundle `teacher_t45_v4a` (v4d + back
  v1c + hand-over), 20 episodes, seed 7000: 4-2 D0 1.00, D1 0.70 (collapse-only 0.85); 4-3 D0 0.95, D1 0.60 (0.70).
  No gain over v2 (1.00 / 0.85 / 0.95 / 0.70) within the noise of 20 episodes: the hand-over already finishes the
  stalled last blocks.
- **Final Task 4 evaluation (bundle teacher_t45_v2: push left v3, back v1c, hand-over; seed 9000, 40 episodes per
  cell)**: 4-2 D0 39 / 40 = 0.97 (Wilson 0.87-1.00; collapse-only 1.00), D1 35 / 40 = 0.88 (0.74-0.95); 4-3 D0
  36 / 40 = 0.90 (0.77-0.96; collapse-only 0.95), D1 29 / 40 = 0.72 (0.57-0.84). ODR 0.002-0.062. D1 failures:
  combined (9), high_friction (4), uneven / irregular / shifted (1 each).
- Regression check of Tasks 1-3 after the session's executor / runner changes: teacher_v6, 8 episodes per cell, Task
  1-2 and 2-3 D0 / D1 all 8 / 8.
- Repo note: `.gitignore` line `data/` also matches the package `tower_rl/data/` (layouts.py, episode.py, remote.py,
  ...): those sources are not in git (commit 22132a4 lacks them). A root-anchored `/data/` (and `/results/`,
  `/assets_m/`) would fix it; left for the user.
- Debris fine-tunes, skill eval in the debris env (500 episodes): left v4d D0 0.988 / D1 0.958, back v2d (200 it. from
  v1c, `results/skills/push_back/2026-10-20_0302_b5e44dc924_v2d`) D0 0.994 / D1 0.942. In the full task (bundle
  `teacher_t45_v4b` = left v3 + back v2d, same seeds / 40 per cell): 4-2 D0 0.88, D1 0.78; 4-3 D0 0.93, D1 0.68: 130 /
  160 vs v2's 139 / 160. **teacher_t45_v2 stays the Task 4 / 5 bundle**; the debris fine-tunes are kept as runs only.
- **Rotated pad-B stack, corrected diagnosis**: every failing shifted / irregular 5-2 D1 episode has `grasp_rot_deg`
  40-41 right after the grasp and a pre-grasp `goal_adjust_deg` ~4.5: at the left arm's workspace edge
  `feasible_orientation` lets IK pick the tilt about the jaw axis, and it picks ~40 deg; the block is held centred,
  but the pitched gripper is carried to the slot and its body sweeps into the tower as the insert starts (143-264 mm).
  A free yaw during the lift (tried: IK free axis = vertical) does not change it (24 episodes: 5-2 D1 17 vs 18, 5-3 D1
  15 vs 14) and was reverted. Fix to try next: bound the jaw-axis tilt the pre-grasp may choose (or prefer the side
  that leans away from the slot), and fail the transfer early rather than insert with a pitched gripper.
- **push_in back v1** stopped at it. 111 / 250 (success 0.00). Its scripted oracle on GPU scores 0 / 128 too (118
  timeouts; 10 resets where the part-inserted block tips out of the slot, up to 36 deg): the back arm's only push-in
  is 5-3's layer-2 gap crossing, the coplanar edge case, which the GPU env does not get past. The env (reset depth,
  crossing) needs work before training; the planned push-in with the crossing window is the Task 5 default anyway.
  Insert back (same slot) not trained; insert left v1 queued (300 it.).
- **Final bundle `results/bundles/teacher_t45_final`** (2026-10-20 03:52): the skills of teacher_t45_v2 (identical
  weights) with the final graphs and configs (push hand-over parameters; v2's push.yaml predates them, the final
  evaluations above already ran with them). Export check with `--bundle_configs` (configs read from the bundle), D0, 4
  episodes each: Task 1-2, 4-3, 5-3 all 4 / 4.
- **insert left v1** stopped at it. 79 / 300: success 0.00, every episode a failure after ~4 steps (the block released
  outside the slot): a randomly initialised actor opens at once (release when the logit < -0.3) and PPO settled on
  failing fast (mean return -7). v2: the release is ignored until the perceived block is >= `release_min_along_m`
  (-0.105, the success depth) in the slot, in the env (`_grip_mask`) and in the executor's learned insert step alike.
- **insert left v2** (release masked; `results/skills/insert_left/2026-10-20_0407_316564d2c9_v2`, 300 it.): GPU training
  success 0.27 (its scripted oracle reaches ~0.5 there). CPU rehearsal (bundle `teacher_t45_x_insert` = final +
  insert left v2; 5-2, seed 5000, 16 per tier): 0 / 32, every insert timed out. Trace: from the pre-insert pose the actor
  drives the held block down to the table and sideways, away from the slot, from its first rows. Unresolved: the
  observation and action pipelines were compared term by term (gripper flag, width command, phase, progress, action
  scale, EMA, lead limit) and agree, and the block's axis convention is the same in env and executor (x along the
  insertion direction, checked); remaining suspects are the start-state distribution (how the block hangs in the 3 N
  grasp on GPU vs CPU) and the GPU contact artifact. Next: replay the executor's first insert observation inside the
  GPU env and diff the vectors / actions.
- End of session 3 (05:00): GPU idle. Recommended for Tasks 4 / 5: `teacher_t45_final`, where Task 5 uses the planned
  grasp / insert / push-in.

## 2026-10-20/21 (session 4): pose estimator v1

Goal (user, 2026-10-20 evening): build the multi-view RGB pose estimator on the critical path (Teacher_build_session.MD
section 11), "the pose estimator that gives the best estimates". Code: `tower_rl/perception/` (Isaac-free except where
noted), scripts `render_perception_data.py`, `fetch_dataset_frames.py`, `train_perception.py`, `eval_perception.py`,
`check_camera_model.py`, `rehearse.py --perception`.

### Off-the-shelf models (decision)

No off-the-shelf model fits: FoundationPose (the strongest zero-shot 6D pose model) needs depth, which the challenge does
not provide; RGB-only novel-object models (MegaPose, GigaPose, FoundPose, Co-op) are single-view, centimetre-level on
symmetric textureless objects, slow, and know nothing of identical stacked instances. We have what they lack: exact
calibration, the exact block model and an unlimited, perfectly labelled renderer of the evaluator's scene. So: an
off-the-shelf ImageNet backbone (torchvision ResNet-34) in a DOPE/PVNet-style keypoint network (block corners + block
edges + visible-block mask, full resolution) trained on our renders, and a model-based multi-view fit (Gauss-Newton on
corner and edge correspondences in all four cameras) with tracking priors from the executor's world model.

### Image formation (measured; supersedes "a ~1 deg recalibration" of session 1)

- The 12-17 px offset between our renders and the dataset JPEGs is exactly (cx - 320, cy - 240) on all four cameras
  (sub-pixel registration: 11.9 / 3.0, 12.1 / 17.0, 4.5 / 6.6, 4.5 / 6.3 px; correlation 0.74-0.95): **Isaac Lab cameras
  ignore the principal point** (aperture offsets unsupported: the optical axis goes through the image centre; fx / fy
  averaged), the generator applies it (`opencv_pinhole_applied_explicitly`). The camera poses of the published station
  equal the dataset's (Sep-9 Tasks 1-3 and Sep-14 Tasks 4/5 data alike): there was no recalibration.
- Both renderers count the principal point from the pixel corner: in pixel-centre coordinates the evaluator's is
  (cx - 0.5, cy - 0.5) and Isaac's (319.5, 239.5). With that, projected block edges of the recorded poses sit on the
  dataset JPEGs' edges within 0.25 px on every camera (`results/perception/calib/model_check_t42_ep70/`; one 0.5 px case),
  and on our renders within 0.25 px (FK wrist poses). Static images are delivered turned by 180 deg (docs: observation
  arrays match the dataset JPEG orientation; wrist views as is).
- Consequence: the organizers' 531 episodes are perfectly labelled evaluator-renderer images (recorded block poses + FK).
  `data/perception/dataset_val/`: 3 episodes x 24 frames per variant (720 frames) for validation.
- The rendered block is the station hull: 150 x 49.76 x 28.3 mm (+ two keel vertices within 1 mm of two corners), not
  150 x 50 x 30; labels use these extents (Task 5 side blocks scaled by their dims).
- Foundations are not fixed in the dataset: 2 mm spread across episodes, up to 3.8 mm from our layouts; the estimator
  measures them (5 mm prior toward nominal on the first call only).
- Lighting: one RectLight (900 x 600 mm, 3000, 5600 K); RTX ambient light (intensity 1.0) gives most of the base
  illumination, so x0.25-x2 of the RectLight moves mean brightness 108-145 (x0 still 105); colour temperature shows
  clearly. Renders after a light change need more than one 4-subframe capture to converge.

### Training data

`scripts/render_perception_data.py`: the teacher (`teacher_t45_final`, privileged, CPU physics) runs episodes with the
four cameras on (RGB + depth, the generator's RTX settings, 4 subframes); random rows are captured (p 0.1 per 0.1 s
row). Labels: block poses, camera models, per-corner visibility (Isaac depth), per-pixel block ids (analytic boxes
where Isaac's depth agrees; 254 = a block hidden by an arm or finger). Light randomized per env and batch: 40 % nominal,
else intensity x0.25-2.0 (log-uniform), 3000-8500 K, azimuth +-60 deg about the table centre, tilt <= 30 deg in half.
Run r1: all ten variants, D0 + D1, 16 episodes per tier (seed 20000) + validation 4 per tier (seed 90000); ~240 KB per
frame (4 JPEG q95 + 4 PNG); Task 2-3: 2766 frames in 13 min.

### Fit (oracle heatmaps, before any network)

With heatmaps splatted from the frames' own labels (`eval_perception.py --oracle`), Task 2-3 renders:
- from a perturbed truth (4 mm / 3 deg yaw / 1 deg tilt): first version median 0.10 mm, p99 10.8 mm, 3 % of blocks > 5 mm
  (wrong edge associations snowballing; 15 mm priors out of the windows); after coarse-to-fine association (6 -> 3 px),
  Huber -> Tukey weights, visibility re-predicted during the fit and a re-search + refit of every block whose inlier
  ratio (inliers / predicted visible points) is < 0.6: median 0.06 mm, p90 0.36 mm, p99 2.7 mm, 0.2 % > 5 mm (no
  movable block); held blocks 0.01 mm;
- first call from the nominal layout + coarse search: 0 of 360 blocks > 2 mm, p99 0.55 mm.
The ratio separates good fits: all fits < 0.5 were wrong, half of 0.5-0.6, 1.5 % above 0.8.
- Task 1-3 (two parallel blocks per layer), same tests: first call 0 of 480 blocks > 2 mm (p99 0.45 mm). Perturbed
  truth 4 mm / 3 deg: 2.3 % > 5 mm: blocks slid ~10 mm along their length (long edges cannot see that) or truly
  ambiguous (one 28 mm-off pose outscored the truth). A minimum-over-point-groups inlier ratio (corners, x / y / z
  edges) was tried against the slide and reverted: worse on both variants (4.4 % / 1.7 % > 5 mm), because points hidden
  by an arm (not in the visibility prediction) drag single groups down even at the true pose.
- Realistic tracking priors (1.5 mm / 1 deg / 0.5 deg): no block > 5 mm; p99 0.97 mm (Task 1-3) / 0.94 mm (Task 2-3),
  movable blocks p90 0.34 / 0.15 mm. The 4 mm prior test stays as a re-acquisition stress test.
- Table search for lost movable blocks (Task 4 debris, knocked blocks): flat or on its side, any yaw, +-12 cm, away
  from other blocks; oracle test: a top block 8.5 cm off its tower, turned 34 deg, found within 2 mm.

### Estimator details added before the first network (2026-10-20 23:00-23:40)

- A fourth network head, "hidden" (a block surface covered by an arm or finger: ids 254), keeps such points out of
  the predicted visibility (arms are not modelled geometrically). Using the visible-block mask for this instead was
  tried and dropped (stress test 5.2 % / 2.8 % > 5 mm): it also says "no block" on background, which excuses the points
  of a misplaced pose.
- Resting blocks: a fit within 1.5 mm / 0.75 deg of the previous estimate moves it by 0.3 of the difference (noise);
  beyond, it is taken whole (moved). Held blocks: reported as the gripper's FK pose times a filtered offset (gain 0.15,
  or 0.5 after a change > 1 mm / 0.5 deg: a slip); the hold ends when the fingers open past 65 mm or close past 46 mm
  (dropped: the block is 49.7 mm wide).
- Closed loop without rendering (`rehearse.py --perception oracle`: heatmaps splatted from the true visible corners
  and edges, boxes only, estimator on CPU) checks the integration with the executor: Task 2-3 D0, 2 episodes: TSR 1.00,
  PSR 1.00, ODR 0; estimation errors along the trajectories median 0.07 mm, p99 0.45 mm; ~180 calls per episode.

### Night of 2026-10-21 (00:50-): r1 data done, oracle closed loop on D1, tooling

- r1 training renders finished 00:54: 28,227 frames (4 views each) over all ten variants, D0 + D1, 16 episodes per tier
  (Task 1-3 4252, 4-3 3790, 5-3 3966, 4-2 3202, 1-2 3299, 2-3 2766, 5-2 2520, 3-3 1756, 2-2 1456, 3-2 1220); 6.5 GB.
  Validation renders (seed 90000, 4 per tier) follow, then labelling of the organizer frames, then kp_v1 training
  (`results/perception/gpu_queue_train_v1.sh`), its evaluation (`gpu_queue_eval_v1.sh`) and the closed loop on
  rendered cameras for every variant (`gpu_queue_closed_v1.sh`, 4 episodes per tier, CPU physics).
- `train_perception.py` smoke-tested on CPU (6 iterations on the full r1 + organizer data): runs end to end.
- `label_dataset_frames.py` (first run is in the queue) checks that the written state reaches the renderer: the share
  of box pixels whose rendered depth lies beyond the box; it aborts (exit 2, no labels) if the first three frames are
  all > 20 %, and reports the mean / max per variant.
- Closed loop: `CameraPerception.save` writes every call's estimates, truth, widths, held flags, inlier ratios
  (`perception_log.pkl` next to perception.json); the D1 case "lighting" now reaches the rendered station light
  (`tower_rl/sim/lights.py`, `run_batch` calls `observe.set_domain(world, dom)`; before, rehearsals rendered nominal
  light in every case).
- Estimator: a lost movable block not found by the table search waits 2, 4, 8, 16 calls before the next search (Task
  4 debris was searched every call: the CPU oracle run of 4-3 D1 took > 80 min for one batch).
- Oracle closed loop, D1, 2 episodes each (`results/perception/closed_loop/oracle_d1/`): Task 3-3 TSR 1.00, errors p99
  0.60 mm (held p99 0.72 mm). Task 1-3 TSR 0.50: seed 7000001 (uneven_weights) failed its first place (yaw 6.3 deg),
  re-grasped 27 mm off-centre, failed again (9.2 mm) and the tower then fell (the same seeds pass on truth); confident
  estimates > 5 mm in 12.5 % of the calls after the fall. Trace rerun with the per-call log in `oracle_d1_trace/`.
- Task 1-3 D1 trace (`oracle_d1_trace/`, per-call log): perception matched the truth to < 0.3 mm / 0.4 deg through
  both failed places of seed 7000001, and the same seeds on truth state (`truth_d1_same/`) fail identically (TSR
  0.50, t0 place 6.3 deg off): a teacher failure under uneven weights, not perception. Two estimator faults showed:
  (1) at 48.8 s the back arm's fingers knocked block l3_a (tilted 9 deg, 10 mm); the prior fit and the planar search
  failed, the table search ran and its refit slid onto the destination foundation block 2 (a duplicate pose with the
  other block's image features, inlier ratio 0.69, confidence 1), where it stayed while the block came to rest on the
  foundation layer turned 55 deg; (2) a grasp was detected while the fingers were still closing (51.4 mm on the
  49.8 mm block) and the block shifted ~8 mm in the pinch frame within a call; the filtered offset took ~10 calls.
  Fixes: `_overlap` (a pose whose centre line or corners, pulled in 3 mm, lie inside another block shrunk by 3 mm, or
  that contains another block's centre) rejects table-search results, re-search results and any fit that moved > 5 mm
  into another block (the block is then lost); the lost-block search also tries lying flat on the top face of every
  nearby block (layers), and its hypotheses must not interpenetrate (replaces the 45 mm centre-distance filter); the
  first 3 held-block fits after a grasp are taken whole (`settle_calls`). Test: `test_knocked_block_found_on_a_layer_
  not_on_another_block`. The old table test's block lay partly inside the foundation (the 45 mm filter let that pass).
- Re-acquisition stress test on all ten variants (oracle maps incl. the arm-hidden map from the ids; priors 4 mm /
  3 deg yaw / 1 deg tilt; 400 val frames, 4873 blocks; `results/perception/eval/oracle/stress_r1val_*.json`): baseline
  foundations 1.80 % / movables 1.59 % > 5 mm (p99 7.4 / 6.2 mm). Most foundation failures sank or tilted (z errors up
  to 11 mm under the towers that hide them); most movable ones slid along their length (corners lost, long edges kept).
  (1) Foundations lie flat on the table: their fit pins z (nominal), roll and pitch (prior weight x1e4; `_flat`): 0.29 %.
  (2) A fit that matches < 75 % of its >= 3 predicted visible corners is searched again (planar), the re-search
  accepted on inlier ratio + 0.5 corner ratio (0.5 caught half of the slides: 1.30 %; 0.75: 0.65 %). (3) The rule "a
  fit that moved into another block is rejected" now spares a fit whose inlier ratio beats the other block's by
  >= 0.15 (in the stress test the neighbour's perturbed prior was the wrong one): foundations 0.17 % / movables
  0.57 % > 5 mm (p99 1.2 / 3.4 mm). Realistic tracking priors (1.5 mm / 1 deg / 0.5 deg): 0 of 4860 > 5 mm, p99
  1.06 mm; first call: 0 of 880 > 5 mm, p99 0.66 mm. CPU call 0.19 s tracking / 0.33 s stress / 0.47 s first call.
- **Organizer data quirk: recorded foundation poses are not where the renderer drew the foundations.** First read of
  kp_v1 (checkpoint 7500, CPU, 120 organizer val frames, tracking priors): movable blocks median 0.22 / p90 0.79 /
  p99 1.55 mm, held p90 0.41 mm, but foundations 1-4 mm off, the three blocks of a pad off together, z exact. A
  network-free check (edge registration of projected recorded poses on the static JPEGs) agrees: movable blocks
  register within 0.1 px (never > 1 px), recorded foundations are off by > 1 px in 41 % of the views (up to 2.7 px).
  Per pad and episode the drawn foundations sit at a rigid offset from the recorded ones (median 1.0-1.2 mm, p90
  3.9-4.8 mm, max 6.9 mm; no yaw), neither at the recorded nor at the nominal layout position. The episodes' goals
  (`target_mapping`) are built on the same jittered tower centres as the recorded foundations (`tower_jitter_m`
  0.003). The docs fix foundations and destination anchors in the evaluation, so drawn and recorded should agree
  there; our executor takes goals from the perceived (drawn) foundations either way. Consequences here: the organizer
  frames' foundation labels (training) and truth (evaluation) were wrong by up to ~7 mm.
  `scripts/correct_dataset_foundations.py` registers each pad per episode (coarse-to-fine translation search on the
  image gradient along the pad's visible edges, both static cameras, 4 frames; idempotent; keeps `pos_recorded`) and
  rewrites the records; `eval_perception_sequence.py --foundation_corrections` applies them to the HDF5 truth.
  kp_v1 trained on the uncorrected labels; the kp_v1 evaluation queue (`gpu_queue_eval_v1b.sh`) uses the corrected
  truth. Also fixed: `eval_perception.py` assumed records list blocks in the estimator's order (Task 5 records list the
  destination blocks first: the dataset evaluations would have stopped at the first Task 5 frame).
- More organizer training frames: subset episodes 8-9 of every variant (40 frames each) into `dataset_train`.
- Estimator: lost blocks back off up to 64 calls between searches; after 3 failed searches a block is a "ghost"
  (confidence 0; it hides nothing and blocks no other pose until found again; Task 4 debris).

### kp_v1: trained network results (2026-10-21 03:30-)

- Training (`results/perception/models/kp_v1/`): 25k iterations x 12 crops (512 x 384), ~100 min on the A5000 (4.3
  it/s); r1 renders + organizer frames x3. Held-out renders at 25k: corners recall 98.4 % / precision 97.4 % within
  3 px, matched error median 0.20 / p90 0.67 px; edges recall 99.4 % / precision 98.9 %; block mask IoU 0.995;
  arm-hidden IoU 0.93 (plateau from 20k). The process hung after TRAIN_DONE (persistent loader workers): killed;
  `train_perception.py` now ends with `os._exit(0)`.
- Single frames (`results/perception/eval/kp_v1/`; organizer truth with corrected foundations):

  | set | mode | all blocks p50 / p90 / p99 mm | > 5 mm | movable p90 / p99 mm | held p90 mm | call s |
  |---|---|---|---|---|---|---|
  | renders | tracking prior | 0.13 / 0.63 / 1.29 | 0.06 % | 0.72 / 1.62 | 0.42 | 0.27 |
  | renders | first call | 0.14 / 0.39 / 0.84 | 0 | 0.51 / 0.96 | - | 0.35 |
  | renders | stress 4 mm / 3 deg | 0.12 / 0.42 / 2.09 | 0.39 % | 0.58 / 3.08 | - | 0.36 |
  | organizer | tracking prior | 0.31 / 1.15 / 4.05 | 0.37 % | 0.69 / 1.61 | 0.25 | 0.10 |
  | organizer | first call | 0.29 / 0.78 / 4.07 | 0 | 0.39 / 0.59 | - | 0.13 |
  | organizer | stress | 0.29 / 1.32 / 4.39 | 0.60 % | 0.67 / 2.99 | 0.14 | 0.28 |

  The network adds little to the fit's own error (oracle maps: tracking p99 1.06 mm). On the organizer renderer
  movable and held blocks are as accurate as on ours; the remaining large errors are foundations (destination pads
  under towers: Task 3-2, 5-2, 5-3).
- Whole organizer episodes at 10 Hz (10 held-out episodes, `sequences.json`): calls 0.09-0.17 s mean (GPU), held
  blocks p90 0.10-0.24 mm, placed blocks 0.02-0.28 mm from the truth one second after release; but relative to the
  destination foundation 0.1-0.8 mm (Tasks 1-3, 2-x) and ~2 mm (Task 3-x), 3.2-3.6 mm (Task 5-3, where the slot comes
  from the side blocks instead): the destination foundation estimate drifts once blocks stand on it. Static p99
  3-6 mm in Task 3 (foundations).
- Foundations are now frozen after 10 ok fits (mean x, y, yaw; `freeze_after`): no longer fitted, still occluders.
  Oracle closed loop Task 2-3 D0 with it: TSR 1.00, errors p99 0.27 mm (was 0.45).
- Closed loop with the network on rendered cameras (`gpu_queue_closed_v1.sh`, `results/perception/closed_loop/kp_v1/`,
  4 episodes per tier, CPU physics, renders + network + estimator on the GPU): Task 2-3 D0 TSR 1.00 (4/4), PSR 1.00,
  ODR 0, errors along the trajectories median 0.07 / p90 0.30 / p99 1.5 mm; D1 TSR 1.00 (4/4), p99 0.98 mm.
- Closed loop, network, continued: Task 1-3 D0 / D1 TSR 1.00 / 1.00 (4/4 each), errors p99 0.73 / 1.16 mm, held p90
  0.17 mm; Task 3-3 D0 / D1 1.00 / 1.00 (4/4 each). Target block of each skill p90 0.16-0.32 mm.
- Foundation drift confirmed (per-call trace through organizer episodes, `eval_perception_sequence.py --trace`): Task
  3-3's destination pad centre is off 0.4-0.6 mm while the pad is empty, ~2 mm from the first block on it (its middle
  block 5.2 mm median); Task 1-2 0.1-0.4 mm empty, 1-2 mm later. Hence the freeze after 10 ok fits.
- Task 4 (oracle closed loop, D1 4-3, `oracle_d1_fix1` / `fix2`): the pushed block moved 31 mm in one 0.1 s call
  (the push actor strikes first), then 125 mm within 0.4 s; the estimate stayed on the tower (fit along its long edges,
  ratio 0.64; then lost and searched only within 12 cm, the debris lands 160-175 mm out), so the push node never saw
  the departure and pushed on into the tower ("structure" failures from the first push; truth reaches p2-p7).
  Fixes: the planar re-search adds hypotheses along the block's own long axis up to +-100 mm (and drops hypotheses
  inside other blocks); a block that moved > 3 mm last call is predicted to move as much again (<= 40 mm; no static
  low-pass while moving); the lost-block search spans +-16 cm on 5 x 5 max-pooled heatmaps (8 mm steps) and tries
  leaning poses (low end on the table, tilt 25-82 deg about the short axis, yaw over the full turn): a pushed-off
  block that came to rest leaning against the tower (60 deg) was otherwise never found (its stale pose kept the push
  going: timeouts). Pushes are now tracked through the slide (errors 0.0-0.5 mm at 9-43 mm per call, `fix2`); the
  lean search is new in `fix3`. Tests: `test_pushed_block_followed_along_its_length`,
  `test_block_leaning_against_the_tower_found` (fails without the lean search).
- Freeze verified on the organizer Task 3-3 episode (CPU replay, kp_v1): static p99 5.58 -> 0.84 mm, > 5 mm 8.5 % ->
  0.01 %; placed blocks relative to the destination foundation 2.21 / 1.90 / 1.79 -> 0.63 / 0.54 / 0.41 mm.
- Closed loop, network: Task 5-3 D0 TSR 0.75 (3/4, all G and P ok), D1 0.75 (3/4; truth on 2 of these seeds 0.50).
  Insert target p90 0.085 mm, held p90 0.12 mm; but every i1 push-in ended "failed" (twice "structure 11.6 / 12.5 mm"):
  while the insert slid in next to it (1 mm clearance), the side block's estimate crept 1.5 -> 12.8 mm (it never
  moved: its fit took the insert's parallel edges, each call's step inside the static gate), and the executor's
  structure check saw it move. Fix: a free block unmoved for 10 calls (< 0.5 mm / 0.3 deg per call) is anchored, its
  fit held toward the anchor with sigma 0.5 mm (`rest_calls`, `rest_prior_m`); a real move fails the fit or the corner
  test and the re-search (weak prior) moves it, which ends the anchor.
- Task 4 oracle `fix3` (lean search): pushed blocks found leaning (0.0-1.0 mm), but an identity swap appeared: a
  still top-layer block whose fit failed for a call (pusher in front) was searched wide and taken onto the leaning
  pushed block's features (ratio 1.00: identical blocks), and the pushed block then onto its vacated place. Fix: a
  block lost while still is taken as occluded (pose kept) for 10 calls before the wide search; blocks lost while
  moving are searched at once (`occluded_calls`). Test: `test_still_block_out_of_view_keeps_its_pose`; the table,
  layer and lean tests now give their block its last motion (falling, knocked, pushed).
- Anchoring verified (oracle closed loop, D0, 2 episodes each, `oracle_anchor/`): Task 5-3 TSR 1.00 with every
  push-in ok (push-in target p90 1.3 mm, was 20.4 mm with the network before the anchor), insert target p90 0.04 mm;
  Task 2-3 TSR 1.00, p99 0.32 mm.
- Closed loop, network, Task 4-3 D0 (code with the lean search, occlusion delay): TSR 0.00 (0/4), pushes 50 %;
  the same seeds on truth: TSR 1.00 (4/4, all 9 pushes). Two traces: (a) seed 7100001: a layer-1 block deep in the
  tower (never touched) failed its fits while hidden for > 10 calls, was searched wide and teleported, and a pushed
  block took its place (identity swap); (b) seed 7000000: every estimate within 0.5 mm at the start of push p3 (debris
  on the table found correctly), the pushed block tracked to 0.1 mm over 149 mm, yet the push dragged both layer-2
  neighbours 9-10 mm ("structure"), which truth does not. Open; an oracle run on the same seeds separates estimator
  logic from network errors (`oracle_d0_fix5`).
- Fixes: (1) a still block is searched wide only if it was touched, i.e. a gripper's pinch came within 6 cm of it or
  a moving block's centre within 10 cm, during the last 20 calls (`touch_m`, `touch_block_m`, `touch_calls`; replaces
  the 10-call occlusion delay): blocks do not teleport. (2) 57 % of the reported quaternions were half-turned twins of
  the simulator's (the same box; searches and the foundation freeze pick any of the four), while the learned skills
  observe raw quaternions trained on the simulator's continuous ones: each call now reports the representative
  nearest its previous output.
- Quaternion twins traced in the Task 4-3 D0 network run: foundations turned at their freeze (the doubled-angle mean
  of a -90 deg yaw came back +90 deg) and stayed so; a pushed block re-found by the search at yaw 177 deg instead of
  -3 deg mid-push, so the push actor's observed yaw change jumped by 180 deg. The continuity step (above) covers
  both; test `test_quaternions_stay_continuous_through_the_foundation_freeze` (180 deg without it).
- Queue after the kp_v1 closed loop (`gpu_queue_v2c.sh`): Tasks 4-3 / 5-3 again with kp_v1 and the later estimator
  fixes (`closed_loop/kp_v1b/`), then relabelling of the corrected organizer frames, kp_v2 (fine-tuned from kp_v1,
  12k iterations) and its evaluation.
- Task 4-3 D0 oracle closed loop with the touch rule and quaternion continuity (`oracle_d0_fix5`, same seeds as the
  truth 4/4): TSR 0.25 (1/4), GSR 0.86 (was 0.50 with the network and the older code), PSR 0.83, ODR 0.11. The
  estimates are good while tracked (median 0.08 mm); failures: two "structure" pushes (12.7 / 13.2 mm, real
  neighbour drags), a collapse with p8 timeouts, and one pushed block that flew off the table (z -18 m in the
  truth; this is the 5e7 mm p99) after being lost for two calls while tipping over the tower edge: its frozen pose
  told the push actor the block had stopped. The learned push actor reacts to perception artifacts that truth never
  shows. Change: a moving block not found keeps its motion for up to 5 calls (decaying x0.8, `coast_calls`) instead of
  freezing. The push actor's sensitivity to perceived poses remains the open Task 4 issue (candidates: train the push
  skill with perception noise and short dropouts, as Step 8.1 did for grasp/place; or have the executor hold the
  push while its target's confidence is low).
- Executor focus: `PoseEstimator.__call__(..., focus=[block indices])` names the blocks the policy's current node acts
  on; they are never anchored and are searched wide when lost (`CameraPerception` passes the current node's block; a
  policy server passes the executor's). Reason: the anchored target of a push lagged its first 1-3 mm of motion (the
  long edges still matched, corners within the 2 px inlier window), and the push actor, seeing a block that did not
  move, pushed harder. Task 5's side blocks stay anchored while the insert (the focus) moves.
- Truth reference in the loop: `rehearse.py --perception truth` passes the states through unchanged and logs them like
  the estimator runs (Task 4-3 D0, same seeds: 4/4, as plain truth). Aligned call by call with the oracle run,
  the physical trajectories part in the very first push (p0, t 5.8-6.4 s), exactly while the pushed block tips over
  the tower edge and the estimator loses it for 1-3 calls (a block rotating about an edge leaves the fit's basin):
  the actor then acts on a stale or coasted pose, the push and everything after it differ, and later pushes meet
  other debris. Executor change (`PushNode.step`): while the push target's confidence is below CONF_MIN, hold the
  pusher (no actor step) for up to `lost_hold_rows` (15) rows; lost longer after it was last seen >= 60 mm along the
  push, it has left the tower ("lost_departed", retreat). On truth (confidence 1) nothing changes (verified 4/4).

- kp_v1 closed loop, all variants (`results/perception/closed_loop/kp_v1/`, network on the four rendered cameras, 4 episodes
  per tier, seeds 7000; the estimator changed during the queue: Tasks 2-3 / 1-3 / 3-3 / 5-3 / 4-3 ran before the freeze /
  anchor / touch / continuity / focus / push-hold changes, the 2-layer variants with most of them):

  | variant / tier | TSR | PSR | ODR | all blocks mm (median / p90 / p99) | held p90 mm | target block p90 mm by skill | calls |
  |---|---|---|---|---|---|---|---|
  | task1-2_tower_transfer_2 D0_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.16 / 0.34 / 0.54 | 0.19 | grasp_lift 0.29, place_release 0.21, transit 0.29 | 942 |
  | task1-2_tower_transfer_2 D1_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.17 / 0.33 / 0.66 | 0.19 | grasp_lift 0.32, place_release 0.21, transit 0.30 | 940 |
  | task1-3_tower_transfer_3 D0_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.19 / 0.37 / 0.73 | 0.17 | grasp_lift 0.18, place_release 0.17, transit 0.16 | 1446 |
  | task1-3_tower_transfer_3 D1_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.20 / 0.44 / 1.16 | 0.17 | grasp_lift 0.20, place_release 0.23, transit 0.19 | 1455 |
  | task2-2_cross_tower_transfer_2 D0_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.08 / 0.36 / 0.70 | 0.15 | grasp_lift 0.28, place_release 0.40, transit 0.24 | 484 |
  | task2-2_cross_tower_transfer_2 D1_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.09 / 0.35 / 0.41 | 0.18 | grasp_lift 0.22, place_release 0.44, transit 0.21 | 484 |
  | task2-3_cross_tower_transfer_3 D0_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.07 / 0.30 / 1.50 | 0.16 | grasp_lift 0.17, place_release 0.32, transit 0.17 | 712 |
  | task2-3_cross_tower_transfer_3 D1_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.09 / 0.32 / 0.97 | 0.17 | grasp_lift 0.18, place_release 0.26, transit 0.20 | 767 |
  | task3-2_vertical_stacking_2 D0_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.07 / 0.21 / 0.97 | 0.16 | grasp_lift 0.39, place_release 0.25, transit 0.35 | 465 |
  | task3-2_vertical_stacking_2 D1_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.08 / 0.26 / 0.50 | 0.17 | grasp_lift 0.32, place_release 0.26, transit 0.31 | 463 |
  | task3-3_vertical_stacking_3 D0_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.07 / 0.37 / 1.00 | 0.17 | grasp_lift 0.19, place_release 0.26, transit 0.23 | 691 |
  | task3-3_vertical_stacking_3 D1_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.08 / 0.36 / 0.87 | 0.18 | grasp_lift 0.21, place_release 0.28, transit 0.22 | 693 |
  | task4-2_full_tower_pushdown_2 D0_perception | 0.75 (3/4) | 0.75 | 0.102 | 0.12 / 0.36 / 150668809.45 | - | push 0.56, transit 0.48 | 2358 |
  | task4-2_full_tower_pushdown_2 D1_perception | 0.50 (2/4) | 0.92 | 0.040 | 0.09 / 0.36 / 198.69 | - | push 15.97, transit 0.45 | 3990 |
  | task4-3_full_tower_pushdown_3 D0_perception | 0.00 (0/4) | 0.47 | 0.500 | 0.10 / 0.37 / 359.66 | 0.26 | push 221.94, transit 474.90 | 5543 |
  | task4-3_full_tower_pushdown_3 D1_perception | 0.00 (0/4) | 0.64 | 0.434 | 0.13 / 15.67 / 190.74 | 0.44 | push 130.47, transit 19.95 | 6460 |
  | task5-2_tower_insertion_2 D0_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.11 / 0.23 / 6.43 | 0.11 | grasp_lift 0.12, insert 0.09, push_in 31.58, transit 0.19 | 895 |
  | task5-2_tower_insertion_2 D1_perception | 1.00 (4/4) | 1.00 | 0.000 | 0.10 / 0.24 / 14.32 | 0.10 | grasp_lift 0.14, insert 0.08, push_in 16.44, transit 0.13 | 939 |
  | task5-3_tower_insertion_3 D0_perception | 0.75 (3/4) | 1.00 | 0.000 | 0.18 / 0.90 / 2.79 | 0.12 | grasp_lift 0.22, insert 0.09, push_in 20.41, transit 0.26 | 3015 |
  | task5-3_tower_insertion_3 D1_perception | 0.75 (3/4) | 0.75 | 0.529 | 0.17 / 0.40 / 2.36 | 0.13 | grasp_lift 0.19, insert 0.08, push_in 15.13, transit 0.20 | 3016 |

  Tasks 1-3 (2 and 3 layers): 48 / 48 episodes; Task 5-2 8 / 8, 5-3 6 / 8; Task 4-2 5 / 8, 4-3 0 / 8 (truth on the same
  4-3 D0 seeds 4 / 4). The p99 outliers of Task 4 are pushed-off blocks that left the table (truth z far below 0).
- With the push hold (Task 4-3 D0, first batch of 2 episodes): oracle maps TSR 1.00 (2/2), GSR 1.00, ODR 0 (was 1/2,
  GSR 0.67-1.00 without it); network on rendered cameras (`kp_v1b`, all estimator fixes + hold) TSR 0.50, GSR 0.83,
  ODR 0.08 (kp_v1 before the fixes: 0/2, GSR 0.56).
- Push hold, results Task 4-3 D0 (4 episodes): oracle maps TSR 0.75 (3/4), GSR 1.00 (36/36 pushes), ODR 0.01; network
  (`kp_v1b`) TSR 0.50 (2/4), GSR 0.83, ODR 0.08. Both network failures at p5 (the last layer-2 block): trace 7100001:
  tracked to 0.1 mm while pushed, lost 6 calls while tipping (the pusher held still), found on its side on the table
  at 0.1 mm, but 153 mm from the tower axis (< 160 mm: not departed), so the actor resumed and pushed into layer 1
  ("structure"); on truth the actor follows through while the block tips and it lands beyond the line. Change: while
  the target is unseen the pusher continues the actor's last commanded motion, decaying x0.8 per row (instead of
  holding still). Oracle check `oracle_d0_fix9`.
- `kp_v1b` (kp_v1, all estimator fixes, push hold): Task 4-3 D0 TSR 0.50 (2/4), GSR 0.83; D1 0.00 (0/4), GSR
  0.72-0.78 (truth on two of these D1 seeds: 0/2); Task 5-3 D0 1.00 (4/4; was 0.75 with kp_v1 before anchoring and
  focus).
- Follow-through rejected: oracle Task 4-3 D0 (4 episodes, same seeds) with the pusher continuing its last motion while
  the target is unseen: TSR 0.25 (1/4), GSR 30/36 (`oracle_d0_fix9`), against holding still: 0.75 (3/4), GSR 36/36
  (`oracle_d0_fix8`). Reverted to the hold. Task 5-3 with anchoring + focus (`kp_v1b`, network): D0 1.00 (4/4), D1
  0.75 (3/4) (kp_v1 before: 0.75 / 0.75; truth on two of the D1 seeds 0.50).

### kp_v2 (2026-10-21 09:45-)

Fine-tuned from kp_v1 (12k iterations, lr 5e-4) on the relabelled organizer frames (foundation poses corrected per
episode; 70 training episodes = 2800 frames x3) + r1. Held-out renders unchanged (corners recall ~98.4 %). Organizer
validation frames (corrected truth):

| set | kp_v1 all p50 / p90 / p99 mm | kp_v2 | kp_v1 foundations p50 / p90 / p99 | kp_v2 | movables p90 v1 / v2 |
|---|---|---|---|---|---|
| tracking prior | 0.31 / 1.15 / 4.05 (> 5 mm 0.37 %) | 0.20 / 0.70 / 1.72 (0.07 %) | 0.48 / 1.78 / 4.41 | 0.21 / 0.74 / 1.75 | 0.69 / 0.67 |
| first call | 0.29 / 0.78 / 4.07 | 0.18 / 0.41 / 0.91 | | | 0.39 / 0.36 |
| stress 4 mm / 3 deg | 0.29 / 1.32 / 4.39 | 0.19 / 0.63 / 2.89 | | | 0.67 / 0.71 |

The corrected labels removed most of the foundation error on the evaluator's renderer; movable blocks are unchanged.
**kp_v2 is the model to deploy** (`results/perception/models/kp_v2/model_last.pt`). Last queue (`gpu_queue_v3.sh`):
Task 4-2 closed loop with kp_v2 and the final estimator + executor (`closed_loop/kp_v2/`).
- Whole organizer episodes at 10 Hz (10 held-out episodes, kp_v2 + final estimator incl. the foundation freeze,
  `results/perception/eval/kp_v2/sequences.json`; kp_v1 with the estimator of 03:40 in brackets): placed blocks
  relative to the destination foundation 0.09-0.55 mm (0.1-3.6 mm); static p99 0.39-1.02 mm on Tasks 1-4 (0.86-6.3);
  Task 5 still has a tail (5-2: 1.2 % of static estimates > 5 mm, p99 11.4 mm; 5-3: 0.8 %, p99 3.0 mm), open; held
  p90 0.10-0.23 mm; 0.085-0.13 s per call.
- Task 5-2 tail traced (CPU replay, kp_v2): the only block > 5 mm is the insert during the demo's push-in (t 51-59 s,
  under the roof, mostly hidden: up to 17 mm along its length). The sequence replay has no executor focus; in the
  closed loop the push-in succeeds (Task 5-2 8/8, 5-3 7/8). Open, minor.
- Task 4-2 with kp_v2 and the final estimator + executor (`closed_loop/kp_v2/`): D0 TSR 1.00 (4/4, all pushes).
- Task 4-2 with kp_v2 and the final code: D0 1.00 (4/4); D1 0.25 (1/4; pushes 92-96 %, truth ~0.88 on 40 episodes).
  The D1 failures follow "lost_departed" ends (target unseen 15 rows after >= 60 mm along: sometimes still on the
  tower edge) and then "structure" failures; one is the lighting case.

### Session 4 end state (2026-10-21 11:30)

- Deploy `results/perception/models/kp_v2/model_last.pt` with `PoseEstimator(..., source="evaluator")`; per episode
  `reset(variant)`, per call `est(images, q, widths, focus=[executor node's target block index])` (README).
- Closed loop with the network on rendered cameras (4 episodes per tier): Tasks 1-3 (2 and 3 layers) 48 / 48 (kp_v1);
  Task 5-2 8 / 8 (kp_v1), 5-3 7 / 8 (kp_v1 + fixes); Task 4-2 D0 4 / 4, D1 1 / 4 (kp_v2 + fixes); Task 4-3 D0 2 / 4,
  D1 0 / 4 (kp_v1 + fixes; truth D0 4 / 4, D1 0 / 2 on these seeds).
- Organizer (evaluator renderer) accuracy, kp_v2: tracking all blocks p90 0.70 / p99 1.72 mm, movable p90 0.67 mm,
  held p90 0.24 mm; first call p99 0.91 mm; whole episodes: placements relative to the destination 0.09-0.55 mm.
- Open: (1) Task 4: the learned push actor and perception (target lost while tipping over the tower edge; the
  executor holds the pusher, then "lost_departed" may fire early; lighting case). Most promising: fine-tune the push
  skill with perception-like dropouts and stale poses, as Step 8.1 did for grasp / place, or improve tracking through
  the tip-over (rotation about the edge). (2) Task 5 push-in: the insert deep under the roof is tracked to 1-17 mm
  (closed loop succeeds). (3) The policy server (`tower-infer-v1`) wiring the estimator, the executor and the bundle.

### Write-up and videos (2026-10-21 afternoon)

- `~/Documents/corl_challenge/Pose-estimator-v1.MD`: the session's narrative and reference (build, closed loop, results,
  commands per task).
- `rehearse.py --perception <model.pt> --video <dir>` (`tower_rl/perception/video.py`, hooked into `CameraPerception`):
  one mp4 per episode (`<variant>_<tier>_seed<seed>_<success|fail>.mp4`), a frame per estimator call: the four
  delivered images with every block's estimated box (visible edges, ray cast against the estimates; colour = state:
  tracked, foundation, held, node target, weak, not seen, ghost), the true box (white) of blocks > 2 mm off, and a
  status bar (node, target error and confidence, counts, held blocks). `--video_maps` adds the edge map and corner
  peaks; `--live` renders the stage after every row (without `--headless`: the Isaac Sim window, play_episode.py's view).
  Drawing costs ~0.02 s per frame (0.1 s with maps).
- Checks (kp_v2, final code, one episode each): Task 2-2 D0 TSR 1.00, 121 calls, errors p99 0.28 mm, 51 s; Task 3-2 D1
  (`--live` headless, `--video_maps`) TSR 1.00, 117 calls, p99 0.43 mm, 72 s. Test
  `test_video_frame_draws_the_estimates_and_the_truth_where_off`; full suite passes.
