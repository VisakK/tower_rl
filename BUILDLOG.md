# BUILDLOG: agentic teacher (TOWER challenge)

Build instructions: `~/Documents/corl_challenge/TOWER_BUILD_INSTRUCTIONS.md` (the "build file"). One entry per session:
what was built, acceptance numbers, deviations from the build file with reasons.

## Status (updated 2026-11-05)

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
| Student: runtime and oracle (session 5) | Steps 1-2 passed; **G3a passed** (480 / 480 episodes with identical node events vs the executor); **G3b passed** (aggregate 447 vs 446 / 480; the Task 1-3 D1 cell re-checked on 48 new episodes: 47 / 48 each, events identical) | "session 5" |
| Student: switcher | **G6 passed** (3,000 episodes, 50,457 decisions); bc_r0: **G7a** nominal 1.000 / AUC 0.993 pass, non-nominal 0.954 < 0.98; **G7b**: 393 / 480 vs oracle 405 (fail: Task 1-3 redo of a delivered goal near the yaw tolerance) -> with the opt-in `keep_delivered` rule **402 / 480: passed**; **G8 passed within noise** with `bc_r2` *without* the rule (404 vs oracle 405 / 480; recovery 55.5 vs 59.3 %; the rule is dropped) | "G6", "G7b", "Results of the 08:00 queues" entries |
| Student: skills | **9c passed** (distilled Task 5 grasp / insert / push-in, both arms: 83 / 96 vs planned 78 / 96); **9a** GPU gates passed (left 98.8 %, back 95.4 %), task-level gate failed (Task 4 under v1 noise 53 / 96 vs privileged 84); **9b** fine-tunes done (grasp better at 3 N), grip matrix not re-run | "Step 9a / 9b / 9c", "G9c", "Results of the 08:00 queues" entries |
| Student: camera loop (Step 10) | `student_v1` frozen (bc_r2 + distilled v2 + 9a pushes + kp_v2); **G10 not passed, only on the Task 4 bar**: 135 / 160 vs the camera-loop teacher's 131 / 160, every cell >= teacher - 1 / 8, Tasks 1-3 96 / 96, Task 4 18 / 32 (teacher 15; bar 7 / 8 D0, 5 / 8 D1), Task 5 21 / 32 (22); `pushin_kin` A/B 14 vs 21 / 32: stays off | "Results of Step 10 / G10" |
| Server | `StudentAdapter` (observation keys checked against the live docs); submission flow re-read 2026-10-24; **next: a first official evaluation of `student_v1`, before any fix**; `scripts/serve_student.py` waits for the starter kit and `track.json` (HUMAN: the site resets TLS from this machine) | "Step 11: protocol details from the live docs", "Submission before fixes" |
| Official run 1 | `student_v1`: **59.32 points**, est. rank 1 (site 2026-10-27, `awaiting_review`); Tasks 1-3 blind (fixed-camera orientation), Task 4 TSR 0 | `corl_challenge/Lessons-v1.MD`, `Plan-v1.1.MD` |
| `student_v1_1` (session 7) | **frozen and served**: orientation per episode (125 / 125 official frames), finished guard (Tasks 1-3 120 / 120 vs 1 / 120 without, noise model), 16-row idle holds, blind guard, `push_settle` (+2.1 points on Task 4, two seeds); camera loop Tasks 1-3 60 / 60 to 210 s; end to end through the server clean; endpoint `official_v1_1` port 9002 (HUMAN: Save model, Check endpoint, Queue evaluation) | "session 7", `corl_challenge/Student-v1.1.MD` |
| Official mirror | `rehearse.py --mirror` (to 210 s, final arrangement, Task 4 rest states), `scripts/mirror_points.py`; reproduces official 4-2 LHPR exactly, 4-3 within 3.4 points | "session 7" |
| Recovery node | contracts + rules (`tower_rl/recovery/`), teacher `oracle_state_v2` (+0.4 points on truth: remaining failures are prevention problems), student rule layer (no effect), features-v2 router (`bc_v2_r*`, DAgger) | "session 7" |
| `student_v1_2` (session 7) | `student_v1_1` + the end-of-plan re-push rule on Task 4; two official evaluations cut off by machine resets | "session 7", "session 8" |
| `student_v1_3` (session 8) | router `bc_v2_r3`, `offcentre_place`; every gate passed; **the final submission: official evaluation complete 2026-10-31, 100 episodes, 0 errors on the server** | "session 8", `docs/Student-v1.3.MD` |
| Learned reaching, `student_v1_4`, `student_v1_5` (sessions 8-11) | after the deadline, never evaluated | `docs/Reacher-V3.MD`, `docs/Student-v1.4.MD` |
| Reproduction package (session 11) | `reproduce/`: rebuilds `student_v1_3` from the public downloads, one script per stage; the training stages not rerun | `reproduce/README.md` |
| Review handover (2026-11-05) | `eval/*` tags with the exact evaluated code, `REVIEW.md`, `reproduce/evaluated/`, `reproduce/environment/` | "2026-11-05" |

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

## 2026-10-21/22 (session 5): student

Plan: `~/Documents/corl_challenge/Student-distillation-plan.MD` (the "plan"; approved 2026-10-21). The student (option
switcher + skills + kp_v2) is the submission; the executor, the graphs and the teacher bundles stay unchanged as the
parity reference and the fallback. Steps: 0 setup + grip check + kp_v2 teacher camera loop; 1 task spec; 2 option
runtime; 3 oracle + parity (G3a, G3b); 4 tokens + logging; 5 estimator noise model; 6 round-0 collection; 7 switcher
BC; 8 DAgger; 9 skills (9a push with estimator failure modes, 9b grip, 9c Task 5 distillation); 10 camera-loop
validation; 11 server + official evaluations; 12 AWR over options (optional); 13 freeze and submit.

### Step 0: setup (2026-10-21 23:35)

- Session-4 work was already committed by the user (`f5e8cce Pose estimator - v1`); working tree clean at the start.
- Code backup: `results/backups/code_2026-10-21_pre_student.tgz` (BUILDLOG, README, configs, scripts, tests, tower_rl).
- `--finger_effort <N>` added to `rehearse.py` (`make_station_scene_cfg(..., finger_effort=)`: overrides the layout's
  `physics.finger_effort_n`; default unchanged) and passed through by `rehearse_many.py` (also new: `--student`,
  `--extra` = further rehearse.py arguments verbatim).
- Step 0.4 (kp_v2 teacher camera loop, GPU) launched 23:36: `results/student/gpu_queue_camera_teacher.sh`, all ten
  variants, D0 + D1, 8 episodes per tier, seed 7000, 2 envs -> `results/student/closed_loop/teacher_kp_v2/`, logs
  `results/student/logs/camera_teacher_kp_v2_<variant>.log`. It runs from a frozen copy of the code
  (`~/Documents/tower_rl_snapshots/camera_teacher_kp_v2`: tower_rl / scripts / configs copied, data / assets_m /
  results symlinked), because its ten processes start over ~5 h while the student code changes.
- Step 0.3 (grip check, CPU) launched 23:40: `results/student/cpu_queue_grip_check.sh` (same frozen snapshot):
  teacher_t45_final on the six Tasks 1-3 variants, D0 + D1, N = 24, seed 5000, 4 envs, configurations baseline
  (22 N, 0.040), A (3 N, 0.040), B (3 N, 0.0425), C (22 N, 0.0425) -> `results/student/grip_check/<config>/`.

### Step 1: task spec (2026-10-21 23:45): PASSED

`tower_rl/student/spec.py`: `TaskSpec.from_graph / load(variant, bundle, graph_set)`, `Goal` (type, block, prescribed
arm, stages with the graph's node specs, geometry), `Decision(option, goal, arm, park, alt)`, `park_spec(arm, k, kind)`
(graph / end / alt / auto, named as the executor names them), `option_nodes(decision)` (deep copies; another arm ->
`_alt` ids and `first_of_transfer` false, as `_schedule_arm_switch`), `graph_nodes()` (the round trip). Stages are
recognised by (skill, transit target), ids checked against `{t,p,i}{k}_<suffix>`; parks are validated as "between
transfers that switch arms, parking the previous arm" and "once at the end".
- Gate: `pytest tests/test_student_spec.py` 40 passed: all ten graphs, repo and `teacher_t45_final` copies, rebuild
  their node lists exactly (every field); the alt park + other-arm acquire / deliver + park equal the node list the
  executor's own `_advance` queues (Tasks 1-3, every goal); malformed graphs are rejected.
- Note: `tower_rl.data.layouts.VARIANTS` lists only Tasks 1-3 (`ALL_VARIANTS` has all ten); `rehearse_many.py`
  defaults to `VARIANTS`, so every student run passes `--variants` explicitly.

### Step 2: option runtime (2026-10-21 23:50-00:20): PASSED

`tower_rl/student/runtime.py`: `StudentRun(EnvRun)` (option queue instead of the graph pointer; decides when an
option ends inside the same `act` call, as `_advance` moves the pointer; the queue head is the next node, so
`focus_block` / `node_info` match the executor's pointer), `Memory` (per goal tries / attempts per arm / last option,
status and failure category / arm-switch state; per arm last option, moved since park, holding goal; option log),
`View`, `IdleNode`, `StudentExecutor(Executor)` (spec built at `reset` from the graph dict, so `--graph_set` applies),
`make_decider`. Mechanical rules: the gripper of an arm that holds a block the next option does not deliver / insert
is opened first (this reproduces the executor's resets on skip / restart / arm switch exactly); `arm_clear` parks as
inserted nodes (not for the oracle's arm-switch items, as the executor); masks (invalid -> idle, counted); idle
re-query every 2 s (`configs/student/switcher.yaml`); rows clipped to the joint / gripper limits, NaN -> hold chunk.
`Executor.focus_block / node_info` added (today's semantics) and `CameraPerception` uses them. `rehearse.py --student
oracle_graph | oracle_state | <switcher.pt>` (dirs `<tier>_student-<name>`), passed through by `rehearse_many.py`.
- Gate: `pytest tests/test_student_runtime.py` (scripted decider + stub nodes: acquire -> deliver -> park -> idle,
  failure aborts the option and opens an empty gripper, a held block survives a failed deliver but not the next
  non-deliver option, masks, inserted parks, focus, clipping, idle re-query) passes; smoke rehearsal Task 2-2 D0 with
  `--student oracle_graph`: TSR 1.00 (`results/student/smoke/`).
- Extra pre-check without Isaac: `tower_rl/student/fakeworld.py` (kinematic stand-in: arms settle on the sag
  equilibrium of their commands, closing fingers grab a block within 12 mm of the pinch, released blocks drop onto
  their support, scripted release offsets / drops / reachability) and `scripts/student_fake_parity.py`: the executor
  and the runtime with `oracle_graph` side by side, planned skills, 210 s, all ten variants x 8 scenarios (nominal,
  misplacements -> retries and skips, arm switch and a re-miss after it, a block fallen off the table, empty grasps,
  a block dropped in a held transit): **80 / 80 identical rows and node events** (`results/student/parity/fake_world.json`).
- **Teacher bug found** (fake world, then confirmed in old logs): `_clearance_park` re-parks an arm that is already at
  home when the next transit's target lies within `ARM_CLEAR_RADIUS_M` (0.18 m) of that arm's home pinch (a dropped or
  knocked-off block): an inserted park every 0.5 s until the episode ends. In all past rehearsal logs: 54 of 7,909
  episodes (all failures), Task 1-3 20 / 1208, Task 4-3 12 / 428, Task 2-3 6 / 1294, Task 3-3 6 / 874, others 0-3. The
  executor stays unchanged; the student runtime skips the inserted park when the other arm has not moved since its
  last park, except for exact-executor deciders (`Oracle("graph")`, `exact=True`, so G3a can hold).

### Step 3 (in progress) and Step 4: tokens and logging (2026-10-22 00:00-00:25): Step 4 PASSED

- `tower_rl/student/oracle.py`: `Oracle("graph")` (exact `EnvRun._advance` semantics as a function of the runtime memory:
  per-goal stage machine, `on_fail`, fell-off skip, retries, arm-switch sequence, the graph's structural parks) and
  `Oracle("state")` (the training expert; changes from graph mode, all reading truth only through `privileged`):
  1. satisfaction from truth; 2. a goal truth finds satisfied is closed even after a failed verdict; 3. a closed goal
  that truth no longer finds satisfied is reopened (transfer family, block on top of the stack and on the table level
  or above, >= 30 s left, once); 4. re-grasps use the prescribed arm if it reaches the pre-grasp, else the other arm
  (transfer family only: insertion directions are arm-specific); this replaces graph mode's arm-switch sequence;
  5. parks by "the other arm moved since its last park" (graph mode parks by the graph's structure, so it re-parks an
  arm the arm-switch sequence already parked, e.g. `park_back_1` right after `t0_alt_park_back`: kept for parity);
  6. an arm holding an open goal's block delivers it when that goal is next (or, transfer family, when every
  lower-layer goal is satisfied): labels for states the switcher reaches off the graph. Deviation from the plan's
  wording: changes 4-6 are made explicit here; the plan's pseudo-code parks by "moved since park" in both modes, which
  breaks parity after an arm switch (above).
- Truth for the oracle and the progress head: `OperationTracker.instant / PushTracker.instant` (the P condition
  without its settle time; `own=True`: the goal's own condition, without the lower-layer / structure terms, which the
  state oracle uses so a correctly placed block is not re-grasped when a lower goal was given up), `latched()`,
  `disturbed_so_far()`; `run_batch` attaches `ops`, `ops_own`, `ops_latched`, `disturbed` to every state's
  `privileged` (the executor never reads it).
- `tower_rl/student/tokens.py` (features v1, Appendix B: block 38, goal 88, arm 27, global 16; action masks; encode /
  decode), `TokenTracker` (per-episode histories: start poses, 1 s speeds, sightings, reachability cached until a
  block that moved comes to rest: `_pregrasp_reachable` costs ~0.12 s of IK). `tower_rl/student/logging.py` (Appendix
  C shards + `index.jsonl`, `load_shards` refuses other feature versions). Runtime: `collect`, a labeller (the
  state-mode oracle by default), DAgger `beta` mixing and exploration `explore_eps` (random valid alternative, own
  per-episode RNG); `rehearse.py --collect <dir> --explore --beta`.
- Gate: `pytest tests/test_student_tokens.py` 15 passed (shapes and masks on all ten variants, a `privileged` that
  raises on any read, half-turn symmetry on Tasks 2-3 / 4-2 / 5-3, transport masks, encode / decode). Collection
  `--student oracle_state --collect` on Task 2-3 D1, 4 episodes (TSR 1.00): 176 rows (36 decisions, 140 progress
  samples), all 36 labels equal the executed decisions and are valid under the logged masks, progress padding -1.

### Step 3: parity G3a, a bug found and fixed (2026-10-22 00:10-00:50)

- First G3a run (executor vs `--student oracle_graph`, separate processes, all ten variants, D0 / D1, N = 24, seed
  5000): node ids, statuses and times matched, but the info values differed from the first **left-arm** grasp on
  (Task 1-3 t0, Task 2-3 t1, Task 5-3 i1); back-arm-only Task 3-3 batch 0 and push-only Task 4-3 were identical.
- Tools added to find it: `ShadowPolicy` + `rehearse.py --shadow` (the executor drives the simulation, the student
  runtime gets the same states in the same process; per episode the row deviation, the first call beyond 1e-4, and
  whether the node events are equal; `TOWER_SHADOW_DUMP=<dir>` pickles the state of the first inexact call), and a
  control (`--shadow` without `--student`: executor vs a second executor instance, bit-identical: 0 inexact calls).
  Separate-process reproducibility of the executor itself was confirmed (two processes, identical events).
- Cause: the runtime's row clipping to the gripper range [0, 0.1]. Isaac reports the open left gripper at
  0.10000002 m (float32), the executor commands that back on its first rows, and the clip changed it by 2.4e-8; the
  physics amplified it at the first left-arm grasp. Reproduced offline from the dumped states (bit-for-bit equal to
  the Isaac rows). Fix: clip only values beyond the limits by more than 1e-6 (`CLIP_TOL`); exact clipping to the
  track's action bounds belongs in the server (Step 11). The first run's results are kept as
  `results/student/parity/old_clipbug_*`; G3a relaunched 00:50 from a fresh snapshot.
- Shadow results before the fix (Task 1-3 and 2-3 D0, 4 episodes each): node events equal in 8 / 8 episodes despite
  the 1-ulp rows (the decision layer itself agreed).

### Steps 5-7: code (2026-10-22 00:00-01:00; gates pending)

- **Step 5, estimator noise:** `scripts/fit_estimator_noise.py` (per call and block a class: foundation / static / held /
  target:<skill> / moving (truth speed > 1 cm/s) / fallen (truth z < 0); position error, rotation-error vector (box
  symmetry nearest the truth) and confidence level per call; contiguous same-class runs = replay traces; push targets
  also P(lost | tilt, travel) + lost-run lengths) -> `configs/student/noise.yaml` + `results/student/noise/traces.npz`.
  `tower_rl/student/noise.py`: `NoiseModel` (fitted model, one `EstimatorNoise` per env via `make`), replay per class,
  foundation / static traces keep their last error (the estimator freezes / anchors them), "not seen" = previous
  output coasted x0.8 for <= 5 calls when moving, ghosts keep their pose, quaternions continuous, truth kept in
  `privileged`; `run_batch(noise=NoiseModel)` (focus = the policy's `focus_block` / `node_info`);
  `rehearse.py --est_noise <yaml>` (dirs `_estnoise`; per-call log -> perception.json / perception_log.pkl, off when
  collecting). `pytest tests/test_estimator_noise.py` 5 passed.
- Fit of 01:00 (29 logs; kp_v2 preferred when it has >= 40 calls of a class): foundation / static / held / transit /
  grasp / place / push / moving from kp_v2, insert / push_in from kp_v1b, fallen from kp_v1. Moving blocks: kp_v1 /
  kp_v1b (early estimator code, Task 4-3) had p90 125-133 mm at full confidence (mis-tracked blocks tipping off the
  tower); kp_v2 (final code) p90 0.53 mm, p99 1.18 mm (only 50 calls yet). To refit when the Step 0.4 camera loop
  (all ten variants, kp_v2) has finished.
- **Step 6, injection + exploration:** `tower_rl/student/inject.py` + `configs/student/collect.yaml`; opt-in hooks in
  the executor (`spec["inject"]`, absent from every graph): shifted pre-grasp / pre-push (`target_shift`), a drop part-way
  up the lift (`drop_at_lift_m`, open 3 rows), a drop in the held transit (`drop_after_rows`), a shifted or yawed goal
  (`goal_shift` / `goal_yaw_deg` in `dest_goal`), an entry offset (`slot_shift` in `slot_pose`). Deviation: the lift
  drop is triggered by height (2.5-5 cm), not by time (the grasp node ends at 6 cm, often before 0.5-2 s).
  `rehearse.py --inject <p>`; `pytest tests/test_student_inject.py` 6 passed (each perturbation produces its failure
  through the normal node checks on the fake world).
- **Step 7, switcher:** `tower_rl/student/switcher.py` (`SwitcherNet`: per-type projections + type embeddings, pre-LN
  encoder 3 x d 128 x 4 heads, ff 384; option head, goal pointer (query from global + option embedding), arm head
  (global + option + chosen goal, a learned null goal for parks), progress and value heads; 0.67 M parameters;
  masked argmax `decide` is pure tensor code; `export` = TorchScript + JSON with the normaliser; `SwitcherDecider`
  refuses another feature version), `scripts/train_switcher.py` (episode split 90 / 10, normaliser fit on train rows,
  AdamW + one-cycle, best epoch by validation decision accuracy), `scripts/eval_switcher_offline.py` (per head, per
  variant, nominal / non-nominal, option confusion, progress AUC + calibration). Smoke: 4 episodes -> held-out
  episode decision accuracy 1.0, progress AUC 0.99; scripted call ~5 ms on the loaded CPU.
- Shard format change: one `index.jsonl` per `<variant>/<tier>` (parallel rehearsals must not append to one file);
  `load_shards` numbers the episodes. Shards also carry `nonnominal` (after a failed option, an exploration or an
  injection in that episode). Repeated idle decisions (an episode idling to 210 s after its goals) are logged once per
  10 re-queries (they were 275 of 418 decisions in a smoke run).
- State-oracle fixes from the first collection smoke run (inject 0.3, explore 0.15, Tasks 1-2 / 4-2 / 5-2 D1):
  a disturbed lower goal was reopened while the other arm held the next block, which the mechanical rule then dropped:
  reopens now wait while an arm holds a block, and a held block is deliverable unless an open goal lies on a lower
  layer; change 7: a block lying tilted > 15 deg is given up instead of re-grasped (the learned grasp timed out 3 x
  15 s on dropped blocks). G3b's run started before these changes (they only act after failures).

### G3a (2026-10-22 01:27): executor vs `oracle_graph`, separate processes

All ten variants, D0 + D1, N = 24, seed 5000, 4 envs, simulator poses, `teacher_t45_final`, one snapshot
(`results/student/parity/{executor,oracle_graph}`, table `results/student/parity/g3a.md`): **476 / 480 episodes with
identical node-event sequences (ids, statuses, times, every info value); TSR identical in every cell (446 / 480 =
0.929 both).** The 4 others are one batch (Task 1-3 D1, seeds 5500000-5500003, the 6th D1 batch of that process):
same ids / statuses / times, info values differ from the first grasp on, success identical. All earlier batches of
that process were identical, so the batch's PhysX start state differs; follow-up: a shadow run of the whole Task 1-3
D0 + D1 sequence (below). The kinematic fake world: 80 / 80 scenarios bit-identical rows (Step 2).

Teacher numbers on these seeds (simulator poses): Tasks 1-3 0.917-1.000 per cell; 4-2 D0 / D1 0.958 / 0.875; 4-3 0.792 /
0.708; 5-2 1.000 / 0.750; 5-3 1.000 / 0.750.

### Step 0.3: grip check (2026-10-22 01:20)

teacher_t45_final, Tasks 1-3, D0 + D1, N = 24, seed 5000, 4 envs (`results/student/grip_check/<config>`; the frozen
snapshot of 23:36, i.e. before the student code):

| variant | tier | baseline 22 N, 0.040 | A 3 N, 0.040 | B 3 N, 0.0425 | C 22 N, 0.0425 |
|---|---|---|---|---|---|
| task1-2_tower_transfer_2 | D0 | 24 / 24 | 24 / 24 (+0) | 23 / 24 (-1) | 23 / 24 (-1) |
| task1-2_tower_transfer_2 | D1 | 22 / 24 | 23 / 24 (+1) | 19 / 24 (-3) | 18 / 24 (-4) |
| task1-3_tower_transfer_3 | D0 | 23 / 24 | 24 / 24 (+1) | 24 / 24 (+1) | 24 / 24 (+1) |
| task1-3_tower_transfer_3 | D1 | 23 / 24 | 19 / 24 (-4) | 21 / 24 (-2) | 20 / 24 (-3) |
| task2-2_cross_tower_transfer_2 | D0 | 24 / 24 | 21 / 24 (-3) | 24 / 24 (+0) | 24 / 24 (+0) |
| task2-2_cross_tower_transfer_2 | D1 | 23 / 24 | 18 / 24 (-5) | 23 / 24 (+0) | 22 / 24 (-1) |
| task2-3_cross_tower_transfer_3 | D0 | 24 / 24 | 24 / 24 (+0) | 23 / 24 (-1) | 24 / 24 (+0) |
| task2-3_cross_tower_transfer_3 | D1 | 23 / 24 | 22 / 24 (-1) | 23 / 24 (+0) | 23 / 24 (+0) |
| task3-2_vertical_stacking_2 | D0 | 24 / 24 | 24 / 24 (+0) | 24 / 24 (+0) | 23 / 24 (-1) |
| task3-2_vertical_stacking_2 | D1 | 24 / 24 | 24 / 24 (+0) | 24 / 24 (+0) | 24 / 24 (+0) |
| task3-3_vertical_stacking_3 | D0 | 24 / 24 | 24 / 24 (+0) | 24 / 24 (+0) | 24 / 24 (+0) |
| task3-3_vertical_stacking_3 | D1 | 24 / 24 | 24 / 24 (+0) | 24 / 24 (+0) | 24 / 24 (+0) |
| **all** | | 282 / 288 (97.9 %) | 271 / 288 (94.1 %) | 276 / 288 (95.8 %) | 273 / 288 (94.8 %) |
- At 3 N with today's closed width (A) the teacher loses 4 points (Task 2-2 D1 18 / 24, Task 1-3 D1 19 / 24): the
  open 3 N question matters. 0.0425 m gives the same grip at both efforts (B and C: ~2.9 N per finger, unsaturated;
  95.8 / 94.8 %), but loses 3-4 / 24 on Task 1-2 D1 (and 2-3 / 24 on Task 1-3 D1) against the 22 N / 0.040 baseline
  (3.9 N per finger).
- **Step 9b decision (the plan's rule):** B and C are not within 2 / 24 of the baseline in every cell (Task 1-2 D1 -3
  and -4), so 0.0425 is not adopted as is; next is the fine-tune of `grasp_lift` / `place_release` (both arms) at a
  closed width of 0.0425 with the finger effort drawn per env from {3, 22} N (GPU, ~4 x 25 min; after the camera loop).
  Separate-process A/B noise at N = 24 is about +-2 / 24 per cell, so the deciding cells are close to the rule.

### G3b, first run (2026-10-22 01:25) and a better change 4

- `oracle_state` (code of 00:45, i.e. with the original change 4) vs G3a's `oracle_graph`, same seeds / settings
  (`results/student/parity/g3b.md`): aggregate **448 / 480 vs 446 / 480** (passes), but Task 1-3 D1 **21 / 24 vs
  23 / 24** (fails "no cell worse by > 1 / 24"); better cells: Task 1-2 D1 +1, 2-2 D1 +1, 4-3 D1 +2. Both Task 1-3 D1
  losses: a near-miss first place (5.4 / 6.3 mm) -> change 4 re-grasped with the other arm, whose grasp timed out, up
  to three tries, then the episode collapsed (later transits timed out / pre-grasps unreachable).
- Evidence from every past teacher rehearsal log (7,909 episodes): the arm-switch re-grasp (`*_grasp_alt`) timed out in
  **206 of 217** attempts (8 ok); **205 of 211** episodes with an arm switch failed. Grasp retries after a timeout
  succeed 17 / 131; after a drop 20 / 38.
- **Change 4 revised** (deviation from the plan's "whichever arm can reach"): a re-grasp uses the prescribed arm; when it
  cannot reach the pre-grasp, the goal is given up (no other-arm attempt; also no reopen). Test
  `test_state_mode_gives_up_what_the_prescribed_arm_cannot_regrasp`. The round-0 collection and the eval reference
  that had started at 01:15 / 01:27 with the old rule were stopped (outputs kept as `*old*_change4`) and restarted
  01:38 from a fresh snapshot; G3b is re-run with the final state oracle (`results/student/parity/oracle_state_v2`).

### Kit file watchers exhaust inotify (2026-10-22 01:10)

The Step 0.4 camera-loop process of Task 3-3 aborted at Kit start-up in omniclient's file thread (exit 134); the
earlier "Failed to create change watch ... errno=28" warnings were the cause: 63,948 of the user's 65,536 inotify watches
were in use (GNOME's tracker-miner-fs 33.6k, VS Code ~24k, ~384 per Kit process). `rehearse.py` now passes
`--/app/extensions/fsWatcherEnabled=0` (a non-persistent /app setting) unless `--kit_args` sets it: 384 -> 0 watches
per process, results unaffected (patched in the repo and in the three snapshots, whose later processes read it). Task
3-3 is queued again after the camera loop (`results/student/gpu_queue_camera_teacher_rerun.sh`). HUMAN (optional):
`sudo sysctl fs.inotify.max_user_watches=524288` would remove the limit for good.

### G3a follow-up: shadow parity on Task 1-3 and a second clipping difference (2026-10-22 02:05)

- Shadow run (`--shadow`, Task 1-3 D0 + D1, N = 24, seed 5000): D0 bit-identical over 9,466 calls; D1 all 24 episodes
  with equal events, one call different by 0.0325 (env 0, t 130.1 s). Replayed offline from the dumped states: after
  three instant "pregrasp unreachable" failures the executor starts `park_back_4` from an extreme grasp posture, and
  `ParkNode` sends the sag-compensated command unclamped, past back joint 4's limit for 4 rows; the runtime clipped them.
  This is also what shifted the PhysX history of G3a's one divergent batch. Fix: no clipping for exact-executor
  deciders (`oracle_graph`); the student (non-exact) clips to the joint limits, and the server will clip to the
  track's action bounds (Step 11; note: the executor's own rows can exceed the joint limits by ~0.03 rad in parks).
  After the fix the 794 calls replay identically. Confirmation run queued: Task 1-3 executor vs `oracle_graph`, D0 + D1,
  N = 24, separate processes (`results/student/parity/t13_v3.md`).
- The round-0 collection (state oracle, non-exact) is unaffected.
- The G7b reference run (`oracle_state`, `--est_noise`, seeds 50000) was paused with SIGSTOP at 01:58 to give round 0
  the CPU; to be resumed (SIGCONT) before the switcher's closed-loop evaluation.

### Step 9a: code (2026-10-22 02:00-02:20; GPU training queued behind the camera loop)

- `push_actor_obs(..., seen=, since_seen=)`: 67-D when given (65-D otherwise, unchanged).
- `PushEnv` with `obs_noise.estimator` (off by default; then `observation_space` 67 / `state_space` 161): the target is
  lost with the fitted P(lost | tilt, travel) (`configs/student/noise.yaml` `push_lost`) for recorded run lengths;
  while lost, the observed target is the coasted pose (last seen + last per-step displacement x 0.8^k, k <= 5, only if
  it was moving) or stale, `seen` = 0, `since_seen` = k / 5; applied inside `noisy_block_poses` (once per step, so the
  action scaling sees the same estimate); `target_unseen_frac` logged. Other blocks: the existing `dropout_p`.
- `scripts/widen_push_checkpoint.py`: v3 / v1c -> 67-D (zero first-layer columns for the new inputs, critic columns
  inserted after the actor part at index 65, normalizer mean 0 / var 1; the widened actor's outputs equal the original's
  exactly); VERIFY done: rsl_rl's buffers are `actor_obs_normalizer._mean/_var/_std` + `count`.
  `train_skill.py --resume_no_optimizer`. CPU smoke: 2 iterations of the 67-D env resumed from widened v3, no errors.
- Executor (opt-in by the policy): `SkillPolicy.obs_dim` from the export JSON; `PushNode` with a 67-D policy feeds
  `seen` / `since_seen`, skips the hand-coded hold (keeps `lost_departed` as a mechanical end), and counts hand-over
  rows only when the target is seen. 65-D policies behave exactly as before.
- Queue `results/student/gpu_queue_push9a.sh` (started; waits for both camera-loop queues): refit the noise model on all
  kp_v2 logs into `traces_v2.npz` (running collections keep their bank), then per arm 250 iterations from the widened
  checkpoint (`--override` estimator noise + 1.5 % dropouts) and the gate evaluation (500 episodes, D1, same noise).

### Round 0 (Step 6.2), running

Restarted 01:38 (D1, 200 per variant, seed 105000, 8 processes); throughput ~570 episodes / h on the shared CPU; the
D0 part (100 per variant, seed 100000) was started in parallel at 02:15 (5 processes) instead of after D1.

### G3b v2 (2026-10-22 02:40): final state oracle vs graph mode

`results/student/parity/g3b_v2.md` (state oracle with the revised change 4 and the 01:00 fixes, same seeds / settings as
G3a): **aggregate 447 / 480 vs 446 / 480 (passes)**; better or equal everywhere except Task 1-3 D1 **21 / 24 vs
23 / 24** (the per-cell rule "no cell worse by > 1 / 24" fails there, as in the first run), with higher GSR / PSR
there (1.000 / 0.979 vs 0.972 / 0.958) and ODR 0 vs 0.031; Task 4-3 D1 +2, Task 2-2 D1 +1. The three lost episodes:
near-miss first places (yaw 5.03 / 5.57 deg, i.e. just over the 5 deg tolerance; one 9.8 mm) whose re-grasp and re-place
failed worse (11, 17, 218 mm). Graph mode retries near misses the same way; its run had near misses in other episodes
(once any decision differs, the process's later physics differs, so per-episode outcomes are not paired). Reading:
within noise at N = 24; open: an extended Task 1-3 D1 comparison (more seeds) when the CPU is free. Note for later:
near-miss re-places rarely succeed (both oracles); the evaluator's tolerances are an open question to the organizers.

### G3a: PASSED (2026-10-22 02:40)

Task 1-3 re-run with the exact-mode fix (executor vs `oracle_graph`, D0 + D1, N = 24, separate processes,
`results/student/parity/t13_v3.md`): **48 / 48 identical event sequences**. With the other nine variants of the full run
(432 / 432, rows unaffected by the change), all ten variants x D0 / D1 x 24 episodes are identical: graph mode
reproduces the executor exactly.

### Step 7 preview: BC on the first 584 round-0 episodes (2026-10-22 02:35)

`results/student/models/bc_r0_early` (CPU, 15 epochs, all decisions + 30 % of the progress rows per epoch; 108k rows,
9,771 decisions, no invalid labels; ~20 s per epoch): best held-out decision accuracy 0.932, **nominal 1.000** (171),
non-nominal 0.916 (752), progress AUC 0.987 (training loss still falling while validation accuracy plateaus: more data
needed). Offline (`eval_switcher_offline.py`, val episodes): errors are acquire <-> park (20 of 40) and goal choice in
the transfer family, in late multi-failure states; Tasks 2-2, 4-3, 5-3 non-nominal 100 %. Inspected cases: some labels
are not inferable from the tokens: (a) truth-based decisions at a tolerance boundary (a block perceived 4.9 mm from its
goal reopened; a failed place verdict on a block truth calls placed), (b) the oracle's own "given up" set after several
failures. Reopens are 107 of 11,455 round-0 decisions, 35 of them with a perceived offset < 8 mm: kept as they are (an
oracle change now would make the rounds' labels inconsistent); a known label-noise source (~0.3 %).

History check that shaped the oracle: after a near-miss place failure (<= 8 mm, <= 8 deg), the re-grasp and re-place
succeed in 134 of 821 cases and 123 of those episodes succeed (15 %): retrying near misses stays (TSR needs every P).
- Closed-loop check of `bc_r0_early` (`--est_noise`, eval seeds 50000, 8 episodes per tier,
  `results/student/eval/bc_r0_early`): **Task 2-2 D0 8 / 8, D1 8 / 8 (the state oracle on the same seeds: 8 / 8, 8 / 8)**;
  Task 4-2 D0 5 / 8, D1 3 / 8 (no Task 4-2 episodes were in its training data yet). Its Task 4-2 failures are mostly push
  failures under the noise model (shoved / structure, as for the oracle: the reference run has Task 4-3 D0 at 6 / 24
  with `--est_noise` vs 19 / 24 on simulator poses), plus erratic recovery decisions after them (e.g. p1, p2, p1, p4).
- `results/student/cpu_queue_after_r0.sh` (queued 02:50): after round 0, shard summary, `bc_r0` (CPU), offline
  evaluation, then the G7b evaluation and DAgger round 1 (beta 0.5, 1,500 episodes) in parallel.

### Queued pipeline (2026-10-22 03:10)

CPU (all from the `round0` snapshot, so every switcher round uses the same code and the 01:00 noise fit v1):
1. `cpu_queue_after_r0.sh`: after round 0 -> `r0_summary.json` (G6) -> `bc_r0` -> offline evaluation (G7a) -> in
   parallel the G7b evaluation (`eval/bc_r0_est`, to compare with `eval/oracle_state_est`) and DAgger round 1
   (`data/r1`, beta 0.5, D1 100 + D0 50 per variant, seeds 115000 / 110000).
2. `cpu_queue_after_r1.sh`: after round 1 -> `bc_r1` (r0 + r1) -> DAgger round 2 (`data/r2`, beta 0) -> `bc_r2`
   (r0 + r1 + r2) -> G8: `eval/bc_r2_est` (seeds 50000) and, with injection 0.15 at seeds 60000, `eval/bc_r2_inject` vs
   `eval/oracle_state_inject`.
3. `cpu_queue_g5_after_refit.sh`: after the noise refit (below) -> the teacher with the refit noise on the camera
   loop's episodes (`noise/g5_v2`) -> `noise/g5_v2_calibration.md` (G5).
GPU (one job at a time): the Step 0.4 camera loop -> its Task 3-3 re-run -> `gpu_queue_push9a.sh` (noise refit v2 into
`traces_v2.npz`, then Step 9a training + gate evaluation, both arms).
New code in this period: `tower_rl/student/adapter.py` (`StudentAdapter`, Step 11.2's session object: observation
-> estimator with the runtime's focus -> runtime -> clipped float32 chunk; hold chunks for blank images, no confident
block on the first calls, any exception; `tests/test_student_adapter.py` 3 passed with a stub estimator; observation
keys to VERIFY against the starter kit), `scripts/freeze_student_bundle.py`, `tests/test_switcher.py` (3 passed).

### Task 4 under the estimator noise model (2026-10-22 03:20)

The state oracle with `--est_noise` (eval seeds 50000, `eval/oracle_state_est`): Task 4-3 D0 6 / 24, D1 9 / 24; 4-2 D0
16 / 24 (simulator poses, G3b v2: 19 / 24, 19 / 24, 23 / 24). The failures are real: the tracker's truth finds the
remaining tower broken in 18 / 24 (4-3 D0) episodes (11 also caught by the push node's structure check, 7 not), vs
4 / 24 on simulator poses. The push actor (65-D, v3 / v1c) breaks the tower when its target is observed noisily and
intermittently lost while tipping, which is the camera loop's known Task 4 weakness (session 4: 4-3 D0 2 / 4, D1
0 / 4); the noise model reproduces it, and Step 9a (queued) retrains the push against exactly these losses. Tasks
1-3 and 5 are unaffected by the noise model (e.g. Task 2-3 / 3-3 24 / 24 in both tiers).

### Step 9c: code (2026-10-22 03:00-04:00)

- `tower_rl/student/distill/`: `core.py` (68-D observation `DistillObs` from the planned nodes' own quantities: pinch
  pose / velocity / lead, width and gripper commands, the block in the pinch frame, the reference pose (slot centre and
  axis; for the grasp the grasp target and the block's long axis), the block's offsets from it over the last 3 calls,
  node features, the previous action; 7-D action = target delta in the base frame / `POS_LIM` (grasp 8, insert 6,
  push-in 12 mm per row: the planned nodes' largest steps, never reached in a nominal run), rotation delta / 1.5 deg,
  gripper bit; `Dart` (Ornstein-Uhlenbeck target shift, plan sigma 1 mm / 0.15 deg per row, theta 0.3); `DistillNet`
  (256-256-256 ELU, normaliser inside, tanh motion + gripper logit)), `nodes.py`, `data.py`.
- Distilled nodes (`DistilledGraspNode` / `DistilledInsertNode` / `DistilledPushInNode`, subclasses of the planned
  ones): the network moves the target and decides when to close (grasp) / release (insert). Kept from the planned
  nodes: attach on the 14th closed row, the payload, grasped / empty-gripper verdicts, refresh of the held offset, the
  insert's release wait / retreat / slot verdict, the push-in's retreat / verdict, the structure checks, timeouts.
  Mechanical guards: the grasp closes once and stays closed (no motion while the fingers close, as phase 1); a block
  1 s at the grasp target closes; no release more than 1 cm short of the entry depth; a block 1 s at the entry depth is
  released; the push ends when the block is home or past it (> -1.5 mm), stalls after moving, or after 200 rows.
- Executor (opt-in, default off): `make_node` builds the distilled node when the graph (or node) has
  `"distilled": true` and the bundle has `skills/<skill>_bc_<arm>.pt` (`grasp_t5_bc`, `insert_bc`, `push_in_bc`), else
  the planned node as before; `Executor.node_hook` (None by default) lets data collection wrap nodes. Fake-world
  parity after the change: 80 / 80 identical; graph / student tests pass.
- Collection: `DartCollector` (the node hook) runs every planned Task 5 node under DART (per node a scale U(0, 1.5) of
  the base sigma; no shift while the fingers close or after the release) and records (observation, the planned
  node's command from the shifted target) for the rows the distilled node will produce (grasp: all; insert: the slide
  and the release row; push-in: descent and push). `rehearse.py --distill_collect <dir> [--dart_scale]` writes
  `tower_rl.student.distill.data` shards per batch; `scripts/collect_distill.py` runs one process per variant, tier
  and seed chunk; `scripts/train_distill.py` trains per skill and arm (episode split, MSE + 0.1 BCE with the rare
  gripper class weighted, best validation epoch, TorchScript + JSON).
- `tests/test_student_distill.py` (4 passed): on the fake world (with a scripted pusher) the hook records the planned
  Task 5-2 run, and **the distilled nodes driven by the recorded labels (a replay network) reproduce the planned run**
  (same node events, rows within 2e-3); DART labels are corrective; shards round-trip; a 2-epoch training exports a
  policy the executor loads.

### Step 0.4 progress: Task 5-3 on the camera loop (2026-10-22 03:18)

teacher_t45_final on kp_v2 (seed 7000, 8 per tier): **D0 7 / 8, D1 3 / 8 (ODR 1.30)**. D1's perception summary shows
toppled blocks (movable static pos p90 194 mm, rot p90 89 deg), i.e. structure collapses, not estimator errors on
intact blocks (target@insert p90 0.085 mm). For comparison the state oracle with `--est_noise` (eval seeds 50000):
5-3 D1 14 / 24. Compared per episode in Step 10.

### Step 9c: DART pilots (2026-10-22 03:20-03:40)

- Pilot 1 (`distill/data/pilot`, plan sigma 1 mm / 0.15 deg per row for every skill, 4 episodes per Task 5 cell): 2 / 16
  episodes; the planned grasp 24 / 24 ok, but the planned **insert timed out in 14 of 24 segments** (ok only at node
  scales <= 0.23) and failed "structure" in 5: it advances only while the held block is aligned within 0.8 mm /
  0.6 mm / 0.4 deg / 0.3 deg and otherwise holds its depth, so a 1 mm OU shift keeps it waiting. Per-skill sigma since
  (`DartCollector.SIGMA`): grasp 1 mm / 0.15 deg, **insert 0.2 mm / 0.04 deg**, push-in 0.5 mm / 0.1 deg, times U(0, 1.5)
  per node; the collector is reseeded per batch with the variant / tier as salt (pilot 1 drew the same scales in every
  cell).
- Pilot 2 (`distill/data/pilot2`, 8 per cell, seeds 291000): **25 / 32 episodes** (5-2 D0 8 / 8, D1 5 / 8; 5-3 D0 7 / 8,
  D1 5 / 8; the planned teacher without noise: D0 24 / 24, D1 18 / 24 in G3b): all 48 grasps and 16 back inserts ok,
  left inserts 28 / 32 (structure), push-ins 39 / 44 (structure, D1); 8,613 rows (grasp ~40, insert ~95, push-in ~45
  rows per segment).
- A 40-epoch training on pilot 2 and a CPU smoke run with those networks (`distill/smoke_pilot2`, Task 5-2 D0, 4
  episodes, `--graph_set '{"distilled": true}'`): no errors; distilled grasps 4 / 4 (lift 62 mm, gripper 1.4-1.8 deg
  from the canonical grasp vs ~0.5 planned); distilled inserts 0 / 4 (timeouts: 32 training segments). The trainer's
  rare-class weight is capped at 10 (`--w_rare_max`; at 50 the pilot network predicted the release 14 rows early).
- Full run queued 03:37: `results/student/cpu_queue_distill.sh` (snapshot `distill_v1`, PAR 4): collection
  `distill/data/c1` (300 episodes per cell, seeds 200000 / 210000 / 220000) -> `train_distill.py` -> bundle
  `results/bundles/student_9c_v1` -> gate runs (`distill/gate/{distilled_v1,planned}`, eval seeds 50000, N = 24) ->
  `distill/gate/distilled_v1.md`.

### Step 9b: code and queue (2026-10-22 03:20)

- Skill envs (`base_env.py`, opt-in): `gripper_closed_width` sets the env's closed command; `finger_effort_choices`
  draws a finger effort limit per env at every reset and writes it with `Articulation.write_joint_effort_limit_to_sim`
  (PhysX `set_dof_max_forces`). Checked on a 16-env CPU grasp env: the drawn limits {3, 22} N reach PhysX for both
  finger joints and are redrawn only for the reset envs.
- `results/student/gpu_queue_grip9b.sh` (after the 9a queue): fine-tune grasp_lift and place_release (both arms) from
  the teacher's checkpoints with their own training configs + closed width 0.0425 + effort {3, 22} N, 150 iterations
  each (`--tag 9b_grip`), then skill evaluations (500 episodes, D1) of the fine-tuned and the source checkpoints at
  0.0425 with 3 N and with 22 N -> `results/student/grip9b/eval_*.json`. The grip matrix re-run (CPU) follows once
  those look right.

### Step 11: protocol details from the live docs (2026-10-22 03:40)

The starter kit still cannot be downloaded from this machine (challenge.tower-benchmark.com resets TLS from here;
`curl` of `docs/downloads/tower-policy-example.zip`: connection reset), so the docs pages were read with WebFetch
(Observations, Actions, Reset and seeds, Model interface, Binary encoding, Python reference server). Recorded for the
server (Step 11.1):
- **Observation** (`infer` request; envelope `protocol: tower-infer-v1`, `type`, `session_id`, `trial_id`,
  `request_id`, `policy_id`): `observation.schema = tower-challenge-observation-v1`, `step_index` (control actions
  already executed), `sim_time_s`, `task_variant`, `instruction`, `joint_positions` float32 [2, 7] (left, back),
  `gripper_widths_m` float32 [2], `images` {left_back_camera, right_front_camera, left_wrist_camera,
  back_wrist_camera}: uint8 [H, W, 3] RGB at the release's size (Task 4/5 images in the dataset's JPEG orientation, no
  rotation). `StudentAdapter.parse_observation` already reads exactly these keys (docstring updated).
- **infer_result**: the five identity fields echoed (protocol, session_id, trial_id, request_id, policy_id), `type`,
  `plugin`, `plugin_version` (as in the metadata), `base_seed`, `seed_scheme: tower-seed-stream-v1`, `sample_index`
  (+1 per call), `sample_seed = sample_seed(base_seed, sample_index)`, `actions` little-endian float32 [H, 16]
  (1 <= H <= track `max_chunk_actions` = 16; absolute radians and full opening widths in metres; within the release's
  `action_lower` / `action_upper`, else the chunk fails validation), `dt_s = track["action_dt_s"]` (0.1, or exactly 1/30).
- **reset** (identity fields + uint32 `seed`) -> `reset_ack` {identity, `base_seed` = seed, `seed_scheme`,
  `next_sample_index: 0`}; clear all per-episode state on every reset even when the seed repeats; a reset can recur on
  the same connection; the seed controls sampling only (not the scene case). `sample_seed`: first 8 bytes of
  SHA-256(b"tower-seed-stream-v1\0" + seed as 4 big-endian bytes + sample_index as 8 big-endian bytes), big-endian.
- **metadata** within 10 s of accepting: protocol, `type: metadata`, `ready: true`, `policy_id` (1-128 of
  [A-Za-z0-9_.-]), `artifact_identity: sha256:<64 hex>`, plugin, plugin_version, seed_scheme, `seed_mode`
  (`deterministic` for us), `supports_seed_reset: true`, `observation_spec(track)`, `action_spec(track)` (kit helpers);
  optional `reconnect` {protocol tower-reconnect-v1, idempotent_requests, retention_s >= 60 + inference timeout}.
- **Encoding**: binary msgpack frames (<= 64 MiB, websocket compression off); NumPy arrays as msgpack ExtType 42 with
  payload packb([dtype string, shape list, contiguous bytes]); wire dtypes |u1, <f4, <f8.
- **Reference server** (kit `server.py`): `websockets.asyncio.server.serve(max_size=64 MiB, compression=None,
  ping_timeout=None)`; `handler_factory(track, policy_id, artifact_identity, predict=hold_policy)`; sessions by
  `session_id` with an asyncio lock, kept across reconnects (retention max(300, inference timeout + 60) s, <= 64);
  headers X-Tower-Task-Variant / -Track-Version / -Evaluation-Version bound per session; request dedupe by SHA-256 of
  the raw request (`request_payload_mismatch`); `reset` needs 0 <= seed < 2^32, a repeated `trial_id` is rejected;
  `infer` needs a prior reset (`reset_required`) and a non-empty instruction and runs `predict` in `asyncio.to_thread`;
  errors {protocol, `type: error`, request_id, error {code, message}} then close 1008. msgpack 1.1.2 and websockets
  15.0.1 are installed in the venv. Whether `predict` receives the session (for one `StudentAdapter` per session) is
  still to be read from the kit.

### Pre-flight of the unattended DAgger rounds (2026-10-22 03:45)

`rehearse.py --student bc_r0_early/switcher.pt --collect ... --beta 0.5 --inject 0.1 --est_noise` from the `round0`
snapshot (Task 2-2 D1, 4 episodes, seed 999000; the exact path `cpu_queue_dagger.sh` runs): no errors, 3 / 4 episodes,
shard written with the decision sources (switcher / oracle / inject) and beta recorded. The round-1 / round-2 queues
should therefore run through.

### Task 5-3 camera-loop failures, first look (2026-10-22 03:46; for Step 10)

From `closed_loop/teacher_kp_v2/task5-3_tower_insertion_3/*/perception_log.pkl` (the push-in's target block):
- **The inserted block's estimate drifts as it disappears into the tower** at the end of the push-in, in most
  episodes: the error grows over ~5 calls to 6-9 mm, almost all in z (+6 to +9 mm), with 5-7 deg rotation error and
  confidence 0.6 (e.g. l1_insert at t = 78 s in 7100001 / 7300000 / 7300001 D0, 7000001 / 7100000 D1; l2_insert at
  t = 36.5 s, smaller). In one episode (7000000 D0, the cell's only failure) the error is along the slot axis (x up to
  -18 mm, p50 37 mm), which is what the planned push-in reads as progress. The push-in verdict (perceived yaw <= 5 deg,
  xy <= 4 mm) also fails on these errors (D0 yaw p90 28 deg); its `on_fail: skip` keeps that harmless for the
  executor, but a switcher that learned to re-push after a failed verdict could act on it.
- D1 (3 / 8): failures are i0_push_in "structure" (4, high friction) and the next i1_insert "structure" (4): the same
  modes as the privileged teacher on simulator poses (G3a's oracle_graph run, 5-3 D1: i0_push_in structure 5,
  i1_insert structure 4), so mostly physics (the push in high friction drags the side blocks), not perception.
- Candidate fix for Step 10 (not made; it changes node behaviour): during the push, track the pushed block's along
  position kinematically from the pusher (pinch along + 86 mm contact offset) when the estimate is low-confidence or
  disagrees, and judge the push-in on that track.
- (03:49) Two more guards in the distilled nodes, the planned nodes' implicit bounds, for camera-loop pose drift: the
  push-in target never goes more than 6 mm past "block centred" along the slot axis (the planned push's target stops
  at the contact point + at most 6 mm of nudges), and the held insert gets no forward motion once the perceived block
  is 3 mm past the entry depth. The replay test still reproduces the planned run (the guards never fire there); the
  `distill_v1` snapshot got the updated `nodes.py` (its collector part is unchanged, so the running collection is not
  affected; the gate step will use the guards).

### Opt-in `pushin_kin` for the camera loop's push-in (2026-10-22 03:52)

From the Task 5-3 camera-loop analysis above. `PushInNode` (planned, learned and distilled paths) with the graph (or
node) flag `"pushin_kin": true`: the block's along-axis position is max(perceived, measured pinch along + 86 mm), a
lower bound because the pushed block is never behind the pusher's face; it feeds the speed schedule, the gap-crossing
window, the nudges and the stop, and the distilled push-in's ends. The verdict uses max(perceived, the contact bound
at the end of the push) and, once that says the block is inside the slot (> -50 mm), judges the depth only
(|along| <= `tol_pos_m`): the walls (1 mm clearance) fix the lateral offset and yaw there, and those are what the
drifting estimate gets wrong (yaw p90 28 deg at 5-3 D0). Off by default (fake-world parity re-run). Test
(`test_pushin_kin_ignores_a_drifting_estimate`, fake world, scripted pusher): the inserted block's estimate drifts
-18 mm along the axis and +7 deg in yaw at confidence 0.6 once the push starts; with the flag the push-in ends ok
(|along| <= 4 mm), without it the verdict fails. To be checked on simulator poses (no change expected) and in the
Step 10 camera loop.

### G6 preview on partial round-0 data (2026-10-22 04:12)

`summarize_shards.py` on the first 1,956 episodes: 33,789 decisions (on track for ~52k at 3,000: gate >= 40k),
340,519 progress samples (gate >= 300k), non-nominal 83 % overall (transfer 81 %, pushdown 88 %, insertion 83 %;
gate >= 10 % per family). Labels: acquire 8,476, idle 8,668, park 7,708, deliver 4,715, push_off 3,369, insert 508,
push_home 345; sources: oracle 25,376, explore 5,987, inject 2,426; no masked decisions. Episode TSRs are low by
design (injection 0.15 per option, exploration 0.1 per decision, estimator noise): e.g. 2-2 D1 0.63, 4-3 D1 0.14.

### Step 9c: early training on partial DART data (2026-10-22 04:22)

`train_distill.py` on the first 1,347 segments (82,563 rows; left arm: 5-2 D0 complete, 5-3 partial), 40 epochs, best
validation epoch kept:
| skill (left) | val rows | val loss | RMS motion error per row | gripper | switch row vs planned (held-out segments) |
|---|---|---|---|---|---|
| grasp_t5 | 2,003 | 0.023 | 0.65 mm, 0.10 deg | acc 0.952 | close: 41 / 47 within +-1 row, 6 early |
| insert | 4,223 | 0.0085 | 0.30 mm, 0.13 deg | acc 0.992 | release: 30 / 39 within +-1, 9 early (median -1) |
| push_in | 1,766 | 0.00013 | 0.19 mm, 0.004 deg | 1.000 | - |
Grasp and insert overfit after a few epochs (train 0.004, val rising to 0.077 / 0.015 by epoch 39; the best epoch is
kept). Input-noise augmentation (`--obs_noise` 0.1 / 0.2 std, weight decay 1e-3) did not lower the best validation
loss (grasp 0.024 / 0.023, insert 0.0091 / 0.012) and made the release earlier: defaults unchanged (no noise, wd 1e-4);
the full data set (3x) is the remedy. Guard added for the early closes: the distilled grasp's network may close only
within 5 mm of the grasp target (the planned grasp closes within 4 mm). `distill_v1` snapshot re-synced: `nodes.py`,
`train_distill.py`, and `executor.py` (the distilled push-in calls `PushInNode.kin_along`; with the flag off the
planned nodes are unchanged, so the running collection is unaffected).

### GPU queue reordered (2026-10-22 04:40)

The camera loop runs ~20 min per batch of 2 episodes under the night's CPU load (Task 4-3 D0: 4 batches 279 /
1,091 / 1,761 / 1,321 s; failed Task 4 episodes run the full 210 s), so the six remaining teacher baselines would have
held Step 9a / 9b until the evening. New order: the running Task 4-3 camera run (PID 1932013) -> 9a
(`gpu_queue_push9a_v2.sh`: same as `gpu_queue_push9a.sh`, waiting for that PID; the noise refit then sees the kp_v2
logs of 2-3, 1-3, 5-3, 4-3 and falls back to kp_v1 for the rest) -> 9b (`gpu_queue_grip9b.sh`, unchanged) -> the
remaining baselines (`gpu_queue_camera_teacher_rest.sh`: 4-2, 5-2, 3-3 re-run, 2-2, 1-2, 3-2). The old queue shells
(camera, its re-run, the 9a wait) were stopped with `kill -9` by PID; the crashed 01:10 Task 3-3 log is kept as
`camera_teacher_kp_v2_task3-3_vertical_stacking_3_crash0110.log`. Camera loop so far (teacher_t45_final + kp_v2, seed
7000, 8 per tier): 2-3 D0 8 / 8, D1 8 / 8; 1-3 8 / 8, 7 / 8; 5-3 7 / 8, 3 / 8; 4-3 D0 4 / 8 (D1 running). G5 (after the
refit) will compare against the variants available then and is re-run once all baselines exist.

### G6: PASSED (2026-10-22 05:43)

Round 0 complete (`results/student/data/r0`; D1 14,700 s, D0 10,560 s wall): **3,000 episodes, 50,457 decisions
(gate >= 40k), 516,380 progress samples (>= 300k), non-nominal 83 %: transfer 81 % (28,727 decisions), pushdown 88 %
(13,343), insertion 83 % (8,387) (>= 10 % each)**; no masked decisions. Labels: idle 13,838, park 11,637, acquire
11,538, deliver 6,229, push_off 5,695, insert 917, push_home 603; sources: oracle 37,664, explore 9,286, inject 3,507.
Episode TSR (perturbed collection, not a policy measure): Tasks 1-3 0.17-0.71, 4-2 0.33 / 0.38, 4-3 0.22 / 0.16,
5-2 0.68 / 0.56, 5-3 0.40 / 0.33 (D0 / D1). `results/student/data/r0_summary.json`. Next (queued): bc_r0, G7a, then
G7b in parallel with DAgger round 1.

### Step 9c: full collection and training (2026-10-22 06:00)

- Collection `distill/data/c1` (8,240 s): 1,200 episodes (300 per Task 5 cell), 935 task successes under DART (D0
  ~0.93, D1 ~0.49-0.69 per chunk), **328,113 rows**. Segments: grasp left 1,242 (1,223 ok, 11 empty gripper, 8
  timeouts), back 600 (all ok); insert left 1,191 (1,031 ok, 122 structure, 38 timeouts), back 600 (599 ok); push-in
  left 1,031 (930 ok), back 599 (521 ok, 78 structure).
- Training `distill/models/v1` (60 epochs, best validation epoch, all segments), held-out episodes:
| network | val rows | val loss | RMS per row | gripper acc | switch row vs planned |
|---|---|---|---|---|---|
| grasp_t5 left | 5,179 | 0.0052 | 0.20 mm, 0.025 deg | 0.986 | close 123 / 126 within +-1 row, 3 early |
| grasp_t5 back | 2,288 | 0.0047 | 0.19 mm, 0.014 deg | 0.990 | close 54 / 57, 3 early |
| insert left | 12,058 | 0.0024 | 0.22 mm, 0.038 deg | 0.997 | release 94 / 102, 6 early, 2 never |
| insert back | 5,190 | 0.0032 | 0.12 mm, 0.007 deg | 0.994 | release 42 / 57, 13 early, 2 never |
| push_in left | 4,831 | 0.0001 | 0.17 mm, 0.002 deg | 1.000 | - |
| push_in back | 2,788 | 0.0001 | 0.20 mm, 0.003 deg | 1.000 | - |
  (The guards cover the misses: no close beyond 5 mm of the grasp target, no release beyond 1 cm short of the entry
  depth, release after 1 s at depth.) Bundle `results/bundles/student_9c_v1`; gate runs started.

### Step 9c: a train / inference mismatch found and fixed (2026-10-22 06:12)

The first gate batches of the distilled skills failed every Task 5 episode (0 / 8 per variant at D0; a 2-episode
diagnostic: distilled grasps timing out 3 / 5, the two that lifted held the block 33-51 deg off the canonical grasp).
Cause: the distilled nodes fed the network's raw gripper **logit** (e.g. +-5) back as the "previous action" input,
while the training labels (and so the training inputs) carry the executed gripper bit 0 / 1: from the second row on
the networks saw out-of-distribution inputs. The replay test had missed it because its replay "network" ignores its
input. Fix: the nodes feed back the executed action (motion as applied, gripper 0 / 1). The replay test now also
checks that the observations the distilled nodes produce equal the recorded training observations (all 68 features
agree to ~1e-11 with the fix; without it the test fails on feature 67, the previous gripper bit). The invalid gate run
was stopped (kept as `distill/gate/distilled_v1_prevbug`); the distilled gate re-runs as `distill/gate/distilled_v1b`
(same bundle and seeds; the planned run `distill/gate/planned` continues).

### Step 9c: the networks copied their previous action; v2 without it (2026-10-22 06:30)

With the input fix, the v1 networks grasped (GSR 1.0) but the distilled inserts still failed (2-episode diagnostic:
a timeout and a structure failure). Per-row traces (`TOWER_DISTILL_TRACE=<file>`: one JSON line per distilled row,
off by default) showed classic behaviour-cloning drift: in one episode the insert stopped advancing at the tower face
with a 1.1-1.5 mm height error it never corrected (the planned insert corrects 0.6 x the error per row and holds the
depth until aligned); in the other it output a near-constant rotation (0.27 of the limit per row for 100+ rows) while
the lateral error grew to 13 mm. Offline ablation on the held-out rows: zeroing the previous-action input raised the
validation MSE 200x for insert left (0.00096 -> 0.196), 270x for insert back, 12x for the grasp: the networks were
mostly continuing their last action (copycat), which works on smooth demonstrations and drifts in closed loop.
`DistillNet` got a feature `mask` (inside the TorchScript; the 68-D interface is unchanged) and `train_distill.py`
masks the previous action by default (`--use_prev_a` restores it). v2 (`distill/models/v2`): validation about as
before (insert left 0.27 mm / row RMS vs 0.22; grasp 0.20; push-in 0.18-0.21; release 95 / 102 and 43 / 57 within
+-1 row). **Closed loop (the same 2 diagnostic episodes, Task 5-2 D0): 2 / 2**, distilled grasp (0.9 deg), insert
(-90.5 mm, lateral 0.1 mm, 9.4 s like the planned 9.7 s) and push-in (1.7 mm) all ok. Gate re-run with v2:
`distill/gate/distilled_v2` (bundle `results/bundles/student_9c_v2`, eval seeds 50000, N = 24); the v1b run was
stopped (kept as `distilled_v1b_copycat`).

### G3b follow-up: Task 1-3 D1 settled (2026-10-22 06:35)

`parity/g3b_ext` (48 more episodes per decider, seeds 5600-6700, same snapshot and settings as G3b v2): graph mode
47 / 48, state oracle 47 / 48, **node events identical in 48 / 48**. With G3b's 24 (23 vs 21) the cell is 70 / 72 vs
68 / 72: the -2 / 24 was noise; G3b is closed.

### `pushin_kin` on simulator poses (2026-10-22 06:35)

The planned teacher with `--graph_set '{"pushin_kin": true}'` vs without (the 9c gate's planned run; same seeds
50000, N = 24, simulator poses; `pushin_kin/sim_vs_planned.md`): D0 24 / 24 both variants either way; **D1 5-2
14 -> 17 / 24, 5-3 16 -> 18 / 24 (78 -> 83 / 96)**. All five flips are failures avoided (3 push-in "structure" in
high friction / slippery, 2 later insert "structure"); none the other way. Mechanism: the contact bound reads the
block ~1.4 mm further in than its pose, so the push stops earlier: successful push-ins end at median 2.6 mm (p90
2.7-3.1 mm) from the slot centre instead of 1.2-1.4 mm (p90 1.3-2.1 mm), still within the node's 4 mm verdict; the
last millimetre of pushing is what drags the structure in D1. Risk to keep in mind: an evaluator slot tolerance below
~3 mm (unknown; organizer question 4) would turn these into misses. Candidate for the student bundle; decided with the
Step 10 camera loop.

### G9c: PASSED (2026-10-22 06:58)

Distilled v2 (`results/bundles/student_9c_v2`, graph flag `"distilled": true`) vs the planned controllers
(`teacher_t45_final`), executor, simulator poses, eval seeds 50000, N = 24 (`distill/gate/distilled_v2.md`):
| cell | planned | distilled | gate |
|---|---|---|---|
| 5-2 D0 | 24 / 24 | 24 / 24 | >= 23 ok |
| 5-2 D1 | 14 / 24 | **17 / 24** | >= 12 ok |
| 5-3 D0 | 24 / 24 | 23 / 24 | >= 23 ok |
| 5-3 D1 | 16 / 24 | **19 / 24** | >= 14 ok |
| all | 78 / 96 | **83 / 96** | |
Failure modes are the planned ones (D1: insert / push-in "structure", held block dropped in the pre-insert transit,
insert timeouts), fewer structure ends; distilled push-ins stop slightly earlier (place error median 1.55 vs 1.25 mm).
The 5-3 D0 miss: all inserts placed (PSR 1.0), a disturbance. All six Task 5 controllers (grasp, insert, push-in x
two arms) are now networks for the student; the planned controllers stay as the fallback (flag off).

### Step 9a restarted on the refit noise model (2026-10-22 07:00)

- Task 4-3 camera-loop baseline done: **D0 4 / 8, D1 2 / 8** (teacher_t45_final + kp_v2; ODR 0.09 / 0.30).
- The 9a queue started at ~06:05 but its noise refit crashed (`fit_estimator_noise.py`: `Path(out_bank).relative_to`
  on a relative path; fixed with `.resolve()`), so the left push trained 200 iterations on the 00:34 v1 fit. The refit
  (07:00, `configs/student/noise.yaml` + `results/student/noise/traces_v2.npz`; v1 kept as
  `results/student/noise/noise_v1_0034.yaml`) adds the kp_v2 Task 4-3 camera logs and changes the push target a lot:
  **5,129 calls, lost (confidence 0) in 23.6 % (v1: 1,086 calls, 4.2 %)**, confidence 0.3 in 7.6 %; P(lost | tilt,
  along) 0.004-0.08 below the first tilt bin and 0.35-0.99 above it; 148 lost runs (median 2 calls). The "moving" class
  now has 823 calls (v1: 50).
- 9a stopped (kill -9 by PID) and restarted as `gpu_queue_push9a_v3.sh` (same log name, so the 9b queue still waits on
  it): left continues from the first run's `model_500.pt` for 150 iterations under v2 (`--tag 9a_est_v2`); back 250
  iterations from the widened v1c; then the gate evaluations under v2 (`results/student/push9a/eval_<arm>_D1_est_v2.json`).
- The refit triggered G5 (`cpu_queue_g5_after_refit.sh`: the teacher with the v2 noise on the camera loop's episodes,
  seed 7000), compared against the camera baselines that exist now (2-3, 1-3, 5-3, 4-3); to be re-run when the
  remaining baselines are in. The switcher rounds keep the v1 noise (their snapshot), so its data stay consistent.

### G7b, round 0: not passed; the cause and an opt-in rule (2026-10-22 07:15)

`bc_r0` (best epoch 17: held-out decisions 0.963, nominal 1.000, non-nominal 0.954, progress AUC 0.993; **G7a**: nominal
and AUC pass, non-nominal below 0.98) in closed loop, `--est_noise` (v1), eval seeds 50000, N = 24
(`eval/g7b_bc_r0.md`): **393 / 480 vs the state oracle's 405 / 480 (-2.5 %; gate -1 %)**. Cells: Tasks 3 and 5 equal
(Task 3 and 5-2 D0 with identical events), Task 4 within 2 (4-3 D1 8 vs 10), 2-2 -1 / -1, **1-3 D0 18 vs 24, D1 20 vs
23** (> 2: fail). In 7 of the 9 lost 1-3 episodes the switcher acquired goal 0 again right after its deliver ended ok
(place yaw ~4.3 deg, near the 5 deg tolerance) and the re-place landed 10-14 mm off; the other 2 were place verdicts
at 5.1-5.2 deg. The state oracle re-acquires a just-delivered goal only when truth says it is outside tolerance, so
near the yaw boundary the labels are not inferable from the (noisy) tokens: the switcher learned to redo with some
probability. DAgger (round 1 running) labels the switcher's own redo states and should move it to the majority label
("continue"); as insurance, opt-in graph flag **`keep_delivered`**: a goal whose deliver ended ok is not acquired again
while its block's rest estimate stays within 5 mm of where it was released (masked in `action_masks` and
`StudentRun.valid`; `test_keep_delivered_blocks_a_redo_until_the_block_moves`). Retrying a just-placed near miss
succeeded in only 134 / 821 past cases, so the rule gives up little. Check running: `eval/bc_r0_keep` (Task 1-3, 2-2).
- (07:20) Distilled v2 + `pushin_kin` on simulator poses (`distill/gate/distilled_v2_vs_kin.md`): 81 / 96 vs 83 / 96
  for distilled v2 alone (5-2 D1 14 vs 17, 5-3 D0 24 vs 23; within noise). The distilled push-in already stops a bit
  early (median 1.55 mm), so the flag's simulator gain for the planned push-in does not add up; its value is the
  camera-loop drift, to be checked in Step 10.
- (07:40) `eval/bc_r0_keep` (bc_r0 + `keep_delivered`, same seeds and v1 noise as G7b): **Task 1-3 D0 24 / 24, D1 23 / 24**
  (bc_r0 18 / 20, state oracle 24 / 23); Task 2-2 D0 23 / 24, D1 23 / 24 (unchanged; oracle 24 / 24). The rule only
  touches the transfer family (deliver), so Tasks 4 / 5 keep bc_r0's G7b numbers; the other four transfer variants
  are running with the rule into the same directory for a full G7b-with-rule table.

### Step 9a, left arm: gate passed (2026-10-22 07:50)

`results/skills/push_left/2026-10-22_0700_aa4a5bb909_9a_est_v2` (67-D observation with `seen` / `since_seen`; 200
iterations under the v1 noise fit, then 150 under v2; training success ~0.935 with the target unseen in ~10 % of
steps): **D1, 500 episodes, v2 estimator noise: 494 / 500 = 98.8 % (Wilson 97.4-99.5 %; gate >= 95 %)**, disturbance
0; by case: combined 45 / 47, high friction 39 / 40, irregular layers 54 / 56, shifted assembly 59 / 60, the rest 100 %
(`results/student/push9a/eval_left_D1_est_v2.json`). Back arm training now (250 iterations from the widened v1c).
The plan's second criterion (CPU Task 4-2 / 4-3 rehearsal with `--est_noise`, >= the privileged teacher - 2 / 24) needs
a bundle with both new pushes: after the back arm.

### G7b with `keep_delivered`: passed (2026-10-22 07:51)

bc_r0 + `keep_delivered` (all six transfer variants re-run with the rule; Tasks 4 / 5 have no deliver option, their
cells are bc_r0's) vs the state oracle, eval seeds 50000, N = 24, v1 noise (`eval/g7b_bc_r0_keep.md`, the combined
root `eval/bc_r0_keep_full` links the cells): **402 / 480 vs 405 / 480 (-0.6 %; gate -1 %)**; worst cells 4-3 D1 8 vs
10 (-2), 2-2 D0 / D1 23 vs 24; Task 1-3 24 / 24 and 23 / 24 (= oracle); Tasks 1-2, 2-3, 3-x 24 / 24. **G7b passes with
the rule** (without it: 393 / 480). The rule goes into the student bundle unless DAgger makes it unnecessary; G8 is
evaluated both ways: `cpu_queue_after_r1.sh` (no rule, as planned) and `cpu_queue_g8_keep.sh` (bc_r2 + rule, eval
seeds 50000 and the injection set 60000; queued after it).

### G5 with the v2 refit: not passed, too harsh (2026-10-22 07:52)

The teacher with the v2 noise model on the camera loop's episodes (seed 7000, 8 per cell; the four variants with
camera baselines so far; `noise/g5_v2.md`, `noise/g5_v2_calibration.md`): **27 / 64 vs the camera loop's 47 / 64**,
outside the camera Wilson intervals in several cells (5-3 D0 2 / 8 vs 7 / 8, 4-3 D0 0 / 8 vs 4 / 8, 1-3 D1 4 / 8 vs
7 / 8). Per class: foundation, held and grasp targets within 25 %; static p90 2.1 mm vs 164 mm and moving / push
targets differ (the camera's tails come from collapsed towers); the v2 bank replays large-error traces (static p99
~400 mm) into intact episodes, which the camera loop shows only after collapses. The v1 fit (00:34, the switcher
rounds' model) matches the camera TSRs much better (e.g. state oracle with v1: 1-3 / 2-3 24 / 24 and 23-24 / 24, 5-3
22 / 24 D0, 4-3 6 / 24 D0 vs camera 4 / 8). Consequences: the 9a CPU criterion now runs under v1
(`cpu_queue_push9a_check_v1.sh`; the v2 version was stopped before it started); the left push trained and passed its
gate under the harsher v2, which only makes it more conservative. To do (Step 5 v3): fit the static / moving classes
only from intact-tower calls, or replay large-error traces only after a structure change.

### State at 08:00 (2026-10-22) and what runs next

Tests: 277 passed, 3 skipped. Nothing committed (the user commits); new since f5e8cce: `tower_rl/student/` (incl.
`distill/`), `configs/student/`, 11 scripts, 9 test files, opt-in hooks in `executor.py` (`node_hook`, `distilled`,
`pushin_kin`), `base_env.py` (9b grip), `rehearse.py` (`--distill_collect`, `--dart_scale`, ...).
Queues (logs in `results/student/logs/`):
- CPU: DAgger r1 (D1 done, D0 running; `round0` snapshot, v1 noise) -> `cpu_queue_after_r1.sh` (bc_r1 -> r2 ->
  bc_r2 -> G8 runs without the rule) -> `cpu_queue_g8_keep.sh` (bc_r2 + `keep_delivered`: seeds 50000 and the
  injection set 60000); `cpu_queue_push9a_check_v1.sh` (after the 9a back eval: Task 4-2 / 4-3 with both new pushes vs
  the old pushes and the privileged teacher, v1 noise).
- GPU: 9a back (250 iterations) + eval -> `gpu_queue_grip9b.sh` (9b fine-tunes + skill evals at 3 / 22 N) ->
  `gpu_queue_camera_teacher_rest.sh` (camera baselines 4-2, 5-2, 3-3, 2-2, 1-2, 3-2) -> Step 10 (`gpu_queue_step10.sh`,
  to launch once bc_r2 / G8 and the skill gates fix the bundle: switcher + distilled v2 Task 5 + 9a pushes [+ 9b] +
  `keep_delivered` [+ `pushin_kin` if the camera loop agrees]).
HUMAN: (1) download the starter kit `tower-policy-example.zip` and the 10 Hz all-ten `track.json` into
`~/Documents/tower_rl/data/challenge/` (Step 11; the site resets TLS from this machine); (2) the organizer questions
(plan Appendix D), now with two concrete stakes: Q2 (3 N vs 22 N fingers; 9b) and Q4 (the Task 5 slot tolerance: the
distilled / `pushin_kin` push-ins stop 1.5-2.6 mm short of the slot centre, inside our 4 mm verdict); (3) commit.

### Results of the 08:00 queues (2026-10-22, read at 19:10)

All queues ended by 14:28 (CPU 13:09); nothing has run since (Step 10 waits for the bundle decision below). Killed an
orphaned `nvidia-smi -l 1` monitor of the killed 9a v2 run (pid 2040871, running since 06:04, writing
`logs/push9a_train_left.log.gpu.csv`).

**G9a.** GPU gates passed (D1, 500 episodes, v2 noise): left 98.8 % (`push_left/2026-10-22_0700_aa4a5bb909_9a_est_v2`),
back 95.4 % (`push_back/2026-10-22_0736_49b7430e5b_9a_est_v2`; gate 93 %). CPU Task 4 (v1 noise, seeds 50000, N = 24,
`results/student/push9a/cpu_*`):

| cell | privileged, simulator poses | old pushes, v1 noise | 9a pushes, v1 noise |
|---|---|---|---|
| 4-2 D0 | 24 | 12 | 20 |
| 4-2 D1 | 20 | 11 | 12 |
| 4-3 D0 | 21 | 10 | 11 |
| 4-3 D1 | 19 | 11 | 10 |
| all | 84 | 44 | 53 |

The task-level gate (privileged - 2 / 24 per cell) fails in every cell; the gain is in 4-2 D0 only. Task 4 stays the
weakest family (camera teacher 15 / 32 below); its failures under noise are not diagnosed yet.

**9b.** Fine-tunes of `grasp_lift` / `place_release`, both arms (150 iterations, closed width 0.0425, finger effort per
env from {3, 22} N). Skill evaluations, D1, 500 episodes, source -> fine-tuned, at 3 N / 22 N
(`results/student/grip9b/`):
- grasp_lift left 95.4 / 96.2 -> 97.2 / 97.4 %; back 96.0 / 98.2 -> 98.8 / 99.0 % (heavy blocks at 3 N: 73.5 -> 88.2 %
  left, 73.5 -> 100 % back);
- place_release left 92.8 / 93.6 -> 94.4 / 94.8 %; back 86.0 / 85.0 -> 84.2 / 85.6 % (no change; the back place is the
  weak skill at either effort).
G9b itself (the task-level grip matrix with the fine-tuned skills) has not been repeated yet.

**Step 8 / G8.** DAgger r1 (beta 0.5; 1,500 episodes, 16,870 decisions) and r2 (beta 0; 1,500 episodes, 16,946
decisions), injection 0.1. bc_r1 (r0 + r1; best epoch 19: nominal 99.94 %, non-nominal 94.4 %, AUC 0.995); bc_r2
(r0 + r1 + r2; best epoch 5: nominal 99.96 %, non-nominal 93.6 %, AUC 0.997). Closed loop, v1 noise
(`results/student/eval/`):
- Seeds 50000, N = 24: bc_r2 404 / 480 vs the state oracle 405; per cell 4-3 D0 +1, 4-3 D1 -1, 5-3 D1 -1, all others
  equal. No cell worse by more than 1 / 24; the aggregate is one episode short of ">= oracle".
- DAgger removed the Task 1-3 re-grasps: without `keep_delivered`, 1-3 is 24 / 24 and 23 / 24 (bc_r0: 18, 20). With the
  rule the total is also 404 (`g8_bc_r2_keep.md`).
- Injection 0.15, seeds 60000: TSR oracle 334, bc_r2 319, bc_r2 + rule 308 / 480. The eval runs do not log which
  episodes were injected; new `scripts/inject_recovery.py` replays the draws (`default_rng([seed, 991])`, once per
  goal option, from the option sequence in the policy log). Check: of 80 replayed deliver yaw injections (4.4-8.0 deg)
  in bc_r2's run, the place that followed logged >= 3 deg in 75 (median 5.7 deg; places without injection: median
  0.7 deg, 2.4 % >= 3 deg). Recovered (success among episodes with >= 1 injection): oracle 185 / 312 = 59.3 %, bc_r2
  172 / 310 = 55.5 % (-3.8 points; gate -5: pass), bc_r2 + rule 164 / 311 = 52.7 % (-6.6: fail).
- Verdict: bc_r2 without `keep_delivered` passes G8 within noise (1 / 480 short on the aggregate). The rule is dropped:
  no gain after DAgger, and worse recovery (it blocks re-doing a delivered block that a knock moved by less than its
  5 mm threshold).

**Step 0.4 complete** (the teacher on kp_v2 perception in the camera loop, seed 7000, 8 per tier): D0 72 / 80, D1 59 / 80,
all 131 / 160.

| | 1-2 | 1-3 | 2-2 | 2-3 | 3-2 | 3-3 | 4-2 | 4-3 | 5-2 | 5-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| D0 | 8 | 8 | 8 | 8 | 8 | 8 | 5 | 4 | 8 | 7 |
| D1 | 8 | 7 | 8 | 8 | 8 | 7 | 4 | 2 | 4 | 3 |

Next: freeze `student_v1` (bc_r2, distilled v2 Task 5, 9a pushes, kp_v2; 9b once the grip matrix passes) and run
Step 10 (`gpu_queue_step10.sh`); repeat the grip matrix with the 9b skills (CPU); diagnose Task 4 under noise with the
9a pushes; G5 with v1 noise against all ten camera baselines.

### `student_v1`, randomization flag, videos (2026-10-22 19:20-19:50)

- **Bundle `results/bundles/student_v1`** (`freeze_student_bundle.py`; tar sha256 `01ab88da17d7f900...`): teacher bundle
  `teacher_t45_push9a` (= `teacher_t45_final` with the two 9a pushes, 67-D; its MANIFEST was copied unchanged and still
  names the old push sources, the skills' own JSONs name the 9a checkpoints), the six distilled Task 5 networks
  (`distill/models/v2`), switcher `bc_r2`, kp_v2, graph flag `{"distilled": true}` (no `keep_delivered`, no
  `pushin_kin`; 9b skills not included until the grip matrix passes).
- Smoke tests: Task 5-2 D0 on simulator poses + v1 noise 2 / 2 with distilled grasp / insert / push-in (`distilled`
  marked in every Task 5 node); Task 4-2 D0 1 / 2 (a disturbance); camera loop Task 2-2 D1 slippery 1 / 1 with video.
  The full student (switcher + distilled + 9a) had not been run together before.
- **`rehearse.py --case <name> [...]`** (opt-in): forces the tier's robustness case(s), episode j of a tier gets the j-th
  (cycling); `run_batch(case=)` -> `DomainSampler.sample(case=)` (now also a per-env list); result dirs get
  `_case-<names>`. Default unchanged (case drawn from the seed).
- Videos: file names carry the case (`<variant>_<tier>_<case>_seed<seed>_<success|fail>.mp4`); with the student the
  status bar shows the current decision (`decision deliver goal 0 (left)`).
- **`scripts/try_student.py`**: front end for trying the student (or `--teacher`) on one task or all ten, any tier /
  case, camera loop with videos (default), simulator poses or the v1 noise model; `--live` opens the Isaac Sim window.
- `scripts/inject_recovery.py` (G8's recovery rate, above). Tests: 277 passed, 3 skipped.
- GPU (19:50): `results/student/gpu_queue_videos_v1.sh` (snapshot `student_v1_videos`): student_v1 in the camera loop,
  all ten tasks x 5 episodes (D1: nominal + four robustness cases rotated over the tasks), seed 7000, 1 env ->
  `results/student/videos/student_v1/*.mp4`, results `results/student/closed_loop/student_v1_videos/`.

### Results of the `student_v1` video run (camera loop, finished 2026-10-22 23:00)

50 episodes (10 tasks x 5, D1: nominal + four forced robustness cases, seed 7000, 1 env), 3 h 10 min.
**40 / 50** (Wilson 0.67-0.89): Tasks 1-3 **30 / 30**; 4-2 2 / 5, 4-3 2 / 5, 5-2 4 / 5, 5-3 2 / 5. By case: nominal 9 / 10,
lightweight 4 / 5, heavy 3 / 5, uneven_weights 5 / 5, slippery 3 / 5, high_friction 3 / 4, shifted_assembly 4 / 4,
irregular_layers 3 / 4, lighting 4 / 4, combined 2 / 4. Not paired with the teacher baseline (other seeds / cases);
G10 does that. No policy errors; after every failure the switcher did what the graph / oracle does (close the goal,
park, next goal or idle). All 15 Task 5 grasps (distilled) ok, rotation error <= 0.7 deg: no sign of review issue 2.

The 10 failures (from `episodes.json` + the perception logs' true poses):
- **4-3 heavy, 4-3 slippery: a pushed block went off the table edge** (heavy: the last push knocked the earlier-pushed
  l3_0 off at 60.6 s; slippery: the last target l1_0 slid off at 57.6 s). Every push had succeeded in order (heavy 9 / 9
  P, slippery 8 / 9: the ninth block never came to rest), structure intact. The scene has no floor (the station usdz
  has no floor / ground collider either, checked with pxr), so the block falls forever (z -110 km at 210 s), the
  PushTracker's "all movable blocks static" never holds and the episode runs to 210 s. With a floor both would likely
  have succeeded. The camera-loop teacher had the same in 2 of 16 D1 Task 4 episodes (4-2 combined, 4-3 combined).
  Unknown whether the official scene has a floor / whether a block off the table fails Task 4: organizer question.
- 4-2 nominal (all six pushes ok, push 5 dragged block 6 > 10 mm: structure broken; collapse-only would pass),
  4-2 irregular_layers (pushes 4 / 5 departed out of order), 4-2 combined (push 0 "structure" x3, then out of order),
  4-3 high_friction (12 node failures, drags 12-147 mm): the known neighbour-drag failure mode.
- 5-2 combined, 5-3 slippery: push-in "structure" (10.6 / 16.8 mm, roof / side block moved); no retry (graph and
  oracle close the goal); 5-2's block stays 34 mm out. 5-3 lightweight: first insert "structure" (10.5 mm).
- 5-3 heavy: every node ok except the last push-in's own verdict; the lower-slot block ended **6.0 mm** from its goal
  (tolerance 5 mm).
- **Distilled push-in stops short on the 5-3 lower slot.** True final xy error where the block went in: student 2.4,
  2.4, 3.8, 3.8, 6.0 mm; camera-loop teacher (planned push-in) D0 1.8-2.1 in 6 of 8 (one 4.0, one failed 28 mm
  out), D1 0.8-1.9 where it went in. 5-2: student 0.9-4.7,
  teacher 0.7-3.2 (D0). Likely cause: the push-in ends when the *perceived* along > -1.5 mm, and the hidden block's
  estimate runs ahead (final estimation error 13-34 mm, confidence 0-0.6 in 4 of 5 5-3 episodes). `pushin_kin` does
  not cover an estimate that is ahead (it takes max(perceived, kinematic)). Candidate: end the push-in on the pusher
  kinematics (block along >= pinch along + 86 mm), keeping the 6 mm cap.
- The push-in's own verdict is unreliable in the camera loop: 6 of the 12 push-ins without a structure failure reported
  "failed" (estimated place error 4.4-21 mm) although the tracker scored the block placed. Harmless now (nothing
  retries a push-in), but any retry rule needs a better verdict first (e.g. the kinematic depth).

## 2026-10-23 (session 6): student, Step 10

### Step 10 / G10 launched (2026-10-23 08:34)

- **What runs.** `results/student/gpu_queue_step10.sh ~/Documents/tower_rl_snapshots/student_v1_step10
  results/bundles/student_v1 student_v1_kp_v2 "" --video results/student/videos/step10_student_v1` (log
  `results/student/logs/gpu_queue_step10_student_v1.log`, one log per variant `step10_student_v1_kp_v2_<variant>.log`):
  `student_v1` in the kp_v2 camera loop, all ten variants, D0 + D1, 8 episodes per tier, seed 7000, `--num_envs 2`.
  Results go to `results/student/closed_loop/student_v1_kp_v2/`; the per-cell comparison with the teacher baseline to
  `results/student/closed_loop/student_v1_kp_v2_vs_teacher.md`.
- **Inputs checked.** Bundle `student_v1` unchanged (tar sha256 `01ab88da17d7f900...`; the directory equals the tar,
  only mtimes differ). Snapshot `student_v1_step10` (08:33) = the current code, identical to `student_v1_videos` except
  `scripts/try_student.py`. Tests: 277 passed, 3 skipped.
- **Paired with the teacher baseline.** Same arguments and seeds as Step 0.4 (`closed_loop/teacher_kp_v2`, snapshot
  `camera_teacher_kp_v2`). Without a forced case `DomainSampler.sample` / `run_batch` draw exactly as in that snapshot,
  so each episode seed gets the same scene, robustness case and dynamics; `compare_runs.py` pairs episodes by seed.
- **Deviation from the plan's Step 10.3** ("re-run the failed episodes with `--video`"): every episode is recorded
  during the run. The video hook (`CameraPerception._video_frame`) only reads the delivered images and the estimates (no
  RNG, no estimator state), so it cannot change the run, while a re-run would not reproduce an episode exactly (CPU
  PhysX history; GPU rendering and network in the loop). Cost: ~3 MB per episode (~0.5 GB; 27 GB free on the disk).
- **Expected duration.** ~11 h: the teacher baseline's ten variants took 11.3 h at 2 envs (Task 4 4.8 h, Task 5 3.2 h,
  Tasks 1-3 2.3 h). Order: 4-3, 5-3, 4-2, 5-2, then Tasks 1-3.
- **Queued behind it** (waits for the main queue's `QUEUE_DONE`): `results/student/gpu_queue_step10_pushin_kin.sh` (log
  `gpu_queue_step10_pushin_kin.log`), the Task 5 A/B of progress report 1 (next steps, item 2): `--graph_set
  '{"pushin_kin": true}'`, 5-3 and 5-2, same seeds, no video; table
  `closed_loop/student_v1_kp_v2_pushin_kin_vs_student_v1_kp_v2.md`. ~3 h. `pushin_kin` takes max(perceived, kinematic)
  depth, so it guards only against a lagging
  estimate, not against the video run's short 5-3 push-ins (estimate ahead of the block, report issue 10).
- **Gate G10:** per variant and tier >= the teacher's camera loop - 1 / 8; aggregate >= 131 / 160; Task 4 D0 >= 7 / 8 and
  D1 >= 5 / 8; perception errors in line with `Pose-estimator-v1.MD` section 8.3; failures diagnosed.

### Results of Step 10 / G10 and the `pushin_kin` A/B (queues done 2026-10-23 17:00 / 20:19; read 2026-10-24 05:45)

**Runs.** Main queue 08:34-17:00 (8 h 26 min; the teacher baseline took 11.3 h), A/B 17:00-20:19. 160 + 32 episodes,
policy errors 0 in all 24 cells, no Traceback / `[Error]` in any log. 160 videos (25 failures, 563 MB) in
`results/student/videos/step10_student_v1/`. Table: `closed_loop/student_v1_kp_v2_vs_teacher.md`.

| variant | D0 student / teacher | D1 student / teacher |
|---|---|---|
| 1-2 | 8 / 8 | 8 / 8 |
| 1-3 | 8 / 8 | 8 / 7 |
| 2-2 | 8 / 8 | 8 / 8 |
| 2-3 | 8 / 8 | 8 / 8 |
| 3-2 | 8 / 8 | 8 / 8 |
| 3-3 | 8 / 8 | 8 / 7 |
| 4-2 | 6 / 5 | 3 / 4 |
| 4-3 | 5 / 4 | 4 / 2 |
| 5-2 | 7 / 8 | 5 / 4 |
| 5-3 | 6 / 7 | 3 / 3 |

**135 / 160 vs 131 / 160** (Wilson 0.78-0.89 vs 0.75-0.87). Tasks 1-3 96 / 96 (teacher 94), Task 4 18 / 32 (15), Task 5
21 / 32 (22). Paired by seed: both 121, student only 14, teacher only 10, neither 15. Task 4 outcomes are nearly
independent between the two runs (both 8, student only 10, teacher only 7, neither 7): small differences decide a
Task 4 episode.

**G10: not passed, only on the Task 4 bar.**
- Per variant and tier >= teacher - 1 / 8: passed in all 20 cells (at the limit: 4-2 D1 3 vs 4, 5-2 D0 7 vs 8, 5-3 D0
  6 vs 7).
- Aggregate >= teacher: passed (135 vs 131).
- Task 4 D0 >= 7 / 8, D1 >= 5 / 8 with the 9a push: failed (4-2 6 / 8 and 3 / 8, 4-3 5 / 8 and 4 / 8), although the student
  beats the teacher there (18 vs 15 / 32).
- Perception in line with `Pose-estimator-v1.MD` 8.3: yes. Confident estimates per cell: median 0.06-0.20 mm, p90
  0.21-1.08 mm (the teacher run's p90 reached 174 mm in 4-3 D1 and 67 mm in 5-3 D1). The large all-blocks p90 values
  (4-3 D0 19 mm, D1 170 mm) are blocks knocked off in failed Task 4 episodes, as in 8.3.
- Failures diagnosed: below.

**Task 4 failures (14; teacher 17).** Classified from `disturbed_ids`, the events and the perception logs' true poses.
Block names per the graphs: the top row is pushed first, then the rows below; in each row, `_2`, then `_1`, then `_0`.
- **7: one neighbour dragged by a push**, the other pushes fine ("structure broken"):
  - pushing bottom-row `l1_1` drags `l1_0`: 4 (4-2 D0 7100000, 4-2 D1 7000001, 4-3 D0 7000000, 4-3 D1 7000001; in 4-3 D1
    7000001 all 9 pushes counted);
  - pushing `l2_0` drags `l1_2` below it: 2 (4-2 D0 7200001, 4-3 D1 7200000);
  - pushing `l1_2` drags `l1_1`: 1 (4-2 D1 7200001).
  - The teacher had 5 of these.
- **5: an early push** (P0 / P1, top row) moved several blocks of the rows below, then pushes departed out of order (4-2 D1
  7100001 / 7200000, 4-3 D0 7100000, 4-3 D1 7000000 / 7300001). The teacher had 11 multi-block disturbances.
- **2 other:**
  - 4-3 D0 7200000: the transit to the last pre-push shoved the target `l1_0` 42 mm, then the push failed "shoved".
  - 4-2 D1 7000000 (high_friction): 5 / 6 pushes counted; structure broken at 17.3 s with no block attributed.
- Blocks off the table (no floor, report issue 12): in 3 student failures, always after the structure had broken. Not a
  primary cause in this run.

**Task 5 failures (11; teacher 10).** The final true distance of each inserted block from its goal
(`destination_base_1` frame; the success threshold is 5 mm: 4.7 passed, 5.1 failed), D0, where the block went in:
- student: 5-3 upper 1.3-1.9 mm, lower 0.8-2.7 mm; 5-2 0.9-3.2 mm;
- teacher: 5-3 upper 1.0-3.5 mm, lower 1.8-4.0 mm; 5-2 0.8-3.2 mm.

When perception holds, the distilled push-in is now as precise as the planned one. The video run's 2.4-6.0 mm
lower-slot shortfalls (report issue 10) recurred only mildly (5-2 7200000). All three D0 losses are push-ins ending on a
wrong perceived depth:
- **5-3 D0 7300000** (teacher passed):
  - The upper-slot block was ~8 mm from home and nearly hidden (confidence 0.6). Its estimate jumped to the *empty
    lower slot directly below*: same xy, 32 mm lower, 90 deg yaw, confidence 1.0. The push-in, which ends on the
    perceived depth, stopped 5.8 mm short.
  - During the second insert the two insert blocks' estimates swapped. When the lower block's estimate jumped to the
    upper block (whose xy is the goal's), its push-in ended and left the block 28.7 mm out.
  - The teacher had the same sequence on 7000000 (upper 3.8 mm, inside the tolerance; lower 28.3 mm out).
- **5-3 D0 7300001** (teacher passed): near the end of the push-in the lower block's estimate jumped to the upper slot's
  height, 34 mm ahead. The push stopped 10.3 mm short.
- **5-2 D0 7200000** (teacher passed): the push-in ended 5.1 mm from the goal (estimate ~1 mm ahead at the end).
- **5-2 D1 (3) and 5-3 D1 (5):** all on seeds the teacher also fails.
  - The insert or the push-in moved the structure > 10 mm (high_friction, combined, shifted_assembly).
  - In 5-3 shifted_assembly / combined the second grasp came out ~40 deg rotated (the teacher's 41-43 deg on the same
    seeds), and its insert failed "structure".

**`pushin_kin` A/B: worse, stays off.** `closed_loop/student_v1_kp_v2_pushin_kin`: 14 / 32 vs 21 / 32.
- Per cell: 5-2 D0 8 vs 7, D1 2 vs 5; 5-3 D0 3 vs 6, D1 1 vs 3.
- Paired: `pushin_kin` only 2, `student_v1` only 9 (exact McNemar p ~ 0.065).
- The final distances show why: 5-2 D0, every block 2.7-3.5 mm from its goal (`student_v1` 0.9-3.2 mm); 5-3 D0 lower
  slot, 6.4-10.5 mm in 5 of 8 (`student_v1` 0.8-2.7 mm outside the two perception failures).
- The push-in ends when max(perceived, kinematic) along > -1.5 mm, or when that value stalls for 15 rows, so the flag
  can only end it earlier.
- The 5-2 fused depth read -1.1 to -1.8 mm while the blocks were 2.7-3.5 mm short: one of the two readings is 1-2 mm
  optimistic.
- The issue 10 candidate stays: end on the kinematic reading *alone*, not the max, with its contact offset checked
  against simulator truth first. It would also cover both 5-3 D0 perception failures.

**Next candidates (none started):**
1. Push-in end on the kinematic depth alone (offset calibrated on truth). Test on CPU on simulator poses and v1 noise,
   then in the 5-2 / 5-3 camera loop.
2. Estimator: a block nearly hidden in a slot must not be re-acquired in the other, empty slot of the same column.
3. Task 4: the bottom-row drag (pushing `l1_1` drags `l1_0`) is the largest single Task 4 failure.

### Submission before fixes; the live docs' submission flow (2026-10-24 07:10)

The user decided to run a first official evaluation of `student_v1` before any fix (plan Step 11, G11b). Progress
report 1 updated: §6.7 (Step 10), §10 issues 13-14, §11 item 3 (the submission steps), Appendix A. The live docs were
re-read with WebFetch (Getting started, Video walkthrough, Endpoint qualification, Results and videos, Python reference
server, WebSocket connection, Actions, Observations, Task prompts):
- Flow: "Create an account" (team name, email, password >= 12 characters) -> "Connect your model": task selection and
  control rate -> "Download track.json" -> serve at a public `wss://` URL on port 443 (quick tunnel to 127.0.0.1:9001;
  the model URL has no port) -> "Model name" / "Model URL" -> "Save model" -> "Check endpoint" -> "Qualification:
  passed" -> "Queue evaluation" -> "Your runs -> Evaluations" -> "View result" / "View replays". No API key in this
  flow (the repo's `TOWER_API_KEY`, `check_policy.py` and `submission.json` belong to the old `tower-openpi-v1`).
  Results are reviewed by the organizers before publication; leaderboards are per release and per control rate.
- The kit's `predict(observation, *, prompt, seed)` does not receive the session, so `serve_student.py` keeps one
  `StudentAdapter` per `session_id` (a minimally forked handler), cleared on every reset.
- Up to six queued or running evaluation submissions per team, one running at a time; up to 8 concurrent episodes plus
  one check session. Qualification pins `policy_id` and `artifact_identity`. Out-of-bounds chunks fail validation (the
  evaluator does not clip). Python servers: `ping_timeout=None`.
- Load, from the Step 10 perception logs: 116-1,254 calls per episode (mean 459), ~53,000 per all-ten evaluation, each
  with four 640 x 480 images (3.7 MB uncompressed): ~190 GB downloaded per evaluation. Download from Cloudflare's edge
  measured ~80 MB/s here (100 MB test). kp_v2 ~0.1 s + runtime ~30 ms per call: ~2 h of GPU-serialized compute per
  evaluation, up to ~20 min of our wall time for a 5-3 D1 episode (to compare with `episode_wall_timeout_s`).
- Not installed: `cloudflared`, `tmux`. The venv has Python 3.11.14, websockets 15.0.1, msgpack 1.1.2 (the kit needs
  Python >= 3.11).
- HUMAN: the account, `track.json` (all ten tasks, 10 Hz) and `tower-policy-example.zip` into `data/challenge/`.

### Step 11: the policy server (2026-10-24 07:40-08:05)

The starter kit and the all-ten 10 Hz `track.json` are in `data/challenge/` (HUMAN). Kit (`tower-policy-example.zip`,
sha256 b0a4cfdd...): `server.py` (reference handler, hold policy), `tower_protocol.py` (codec, `sample_seed`,
`observation_spec` / `action_spec`), `requirements.txt` (msgpack 1.2.3, numpy 2.4.6, websockets 15.0.1) and a coding-agent
skill (`SKILL.md`, `references/contract.md`). The venv keeps its numpy 1.26.0 and msgpack 1.1.2 (Isaac Sim needs them;
the codec uses only APIs both versions have); websockets 15.0.1 matches. `track.json` (version
`20261013-v1-task45-10hz-reconnect`, sha256 efbc4adb...): cameras 480 x 640 (all four), `action_dt_s` 0.1,
`max_chunk_actions` 16, bounds +-6.2832 rad for the joints and [0, 0.10000000149] m for the grippers,
`inference_timeout_s` **180**, `episode_wall_timeout_s` **7200**, `simulation_limit_s` 210, `policy_seed_mode` fixed,
`infrastructure_retries` 0.

New code (the teacher and the student's behaviour unchanged):
- `tower_rl/server/tower_protocol.py`: the kit's file, verbatim (sha256 bba5de96...).
- `tower_rl/server/handler.py`: the kit's `handler_factory` with per-session model state. The kit's `predict(observation,
  *, prompt, seed)` gets no session, so every session record also holds a model object: `start(trial, seed)` at every
  reset (before the ack, even for a repeated seed), `infer(observation, index)` per call. It lives in the kit's session
  registry, so `tower-reconnect-v1` holds (duplicate requests get the committed bytes, no second model call; advertised).
  Model calls run in worker threads, one at a time across sessions (one kp_v2 on the GPU), so the event loop keeps
  sending metadata within 10 s. Changes are marked `# student:`.
- `tower_rl/server/student.py`: `StudentModels` (bundle files checked against MANIFEST.json, kp_v2, switcher, track
  bounds, warm-up), `SessionModel` (a `StudentAdapter` plus logs: per call `episodes/<variant>/<trial>.jsonl`, the four
  images of call 1 and every 100th call, `<trial>_summary.json` with node events and options), `EventLog`
  (`server.jsonl`).
- `tower_rl/student/adapter.py`: `prepare(variant)` (built at the reset from the worker's `X-Tower-Task-Variant` header),
  a per-call record, and `clip32`: clipping to the track bounds in float32 that stays inside the float64 bounds (6.2832
  rounds up in float32); clips beyond 1e-6 are counted (the open gripper's 0.10000002 m is not a mapping error).
- `scripts/serve_student.py` (+ `configs/student/server.yaml`): policy_id `tower-student-v1`, plugin `tower-rl-student` /
  version 1, `seed_mode` deterministic, artifact identity = the `student_v1` tar's sha256 (`sha256:01ab88da...`),
  `ping_timeout=None`, a status line a minute, graceful stop on SIGINT / SIGTERM.
- `scripts/probe_server.py`: the worker's checks (metadata vs the track's specs, identity echoes, seed fields and
  counters, `dt_s`, plugin fields, chunk dtype / shape / finiteness / bounds); `--replay` streams organizer episodes.
- `scripts/ws_loop.py`: G11a, the station scene through the server (server with `--camera_source render`).
- `tests/test_server_protocol.py` (7 tests, loopback WebSockets, fake models and the fake world): metadata, reset /
  infer echoes and counters, a second reset on one connection, independent sessions, reconnect without a second model
  call, rejections (unknown session, reused request id, identity, instruction), `clip32`, the adapter end to end.
  The suite: 284 passed, 3 skipped.
- `results/student/serve_official.sh <name>` / `stop_official.sh`: server (snapshot `server_v1`) and a Cloudflare quick
  tunnel (`~/.local/opt/cloudflared/cloudflared` 2026.10.3, checksum checked against the release notes,
  `--no-autoupdate`) in tmux session "tower", under `systemd-inhibit`.

Local results (server from the repo, real bundle, kp_v2 on the GPU; `results/student/server/local_probe_1/`):
- Load and warm-up 1.6 s.
- Qualification-like probes (`probe_check.json`): all ten variants at once, two resets each, zero images, joints drawn
  within the bounds: all replies valid; metadata within 6 ms; calls 33-88 ms mean (hold path).
- Organizer episodes (`probe_replay.json`): the ten local full episodes (`data/episodes_full`), 300 calls each (the
  first 30 s), ten sessions at once. Confident movable-block estimates vs the recorded poses: median 0.12-0.23 mm, p90
  0.21-0.57 mm, confident 99.9-100 % of calls; foundations p90 0.21-0.89 mm (as `Pose-estimator-v1.MD` on organizer
  frames). So decoding, image orientation, the evaluator camera model and the joint mapping are right through the wire.
  Server time per call 93 ms mean (estimator 86, runtime 9; p99 147 ms, max 1.56 s), 3,000 calls in 289 s;
  per-session latency with ten sessions 0.95 s mean, p99 2.3 s (timeout 180 s). No error.
- Expected official load: ~53,000 calls, i.e. ~85 min of serialized server time; the longest episodes (5-3 D1, ~1,250
  calls) ~16 min of wall time at 8 concurrent sessions (limit 7200 s).

### G11a: the closed loop through the server (2026-10-24 08:02-08:46): passed

`results/student/gpu_queue_g11a.sh` from the snapshot `server_v1` (the code the official server runs): the server with
`--camera_source render` on port 9011, `scripts/ws_loop.py` (the station scene renders the four cameras, a WebSocket
client per env sends each call's observation, the chunk is executed row by row), Task 2-2 and 4-2, D0 + D1, 4 episodes
per tier, seed 7000, 2 envs: the first 4 episodes per tier of the Step 10 run, compared per seed
(`results/student/ws_loop/g11a/`, server logs `results/student/server/g11a_ws_loop/`).

| Cell | Step 10 (in process) | ws_loop (through the server) | Same outcome |
|---|---|---|---|
| 2-2 D0 | 4 / 4 | 4 / 4 | 4 / 4 (durations within 0.3 s, 121 calls per episode in both) |
| 2-2 D1 | 4 / 4 | 4 / 4 | 4 / 4 |
| 4-2 D0 | 3 / 4 | 4 / 4 | 3 / 4 (7100000: failed in Step 10, succeeded here) |
| 4-2 D1 | 1 / 4 | 1 / 4 | 2 / 4 (7000001 and 7100000 swapped) |
| all | 12 / 16 | 13 / 16 | 13 / 16 |

- Task 2-2 reproduces call for call. The Task 4 differences are separate-process divergence: Task 4 outcomes are
  nearly independent even between two runs of one policy (Step 10 results, paired split 8 / 10 / 7 / 7).
- No server error, no adapter error, no hold after the first calls. Client-side latency per call 165-266 ms mean, p99
  0.47-1.37 s (2 sessions; the client also renders). Failed Task 4 episodes run to 210 s at ~1,800 calls each (one row
  per call while idle or pushing): 5,694 calls for 4-2 D1.
- Server memory with 12 sessions: 2.1 GB RAM, 2.1 GB GPU.
- G11a: probes pass, the closed loop reproduces the in-process results within the known noise, timing far within
  `track.json` (180 s per call, 7200 s per episode). Next: the official endpoint (`serve_official.sh`).

### Official run 1 (`student_v1`): endpoint, qualification, evaluation queued (2026-10-24 08:46-11:25)

- 08:46 `serve_official.sh official_1`: server from the snapshot `server_v1` (`student_v1`, artifact identity
  `sha256:01ab88da...4bccdd`, policy_id `tower-student-v1`, load 1.6 s) and a Cloudflare quick tunnel
  (`wss://mention-sampling-prove-neo.trycloudflare.com`; quick tunnels hold one connection, `ha-connections 1`,
  registered at sjc10). `probe_server.py` through the public URL: all ten variants valid, metadata 0.37 s, ~0.8 s
  per call with ten sessions (`results/student/server/official_1_probe_via_tunnel.json`).
- The site's "Check endpoint" (11:07:43-11:09:43, "Qualification: passed"): the ten variants one after another, one
  session each: metadata, reset (seed 0), one `infer` with blank images, a second reset (seed 0), one more `infer`,
  disconnect. Every call held ("blank", one row); resets 62-161 ms; no error. The check's sessions stay in the
  server's table until they expire (`episode_wall_timeout_s`), as did our probe's (expired 11:07:43).
- The user queued the evaluation at ~11:20; at 11:25 no evaluation session yet (queued on the site).
- Outcomes from the server's side: `scripts/server_episodes.py` summarises a server log directory per variant and
  per episode. An episode whose last chunk reaches 210 s ran to the limit; one that stopped before was ended by
  the evaluator, which locally happens only on success. On the G11a server logs this gives 8 / 8 and 5 / 8 early
  ends, the local evaluator's 8 / 8 and 5 / 8 successes, episode for episode. The failed Task 4 episodes there
  finished every node "ok" (all pushes, `park_left`) and idled to the limit: the runtime does not see those
  failures.

### System write-ups `Teacher-v1.MD` and `Student-v1.MD` (2026-10-24 17:40)

At the user's request, while official run 1 is queued: two reference documents in `~/Documents/corl_challenge/`. Each
covers every component of the teacher and student systems: inputs and outputs, how it is trained or set up, how it
connects to the rest. Each closes with a "facts for the open design questions" table (observability, failure detection,
recovery, hooks) and a doc-vs-code list. The facts were surveyed from the code, this log and the design docs, then
spot-checked against the code. Diagrams: 11 SVGs in `corl_challenge/figures/{teacher-v1,student-v1}/`, drawn by
`scripts/doc_figures.py` (matplotlib; no Graphviz here): `python scripts/doc_figures.py --out
~/Documents/corl_challenge/figures [--png <dir>]`. No code or results changed.

### Experiments beside the official endpoint: `scripts/eval_guard.py` (2026-10-25 07:00)

Official run 1 is still "Queued" (19 h at 06:35; other teams report the same on Discord), so the user wants to keep
working on the GPU meanwhile. Facts that decide how:
- The server and the tunnel live in VS Code's cgroup (`app.slice/snap.code.code-*.scope`), as does every shell
  started from VS Code or Claude Code. systemd-oomd is active and watches `user@1000.service` (memory pressure 50 % for
  20 s) and kills a whole cgroup, so a job that runs the machine out of memory from such a shell takes the endpoint
  with it. Kernel OOM kills pick the largest process instead (the server: 1.8 GB RSS, `oom_score_adj` 200).
- Every new episode rebuilds its executor from the bundle (`Executor.from_bundle`: `graphs/<variant>.json`, all
  `skills/*.pt`), so `results/bundles/student_v1` must not change while the endpoint is up (a new bundle needs a new
  name). Code edits are safe: the server runs from the snapshot `server_v1`, whose `data/`, `assets_m/` and
  `results/` are symlinks into this repository.
- Server GPU memory 2.0 GB now and 2.1 GB with 12 sessions in G11a. The start-up warm-up ran one heatmap pass, but
  the qualification's calls were blank holds that skip the estimator, so the first real calls may still allocate a
  little. Calls are serialized over sessions (~93 ms each): with
  8 sessions the longest episodes (~1,800 calls) take ~22 min against the 7,200 s limit, a margin of ~5x for GPU
  contention.
- `scripts/eval_guard.py -- <command>`: runs the command in its own `systemd-run --user --scope` (MemoryMax, default
  36G, no swap), at nice 10 and `oom_score_adj` 1000; SIGSTOP to its process group as soon as a session opens or a
  status line shows connected sessions or calls, SIGCONT after `--idle-min` (20) quiet; kills it if free GPU memory
  falls below `--min-free-gb` (2) while the server runs, and refuses to start below it. It only reads the
  endpoint's logs and `run.json`.
- Tests against a fake endpoint (scratch results tree, a `sleep` as the server's pid): scope, cap, swap 0, adj 1000,
  nice 10; pause on a session_open within one poll (sessions before the guard ignored), stays paused while
  connected, resumes after the idle time, pauses on calls alone, resumes when the endpoint dies; Ctrl-C resumes and
  forwards (also for a guard started in the background, whose SIGINT is inherited ignored), a second Ctrl-C kills
  the group; floor above free memory: not started; 1.5 GB allocated by another process below the floor: killed
  within 15 s.

### Official run 1: the evaluation started; its observations show no blocks (2026-10-26 11:56-13:10)

- Started 11:56:48, ~48.5 h after queuing. The first session reset and disconnected after 98 s without a call; the
  worker restarted the episode in a new session 3 s later (`server_episodes.py` now reports such sessions as
  "empty"). From 12:58 several workers ran Task 1-2 episodes in parallel.
- Official episodes always run to the 210 s limit, also after success (the first ran 11:58:29-12:54:02, 1,477 calls,
  55.5 min); "early" never happens here, so outcomes come only from the site. Measured pace with one worker: the
  next call arrives ~1.0 s + 0.91 s per row of the previous reply after it (evaluator simulation ~9x slower than real
  time); our server ~0.2 s per call. Worst case (2,100 one-row calls) ~68 min, well inside the 7,200 s wall limit.
  Idle 1-row holds after the plan finishes cost ~2 s each: v2 should hold with 16-row chunks (`max_chunk_actions`).
- **The delivered images contain no blocks in most episodes** (5 of the first 6 Task 1-2 episodes; the sixth, `2a3ccfdc`, shows them, 4-9 % wood-coloured pixels like the dataset frames, 0 % in the others): every saved frame (calls 1, 100, ..., 1400) of the first episode and the
  first frame of the next one show the station, the arms and the empty table, in all four cameras; at call 100 the
  back wrist camera shows the fingers closing on nothing. The blocks exist in the physics: every grasp verdict needed
  a measured opening of a block's width (44-56 mm) while lifting, and all four were "ok". kp_v2 found nothing
  (confidence 0.30 = prior, 0.00 = kinematic), so the student ran its plan on the descriptor's nominal poses.
  Evidence: `results/student/official_1_observations_vs_dataset.png` (official frames vs the public dataset's
  task1-2 frames).
- **Orientation**: the fixed cameras (`left_back`, `right_front`) arrive rotated 180 deg relative to the public
  dataset frames of task1-2 (normalized correlation 0.85 / 0.87 rotated vs 0.02 / -0.09 as is); the wrist cameras
  match as is (0.84 / 0.73). The live docs ("Observations"): "Official Task4/5 observations match the JPEG orientation
  in the pinned public dataset", "Both wrist views retain their original orientation", "Apply any transformation
  needed by your own checkpoint inside its model adapter". kp_v2 was validated only on dataset frames (our probe), so
  the rotation went unnoticed; for Tasks 1-3 a v2 adapter must rotate the fixed cameras (to confirm per task with the
  organizers, and check Task 4/5 frames when they arrive).
- **The orientation is ours to fix**: `perception/cameras.py::deliver` turns our static-camera renders by 180 deg to
  the dataset JPEGs' orientation (memory: the stored Task 1-3 JPEGs are turned), and kp_v2 learned that orientation;
  the official Task 1-2 arrays come unturned (the docs promise the dataset orientation only for Tasks 4/5). Offline on
  CPU (`orient_test.py`, the official frames of `2a3ccfdc`, q from the dataset's first frame): as delivered every
  block stays at the prior (conf 0.30); with the static cameras turned 180 deg conf 0.60-1.00 and poses within 3-5 mm
  of the dataset episode's start. So even with blocks rendered, `student_v1` is blind on Task 1-2 officially; a v1.1
  adapter must turn the static cameras (per task, or chosen per episode by detection confidence on the first call).
- By 14:30: blocks missing in 8 of the first 14 episodes with frames (Task 1-2: 6 of 10; Task 1-3: 2 of 4); the
  others render them (max wood-coloured share 0.086-0.103 vs 0.000-0.007). With eight workers, a one-row call cycle
  is 1.76 s (one worker: 1.95 s), our server 0.22 s per call at ~4 calls/s: no wall-limit risk (2,100 one-row calls
  ~62 min). The monitor's `grep` here is ugrep 7.8.4, which held back lines even with `--line-buffered`: the watcher
  filters its start line itself (`WATCH_QUIET_START=1`).

### Lessons write-up `Lessons-v1.MD` (2026-10-26 19:30)

At the user's request: `~/Documents/corl_challenge/Lessons-v1.MD` holds the lessons of official run 1 and the
handoff: state, timeline, the orientation miss (ours) and the missing blocks (theirs), official episode semantics and
pace, what the blind student did, process lessons, a v1.1 checklist and how to resume (watchers, analysis, organizer
message). The session's tools moved from the scratchpad to `scripts/official/` (`watch_official.py`, `obs_check.py`,
`orient_test.py`, `call_pace.py`). Further facts while writing:
- `orient_test.py` on task1-3 (`671f5dc1`) and task2-2 (`259004de`): every block conf 0.30 as delivered, 1.00 with the
  fixed cameras turned. Task 1-3 correlations: rot180 +1.00 / +0.99 (fixed), same +0.99 / +1.00 (wrist).
- Task 1-2 splits by rendering: the 6 blank episodes all ran the nominal plan (t0 / t1 places "ok", t2 / t3 "failed");
  the 4 with blocks had a few confident detections (6-18 calls) that changed the order and the grasp targets (every
  place "failed", grasp timeouts; `6b7e4c9a` placed nothing). Misoriented but visible was worse than blank.
- Task 2-2: in all 8 episodes both transfers ran (`t0_place` "failed", `t1_place` "ok"), the plan ended at ~50 s,
  then 1,595 one-row idle holds. Task 1-3: later grasps time out in all 10 (`t2_grasp_alt` in 9); only `c6ba40b4`
  placed all six.

### Official run 1 resumed (2026-10-26 22:18)

- After 4 h 38 min without sessions (17:40:35-22:18:49 PDT), the evaluation resumed: two Task 2-2 sessions (22:18:49,
  22:19:18), then Task 2-3 from 22:20:05. The two stale sessions (`1b9cf5ba`, `8c71a690`) expired when the first new
  one opened. The cut episode (`1e6d738a`, 164 s) was not re-run: with the two new ones Task 2-2 has 10 sessions with
  calls. The first call comes 1-2 min after the reset (73 s and 136 s in earlier episodes).
- All three new episodes render blocks (wood share 0.04-0.08 in every camera). `obs_check.py`: Task 2-3 is rotated
  like Tasks 1-2 / 1-3 / 2-2 (fixed cameras rot180 +1.00 / +1.00 vs same +0.02 / -0.15; wrist cameras same +0.99 /
  +1.00). `orient_test.py` on `fb7911c5`: as delivered one block 0.60, the rest 0.30; fixed cameras turned: every
  block 1.00, poses 1.5-4.5 mm from the dataset episode's start. `Lessons-v1.MD` sections 1-4 updated.
- **Count corrected: 15 of the first 28 episodes render no blocks, not 16** (Task 2-2: 3 with / 5 without). The wood
  share fails under the scene's lighting cases: sepia light makes a blank table "wood" (`72f06002`, Task 2-3, 0.72),
  and cool light makes blocks pale (`e8dcf797`, Task 2-2, max 0.016: counted blank, but it shows blocks). The
  edge density at the table centre of the back wrist camera's first frame separates them cleanly (blank
  0.0013-0.0032, blocks 0.0105-0.0216; every tinted episode checked by eye); `obs_check.py` now lists this verdict
  per episode (threshold 0.006). Since the resume (to 22:40): 5 of 7 render blocks (Task 2-3 `72f06002`,
  `4844c92d` blank). `Lessons-v1.MD` section 4 and the organizer message updated.
- 23:05:23: cloudflared's only edge connection failed ("datagram manager error: timeout: no recent network
  activity") and re-registered in 1 s (sjc01). All 8 sessions disconnected at once; all 8 were connected again by the
  23:05:53 status line, and two resent their in-flight request (`duplicate_request`: the kit's dedupe returned the
  committed bytes, no model call). The reconnect protocol works over a tunnel blip; the watcher reports
  `duplicate_request` as an error kind, which is benign here.
- 23:09:19: session `5565e87c` (Task 2-3 `72f06002`, at 177 s) disconnected once more; the watcher's tally called it
  "early", but calls resumed within a minute (the server logs no reconnects; its status line still counted 8
  connected). `server_episodes.py` now closes an episode on a disconnect only once the disconnect is
  `RECONNECT_GRACE_S` (120 s) old with no later call. So far the only episode really cut is `1e6d738a` (17:40).
- 23:57: Task 3-2 started (first episode `77ca0561`, blank). `obs_check.py`: fixed cameras rot180 (+0.98 / +0.93 vs
  same +0.05 / -0.07), wrists same: the rotation holds for every Tasks 1-3 variant seen (1-2, 1-3, 2-2, 2-3, 3-2).
  Task 2-3 complete as to starts: 6 of 10 episodes render blocks. Task 2-2 finished (9 at the limit, 1 cut at 164 s).
  Workers varied 2-8 after the resume.
- 00:11: first Task 3-2 episode with blocks (`5d284ee9`; its flat three-block layer gives an edge density of only
  0.0072, above the 0.006 threshold but with less margin). `orient_test.py`: as delivered every block 0.30; fixed
  cameras turned: every block 1.00, poses 0.1-3.5 mm from the dataset episode's start.
- 01:16: Task 3-3 started (`2a866e8d`, blocks render). `obs_check.py`: fixed cameras rot180 +1.00 / +1.00 (same
  +0.06 / -0.12), wrists same. `orient_test.py`: as delivered every block 0.30; turned: every block 1.00, bottom layer
  0.0-0.2 mm and the three stacked blocks 4.7-8.9 mm from the dataset episode's start (not the official truth). All
  six Task 1-3 variants are now confirmed: fixed cameras arrive unrotated (raw), kp_v2 needs them turned.
- **Correction: official episodes end on success** (organizers' README, "Evaluation": "An episode ends on success,
  the per-task time limit, or a policy error"). The 10-26 note above ("always run to the 210 s limit, also after
  success") generalized from the first episode, which failed. So "limit" = failure, and `server_episodes.py`'s
  original reading (early = success) holds, apart from evaluator-side cuts (`1e6d738a`, Task 2-2, 164 s). By 01:30:
  Task 1-2 / 1-3 / 2-2 / 2-3: every finished episode at the limit (39 failures + the cut one). Task 3-2: 5 finished,
  every verdict "ok" in all 5 (blind plan, same timing); 4 at the limit, `218cdf6e` (blank) ended at 74.1 s, 30.6 s
  after `park_back`, no reconnect: probably a success, to confirm on the site. Task 2-3: `t1_place` "failed" in 9 of
  10. `Lessons-v1.MD` sections 0, 5, 6 corrected.
- 01:46: second probable success, Task 3-3 `4fcafedc` (blank): places "ok" at 21.2 / 40.7 / 61.8 s, ended at 91.0 s.
  Both probable successes ended ~30 s after the final place (33.4 s and 29.2 s): the evaluator seems to declare
  success once the arrangement has held ~30 s. In it, `park_left` re-fired every 0.5 s from 62.3 to 77.3 s (31
  events) before settling.
- 01:54-01:57: four new Task 1-2 sessions (two empty, two running) although Task 1-2 had all 10 episodes at the limit
  and `infrastructure_retries` is 0: a manual "Retry failed episodes" or an organizer re-run; not visible from our
  side.
- 02:30: Task 1-3 sessions started too, after 7 new Task 1-2 episodes (01:54-02:29, plus 5 empty sessions). The
  evaluator is going through the variants from the start: a new or restarted evaluation from 01:54:43, while the
  first pass stopped at Task 3-3 3 / 10 (Tasks 4-5 never ran). The server cannot tell runs apart (same binding); use
  `--since 2026-10-27T01:54` for the second pass. Its first 3 Task 1-2 episodes with frames render blocks (the first
  pass: 4 / 10); the fixed cameras are still rotated.
- 02:40: sessions without a call ("empty") rose to 23 of 82 (14 since 01:50). Our reset replies took 62-264 ms; the
  worker disconnected 56-154 s after the reset; no server or tunnel errors since 23:05. Evaluator-side (scene start
  failures or timeouts under load); the worker retries.
- 03:00: second pass (from 01:54): 9 of 9 episodes with frames render blocks (Task 1-2 4/4, 1-3 3/3, 2-2 2/2);
  ~0.2 % by chance at the first pass's ~50 % blank rate, so the restart probably fixed the rendering. Orientation
  unchanged. Second-pass Task 1-2: 3 at the limit so far (failures). Variants now start in parallel (1-2, 1-3, 2-2).
- 03:10:54: second tunnel blip (re-registered in 2 s, sjc10). All 8 sessions dropped; every running episode
  reconnected (none closed after the 120 s grace), one resent its in-flight request (`duplicate_request` #3).
- 05:30: second-pass Task 3-2: 4 more episodes, the same all-"ok" plan, all at the limit (1 probable success in 14
  overall). A failed final arrangement (`56cce224`, call 1800) looks almost like the dataset's success frame, with the
  top block slightly shifted and turned in the wrist views (`results/student/official_1_task32_final_vs_dataset.jpg`):
  the blind plan comes close but usually misses the tolerance. In the dataset HDF5 `sim/success` and `sim/terminal`
  rise together, 8-9 s after the last release (generator phases 9 -> 10 -> 12 -> 13), for Tasks 1-2, 2-2, 2-3, 3-2,
  3-3.
- 06:10: **Task 4-2 started (first Task 4 episodes); its frames match the dataset as delivered** (`obs_check.py`:
  all four cameras "same" +1.00; rot180 +0.02 / -0.12 for the fixed ones). `orient_test.py` on `e7d9fdf0`: as
  delivered every block 1.00, poses 0.0-0.3 mm from the dataset episode's start; turned: 0.30 (two at 0.60). The docs'
  Task 4/5 promise holds for 4-2, so `student_v1` sees normally there (Step 10 locally: Task 4 18 / 32).
- 07:06: first Task 4-2 episode (`e7d9fdf0`) finished at the limit although it saw normally (conf 1.00) and pushed all
  six in order with every verdict "ok" (along 152-165 mm; plan done 42.2 s; local Task 4-2: 9 / 16 successes, median
  41.8 s). Final frame: blocks off the tower, one group of three fanned at angles (dataset success: side by side);
  `results/student/official_1_task42_final_vs_dataset.jpg`. If more Task 4-2 episodes fail like this, the official
  push success rule differs from `PushTracker` (our reconstruction). Dataset `layout_metadata_json`: "terminal_scene":
  "dynamic scattered blocks; no authored terminal target"; config `final_settle_seconds` 2.0.
- 07:20: Task 4-3 started; frames match the dataset as delivered (all four cameras "same" +1.00), blocks render;
  `orient_test.py` on `d08100d8`: as delivered every block 1.00, turned 0.30.
- 07:30: Task 4-2 3 / 3 failures: `e7d9fdf0`, `63628fa8` (all six pushes "ok", in order) and `70edfc49` (p3 skipped:
  `source_l1_2` had moved during the top-layer pushes, lat errors 16-25 mm; then `l1_1` pushed 250 mm: order broken).
  The first two need the site's per-operation result: which push did the official scorer fail?
- 08:00: Task 4-2 finished 7 of 7 at the limit with working perception (local 9 / 16 successes; P(0 / 7) at that
  rate ~0.3 %). Either the official Task 4 success rule is much stricter than `PushTracker`, or official episodes do
  not end on success and the two early ends (Task 3-2 `218cdf6e`, Task 3-3 `4fcafedc`) were cuts. Only the site's
  results decide; the user reports replays of Tasks 1-3 "seem to do okay".
- 08:32: Task 5-2 started; frames match the dataset as delivered (all cameras "same" +0.99-1.00), blocks render;
  `orient_test.py` on `26fc69cc`: as delivered every block 1.00, turned 0.30.
- 09:05: Task 5-2 (perception works): 3 of 4 episodes inserted fine (along -91 mm) but the distilled push-in
  "failed" (6.9 / 25.1 / 48.2 mm from goal, structure moved 6.2-8.3 mm); `cc2b8ac0` dropped the block on the first
  carry, then the insert aborted ("structure", 20.5 mm). All idle afterwards, none ended early. Local Task 5: 21 / 32
  with push-in losses of a few mm. The official 5-2 contact behaviour differs from ours: a v1.1 question.
- 10:01: Task 5-3 started; frames match the dataset as delivered (all cameras "same" +0.99-1.00); `orient_test.py` on
  `7e4f563a`: as delivered every block 1.00, turned 0.30. **All ten variants verified:** Tasks 1-3 fixed cameras
  arrive rotated 180 deg vs the dataset (raw), Tasks 4-5 as the dataset. The v1.1 adapter rule can be per task.
- 10:30: Task 5 pattern (5-2: 11 episodes, 5-3: 5): mostly "structure" aborts (tower side/roof blocks moved 13-27 mm
  during insert or push-in), else push-ins ending 7-71 mm from the goal; none ended early. Locally the structure
  break was the D1 failure mode; officially it dominates.
- 12:30:20: the last session ended (Task 5-3, at the limit); no sessions since. Server side over the whole run: 113
  episodes with calls (110 at the limit, 3 early: Task 2-2 cut 164 s, Task 3-2 `218cdf6e` 74 s, Task 3-3 `4fcafedc`
  91 s) and 81 sessions without a call. Per variant: 1-2 14, 1-3 13, 2-2 15, 2-3 13, 3-2 14, 3-3 10, 4-2 7, 4-3 8,
  5-2 9, 5-3 10. The endpoint stays up until the site shows "evaluated" with no retry pending (the user's call).
- 14:18:50: a third pass started (Task 1-2 again; same binding `20261013-v1-task45-10hz-reconnect`, same endpoint and
  bundle). Unknown whether a retry, a resubmission or an organizer re-run: ask the user what the site shows. First
  episode with frames (`2422a24a`, reset 14:20:50): blocks render (edge 0.0151), fixed cameras still rotated 180 deg
  vs the dataset (rot180 +0.86 / +0.99, same ~0.0), so the student is blind on Tasks 1-3 again. Empty sessions recur
  (`bc5704e7` at 14:18:50, `d910dea7` at 14:20:50).
- 14:45: third pass, first frames of 7 episodes (Task 1-2 x2, 1-3, 2-2, 2-3, 3-3, 4-2; up to 7 workers at once,
  variants interleaved): all render blocks. Two read low on the edge measure but show blocks by eye: Task 3-3
  `32d9d53e` 0.0062 (a flat layer of three, sepia light; just above `BLOCKS_MIN` 0.006) and Task 2-3 `e98c61cd`
  0.0089. A reading below ~0.01 needs a look. Duplicate request #6 (14:35:59, session `301040aa`): a single-session
  reconnect 3 s after a disconnect, deduped; no tunnel error.
- 15:15: third pass, Task 4-2 `e20b7613` (reset 14:30:44, perception working): all six pushes "ok" and in order, parked
  at 44.5 s, yet the episode ran to the limit (last call t 209.9). The third Task 4-2 episode with all-"ok" pushes
  that did not end early (after `e7d9fdf0`, `63628fa8`). Final scene
  (`results/student/official_1_pass3_task42_e20b7613_first_vs_last.jpg`, rows: first call, call 1600): three pushed
  blocks lie on their sides in a row as in the dataset's success frame, the other three fanned at angles; the
  dataset's success frame has both pushed layers as neat parallel groups. Either the official Task 4 success rule
  checks how the pushed blocks land, or episodes do not end on success at all; the site's results decide. Task 1-2
  `284cfd05` and Task 1-3 `79eb4675` ran to the limit with failed places (blind).
- 16:00: third pass, Task 4-2 `8d5a38bb` (reset 15:13:17): all six pushes "ok", parked, ran to the limit (t 209.9):
  the fourth such Task 4-2 episode. Here the last frame (`results/student/official_1_pass3_task42_8d5a38bb_first_vs_last.jpg`)
  shows one pushed block lying across the base layer (back wrist and right_front cameras), a plausible failure under
  any "off the tower" rule; our per-push "ok" is judged at push time, not after the final settle.
- 16:20: third pass, more all-"ok" episodes at the limit: Task 4-3 (resets 15:24:06, 15:27:46; 21 events each, every
  verdict "ok", parked) and Task 5-2 `80aa6968` (15:35:52): insert along -91.0 mm / lat 0.1 mm / moved 0.0 mm, push-in
  place 2.97 mm / yaw 0.14 deg / structure moved 8.1 mm, parked at 42.8 s, then idle to 210 s. Across all passes no
  Task 4 or 5 episode has ended early (4-2 10, 4-3 10, 5-2 10, 5-3 10 so far), including 17 with every verdict "ok"
  (4-2 8, 4-3 7, 5-2 1, 5-3 1; `server_episodes.py --json`, episodes with no "not ok" event). With perception working,
  17 of 17 clean runs by our runtime's verdicts did not end early.
  Either the official success rules are stricter than ours on exactly the things our runtime does not re-check (the
  settled scene; Task 5 structure moved 8.1 mm), or official episodes do not end on success and the three early ends
  in Tasks 2-3 were cuts. The site's per-episode results decide.
- 16:30: the third pass ended (no session connected since ~16:20): 12 episodes, all at the limit (1-2 2, 1-3 1, 2-2 1,
  2-3 1, 3-3 1, 4-2 3, 4-3 2, 5-2 1), plus 4 empty sessions. **The three passes add up to one evaluation of 10
  episodes per variant, in which every blank-render episode and the cut episode were re-run.** Per variant, the episodes
  with calls in passes 2 + 3 equal pass 1's blank-render episodes (`obs_check.py`, all 125 episodes) plus the episodes
  pass 1 never ran: 1-2 6 = 6 blank; 1-3 4 = 4 blank; 2-2 6 = 5 blank + the cut `1e6d738a` (blocks, cut at 164 s);
  2-3 4 = 4 blank; 3-2 4 = 4 blank; 3-3 8 = 1 blank + the 7 not run; Tasks 4-2, 4-3, 5-2, 5-3 10 each (not run in
  pass 1). Every episode in passes 2 and 3 rendered blocks (72 of 72: 60 + 12). So the scored set is presumably 100 episodes, all with
  blocks visible. Both early ends that read as probable successes (Task 3-2 `218cdf6e`, Task 3-3 `4fcafedc`) were
  blank-render episodes, so they were most likely replaced. Every episode in the presumed scored set ran to the limit.
  If "limit" means failure that is 0 / 100; if episodes do not end on success the server cannot say. The site decides.
  The many empty sessions of passes 2-3 (72 + 4) may be the evaluator discarding blank renders at reset (a guess).
  The organizer message's point (1) and the cut-episode question look answered (re-run); point (2), the Tasks 1-3
  orientation, stands. The run is probably complete: check the site for "evaluated".
- 16:40: the user reports the site shows the run as `awaiting_review` with **estimated rank 1**. That fits badly with
  "every scored episode ran to 210 s, so 0 / 100". Either the site credits successes inside episodes that ran to the
  limit (episodes do not end on success), or the rank rests on the progress rate or on weak competition. Waiting for the
  per-task success and progress rates. The endpoint stays up: `awaiting_review` is not "evaluated".

### Official run 1: the scorecard (2026-10-27 16:50, site status `awaiting_review`)

The user's screenshot: `~/Documents/corl_challenge/Screenshot from 2026-10-27 16-50-44.png`. **59.32 points,
estimated overall rank 1.** Per variant, as TSR / LHPR / ODR / task score -> points (weight), with the estimated task
rank:

| Variant | TSR | LHPR | ODR | Task score | Points (weight) | Est. task rank |
|---|---|---|---|---|---|---|
| 1-2 | 0% | 78.8% | 0.151 | 51.81 | 2.88 (x1) | #3 |
| 1-3 | 10% | 48.3% | 0.791 | 33.99 | 1.89 (x1) | #3 |
| 2-2 | 80% | 90.0% | 0.026 | 86.44 | 4.80 (x1) | #2 |
| 2-3 | 80% | 86.7% | 0.000 | 84.44 | 4.69 (x1) | #3 |
| 3-2 | 90% | 97.5% | 0.025 | 94.77 | 5.27 (x1) | #1 |
| 3-3 | 80% | 96.7% | 0.067 | 90.54 | 5.03 (x1) | #1 |
| 4-2 | 0% | 73.3% | 0.259 | 47.88 | 7.98 (x3) | #1 |
| 4-3 | 0% | 56.7% | 0.984 | 35.90 | 5.99 (x3) | #2 |
| 5-2 | 50% | 75.0% | 0.700 | 63.92 | 10.65 (x3) | #1 |
| 5-3 | 40% | 75.0% | 0.639 | 60.86 | 10.14 (x3) | #1 |

- **The scoring.** The site says weights are 1:1:1:3:3 per task and task ranks use LHPR.
  - Points = task score x weight / 18, which reproduces 59.31.
  - A least-squares fit of task score is about 0.34 TSR + 0.6 LHPR + a few points for low ODR (residuals up to
    about 1 point; not the published formula).
  - LHPR is the README's "longest completed in-order prefix of task points / all task points". ODR is probably its
    "distinct non-target blocks disturbed per episode".
- **Episodes do not end on success.** Every scored episode ran to 210 s while TSR reached 90%. The "limit = failure"
  reading and the "in doubt" notes above were wrong. The server never sees success.
- **Blind Tasks 2-3 succeeded 80-90%**: the nominal plan is good enough there. Task 1 got 0% and 10%: blind "failed"
  verdicts set off retries and re-grasps (ODR 0.79 in 1-3).
- **Task 4: 0 / 20, with 15 runtime all-"ok" runs.** The official checker rejects pushes we call "ok". LHPR of
  about 4.4 / 6 and 5.1 / 9 pushes in order fits a later push in each episode not counting. Local 9 / 16 per variant
  was optimistic.
- **Task 5: 50% and 40%, with one runtime all-"ok" run each.** Our push-in and structure verdicts are stricter than the
  official rule.
- **Points lost (40.7):** Task 4 19.4, Task 5 12.5, Task 1 6.3, Task 2 1.6, Task 3 0.8. `Lessons-v1.MD` sections 0
  and 8 have the v1.1 priorities.

## 2026-10-27/28 (session 7): `student_v1_1` and the recovery node

Plan: `~/Documents/corl_challenge/Plan-v1.1.MD` (Step 1: orientation + the official mirror; `student_v1_1` for submission
on 10-28; then the recovery node, teacher first, then the student). Everything new is opt-in (graph flags, rehearse.py
flags), so the teacher, `student_v1` and the parity tests are unchanged: fake-world parity 80 / 80 after the changes,
full unit suite passing.

### Code (session 7)

- **Orientation** (`tower_rl/perception/orient.py`, `scripts/build_orient_refs.py`): per episode, on the first call
  with images, each fixed camera and its 180 deg rotation are correlated (160 x 120 normalised grayscale) with the
  variant's dataset first frame (`results/perception/orient_refs.npz`, 20 references); the better wins if >= 0.5 and
  ahead by >= 0.3, else the per-task fallback (rotate Tasks 1-3). Checked on every official first frame of official
  run 1: **125 / 125 chosen by correlation as the per-task rule says**; right orientation >= 0.86, margin >= 0.89 (Task
  1-2 lowest). On our Isaac renders (G11a frames) the right orientation scores ~0.60 (left_back 0.78, right_front
  0.42): our renderer differs, still above the thresholds. `StudentAdapter` applies it before the estimator (the
  recorder keeps the delivered images) and logs the choice in the first call's record and the summary.
- **Blind guard** (adapter + graph flag `blind_guard`): when the runtime first acts and no movable block was ever
  confidently seen, the episode is BLIND; transfer family: `Oracle("graph")` and place verdicts taken as ok.
- **Idle chunks** (graph `idle_chunk`): idle emits up to N hold rows per call; the re-query stays at 2.0 s (16 + 4 rows).
- **Finished guard** (graph `finished_guard`): once every goal's final stage ended ok, a goal option is vetoed unless its
  block's rest estimate moved > 10 mm for >= 2 s (vetoes logged in the summary).
- **The official mirror** (`tower_rl/eval/mirror.py`; `rehearse.py --mirror`, `ws_loop.py --mirror`): every episode to
  210 s, success judged on the final arrangement (`run_batch(full_episode=True)`, `extra.t_first_success`); Task 4 P per
  push on the block's final rest state (`OpCfg.push_rule="rest"`: flat on the table, >= 2 mm outside the foundation
  footprint in the foundation's frame, static; G = P), `extra.rest`, `task_success_strict` (+ no debris moved > 10 mm),
  `task_success_radial` (the v1 rule). `scripts/official/task4_outcomes.py` now uses the same rule: official run 1
  re-scored 4-2 44 / 60 = 73.3 % (site 73.3 %), 4-3 48 / 90 = 53.3 % (site 56.7 %); mirror TSR 1 / 10 and 0 / 10 (site
  0 and 0): within the plan's acceptance (TSR within one episode, LHPR within 5 points). `scripts/mirror_points.py`:
  TSR / LHPR / ODR, fitted task score and points per variant.
- **Camera loop with the official orientation** (`rehearse.py --official_orient`, `ws_loop.py --official_orient`): the
  fixed cameras of Tasks 1-3 delivered unrotated and turned back by the orientation choice.
- **`push_settle`** (graph flag; `PushNode`): v1's push until its radial test fires, then hold still until the block's
  hull is confirmed >= 2 mm outside the foundation footprint on 5 consecutive confident calls at rest; a block still
  seen over the footprint is finished by the planned straight push; unconfirmed after 4 s it ends as v1 would. A first
  version (planned slow push from 60 mm on) dropped blocks onto the base and dragged neighbours (D0 smoke): dropped.
- Endpoint scripts take `PORT`, `SESSION`, `SNAP`, `BUNDLE`, `EXTRA` (defaults = official_1); `scripts/make_snapshot.sh`.

### Results (session 7, night of 2026-10-27/28)

**Post-success: the finished guard is necessary.** The camera-loop smoke (Task 1-2, lighting case, our renders with
the official orientation, to 90 s) placed all four blocks in tolerance (first success 82.0 s) and then the switcher
started `t1_to_pregrasp` at 88.9 s: re-acquiring a placed block. CPU A/B under the estimator noise model v1, the
mirror (to 210 s), the ten cases, 20 episodes per cell, `finished_guard` off vs on (`results/v1_1/post_noise/guard_ab`):

| Variant | off: TSR, ODR | on: TSR, ODR |
|---|---|---|
| 1-3 | 20 / 20, 0.00 | 20 / 20, 0.00 |
| 2-3 | **0 / 20, 0.43** | 20 / 20, 0.01 |
| 3-3 | **0 / 20, 0.40** | 20 / 20, 0.00 |
| 1-2 | **0, 0.42** (first 16) | 20 / 20, 0.00 |
| 2-2 | **0, 0.40** (first 4) | 4 / 4 (first batch) |

Without the guard the switcher reworks finished goals in the 100-160 s after the plan (it was trained to reopen goals
that look disturbed, and every run before stopped at success); with working perception that would have cost most of
Tasks 1-3 officially. In official run 1 Tasks 1-3 were blind (estimates frozen at the prior), so it never showed.

**Task 4 `push_settle` (CPU, simulator poses, mirror, ten cases, 24 per cell, seed 7100;
`results/v1_1/ab_task4/settle_v2`):**

| Arm | 4-2 TSR / LHPR | 4-3 TSR / LHPR | points (4-2 + 4-3) |
|---|---|---|---|
| v1 push | 37.5 / 84.7 % | 0 / 67.6 % | 19.04 |
| `push_settle` | 62.5 / 92.4 % | 0 / 74.1 % | 21.92 |

Rest states: 4-2 leaning 5 -> 0, tipped 14 -> 9; 4-3 "on a block" 38 -> 38 (layer-1 blocks landing on layer-3 debris,
pushes 6-8 in 10-15 of 24 episodes each), tipped 18 -> 10. The planned finish fired in only 3 / 146 and 7 / 230
pushes: most of the gain is holding still until the block is confirmed at rest instead of retreating at once.
G10's camera-loop Task 4 episodes (v1) re-scored with the mirror from their logged truth: 4-2 90.6 % flat clear (9 / 16
all clear), 4-3 67.4 % (0 / 16; on a block 19, tipped 15, leaning 10).

**4-3 follow-through (dropped):** the demos push 4-3's top layer 0.25 m at up to 0.18 m/s (`push_speed_m_s`); those
blocks stand on end, then topple outward to 316-347 mm, leaving the layer-1 landing zone (161 mm) clear. A planned
follow-through to 0.265 m of pinch travel at 120 mm/s on layer 3 made 4-3 worse (LHPR 0.65 vs 0.74, ODR 0.65 vs 0.07
after 16 episodes); not investigated further.

**Adapter on the official frames** (`scripts/official/adapter_replay.py`, release candidate `student_v1_1_rc1`):
every episode with blocks rendered: orientation chosen by correlation (0.93-1.00), all movable blocks confident
(8-9 / 9 in 4-3, 6 / 6 in 5-2, 3 / 3 in 2-3), no holds; blank renders: three "unconfident" holds, then BLIND and the
runtime acts.

**Finished guard, complete** (`results/v1_1/post_noise/guard_ab`, Tasks 1-3, 20 per variant, noise v1, mirror, ten
cases): guard on **120 / 120** (33.23 points over the six variants), off **1 / 120** (23.39 points; goal nodes started
after the first success: 177-416 per variant vs 0-11 with the guard; ODR 0.37-0.43). The guard ships in v1.1.

**The teacher's recovery node, first rules** (`results/v1_1/teacher_recovery/r1`, state oracle vs state oracle v2 on
truth, Tasks 4-5, push_settle, mirror, ten cases, 20 per cell): 47.19 vs 46.81 points (4-3 TSR 1 / 20 vs 0 / 20, LHPR
71.7 vs 70.0 %; 4-2 and 5-x unchanged). `scripts/recovery_review.py`: in 4-3 the failing rest states are layer-1 blocks
on layer-3 debris (pushes 6-8: 29), tipped (12), off the table (3); none is a re-push case. In 5-3 every failure is a
structure break (10-23 mm) during the layer-1 insert or the upper push-in; the rules abstain ("prevention's job"). The
first R4.1 also fired on blocks flat against the base (pre-push pose inside the foundation: transit timeouts) and, in a
heavily injected smoke, on a block pushed only 38 mm with the layer above still on the tower (structure failures):
tightened (push ended ok, moved >= 0.10 m, near end >= 45 mm high, no higher-layer block on the tower), the same test
masks `repush` in features v2, and exploration never picks a recovery option.

**Why the 4-3 follow-through failed** (`results/v1_1/ab_task4/settle_v2/follow_diag`, 8 episodes, same seeds as the
A/B): the first top-layer push (`p0`) failed its structure check in 8 / 8 episodes (the rest of the tower moved
19-87 mm at 164-222 mm of travel), and the windows of p0 / p1 record 4-7 disturbed blocks each. Continuing the stroke
past the radial test drags the gripper along the vacated slot through the rest of the top layer: our pinch rides only
~0-2 mm above the top surface at the 45 deg tilt, while the demo's tool has a 10 deg forward pitch with its pad 15 mm
above the block centre (`motion_choices_json`). A follow-through needs a raised, more upright pusher; open.

**Push speed near the edge (graph `push_settle.edge_slow`, a probe for v1.2; CPU, same seeds and cells as the A/B):**
x0.5 from 50 mm out: 17.69 points (4-2 TSR 16.7 %, LHPR 84.7 %; 4-3 LHPR 66.2 %, on a block 48); x1.5: 20.77 points
(4-2 identical to push_settle, 62.5 / 92.4 %; 4-3 LHPR 63.9 %, tipped 29 vs 10, ODR 0.23). Plain `push_settle`
(21.92) stays: slower pushes drop blocks next to the base, faster ones tip more of 4-3's blocks.

**Finished guard on Task 5** (`results/v1_1/post_noise/guard_ab_t5`, noise v1, mirror, ten cases, 20 per cell): on and
off identical (5-2 TSR 65 %, LHPR 82.5 %; 5-3 65 %, 87.5 %; 25.78 points): the switcher does not rework Task 5 after
success, and the guard costs nothing there.

**Release candidate checks** (`student_v1_1_rc2` = core + push_settle, served from snapshot `server_v1_1` on a local
port): the qualification-like probe passes on all ten variants; dataset replay through the server (Task 2-2, 4-2):
estimates p90 0.23 / 0.39 mm, no holds, orientation kept as delivered by correlation (1.00), not blind, no errors.
Camera loop with the v1.1 core (+ guard), mirror, ten cases: Task 1-3 10 / 10, Task 2-3 10 / 10 so far.

**Task 5 failure clusters, student under noise v1, mirror, ten cases** (guard on, 20 per variant): 5-2 7 failures, 4
with `i0_push_in` "structure" (> 10 mm) and 2 with the push-in verdict failed (block short, structure intact); 5-3 7
failures, 6 with a push-in "structure" (upper slot `i0`, then `i1` insert or push-in). Cases: high_friction, combined,
uneven_weights, heavy, nominal. The push-in is the Task 5 failure point. An earlier abort alone would trade "home but
structure moved" (P still counted) for "short" (P lost); prevention needs a push-in that drags less (contact height and
line, pace under friction), then R5.2 can retry the short ones.

**The student's rule-based recovery node** (graph `recovery`, rules on rest estimates; `results/v1_1/student_recovery`,
noise v1, mirror, ten cases, 20 per cell): it fired once in 4-2 and twice in 4-3 (R4.1; all three episodes failed
anyway) and never in Task 5. Off vs on: 4-2 TSR 55 vs 35 %, LHPR 91 vs 90 %; 4-3 0 / 68 vs 0 / 70 %; 5-2 and 5-3
identical. The four 4-2 episodes that flipped had no recovery decision: separate-process divergence after the one
decision in batch 2 changed the physics history (the known +-2-4 / 20 noise of such A/Bs). It stays off.

**`push_settle`, second seed** (seed 9100, same setup; `results/v1_1/ab_task4/seed9100`): off 20.85 vs on 22.10 points
(4-2 TSR 62.5 -> 70.8 %, LHPR 88.9 -> 93.1 %; 4-3 LHPR 67.1 -> 71.3 %, TSR 0; 4-3 ODR 0.04 -> 0.17, tipped 16 -> 24,
on a block 43 -> 29). Both seeds (48 per cell): 4-2 TSR 24 / 48 -> 32 / 48, LHPR 86.8 -> 92.8 %; 4-3 LHPR 67.4 -> 72.7 %;
+2.1 points on simulator poses. The direction holds on both seeds; tipping in 4-3 does not (18 -> 10, 16 -> 24).

**Post-success camera loop, complete** (`results/v1_1/post_success/core_guard`; our renders -> kp_v2 -> student_v1's
switcher with the v1.1 core flags incl. the finished guard; Tasks 1-3's fixed cameras delivered unrotated and turned
back by the orientation choice; the mirror (to 210 s); the ten cases, 10 episodes per variant): **60 / 60** (33.23
points of 33.3); orientation rotated by correlation in 60 / 60 (winner 0.55-0.61 on our renders); confident estimates
p90 0.15-0.41 mm; goal nodes after the first success: 7 (Task 1-2), 0 elsewhere.

### `student_v1_1` frozen and served (2026-10-28 02:35)

- Bundle `results/bundles/student_v1_1` (from `student_v1`: same switcher bc_r2, skills, kp_v2; + `perception/
  orient_refs.npz`; graphs + `{"idle_chunk": 16, "blind_guard": true, "finished_guard": true, "push_settle": true}`),
  73 files; tar sha256 **`d10a69c557d42bf9faaa63e89e1793f8f81169017128584fda59c2727078ec93`** (the artifact identity).
- Endpoint `official_v1_1`: code snapshot `tower_rl_snapshots/server_v1_1`, port 9002, tmux session `tower_v1_1`
  (windows server, tunnel), policy id `tower-student-v1-1`, plugin version 2, camera source evaluator. Model URL
  **`wss://assignment-miss-luke-documents.trycloudflare.com`** (tunnel metrics 127.0.0.1:20242). Started by
  `results/v1_1/finalize_v1_1.sh`; the qualification-like probe passed 10 / 10 locally and through the tunnel
  (`results/v1_1/official_v1_1_probe_{local,tunnel}.json`). `official_1` (port 9001, session `tower`) unchanged.

**End to end through the v1.1 server** (`results/v1_1/gpu_ws_e2e.sh`: `student_v1_1_rc2` = the frozen contents,
snapshot `server_v1_1`, camera source render, `ws_loop.py --official_orient --mirror`, cases nominal + lighting, 2
episodes each): Task 1-2 2 / 2 (orientation rotated by correlation; 365 calls per 210 s episode with the 16-row idle
holds; **the finished guard vetoed 64 post-completion reworks per episode**), Task 4-3 0 / 2 (LHPR 50 %), Task 5-3 1 / 2
(LHPR 75 %); in all six episodes no adapter error, no hold, no clipped value, never blind; server time per call p99
0.3-1.5 s with the camera loop sharing the machine.

**4-3 follow-through with a raised pusher** (`push_settle.follow_raise_m` 0.012; stopped after 8 episodes): LHPR
0.58-0.65 and ODR 0.61-0.63 (plain push_settle 0.74 / 0.06): raising the pinch 12 mm does not keep the stroke from
disturbing the tower. The follow-through is dropped; 4-3's landing zone needs another pusher geometry.

### The recovery node in the switcher: features v2 (session 7)

- **Round 0** (`results/student/collect_v2/r0`; `results/v1_1/cpu_collect_v2.sh`, snapshot `v2_collect`): oracle_state_v2
  acting, exploration 0.1 (never a recovery option), injection 0.15 incl. the short push (`--inject_set
  '{"push_off": {"stop_r_m": [0.125, 0.15], "short_frac": 0.35}}'`), noise v1, mirror, D1 with the ten cases, the v1.1
  flags, 120 per variant: 1,200 episodes, 25,288 decisions (labels: idle 10,528, park 4,668, acquire 4,594, deliver
  2,446, push_off 2,352, insert 372, push_home 260, **repush 68**), 156,133 progress rows, 86 % non-nominal. A control
  run of the same oracle without exploration / injection: Task 2-3 8 / 8.
- **bc_v2_r0:** val decision accuracy 0.956 (nominal 1.000, non-nominal 0.949), progress AUC 0.9986 (best epoch 20).
- **Round 1** (bc_v2_r0 acting, beta 0.5, injection 0.15, 60 per variant): 600 episodes, 9,984 decisions (repush 30),
  51,895 progress rows.
- **bc_v2_r1** (rounds 0 + 1): val 0.965 (nominal 0.9985, non-nominal 0.957), progress AUC 0.9992 (best epoch 19). For
  scale, bc_r2 (features v1, 6,000 episodes): 0.955 / 0.9996 / 0.936, AUC 0.997 (other validation sets).

**Closed loop, v1 vs v1.1 vs the v2 router** (`results/v1_1/v2_eval`; noise v1, mirror, ten cases, seed 7700, 12 per
variant, the same seeds for all three):

| | Tasks 1-3 | Task 4 (4-2 TSR / LHPR, 4-3 LHPR) | Task 5 (5-2, 5-3 TSR) | points |
|---|---|---|---|---|
| `student_v1` as submitted (no flags) | 22.94 (TSR 0 in five variants) | 19.89 (50 / 88.9 %, 65.7 %) | 26.08 (67, 67 %) | **68.90** |
| `student_v1_1` (bc_r2 + v1.1 flags) | 32.41 | 20.02 (41.7 / 91.7 %, 68.5 %) | 26.08 (67, 67 %) | **78.50** |
| bc_v2_r1 + v1.1 flags | 32.46 | 19.75 (41.7 / 88.9 %, 68.5 %) | 27.42 (67, 83 %) | **79.62** |

v1.1 is +9.6 points over v1 on this mirror (almost all Tasks 1-3: the finished guard). The distilled router with the
recovery features holds bc_r2's level with a fifth of the data (+1.1 points; 5-3 10 / 12 vs 8 / 12, within the noise of
12-episode cells). DAgger round 2 (bc_v2_r1 acting alone) queued: `results/v1_1/cpu_v2_round2.sh`.

**Task 4 camera check: `push_settle` loses on 4-2 under perception** (`results/v1_1/t4_camera/settle/{on,off}`; our
renders -> kp_v2 -> student_v1's switcher, the v1.1 core flags, mirror, ten cases, seed 7400, 10 per arm): 4-2 with it
TSR 4 / 10, LHPR 81.7 %, 11.27 points; without it TSR 6 / 10, LHPR 90.0 %, 13.28 points. With it, 5 pushes ended
leaning on the base (2 without) and 2 off the table: the hold at the radial test stops the block's momentum, and the
2 mm footprint confirmation trusts the estimate at the base edge, where it is a few mm off; on simulator poses
(truth) the same rule removed every leaning block. 4-3 with it: LHPR 71.1 % (control arm running). Decision for
`student_v1_1`: no `push_settle` on 4-2; 4-3 by its control arm.
- **Round 2** (bc_v2_r1 acting alone, beta 0, injection 0.15, 60 per variant): 600 episodes, 9,919 decisions (repush
  labels 49). **bc_v2_r2** (rounds 0-2): val 0.964 (nominal 1.000, non-nominal 0.954), AUC 0.9986 (best epoch 4).
  Closed loop (same seeds): **77.67** points (5-2 TSR 58 %, 5-3 67 %; the rest as bc_v2_r1). bc_r2 78.50, bc_v2_r1
  79.62, bc_v2_r2 77.67: parity within the noise of 12-episode cells; the third round added nothing measurable.

**4-3 camera check:** with `push_settle` LHPR 71.1 % (7.89 points), without 66.7 % (7.37), TSR 0 both; the direction of
both CPU seeds (+6.5, +4.2 LHPR). Decision: `push_settle` on 4-3 only.

### `student_v1_1` rebuilt and re-served (2026-10-28 05:23)

`results/v1_1/rebuild_v1_1.sh task4-3_full_tower_pushdown_3`: the first build (push_settle on both Task 4 variants)
kept as `student_v1_1_settle_all` (bundle, tar, `results/student/server/official_v1_1_settle_all/`, logs); the new
`student_v1_1` = student_v1 networks + `orient_refs.npz` + `{"idle_chunk": 16, "blind_guard": true,
"finished_guard": true}` on every graph + `{"push_settle": true}` on 4-3 only (freezer `--graph_set_variant`). Tar
sha256 **`41fddf485c156ce6bc20666db5098258715810ff758d006d0dd62adf85dc218a`**. Endpoint `official_v1_1` (port 9002,
tmux `tower_v1_1`, tunnel metrics 20242), model URL **`wss://epson-korea-analog-patients.trycloudflare.com`**; the
qualification-like probe passed 10 / 10 locally and through the tunnel. Nobody had qualified the first URL.

### The features-v2 router in the camera loop: not passed; the re-push bug it exposed (2026-10-28 05:27-06:48)

`results/v1_1/gpu_v2_camera.sh` (snapshot `server_v1_1`): bc_v2_r1 with student_v1_1's configuration (push_settle on
4-3 only), our renders -> kp_v2, mirror, ten cases, seed 7400, 10 per variant, against the bc_r2 arms of the same
seeds (`t4_camera/settle/off` for 4-2, `settle/on` for 4-3). `results/v1_1/t4_camera/v2_r1_points.md`.

| | 4-2 TSR / LHPR / points | 4-3 LHPR / points | Task 4 points |
|---|---|---|---|
| bc_r2 (student_v1_1) | 6 / 10, 90.0 %, 13.28 | 71.1 %, 7.89 | 21.17 |
| bc_v2_r1 | 2 / 10, 83.3 %, 10.33 | 71.1 %, 7.84 | 18.18 |

- **Run-to-run noise is large:** in 6 of the ten 4-2 episodes both routers chose the same node sequence, and still
  bc_r2 succeeded in 4 of them and bc_v2_r1 in 2 (heavy: 3 blocks tipped vs none, same decisions). The camera loop is
  not deterministic (rendering, estimates), and Task 4 tipping amplifies a few mm. 10-episode camera A/Bs of Task 4
  resolve only large effects (this also qualifies the 4-2 push_settle decision, which kept v1's behaviour).
- **The v2 router's re-pushes did nothing:** it chose `repush` in 3 of the 10 4-2 episodes (4 nodes), and in two of
  them the block it chose was the episode's only failure at the end (lightweight p4, combined p2: both "on/leaning on
  base"). Every 4-2 re-push ended after 4 rows with along 0 mm and gap 3.3-4.8 mm, and one second attempt timed out
  in its transit (15 s). Cause: the re-push starts on a block that is already radially out, so the push_settle logic
  went straight to hold-and-confirm on its first row, and the per-row estimate put the block 3-5 mm clear (R4.1 had
  judged it leaning on the rest estimates: gap < 2 mm, near end raised); the gap alone was confirmed. In the CPU noise
  model the same node pushed 37-94 mm (the noise model does not reproduce the estimate's offset at the base edge). In
  4-3 (push_settle on) its two re-pushes moved the blocks 51-66 mm.
- **Fix (opt-in graph flag `repush_v2`, `REPUSH_V2` in `executor.py`, re-push nodes only):** the planned push first
  runs `min_push_m` 0.04 past its contact pose, before any departure test or confirmation; the block is confirmed only
  "flat clear" (`mirror.rest_state` on the per-row estimate); not flat within `near_m` 0.03 of the footprint counts as
  still over it (finished by the planned push); no finishing past `max_total_m` 0.15. Unit test
  `test_repush_v2_pushes_its_stroke_and_confirms_flat` reproduces the camera-loop record with the flag off (4 rows,
  along 0, gap 3.7 mm) and checks the stroke and the cap with it on. Full suite passes. Camera check with the flag
  (snapshot `v2_repush`, `t4_camera/v2_r1_repush2`) started 06:48.

### `repush_v2` in the camera loop, and the cause of 4-2's leaning blocks: coasted estimates (2026-10-28 06:48-08:30)

`t4_camera/v2_r1_repush2` (bc_v2_r1 + `repush_v2`, same seeds): 4-2 TSR 3 / 10, LHPR 78.3 %, 10.30 points; 4-3 LHPR
66.7 %, 7.41 points; Task 4 17.71 (bc_v2_r1 18.18, bc_r2 21.17). The flag only changes re-push nodes, and there was one
re-push in the 20 episodes (plus a re-push transit that timed out in 4-2 combined), so the totals are the camera loop's
noise (4-2 nominal: same decisions as the bc_v2_r1 run, success there, failure here).

The one re-push (4-2 lightweight, p5 = `source_l1_0`) pushed its whole stroke (along 143.6 mm, "unconfirmed") and the
block still ended leaning on the base. Perception log (`perception_log.pkl`, seed 7400001):
- p5's own push: the block stopped at x = 535 mm leaning on the base edge (z 34.6 mm, tilt 15.8 deg, true r 127 mm)
  while the camera lost it (conf 0.3 from 37.4 s); the estimator **coasted** it on outward (548 -> 640 mm, decaying
  x 0.8 per call, `coast_calls` 5) and the push node's radial test fired on that coasted pose (along 148 mm): "ok".
- 38.0 s: re-found (conf 1.0) at 535 mm. The estimator took the velocity from that correction (-105 mm, capped at
  `max_step_m` 40 mm per call), lost the block again, and coasted it 145 mm back towards the tower (390 mm, conf 0.3).
- R4.1 judged the block leaning on the rest estimate (confident samples only, correct), but the re-push node aims at
  the per-row estimate: phase 0 went to the contact pose behind the coasted 390 mm pose (radial_along -144 mm), and the
  0.15 m stroke from there ended about where the block's near end was. The block never moved (true x 535.0 throughout).

All 80 camera-loop Task 4 episodes (`settle/off`, `settle/on`, `v2_r1`, `v2_r1_repush2`), every push that ended "ok":
the row where its estimate first passed `push.departed`, and the truth at that row
(`scripts/radial_truth.py`):

| | estimate confident there | estimate not confident (conf < 0.5) |
|---|---|---|
| 4-2: truly departed | 188 / 189 | 23 / 36 |
| 4-2: final "on/leaning on base" | 2 / 189 | **10 / 36** (all 10 not truly departed, true r 107-157 mm) |
| 4-3: truly departed | 271 / 280 | 17 / 35 |

(planned pushes; `--re` adds the re-pushes). So in 4-2, 10 of the 12 blocks that ended leaning on the base after
their push were pushes ended by the radial test on a coasted estimate, the block still at the base edge. The learned push was trained to push through such losses: its training
env (`push_env.noisy_block_poses`, Step 9a) shows it the same coasted pose with `seen` = 0 and ends the episode on the
**true** departure; deployment ended it on the coasted one.

**Fixes (opt-in, `executor.py`; snapshot `tower_rl_snapshots/depart_seen`):**
- Graph flag **`depart_seen`** (`PushNode`): the departure test counts only rows where the target is seen
  (confidence >= `CONF_MIN` 0.5; `self.lost_rows > 0` means this row's pose is coasted or stale). The learned push
  (67-D, seen inputs) then pushes on through the loss as in training; a target lost > 15 rows after it was last seen
  >= 60 mm along still ends the push (`lost_departed`, unchanged). With push_settle (4-3) the settle logic starts only
  on a confident departure as well.
- Under **`repush_v2`**, the re-push acts on the target's last confident pose (the rest estimate at its start) on rows
  where the target is not seen, instead of the coasted / stale per-row pose.
- Tests: `tests/test_push_node.py` (a still 67-D actor on a static scene: without the flag a coasted departed pose ends
  the push within 6 rows; with it the node runs on until a confident departed row, or ends `lost_departed` after
  > 15 lost rows once seen 80 mm out), `test_repush_v2_aims_at_the_last_confident_pose` (coasted -145 mm: the stroke
  starts at the true contact pose; with the substitution disabled it starts at -145 mm, the camera-loop record).
  Full suite passes.
- The estimator's coasting (velocity from a correction jump) is left as is: it changes perception for every task, and
  the push actor was trained with exactly this coasting (`push_env` replays it).

**Camera A/B started 08:17** under `scripts/eval_guard.py --name official_v1_1` (pauses on an evaluation session):
`SNAP=.../depart_seen ADD='"depart_seen": true' results/v1_1/gpu_v2_camera.sh <student_v1 switcher (bc_r2)>
bc_r2_seen 10`, i.e. student_v1_1's configuration + `depart_seen`, the same seeds as `t4_camera/settle/off` (4-2) and
`settle/on` (4-3). Output `results/v1_1/t4_camera/bc_r2_seen`.

**The same in official run 1** (`scripts/official/push_end_conf.py`: the server's per-call logs run push nodes one row
per call, so call `steps` of a push's window is the row its departure test fired; rest states from the last call's
estimates, the rule that reproduced the site's LHPR):

| official run 1, pushes that ended "ok" | ended on a confident estimate | ended on an unconfident one |
|---|---|---|
| 4-2 | 42: flat clear 36 (86 %), tipped 4, on a block 2 | **16: flat clear 7 (44 %), leaning on base 3, tipped 1, unseen 5** |
| 4-3 | 68: flat clear 40, on a block 16, tipped 6, other 6 | 9: flat clear 3, unseen 6 |

All three official 4-2 blocks left leaning on the base came from pushes that ended on an unconfident estimate; a
quarter of 4-2's pushes ended that way.

**`depart_seen` camera A/B, 4-2 (`t4_camera/bc_r2_seen`): not adopted.** bc_r2 + `depart_seen` vs `settle/off` (same
seeds): TSR 3 vs 6 / 10, rest states flat 50 vs 54 / 60, leaning 1 vs 2, tipped 4 vs 3, touching 2 vs 1, **off the
table 3 vs 0**; push timeouts 2 vs 0 (`scripts/t4_camera_cmp.py`). `scripts/depart_counterfactual.py`: in 13 pushes
the v1.1 rule would have ended the push on an unconfident row; 2 of those blocks had not departed (r 142, 160 mm) and
with the flag ended flat clear (the intended effect); 11 had truly departed (the coasted pose was right), and pushing
on cost one of them: high_friction p2 was shoved from 210 to 315 mm and off the table edge during the 15 lost rows
before `lost_departed` (the actor never saw post-departure states in training). The other off-table / timeout cases
were not the flag's (slippery p3 ended on a confident sighting at 233 mm and slid off later; high_friction p5 was
flung off mid-push with a stale, not coasted, estimate at 63 mm). Pushing on blind cannot tell a block that left
from one that stuck: the fix has to look before it pushes again. The 4-3 half was stopped at 08:50 (no decision
depends on it). The flag stays in the code, off.

Next: the recovery node's R4.1 does exactly "look, then push again" (confident rest estimates, then the planned
re-push), and graph `recovery` runs it in the student runtime with any router. Its gate `_static_seen` needs every
call in the last 1.0 s confident and >= 2 calls, but with `idle_chunk` 16 the calls after the plan come every 1.6 s, so
it can never pass in the idle phase (likely why it fired once in 20 4-2 episodes in the noise model).

### The student's recovery node, gate for sporadic sightings (2026-10-28 08:55-09:06)

- Camera log of the 4-2 lightweight episode (v2_r1_repush2): during the idle phase after the last push (calls 1.45 s
  apart) the leaning block was confidently seen on 4 of ~30 calls, never two in a row; in between conf 0.3 (coasted)
  or 0 (ghost). `RecoveryNode._static_seen` (every call of the last `static_s` 1.0 s confident, >= 2 calls) can never
  pass at that call rate, and `Contracts.check` also needs the decision call itself confident.
- **`RecoveryNode(seen_s=...)`** (estimate source; graph `recovery: {"seen_s": 15.0}` passes the options through, plain
  `true` keeps the old gate): a block counts as seen if it had a confident sighting within `seen_s` within 4 mm of its
  rest estimate; "at rest" = >= 2 confident sightings within `seen_s`, all within 4 mm of the rest estimate (the
  unconfident calls are ignored, not vetoes). History kept max(3 s, `seen_s`). 15 s because the sightings in that
  episode were ~11.6 s apart.
- Test `test_rules_seen_s_acts_on_sporadic_sightings` (idle calls 1.6 s apart, two sightings 11.2 s apart, coasted
  145 mm back in between): the old gate does not fire, `seen_s` 15 fires R4.1, one sighting or two that disagree by
  10 mm do not. Full suite passes.
- Camera mechanism test started 09:06 (snapshot `recovery_v12`, `results/v1_2/gpu_recovery_camera.sh`, `INJ=1`: 30 %
  of the pushes stopped short at 0.125-0.15 m so R4.1 has work): bc_r2 + v1.1 core + `{"recovery": {"seen_s": 15.0},
  "repush_v2": true}`, 4-2, 10 episodes, seed 7400 -> `t4_camera/rec_inj`; per event `scripts/recovery_events.py`.

**Mechanism test result (`t4_camera/rec_inj`, 09:06-09:32; `scripts/recovery_events.py rec_inj 4-2`):** 24 pushes
stopped short by injection in 10 episodes (TSR 0 throughout: a block left short breaks the push order once the next
one departs, so this run measures only the mechanism).
- R4.1 fired 3 times, each on a block truly "on/leaning on base" at that moment (no false positive), and **all three
  re-pushed blocks ended flat clear**. The two made at the plan's end (nominal goal 1 at 45 s, uneven_weights goal 4
  at 48 s) pushed 28.7 mm and were confirmed 15-16 mm clear of the footprint.
- The third (heavy, goal 0 = `source_l2_2`, at 21.8 s after layer 2 with layer 1 still on the tower): the block had
  fallen against the end of layer-1 block `source_l1_2` (z 47.5 mm, tilt 28 deg). The re-push came in over layer 1 at
  the leaning block's height: its transit disturbed a block by 9.6 mm and the push failed "structure" (`l1_1` lifted
  44 -> 52 mm, `l1_2` shifted 22 mm). R4.1's "no higher-layer block on the tower" does not cover a lower layer under the
  re-push's path.
- 5 more leaning states were refused by R4.1's preconditions (e.g. pushed < 0.10 m). Of the 24 injected blocks, the
  3 re-pushed ended flat clear; of the 21 never pushed again, 14 settled flat clear on their own, touching base 3,
  leaning 1, tipped 1, on a block 1, off the table 1.

**`RecoveryNode(at_end=True)`** (graph `recovery: {..., "at_end": true}`): R4.1 only once every goal was pushed, no
block is "on tower" and the base decider idles or parks. Test `test_rules_at_end_waits_for_an_empty_tower` (the heavy
state: the old rule re-pushes, `at_end` waits; plan done + idle: re-push). Full suite passes. The mirror's TSR needs
the pushes in order, so a re-push at the end rescues the block's rest state (LHPR, 2/3 of the score) and the TSR only
for the last push; LHPR is where official 4-2 lost 3 blocks to leaning.

**No-injection camera A/B started 09:38** (snapshot `recovery_v12b`): bc_r2 + v1.1 + `{"recovery": {"seen_s": 15.0,
"at_end": true}, "repush_v2": true}`, 4-2 then 4-3, 10 each, seed 7400 -> `t4_camera/rec_end`, against `settle/off`
(4-2) and `settle/on` (4-3).

**Official run 1, replayed for R4.1 (`scripts/official/repush_candidates.py`, server logs, estimates only):** the
blocks whose last confident estimate rests leaning on the base and that pass R4.1's preconditions with the `seen_s`
gate: 4-2 four (06aba879 goal 5, 2c198e9f goals 0 and 4, e820425f goal 5; pushed 111-156 mm, near ends 58-73 mm,
seen at that pose on 3-1,675 calls after the plan), 4-3 one (d08100d8 goal 8). In 06aba879 the leaning block was the
last push and every other block was flat clear with no push failure: a re-push that works turns that episode into a
success. If each re-push lands flat clear: 4-2 LHPR 44 -> 48 / 60 and TSR 0 -> 1 / 10, 4-3 LHPR + 1 / 90, about +1.3
points at the mirror's weights (points per variant = score / 6, score ~ 100 (TSR / 3 + 2 LHPR / 3)). Several other
leaning blocks were never confidently seen after their push (no rule can act on those).

**No-injection A/B, 4-2 (`t4_camera/rec_end` vs `settle/off`, same seeds):** TSR 5 vs 6 / 10, flat clear 51 vs 54 / 60,
tipped 5 vs 3, leaning 2 vs 2, on a block 1 vs 0; push-node failures 1 vs 4. R4.1 fired once (combined, goal 5 at
53.7 s, the plan done): the block was truly leaning; the re-push pushed 33.6 mm and confirmed it 35 mm clear; it
ended flat clear. No other episode had a recovery decision, so their differences are the camera loop's run-to-run
noise (e.g. lightweight: same decisions, goal 1 tipped here and not there). End-of-plan re-pushes so far: 3 / 3 flat
clear, no collateral.
- Missed: irregular_layers, where the switcher skipped push 3 (sequence 0 1 2 4 5) and goal 5 ended leaning; `at_end`
  required every goal pushed. **Refined:** every goal pushed *or its block confidently seen off the tower* (an unseen
  block still blocks it), none on the tower, the decider idling. Test extended (skipped goal off the tower: re-push;
  still standing on the tower: none). Full suite passes. (The 4-3 half of this A/B runs the stricter version; they
  differ only when the switcher skips a goal.)
- Server episode summaries (`tower_rl/server/student.py::_summary`) now carry the recovery node's decisions and
  abstentions (`recovery`, from `episode_extra`) when graph `recovery` is on, for the analysis of a v1.2 official
  run (inside `_write_summary`, which never raises). Server / adapter tests pass. Served v1.1 runs from its own
  snapshot and is unaffected.
- Prepared (not run): `results/v1_2/build_v1_2.sh` (freeze `student_v1_2` = v1.1 + Task 4 flags `T4`, default
  `{"recovery": {"seen_s": 15.0, "at_end": true}, "repush_v2": true}`; bundle check against v1.1; serve on 9003 /
  `tower_v1_2`; probes) and `results/v1_2/gpu_ws_e2e_v1_2.sh` (its server on 9004 with our renders, ws_loop on 4-2 /
  4-3, nominal + combined, 2 each).

**No-injection A/B, 4-3 (`rec_end` vs `settle/on`):** R4.1 made **no** decision in 4-3 (its abstentions: 2 leaning
blocks failing its preconditions, the rest on a block / tipped / unseen), so this arm behaved exactly as v1.1 (code
checked: the `View` is a plain container; the rule's `observe` / `propose` only read estimates and log). Yet LHPR 56.7
vs 71.1 %, tipped 19 vs 9, structure failures 28 vs 9: four episodes (slippery, high_friction, shifted_assembly,
combined) cascaded from a mid-plan structure failure, as other arms do in other episodes. **This is the camera loop's
run-to-run noise for 10-episode 4-3 cells: ~14 pp LHPR between identical configurations** (`mirror_points`: Task 4
18.52 vs 21.17 with one recovery decision in 20 episodes). Such A/Bs cannot measure a ~1-point effect; the evidence
for v1.2 is the mechanism (end-of-plan re-pushes 3 / 3 flat clear, 4 / 4 R4.1 decisions on truly leaning blocks), the
official replay (5 eligible blocks, one TSR flip) and its construction (acts only at the plan's end, on a block seen
leaning; nothing else changes).

**`student_v1_2` frozen (11:00):** `student_v1_1` + Task 4 graphs `{"recovery": {"seen_s": 15.0, "at_end": true},
"repush_v2": {"max_total_m": 0.1}}` (the re-push stroke capped at 0.10 m: the successful ones were confirmed after
29-34 mm, and an unconfirmed one should not sweep 0.15 m through the debris; test extended). Tar sha256
**`f726c71a453da23943757cf9fb8506d101f83b196b22770f966073a5c3279aa9`**; the bundle check passed (networks /
perception byte-identical to v1.1; graphs differ only by those flags on Task 4). Server snapshot `server_v1_2` (= repo
code). A first build with the default 0.15 m cap is kept as `student_v1_2_cap015` (its end-to-end run was stopped
after 2 min to test the final bundle instead). End to end through its server (port 9004, our renders, 4-2 / 4-3,
nominal + combined): started 10:55.

**`student_v1_2` end to end and served (10:55-11:10).** Through its server (port 9004, our renders, `ws_loop.py`, the
mirror, 4-2 / 4-3 x nominal + combined): 4 episodes to 210 s, 0 errors, call latency p50 ~145 ms / p99 1.0-1.25 s;
4-2 TSR 1 / 2, LHPR 83 %; 4-3 LHPR 67 %. The episode summaries carry the recovery node's log (source "estimate"): no
decision (4-2 had no leaning block; in 4-3 one leaning block failed R4.1's preconditions, logged). Then
`results/v1_2/build_v1_2.sh`: endpoint `official_v1_2`, port 9003, tmux `tower_v1_2`, tunnel metrics 127.0.0.1:20243,
policy id `tower-student-v1-2` (plugin version 2), artifact identity `sha256:f726c71a...9aa9`, model URL
**`wss://windsor-lighting-lean-iii.trycloudflare.com`**; the qualification-like probe passed 10 / 10 locally and
through the tunnel. `official_1` (9001) and `official_v1_1` (9002) untouched and healthy (0 errors, tunnels connected).

**`student_v1_2` queued (2026-10-28 15:31).** The user saved `wss://windsor-lighting-lean-iii.trycloudflare.com` and
ran Check endpoint: ten sessions 15:31:02-15:31:59 on `official_v1_2`, one per variant (binding
`20261013-v1-task45-10hz-reconnect`), each reset seed 0 + one blank-image call, 0 errors (official_1's qualification
pattern). Evaluation queued (all ten, 10 Hz). Watcher armed as a one-shot (`WATCH_NAME=official_v1_2
scripts/official/watch_official.py 2026-10-28T15:33:00 600 600 first`: exits at the evaluation's first session or any
endpoint event). All three endpoints stay up (GPU 6.5 / 24.5 GB). Official run 1 waited ~48 h in the queue (queued
10-24 11:20, started 10-26 11:56).

**`student_v1_2` official evaluation started (2026-10-28 19:55:24, ~4.4 h after the check).** First session
task1-2 (binding `20261013-v1-task45-10hz-reconnect`); first calls ~20:00. First two task1-2 episodes: orientation
`rotate` with correlation rot180 0.929 vs same -0.073 (the official renders' 180° turn, as in run 1's frames), `blind`
false, 10 / 10 blocks confident on the first and the latest call, call latency p50 ~100 ms / p99 0.7-1.0 s. Watcher
re-armed as a 30-min one-shot (`watch_official.py 2026-10-28T19:55:00 1800 600 first`, `WATCH_QUIET_START=1`);
live per-episode view: `python3 scripts/official/live_episodes.py "2026-10-28 19:55:00" official_v1_2`.

**v1.2 official, Task 1-2 wave (19:55-20:44).** 12 sessions: 8 episodes to 210 s, 4 empty (opened and reset, closed
without a call: worker restarts, as in run 1). Every episode: orientation `rotate` (rot180 0.921-0.929, same ~-0.07),
10 / 10 blocks confident throughout, 0 errors, call p50 ~95 ms / p99 0.27-0.63 s. Every plan completed by ~85 s with
all 18 nodes ok (4 transfers + parks); runtime place errors (our estimates) 0.2-4.7 mm (worst: c074af63 t3 4.74 mm,
3c9493a9 t2 4.30 mm; tolerance 5 mm), yaw <= 2.6 deg, placed blocks static to the end. After the plan the switcher
asked to re-acquire goal 1 at every decision from ~85 s to 209 s (63 times per episode; goal 1's place error is the
largest, 2.2-3.7 mm): all vetoed by the finished guard, as designed (Student-v1.1.MD §0 item 1). The server then idled
(no session from 20:44); run 1 also paused between variants.

**v1.2 official, Task 1-3 wave (21:09-22:05; one episode still running).** Orientation `rotate` (rot180 0.977-0.996),
12 / 12 blocks confident, 0 errors. Of 6 finished episodes, 3 completed every place within 0.5-3.3 mm (our
estimates); 3 have a failed place verdict:
- 584b1314 and 7a9650ef: **t4_place 8.96 / 7.19 mm** (tolerance 5 mm). Both followed a t4 grasp with a large in-hand
  offset (grasp_offset_mm x -5.5 / -6.5; the good episodes' t4 grasps: |x| <= 1 mm). The switcher then went on to t5
  and idled to 210 s without reworking t4 (a top-layer block, so a re-place was possible).
- 588e8a2a: **t1_place 5.08 mm**; at 141 s the switcher went back to t1, a bottom-layer block under two layers, and
  `t1_grasp` failed "dropped" at 147 s.
- 1b58c727 (still running at 189 s): plan done at 131 s, then the switcher reworked t2 (a middle-layer block) at
  133-151 s, then t3, then t4 twice (grasp at 170 s and 189 s). Its summary is not written yet.
New failure modes vs the local camera loop (Tasks 1-3 60 / 60): a grasp offset not compensated at the place, the
switcher not reworking a misplaced accessible block, and reworks of blocks with blocks on top of them.

**The Task 1-3 collapse (1b58c727, the user's video "Task1-3 Repeat 7", 22:30).** Goals 0-3 placed 1.4-2.6 mm. The
left arm gripped goal 4 (`source_l1_a`) 5.8 mm off-centre along the block (z -3.2 mm); the carry touched nothing
(max_block_disp 0.12 mm); while setting it on layer 2 at 106.8-107.2 s it hit the layer below: the two layer-2
blocks moved 65-76 mm, a layer-1 block 19 mm; `t4_place` failed (23 mm, 7°, 38 steps vs 17-19). Goal 5 then placed
"ok" (2.95 mm). From 133 s the v1 switcher (no recovery node on Tasks 1-3 in v1.2; the finished guard allowed it,
blocks had moved) went back: goal 2 re-gripped at 10° tilt and set down 18.5 mm off, goal 3 "dropped", goal 4 two
15-s grasp timeouts (goal_adjust 29.7 mm / 8°: the block far from its planned pose). Frame 300 (107.0 s) shows the
tower falling. The same goal-4 grip (x -5.5 / -6.5 / -5.8 mm) is behind all three Task 1-3 place failures so far
(7.2, 9.0, 23 mm); the other four episodes gripped it within 2.3 mm and placed it within 1.5-2.2 mm. Locally the
place absorbs such grips (v1.1 camera loop: 5 of 60 Task 1-3 grips >= 4 mm, max 5.3, no place > 5 mm; the place
skill trains with offsets up to ±10 mm): why it does not here (a wrong offset estimate on the official images, or
slip after the grip) is open.

**v1.2 official, Task 1-3 complete (10 episodes, 23:04).** By our estimates 6 / 10 clean (every place <= 4.2 mm);
the 4 others: 584b1314 and 7a9650ef goal 4 misplaced 9.0 / 7.2 mm, 1b58c727 the collapse, 588e8a2a goal 1 at 5.08 mm
then the re-grip of that buried block ("dropped"). Goal 4 gripped >= 4 mm off-centre only in the three goal-4
failures. Task 2-2 under way: 5 episodes, every place <= 3.7 mm so far; two ended before 210 s (evaluator-side, as
in run 1).

**v1.2 official, Task 2-2 complete (10 episodes, 22:48-23:31).** Orientation `rotate` (rot180 0.993-0.997), 8 / 8
blocks confident, 0 errors. 9 / 10 clean: both places <= 3.7 mm (our estimates), plan done by ~46 s, then the
finished guard vetoed ~83 re-acquire requests per episode, as in Task 1-2. The failure, **41780038, was not caused by
the robot**: its start was an irregular-layers case with the top block (`source_l2_center`, a single block resting
on the single layer-1 block) 20 mm off its support along x (layer 1 at (+6, -7) mm, layer 2 at (-14, -2) mm from
nominal). At the first call (t 0.0) it was already tilted 2.3°; by the second call (1.6 s, the left arm still on its
way) it lay on the table beside the tower (x -103 mm, z 32.5: leaning on the foundation). The student grasped it
there (offset (19.0, 0.3, 17.5) mm, 8.4°), set it on the goal 8.0 mm / 5.4° off, re-gripped it twice from the goal
(offsets 18.4 / 21.6 mm along the block) and set it down 10.0 then 12.4 mm off, parked, delivered goal 1 (1.7 mm)
and idled from 66 s. Why the re-grips off the goal stay ~20 mm off-centre along the block is open (the place
absorbs up to ~10 mm, its training range).

**The evaluator's initial states (all v1.2 episodes so far).** Three of ~10 episodes per variant start perturbed;
the others start within 0.4 mm of the per-variant median: a rigid shift of every movable block with one shared yaw
(1-2 6aee6291 -7.2°; 1-3 7a9650ef -7.5°, up to 15 mm; 2-2 920eadc2 (-11, -11) mm, +4.5°: shifted assembly or
combined), and per-layer shifts and yaws (2-2 41780038 above; 2-3 d97f2ef3 layer 2 (-15, -5) mm, layer 3 (-22, -6)
mm, layer 1 nominal; 2-3 92114e83 and 1-3 505c7e28 layers (+5..+7, -8..+10) mm with +3.4..+4.5° on single layers:
irregular layers). Our D1 cases (`configs/tiers.yaml`): shifted ±15 mm / ±10°, irregular ±8 mm / ±5° per layer on
top of the assembly shift, plus ±2 mm per-block jitter, so adjacent layers can differ by up to 16 (20 with jitter)
mm: the official cases sit at or past the edge of that range (20 mm between adjacent layers in 41780038, 22 mm
between layers 1 and 3 in d97f2ef3), where ours are rare corners of a uniform draw. Failures vs start: of the five
Tasks 1-3 failures so far, one started perturbed (1-3 7a9650ef; the collapse 1b58c727 started nominal); in 2-2 the
perturbation itself dropped the top block.

**The machine hard-reset during the v1.2 evaluation (2026-10-29 ~00:11); the site marked the evaluation failed.**
No shutdown record (`last -x`), nothing in syslog / the journal after 00:00:03 (a quiet system: no kernel panic,
Xid, MCE or thermal line survived), boot 00:11:33. The server's own log ran to 00:10:29: 8 sessions connected, 0
errors, mean call 95-99 ms, GPU 6.8 GB with the three endpoints. Cause unknown (power / hardware; the second hard
reset after 10-18 14:36). Before it: Tasks 1-2, 1-3, 2-2 complete (10 episodes each), 2-3 at 9 episodes in progress.
All logs survived (per-episode logs in `results/student/server/official_v1_2/episodes`); the scratchpad in /tmp did
not.

**`official_v1_2` re-served (2026-10-29 05:28).** Old endpoint logs moved to
`results/student/server/official_v1_2/attempt1_crash_20261029/` (server / tunnel logs, run.json, server.jsonl) so
the restart could not read the stale URL; bundle tar checksum OK (`f726c71a...9aa9`). Same bundle, snapshot
`server_v1_2`, port 9003, tmux `tower_v1_2`, policy id `tower-student-v1-2`; server pid 19830, READY in 2.0 s, GPU
2.1 GB. New model URL **`wss://setting-insights-perfume-lesson.trycloudflare.com`**; probes 10 / 10 locally (2.2 s)
and through the tunnel (11.4 s) (`results/v1_2/official_v1_2_probe_{local,tunnel}_restart1.json`). `official_1`
(9001) and `official_v1_1` (9002) not restarted: no evaluation uses them. Watcher re-armed one-shot from 05:30.

**Site check on the new URL (2026-10-29 06:03:42-06:06:32): passed.** One session per variant (task1-2 ... task5-3,
binding `20261013-v1-task45-10hz-reconnect`), each two resets (seed 0) with one blank-image call per trial, 0
errors; the same event pattern as the 10-28 15:31 check (10 open, 20 reset, 10 episode_end, 10 disconnect). Our 20
probe sessions from 05:28 expired at 06:03:42. Watcher armed one-shot from 06:07 for the evaluation's first session.

**`Plan-v1.3.MD` (corl_challenge, 2026-10-29 morning): what is hand-written, and the last submission.** Findings
behind it:
- Finished-guard vetoes, local noise-model mirror (`results/v1_1/v2_eval`, 120 episodes each): `bc_r2` (v1.2's router)
  4,321 (per episode 1-2 63, 1-3 0, 2-2 83, 2-3 71, 3-2 85, 3-3 76; Tasks 4-5 0), `bc_v2_r1` 2, `bc_v2_r2` 1. The
  official v1.2 counts match `bc_r2`'s exactly (1-2 63 x 9, 1-3 0 x 10, 2-2 up to 83, 2-3 71): the v2 routers,
  trained on 210 s episodes, learned post-success idling; the guard is a crutch for `bc_r2` only.
- Goal 4 of 1-3: block 6's pre-grasp estimate stable to 0.2 mm and identical in the seven nominal-start episodes
  (x 422.9-423.0), yet recorded grip offsets along the block -6.5 ... +2.0 mm; all >= 5.5 mm failed their place
  (8.96, 7.19, 23 mm), all <= 2.3 placed <= 2.2 mm. The held offset is one estimate at lift (`record_held_offset`),
  pre-compensated by `place_pinch` / `release_pose`. Open: H1 in-hand estimate error vs H2 real offset + release
  geometry at that goal (a CPU injection test separates them).
- Teacher gap: `oracle.py::_end_state` restarts a failed goal without `_accessible` (checked only when reopening
  closed goals); the student's out-of-order state (588e8a2a: goal 1 failed at 5.08 mm, reworked at 141 s under two
  layers) would be labelled "rework" by DAgger. Fix in the oracle, then retrain.
- CPU cycle timings (10-28 night): round-0 collection ~2 h (1,200 episodes, ten variants in parallel), BC 10-20 min,
  a DAgger round 30-60 min (600 episodes), a router evaluation ~10 min (120 episodes). Camera: Tasks 1-3
  post-success 60 episodes ~2.5-3 h, Task 4 20 episodes ~1 h.

**Goal-4 failures of 1-3 resolved: real off-centre grips, then an early release (2026-10-29 06:30-06:55).** From the
official per-call estimates (block 6, goal 4 = `source_l1_a`, left arm):
- During the place's descent the held block's along-axis error from the goal (reference: the mean final position of
  the seven ok places, which agree within 0.6 mm) was +0.6..+2.6 mm in the failing episodes too: the offset
  compensation worked, so the recorded offsets (-5.5 / -6.5 / -5.8 mm) were real. H1 (a wrong in-hand estimate) is
  out; the held offset comes from the attach at rest anyway (`attach`, `record_held_offset` keeps `att[1]`).
  Confidence was 1.0 at the lift in all ten episodes; the block moved 40-50 mm/s along its axis there in all ten.
- Then: 584b1314 one-call drop 40 mm (z 146 -> 105: released ~4.5 cm up; landed -6.7 mm along), 1b58c727 81 mm (the
  collapse), 7a9650ef a normal 12 mm release, then slid -2.0 -> -5.5 mm along its axis over 0.3 s. Every ok place in
  the four variants: largest one-call drop <= 21 mm (p95 16).
- All 122 official v1.2 places: grips >= 4 mm off-centre along the block failed 7 / 12 (incl. 2-2's fallen-block
  re-grips at 18-22 mm and 1b58c727's post-collapse rework), others 1 / 110 (588e8a2a's 5.08 mm). Some ~5.6 mm grips
  in 1-2 placed fine: a risk, not a certainty.
- The learned place opens whenever its policy says so (no release-height guard; the insert has
  `InsertEnv._grip_mask`).
Stepwise plan for v1.3: `corl_challenge/Studentv1_3_plan.MD` (tracks: A learned router `bc_v2_r3` with the oracle's
buried-goal fix and new injections; B planned place after grips >= 4 mm off-centre, `offcentre_place`; C camera
checks, freeze, serve). Cutoff: freeze by 10-30 08:00.

## 2026-10-29 (session 8): `student_v1_3` (plan: `corl_challenge/Studentv1_3_plan.MD`)

**Step 0-1 (07:10-08:00), code, all opt-in:**
- `scripts/eval_guard.py --no-pause`: CPU jobs keep running through an evaluation but still get their own systemd
  scope (memory cap, no swap), nice 10 and OOM score 1000. Needed because the re-served endpoint (pid 19830) and its
  tunnel share VS Code's cgroup with this Claude Code session (systemd-oomd kills whole cgroups). Smoke: own scope
  `guarded-<pid>.scope`, nice 10, oom_score_adj 1000.
- Teacher (`student/oracle.py`, graph `oracle_skip_buried`, read only by the state oracle): a failed goal whose block
  now has a block resting on it is closed, not re-acquired (in `_decide_state` and at the failure in `_end_state`).
- Injection (`student/inject.py`, `LearnedGraspNode`): `acquire: {grasp_along_m: [lo, hi], grasp_along_frac}`: the
  grasp policy sees the block shifted along its length, so it grips off-centre for real (attach and lift test keep the
  true pose); `goals: [...]` per option limits injection to those goals. Configs without the new keys draw the same
  random numbers as before (test).
- `configs/tiers.yaml`: tier `D1e` = D1's ten cases + `irregular_edge` {assembly 7 mm, layers 11 mm, 5°} (adjacent
  layers p50 11, p90 19, max 30 mm; >= 15 mm in 28 %, >= 20 mm in 6 %: the official 15-22 mm range in the tail; 15 mm
  per layer gave p90 26 / max 40 and was cut). D1 unchanged (its random case draws stay as they were).
- `student/tokens.py`: graph `router_repush: false` masks the router's repush (v1.3 Task 4 keeps v1.2's rule node).
- `graph/executor.py`: graph `offcentre_place: {"min_mm": m}` -> the planned place (`PlannedPlaceNode`: slow descent
  to the offset-compensated release pose, open there) when the held block is >= m mm off-centre along its length;
  then its verdict also checks height (as the learned one) and its info carries `offcentre`. Both place nodes now log
  `grip_along_mm` and `release_above_mm` (the block centre above its goal centre when the gripper opened). Helper
  `held_along_mm`.
- Tests: `test_state_mode_skip_buried_is_opt_in`, `test_grasp_along_and_goal_filter`, `test_old_configs_draw_as_before`,
  `test_router_repush_flag_masks_the_routers_repush`, `tests/test_executor_v1_3.py` (factory switch): 5 / 5. Baseline
  suite before the changes: 304 tests (3 skipped), exit 0.
- Per-variant switchers (insurance for Step 7, opt-in): a bundle graph's `"switcher"` key (a path inside the bundle)
  makes `StudentModels` load that switcher for the variant (`student/switcher.py::VariantDeciders`); the adapter asks
  `for_variant` before it builds the episode's runtime (the runtime takes the token version from the decider it
  gets). Freezer: `--graph_set_variant "<v>={\"switcher\": \"student/switcher_x.pt\"}"` + `--extra` for the .pt and
  its .json. Test `test_per_variant_decider_is_chosen_at_init`. Rehearsals take `--student` per variant as before.
- Full suite after these changes: 309 tests (304 + 5 new), exit 0 (the adapter test added later: adapter / server /
  switcher files pass). Scripts written: `results/v1_3/cpu_grip_repro.sh` (Steps 2-3), `cpu_collect_v3.sh` (Step 4),
  `cpu_v3_gates.sh` (Step 6), `gpu_v3_camera.sh` (Step 7), `scripts/grip_report.py` (places by grip offset; old
  noise-model run `v2_eval/student_v1_student`: grips 4-8 mm placed 11 / 11, so the CPU loop may not reproduce the
  official failure).

**Steps 2-3: off-centre grips on CPU (07:31-07:51; `results/v1_3/grip_repro`, `scripts/grip_report.py`).** bc_r2 +
v1.1 flags, noise model v1, mirror, ten cases, 1-3 / 1-2 / 2-3 x 12, seed 7800, same seeds per arm:

| Arm | TSR | LHPR | places 4-6 / 6-8 / >= 8 mm: truth ok | release p50 | points (3 variants) |
|---|---|---|---|---|---|
| base | 97.2 % | 100 % | 6 / 6, -, 1 / 1 | 10.8 mm | 16.45 |
| inject (every acquire gripped 4-8 mm off) | 80.6 % | 98.1 % | 42 / 42, 40 / 43, 9 / 12 | 10.4-12.5 mm | 15.32 |
| inject + `offcentre_place` | 97.2 % | 99.8 % | 50 / 51, 37 / 37, 18 / 18 (all planned) | 4.9 mm | 16.44 |
| `offcentre_place`, no injection | 100 % | 100 % | 8 / 8 (planned) | 10.9 / 4.9 mm | 16.62 |

- D1: partly reproduced. Off-centre grips >= 6 mm fail places on CPU (10 of 55), but no release was > 25 mm above the
  goal: the official early releases (40 / 81 mm drops) need official physics or renders; on CPU the failures are
  slides and misplacements.
- D2: passed. The planned place after grips >= 4 mm removed them (1 of 106 off-centre places failed, at 4-6 mm) and
  did no harm without injection (8 natural triggers, all ok). It releases ~5 mm above the goal (the learned place ~11
  mm). `offcentre_place: {"min_mm": 4.0}` goes into v1.3 (Tasks 1-3).
- Step 4 (DAgger round 3) was launched at 07:32 before D2, without the flag: its labels are the oracle's either way.
- Camera check of the flag started 07:52 (`results/v1_3/gpu_oc_camera.sh`: inject + flag, 1-3 / 1-2 x 6, under
  eval_guard with pausing): the planned release with camera estimates.
- **Camera check of `offcentre_place` (07:52-08:35, `results/v1_3/camera/oc_inject`):** our renders -> kp_v2, every
  acquire gripped 4-8 mm off-centre, bc_r2 + v1.1 flags + the flag, 1-3 / 1-2 x 6: TSR 12 / 12, LHPR 100 %, no
  disturbance; 37 off-centre places, all planned, all ok (release 4.9-5.2 mm above the goal), and 23 normal grips
  placed by the learned place ok. Not paused (no evaluation session).

**Step 4: DAgger round 3 collected (07:32-08:48, `results/student/collect_v2/r3`, `results/v1_3/cpu_collect_v3.sh`).**
`bc_v2_r2` acting with beta 0.5, labels `oracle_state_v2` + `oracle_skip_buried`, tier D1e (ten cases +
`irregular_edge`), injection 0.15 (short pushes; a third of acquire injections an off-centre grip 4-8 mm), noise
model v1, mirror, v1.1 flags (+ push_settle on Task 4), seed 8300, 120 per variant. 1,200 episodes, 20,051 decisions
(transfer 12,053 / pushdown 4,645 / insertion 3,353; non-nominal 57 / 78 / 65 %); labels acquire 3,299, deliver 2,435,
park 2,477, idle 8,979, push_off 2,112, repush 63, insert 368, push_home 318; sources switcher 9,316, oracle 9,400,
inject 1,293, recovery 42. Collection TSR (injected, half the student) 1-2 0.63, 1-3 0.50, 2-2 0.78, 2-3 0.63, 3-2
0.83, 3-3 0.68, 4-2 0.43, 4-3 0.03, 5-2 0.77, 5-3 0.48. Training `bc_v2_r3` on r0-r3 started 08:48, the CPU gates
follow automatically (`results/v1_3/cpu_train_gates.sh`; the injected cells run without `offcentre_place` so failed
places occur, REF and CAND on the same injected seeds).

**Step 5: `bc_v2_r3` trained (08:48-09:21, `results/student/models/bc_v2_r3`, log `results/v1_3/logs/train_bc_v2_r3.log`).**
Features v2, rounds r0-r3 (3,600 episodes, 424,682 rows, 65,242 decisions). Best epoch 7: val acc 0.968 (gate >=
0.965), nominal 0.999, non-nominal 0.955 (gate >= 0.95), progress AUC 0.9989 (gate >= 0.995): offline gate passed.
Later epochs overfit (loss 0.008, acc 0.958-0.965).

**Step 6: CPU gates (09:21-09:56, `results/v1_3/gates/r3`, `scripts/gates_compare.py`; run it with the env python, the
system python3 is < 3.10 and the rework report it spawns fails to import).** REF = v1.2 (bc_r2 + v1.2 flags), CAND =
bc_v2_r3 + v1.2 flags + `router_repush: false` on Task 4 + `offcentre_place` on Tasks 1-3; noise model v1, mirror,
ten cases, seeds 7700 / 7900 x 12 per variant, the same seeds per arm:

| variant | REF succ / LHPR / vetoes / points | CAND succ / LHPR / vetoes / points | diff |
|---|---|---|---|
| 1-2 | 24/24 100.0 1460 5.54 | 24/24 100.0 0 5.54 | +0.00 |
| 1-3 | 20/24 97.2 0 5.12 | 21/24 99.0 0 5.27 | +0.14 |
| 2-2 | 24/24 100.0 1887 5.54 | 24/24 100.0 0 5.54 | +0.00 |
| 2-3 | 21/24 97.9 1554 5.23 | 23/24 99.3 0 5.43 | +0.21 |
| 3-2 | 24/24 100.0 2040 5.54 | 24/24 100.0 0 5.54 | +0.00 |
| 3-3 | 24/24 100.0 1818 5.54 | 24/24 100.0 0 5.54 | +0.00 |
| 4-2 | 10/24 86.1 0 11.84 | 10/24 86.1 0 11.84 | +0.00 |
| 4-3 | 0/24 68.5 0 7.57 | 0/24 70.4 0 7.80 | +0.23 |
| 5-2 | 16/24 83.3 0 12.89 | 15/24 81.2 0 12.47 | -0.42 |
| 5-3 | 16/24 88.5 0 13.39 | 18/24 89.6 0 13.92 | +0.53 |
| total | 78.19 | 78.88 | +0.69 |

- D3 passed: CAND >= REF - 0.5 (+0.69); no Tasks 1-3 variant lost an episode (the 1-3 / 2-3 gains are
  `offcentre_place`'s: in these cells the routers otherwise act alike); Task 4 LHPR (= the per-push tally) equal on
  4-2, +1.9 pp on 4-3; Task 5 -1 / +2 episodes (allowed: at most 1 lost per variant); finished-guard vetoes 0 / 240
  episodes (REF 8,759); buried reworks 0.
- Injected cells (Tasks 1-3, every acquire 4-8 mm off-centre with p 0.3, no `offcentre_place`, seed 7700 x 12): REF
  and CAND identical outcomes (69 / 72, LHPR 98.9 %, 2 failed places each, none redone at once, 0 buried reworks);
  REF needed 4,209 vetoes for it, CAND 0. So the guard was all that kept bc_r2 from post-success reworks; bc_v2_r3
  routes the same way without it.
- Edge cell (CAND, tier D1e `irregular_edge`, 2-2 / 2-3 x 12): 20 / 24, LHPR 95.5 %, 3 failed places (grips 0-2 mm:
  the layers, not the grip), 1 redone at once, 0 buried reworks. REF on the same seeds (`results/v1_3/cpu_ref_edge.sh`,
  `gates/r3/REF_edge`): also 20 / 24, LHPR 95.5 %, 10.12 vs 10.13 points, but **1 buried rework** (2-2 seed 7800002 at
  50.8 s: bc_r2 re-acquired a failed goal with a block on it, as official 588e8a2a) and 0 failed places redone at once
  (CAND 1); disturbed non-targets 4 vs 3. Small cells, but the direction track A was built for.
- Camera reference for Task 5 (`results/v1_3/camera/ref/t5_ref`, bc_r2 + v1.1 flags, seed 7500 x 10): 5-2 7 / 10
  (LHPR 90.0), 5-3 3 / 10 (LHPR 77.5); 23.88 points.
- **Tasks 4-5, two more seeds (8100 / 8500 x 12, `results/v1_3/cpu_v3_gates_t45.sh`, 10:03-10:14), pooled with 7700 /
  7900 = 48 per variant:** 4-2 REF 18 / 48, LHPR 87.2, 11.71 pts vs CAND 19 / 48, 87.8, 11.89 (+0.19); 4-3 0 / 0, 70.4
  vs 68.5, 7.79 vs 7.58 (-0.21; the new seeds alone -0.64); 5-2 33 vs 32 / 48, 13.08 vs 12.87 (-0.21); 5-3 31 vs 34 /
  48, 13.22 vs 13.61 (+0.39). Tasks 4-5 together +0.16: parity.
- **Where the arms differ (node sequences, same seed and case):** 4-2 45 / 48 episodes identical; 5-2 47 / 48; 5-3 36
  / 48; 4-3 only 12 / 48, but there the first difference is nearly always the same node with another outcome
  (`p1_push` ok vs failed, 33 of 36): the routers made the same decisions and the episode diverged in the physics /
  noise draws (4-3's toppling top layer and `push_settle`'s estimate checks amplify tiny differences). So the 4-3
  deltas are noise, not routing. Real routing differences: **5-3, after a push-in that ended short (`i1_push_in`
  failed, place_xy 5.0 mm), bc_r2 parks and gives up, bc_v2_r3 goes back to the pre-push-in pose and pushes it home
  (2.1 mm, success): 3 episodes (7700 nominal, 8100 nominal, 8100 lightweight), all won** (R5.2, learned from
  `oracle_state_v2`'s labels; its transit back moved a block 16.9 mm once, the episode still succeeded); a few
  different push / grasp orders (4-3 8500 uneven_weights: goal 1 before goal 0, 16 vs 12 ok ops).
- **Camera reference for Task 4, seed 7500 (`results/v1_3/camera/ref/t4ref_7500`, bc_r2 + v1.2 flags, 09:50-11:05):**
  4-2 4 / 10, LHPR 85.0 %, 11.60 pts; 4-3 0 / 10, LHPR 60.0 %, 6.68 pts (18.28). Seed 7400's v1.2-like reference
  `results/v1_1/t4_camera/rec_end`: 4-2 5 / 10, 85.0 %, 12.22; 4-3 0, 56.7 %, 6.30 (18.52). The candidate chain
  (`gpu_cand_chain.sh`: t13 -> t4 -> t5) started 11:05.

**Step 7a: Tasks 1-3 camera check of the candidate (11:05-13:10, `results/v1_3/camera/r3/t13`; our renders -> kp_v2 ->
bc_v2_r3, v1.3 flags incl. `offcentre_place`, `--official_orient`, mirror, ten cases, seed 7200 x 10 per variant):
59 / 60, 32.87 points (reference `post_success/core_guard`, v1.1 router + guard, same seeds: 60 / 60, 33.23).** Gate
passed: >= 58 / 60; 0 finished-guard vetoes (the guard never had to act); 0 releases > 25 mm above the goal (202
places; release p50 6.9-10.6 mm); 2 `offcentre_place` triggers (4-6 mm grips), both ok; 0 buried reworks. The "goal
nodes after success" (1-2: 9, reference 7) are the last place's own verdict landing 0.1 s after the success
time, not reworks.
- The one failure, 2-2 high_friction (seed 7400001): goal 0's learned place (grip 2.1 mm, under the threshold, so the
  same skill as v1.2) took 37 steps (reference run 18) and opened 14.4 mm above the goal; on opening, the high-friction
  fingers flung the block ~100 mm sideways (truth x 174 -> 83 mm, y -525 -> -577 mm in 0.3 s) out of the cameras'
  view (conf 0.3 coasting at the release pose, then 0). The verdict failed it (unseen); the router went on to goal 1
  (nothing else was visible) and placed it in goal 0's spot one layer low, retried twice, parked. The decisions up to
  the release were the reference's; the camera loop is not deterministic (render noise -> the place policy's
  observations), so this is a skill / physics event, not a router one. The same can happen to v1.2.
- D4 (Tasks 1-3): the candidate keeps Tasks 1-3.

**Step 7b (Task 4 camera), partial (15:55).**
- 4-2 seed 7400 (13:11-13:35, `results/v1_3/camera/r3/t4_7400`): **8 / 10, LHPR 91.7 %, 14.60 points** vs the v1.2-like
  reference `t4_camera/rec_end` 5 / 10, 85.0 %, 12.22 (`scripts/t4_camera_cmp.py`: pushed blocks flat clear 55 vs 51 /
  60, tipped 0 vs 5, leaning on base 3 vs 2, touching 0 vs 1, on a block 2 vs 1). One R4.1 re-push (rule node, as
  v1.2), on a truly leaning block, ended flat clear. Camera A/Bs are noisy (10 episodes); not worse is what counts.
- **4-3 seed 7400 hung in the renderer**: started 13:36; the Kit log's last line 13:37:41 `[omni.syntheticdata.plugin]
  SdRenderVarPtr missing valid input renderVar LdrColorSDhost`; from ~13:44 the GPU sat at 13 % / 210 MHz while the
  main thread spun at 100 % CPU for 2 h with no output. SIGABRT (to get a crash stack) was caught and never
  handled, i.e. the main thread was in native code, not Python, so not a student / runtime livelock; killed (SIGKILL)
  15:52, the chain moved on to seed 7500. Same class as the earlier "hung at Kit start" gotcha. The server never renders
  (it gets the evaluator's images), so this cannot occur in an official run.
- Rerun queued after the chain (`results/v1_3/gpu_t4_7400_43_rerun.sh`, guarded, hard 75 min limit) and a hang
  monitor (`results/v1_3/hang_watch.sh`: a camera process running, guard not paused, GPU < 30 % for 10 min).

**v1.2's official evaluation started 2026-10-29 16:06:23** (queued 06:07; first session task1-2; same reconnect pattern
as attempt 1: several sessions open / disconnect in the first minutes before calls). Watcher re-armed
(`WATCH_NAME=official_v1_2 WATCH_QUIET_START=1 watch_official.py 2026-10-29T16:06:00 3600 1800`). Attempt 1 ran
~35 episodes in ~4 h, so the full run likely ends ~02:00-04:00 on 10-30.
- **User decision 16:15: no GPU work beside the evaluation.** The guards paused the candidate chain at 16:06 (Task 4
  seed 7500, 4-2 in progress) and resume it after 20 min without evaluation activity. Queued behind it, each under its
  own pausing guard: the 4-3 seed 7400 rerun (`gpu_t4_7400_43_rerun2.sh`, guard log `guard_t4_rerun2.log`), then the
  end-to-end test of the current bundle (`gpu_e2e_after.sh` -> `gpu_ws_e2e_v1_3.sh`, server_v1_3 snapshot, port 9005,
  guard log `guard_e2e.log`). Hang monitor `hang_watch.sh` covers them. Then D4, (rebuild if a family falls back),
  serve on 9004, URL to the user (~07:00, before the 08:00 cutoff).
- Lesson: never wrap a guarded job in `timeout` (without `--foreground`): timeout makes its own process group, so the
  guard's SIGSTOP to the command's group no longer reaches the Isaac process. The first rerun script had it; it was
  killed (16:18, never ran) and replaced.

### Third hard reset during an official evaluation (2026-10-29 16:41): investigation
- **When:** files written until 16:41:14 (server status 16:40:57, episode logs 16:41:14), next boot ~16:41:45 (kernel
  16:42:30): a reset with an automatic restart, not a shutdown; the journal's last persisted entry was 16:33:28 (Chrome
  crashpad), nothing after. v1.2's evaluation attempt 2 was 35 min in (started 16:06; 24 episode summaries, Tasks 1-2
  / 1-3; 8 sessions connected, 51-75 calls/min, mean call 92-94 ms, 0 errors). Our GPU jobs had been SIGSTOPped by
  the guards since 16:06 (their GPU telemetry stops at 16:06:22); load ~2.
- **All abrupt boot ends in the journal (since 08-22):** 09-07 ~12:35 (light desktop use; Chrome GPU errors 12:31; back
  12:49), 10-18 ~14:36 (idle: tracker jobs only; back 14:37:38), 10-29 ~00:10:55 (v1.2 evaluation attempt 1, 4 h in;
  last episode write 00:10:51, back 00:11:33), 10-29 16:41:15 (attempt 2). 09-07 -> 10-16 ran 39 days and ended with
  a user `sudo reboot`. None during the hundreds of hours of sustained full load (10-29 07:30-10:15: load ~26 on 24
  threads + GPU 100 %).
- **Ruled out / not seen:** a kernel panic (kernel.panic=0: a panic would freeze, not reboot; no messages); MCE /
  hardware-error / BERT records at the following boots (none); PCIe AER errors (none since boot); GPU thermal or
  power-brake slowdowns (counters 0; limit 230 W, shutdown 98 C); temperatures (CPU idle 43-53 C, GPU 41 C, NVMe 48 C);
  memory / disk; software changes (no packages since tmux 10-24; NVIDIA 580.105.08 since 2025-12-25).
- **Platform:** i9-12900KS on ASUS PRIME Z690-A, BIOS 2004 (2022-08-24), CPU microcode 0x2c (intel-microcode
  20230214; Ubuntu 22.10 is EOL so no newer package), kernel 5.19; Intel I225-V Ethernet (igc); USB Wi-Fi rtl88x2bu
  (out-of-tree); the desktop and Chrome on the Intel iGPU (i915 GPU hangs in Chrome 08:08, 14:59; the A5000 drives no
  display). At idle cpu0 spends ~65-70 % in C10, single cores boost to 5.2 GHz.
- **Reading:** hardware-level resets at idle or light, bursty load, never under sustained heavy load: most consistent
  with a low-load / power-state-transition instability (CPU + old BIOS / microcode, or the PSU's low-load / transient
  behaviour); less likely the I225-V (two of four had no network load), mains blips, or the iGPU (Chrome GPU errors
  before two of four).
- **Done:** `scripts/sys_telemetry.py` started (pid 18865, setsid): one fsynced CSV line per second
  (`results/v1_3/logs/sys_telemetry_*.csv`: load, CPU temps / MHz, cpu0 idle-state shares, GPU W / C / util / clock /
  P-state / reasons, NIC rates), so a next reset leaves its last seconds. v1.2 re-served 16:50 (old logs in
  `results/student/server/official_v1_2/attempt2_crash_20261029_1641/`): **wss://wish-rico-sic-willing.trycloudflare.com**,
  identity f726c71a...9aa9, probes 10 / 10 locally and through the tunnel.
- **Suggested to the user (root needed):** disable C8 / C10 until the next reboot (`.../cpuidle/state{3,4}/disable`),
  keep Chrome closed (or without hardware acceleration) during evaluations, UPS / another outlet, PSU model; after the
  deadline: BIOS update (newer microcode), check MultiCore Enhancement, memtest.
- **User decision 16:55: v1.3 is the final run (v1.2 not re-queued; its endpoint stays up as a fallback).** The reset
  killed every queued job; the remaining checks restarted 17:20 as one guarded chain (`results/v1_3/gpu_v13_final.sh`,
  guard log `guard_v13_final.log`, monitor `hang_watch.sh <guard log>`): end to end of `student_v1_3` (server_v1_3
  snapshot, port 9005; 17:21: calls flowing, 0 errors, mean call 124 ms), then Task 5 (seed 7400), Task 4 seed 7500
  (4-2, 4-3), Task 4-3 seed 7400 (the rerun). Then D4, serve on 9004 (~21:00).
- **Chain done 20:14** (results under `results/v1_3/camera/r3`, `results/v1_3/ws_e2e`, server logs
  `results/student/server/e2e_v1_3`):
  - **End to end (Step 8) passed:** 8 episodes through `serve_student.py` (server_v1_3 snapshot, the frozen bundle):
    1-3 2 / 2, 2-2 2 / 2, 4-2 0 / 2 (GSR 0.75), 5-3 0 / 2 (GSR 1.00, PSR 0.25); 0 errors, holds, clipped values,
    vetoes, masked calls; not blind; client call p99 426 / 346 / 1,335 / 324 ms (4-2 as v1.2b's 1,248 ms: our renderer
    shares the GPU); summaries carry `release_above_mm` (places, 5.0-15.4 mm), `vetoed` / `vetoes`, Task 4's
    `recovery` (abstentions only). 5-3 "open" = the last variant's sessions, as in v1.2b (server stopped first).
  - **Task 5 (seed 7400):** 5-2 8 / 10 (14.34) vs `ref/t5_ref` 7 / 10 (13.80); 5-3 2 / 10 (9.00) vs 3 / 10 (10.08);
    23.34 vs 23.88. Same decisions up to every failure (uneven_weights: cand i1_push_in 4.0 mm failed, ref 2.3 mm ok,
    identical skills -> camera-loop noise). Neither router pushed a short insert home on camera: both park after a
    failed push-in (CPU: cand pushed home 3 / 48). Gate (not worse by > 1 success per variant): passed -> T5 = cand.
  - **Task 4 seed 7500:** 4-2 5 / 10, LHPR 93.3, 13.06 vs `ref/t4ref_7500` 4 / 10, 85.0, 11.60; pushed blocks flat
    clear 56 / 60 vs 51 / 60. 4-3 0 / 10, 61.1, 6.73 vs 0 / 10, 60.0, 6.68; flat clear 55 / 90 both (tipped 18 vs 12, on
    a block 10 vs 18). 4-2: 2 R4.1 re-pushes, both truly leaning, both ended flat clear. With seed 7400 (4-2 55 / 60 vs
    rec_end 51 / 60): per push not worse -> T4 = cand.
  - **4-3 seed 7400 hung in the renderer again:** batch 1 done 19:58 (TSR 0, GSR 0.72), GPU idle (12 %, 210 MHz) from
    20:02, main thread 100 % CPU in native code, helpers idle (same signature as 13:40; ptrace_scope 1, no stack);
    SIGTERM ignored, killed -9 20:14. Not reproducible at a fixed point (first run ~5 min in, this ~12 min in); the
    server never renders, so no bearing on the submission. Not run a third time.
- **D4: all families keep the candidate; the trial freeze is final** (`student_v1_3`, sha256 76ec7f3d...263d).
- **Served 20:16 as official_v1_3: wss://norfolk-charleston-recommended-enough.trycloudflare.com** (port 9004, tmux
  `tower_v1_3`, policy `tower-student-v1-3`, artifact identity sha256:76ec7f3d...263d); bundle check vs v1_2 ok (only
  the switcher files, `offcentre_place` on Tasks 1-3, `router_repush` on 4-2 / 4-3 differ); probes 10 / 10 locally and
  through the tunnel (second try, 14 s). Watcher armed from 20:20 (`WATCH_NAME=official_v1_3 ... first`). v1.2 stays
  up on 9003 (no platform session since its 16:50 re-serve). At 20:18 C8 / C10 still enabled and Chrome open.
- 20:44:24-20:46:21: the site's Check endpoint on official_v1_3 (10 tasks, 2 resets each, seed 0, 0 errors). 20:46:
  C8 / C10 disabled on all 24 CPUs at the user's request (`pkexec`, by state name; until the next reboot); telemetry
  shows cpu0's idle time in C6 (~0.86), 0 in C8 / C10. Watcher re-armed from 20:46:30 for the evaluation's start.
- **~20:48: the user queued official_v1_3 (final run)**, with Chrome closed and C8 / C10 off; no session yet at 20:49.
- **2026-10-30 03:28:34: official_v1_3 evaluation STARTED** (queued ~20:48, ~6.7 h in the queue; no reset while waiting, up since 10-29 16:42, C8 / C10 off). Start as v1.2's: task1-2 sessions reset (seed 0), no calls for the first minutes while the evaluator sets up. Watcher re-armed (WATCH_QUIET_START=1, from 03:28:00, progress 3600 s, idle 1800 s).
- **Fifth hard reset 2026-10-30 03:53:43** (v1.3 evaluation 25 min in: 9 episodes started, 4 summaries; 2-5 sessions,
  25 calls / min). Telemetry (fsynced 1 Hz) ends 03:53:43, kernel up 03:54:19: instant, nothing abnormal in the last
  minute (load 1.1, CPU 49-67 C, idle in C6 with C8 / C10 off, GPU 77-112 W at 71-73 C, rx 4.5 MB/s). No panic
  (panic=0 would hang), no MCE / AER / Xid / thermal events, no hardware watchdog (RuntimeWatchdog off, no
  /dev/watchdog). Router and Wi-Fi AP stayed up at every reset (link up 6 s, Wi-Fi 9 s after kernel start): not a
  whole-house outage. 10-29 16:41 was a double reset: a hidden boot (b740f5e5, kernel 16:41:51) died ~4-15 s into
  early boot before the 16:42:30 boot. Pattern: 09-07, 10-18, then 10-29 00:10 / 16:41 (+ boot) / 10-30 03:53: the
  last three all during evaluations (~5 h of evaluation vs ~27 h without one since 10-28 20:10), but official_1's
  evaluation ran 22 h on 10-26/27 (5.2 sessions, 155 calls / min, idle machine) without a reset, and the heavy GPU
  chain 17:20-20:14 (157 W mean, 205 W max, 89 C GPU, 97 C CPU) and 7 h idle passed. Conclusion: hardware / power
  delivery fault whose rate jumped since 10-28 (PSU, connector / outlet, board VRM or CPU); the C-state mitigation
  did not stop it; the logs cannot tell which part.
- Telemetry restarted (`sys_telemetry_20261030_0608.csv`). **v1.3 re-served 06:09 at
  wss://wilson-angela-maui-pump.trycloudflare.com** (old logs in `server/official_v1_3/attempt1_crash_20261030_0353/`);
  probes ok locally and through the tunnel. v1.2 not re-served.
- GPU question (user, 10-30 ~06:15: past resets with an incompatible GPU driver during RL runs): the endpoint runs every
  student model on CUDA (`serve_student.py --device cuda`; ~2 GB; 70-110 W, P2 during the evaluation); the simulation
  and rendering are the organizers'. Driver NVIDIA 580.105.08 (proprietary, DKMS on 5.19, GSP firmware on, persistence
  on; nvidia-modprobe / nvidia-settings 590.48.01), installed 2025-12-25, unchanged since (only Chrome upgraded,
  10-29 14:40); no Xid ever logged. CPU-only server test (port 9006, CUDA hidden, organizer replays): 1 thread (the
  server's `torch.set_num_threads(1)`) 3.8 s per call (10 sessions, 16 calls / min); 16 threads ~1.0 s per call
  (4 sessions, 63 calls / min) vs ~0.1 s on the GPU: too slow for an evaluation. Test servers stopped.

## 2026-10-30 (session 8, cont.): steady GPU load through the evaluation + learned reaching (PPO) setup

- **User (10-30 ~07:40):** re-register v1.3 (`wss://wilson-angela-maui-pump.trycloudflare.com`, probes ok 07:45) and
  queue; keep a steady GPU load through the whole evaluation (no reset ever came under sustained load); use that GPU
  time to start a goal-conditioned PPO reacher (JAiLeR, arXiv 2011.06332); plan it. Plan:
  `corl_challenge/Reach-PPO-plan.MD`.
- **Latency A/B** (second server on 9006, organizer replays, 4 sessions): 86-89 ms per call with the GPU otherwise
  idle; 123-132 ms with `scripts/gpu_steady_load.py --duty 0.6`. Running stand-in duty 0.3: ~30 % util, ~160 W,
  84 C, fan 57 %. Duty 0.5 drew 205 W at 86 C (too hot to hold for hours).
- **`scripts/bg_load_keeper.py`** (started 07:55, log `results/reach/logs/bg_load_keeper.log`): the stand-in under
  `eval_guard --no-pause --mem 4G` whenever no training named in `results/reach/train.pid` has run for 2 min;
  `touch results/reach/STOP_BG` stops it.
- **Reacher P0 code:** `tower_rl/skills/reach_env.py` (arm-only DirectRLEnv: joint-velocity actions integrated to
  q_des, sag-compensated rows with the command delay; three-point pose error; sequential goals with a settle test;
  nested-region curriculum on goal displacement; goals = FK of perturbed joints / demo posture priors filtered to the
  table workspace with the approach axis within 60 deg of down), `configs/skills/reach.yaml`, `--skill reach` in
  `scripts/train_skill.py`, `scripts/reach_check.py` (zero / IK-oracle / trained-policy checks),
  `results/reach/run_train.sh` (left then back, 3000 iterations, 4096 envs, guarded, writes train.pid).
- Smoke (64 envs, 3 iterations): obs 55 / critic 65, finite. Hold test (zero actions, 256 envs, level 3): max joint
  error 3.8 mrad (sag compensation holds the pose); `app.close()` hangs in Isaac Sim 5.1, so the check ends with
  `os._exit(0)` like train_skill.py.
- IK oracle (`reach_check.py --policy ik`, left, level 3, 256 envs, 300 rows): settled 97.5 % of 5,671 goals, end
  error p50 0.75 mm / 0.12 deg, p95 1.81 mm / 0.26 deg, settle p50 1.1 s; pinch min z -21 mm (joint-space straight
  lines dip into the table: the table penalty matters).
- Training started 08:15 (`run_train.sh 3000 4096`): 4.8 s per iteration, 3.6 GB, 89 % util, 154 W; the keeper
  handed over at 08:17:43. Server latency with it (A/B replay, 4 sessions): 266-270 ms per call (idle 86-89).
  Restarted 08:22 at 1,024 envs (`run_train.sh 6000 1024`): still 4.8 s per iteration (288 physics steps per
  iteration dominate), 81 % util, 135 W; latency 210-217 ms. official_1 (22 h, completed) had 328 / 1,314 minutes at
  >= 200 ms per call, so this is within what the evaluator has handled. Kept 1,024 envs while the evaluation runs.
- Gotcha: `pgrep -f '<train_skill pattern>'` also matches the shell running the command; killing its result killed
  my own shell. Kill by the exact PIDs from `ps` (filter out `bash -c source`).
- Site Check endpoint 08:41:06-08:43:24 (10 tasks, 0 errors) with the training running; watcher re-armed from 08:44.
- **Reach run 1** (08:22-08:55, 808 iterations, `train_reach_left_p0_run1.log`): learned to stand still. The abort at
  d > 2.4 m cut early random drift, action std 0.5 -> 0.04; end of goal 90 mm / 35 deg on level-0 goals
  (<= 10 cm / 15 deg), 1 % settled. **Run 2** (08:55, `_run2.log`): + `reward.linear` 2.0 (-2 d), abort off, entropy
  0.01: end of goal 90 -> 25 mm / 32 -> 12 deg by iteration 392, then a plateau at ~20-25 mm / 10-12 deg with
  std ~0.3. Mean policy (`reach_check`, model_1100, level 0): p50 5.1 mm / 4.3 deg, p95 24 mm / 21 deg, 82.5 % within
  1 cm, 0.1 % settled (2 deg). **Run 3** (09:39): `ppo.entropy_schedule` {to 0.0005, iters 2000} (new option in
  train_skill.py: linear anneal of rsl_rl's `alg.entropy_coef`), resumed from run 2's model_1100.
- Run 3 (09:39-10:28, `_run3.log`): entropy annealing moved slowly (std 0.33 -> 0.26 by iteration 2,332, end of goal
  25 -> 15 mm / 11 -> 8 deg). **Run 4** (10:28, from run 3's model_2300): `ppo.resume_noise_std` 0.12 (new option:
  reset the checkpoint's learned action std), entropy 0.0005 constant (`results/reach/override_left_r4.json`, new
  `--override @file`), + reward `mid` 1.0 (sigma 6 cm on d). run_train.sh takes RESUME_LEFT / OVERRIDE_LEFT /
  RESUME_BACK.
- Run 4 by iteration 3,420: noise std 0.04, settled (5 mm / 2 deg, stopped) 0 -> 22 % of level-0 goals. Mean policy
  (model_3400, level 0): p50 2.56 mm / 2.01 deg, p95 19.6 mm / 14.7 deg, 92 % within 1 cm, settled 21.7 % (the median
  rotation sits on the 2 deg settle limit). **Pinch min z -0.178 m:** beside the table (near the arm base, outside the
  table footprint) the arm swings below table height; the table term only covers the table. P3 must add the stand /
  station structure as obstacles.
- Run 4 to iteration 5,044 (killed 12:20, `_run4.log`): training settled fraction fell 22 -> 7-10 %. Mean policy
  (model_5000, level 0, `reach_check --dump`, new: per-goal start / end errors, end joint speed, limit distance):
  p50 2.46 mm / 1.91 deg, but settled 7.8 %, p95 47 mm / 56 deg, goal time p50 = the full 6 s. **Reward hacking:**
  52 % of goals ended inside the settle tolerance (5 mm / 2 deg) yet unsettled, max |qd| p50 0.17 rad/s (limit 0.05):
  settling ended the goal, and the next goal (~5 cm away) pays far less than hovering near the old one (~3 per row for
  ~20 rows vs a 5 bonus), so PPO learned to jitter on the goal. Separately 6.2 % of goals ran away (> 5 cm or
  > 30 deg off at the end, some 1 m off / 180 deg), 64 % of them pinned at a joint margin. Without those the mean was
  3.5 mm / 2.6 deg, inside the curriculum bar (1 cm / 3 deg), so the runaways alone held it at level 0.
- **Run 5** (12:21, from run 4's model_3400, `override_left_r5.json`: noise std reset 0.08, entropy 0.0005; 3,000
  iterations at 1,024 envs, back arm after): `goal.switch: timer` (new; goals last a random 3-6 s whatever the arm
  does; `settle` keeps the old behaviour), `reward.hold` 2.0 per row x exp(-(pos/4 mm)^2 - (rot/2 deg)^2 -
  (max|qd|/0.05)^2), `reward.joint_limit` 0.1 -> 0.5, `curriculum.trim` 0.05 (the advance test drops the worst 5 % of the
  window). New logs: `reach/ever_settled_frac`, `reach/settle_time_s` (first settle). IK oracle with the timer
  (level 0, 128 envs): settled 99.9 %, p50 0.6 mm / 0.14 deg, settle 0.6 s.
- Run 5 first try (12:21-12:23, `_run5a.log`): ever-settled 0 % — action noise 0.08 x v_max 1.5 moves each joint
  ~0.12 rad/s, above the 0.05 settle / hold limit, so under exploration the arm is never still and `hold` pays
  nothing. **`action.map: square`** (new; joint speed v_max * a|a|, same top speed, ~6x finer near zero;
  `reach_check --policy ik` inverts it). Restarted 12:24 as run 5 (same resume / override); first iterations
  settled 3.8 %, end of goal 9.4 mm / 6.7 deg (vs 0 % / 19 mm under the linear map).
- Run 5 (12:24-13:12, `_run5.log`, to iteration 4,655): training settled 4 -> 11-14 %, ever settled ~15 %. Mean
  policy (model_4500, level 0): p50 3.6 mm / 3.0 deg, p95 35 mm / 36 deg, 26 % in tolerance at the goal end, 12 %
  settled, 5.5 % runaways. `reach_check --trace N` (new: per-row q, raw action, q_des, q_cmd, errors of the first N
  envs, into the --dump JSON) showed the real cause of the "jitter" (also run 4's 0.17-0.2 rad/s mode): **joint 1
  (base yaw) bang-bang** — the raw action mean alternates ~+4 / -9 every row (max 24), clamped to +-1, so q_des swings
  0.15 rad each row and the arm averages it out around the goal at ~0.19 rad/s. 98 % of rows, all 64 traced envs;
  the other joints exceed +-1 on ~1 % of rows. The env clamps silently, so PPO saw no cost and no gradient beyond the
  clamp. The runaways pinned at a margin are likely the same (a saturated mean).
- **Run 6** (13:12, from run 5's model_4600, `override_left_r6.json`: noise reset 0.06): `reward.action_bound` 0.1 per
  unit^2 of raw action beyond +-1 (new), `action_rate` 0.01 -> 0.05.
- Why joint 1: its drive (configs/robot/nero.json, the evaluator's identified values) is weak and friction-bound —
  kp 75 (joints 2-4: 4,815 / 2,709 / 2,085), kd 0.06, static friction 0.23 N m, viscous 5.0 N m s/rad. The static
  friction alone leaves a +-3 mrad deadband around q_des (~1.5 mm at the pinch at 0.5 m reach) and the viscous term a
  ~67 ms lag. The bang-bang is a learned dither past that friction. The IK oracle's p95 1.6 mm is the same deadband,
  so ~1.5 mm is the floor for any controller that sends q_des = the IK solution; the policy can do better only by
  overshooting q_des on joint 1 deliberately (smoothly, not by dithering).
- Run 6 (13:12-13:58, `_run6.log`, to iteration 5,703): the bound penalty halved joint 1's raw output (median 6.0 ->
  3.2) but the dither stayed (saturated sign flips on 92.5 % of row-envs): it lives at +-1, where only action_rate
  (0.05 x 2^2 = 0.2 per row) costs anything. Worse overall: mean policy (model_5700) p50 4.4 mm / 3.1 deg, settled
  2.2 %, runaways 5.8 %; reward fell to -288 per episode on the change, value loss up to 6,000.
- **Run 7** (13:58, from run 6's model_5700, `override_left_r7.json`, noise 0.05): `action_rate` 0.05 -> 0.5 (the dither
  costs 2 per row). Fallback if it persists: a fresh run with all of run 5-7's changes from iteration 0.
- Run 7 (13:58-14:37, `_run7.log`, to iteration 6,630): the curriculum opened level 1 at iteration ~6,400 (trimmed
  window mean 4.1 mm / 3.00 deg), but joint 1 still saturated on 96 % of rows (model_6600, level 0: p50 3.6 mm /
  2.8 deg, settled 10 %, runaways 6.5 %). Cause: action_rate was computed on the clamped action; with raw means at
  +-3 every exploration sample clamps to the same +-1, so the penalty does not vary with the sample and PPO gets no
  gradient from it at any weight. **Run 8** (14:38, from run 7's model_6600, noise 0.05): `action_rate` 0.1 on the
  raw action change (a +-3 dither: 3.6 per row), `action_bound` 0.1 -> 0.5.
- Run 8 (14:38-15:12, `_run8.log`, to iteration 7,407): worse. Episode reward swung -200 .. -1,650 (the squared bound
  penalty on raw outliers up to 66), nothing settled, joint 1 still saturated on 81 % of rows; mean policy
  (model_7400, level 0) p50 6.7 mm / 3.5 deg, runaways 7.9 %. Four resumed runs (5-8) did not unlearn the dither.
- **Run 9** (15:13): joint-1 head surgery instead of a fresh start. `scripts/reach_reset_joint.py` (new) zeroes joint
  1's row of the actor output layer (mean 0) and sets its noise to 0.3 (others keep 0.02-0.05) in run 7's model_6600
  (`results/reach/ckpt/left_run7_6600_j1reset.pt`); resumed with `--resume_no_optimizer`, entropy 0.0005
  (`override_left_r9.json`). `action_bound` now linear (0.5 per unit beyond +-1). First iterations: episode reward
  +490 (run 8: -1,000s).
- Run 9 by iteration 7,427 (30 min): **the surgery worked.** Joint 1 beyond +-1 on 0.2 % of rows (run 8: 81 %);
  goals inside tolerance are still (max |qd| p90 0.017 rad/s, was ~0.2). Mean policy (model_7400, level 0): settled
  28 % (run 7: 10 %), settle p50 1.1 s, p50 3.25 mm / 2.87 deg, 29 % in tolerance at the goal end; runaways 5.7 %.
  Curriculum level 1 at iteration ~7,250 (trimmed mean 3.7 mm / 2.99 deg). Rotation is now the main miss (p50 above
  the 2 deg tolerance). The runaways start in unusual configurations: in the traced one an episode began with joint 1
  at 2.5 rad (a `reset.uniform_frac` start) and the arm walked away from a 33 mm goal to 122 mm / 104 deg — joint
  space the policy has not learned yet; left to the larger curriculum levels for now.
- Lesson: in this env every action cost must be on the raw policy output, not the clamped command (no gradient past
  the clamp), and a squared cost on unbounded raw outputs swamps the returns. The back-arm run that follows uses
  these terms from iteration 0, which also tests whether a fresh policy avoids the joint-1 dither.
- Run 9 to iteration 9,187 (16:52, `_run9.log`): level 1 from ~7,250 on, no level 2. Mean policy (model_9100,
  level 1): settled 19.6 %, p50 4.1 mm / 3.6 deg, p95 41 mm / 42 deg, runaways 5.7 %; without them 5.0 mm / 4.05 deg
  (bar 1 cm / 3 deg). Still on the goal (max |qd| p90 0.021), joint 1 in range. The miss is precision in both position
  and rotation (the IK oracle: 0.14 deg, so not friction).
- **Run 10** (16:52): `obs.fine_m` 0.01 (new): + tanh(three-point error / 1 cm) appended to the actor observation (and
  the critic's actor part), 9 inputs, 55 -> 64 (critic 65 -> 74); the raw error in metres, normalised by a ~5 cm
  spread, gave 2 mm ~0.04 units. `scripts/widen_push_checkpoint.py` takes `old new` (here 55 9): run 9's model_9100
  -> `results/reach/ckpt/left_run9_fine.pt` (zero columns; level-1 check unchanged: p50 4.05 mm / 3.51 deg). Resumed
  with `--resume_no_optimizer`, curriculum start level 1, entropy 0.0005 (`override_left_r10.json`). The back arm
  that follows trains from scratch with every change of runs 5-10. `reach_check.py` now loads weights only.
- Run 10 (16:52-17:47, `_run10.log`, to ~10,500): small gains. Mean policy (model_10400, level 1): settled 24.5 %
  (run 9: 19.6), p50 4.0 mm / 3.3 deg, runaways 6.0 %, noise std down to 0.01. Trace: 2+ s into a goal, 2-6 deg off,
  the arm is parked (rotation change over the next 1 s p50 -0.04 deg); the largest output p50 0.076 -> 0.009 rad/s
  through the square map, whose slope at 0 is zero (closing 3 deg in ~1 s needs ~0.05 rad/s).
- **Run 11** (17:48, from run 10's model_10500.pt, with optimizer, start level 1): `action.map: blend` (new), v_max *
  (0.3 a + 0.7 a|a|): same top speed, ~4.6x the speed at a = 0.076, +30 % at a = 0.5; reach_check's IK oracle inverts it.
- Run 11 (17:48-19:42, `_run11.log`, to 13,499): **level 2 (R3) at ~11,200**, the first new level since run 9.
  Mean policy, level 2 (model_13300): settled 33.9 %, p50 3.3 mm / 2.55 deg, p95 64 mm / 63 deg, within 1 cm 81.5 %,
  runaways 7.6 %.
- **Where the runaways come from** (the 64-env trace, goals in order per env): the first goal of an episode ran away
  21 % of the time, a goal after a good one 2.6 %, a goal after a runaway 48 % (the arm sits pinned at a limit and the
  next goal is sampled around it). Splitting the first goals by start: **0 / 101 from home / demo postures, 27 / 28
  from `reset.uniform_frac` joint-uniform starts** (mostly sideways or upward pinches, outside the goal distribution).
  New `reset.uniform: workspace` (default now): the "anywhere" start is a joint-uniform pose whose pinch is inside
  `goal.workspace` and points down (32 candidates, else the posture start). Run 11's final policy on those starts,
  level 2: settled 38.8 %, p50 2.8 mm / 2.3 deg, p95 26 mm / 22 deg, within 1 cm 87 %, runaways 4.6 %.
- **Why the last degrees are not closed** (same trace): rotation is the binding miss (of the non-runaway goals 30 %
  are within 5 mm but over 2 deg, ~0 % the other way); longer goals do not end better (unsettled goals at 5-7 s:
  p50 3.75 deg, at 3-4 s: 3.4), so the policy parks off the goal. In the parked rows the wrist outputs are ~0.003
  while joints 1 and 3 turn against each other at 0.03-0.06 rad/s, moving the arm without moving the pinch. The
  wrist deadbands are ~2 mrad (0.1 deg; friction / kp 0.27 / 153 and 0.31 / 162), so it is not the drives. The hold
  term is a product, ~0.02 per row at 3 mm / 3 deg whatever the arm does: no gradient there toward the goal or toward
  stopping the null-space drift. New `reward.hold_wide` (+ `hold_wide_sigma` 10 mm / 5 deg / 0.05 rad/s), the same
  product with wider sigmas; 0 by default.
- Back arm from scratch (19:43, 1,024 envs, 3,000 iterations): every change through run 11 plus the workspace starts
  (not `hold_wide`). `run_train.sh` takes `ARMS`. Queued after it: **left run 12**
  (`results/reach/queue_left_r12.sh`): run 11's model_13499, start level 2, entropy 0.0005, `hold_wide` 2.0
  (`override_left_r12.json`), workspace starts.
- **Back arm from scratch, first hour (to 1,447):** curriculum R2 / R3 / R4 opened by iteration ~450 (trimmed
  window means 5.1-5.6 mm / 3.0 deg); in training at R4, 75 % of goals settled by 1,447 (left at R3: ~30 %). Mean
  policy (model_1400): R1 settled 91 %, p50 1.33 mm / 0.69 deg; **R4 from posture starts settled 92.5 %, p50 1.04 mm
  / 0.57 deg, p95 8.8 mm / 8.2 deg, within 1 cm 95.2 %, runaways 4.1 %** (left run 11 on the same test: 49.7 %,
  2.2 mm / 1.5 deg, 81 %, 17 %); workspace starts 89.9 %, p95 15 mm / 23 deg, runaways 4.8 %. No dither (raw beyond
  +-1 on <= 3 % of rows per joint). The back drives are not easier: joints 1 and 4 have 3.9 and 3.3 mrad deadbands
  (left joint 1: 3.05). So the left lineage's habits (dither, null-space drift) came from runs 1-4's env and survived
  the resumes. **Left run 12 is now from scratch** with the same config (`results/reach/queue_left_scratch.sh`,
  replaces `queue_left_r12.sh`; `hold_wide` stays off so both arms train the same).
- Lesson: after a redesign of the env or reward, retrain from scratch instead of resuming a policy shaped by the old
  one; here an hour from scratch beat eight resumed runs.
- Back arm final (19:43-21:42, `_run1.log`, model_2999): R4 from posture starts settled 93.0 %, p50 0.81 mm / 0.38
  deg, p95 7.65 mm / 9.5 deg, within 1 cm 95.3 %, runaways 4.1 %. **Goals that need < 90 deg of turn (95 % of R4):
  settled 96.9 %, p50 0.79 mm / 0.37 deg, p95 1.72 mm / 0.94 deg, within 1 cm 99.0 %, settle p50 1.5 s: the P1
  accuracy gate** (the time gate against the planned transit is not measured yet). Workspace starts, < 90 deg: p95
  1.90 mm / 1.10 deg. The remaining runaways are the large turns: 61 / 91 / 95 % of goals needing 90-120 / 120-150 /
  150-180 deg (they end pinned, mostly joint 6).
- **Left run 12, from scratch** (21:42, the back arm's config): R1-R4 open by iteration ~600. Mean policy at 1,300
  (55 min): R4 posture starts settled 84.9 %, p50 1.04 mm / 0.96 deg, p95 5.05 mm / 4.57 deg, within 1 cm 96.1 %,
  runaways 3.0 % (run 11's final on the same test: 49.7 %, 2.2 mm / 1.5 deg, 81 %, 17 %); < 90 deg: p95 2.53 mm /
  2.57 deg; R1: settled 91.6 %, p95 2.79 mm / 2.25 deg, 99.7 % within 1 cm.
- **Payload (P2):** `payload.force` (new, default on): the block's weight (payload.kg x g, world down) applied at the
  pinch point on `gripper_base` each substep (`set_external_force_and_torque`, local frame, rotated by the body's
  orientation); before, a payload only changed the compensated command. `scripts/reach_payload_check.py` (zero
  actions, half the envs holding the block, pinch vs FK(q_des) after 30 rows): left: 0.23 mm with the block and the
  force, 0.21 mm without a block, 0.21 mm with compensation but no force (the left drives are stiff; wrist friction
  0.27-0.31 N m exceeds the block's ~0.19 N m). Back: 2.74 / 2.84 / 0.76 mm: with the force the block is compensated
  exactly, and the unloaded back arm sags 2.6 mm at rest (joint 4 kp 68), which the policy corrects in closed loop.
  (`app.close()` hangs in Isaac Sim 5.1: the first version of the script hung after printing; scripts end with
  `os._exit(0)`.)
- Queued after left run 12 (~23:35): **P2a** (`results/reach/queue_p2a.sh`): each arm resumes its from-scratch run
  for 3,000 iterations at R4 with a held block in 30 % of episodes, entropy 0.0005 (`override_p2a.json`).
  `run_train.sh` takes `OVERRIDE_BACK`.
- **Left run 12 final** (21:42-23:41, `_run12.log`, model_2999), R4: posture starts settled 93.1 %, p50 0.66 mm /
  0.45 deg, p95 4.32 mm / 3.63 deg, within 1 cm 96.5 %, runaways 3.0 %; **< 90 deg of turn: settled 96.1 %, p50 0.65
  mm / 0.44 deg, p95 1.53 mm / 1.28 deg, within 1 cm 99.4 %: the P1 accuracy gate.** Workspace starts < 90 deg: p95
  1.68 mm / 1.56 deg. Holding a block (payload 1.0, force on, never trained with it): unchanged (p95 1.53 mm / 1.27).
- **P1 time gate:** `reach_check.py --dump` now records each goal's start and goal pinch (station frame); the planned
  transit's duration for the same goals from `tower_rl/skills/transit.py` (rise 2 cm, straight at travel height,
  descend; quintic segments at 0.25 m/s, 0.6 m/s^2, 1 rad/s, >= 0.3 s each; + 0.4 s settle). Settled goals, R4
  posture starts: left reacher p50 1.4 s / p90 2.1 s vs planned 3.9 / 5.5 s, faster on 99.7 %; back 1.5 / 2.1 vs 3.9 /
  5.4 s, 99.9 %; ratio p50 0.38-0.40 at every distance (0.45+ m: 1.9 vs 5.9 s). Caveat: the reacher's straight joint
  paths ignore obstacles; the pinch went 1.6 cm (left) / 9.4 cm (back) below the table top in these checks. P3 has to
  bring that to zero and will cost time.
- P2a (23:41): left resumed at 3,000 with the payload; by 4,086 settled 93 % in training (level R4, noise 0.01).
- **CPU PhysX** (`reach_check.py --device cpu`, 64 envs, 600 rows, R4 posture starts; the GPU-trained final models):
  goals < 90 deg: left settled 95.9 %, p50 0.65 mm / 0.44 deg, p95 1.64 mm / 1.25 deg, within 1 cm 98.6 % (GPU 96.1 %,
  1.53 / 1.28, 99.4 %); back 96.4 %, 0.79 / 0.38, p95 1.94 mm / 1.06 deg, 98.0 % (GPU 96.9 %, 1.72 / 0.94, 99.0 %).
  Settle p50 1.5 s on both. The policies carry over to the evaluator's physics.
- Table clearance (`reach_check.py` records `path_z_min`, the lowest pinch during each goal): R4 posture starts, final
  models: the pinch went below the table top on 0.1 % (left) / 0.2 % (back) of goals and below 1.5 cm on 1.0 / 1.2 %;
  path minimum p5 25-28 mm, the same as the lower of start and goal (p5 28 mm), so the paths rarely dip. The -1.6 /
  -9.4 cm minima above were single outliers (runaways). The table penalty does its job; P3's work is towers, blocks
  and the other arm.

### 2026-10-31 (session 8, cont.): reacher P2a results, task goals, speed limit

- **Left P2a** (23:41-01:46, `reach_left/2026-10-30_2341_47edad1d9d_p0/model_final.pt`: run 12's model_2999 resumed
  3,000 iterations at R4, a held block in 30 % of episodes, entropy 0.0005). R4 posture starts: settled 93.5 %, p50
  0.58 mm / 0.36 deg, p95 3.69 mm / 2.95 deg, within 1 cm 96.5 %; **< 90 deg: settled 96.5 %, p50 0.57 mm / 0.35 deg,
  p95 1.42 mm / 1.25 deg, within 1 cm 99.4 %** (run 12: 96.1 %, 0.65 / 0.44, 1.53 / 1.28). Holding a block: the same
  (p95 1.43 mm / 1.26 deg). Back P2a from 01:46 (to ~04:10; 2.8 s / iteration while CPU rehearsals run beside it).
- **Task goals (P2):** `rehearse.py --transit_log <file>` (new; `tower_rl/skills/reach_tasks.py` `TransitRecorder`, an
  executor node hook) records every transit when it starts: arm, target, held block, both arms' measured joints, the
  goal pinch pose (station frame), the graph's arrival tolerance, the planned rows, every block's pose; and its end
  (status, time). `results/reach/tasks/collect.sh` (teacher = the student_v1_3 bundle's executor, CPU PhysX, all ten
  variants, D0 + D1, 12 episodes each, seed 9200): 1,559 transits (left 762, back 797; pre-grasp 557, pre-place held
  476, pre-push 385, pre-insert held 74, pre-push-in 67); the planned transits: 1,547 OK, 8 timeouts, 4 failed. Only 8
  (0.5 %) need > 90 deg of turn from where the arm is.
- `scripts/reach_task_check.py` (new; `results/reach/tasks/task_check.sh`): each recorded goal replayed on the reacher
  from its start joints with its payload, 80 rows; settled, end error, inside the next skill's tight band
  (`reach_tasks.BANDS`: grasp 3 cm / 5 deg, place 2 cm / 5 deg, push 1.5 cm / 5 deg, insert / push-in 4 mm / 3 deg,
  their tightest axis) and the graph's tolerance, lowest pinch, pinch speed and acceleration (row to row), the planned
  transit's time.
- **Results (GPU):** left P2a: **in band 99.7 %** (pre-grasp, pre-place, pre-insert, pre-push-in 100 %; pre-push
  99.2 %), settled 87.3 %, arrival p50 1.7 s vs planned 5.7 s. Back P1 model: **99.3 %** (pre-grasp 99.0 %, pre-push
  98.0 %, the rest 100 %), settled 99.0 %, 1.8 vs 5.2 s. Of the 8 goals outside the band, 7 need > 90 deg of turn
  (they run away, as on random goals); the 8th is a back pre-push 54 cm away, still 10 mm / 7.5 deg off after 8 s.
  With the planned transit as the fallback above 90 deg: left 100 %, back 99.9 %.
- Left pre-push goals point 45 deg off vertical (back: 30 deg; the others 4-14 deg), rare among the random goals: the
  left reacher parks 2.9 mm / 1.9 deg off them (settled 68.5 %; inside the push band all the same).
- **Speed:** carrying a block to the pre-place pose, the reacher's pinch peaks at ~0.95 m/s and ~4-5 m/s^2 (row to
  row; p95 13 m/s^2 on the left), against the planned transit's 0.12 m/s held / 0.25 empty and 0.6 m/s^2. Tasks 4 / 5
  hold blocks in 3 N fingers that already slip under the planned transit, so the speed is a real risk.
- **New, opt-in (defaults keep the old behaviour):** `speed.*` in `configs/skills/reach.yaml`: a pinch speed limit per
  goal (log-uniform 0.08-0.6 m/s on `frac` of goals, else 2 m/s = none), an actor input appended last (`obs`; a
  checkpoint without it: `scripts/widen_push_checkpoint.py <in> <out> 64 1`, zero weights, then
  `--resume_no_optimizer`), a penalty `reward.speed` x relative excess, and limited goals last at least distance /
  limit + 2 s. `goal.tasks` / `reset.task_frac`: at R4 a fraction of goals (and of episode starts) come from the
  recorded transits (jittered +-2 cm, +-10 deg yaw); the IK oracle settles 96 % of them (left and back). Smoke runs:
  the widened left P2a model in the speed env behaves as P2a (p50 0.59 mm / 0.39 deg).
- Queued (`results/reach/queue_p2b.sh`, after P2a): **P2b**, each arm's P2a model widened and fine-tuned 2,000
  iterations at 1,024 envs with speed limits on half the goals, 30 % of goals and starts from the recorded transits
  (`override_p2b.json`). A held-out goal set (`heldout_*.jsonl`, seed 9600, 8 episodes per tier) for the checks.
- **Held-out task goals** (`heldout_*.jsonl`, seed 9600, 1,059 transits; `GOALS=heldout results/reach/tasks/task_check.sh`):
  left P2a 515 goals, back P1 544: **in band 100 % on both arms, GPU and CPU PhysX** (back: its 3 > 90 deg pre-push
  goals made it this time; the pinch dipped to 2.2 cm below the table top on them). Held-block pinch speed on CPU even
  higher (left p50 1.35 m/s vs 0.96 GPU).
- **Clearance** (`scripts/reach_clearance.py`, new; `reach_task_check.py --traj` dumps each goal's joint path): gripper
  axis points (gripper_base to 9 mm past the pinch, radius 2.5 cm) and the held block (its recorded offset from the
  pinch) against every other block / foundation box, mid-path only (pinch > 3 cm from start and goal: next to them the
  gripper is near blocks by design; the first version counted those and flagged 28-33 % of the planned transit's
  pre-grasps). **Direct reacher paths would cross other blocks on 37 % (left) / 17 % (back) of task transits** (pre-push
  68 / 31 %, held pre-insert 100 / 100 %, pre-grasp 21 / 15 %, held pre-place 14 / 5 %); the planned transit's paths
  1.6 / 1.3 %.
- **Via-points** (`reach_task_check.py --via`, rule in `reach_tasks.via_points`): the reacher gets the planned
  transit's shape as goals (above the start with the start orientation, above the goal with the goal orientation, the
  goal), the next one when within 1 cm / 20 deg. Travel height: 3 cm above the tallest block top (+3 cm holding a
  block), >= 2 cm above the goal, <= 0.33 m; a start below travel height + 5 cm first rises 5 cm straight up. Found on
  the way: the planned rule (max(start, goal) + 2 cm) put via-points at 0.40 m from the home posture, above the
  training goals (z <= 0.35), and the reacher parked 2.5 cm below them (pre-grasp in band 32 %); heading straight for
  the next via-point the reacher dipped ~1 cm in its first rows beside a block it had just placed (3 cm rise: 7.3 % of
  left pre-grasps grazed it, 5 cm: none); at 10 deg two Task 5 held pre-inserts stalled at a via-point with joint 7 at
  its limit (20 deg: none; 30 deg cut corners, 6 % contacts).
- **With via-points (held-out, GPU): left in band 100 %, contacts 1.0 % (planned 1.6 %), arrival p50 2.1 s (planned
  5.7); back 99.5 % (the three > 90 deg pre-push goals), 1.1 % (planned 1.3 %), 2.1 s (5.2).** Dips below the lower
  of start and goal by > 1 cm: 2.9 / 4.4 % of paths (direct: 24 / 30 %).
- **CPU PhysX differs at these speeds:** back on CPU the same (in band 99.5 %, contacts 0.7 %, dips > 1 cm 7.2 %); left
  in band 100 % but contacts 7.4 % and dips > 1 cm 15.9 %, up to 7 cm: descending 6 cm onto a pre-grasp pose it
  overshoots (pinch at z 0.036 for a 0.105 goal, beside the source tower's foundation) before settling, at 1.66 m/s /
  ~20 m/s^2 peak (GPU 1.19 / 10). Same solver settings on both (`configs/physics.yaml`). P2b's speed limit is the
  first answer; dynamics randomization (plan §2.4) the next if it persists.
- `results/reach/check_p2b.sh <arm> [ckpt]`: the battery for a P2b model (held-out goals with via-points, free and
  0.5 / 0.15 m/s limits, GPU and CPU, clearance, dips; R4 regression). `reach_task_check.py --override @file`.
- **CPU vs GPU, where the gap is not:** `scripts/reach_step_response.py` (new; q_des of one joint per env ramps at
  1 rad/s for 4 rows then holds, joint angles every substep): lag at the stop, overshoot and residual oscillation
  match within 1-2 mrad on CPU and GPU, both arms, every joint. Observation noise (`rand.q_noise_rad`, new, opt-in;
  the policy sees noisy joint angles, rewards use the true ones) of 0.5 mrad on GPU: left via-point check unchanged
  (in band 99.8 %, contacts 1.0 %, no pre-grasp contacts), so the policy is not fragile to small perturbations. The
  left CPU overshoots come from a systematic difference in fast multi-joint motion (next diagnostic: replay the same
  actions open-loop on both). The evaluator is CPU, so the gate stays on CPU.
- **Found it: open-loop replay** (`reach_task_check.py --traj` now also records each goal's actions; `--replay <dump>`
  plays them back without the policy): the left GPU run's actions replayed on GPU reproduce its paths (pinch within
  0.05 mm p50, 5 mm max); replayed on CPU PhysX the pinch strays p50 36 mm, p90 88 mm, max 12 cm from the GPU path
  mid-motion (pre-place held p50 79 mm), and ends at the same place (position control converges on the same q_des).
  Divergence starts at the first row, on several joints at once; the policy drives many joints at saturation (raw
  joint 2 action 3.4). `reach_step_response.py` downward at 1.5 rad/s: the stiff, lightly damped drives differ (left
  joint 2 kp 4815 / kd 1.1: lag at the stop GPU 20 / CPU 60 mrad, overshoot 16 / 59; joint 4 12 / 18 and 12.5 / 22;
  joint 3 0.4 / 2.6); at 1.0 rad/s joint 2 overshoots 3.8 / 17.7 mrad, at 0.6 rad/s 4.3 / 9.9. (The first, upward
  test matched only because joint 2 hardly moved there, against gravity at its 100 N m effort limit.) A numerical
  difference in how the CPU and GPU solvers integrate stiff drives at the 8.3 ms step; it grows with speed.
- **P2c** (`results/reach/queue_p2c.sh`, new): each arm's P2b model fine-tuned on CPU PhysX, 1,500 iterations at 128
  envs (`train_skill.py --device cpu` works: ~1,800 steps/s, 1.7 s / iteration; tag p2c, `override_p2c.json` = P2b's
  with noise 0.03), then the CPU task checks. It does not write `results/reach/train.pid`, so the keeper keeps the
  GPU loaded once P2b is done. The left one runs beside the back arm's GPU P2b.

### 2026-10-31 (session 8, cont.): P2b / P2c results, moving goal, P4 integration (stopped by the user ~08:15)

- **P2b's speed-limit input was not learned:** held pinch 0.85-0.90 m/s with a 0.15 m/s limit, the same as free;
  actor first-layer weight on the input ~0.04 against ~1.2 for the others; `reach/speed_over_frac` 0.19 -> 0.11 in
  250 iterations, then flat. The per-row penalty was about what arriving slowly costs in hold reward.
- **Moving goal instead** (`reach_tasks.Carrot`; `reach_task_check.py --carrot V_HELD V_EMPTY`): the goal advances
  toward each via-point at the speed limit (0.6 m/s^2, 1 rad/s), the reacher tracks it. Held-out goals, P2b models,
  0.15 held / 0.4 empty: in band 100 % left / 99.5 % back (GPU and CPU); held pinch 0.16-0.17 m/s; acceleration
  ~2 m/s^2 (was ~10); dips max 3.5-7.9 mm (CPU direct: 59-70 mm); paths through other blocks 1.9 % left / 1.1 % back
  on CPU (planned 1.6 / 1.3 %); arrival p50 2.8-3.0 s (planned 5.2-5.7). The CPU/GPU gap is gone at these speeds.
- **P2c** (CPU fine-tune, 1,500 iterations): without the moving goal left contacts 9.3 -> 4.3 %, back 0.6 %; with it
  no clear difference from P2b (left pre-push settled 97.5 -> 100 %). `results/skills/reach_*/*_p2c`.
- **Deployable reacher** `tower_rl/skills/reach_policy.Reacher` (no Isaac): `reach_task_check.py --parity` 0.0
  difference in observation, q_des and command over 515 goals. Exports in `results/reach/export/` (`tower_rl.skills.export`).
- **`executor.LearnedTransitNode`**, graph `reach: {mode: learned, policy: {left, back}, ...}` (`REACH_DEFAULTS`):
  via-points from block rest estimates, moving goal, one row per call, planned fallback above 90 deg of turn; options
  `held: planned`, `finish_rows` / `finish_mode` (IK finish into the planned end configuration), `q_des_from`,
  `driver: ik` (diagnostic). `TransitRecorder` also logs learned transits; `rehearse.py --transit_trace` per-row traces.
- **P4 rehearsals** (`results/reach/p4/rehearse_all.sh`, `scripts/reach_p4_compare.py`; held-out seeds 9600, 8
  episodes per tier, CPU, vs `results/reach/tasks/rehearsal_heldout`):
  - all transits learned (`p4/p2b`): TSR 108 / 160 vs planned 141 / 160. D0 Tasks 1-3 fine and ~40 % shorter; Task 5
    D0 0 / 8 (reacher pre-insert 0.8 deg off, insert needs < 0.5 deg; timeouts); D1 drops of held blocks.
  - held drops (trace): heavy blocks (1.4-2x) slide in the fingers from the first rows of a held transit; the IK
    controller following the same moving goal drops them too (7 / 8 first tries), gentler goals (0.12 m/s,
    0.15 m/s^2) still 4 / 8. Only the planned transit's slow quintic (0 / 8) keeps static friction.
  - Task 4: reacher arrives ~1.5 deg and 0.04-0.1 rad of elbow (joints 1 / 3) off the planned configuration; IK finish
    fixes the angle; Task 4-3 D0 over 16 episodes planned 12, IK finish 11, joint finish 10 (noise level).
  - **hybrid** (`p4/hp_ik8`: learned for empty transits, planned for held, 8-row IK finish): TSR 140 / 160 vs
    141 / 160. Tasks 1-3 and 5 equal or better on every cell, 3-19 % shorter episodes; Task 4 17 / 32 vs 19 / 32
    (4-2 D0 8 -> 6, 4-3 D0 6 -> 4, 4-3 D1 1 -> 3). Policy calls per episode +20-50 % (closed loop per row).

### 2026-10-31 (session 9): P2d, the reacher with a real block in the fingers (Reacher-V1.MD §11.1)

- Start checks: evaluation still not connected (server idle since 10-30 08:43); watcher re-armed; `pytest tests/` 307
  passed, 3 skipped (Isaac).
- **`held_block` in `reach_env.py` / `reach.yaml`** (opt-in, defaults unchanged): one station block per env (parked on a
  shelf when empty); held episodes spawn it between the fingers at a grasp offset from the 922 logged grasps
  (`policy_log` `grasp_offset_mm` x p5..p95 -3.5..2.1 mm, z -5.1..1.8, 5 % off-centre +-15 mm, rotation p50 1.3 deg),
  fingers in the Tasks 1-3 (22 N, 40 mm) or Task 4/5 (3 N, 42.5 mm) regime, `settle_rows` with the arm still; mass and
  block / finger friction ramp from the station's nominal to the D1 ranges; slip / drift / drop / table terms;
  `obs`: + opening and regime (actor), block in pinch frame + dynamics (critic). Recorded held starts and held goals
  (`load_task_pool(with_held=True)`), straight rises / descents, held goals 2x longer.
- **`scripts/reach_held_probe.py`** (IK moves at fixed pinch speeds, no policy, per D1 case): every case holds at rest
  (0.0 mm); slip starts at low speeds and sideways is worse than up: CPU, abrupt start, 2x heavy drops at 0.05 m/s
  sideways, nominal drifts 14 mm at 0.15 m/s up; a 4-row ramp or a quintic hardly helps along the block's long axis.
  So acceleration kicks at the 10 Hz command steps matter more than gravity.
- **`reach_task_check.py --block CASE --driver planned|policy`** (+ `results/reach/held/held_check.sh`): held-out held
  transits (left, 164) with a real block. Heavy, CPU: planned transit (graph speeds: v_max_held 0.08, Task 5 0.06)
  **1 drop / 164**, drift max p50 1.1 / p95 5.1 mm (station: 0 drops, ~2 mm creep); P2b reacher + via + carrot 0.15:
  **152 / 164 dropped** (station 7 / 8 first carries). GPU: planned 1, P2b 157. The light env reproduces the station's
  held slip on both back ends, so P2d trains on GPU and gates on CPU.
- `widen_push_checkpoint.py <in> <out> 65 2 22` (critic-only columns); `results/reach/ckpt/{left,back}_p2b_held.pt`;
  `reach_policy.Reacher` and `LearnedTransitNode` pass the measured opening and the regime (graph
  `gripper_closed_width`) to P2d models; `run_train.sh` takes `TAG=`.
- P2d left run 1 (noise 0.1) stopped after 3 min: 0.1 noise is ~0.056 rad/s of joint jitter, above the 0.05 settle
  limit (settled 0.0 vs P2b's 0.93; the hold reward vanished). Relaunched 10:45 with 0.05 (`override_p2d.json`).
- Planned-transit baseline per D1 case (held-out held transits, CPU; `results/reach/held/check_*_planned_*_cpu.json`):
  left 0-1 drops / 164 per case, drift max p95 nominal 0.7, light 0.2, heavy 5.1, uneven 3.2, slippery 0.7, high 0.4,
  combined 1.3 mm; back 0 drops / 208 in every case, p95 <= 0.7 mm. P2b + a slow moving goal (no training): 0.08 m/s
  18 / 164 heavy drops, 0.06 m/s 4 / 164 (drift p95 13.8 mm, arrival 9.1 s vs planned 11.7).
- P2d left with static goals (10:45-10:58, 175 iterations): drops per held episode 0.99 -> 0.95, slip 0.88 -> 0.73 mm
  per row. A goal 30 cm away and a fast policy: the slowdown must come from 0.05 exploration noise. Stopped.
- **`track` (P2e, Reacher-V1.MD §11.2) in `reach_env.py`**: a fraction of goals (`frac` empty, `held_frac` held)
  start at the pinch and slide to the sampled goal as a `reach_tasks.Carrot` with sampled speed (held 0.03-0.2 m/s
  log-uniform, empty 0.05-0.5), acceleration 0.2-1.0 m/s^2; `obs`: + the goal's velocity (6, base frame). Advanced in
  `_get_rewards` after the reward, so the policy sees the moved goal, as `LearnedTransitNode` does (carrot step, then
  the reacher). `Reacher.step(goal_v_w=)`, the executor passes the carrot's displacement per row;
  `reach_task_check.py` drives `env.goal_v` from its own carrot. Smoke: drops 0.02-0.17 per held episode (static goals:
  0.65-1.0).
- P2d-track left (`TAG=p2dt`, `override_p2d_track.json`: held 60 %, track 30 % empty / 80 % held, from
  `results/reach/ckpt/left_p2b_held_track.pt` = P2b widened 65 8 28) from 10:59. (A `pgrep -f` lookup killed the old
  run's `run_logged.sh` wrapper instead of its trainer; the orphan ran beside the new run for ~1 min until killed by PID.)
- `tests/test_reach_policy.py` (obs layout per option, goal velocity frame, `Carrot` limits, `via_points`): 4 passed;
  full suite passes. `reach_task_check.py --parity` with block + via + carrot + held/track inputs: 0.0 / 0.0 / 0.0.
- **Chunking** (`reach.chunk`, default 1; `reach_task_check.py --chunk K`): rows on the reacher's own q_des inside a
  call. Check tool, P2b, held-out goals, no block, GPU: K = 1 / 4 / 8 / 16 all in band 100 %, p50 0.66 mm / 0.6 deg;
  calls to arrival 29 rows -> 14 / 11 / 10; row-to-row pinch acceleration p50 1.9 -> 3.5-4.3 m/s^2. Executor, first
  version (chunks until the carrot is on the final goal, via switching on predicted poses): Task 1-3 TSR 0 / 16, the
  back arm stalled short of via-point 1 (the prediction has no sag / joint friction, so each chunk undid the closed-loop
  correction). Fixed: chunks only while the carrot travels, closed loop while it sits on any slot, via switching on
  measured poses only -> Task 1-3 16 / 16, ODR 0. But no call saving there: calls per episode planned 569 / 774, hybrid
  K=1 680 / 772, K=16 752 / 695 (D0 / D1); learned pre-grasp 21 calls (planned 8); the closed-loop tail (via
  convergence, settle test, 8-row IK finish) dominates. Next if needed: the IK finish in one call, arrival at the graph
  tolerance. Default stays 1.
- P2d-track left at 8500 (500 iterations), heavy held transits, CPU, via + carrot: 0.06 m/s 3 drops / 164 (drift p95
  9.8 mm, arrival 8.8 s), 0.08: 15, 0.10: 45, 0.15: 138, 1.0: 163 (held pinch 0.39 m/s: no self-limiting). P2b at
  0.06: 4, p95 13.8. Training drops ~0.4 per held episode, flat. Heavy blocks slip above ~0.1 m/s whatever the profile
  (probe), so fast / static held goals mostly add unavoidable drops. Refocused (`override_p2d_track2.json`, `TAG=p2dt2`,
  resumed from 8500 with its optimizer and noise 0.03): every held goal moving at 0.02-0.1 m/s, a 0.1-0.6 m/s^2, full
  D1 dynamics from the start (`ramp_steps` 1). First iterations: 0.05 drops per held episode.
- **P2d left gate met** (`p2dt2` model_8700, 200 iterations after the refocus; held-out held transits, 164 per D1 case,
  CPU, via + carrot 0.05 m/s): drops 0 in all seven cases (planned 0 / 0 / 1 / 0 / 0 / 0 / 1); drift max p95 nominal
  0.4, light 0.3, heavy 3.1, uneven 1.1, slippery 0.4, high 0.4, combined 0.8 mm (planned 0.7 / 0.2 / 5.1 / 3.2 / 0.7 /
  0.4 / 1.3); in band 100 %; arrival p50 10.4 s (planned 11.7). At 0.06 m/s: 8.8 s, but heavy 4 and uneven 2 drops.
  Left training plateaued (drops ~0.13 per held episode, slip ~0.04 mm per row); stopped 12:01.
- **Teacher, left P2d held transits** (`results/reach/held/rehearse_p2d.sh`, `held: {left: learned, back: planned}`,
  v_held 0.05, 8-row IK finish; the six variants where the left arm carries, vs the hybrid `hp_ik8`): TSR 90 / 96 vs
  91 / 96; Tasks 1-2, 1-3, 2-2, 2-3 8 / 8 on D0 and D1; 5-2 equal; 5-3 D0 7 / 8 vs 8 / 8, but that episode fails in the
  back arm's push-in and the runs diverge at its first grasp (lift 61.3 vs 62.7 mm), before the left arm moves: CPU
  PhysX run-to-run noise. No time gain at 0.05 m/s (episodes within 1 s); calls +30-50 % (one row per call).
  **Task 1-3 D1 heavy only: 8 / 8, 0 drops** (32 learned left carries, all OK; the all-learned P2b config: 0 / 8).
- **The back arm carries fast**: P2d back (`p2dt2`, model_8200, from P2b widened 65 8 28, `override_p2d_track2_back.json`)
  on 208 held-out heavy held transits: 0 drops at 0.05 / 0.08 / 0.10 / 0.15 / 0.2 / 0.3 m/s, drift p95 0.7-1.0 mm,
  arrival 10.1 / 6.8 / 5.7 / 4.4 / 3.9 / 3.7 s (planned 10.6); Task 5-3 pre-inserts (3 N) 0 drops, 3.5-4.3 s vs 12.1,
  0.5 deg off (the IK finish covers it). Untrained P2b back at 0.15: also 0 drops (p95 0.8). The back arm's soft drives
  (j1 kp 63, j4 68) filter the 10 Hz command steps; the left arm's stiff, lightly damped joint 2 (kp 4,815, kd 1.1)
  turns them into jolts. P4's `held: planned` was only needed for the left arm. `reach.v_held` / `v_empty` may now be
  per arm (dict).
- Rehearsal of all ten variants, both P2d models for every transit, v_held {left 0.05, back 0.15} (`p4/p2d_both`),
  started 12:27.
- **Teacher, both P2d models for every transit** (`p4/p2d_both`: left model_8700, back model_8200, `held: learned`,
  v_held {left 0.05, back 0.15}, 8-row IK finish, one row per call; CPU, ten variants, D0 + D1, 8 episodes, seed 9600):
  **TSR 142 / 160** (planned 141, P4 hybrid 140); GSR 586 / 608 (planned 588), PSR 577 (577). Tasks 1-3 and 5 equal or
  better on every cell; Task 1-3 episodes 15-36 % shorter (1-2 81.9 -> 63.5 s, 1-3 127.8 -> 109.0, 2-2 48.0 -> 33.7,
  2-3 66.2 -> 47.5, 3-2 37.0 -> 23.6, 3-3 56.5 -> 36.5; 5-3 82.0 -> 71.6). Only cell worse than planned by > 1 / 24:
  4-2 D0 8 -> 6 (as the hybrid); 4-3 D1 1 -> 3. Calls per episode ~2x on Tasks 1-3.
- **Chunking held carries fails**: P2d left heavy at 0.05 m/s, `--chunk 4 / 16`: 158 / 160 drops of 164 (closed loop
  0), pinch 0.25-0.34 m/s; q_des less the measured offset (`--chunk_pred offset`): 159. Back heavy 0.15: 3 (K=4), 16
  (K=16) of 208 (closed loop 0). Open-loop rows make the reacher carry at ~2x the moving goal's speed; it was trained
  closed loop against the real arm. Empty transits chunk fine (P2d left, K=16: identical accuracy, calls 10-16 vs
  20-30 rows). `reach.chunk_held` (default 1): held transits stay closed loop. `finish_chunk`: the IK finish rows in
  one call. Named config `configs/reach/teacher_p2d.json` (`rehearse.py --graph_set @file` now works); its rehearsal
  `p4/p2d_both_k16` started 12:54.
- **Chunked config in the teacher** (`p4/p2d_both_k16`: chunk 16 for empty transits, held closed loop, IK finish in
  one call): TSR 138 / 160 vs 142 unchunked, and slower with more calls (3-2 D0 23.6 -> 38.0 s, 2-3 D0 436 -> 1,176
  calls). Trace (`results/reach/chunk/trace_t32_k16.jsonl`): Task 3-2's first back-arm pre-grasp times out at its
  final goal in all 4 episodes, the pinch alternating every row between two poses 1.5 cm apart, 7.5-9 mm from the goal,
  for 12 s of closed loop; the retry arrives in 2.9 s. 16 open-loop rows of travel leave the arm where the policy has a
  2-cycle (likely null-space drift: in prediction the joints move without moving the pinch). Chunking needs training
  with chunked observations; `configs/reach/teacher_p2d.json` now has no chunking (`finish_chunk` kept: the same rows,
  7 fewer calls per learned transit).
- **Back P2d gate met at 0.15 m/s** (model_8200, 208 held-out held transits per D1 case, CPU): 0 drops in all seven
  cases, drift max p95 nominal 0.5, light 0.3, heavy 0.8, uneven 0.7, slippery 0.5, high 0.5, combined 0.7 mm
  (planned 0.5 / 0.3 / 0.7 / 0.7 / 0.5 / 0.5 / 0.6), in band 100 %, arrival p50 4.4 s (planned 10.6).
- **Task 4 with `planned_targets: ["prepush"]`** (`p4/p2d_t4planned`): reproduces the planned run exactly (8 / 4 / 6 / 1
  of 8, the same times and calls): every Task 4 transit is a pre-push. P2d's learned pre-pushes also total 19 / 32.
  Correction to the 5-3 note above: rehearsals are deterministic for identical inputs; the 4 envs of a process share
  one CPU PhysX scene, so changing one env's actions perturbs the others (cross-env coupling, not run-to-run noise).
- Hand-off: `corl_challenge/Reacher-V2.MD`.

### 2026-10-31 (session 10): a chunk-trained reacher (Reacher-V2.MD §7.1)

- v1.3 official evaluation at 14:39: 61 sessions, 4,682 calls, 0 errors, mean call 124 ms (training off 13:46-14:52).
  27 sessions were reset and dropped before their first call ("empty"): the evaluator's workers reconnecting, as in
  official run 1 (85 of 210 resets).
- **`reach_env.py` option `chunk`** (default `lengths: [1]`, off): rows grouped into chunks as the evaluator executes a
  call. A chunk's first row sees the measured joints; the rows after it the reacher's own prediction, q_des less the
  offset q_des - q measured on the first row (sag and lag at that moment), the finger opening measured there, qd from
  the previous row's view. Lengths drawn per chunk from `lengths` / `weights`; a new goal starts a new chunk. `obs`:
  + rows since the measured row / 16 (actor, last input), the true joints and pose error (critic, right after the
  actor's inputs; `widen_push_checkpoint.py <in> <out> 73 1 17`). Rewards use the true state.
- `reach_policy.Reacher.step(None, ...)`: a row inside a chunk on `predicted()` = q_des - offset; `measure(q)` re-measures
  the offset. `reach_task_check.py --chunk K --chunk_final chunk`: chunks to the end (default `closed`: the old
  executor rule); `--chunk_pred` default now `offset` (the deployable's rule); `--parity --chunk 16 --chunk_final chunk`:
  the env's own chunks vs the deployable's, **0.0 / 0.0 / 0.0** (obs / command / q_des, 36 transits, held + empty).
- Executor (`LearnedTransitNode`, chunk-trained reacher and `reach.chunk` / `chunk_held` > 1): `_trained_chunk` runs the
  rest of the call's rows on the prediction, switches via-points on the predicted pinch pose and ends the chunk early
  once the carrot is on the goal, q_des has been still for `settle.rows` rows and the predicted pinch is within
  `settle`; arrival on the next call's measured pose with q_des still (one measurement per call). Models trained closed
  loop keep the old rule.
- Baseline, P2d left widened (new inputs zero) under 16-row chunks to the end, CPU, Task 2-2 held-out, nominal block:
  held 16 / 18 dropped, pre-grasps 0 / 18 settled (p50 10 mm). `tests/test_reach_policy.py` + chunk test: 5 passed.
- **Training `p2fck`** (14:52, 1,024 envs, 1,000 iterations per arm, left then back): from P2d left model_8700 / back
  model_8200 widened (`results/reach/ckpt/{left_p2d8700,back_p2d8200}_chunk.pt`), `results/reach/override_p2f_chunk.json`
  / `_back.json`: P2d's track2 settings + chunk lengths 1 / 4 / 8 / 16 (weights 0.15 / 0.15 / 0.2 / 0.5), empty moving
  goals 0.6 (was 0.3), back held goals 0.03-0.2 m/s (deployed 0.15), noise reset 0.04. First iterations: held drops
  0.36 per episode (P2d end ~0.13), settled 0.19.
- **Left, chunked gate met at model_9000** (300 iterations; CPU, held-out transits, via + carrot 0.05 / 0.4, 16-row chunks
  to the end): all 515 transits in band 99.8 %, settled 99.4 %, p50 0.9 mm / 0.84 deg, p95 1.3 / 1.12, arrival p50
  3.7 s (planned 5.7), **calls to arrival p50 3** (pre-grasp 3, held pre-place 7, pre-push 3, held pre-insert 6,
  pre-push-in 2; closed loop: one per row). Seven D1 held cases (164 each): **0 drops in all**, drift max p95 nominal
  0.4, light 0.2, heavy 2.9, uneven 1.1, slippery 0.6, high 0.3, combined 0.8 mm (closed-loop P2d: heavy 3.1; planned
  5.1), arrival 10.5 s (planned 11.5). Misses: one Task 5-3 held pre-insert (`9600003/67`) parks at via-point 1 at
  20.2 deg, past `via_tol` 20 (closed loop reached it: the wrist-limit case); two 78 deg pre-grasps end 2.7-2.8 deg
  off (in band, not settled). model_8800 (100 iterations) was already 0 / 164 heavy drops; the held pre-insert drift
  p95 ~16 mm is the same closed loop (16.1), not from chunking.
- Teacher rehearsal `p4/p2f_left9000_k16` (left model_9000 chunk 16 for every transit, back P2d closed loop; `chunk` /
  `chunk_held` may now be per-arm dicts) started 15:21.
- **Teacher, left chunked** (`p4/p2f_left9000_k16`: left model_9000 16-row chunks for every transit, held included;
  back P2d closed loop; teacher_p2d's settings; CPU, ten variants, D0 + D1, 8 episodes, seed 9600): **TSR 141 / 160**
  (planned 141, P2d both closed loop 142), GSR 586 (586), PSR 575 (577). Tasks 1-3 8 / 8 on every cell, episode times
  as P2d (1-2 64.5 s vs planned 81.9, 2-2 34.0 vs 48.0, 3-3 36.5 vs 56.5). Calls per episode vs P2d closed loop / planned:
  1-2 D0 392 (624 / 313), 1-3 D0 628 (1,084 / 569), 1-3 D1 773 (1,072 / 774), 2-2 D0 169 (314 / 260), 3-3 D0 339 (366 /
  212; the back arm still one row per call). 4-2 D0 back to 8 / 8 (P2d 6 / 8), 4-3 D0 7 / 8 (planned 6), 4-3 D1 0 / 8
  (planned 1, P2d 3), 5-2 D0 7 / 8 (one push-in disturbed the structure, 62 mm, after a 1.5 mm / 0.3 deg pre-push-in
  arrival). `act_calls` is per batch of 4 envs (a batch runs until its last episode ends), so one long episode
  raises its batch's count (5-2 D0 1,105).
- **The missed pre-insert is the model, not chunking**: model_9000 closed loop parks it the same (4 mm from via-point 1,
  20.2 deg, joint 7 at its 0.03 rad limit margin; P2d 8700 found another posture). New `reach.via_stall_s` (executor,
  default 0 = off) / `reach_task_check.py --via_stall S`: the next via-point also once the carrot has sat on the current
  one S seconds with the pinch within `via_tol`'s position part (via-points are for clearance). With 2.0 s that
  transit arrives (0.7 mm / 0.52 deg, 12.1 s, 8 calls). `check_chunk.sh` / `rehearse_p2f.sh` use STALL=2.0.
- **Left final (`p2fck` model_final = 9700, 1,000 iterations; exported `results/reach/export/reach_left_p2fck9700.pt`)**,
  battery with `via_stall` 2 s (16-row chunks to the end, CPU): all 515 held-out transits in band and settled 100 %,
  p50 1.1 mm / 0.62 deg, p95 1.5 / 0.95, arrival p50 3.6 s (planned 5.7), calls to arrival p50 3 (held pre-place 7);
  seven D1 held cases **0 drops**, in band 100 %, drift max p95 nominal 0.4, light 0.2, heavy 3.6, uneven 1.1, slippery
  0.6, high 0.4, combined 0.8 mm. model_9500: 1 heavy drop (p95 3.8). Training end: settled 0.92, held drops 0.10 per
  held episode (P2d ~0.13). Chosen over 9000 (rotation p50 0.62 vs 0.84 deg; 9000 heavy drift p95 2.9).
- Back `p2fck` (16:02): starts much worse than the left (settled 0.00 -> 0.15 in 95 iterations; left 0.19 -> 0.62 in
  65), held drops 0.25 -> 0.02. The back arm's joints lag q_des more (time constants (kd + viscous) / kp: j4 0.11 s, j5
  0.08, j7 0.14; j1 kp 63 lightly damped) than the left's (< 0.07 s; j2 / j3 ~0.002), so the constant offset held over a
  chunk is further from the truth once the arm decelerates.
- Teacher `p4/p2f_left9700_k16` (left final chunked + via_stall 2 s, back P2d closed loop) started 16:11.
- **Teacher, left final chunked** (`p4/p2f_left9700_k16`, via_stall 2 s, back P2d closed loop): TSR 140 / 160 (planned
  141), **GSR 592, PSR 582, LHPR 1,174** (planned 588 / 577 / 1,165; P2d both 586 / 577 / 1,163). Tasks 1-3 8 / 8
  everywhere, calls 1-3 D1 648 (planned 774), 2-2 D0 177 (260), 1-2 D0 392 (313). TSR differences only on the
  contact-sensitive cells: 4-2 7 / 8 + 3 / 8 (planned 8 + 4), 4-3 4 + 4 (6 + 1), 5-2 6 + 6 (8 + 5), 5-3 8 + 6 (8 + 5).
  5-2 D0's push-in moved the structure more after the learned pre-push-in transit (moved p50 / p90 / max: planned 2.6 /
  5.1 / 6.5 mm, P2d 5.6 / 7.8 / 8.9, chunked 9000 5.5 / 24.6 / 62.3, 9700 4.0 / 23.4 / 24.2: 2 structure failures),
  the pinch arrival identical (1.5 mm / 0.3 deg): the contact skills' sensitivity to the arrival posture, as Task 4's
  pushes. Pre-push (3 s) and pre-push-in (2.5 s) gain nothing from the reacher, so `planned_targets: ["prepush",
  "prepushin"]` is the candidate deployment config.
- **Back chunked gate met at model_8500** (500 iterations; 16-row chunks to the end, via_stall 2 s, carrot 0.15 / 0.4,
  CPU): all 544 held-out transits in band 99.5 %, p50 0.9 mm / 0.35 deg, p95 1.9 / 1.19, arrival p50 3.9 s (planned
  5.2), calls to arrival p50 3 (held pre-place 4); seven D1 held cases (208 each) **0 drops**, in band 100 %, drift
  max p95 nominal 0.5, light 0.3, heavy 0.8, uneven 0.7, slippery 0.5, high 0.5, combined 0.7 mm (closed-loop P2d:
  0.5 / 0.3 / 0.8 / 0.7 / 0.5 / 0.5 / 0.7), held arrival 5.3 s (planned 10.6; P2d 4.4). The 3 misses are Task 4-3 D1
  pre-pushes (seed 9700000, 147 / 154 / 158) stuck at the first via-point, the same with closed-loop P2b (pinch to
  -21 mm). Training metrics: settled 0.00 at the start -> 0.86 at 8547 (P2d end 0.90), held drops 0.25 -> 0.
- Teacher rehearsals with both chunked models (left 9700, back 8500, chunk 16 everywhere, via_stall 2 s), 16:36, one
  process each: `p4/p2f_both_k16` (every transit learned) and `p4/p2f_both_k16_pp` (+ `planned_targets: ["prepush",
  "prepushin"]`).
- **Server timing in the v1.3 evaluation** (per-call `est_ms` / `policy_ms` / `total_ms` in the server's episode rows;
  perception dominates, the policy is 10-21 ms): mean total per variant 1-2 252 ms, 1-3 280, 2-2 135, 2-3 172, 3-2 238,
  3-3 304, 4-2 437 (p90 1,306; 6-33 calls > 1 s per episode, all in push nodes). Lower while only the keeper's stand-in
  ran (2-2 / 2-3, 13:46-14:52) than with the reacher training on the GPU. Official run 1 (no training beside it):
  4-2 perception mean 120 ms, p90 143, 7 of 18,292 calls > 1 s; 1-2 202 ms (171 > 1 s). v1.2's attempt (no GPU work):
  1-2 85 ms. So GPU training costs the server's perception 2-3x, most on Task 4; 0 errors so far.
- **Teacher, both chunked** (left 9700, back 8500, 16-row chunks for every learned transit, via_stall 2 s, CPU, ten
  variants, D0 + D1, 8 episodes, seed 9600):
  - **`p4/p2f_both_k16_pp` (+ `planned_targets: ["prepush", "prepushin"]`): TSR 141 / 160, GSR 588, PSR 576, LHPR 1,164
    (planned 141 / 588 / 577 / 1,165); no cell worse than planned.** Task 4 reproduces the planned run exactly (every
    Task 4 transit is a pre-push). Tasks 1-3 8 / 8 everywhere, episodes 15-34 % shorter (1-2 69.4 vs 81.9 s, 2-2 34.6
    vs 48.0, 3-3 38.8 vs 56.5). **Calls per episode: all cells 13,250 vs planned 13,352 (-0.8 %); Tasks 1-3 3,330 vs
    3,646 (-8.7 %)**; per cell 1-2 384 / 340 (313 / 325), 1-3 528 / 710 (569 / 774), 2-2 130 / 146 (260 / 166), 2-3 244 /
    208 (243 / 252), 3-2 112 / 120 (136 / 150), 3-3 196 / 212 (212 / 246), 5-2 232 / 1,988 (225 / 1,961), 5-3 500 /
    1,974 (464 / 1,830). P2d closed loop: ~2x planned on Tasks 1-3. Gate (plan P4 + Reacher-V2 §7.1: no cell worse
    than planned by > 1, calls <= planned + 10 %) met.
  - `p4/p2f_both_k16` (every transit learned): TSR 140, **GSR 596, PSR 584, LHPR 1,180** (more goals placed than
    planned), but 4-3 D0 5 / 8 (6) and 5-2 D0 6 / 8 (8): learned pre-push / pre-push-in arrivals again.
  - The 2-row limit cycle of closed-loop chunking (Task 3-2's back pre-grasp, 10-31 morning) is gone: 3-2 8 / 8, 24.8 s.
- Named config **`configs/reach/teacher_p2f.json`** (the `_pp` settings).
- **Back final (`p2fck` model_final = 9200, 1,000 iterations; `results/reach/export/reach_back_p2fck9200.pt`)**, battery
  as 8500: all 544 in band 99.5 % (the same 3 pre-pushes), **p50 0.6 mm / 0.32 deg, p95 1.0 / 1.16** (8500: 0.9 / 0.35,
  1.9 / 1.19), pre-grasp calls 2; seven D1 held cases 0 drops, in band 100 %, drift max p95 nominal 0.5, light 0.3,
  heavy 0.8, uneven 0.7, slippery 0.5, high 0.5, combined 0.7 mm, held arrival 5.2 s. Teacher with it
  (`p4/p2f_both9200_k16_pp`, the teacher_p2f settings, one process beside the camera loop) started 17:28.
- **Student camera loop with the chunked reachers** (`results/reach/chunk/gpu_student_cam.sh @configs/reach/teacher_p2f.json
  p2f_t13`; student_v1_3 bundle, bc_v2_r3, kp_v2 on our renders, mirror, `--official_orient`, ten D1 cases, seed 7200
  x 10 per Task 1-3 variant; v1.3 Step 7a reference 59 / 60): started 17:19 (the script's pid in `train.pid`: the
  keeper stopped its stand-in at 17:21:55).
- Camera loop, Task 1-3 (17:19-17:55): **9 / 10** (reference r3 10 / 10), PSR 0.983, ODR 0. The failure (seed 7200001,
  lightweight) is goal 0's first place, judged on the final arrangement: grasp rotated 2.8 deg, the learned held transit
  arrived 3.6 mm / 1.39 deg from the pre-place goal, the place ended at yaw 4.99 deg. The reference run on the same seed:
  grasp 2.6 deg, planned transit 3.5 mm / 1.36 deg (the same arrival: the goal, not the reacher), place yaw 4.69 deg,
  success. A marginal yaw at the tolerance in both. Batches take 435 s vs the reference's 302 s.
- Teacher with back 9200 (`p4/p2f_both9200_k16_pp`): Tasks 1-2, 1-3, 2-2, 3-2, 3-3 8 / 8 everywhere; 2-3 7 / 8 + 7 / 8
  (both left-arm `t1_place` failures at 35 s with the left model unchanged: cross-env coupling with the back arm's
  different actions); Task 4 = planned; 5-2 8 / 8 + 5 / 8.
- **Teacher with back 9200** (`p4/p2f_both9200_k16_pp`): TSR 139 / 160, GSR 588, PSR 574, LHPR 1,162 (8500: 141 / 588 /
  576 / 1,164); the 2 lost episodes are the 2-3 left-arm places above (8 / 8 -> 7 / 8 per tier, their batches' calls
  948 / 990). Other cells equal, times within ~2 s, calls within ~10 %. `teacher_p2f.json` keeps back 8500 (the camera
  loop uses it); 9200 is the more accurate alternative (§ battery) if a later A/B with more episodes favours it.
- Student runtime cost of the chunked reacher (camera loop, Task 1-3, `act_ms_mean`): 24.2 ms per call vs 18.8 ms
  (reference r3): ~5 ms for up to 16 reacher rows (FK + MLP per row); max 1,542 vs 1,497 ms. Calls per mirrored
  episode 614 vs 618 (episodes run to 210 s; idle chunks after the plan). Queued after the Tasks 1-3 camera loop:
  Task 5 the same way (`results/reach/chunk/after_cam_t13.sh` -> `gpu_student_cam2.sh`, SEED 7400, no orient flag,
  `results/reach/student/p2f_t5`; reference r3 t5_cand: 5-2 8 / 10, 5-3 2 / 10, 23.34 points).
- v1.3 evaluation at 18:44: 154 sessions, 79 episodes to the limit, 8 open, 0 errors, mean call 241 ms; Tasks 1-2 to
  4-3 done (9-10 each), 5-2 4 done + 6 open, 5-3 2 open.
- Camera loop (student + chunked reachers): Task 1-3 9 / 10, 2-3 10 / 10, 3-3 10 / 10 (reference r3: 10 / 10 each).
- **Student camera loop, Tasks 1-3, with the chunked reachers (`results/reach/student/p2f_t13`, 17:19-19:54): 59 / 60,
  33.02 points** (v1.3 Step 7a reference r3, same seeds: 59 / 60, 32.87). 1-2, 2-2, 2-3, 3-2, 3-3 10 / 10; 1-3 9 / 10
  (the marginal yaw above; LHPR 99.2). The reference's miss was 2-2 high_friction (LHPR 95.0). No reworks; the
  switcher (bc_v2_r3, trained with planned transits) handled the learned ones without a DAgger round. Task 5 started
  19:54 (`p2f_t5`).
- **Student camera loop, Task 5, with the chunked reachers (`results/reach/student/p2f_t5`, seed 7400 x 10 per
  variant, 19:54-20:57): 5-2 8 / 10, 14.45 points; 5-3 5 / 10, 12.10 points; 26.55 points** (v1.3 reference
  r3/t5_cand: 8 / 10 14.34, 2 / 10 9.00; 23.34). 5-3 LHPR 85 vs 72.5, ODR 0.22 vs 0.40. The difference is the first
  block: the back arm's learned held pre-insert arrives in 4.8 s (planned 12.4) at the same accuracy (3.3 vs 3.0 mm,
  0.5 deg), the first insert goes deeper (along p50 -91.2 vs -87.6 mm; 0 vs 1 insert failures) and its push-in fails
  in 1 episode instead of 5; the second block's push-in fails alike (6-7). 5-2's push-in misses: 4 in 2 episodes vs 2
  (pre-push-in planned in both: camera-loop noise). The privileged teacher showed no 5-3 difference (5 / 8 both), so
  this is a perception-loop effect (a shorter 3 N carry, the estimate at the insert?); 10 episodes, noisy.
- Camera loop in all (eight variants, 80 episodes): 72 / 80, 59.57 points vs the reference's 69 / 80, 56.21.
- Second-seed check of the 5-3 lead (`results/reach/chunk/cam_t53_seed7500.sh`, 21:01): Task 5-3 camera loop, seed
  7500 x 10, the reacher config (`results/reach/student/p2f_t53_7500`) then the unchanged student_v1_3 bundle
  (`results/reach/student/ref_t53_7500`), same code.
- **The 5-3 lead is not confirmed** (seed 7500, `results/reach/student/{p2f,ref}_t53_7500`, same code): the reacher
  config 4 / 10 (LHPR 80, 10.94 points) vs the unchanged student_v1_3 bundle 7 / 10 (LHPR 90, 13.71). Over seeds 7400 +
  7500: 9 / 20 each, 23.04 vs 22.71 points. The bundle's first insert went no deeper (along p50 -88.2 mm) and still
  succeeded, so the depth was not the cause. Task 5-3 camera results swing 2-7 / 10 for one config across seeds; the
  reacher is neutral there.
- v1.3 evaluation at 21:50: 99 episodes to the limit, 1 open (4-3, the last), 0 errors.
- **v1.3 official evaluation complete from the server's side** (11:37:27-22:15): 172 sessions, **100 episodes, 10 per
  variant, all to the 210 s limit, 0 errors**, 72 empty sessions (reconnects: reset, then a new session), last call
  ~22:15. Scores arrive on the challenge site. The keeper's stand-in still runs (stop: `touch results/reach/STOP_BG`).

## 2026-11-01 (session 11): `student_v1_4` = `student_v1_3` + the chunk-trained reachers (user: "build student_v1_4")

Research bundle beside the final submission (`student_v1_3`, untouched); not served (no endpoint, no tunnel).

- **Code (bundle path, Reacher-V3.MD §6.3):** `ExecCfg.bundle` (set by `Executor.from_bundle`, carried into
  `StudentExecutor` with the cfg); `executor.reach_policy()` loads a relative `reach.policy.<arm>` path from the bundle
  when the bundle has the file, else relative to the working directory (so `--graph_set @configs/reach/*.json` on any
  bundle works as before). Test `test_reach_policy_bundle_paths` (bundle copy wins, working-directory fallback, the
  student runtime keeps the bundle dir); suite 314 passed, 3 skipped (Isaac).
- **Code since `server_v1_3`** (10-29): in the runtime only `executor.py` changed (the `_smoothstep` import, the transit
  dispatch behind `reach.mode: learned`, the new REACH_DEFAULTS / reach_policy / LearnedTransitNode) + the new
  `skills/reach_{env,policy,tasks}.py`; configs only gained `configs/reach/` and `configs/skills/reach.yaml`. A graph
  without a `reach` block runs exactly v1.3's code paths. All Task 4 transits are `prepush` (planned in teacher_p2f),
  so Task 4 behaves as in v1.3.
- **Bundle** (`results/v1_4/build_v1_4.sh`, log `results/v1_4/logs/build_v1_4.log`): `freeze_student_bundle.py
  --teacher student_v1_3` (its graphs / skills / configs), v1.3's switcher (bc_v2_r3), kp_v2 and orient_refs, `--extra
  reach/reach_left_p2fck9700.{pt,json}` + `reach/reach_back_p2fck8500.{pt,json}`, `--graph_set` = teacher_p2f.json's
  reach block with bundle paths (`reach/<export>.pt`) on every graph. Checked: 63 files byte-identical to v1.3
  (switcher, skills, perception, references, specs, configs), the 10 graphs = v1.3's + exactly that block, the 4 new
  files identical to the exports; MANIFEST verified and both reachers load from the bundle with cwd /tmp.
  `results/bundles/student_v1_4` (77 files, 106 MB), tar sha256 `78d27952a2b9e9caf74010ca6126304109c3b44a29d71989e295716533b15a60`
  (the server's artifact_identity). Manifest git commit f5e8cce dirty (sessions 8-11 uncommitted): the code is in the
  snapshot.
- **Server snapshot** `~/Documents/tower_rl_snapshots/server_v1_4` (`scripts/make_snapshot.sh`; tower_rl/ identical to
  the repo at 06:00).
- **Checks** (`results/v1_4/gpu_checks_v1_4.sh`, one GPU job, pid in `results/reach/train.pid`, started 06:03):
  1. `gpu_ws_e2e_v1_4.sh`: the server path end to end, as v1.3's e2e (port 9005, 1-3 / 2-2 / 4-2 / 5-3, D1 nominal +
     combined, seed 7600, mirror), alone on the GPU; compare `results/v1_3/ws_e2e/e2e_v1_3*`.
  2. `cpu_parity_v1_4.sh` (CPU, oracle perception): student on v1.3 + `--graph_set @configs/reach/teacher_p2f.json` vs
     on v1.4, seed 7700, 1-3 / 2-3 / 3-2 / 4-3 / 5-2, D1 nominal + heavy, 2 episodes; records must be identical.
  3. `gpu_cam_v1_4.sh`: the Tasks 1-3 camera loop from the bundle (seed 7200 x 10, ten D1 cases, --official_orient);
     compare `results/reach/student/p2f_t13` (59 / 60, 33.02).
- **e2e through the v1.4 server (06:05-06:27, `results/v1_4/ws_e2e`, server logs `results/student/server/e2e_v1_4`):
  0 errors in 8 episodes** (6 to the limit, 2 5-3 left open by the client at the end, as in v1.3's run). Same seeds as
  v1.3's e2e: 1-3 2 / 2 (v1.3 2 / 2), 2-2 2 / 2 (2 / 2), 5-3 TSR 0 / GSR 1.0 / PSR 0.25 (identical to v1.3), 4-2 1 / 2
  vs 0 / 2 (every Task 4 transit planned in both: camera run-to-run noise). Transit calls per episode 71-73 vs
  116-117 (1-3), 21-22 vs 40 (2-2), 26-32 vs 47-53 (5-3); a reacher transit call 75-80 ms of policy time vs 45-50 ms
  planned (16 rows, ~1.7 ms per row), so policy ms per call (mean) 20.4-20.8 vs 19.0 (1-3), 12.0-12.1 vs 11.9-12.2
  (2-2), 10.1-10.3 vs 10.3-11.3 (5-3). Server total mean 104-114 vs 102-110 ms, mostly perception (est_ms).
  Client latency mean 138-206 ms vs 128-228.
- **CPU parity** (`results/v1_4/parity`, compare with `results/v1_4/compare_records.py`; the scripts' own whole-record
  test counts the wall-clock fields act_ms_mean / act_ms_max / wall_s, so their "FAILED" lines are that):
  v1.3 + teacher_p2f vs v1.4 identical up to timing on 1-3, 2-3, 3-2, 4-3, 5-2 (`compare_up_to_timing.txt`); v1.3
  alone vs v1.4 identical on 4-2 and 4-3 (`compare_t4_v13_alone.txt`); negative control v1.3 alone vs v1.4 on 1-3
  differs (last place 127.4 / 129.9 s vs 115.4 / 115.6 s; v1.3 alone 1 / 2, v1.4 2 / 2).
- **Camera loop from the bundle (`results/v1_4/camera`, 06:27-08:41):** Tasks 1-3, 60 episodes: 59 / 60, 33.02
  points (`t13_points.md`) = p2f_t13 (59 / 60, 33.02; v1.3 alone 59 / 60, 32.87); 1-3 TSR 0.9 / PSR 0.983 / LHPR
  0.992 (lightweight fails in both), the other five 10 / 10. `compare_camera.py` (now also whole-episode router node
  sequences; output `camera/compare_p2f_t13.txt`): 60 / 60 episodes alike (outcome, per-goal grasp / place, event
  sequence); event times median 0.0-0.1 s per variant, max 3.2 s, except 8.9 s in 3-2 heavy (the camera path is not
  bit-exact run to run). Node sequences equal in 58 / 60, one each way: 3-2 heavy, p2f_t13's first place failed and
  it re-grasped / re-placed (+8.9 s), v1.4 placed first time; 1-3 lightweight, the first place misses in both runs
  (place_ok[0] false, no P0 event) but the node reported ok in p2f_t13 and failed in v1.4, and both routers went on
  to the next block. The `events` list only records successes, so retries show only in the node sequence.
  mirror_points' "goal nodes after success" (1-2 8 vs 9, 1-3 4 vs 1) differs only by the final place node, logged
  either side of t_first_success + 0.05 s. Batches 346 s vs 435 s (GPU free). At the end train.pid was removed and
  the keeper's stand-in restarted (pid 1777950).
- **v1.4 served (12:37; user: "lets get student v1.4 evaluated"):** `results/v1_4/serve_v1_4.sh` (port 9005, tmux
  tower_v1_4, snapshot server_v1_4, policy_id tower-student-v1-4, artifact sha256:78d27952...), model URL
  wss://bodies-vat-document-bottle.trycloudflare.com, server pid 1802014; probes ok locally and through the tunnel.
  Watcher (`watch_official.py`, WATCH_NAME=official_v1_4) armed from 12:39: its first line will be the site's check.
  README: the best evaluated submission per team is ranked. The site flow (new model, Check endpoint, Queue
  evaluation) is the user's. The keeper still names official_v1_3: its stand-in's guard checks free GPU memory (all
  of it) while that server is up, which also covers v1.4's.
- **student_v1_5 (user: "lets do back reacher"):** `results/v1_5/build_v1_5.sh` = v1.4 with back
  reach_back_p2fck9200 (config `configs/reach/teacher_p2f_back9200.json`), tar sha256 6a0406e1...; bundle check vs
  v1.3 (63 identical, 4 new, graphs = v1.3's + the block) and vs v1.4 (only the back export and reach.policy.back
  differ, 65 files identical); loads from /tmp; code = server_v1_4 (tower_rl/, scripts/ identical: no new snapshot).
  Queues from 12:45: GPU `gpu_checks_v1_5.sh` (pid 1806307, holds train.pid: Task 5 camera seeds 7400 / 7500, Tasks
  1-3 camera seed 7200, e2e on port 9006), CPU `cpu_queue_v1_5.sh` (pid 1806327: parity vs v1.3 + the config and the
  negative control vs v1.4's parity records, then the teacher A/B 8500 vs 9200 on Tasks 1-3 + 5, D0 + D1 x 8, seeds
  10000 / 10200 / 10400, one process at a time). rehearse.py's batch seeds are seed + 100 b, so the earlier 9600 A/B
  used 9600 / 9700.
- **13:18: GPU queue stopped** (user: the screen froze and responded with great delay while it ran; the camera loop
  held the GPU at 94 % / 168 W). Killed by PID (queue 1806307, cam script, guard, rehearse 1814325 in 5-3 seed 7400);
  train.pid removed, the keeper's stand-in back at 13:18:57 (30 % duty). Done before: Task 5-2 camera seed 7400 with
  v1.5: 7 / 10 (TSR 0.7, PSR 0.7, LHPR 0.85; p2f_t5 = 8500: 8 / 10). The GPU checks wait for a time the user is away.
- **Site check of v1.4: 13:18-13:19** (one session per variant, seed-0 resets, blank images), 0 errors on our side.
- **v1.5 CPU parity (`results/v1_5/parity`):** v1.5 = v1.3 + teacher_p2f_back9200 identical up to timing on 1-3, 2-3,
  3-2, 5-2 (`compare_up_to_timing.txt`). Negative control vs v1.4's records: different on 1-3, 2-3, 3-2, identical on
  5-2 (`compare_v14.txt`). In 2-3 nominal v1.5 lost goal 2's place (back arm, t2_place reported ok at 49.3 s,
  place_ok false on the final arrangement; pre-place arrival 3.3 mm); v1.4 placed all. 9200's earlier teacher losses
  were also on 2-3 (left-arm places): watch 2-3 in the A/B.
- **v1.5 teacher A/B done (14:58; `results/v1_5/ab_back/summary.txt`, `results/v1_5/ab_back_summary.py`):** back
  8500 vs 9200, left 9700 in both, teacher on the student_v1_3 bundle (CPU, oracle state), Tasks 1-3 + 5-2 / 5-3,
  D0 + D1 x 8, seeds 10000 / 10200 / 10400 plus the earlier seed-9600 runs (p4/p2f_both_k16_pp vs
  p2f_both9200_k16_pp; graph_set = the config files in all four). 512 episodes per config, paired by (episode seed,
  case): TSR 493 vs 491, PSR 1448 vs 1445 / 1472, GSR 1472 both; 503 pairs identical in outcome; task success only
  with 8500: 5, only with 9200: 3 (exact McNemar p = 0.73). Mirror points over the 8 variants 63.65 vs 63.42 (D1
  only 60.85 vs 60.55). Per seed 8500 / 9200: 122 / 120, 122 / 124, 126 / 124, 123 / 123.
  - 2-3 is the only variant where they differ much: 62 vs 60 / 64. Its main failure, the left arm's t1_place at
    ~35 s (place_xy 5-7 mm), happens with both (8500 twice, 9200 three times), so it is not new with 9200. 9200's
    other 2-3 loss (10200001 heavy) has the parity run's signature: every node ok, t2_place (back) reported ok at
    place_xy 4.11 mm, but P2 never counted (no failure signature; final arrangement not recorded). 8500 never showed
    it.
  - Task 5: 5-2 identical (57 / 57, the same 7 episodes); 5-3 55 vs 54 (one 9200 loss, i0_push_in on the back arm).
  - In the tasks the back arm's arrival errors are the same (held transit p50 3.4 vs 3.3 mm, empty 3.4 vs 3.3, back
    places 1.8 vs 1.7 mm): the IK finish and the graph tolerance set them, so 9200's battery accuracy (p50 0.6 vs
    0.9 mm) does not show. 9200 is ~1-2 s faster on 1-2 / 1-3 (paired, both ok); time does not score.
  - Verdict: no difference worth serving v1.5 for; recommend keeping v1.4 and not spending the ~6 h of GPU checks
    (desktop lag) on v1.5 (user's call). Watcher re-armed 15:04 for v1.4's first evaluation session
    (`results/v1_4/logs/watch_first2.log`); no traffic since the 13:19 site check.
- **Submissions closed (user, after the A/B):** the organizers no longer take submissions after the Oct 31 deadline,
  so v1.4 is never evaluated; student_v1_3 (evaluated 10-31) is the final submission. Watcher stopped (pid 1842521).
  Still up, for the user to keep or stop: the v1.4 server (pid 1802014) + tunnel (cloudflared 1802082), the v1.3
  server (19527) + tunnel (19639), the keeper (46438, stand-in ~30 % GPU / ~160 W), telemetry (19328).
- **All stopped 20:00-20:01 (user: "stop all of them"):** keeper via `touch results/reach/STOP_BG` (keeper 46438,
  guard 1817172, stand-in 1817216 exited in 16 s; STOP_BG left in place, remove it before restarting the keeper);
  v1.4 and v1.3 servers + tunnels via `SESSION=tower_v1_4|tower_v1_3 results/student/stop_official.sh` (Ctrl-C, then
  the tmux sessions closed; the tmux server exited with its last session); telemetry 19328 by kill -9. After: no
  serve / cloudflared / keeper / telemetry processes, ports 9000-9009 free, GPU 0 %, 9 W, 17 MiB.
- **Reproduction package (user: stage everything for v1.3 / v1.4 / v1.5, plus a README for v1.3 on training the
  teacher and distilling the student; weights stay out of git; write-ups copied into docs/):**
  - `reproduce/`: README.md (the guide), env.sh (REPO, PYTHON, DRY_RUN, `run`, `latest_run`), 1_data.sh,
    2_teacher.sh, 3_perception.sh, 4_student.sh, 5_build_v1_3.sh, 6_checks.sh, 7_serve.sh, build_v1_4.sh (v1.4, or
    v1.5 with REACH_CFG / NAME), compare_bundle.py, fetch_full_episodes.py, manifests/ (the bundles' MANIFESTs),
    noise/ (the estimator noise fits v1 + v2 trace banks, 8 MB: the router rounds and the 9a pushes need them).
  - Provenance from four read-only research agents (teacher skills, perception, student, graphs / Claude). The
    hand-stopped runs get the `--max_iterations` that ends at their checkpoint (rsl_rl 3.1.2 resumes at the
    checkpoint's iter, saves every 100). Pre-key runs switch the key off by `--override`.
  - Claude: offline only (code, the graph edit loop, the recovery rules); nothing at inference, training or
    collection calls it.
  - Verified:
    - 5_build_v1_3.sh / build_v1_4.sh re-freeze v1.3 / v1.4 / v1.5 from the original checkpoints byte-identical
      (73 / 77 / 77 files; test bundles deleted).
    - `freeze_bundle.py` with teacher_t45_push9a's six checkpoints: graphs identical, weights identical, TorchScript
      bytes differ (serialization ids). The original push9a was cp + export, so its MANIFEST is stale.
    - The bundle configs differ from configs/ in insert.yaml (release_min_along_m) and tiers.yaml (D1e). The runtime
      reads configs/, and both keys predate every student-era run.
    - Code parity, the repository vs server_v1_3 on the v1.3 bundle: 10 variants, CPU oracle, mirror, D1 nominal +
      heavy x 2, all identical up to timing (`results/release_check/parity_code_v1_3/compare_up_to_timing.txt`).
    - pytest passes. All scripts dry-run; every --flag exists (scratchpad check_flags.py).
  - Known: `eval_switcher_offline.py` cannot read v2 shards (not used). The Task 4 graphs do not recompile
    identically (gripper_closed_width changed since): ship the JSON.

## 2026-11-05 (review handover)

The organizers asked the provisional top five for a technical report, the exact evaluated code, checkpoints and agent records,
for verification before final rankings. This entry covers the repository; the report, checkpoints, logs and agent records went to
the reviewers privately. `REVIEW.md` is the entry point.

- **Team overall is per variant.** The leaderboard takes each team's highest-LHPR result per variant across eligible submissions,
  "including incomplete submissions" (its README and `scoring.js`), so `student_v1`, `student_v1_2` and `student_v1_3` are all covered.
- **No commit existed for any evaluated code:** the servers ran from `tower_rl_snapshots/server_v1*`, and the bundle MANIFESTs say
  `f5e8cce` with `git_dirty`. Now tags, built from the snapshots with a temporary index: `eval/student_v1` `862244d`,
  `eval/student_v1_2` `5db8b5d`, `eval/student_v1_3` `79203a6`. `git archive` of each against its snapshot differs only by the three added
  files (`assets_m/tower_station_m.json`, `.gitignore`, `README.md`) and the snapshot's symlinks.
- **A clean clone could not serve:** every episode reads `assets_m/tower_station_m.json`, which was git-ignored (13 perception tests failed on a clean
  clone for the same reason). The tags and `main` now include it. Fresh clone without `data/`: 198 passed, 8 skipped.
- **Serving needs neither Isaac nor the kit.** Import closure: torch, torchvision, numpy, scipy, cv2, PyYAML, msgpack, websockets
  (`reproduce/environment/requirements-inference.txt`).
- **Checked from clean clones, all three tags, in the evaluated environment and in a clean `uv` venv with only those packages:** served with the
  published tar and the recorded flags; `probe_server.py` 10 / 10, with blank images and with organizer episodes streamed through it; artifact identity on
  the wire equal to the official one; `run.json` equal to the official one apart from pid, times, argv and paths; `strace` shows no read from the
  development repository. About 2 GB of GPU memory.
- **Replay against the official logs:** the saved first-call frames of all 100 `student_v1_3` episodes through the tag and bundle: 1,100 block poses,
  median 0.04 mm and p99 0.87 mm from the logged poses (the frames were saved as JPEG), orientation the same in 100 / 100, repeats exact.
- **CPU regression:** `reproduce/6_checks.sh cpu` on the v1.3 bundle gives mirror points 80.09 and 77.56, equal to the original gates.
- **Serving-path audit against the snapshot:** no recorded action sequence is replayed or used as a label, nothing is keyed on seed or trial, no LLM
  client in any tag. Official v1.3 from the server's logs: 100 episodes, 39,328 calls, 210,630 rows, 0 errors, 0 clipped values, 0 guard vetoes, 0 blind
  episodes, 4 rule-based re-pushes (all 4-2), 11 of 198 places after grips of 4 mm or more.
- **Site retries:** the author used the site's "Retry failed episodes" during `student_v1`'s evaluation. Our logs show its later passes on 10-27, not which click started each.
- New here: `REVIEW.md`, `reproduce/evaluated/` (serving script, verbatim official scripts and `run.json` files), `reproduce/environment/`, `reproduce/manifests/student_v1_2.json`,
  and the two specification documents the README cites. No change to `tower_rl/`, `configs/` or existing `scripts/`, so the parity run stands.
