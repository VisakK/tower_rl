# TOWER / G2A Jenga Challenge — build the agentic teacher in Isaac Lab

Scope of this file: **the privileged agentic system only** (Tier 2's core): Isaac Lab scenes, a local evaluator,
two PPO contact skills, a planned transit controller, and the typed graph that chains them with state-derived
checkpoints, rehearsed in simulation and edited offline by Claude. Everything that turns this into a submission —
the RGB pose estimator, the `tower-infer-v1` policy server, the Tier 1 imitation baseline, the DAgger student and
the submission package — is **deferred** to a follow-up file; §10 lists what the teacher must preserve so those
pieces bolt on later.

Team state when written (2026-10-18): Isaac Lab installed; the TOWER station and block assets download and load
in a scene. Nothing else exists. Deadline for the challenge is October 31, 2026 (AoE), so the teacher should
exist by about October 25.

---

## 0. How to work from this file

1. Work the steps in order. Each has **Deliverables**, an **Acceptance test** and **Notes**. Do not move on until
   the test passes or the human waives it in writing.
2. Keep `BUILDLOG.md` at the repo root: one dated entry per session with what was built, the acceptance numbers,
   and every deviation from this file with its reason.
3. **VERIFY** marks an API name or a value that depends on the installed Isaac Sim / Isaac Lab version or on a
   file we have not opened. Check it against the installed source (`python -c "import isaaclab; print(isaaclab.__version__)"`,
   then `grep -rn` in the Isaac Lab tree) or the file itself before using it. Never guess a class name.
4. **HUMAN** marks something only the human can do (cloud GPUs, a Discord question). Say exactly what is needed and
   continue with independent work.
5. Every tunable that affects scoring (tolerances, jitter ranges, reward weights, chunk lengths) lives in
   `configs/*.yaml`; every experiment writes a JSON result under `results/` keyed by config hash and date.
6. Prefer the simplest thing that passes the test. The first milestone is a **privileged `grasp_lift` PPO policy
   that succeeds ≥ 95% on the nominal layout** in a light scene; everything else builds on that.

---

## 1. Ground truth that shapes the design (verified 2026-10-18)

Sources: docs site `https://challenge.tower-benchmark.com/docs/` (Submission Rules v1.3, 2026-10-15), repo
`https://github.com/yqi19/corl_challenge`, dataset `https://huggingface.co/datasets/tower-benchmark/TOWER-SimData`,
assets `https://huggingface.co/datasets/tower-benchmark/TOWER-Assets`. Where repo and docs disagree, the docs win.

### 1.1 Tasks

| Variant folder | Task | Movable layers | What a layer is |
| --- | --- | --- | --- |
| `task1-2_tower_transfer_2` / `task1-3_tower_transfer_3` | Tower Transfer | 2 / 3 | two parallel side blocks (hollow centre), rotated 90° per layer |
| `task2-2_cross_tower_transfer_2` / `task2-3_cross_tower_transfer_3` | Cross Tower Transfer | 2 / 3 | one centre block per layer, alternating 90° yaw |
| `task3-2_vertical_stacking_2` / `task3-3_vertical_stacking_3` | Vertical Stacking | 2 / 3 | blocks restacked into an aligned column on the destination pad |

Every tower stands on **three fixed foundation blocks** (side by side; footprint 0.15 × 0.15 m; centres at
z = 0.015 m, top at 0.03 m); movable layers sit at a **0.03 m pitch** (layer k centre at z = 0.015 + 0.03·k).
The two pads are about **0.36 m apart**. Operations move the **top layer first** (source layer N → destination
layer 1, N−1 → 2, …). The demonstrations **alternate arms per layer** (`arm: 1` = back, `arm: 0` = left).

The only fully published layout is Task 2-3 (`tasks/cross_tower_transfer_3-cfg08.json`): source tower centre
≈ (0.42854, −0.27528, 0) with foundation yaw −90° and bases at x = 0.37854 / 0.42854 / 0.47854; layer centres
`source_l1_center` z 0.045 with quat (0, 0, −0.7071, 0.7071), `source_l2_center` z 0.075 identity,
`source_l3_center` z 0.105; destination centre ≈ (0.17174, −0.52788, 0), yaw 0, bases at
y = −0.57788 / −0.52788 / −0.47788; transfers l3 → dest layer 1 (arm 1, goal z 0.045), l2 → layer 2 (arm 0,
z 0.075), l1 → layer 3 (arm 1, z 0.105); training-scene jitter `center_block_jitter_m` 0.002,
`tower_jitter_m` 0.003. **Task 1 and Task 3 layouts are generated at runtime by the organizers and are not
published** (`tower:layoutStatus = "generated-at-runtime"`); they are reconstructed from a few demonstrations (Step 1).

### 1.2 Blocks, robot, gripper, cameras, physics hints

- Block: **0.15 × 0.05 × 0.03 m, 0.1056 kg** (twice real Jenga size, ~10× the mass). Asset rigid-body solver iterations 16 / 4.
- Robot: two **AgileX Nero 7-DoF arms** `left` (`StandA_Robot`) and `back` (`StandB_Robot`) on stands at adjacent table corners. Public URDF `https://github.com/agilexrobotics/agx_arm_urdf`. Joint limits (rad): j1 ±2.70526; j2 ±1.74; j3 ±2.75; j4 [−1.01, 2.14]; j5 ±2.75; j6 [−0.7330, 0.8727]; j7 ±1.5708; effort 100; velocity 5.
- Gripper: parallel jaw; prismatic `gripper` joint 0–0.1 m with two mimic finger joints 0–0.05 m (multipliers ±0.5); finger effort limit **3.0 N**; pad friction **2.4 / 2.0** (static / dynamic, raised on 2026-09-13).
- Example initial qpos: left `[-0.0375, -0.8007, -0.0041, 1.7538, 0.0117, -0.0255, 1.2400, 0.1]`, back `[-0.1573, -0.8276, 0.0513, 1.7956, 0.0005, -0.0428, 1.2803, 0.1]` (last value = gripper width, m).
- Cameras (needed later, recorded now): four RGB at 640×480; static `high` (fx 606.97, fy 606.58, cx 332.10, cy 257.00) and `low`, wrist `left_wrist`, `back_wrist` (fx 607.53, fy 607.25). Station update placed `high_camera` at translate (875.459, −34.802, 558.189) mm, orient (0.8332, −0.3619, 0.1605, −0.3859) and `low_camera` at (−151.475, −878.254, 956.252) mm, orient (0.3940, −0.0942, −0.3664, 0.8376) under `/PNP_DUALD1200_STATION/CameraAlignment/` (USD orient = w, x, y, z). Render settings: `RaytracedLighting`, `dlss_mode 2`, `anti_aliasing 4`, `render_subframes 4`, `lighting_samples 16`.
- Station is authored in **millimetres** (`metersPerUnit = 0.001`); block USDs reference the station prims at scale (1000, 1000, 1000). Check units in the stage before trusting any pose.
- Organizers' physics step, PhysX settings and block–block friction are unpublished; they are calibrated by replaying demonstrations (Step 2.3).

### 1.3 Evaluation facts the teacher must respect (even though serving is deferred)

- At test time a policy sees **only** four RGB images, `joint_positions` [2,7], `gripper_widths_m` [2], the instruction, `task_variant`, `step_index`, `sim_time_s` — no block poses, no depth, no force. → Every observation the skills' **actors** use must be something a camera pipeline can estimate (block poses with millimetre noise, TCP pose from FK, gripper width); privileged quantities (contacts, friction, mass, exact poses) go to the **critic** only.
- Actions are absolute joint targets + gripper widths in metres, `float32[H,16]`, 1 ≤ H ≤ 16, each row applied for 0.1 s at 10 Hz (three 30 Hz physics ticks) or 1/30 s at 30 Hz. → Skills run at **10 Hz** in Δ-pose space and are converted to joint targets by IK; the executor emits chunks (H = 1–2 in contact, up to 16 in transit).
- 210 s simulated time per episode; the simulator pauses during inference, so our executor's latency is not a score factor.
- Metrics (docs): **TSR** successful episodes / evaluated (all operations succeed and the final arrangement is stable); **GSR** successful grasps / planned; **PSR** blocks placed stably / planned; **LHPR** successful G+P / planned, successes after an earlier failure still count; **ODR** disturbed non-target block-occurrences / counted steps with each episode contributing `G + P + 1 − task_success` steps; a disturbance is a non-target block whose centre moves **> 10 mm during an operation** (current target, foundations and tools excluded; one count per block per operation). Draft ranking: mean TSR, then progress, then lower disturbance.
- Evaluation randomizes mass, friction, assembly pose, layer pose and lighting (nine randomized episodes per variant plus one nominal). → Dynamics and pose randomization are part of training from day one.
- Rules: learned policies or model-driven agents are allowed, with IK, controllers and safety filters; fixed scripted solutions, simulator-state access at inference and "nominal model wrappers" are banned; LLM agents are allowed with reproducibility materials. → The graph is **authored and edited by Claude offline** and **compiled** for inference; the learned skills and (later) learned perception are the load-bearing parts.

### 1.4 Dataset facts (we download only a few episodes)

One HDF5 per episode, 30 FPS, keys we need: `leader/qpos` [T,16] (commanded joints + gripper widths in m),
`observations/qpos` [T,16], `observations/eef` and `leader/eef` [T,14] (per arm xyz + quaternion xyzw),
`observations/gripper_status` [T,8], `sim/time`, `sim/success`, `sim/terminal`, optional `keyframes` and
`sim/finger_pad_centers_m`. Images are JPEG streams under `observations/images/*` (not needed now). **There are no
block poses in the files.** Episodes are 61–117 s (1,800–3,500 frames) and were generated by a scripted planner
(`jenga-sim-data-generator`, `motion_profile: smooth`), so they are clean and consistent. Lifter channels are
never executed in the benchmark: ignore them. Each file is 0.4–1.2 GB because of the four JPEG streams.

---

## 2. What we are building: the agentic teacher

```
1. Graph authoring (Claude, offline)   block order, arm per layer, typed skill calls, checkpoints on sim state, retry/skip edges
2. Contact skills (PPO, privileged)    grasp_lift and place_release in a light scene; Δ-pose actions through IK;
                                       noisy block pose to the actor, true state + contacts to the critic
3. Rehearsal in Isaac Lab              randomized instances (pose, mass, friction, lighting); per-node pass counts;
                                       settle-and-threshold stability check; the 10 mm disturbance rule
4. Graph and parameter edits (Claude)  swap nodes, tune tolerances, add retry edges; bounded loop (≤ 10 iterations)
5. Frozen teacher bundle               compiled graph JSON, TorchScript skills, tolerances — the object that later
                                       (a) runs on estimated poses behind a policy server, (b) labels DAgger data
```

Design decisions fixed by the evidence (details in the review): reaching is not learned (IK plus planned
transit); RL is spent on the two contact segments; the actor never sees what a camera cannot estimate;
checkpoints run on state, never on frames; the LLM edits code and parameters offline from rehearsal statistics.

Compute: the workstation GPU (24 GB) is enough for every step in this file (state-based PPO with thousands of
envs is cheap; cameras are used only for calibration and eval at 1–16 envs).

---

## 3. Repository layout

```
tower/
  configs/        scene.yaml, physics.yaml, layouts/<variant>.yaml, skills/{grasp_lift,place_release}.yaml, tiers.yaml, eval.yaml
  data/           remote/partial HDF5 reader, episode subset downloader, gripper-event detection, layout reconstruction
  sim/            station_scene.py (full station, cameras; 1–16 envs), skill_scene.py (light cuboid scene; thousands of envs), assets.py
  eval/           operations.py, metrics.py, tiers.py, runner.py (in-process policies only for now)
  skills/         transit.py (IK + via-points), grasp_lift_env.py, place_release_env.py, ppo cfgs, export.py (TorchScript)
  graph/          interface.py, checkpoints.py, graph.py (JSON per variant), executor.py, rehearsal.py, feedback.py, edits/
  tests/          pytest units (metrics, IK, events, layouts) and headless scene smoke tests
scripts/          one entry point per step: fetch_subset.py, replay_demo.py, train_skill.py, evaluate_skill.py, rehearse.py, freeze_bundle.py
results/          JSON results by config hash and date
BUILDLOG.md
```

All code runs inside the Isaac Lab Python environment.

---

## Step 1 — A small data subset, gripper events, and reconstructed layouts (day 1)

Goal: block layouts for all six variants and a handful of command trajectories for physics calibration, **without
the 225 GiB download**.

**Deliverables**
1. `scripts/fetch_subset.py`: download `MANIFEST.jsonl` and `schema/tower_simdata_hdf5.json` first; read the
   schema (it names the 37 fields and the `gripper_status` semantics). Then, per variant, fetch **3 episodes**
   chosen from the manifest (different `source_date`s) with `huggingface_hub.hf_hub_download(repo_id="tower-benchmark/TOWER-SimData", repo_type="dataset", filename="data/<variant>/<episode>.hdf5")`
   into `data/subset/`. Expect ≈ 12–15 GB total. Before that, **try partial remote reads**: open the file over
   HTTP with `fsspec.open("https://huggingface.co/datasets/tower-benchmark/TOWER-SimData/resolve/main/data/<variant>/<episode>.hdf5", block_size=4*2**20, cache_type="blockcache")`
   and `h5py.File(fobj, "r")`, reading only `leader/qpos`, `observations/qpos`, `observations/eef`, `leader/eef`,
   `observations/gripper_status`, `sim/*` — HDF5 chunked datasets are fetched by byte range, so the JPEG streams
   stay on the server. If that works (**VERIFY** on one file; it depends on how the datasets were chunked), extend
   it to 10 episodes per variant for better layout statistics; if not, fall back to the 3-file download.
2. `tower/data/events.py`: gripper **close/open events** per arm from `leader/qpos[:, 7]` and `[:, 15]`
   (hysteresis: width < 0.06 → closed, > 0.09 → open; tune; confirm against `gripper_status`). Pair close → open
   into operations `(arm, t_close, t_open)`. Expected counts: Task 1 = 2 × layers (4 or 6), Task 2 = layers
   (2 or 3), Task 3 = layers (verify from the data).
3. `tower/data/layouts.py`: for each operation take the **commanded** EE pose (`leader/eef`) at `t_close`
   (grasp) and `t_open` (release). With a TCP-to-block-centre offset (the jaws close centred on the 0.05 m face;
   the block centre sits along the approach axis at finger-pad depth — calibrate the offset in Step 2.3), recover
   source and destination block poses per layer. Cluster across the episodes of a variant → nominal layout +
   observed jitter. Cross-check Task 2-3 against the published descriptor (must agree within ~5 mm / 3°) and all
   variants against the repo's preview images. Write `configs/layouts/<variant>.yaml`:
   `foundation_pose`, `layers[k].blocks[i].pose`, `transfers[] = {block, arm, dest_layer, goal_pose}`, `jitter`.
4. `tower/data/semantics.py`: when does `sim/success` turn true relative to the last release (immediately or after
   a settle), does every stored episode end in success, does `terminal` precede `success`? Record in `BUILDLOG.md`.

**Acceptance test:** `pytest tests/test_data.py` — every subset file parses; operation counts match expectations
for every episode; the reconstructed Task 2-3 layout matches the descriptor within 5 mm / 3°; a per-variant top-down
plot of reconstructed footprints over the foundation pads looks right by eye.

**Notes:** keep the 16-D layout explicit (left j1–j7, left width, back j1–j7, back width). The remote-read path
is also how, later, the student can stream the full dataset without a bulk download.

---

## Step 2 — Scenes: light skill scene first, full station for calibration (day 1–3)

### 2.1 Inspect the station (half a day)

Load the station as the team already does and dump the prim tree under the station root to `docs/stage_tree.txt`:
robot articulation roots, joint names and order, gripper joint names (virtual `gripper` vs. the two finger joints),
foundation block prims (`Foundation01–03`, `RotatedBase01–03`), camera prims, physics materials, `metersPerUnit`.
Measure in the stage: the two arm **base poses** in the table frame (needed for the light scene), a foundation
block's extent (confirms units), the pad positions.

### 2.2 Light skill scene (`tower/sim/skill_scene.py`) — the RL workhorse

**Deliverables**
1. `SkillSceneCfg(InteractiveSceneCfg)` (**VERIFY** names): a ground plane; **one Nero arm** as an `ArticulationCfg`
   spawned at the measured base pose of the chosen arm (start with `left`; the back arm is a config switch) — from
   the station's robot prim if it is a usable articulation, else from the URDF via the Isaac Lab URDF converter;
   the three foundation blocks of one pad (second pad on a flag) as fixed box colliders; movable blocks as
   `RigidObjectCfg` boxes 0.15 × 0.05 × 0.03 m, mass 0.1056 kg; no cameras; no station mesh; env spacing 1.5 m;
   `replicate_physics=True`.
2. Actuators: implicit PD on the 7 joints (initial stiffness 400 / damping 40 per joint, calibrated in 2.3),
   effort 100, velocity 5; finger joints as position-controlled prismatic joints with effort limit 3 N.
3. Gripper convention: a **width in metres (0–0.1)** maps to each finger joint as `width / 2` (0–0.05) if the
   articulation exposes the fingers rather than the virtual joint (**VERIFY** in the stage). Our skills use a
   binary target: open 0.10, closed 0.040 (block face is 0.05; tune so the 3 N limit clamps without stalling).
4. Physics (`configs/physics.yaml`): `dt = 1/120`, control at 30 Hz (3 physics steps), policies at 10 Hz (3 control
   ticks per policy step, matching the evaluator); TGS; position iterations 64 for transit, **192 for contact
   skills** (Factory); `friction_correlation_distance = 0.00625`; `gpu_max_num_partitions = 1`; bounce threshold
   0.01; block material start at 0.5 / 0.4 (calibrated in 2.3); pad material 2.4 / 2.0.
5. Collision shapes: boxes for blocks (Isaac Lab warns that thin convex hulls can fall back to CPU collision).

### 2.3 Full-station scene and demo-replay calibration (`tower/sim/station_scene.py`)

**Deliverables**
1. `TowerSceneCfg`: the station spawned fixed via `sim_utils.UsdFileCfg` at the scale that lands it in metres,
   both arms, movable blocks at the reconstructed layout, four cameras at 640×480 with the §1.2 intrinsics
   (focal_mm = fx × horizontal_aperture_mm / 640; with the default 20.955 mm aperture, fx 606.97 → ≈ 19.87 mm) and
   the station-update poses (mm → m), wrist cameras at the offsets found in 2.1; the station's light. 1–16 envs
   only. Cameras can be disabled by flag.
2. `scripts/replay_demo.py --episode <subset file>`: set frame-0 `observations/qpos`, spawn the reconstructed
   layout, and send `leader/qpos[t]` as joint targets at 30 Hz; log measured qpos, EE poses, block poses, contacts.
3. Calibrate until replay reproduces the demonstrations: PD gains (joint RMS error vs `observations/qpos`), block
   friction and the TCP-to-block offset (every operation lifts the block ≥ 2 cm after close and leaves it within
   tolerance after open). PhysX is not reset-deterministic; judge on ≥ 10 replays per episode.
4. Render the four cameras at frame 0 and overlay the demo JPEGs (decode just frame 0 from one downloaded file):
   foundation blocks and arm silhouettes should coincide within a few pixels. Record which static camera is
   `left_back_camera` and which is `right_front_camera`.

**Acceptance test:** (a) `pytest tests/test_scenes.py` opens both scenes headless and steps 300 ticks without NaN;
(b) replay of the Task 2-2 subset episodes completes every operation in ≥ 8 of 10 replays per episode and the
final layout matches the reconstructed destination within tolerance; (c) the camera overlay figure is saved to
`results/calibration/`.

**Notes:** the light scene must reproduce the **arm base pose relative to the pads**, otherwise reachability and
wrist orientations in training will not match the station. Keep the full-station scene for calibration,
rehearsal and, later, rendering; never for thousands of envs.

---

## Step 3 — Local evaluator (day 2–3; in-process policies only)

**Deliverables**
1. `tower/eval/operations.py`: per variant, the ordered operation list from `configs/layouts/<variant>.yaml`:
   one **G** and one **P** per transfer with `(block_id, arm, goal_pose)`. G succeeds when the target is lifted
   ≥ 2 cm with the gripper closed and no other block was disturbed; P succeeds when, after release and a
   **1.0 s settle**, the block is within `tol_pos` (start 10 mm) and `tol_yaw` (start 5°) of its goal, the blocks
   beneath it did not move, and the destination stack is static (max speed < 1 cm/s).
2. `tower/eval/metrics.py`: TSR, GSR, PSR, LHPR, ODR exactly as in §1.3, with the per-episode ODR denominator
   `G + P + 1 − task_success`; a disturbance is a non-target block whose centre moves > 10 mm inside an
   operation window (from the operation's first motion to the end of its settle check), excluding the current
   target and the foundations, one count per block per operation. Our "operation window" is a reconstruction of
   the organizers' definition — document it.
3. `tower/eval/tiers.py`: **D0** nominal layout with descriptor jitter (0.003 m tower, 0.002 m centre block);
   **D1** tower/pad pose jitter ±3 cm and ±15° yaw, block mass ×U(0.7, 1.3), block friction ×U(0.6, 1.4), light
   intensity ×U(0.5, 2.0) and colour-temperature jitter (lighting only matters in the station scene);
   **D2** everything combined. Explicit, logged seeds.
4. `tower/eval/runner.py`: runs an in-process policy object with the evaluator's **chunk semantics**:
   `policy.step(state) -> actions [H,16]`; each row applied for 0.1 s at 10 Hz; stop on success, at 210 s, or on
   an exception; per-episode JSON (per-op outcomes, disturbances, timings) and aggregate metrics with 95% Wilson
   intervals. The privileged executor (Step 7) plugs in here; later, a WebSocket client is added for the real server.
5. `scripts/sweep_tolerance.py`: in the station scene, place blocks at controlled offsets (0–20 mm, 0–10°) on
   3-layer stacks and measure what still stands after 5 s; set `tol_pos`/`tol_yaw` to the largest offsets that keep
   > 95% of stacks standing. Record the values.

**Acceptance test:** `pytest tests/test_metrics.py` reproduces worked examples (e.g. G=3, P=3, two grasps and one
place succeeded, one disturbance in op 2, no task success → GSR 0.667, PSR 0.333, LHPR 0.5, ODR 1/7 ≈ 0.143,
TSR 0). A "hold still" policy scores 0 everywhere. Demo replay on D0 scores TSR ≥ 0.6 on Task 2-2.

---

## Step 4 — Transit controller (no learning; day 3)

**Deliverables**
1. `tower/skills/transit.py`: `DifferentialIKController` (damped least squares, **VERIFY**) driving one arm's
   TCP along straight-line Cartesian segments through via-points: **pre-grasp** (10 cm above the block, jaw axis
   aligned with the block's 0.05 m face), **post-lift** (block ≥ 8 cm above its layer), **travel height** ≥ 15 cm
   above both towers, **pre-place** (8 cm above the goal). Limits: 0.3 m/s, 1.0 rad/s, ≤ 2 cm per 10 Hz step.
2. Collision checks: capsule approximations of the arm links vs. tower OBBs and vs. the parked other arm; on a
   predicted collision raise the travel height or insert a lateral via-point. If cuRobo is already installed, use
   it instead; do not spend time installing it.
3. Termination: the transit **ends inside the next skill's initiation region** (≤ 3 cm / 5° from the pre-grasp
   or pre-place pose) and returns `SkillResult` (Step 7 interface).

**Acceptance test:** from 200 random D1 layouts, transit reaches pre-grasp and pre-place poses within 3 cm / 5°
in ≥ 98% of cases with no arm–block contact (contact forces logged from the simulator).

---

## Step 5 — `grasp_lift` PPO skill (day 3–5) — the first milestone

**Env** (`tower/skills/grasp_lift_env.py`, `DirectRLEnv` or manager-based, **VERIFY**), light scene, 4,096 envs,
one arm (left first; the back arm is a config switch, trained as a second policy).

- **Reset distribution:** the variant's source tower at a sampled stage (which layer is on top varies, including
  layers partly removed); layout jitter per tier; **80% of resets** place the TCP within 3 cm / 5° of the nominal
  pre-grasp pose (the transit's terminal distribution), **20%** within 8 cm / 15° (ManipGen-style wider region);
  dynamics randomization: block mass ×U(0.7, 1.3), block friction ×U(0.6, 1.4), small gripper-pad friction jitter.
- **Actor observation** (everything estimable from cameras + proprioception): TCP pose and linear/angular
  velocity (from FK), gripper width, **noisy** target block pose (per-episode offset σ = 2 mm / 2°, per-step
  jitter 1 mm), target pose relative to TCP, noisy poses of the two nearest non-target blocks, previous action,
  phase one-hot (approach / grasp / lift). **Critic observation:** actor obs plus true poses of all blocks,
  finger-pad contact forces, friction and mass scalars (asymmetric critic).
- **Action:** 6-D Δ-pose of the TCP in the world frame (position ≤ 2 cm, rotation ≤ 0.097 rad per step; EMA 0.2 on
  actions; tighter ≤ 5 mm steps once within 2 cm of the block) → IK → joint targets; plus **one binary gripper
  logit** (open / closed). Policy at 10 Hz (3 control ticks of 30 Hz).
- **Reward (dense, staged):**
  - alignment: keypoint distance between the gripper frame and the centred top-down grasp pose, 4 keypoints, two
    squashing scales (Factory-style `a/(b + d)` with `[50, 2]` then `[100, 0]`);
  - contact bonus `+1` when both finger pads touch the target; grasp bonus `+5` when closed with both pads in contact;
  - lift bonus `+15` when the block is ≥ 4 cm above its start with the gripper closed (as Isaac Lab Lift); success
    `+20` when lifted ≥ 8 cm and held 0.5 s;
  - **disturbance penalty** `−k · max(0, d_nt − 3 mm)` per step summed over non-target blocks, and terminal `−10`
    if any non-target block centre moves > 10 mm (the challenge's rule);
  - action-rate `−1e-4‖Δa‖²`; joint-limit margin penalty; drop the return for steps where block interpenetration
    exceeds 1 mm (IndustReal's fix for reward hacking).
- **Termination:** success; disturbance > 10 mm; block dropped after grasp; 150 policy steps.
- **PPO (rsl_rl, VERIFY config class names):** 24 steps/env, 5 epochs, 4 minibatches, lr 1e-4 adaptive on
  KL 0.016, entropy 0.006, γ 0.99, λ 0.95, actor MLP 256-256-128, critic 512-256-128 (ELU). Expect 1–3 h.
- **Curriculum:** alignment tolerance from 3 cm → 5 mm as success passes 80%; wider reset region as it passes 90%.

**Deliverables:** `scripts/train_skill.py --skill grasp_lift --arm left`, `scripts/evaluate_skill.py` (500
episodes per tier with per-cause failure counts), `tower/skills/export.py` (TorchScript + observation normalizer).

**Acceptance test:** success ≥ 95% on D0 and ≥ 85% on D1, disturbance rate < 0.1 per episode, measured with the
noisy actor observations. Then train the `back` arm policy with the same config.

**Notes:** keep the gripper action binary (the single most important transfer factor in RGB-Stacking). Do not
let the actor see contacts or friction; if the skill only works with them, the student later cannot follow.

---

## Step 6 — `place_release` PPO skill (day 4–6)

Same machinery as Step 5 with the second pad enabled.

- **Reset:** the block spawned already grasped (fingers closed on it, slip offset σ 3 mm), TCP within 3 cm / 5° of
  the pre-place pose (20% wider); destination stack at a sampled stage with jitter; dynamics randomization as above.
- **Actor observation:** TCP pose/velocity, gripper width, noisy goal pose (σ 2 mm / 2°), noisy pose of the held
  block relative to the TCP, noisy poses of the destination layer blocks, previous action, phase one-hot.
- **Action:** Δ-pose as above; ≤ 5 mm per step once the block is within 2 cm of the goal height; binary release.
- **Reward:** two-scale keypoint tracking of the **block** to its goal pose; contact-gated descend bonus;
  **release bonus only if the settle check passes**: after a release action the arm retreats 5 cm upward and the
  simulator runs 1.0 s (30 ticks); then (i) block within `tol_pos` / `tol_yaw`, (ii) every destination block
  displaced < 10 mm, (iii) all destination blocks static → `+20`; releasing early or failing the check → `−5` and
  termination; disturbance penalty as in Step 5; a support-polygon term `−1` per step whenever the destination
  stack's centre of mass projects outside the support of the layer below (privileged, computed from true poses).
- **Reverse curriculum:** start 1 cm above the goal and move the start upward as success grows.

**Acceptance test:** as Step 5 (≥ 95% D0, ≥ 85% D1, disturbance < 0.1), for both arms.

**Fallback:** if `place_release` has not passed by day 6, use a **planned slow descend + settle check** as the place
controller and keep `grasp_lift` learned; note it in `BUILDLOG.md` because it affects how the final agent is
described under the rules.

---

## Step 7 — Graph, checkpoints, executor, rehearsal, Claude loop (day 5–8)

**Deliverables**
1. `tower/graph/interface.py`: `SkillCall(skill, arm, block_id, goal: Pose, tolerance: Tol, timeout_s)` →
   `SkillResult(status ∈ {ok, failed, timeout}, pose_error, disturbance_flag, duration_s)`. Skills: `transit`,
   `grasp_lift`, `place_release`, `park`. Skills read a `State` object (block poses, TCP poses, gripper widths,
   contacts) — filled from the simulator now and from estimated poses later, with identical field names.
2. `tower/graph/checkpoints.py`: predicates on `State` after each node: `grasped` (block z rose ≥ 2 cm and both
   pads in contact — on estimated state: block moved with the TCP ≥ 5 cm), `lifted`, `placed_stable` (the Step 3
   definition), `no_disturbance` (10 mm rule), `arm_clear` (both arms outside the tower bounding cylinders).
3. `tower/graph/graph.py`: the graph per variant as **JSON**: nodes = skill calls; edges `ok` → next node;
   `failed` → retry up to 2× (re-transit, re-grasp), then **skip** to the next operation (LHPR counts later
   successes); `timeout` → skip; final `park`. Block order and arm assignment from `configs/layouts/<variant>.yaml`
   (top layer first, arms alternating per layer as in the demos; Task 1: both blocks of a layer with one arm).
4. `tower/graph/executor.py`: deterministic interpreter at 10 Hz producing `[H,16]` chunks: H = 1 during the
   contact skills, H = 2 in their approach segments, H ≤ 16 in transit; any randomness seeded from a request seed.
   Runs in `tower/eval/runner.py` as a policy object.
5. `tower/graph/rehearsal.py`: N ≥ 100 episodes per variant per tier in the **station scene** (cameras off) with
   privileged `State`; writes `results/rehearsal/<variant>/<tier>/feedback.md`: per-node pass counts, failure
   signatures (which predicate failed, pose-error distributions, disturbance culprits, timing), and 5 failure
   videos (cameras on for those only).
6. **Claude loop, offline and bounded:** the human runs Claude Code on `feedback.md` with permission to edit only
   the graph JSON, `configs/skills/*.yaml` tolerances/limits, or reward weights (then retrain); ≤ 10 iterations,
   ≤ 5 edits each, every edit re-rehearsed; stop at plateau. Prompts and diffs are committed under `graph/edits/`
   (the rules require them for reproducibility if an LLM authored anything).
7. `scripts/freeze_bundle.py`: `bundle/` = graph JSON per variant + tolerance config + TorchScript skills +
   normalizers; `sha256` of the tarball recorded as the bundle identity.

**Acceptance test:** with privileged `State` on D1 the executor reaches **TSR ≥ 0.6 on the 2-layer variants and
≥ 0.4 on the 3-layer variants with ODR < 0.1**, and ≥ 0.8 / ≥ 0.6 on D0. Per-node pass rates and the edit history
are in `results/` and `graph/edits/`.

---

## Step 8 — Hardening the teacher before anything bolts on (day 8–9)

1. **Noise robustness:** re-run Step 5/6 acceptance and Step 7 rehearsal with actor observations corrupted by a
   provisional perception noise model (position σ 3 mm, rotation σ 3°, 2% dropout of a block pose for 0.5 s,
   occasional 1 cm outliers); retrain skills with these noises if success drops > 5 points. This is what makes the
   teacher survive estimated poses later.
2. **Both arms:** rehearse Task 1 variants where arms alternate; confirm `arm_clear` prevents arm–arm contact.
3. **Timing:** measure executor wall time per 10 Hz step; it must stay well under one second per step for the
   later server (the simulator pauses, but a wall-clock limit exists).
4. **Export check:** a fresh process loads `bundle/`, runs 10 episodes per variant on D1 and reproduces the
   rehearsal numbers within noise.

---

## 9. Compressed plan (teacher only)

| Dates | Work | Exit |
| --- | --- | --- |
| Oct 18–19 | Step 1 subset + layouts; Step 2.1–2.2 light scene | layouts for all six variants; light scene steps headless |
| Oct 19–21 | Step 2.3 station scene + replay calibration; Step 3 evaluator; Step 4 transit | replay reproduces demos; metrics tests pass |
| Oct 20–23 | Step 5 `grasp_lift` (both arms); Step 6 `place_release` | ≥ 95% D0 / ≥ 85% D1, disturbance < 0.1 |
| Oct 23–26 | Step 7 graph, rehearsal, Claude edit loop | TSR ≥ 0.6 (2-layer) / ≥ 0.4 (3-layer) on D1 with privileged state |
| Oct 26–27 | Step 8 hardening, frozen bundle | bundle reproduces numbers in a fresh process |

After this, the follow-up file covers: the multi-view RGB pose estimator trained on renders from the station
scene (the teacher then runs on estimated `State`), the `tower-infer-v1` server and adapter, the DAgger student,
and the submission package. The deadline leaves roughly four days for those, so the teacher should not slip
past October 27.

---

## 10. Invariants the teacher must keep so the deferred pieces bolt on

- Actors use only camera-estimable observations (block poses, TCP pose, gripper width, phase); privileged inputs
  live in critics and checkpoints during training only.
- `State` has one schema for simulator truth and for estimated poses (plus per-block confidence fields reserved now).
- Skills act at 10 Hz in Δ-pose space and are converted to absolute joint targets + gripper widths in metres by
  IK, clipped to joint limits; the executor already emits `[H,16]` chunks with H ≤ 16.
- The executor is deterministic given a seed and makes no LLM calls; everything Claude authored is committed as
  JSON/YAML diffs under `graph/edits/`.
- Goal poses come from the variant's layout config **and will later come from estimated poses**; nothing reads
  the organizers' `-cfg08` descriptor values at execution time (that would be hard-coding from test layouts).
- The station scene can render the four cameras at 640×480 with the recorded intrinsics and poses (needed for
  perception data generation and for DAgger).

---

## Appendix A — Starting config values

```yaml
physics:
  dt: 0.008333               # 1/120 s
  control_hz: 30             # 10 Hz policies apply one action for 3 control ticks
  solver: TGS
  position_iterations: 64    # 192 for contact skills
  velocity_iterations: 4
  friction_correlation_distance: 0.00625
  gpu_max_num_partitions: 1
  bounce_threshold: 0.01
block:
  size_m: [0.15, 0.05, 0.03]
  mass_kg: 0.1056
  friction: {static: 0.5, dynamic: 0.4}    # calibrate by demo replay
gripper:
  width_open_m: 0.10
  width_closed_m: 0.040                     # tune; block face is 0.05
  finger_effort_n: 3.0
  pad_friction: {static: 2.4, dynamic: 2.0}
arm:
  pd: {stiffness: 400, damping: 40}         # calibrate by demo replay
  effort_limit: 100
  velocity_limit: 5
eval:
  budget_s: 210
  settle_s: 1.0
  disturbance_mm: 10
  tol_pos_mm: 10                            # sweep in Step 3.5
  tol_yaw_deg: 5
tiers:
  D0: {pose_jitter_m: 0.003, center_jitter_m: 0.002}
  D1: {pose_jitter_m: 0.03, yaw_jitter_deg: 15, mass_scale: [0.7, 1.3], friction_scale: [0.6, 1.4], light_scale: [0.5, 2.0]}
skills:
  action_limits: {pos_m: 0.02, rot_rad: 0.097, ema: 0.2, contact_pos_m: 0.005}
  reset: {tight: {frac: 0.8, pos_m: 0.03, rot_deg: 5}, wide: {frac: 0.2, pos_m: 0.08, rot_deg: 15}}
  obs_noise: {pos_m: 0.002, rot_deg: 2.0, per_step_pos_m: 0.001}
  ppo: {num_envs: 4096, steps_per_env: 24, epochs: 5, minibatches: 4, lr: 1.0e-4, kl_target: 0.016, entropy: 0.006, gamma: 0.99, lam: 0.95}
executor:
  rate_hz: 10
  chunk_transit: 16
  chunk_contact: 1
  retries_per_op: 2
```

## Appendix B — Things that silently break the teacher

- Station in millimetres loaded as metres (everything 1000× too large) — check a block's extent first.
- Light-scene arm base not at the station's stand pose — reachability and wrist angles no longer match.
- Actors fed contacts, friction or exact poses — the policy "works" but cannot be followed from cameras.
- Convex-hull colliders on thin blocks — CPU fallback and slow, jittery contacts; use boxes.
- Reward hacking through interpenetration — drop returns on steps with > 1 mm penetration.
- Continuous gripper actions — slower grasps and worse transfer; keep it binary.
- Per-node success predicates that pass on the wrong thing (a block "lifted" by being knocked upward) — every
  predicate must combine motion with the gripper state and the disturbance rule.
