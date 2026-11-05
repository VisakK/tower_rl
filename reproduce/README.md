# Reproducing `student_v1_3`

`student_v1_3` is our submission to the [TOWER CoRL 2026 challenge](https://challenge.tower-benchmark.com/docs/): all
ten task variants, the 10 Hz track. The organizers evaluated it on 2026-10-31 (100 episodes, 0 errors on our server).
It was served as policy `tower-student-v1-3` with artifact identity
`sha256:76ec7f3d35c4da7b41dc74c26ce2f8484287f87cc4023f3820dc4145a0c8263d`, the sha256 of its bundle tar.

To serve the evaluated submissions as they ran, without rebuilding anything, see [`REVIEW.md`](../REVIEW.md).

This guide rebuilds `student_v1_3` from the public downloads with the scripts in this directory. The model weights are not in the
repository; every one of them is trained here. The scripts reproduce the original runs: the same scripts, configs,
seeds, run chains and checkpoint picks. GPU reinforcement learning and network training are not bit-exact, so a
rebuild gives an equivalent policy, not the same bytes (§10).

**What the policy is.** At test time the challenge sends four RGB images, the joint angles and the gripper widths, and
expects chunks of joint targets.
- A pose estimator turns the images into block poses: **kp_v2**, a keypoint network plus a multi-view pose fit.
- A learned router, the switcher **bc_v2_r3**, decides at every step which option of the variant's task graph runs
  next, on which goal and with which arm.
- The options are learned skills, each with a fixed contract:
  - grasp, place and Task 4 push: PPO;
  - Task 5 grasp, insert and push-in: behaviour cloning;
  - transits and parks: planned.

All of this was distilled from a privileged teacher: an agentic system of task graphs and PPO skills that sees the
simulator state. Design write-ups are in [`docs/`](../docs):
- [Teacher-v1.MD](../docs/Teacher-v1.MD): the teacher;
- [Pose-estimator-v1.MD](../docs/Pose-estimator-v1.MD): the pose estimator;
- [Student-v1.MD](../docs/Student-v1.MD): the student;
- [Student-v1.1.MD](../docs/Student-v1.1.MD): the official-run fixes and the recovery node;
- [Student-v1.3.MD](../docs/Student-v1.3.MD): this version.

The raw notes of all eleven build sessions are in [`BUILDLOG.md`](../BUILDLOG.md).

---

## 1. What you need

**External dependencies.** These are not part of this repository; install them yourself.

- **Isaac Sim 5.1.0 + Isaac Lab 2.3.0**, pip install, Python 3.11. This brings PyTorch and rsl_rl. The versions used:
  - isaacsim 5.1.0.0, isaaclab 0.50.5;
  - rsl-rl-lib 3.1.2;
  - torch 2.7.0+cu126, torchvision 0.22.0+cu128;
  - opencv-python-headless 4.11.0.86, h5py 3.15.1, huggingface_hub, pyyaml, pytest.
  - The full list and the machine: [`environment/`](environment).
- **Claude Code (Anthropic)**: only to redo the *agentic development*, i.e. writing code and editing task graphs from
  rehearsal feedback (§4.1). Rerunning this pipeline does not need it: everything it produced is committed, and no
  step calls Claude or the Anthropic API.
- **To serve the policy:** Isaac is not needed, only [these packages](environment/requirements-inference.txt).
  - `data/challenge/track.json`, downloaded from the challenge site after sign-in;
  - `assets_m/tower_station_m.json`, which is in git (stage 1 rebuilds it);
  - optionally `cloudflared`, for a public `wss://` URL.
  - The challenge kit is not read at run time: its `tower_protocol.py` is vendored in `tower_rl/server/`.

**Hardware used.**
- One NVIDIA RTX A5000 (24 GB), 24 CPU threads, 62 GB RAM.
- About 20 GB of disk for `data/` and a few GB for `results/`.
- GPU jobs run one at a time. Times below are from this machine.

**How to run.**
- Run everything from the repository root with the Isaac Lab environment active. Alternatively, set
  `PYTHON=<that environment's python>`.
- Each script prints every command before running it. `DRY_RUN=1 reproduce/<script>` only prints.
- Outputs go to `data/` and `results/`; both are git-ignored.
- Pure-Python tests, no Isaac: `python -m pytest`. A fresh clone without `data/` gives 198 passed, 8 skipped; with stage 1's data,
  314 passed, 3 skipped.

## 2. Overview

```
stage 1  downloads, asset wrappers                  (the task graphs and layouts are committed)
stage 2  PPO: grasp_lift, place_release, push bases ──► teacher_t45_final
stage 3  teacher_t45_final renders + organizer frames ──► kp_v1 ──► kp_v2, orientation references
stage 4  push bases ──(estimator failure model)──► 9a pushes ──► teacher_t45_push9a
         teacher_t45_final ──(DART on the planned controllers)──► Task 5 BC networks
         teacher_t45_final ──(oracle round + 2 DAgger rounds)──► bc_r2
         teacher_t45_push9a + Task 5 BC + kp_v2 + bc_r2 ──► student_v1
         student_v1 ──(oracle_state_v2 round + 3 DAgger rounds)──► bc_v2_r3
stage 5  student_v1 + bc_v2_r3 + orientation references + v1.3 graph flags ──► student_v1_3
stage 6  checks          stage 7  serve
```

| Stage | Script | What | Compute (this machine) |
|---|---|---|---|
| 1 | `1_data.sh` | assets, dataset metadata, Step 1 subset, ten full episodes, asset wrappers | ~1 h, network |
| 2 | `2_teacher.sh` | 12 PPO runs: grasp, place, push bases; freeze `teacher_t45_final` | ~10 h GPU |
| 3 | `3_perception.sh` | renders, organizer frames and labels, kp_v1 → kp_v2, orientation references | ~7 h GPU |
| 4 | `4_student.sh` | noise models, push fine-tunes, `teacher_t45_push9a`, Task 5 BC, the two routers, `student_v1` | ~2 h GPU + ~20 h CPU |
| 5 | `5_build_v1_3.sh` | freeze `student_v1_3` | seconds |
| 6 | `6_checks.sh` | CPU gate, camera loop, end to end through the server | 40 min CPU; 4 h GPU; 25 min |
| 7 | `7_serve.sh` | the challenge endpoint | |

## 3. Stage 1: data (`1_data.sh`)

The script downloads, in order:
- TOWER-Assets and the TOWER-SimData metadata, at the revisions used.
- Three example episodes, for the tests and validation scripts.
- The **Step 1 subset**: 10 episodes per variant, chosen deterministically from the manifest. Only the non-image arrays
  are read, by HTTP range requests.
- The **ten full episodes** (`fetch_full_episodes.py`, ~8 GB). The orientation references and the sequence evaluation
  read them.

It then builds the meter-unit asset wrappers in `assets_m/`. These hold absolute paths, so rerun
`make_meter_assets.py` after moving the repository.

The configs that Step 1 derived from the subset are committed and used as they are:
- `configs/layouts/<variant>.yaml` (`scripts/build_layouts.py [--t45]`);
- `configs/skills/posture_priors.yaml`;
- `configs/robot/nero.json` (`scripts/extract_robot_model.py`).

The scene and evaluator checks of the original Steps 2-3 are in the main [README](../README.md), under "Agentic teacher".

## 4. Stage 2: the teacher (`2_teacher.sh`)

The teacher is a typed graph of skill calls per variant, run by a deterministic executor (`tower_rl/graph/`). Its
skills are trained with PPO in a light one-arm scene on the GPU. The actor sees camera-estimable inputs; the critic is
privileged. Rehearsals run the whole station on CPU PhysX, as the evaluator does. See
[Teacher-v1.MD](../docs/Teacher-v1.MD).

### 4.1 Task graphs, and what Claude did

The graphs are committed: `tower_rl/graph/graphs/<variant>.json`. **Use them as they are.**

They were compiled by `python -m tower_rl.graph.graph <variant ...>` from the layouts and the skill configs. The Tasks
1-3 graphs then went through an agentic edit loop:
1. `scripts/rehearse.py` writes a `feedback.md` per variant and tier: metrics with Wilson intervals, results by
   robustness case, per-node outcomes, failure signatures, failed seeds.
2. The agent, Claude Code (Opus 5.5) in the build session, proposed up to five edits per iteration.
3. The edits were rehearsed again on the same seeds. An edit was kept only if it beat the interval, or fixed a named
   failure signature without regressions.

Three iterations ran:
- `on_fail: skip_if_close`: reverted;
- `v_max_held` 0.08: kept, graphs v4;
- `perception.jaw_to_goal` 0.5: kept, graphs v5.

Each iteration is recorded in `tower_rl/graph/edits/<date>_<n>/`: `prompt.md`, `diff.patch`, before/after metrics,
`notes.md`. The Tasks 4-5 graphs are the compiler's output with defaults set by hand.

Recompiling today reproduces 8 of the 10 committed graphs byte for byte: compile, then set version 3 on Tasks 1-3 and
apply the two kept `diff.patch` files. The Task 4 graphs predate two config edits, `depart_m` and
`gripper_closed_width` 0.0438 → 0.0425, and the second one is read by the executor. So do not recompile Task 4.

Claude's role overall:
- It wrote the code base in Claude Code sessions run by the author, following a build specification and per-session
  plans.
- It ran the graph edit loop above.
- It wrote the recovery rules (`tower_rl/recovery/rules.py`: R4.1 re-push in Task 4; R5.1 retry the insert and R5.2
  push home in Task 5). The v1.3 router learned their labels.

All of this was offline. Nothing at inference, training or data collection calls Claude. A live Claude call for the
cases where the rules abstain was planned (Plan-v1.1 §5.3) but not built. Abstentions are logged for offline review
(`scripts/recovery_review.py`).

### 4.2 Skills

Every training run is `python scripts/train_skill.py --headless --skill <s> --arm <a> ...`. The seed is 42 (the
default), and the config is `configs/skills/<s>.yaml` plus `--override`. Each run directory holds its `config.json`. The
bundle takes TorchScript exports of the checkpoints in bold:

| Skill, arm | Chain (tag: iterations) | Envs |
|---|---|---|
| grasp_lift, left | v2: 0-100 → v3: 100-300 → v4q_from300: 300-599 (grasp quality gate) → **v5n_noise: 599-848** (perception-noise model on the actor) | 4096 |
| grasp_lift, back | left v3 @ 300 → v1_from_left300: 300-500 → v2q_from500: 500-799 → **v3n_noise: 799-1048** | 4096 |
| place_release, left | **v1: 0-400 (`model_400.pt`)** | 4096 |
| place_release, back | **v1_scratch: 0-599** | 4096 |
| push, left | v3: 0-299 → stage 4 (widened, fine-tuned) | 3072 |
| push, back | v1: 0-299 → v1c: 299-548 → stage 4 | 3072 |

Judgement calls, as made:
- v2's reward had a hovering bug. It was killed at iteration 100, the reward fixed (`success_per_remaining_step`
  3 → 8), and training resumed.
- The back grasp starts from the left grasp's iteration 300.
- Place-left was stopped at 400 for a planned fine-tune. Head to head on CPU, it then beat that fine-tune and stayed.

Where the originals were stopped by hand, the script gives the `--max_iterations` that ends at the same checkpoint. rsl_rl
resumes at the checkpoint's iteration and saves every 100 iterations. Runs made before a config key existed switch
it off by `--override`, with the same behaviour.

### 4.3 Gates (500 episodes, `scripts/evaluate_skill.py`)

| | D0 | D1 |
|---|---|---|
| grasp left | 100 % | 98.8 % |
| grasp back | 99.2 % | 100 % |
| place left (no noise) | 99.6 % | 99.0 % |
| place back (no noise) | 99.8 % | 100 % |

The pushes are gated after stage 4.

### 4.4 Teacher bundle and acceptance

`scripts/freeze_bundle.py` writes `results/bundles/teacher_t45_final/`: the graphs, the configs, the six skills as
TorchScript, a MANIFEST with hashes, and a tar. Insert and push-in stay planned (`planned_fallbacks`): in the CPU
evaluator the planned closed-loop versions beat the GPU-trained ones.

The Step 7 acceptance rehearsal (README "Agentic teacher") is
`scripts/rehearse_many.py --bundle <b> --episodes 100 --parallel 4`. It passes at TSR ≥ 0.8 / 0.6 on D0 and ≥ 0.6 / 0.4
on D1 (2-layer / 3-layer), with ODR < 0.1.

## 5. Stage 3: the pose estimator (`3_perception.sh`)

kp_v2 is an ImageNet ResNet-34 U-Net. It outputs full-resolution maps of block corners, block edges, the visible-block
mask, and block surfaces hidden by an arm. `tower_rl/perception/estimator.py` fits every block's 6-DoF pose to those
maps in all four cameras. See [Pose-estimator-v1.MD](../docs/Pose-estimator-v1.MD). The script follows the original
order, which matters:

1. **Renders.** `teacher_t45_final` runs episodes in Isaac. The four cameras are captured at random rows.
   - Training: `data/perception/r1`, seed 20000, 16 episodes per variant and tier, 28k frames.
   - Validation: `r1_val`, seed 90000, 7k frames.
2. **Organizer frames** from the evaluator's renderer, for the Step 1 subset episodes:
   - validation: episodes 0-2, 24 frames each;
   - training batch 1: episodes 3-7, 40 frames each.

   They are labelled from Isaac depth, occlusion-aware.
3. **kp_v1**: 25,000 iterations on renders plus organizer frames (weight 3), with the frames' raw foundation labels.
4. **Foundation corrections.** The organizer episodes' foundation pads sit a few mm off their recorded poses, so they
   are registered and corrected.
5. **Batch 2**: episodes 8-9. They are corrected before they are labelled, then everything is relabelled.
6. **kp_v2**: kp_v1's weights fine-tuned for 12,000 iterations, lr 5e-4, cosine schedule. `model_last.pt` is deployed.
   - On renders: corner recall 0.986, precision 0.978, median corner error 0.18 px.
   - On organizer validation frames: median position error 0.198 mm tracking, 0.183 mm at an episode's first call.
7. **Orientation references** (`results/perception/orient_refs.npz`): the first frame of each full episode's two fixed
   cameras. The rebuild is byte-identical.

The estimator thresholds are code defaults (`EstCfg`); the camera model is in `cameras.py` and `configs/scene.yaml`.

## 6. Stage 4: the student (`4_student.sh`)

The steps can be run one at a time (`reproduce/4_student.sh distill router_v2`). See [Student-v1.MD](../docs/Student-v1.MD)
for the design: token features, oracle, option runtime, DAgger.

1. **`noise`: estimator noise models.** The student's state-mode training replaces the camera with a fitted model of
   kp_v2's errors: per-call error traces by block class, losses and coasting. Two fits are used:
   - v1 (`noise_v1_0034.yaml` + `traces.npz`) trains every switcher;
   - v2 (`configs/student/noise.yaml` + `traces_v2.npz`) trains the Task 4 pushes.

   They were fitted from the per-call logs of camera-loop rehearsals with kp_v2 (`scripts/fit_estimator_noise.py
   --min_calls 40 [--out_bank ...]`). The logs are not reproducible exactly, so the fits are shipped as data in
   `reproduce/noise/` (8 MB) and installed into `results/student/noise/`.
2. **`push9a`: Task 4 pushes under the estimator's failure model.**
   - The push target is lost with the fitted P(lost | tilt, travel) and coasts. The actor gets two more inputs, seen
     and since-seen (65 → 67, so the bases are widened first).
   - Left: 202 iterations under the v1 fit (in the original, the v2 refit had not landed yet), then 150 under v2.
   - Back: 250 iterations under v2.
   - Gates on D1 under v2: left 98.8 %, back 95.4 %.
   - Then `teacher_t45_push9a` = `teacher_t45_final` with these pushes. The original was a copy with the two exports
     written over it, and its MANIFEST still lists the old pushes; a fresh freeze gives the same graphs and weights.
3. **`distill`: Task 5 grasp, insert and push-in by behaviour cloning.**
   - DART (Ornstein-Uhlenbeck noise, scale 1.5) on the planned controllers' rows: 1,200 episodes, 328k rows,
     ~2.3 h of CPU.
   - One small MLP per skill and arm (`train_distill.py`, defaults). "v2" masks the previous action from the inputs:
     v1 drifted by copying it.
   - Gate G9c, against the planned nodes: 83 / 96 vs 78 / 96.
4. **`router_v1`: the first router `bc_r2`** (features v1).
   - An oracle round: 3,000 episodes, exploration 0.1, failure injection 0.15.
   - Two DAgger rounds, β 0.5 and 0.
   - **Only `student_v1`'s switcher slot uses it.** The v2 rounds pass their decider explicitly and v1.3 replaces the
     slot. So its weights do not matter for v1.3, and it ran on older code (§10).
5. **`student_v1`**: `teacher_t45_push9a` + the six Task 5 BC networks + kp_v2 + `bc_r2`, with every graph flagged
   `"distilled": true`.
6. **`router_v2`: the v1.3 router `bc_v2_r3`** (features v2). Labels come from `oracle_state_v2`, the privileged
   oracle plus the recovery rules. Each round is one CPU process per variant, on `student_v1`:
   - D1 with the ten robustness cases cycling;
   - the estimator noise v1;
   - the official mirror: episodes run to 210 s, success on the final arrangement;
   - failure injection 0.15, including short pushes.

   | Round | Decider | β | Episodes per variant, seed | Then train on |
   |---|---|---|---|---|
   | r0 | `oracle_state_v2` (exploration 0.1) | | 120, 8000 | r0 → `bc_v2_r0` |
   | r1 | `bc_v2_r0` | 0.5 | 60, 8100 | r0-r1 → `bc_v2_r1` |
   | r2 | `bc_v2_r1` | 0 | 60, 8200 | r0-r2 → `bc_v2_r2` |
   | r3 | `bc_v2_r2` | 0.5 | 120, 8300 | r0-r3 → **`bc_v2_r3`** |

   r3 adds the tier D1e (the case `irregular_edge`), injected off-centre grips (4-8 mm, a third of the episodes) and
   skipped buried goals on Tasks 1-3.

   `bc_v2_r3` is a 3-layer, 4-head, d 128 transformer over block, goal, arm and global tokens, trained on 65,242
   decisions. The best epoch was 7: validation accuracy 0.968 (nominal 0.9995, non-nominal 0.955). Offline gates are
   ≥ 0.965 / 0.95 / 0.995 (decision accuracy / non-nominal accuracy / progress AUC).

## 7. Stage 5: freeze `student_v1_3` (`5_build_v1_3.sh`)

`student_v1_3` is `student_v1`'s skills, graphs and kp_v2, plus:
- `bc_v2_r3` as the switcher;
- `perception/orient_refs.npz`;
- v1.3's graph flags:
  - every graph: `idle_chunk` 16, `blind_guard`, `finished_guard`;
  - Tasks 1-3: `offcentre_place` (≥ 4 mm grips take the planned careful place);
  - Task 4: the recovery node's R4.1 re-push (`recovery`, `repush_v2` ≤ 0.10 m), with `router_repush: false`;
  - Task 4-3: also `push_settle`.

The script then runs `compare_bundle.py` against `reproduce/manifests/student_v1_3.json`, the MANIFEST of the bundle
that was evaluated:
- **settings** (graphs, task specs) must be identical;
- **weights** differ after retraining;
- **configs** are copies the runtime does not read (it reads `configs/`).

The task specs are rebuilt from the graphs by the freeze. Rebuilt from the original checkpoints, all 73 files match
(§10).

## 8. Stage 6: checks (`6_checks.sh`)

| Check | What | `student_v1_3` |
|---|---|---|
| `cpu` | The student on CPU, the estimator noise v1 in place of the camera, the official mirror. All ten variants × the ten D1 cases, 12 episodes per variant, two seeds | mirror points 80.09 (seed 7700), 77.56 (seed 7900), of 100 |
| `camera` | The camera loop: Isaac renders the four cameras, kp_v2 estimates, the student acts. 10 episodes per variant | Tasks 1-3, seed 7200: 59 / 60 task successes |
| `e2e` | Through the challenge protocol: `serve_student.py` (rendered images) and a WebSocket client driving the station, 4 variants × 2 episodes | 0 errors |

Ten-episode camera cells resolve about one point, so compare a rebuild with these numbers loosely. "Mirror points" is
our reconstruction of the official score: TSR, LHPR and ODR per variant, weights 1 : 1 : 1 : 3 : 3 across the task
families (`tower_rl/eval/mirror.py`).

## 9. Stage 7: serve (`7_serve.sh`)

`scripts/serve_student.py` serves the bundle with the challenge kit's protocol (`tower-infer-v1`) at
`ws://127.0.0.1:9004`:
- it verifies the bundle's MANIFEST;
- the artifact identity is the tar's sha256;
- the camera source is "evaluator".

The challenge needs a public `wss://` URL on port 443, e.g. `cloudflared tunnel --no-autoupdate --url
http://127.0.0.1:9004`. `scripts/probe_server.py` runs the qualification-like checks: every variant, resets, blank
images. The site flow and our endpoint tooling (watcher, guards, tunnel scripts) are described in the main README,
"Serving the student".

## 10. How exact is this (checked 2026-11-01)

**Reproduces exactly**
- **Freezes.** From the original checkpoints, `5_build_v1_3.sh` rebuilds all 73 files of `student_v1_3` byte for
  byte. `build_v1_4.sh` does the same for all 77 files of v1.4 and of v1.5. Only the MANIFEST timestamp and the tar
  hash differ, so a rebuilt bundle has a new artifact identity.
- **Teacher freeze.** `freeze_bundle.py` with the six checkpoints gives identical graphs and identical weights.
  TorchScript re-exports differ in their serialization ids only.
- **Graphs and specs.** v1.3's graphs are the committed graphs plus the freeze flags, 10 / 10. The task specs match
  10 / 10. The committed graphs recompile 8 / 10 (§4.1).
- **Orientation references.** The rebuild is byte-identical.
- **The code.** The repository runs the evaluated `student_v1_3` bundle exactly as the snapshot it was served from.
  The check: all ten variants on CPU with oracle perception, the official mirror, D1 nominal and heavy, 2 episodes
  each. Every episode record was identical apart from wall-clock timing fields. The perception code is byte-identical
  to that snapshot's.
- **Rehearsals.** Rehearsals are deterministic for a fixed code, bundle and machine: same seeds, same episodes.

**Equivalent, not identical**
- **PPO on the GPU** (PhysX, TF32) and **kp_v2's training** (augmentation seeded from `os.urandom`): a rerun gives new
  weights, to be judged by the gates.
- **The skill runs made before a code change on 2026-10-19** (grasp up to v4q / v2q, both places) drew the
  observation jitter twice per step. The code now draws it once and has no switch for the old behaviour.
- **Rendering** is not bit-exact.
- **CPU training** of the switchers and BC nets can differ across machines and thread counts.
- **DAgger compounds:** every round acts with the previous round's switcher.
- **The v1 router rounds** (`bc_r2`) ran on older code: 15 modules have changed since. This does not matter for v1.3
  (§6, step 4).

**Not in this repository**
- The weights, and every run output: checkpoints, datasets, rehearsal records.
- The Claude Code session transcripts.
- The challenge reviewers received the evaluated bundles, the evaluation logs and sanitized transcripts privately.

**Known issues**
- `scripts/eval_switcher_offline.py` reads shards with features v1 only, so it fails on v2 data. The pipeline does not
  use it: v1.3's offline gate is the training run's validation split.

## 11. Later versions: v1.4 and v1.5 (after the deadline, never evaluated)

`student_v1_4` is v1.3 with learned PPO reachers for the transits (left 9700, back 8500); see
[Reacher-V3.MD](../docs/Reacher-V3.MD) and [Student-v1.4.MD](../docs/Student-v1.4.MD). `student_v1_5` uses back
reacher 9200 instead.

Both freeze with `build_v1_4.sh` from `student_v1_3` and `configs/reach/teacher_p2f*.json`.
- **v1.4** scored the same as v1.3 in the Tasks 1-3 camera loop: 59 / 60.
- **v1.5:** a 512-episode A/B against v1.4 found no difference.

## 12. Where things are

| | |
|---|---|
| `tower_rl/graph/` | graphs, executor (planned nodes, skill nodes, failure handling), graph edit records |
| `tower_rl/skills/` | PPO skill environments, export, reacher |
| `tower_rl/perception/` | pose estimator: network, fit, tracker, cameras, orientation choice |
| `tower_rl/student/` | tokens, oracles, option runtime, switcher, DART and BC nodes, estimator noise |
| `tower_rl/recovery/` | recovery contracts and rules |
| `tower_rl/server/` | the challenge server adapter |
| `tower_rl/eval/` | metrics, operation tracker, official mirror |
| `scripts/` | training, rehearsal, data, freezing, serving; `scripts/official/`: tools for the official runs |
| `configs/` | physics, scene, tiers, layouts, skills, student, reach |
| `reproduce/manifests/` | the MANIFESTs of the bundles as built: every file's sha256 and the freeze arguments |
| `reproduce/evaluated/` | serving an evaluated submission from its `eval/*` tag; the scripts that served the official runs |
| `reproduce/environment/` | package versions, the machine, challenge and data revisions |
| `docs/` | write-ups (copies of the session documents; paths in them refer to the development machine) |
