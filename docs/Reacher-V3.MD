# Reacher v3: several rows per call — session hand-off

*Written 2026-10-31 ~21:10 PDT by Claude Code (Opus 5.5), for the next session after `/clear`. It covers 10-31
14:39-21:10 (session 10): the request was still "read `Reach-PPO-plan.MD` and `Reacher-V1.MD` and continue the
implementation", and this session took `Reacher-V2.MD` §7 steps 1 (chunking) and 2 (the student's camera loop). Paths
are relative to `~/Documents/tower_rl` unless they start with `~` or `corl_challenge/`.*

**Update 2026-11-01:** §7 step 3 is done: `student_v1_4` = v1.3 + these reachers (`corl_challenge/Student-v1.4.MD`).

**Read first:** §0 (state), §7 (next steps). Rules: `Reacher-V1.MD` §1 and `Reacher-V2.MD` §1 still apply; nothing new.

Related: `Reacher-V2.MD` (held-block reacher, P2d), `Reacher-V1.MD` (glossary, P0-P4), BUILDLOG "2026-10-31 (session
10)", memory `tower-reach-ppo.md`.

---

## 0. State at hand-off

### 0.1 What exists now

- **Chunk-trained reachers for both arms**: they act on their own joint prediction inside a call, as the evaluator
  executes up to 16 rows per call. Exports: `results/reach/export/reach_left_p2fck9700.pt`,
  `reach_back_p2fck8500.pt` (74 actor inputs).
- **Env option `chunk`** (`tower_rl/skills/reach_env.py`): the first row of each chunk sees the measured joints. The
  rest see the prediction `q_des − (q_des − q)`, with the offset measured on that first row, plus an input counting
  rows since the measurement.
- **Deployable `Reacher.step(None, ...)`** computes the same prediction. Env/deployable parity: **0.0 / 0.0 / 0.0**.
- **Executor**: chunk-trained reachers run 16-row chunks all the way to the goal (`_trained_chunk`), plus a via-point
  stall rule (`via_stall_s`).
- **Named config `configs/reach/teacher_p2f.json`**: both chunked reachers, the pre-push and pre-push-in transits
  planned.
- Tests: `tests/test_reach_policy.py` gained a chunk test and a via-stall test (6 tests, all pass). The full suite
  passed before the via-stall test (315 collected, 3 skipped).

### 0.2 Headline results (CPU PhysX, the evaluator's physics)

**Gates, 16-row chunks to the goal** (held-out transits, via-points + moving goal):

| Arm (model) | All transits: in band / settled | p50 error | Calls to arrival p50 | Held D1 cases: drops | Held drift p95 (heavy) | Held arrival vs planned |
|---|---|---|---|---|---|---|
| Left (9700), 0.05 m/s held | 515: 100 % / 100 % | 1.1 mm / 0.62° | 3 (held pre-place 7) | **0 in all 7** (164 each) | 3.6 mm (planned 5.1) | 10.5 s vs 11.5 |
| Back (8500), 0.15 m/s held | 544: 99.5 % / 99.5 % | 0.9 mm / 0.35° | 3 (held pre-place 4) | **0 in all 7** (208 each) | 0.8 mm (planned 0.7) | 5.3 s vs 10.6 |

Before training, the P2d models under the same 16-row chunks dropped 16 of 18 nominal held blocks.

**Teacher** (ten variants, D0 + D1, 8 episodes, seed 9600):

| Setup | TSR / 160 | GSR / 608 | PSR / 608 | Cells below planned | Calls per episode, all cells (Tasks 1-3) |
|---|---|---|---|---|---|
| Planned transit everywhere | 141 | 588 | 577 | — | 13,352 (3,646) |
| P2d, closed loop (session 9) | 142 | 586 | 577 | 4-2 D0 | ~2x planned on Tasks 1-3 |
| **`teacher_p2f.json`: chunked, pre-push / pre-push-in planned** | **141** | **588** | 576 | **none** | **13,250 (3,330): −0.8 % (−8.7 %)** |
| Chunked, every transit learned | 140 | **596** | 584 | 4-3 D0, 5-2 D0 | — |

- Task 1-3 episodes are 15-34 % shorter than planned (1-2 69.4 vs 81.9 s, 2-2 34.6 vs 48.0, 3-3 38.8 vs 56.5).
- Task 4 reproduces the planned run exactly: every Task 4 transit is a pre-push.
- The plan's gate is met: no cell worse than planned by more than one episode, and calls ≤ planned + 10 %.

**Student** (camera loop: student_v1_3 on our renders, mirror, ten D1 cases, `teacher_p2f.json`; §3.4):

| Variants (10 episodes per seed) | With the chunked reachers | v1.3 bundle, same seeds |
|---|---|---|
| Tasks 1-3 (seed 7200) | **59 / 60, 33.02 points** | 59 / 60, 32.87 |
| Task 5-2 (seed 7400) | 8 / 10, 14.45 | 8 / 10, 14.34 |
| Task 5-3 (seeds 7400 + 7500) | 5 + 4 = 9 / 20, 23.04 | 2 + 7 = 9 / 20, 22.71 |

- Not worse anywhere, and no switcher retraining was needed.
- Task 5-3 swings 2-7 / 10 for one config across seeds, so read single-seed camera results there as noise.

### 0.3 Main lessons

1. **Training under the deployment's information pattern fixes chunking.**
   - Closed-loop models fail under chunks: they carry at ~2x the moving goal's speed, drop held blocks and fall into
     2-row limit cycles.
   - Trained with chunked observations from P2d, the left arm met its gate after 100-300 iterations. The back arm
     started from settled 0.00 and needed ~500 iterations.
   - The back arm's joints lag q_des longer: (kd + viscous) / kp ≈ 0.08-0.14 s on joints 4, 5 and 7, and joint 1
     (kp 63) is lightly damped. A constant offset held over a chunk is therefore further from the truth after the arm
     decelerates.
2. **A simple prediction is enough if the policy is trained with it.** No arm model is needed: `q_des − offset` plus
   the rows-since-measurement input.
3. **Contact skills are sensitive to where the transit before them leaves the arm.**
   - With learned pre-push-in transits, Task 5-2 D0's push-in moved the structure 23-62 mm in 1-2 of 8 episodes,
     although the pinch arrived identically (1.5 mm / 0.3°). Task 4's pushes are the same.
   - Keeping those two short transits planned costs nothing: they take 2.5-3 s either way, in 2-3 calls.
4. **Via-points are for clearance.**
   - One Task 5-3 held pre-insert parked 4 mm from its via-point, 20.2° off, with joint 7 at its limit margin.
   - `via_stall_s` (2 s: once the carrot has sat there, position alone is enough) fixed it.
   - That miss came from the model (it reproduced closed loop), not from chunking.
5. **GPU training slows the server's perception 2-3x** during the evaluation (it is ~95 % of server time).
   - Task 4-2: mean 437 ms, p90 1.3 s, all in push nodes. Official run 1 without training: 120 ms. Lowest while only the
     keeper's stand-in ran: Task 2-2 135 ms.
   - 0 errors.
6. Rehearsal `act_calls` is counted per batch of 4 envs: a batch runs until its last episode ends, so one long episode
   raises its batch's count. Compare like for like.

### 0.4 Running at hand-off

- **The v1.3 official evaluation is complete from the server's side** (11:37:27 to ~22:15).
  - 100 episodes, 10 per variant, all to the 210 s limit, 0 errors.
  - 172 sessions, 72 of them empty: reconnects, the evaluator's usual pattern. It re-ran missing episodes at the end,
    as in official run 1.
  - Scores arrive on the challenge site. The server (tmux `tower_v1_3`, pid 19527) still serves.
  - Watcher, in case the evaluator comes back:
  ```
  cd ~/Documents/tower_rl && WATCH_NAME=official_v1_3 WATCH_QUIET_START=1 python3 -u scripts/official/watch_official.py 2026-10-30T08:44:00 3600 1800 first
  ```
- The student camera loops with the chunked reachers are done (Tasks 1-3 and 5, §3.4).
- The Task 5-3 second-seed check is done (`results/reach/chunk/cam_t53_seed7500.sh`, 21:01-22:14): no lead (§3.4).
  The keeper's stand-in is the GPU load again. Nothing of this session is still running.
- The back final model (9200) is done (§3.2): more accurate than 8500 in the battery, 139 / 160 in the teacher. The
  two lost episodes are left-arm places, from cross-env coupling. `teacher_p2f.json` keeps 8500.
- Keeper `scripts/bg_load_keeper.py` (pid 46438): stand-in load whenever no live `train.pid`; stop with `touch
  results/reach/STOP_BG`. It was for the evaluation; whether to stop it now is the user's call.

### 0.5 Not done or unverified

- No Task 4 camera check (its transits are planned in `teacher_p2f`), and no switcher DAgger round (not needed so far).
- No bundle / server path for the reacher: exports are not in a bundle, and there has been no websocket end-to-end
  run.
- The left arm still carries at 0.05 m/s. Recovery pushes and manipulation paths (`ReachPathNode`) are not started.
- Nothing is committed (§5).

---

## 1. Rules

Unchanged (`Reacher-V1.MD` §1, `Reacher-V2.MD` §1).
- During the evaluation: one GPU job (training at 1,024 envs or a camera loop, its pid in `train.pid`) and ≤ 2 CPU
  rehearsal / check processes.
- The camera loop counts as one of those CPU processes (physics on CPU).

---

## 2. What was built

### 2.1 Env option `chunk` (`tower_rl/skills/reach_env.py`, `configs/skills/reach.yaml`; default `lengths: [1]`, off)

- Chunk lengths are drawn per chunk from `lengths` / `weights`. Training used 1 / 4 / 8 / 16 with weights 0.15 / 0.15
  / 0.2 / 0.5. A new goal starts a new chunk (`_get_rewards`); a reset starts one too.
- **The actor's view** (`_get_observations`):
  - On a chunk's first row (`ck_k == 0`): measured joints, and the offset `ck_off = q_des − q` and opening `ck_w` are
    re-measured.
  - Inside a chunk: `q_obs = q_des − ck_off`, and the opening is the measured `ck_w`.
  - qd comes from the previous row's view (`q_obs_prev`, set in `_pre_physics_step`, which also advances the chunk
    phase).
  - The pose error and pinch pose come from FK of `q_obs`.
- **`obs`** adds inputs:
  - Actor: rows since the measurement / `k_norm` (16), as the last input (`CHUNK_DIM` 1).
  - Critic, right after the actor's inputs: the true joints (7) and the true pose error (9) (`CHUNK_CRITIC_DIM` 16).
  - Widen a P2d model with `scripts/widen_push_checkpoint.py <in> <out> 73 1 17`.
- Rewards and the settle test use the true state.

### 2.2 Deployable reacher (`tower_rl/skills/reach_policy.py`)

- `ck_obs`, `ck_norm` come from the skill config.
- `predicted() = q_des − off`. `measure(q)` re-measures the offset (`q` None: a row on the prediction, `k += 1`).
- `step(q_meas | None, ...)`; `obs(..., k=...)` takes an int or a per-env tensor.
- An old model fed `step(q)` behaves as before.

### 2.3 Executor (`tower_rl/graph/executor.py`, `LearnedTransitNode`)

- **`_trained_chunk`**: a chunk-trained reacher whose `reach.chunk` / `chunk_held` > 1 runs up to K−1 more rows per
  call on its prediction.
  - Via-points switch on the predicted pinch pose.
  - The chunk ends early once the carrot is on the goal, q_des has been still for `settle.rows` rows and the predicted
    pinch is within `settle`.
  - Arrival is decided on the next call's measured pose, with q_des still (`cmd_still`): one measurement per call.
  - The IK finish (`finish_chunk`) goes out in the same call.
- `chunk` / `chunk_held` may be per-arm dicts, e.g. `{"left": 16, "back": 1}`.
- **`via_stall_s`** (default 0 = off; `teacher_p2f.json` 2.0): the next via-point is also taken once the carrot has sat
  on the current one that long and the pinch is within `via_tol`'s position part (`_via_reached`, `slot_rows`).
- Models trained closed loop keep the old chunk rule (rows on q_des while the carrot travels).

### 2.4 Checks and scripts

- `scripts/reach_task_check.py`:
  - `--chunk K --chunk_final chunk`: chunks to the end. The default `closed` is the old rule.
  - `--chunk_pred` now defaults to `offset` (the deployable's rule).
  - `--via_stall S`.
  - `--parity --chunk K --chunk_final chunk`: the env's own fixed-length chunks vs the deployable.
  - With a chunk-trained model, the env's chunk length is set to `--chunk` (1 = closed loop).
- `results/reach/chunk/check_chunk.sh <arm> <ckpt> <tag> <@override>` (`K`, `VH`, `VE`, `STALL`, `JOBS`): all
  held-out transits (nominal block) plus the seven D1 held cases.
- `results/reach/chunk/rehearse_p2f.sh <left export> <back export> <tag>` (`K`, `STALL`, `JOBS`): the teacher with
  chunked reachers. Add `planned_targets` through `EXTRA` in `p4/rehearse_all.sh`, as `p2f_both_k16_pp` did.
- `results/reach/chunk/gpu_student_cam.sh '<graph_set>' <name> [variants]`: the student camera loop with a reach
  config (Tasks 1-3 settings).
  - `gpu_student_cam2.sh` is the same with `SEED`, `ORIENT` (empty: no `--official_orient`) and `PART` (output
    name), e.g. Task 5: `SEED=7400 ORIENT= PART=t5`.
  - Both write their own pid to `train.pid` (one GPU job, the keeper's stand-in paused).
  - `'{}'` as the graph_set runs the unchanged bundle.

---

## 3. Results in detail

### 3.1 Left (`results/skills/reach_left/2026-10-31_1452_989779b262_p2fck`, from P2d model_8700 widened)

- Training `p2fck`: 1,000 iterations at 1,024 envs, 4.4 s / iteration.
  - Metrics: settled 0.19 → 0.62 after 65 iterations → 0.92 at the end; held drops 0.36 → 0.10 per held episode.
- Batteries (CPU, 16-row chunks to the end, carrot 0.05 / 0.4):

  | Model | Heavy drops / 164 | Heavy drift p95 | All 515: band, p50 | Note |
  |---|---|---|---|---|
  | 8800 | 0 | 2.8 mm | — | pre-insert drift p95 ~16 mm, the same closed loop (P2d 16.1) |
  | 9000 | 0 (all 7 cases) | 2.9 mm | 99.8 %, 0.9 mm / 0.84° | 1 pre-insert via stall (before `via_stall`), 2 pre-grasps with 78° turns settle 2.7-2.8° off |
  | 9500 | 1 | 3.8 mm | 100 %, 1.2 mm / 0.76° | |
  | **9700 (final, chosen)** | 0 (all 7 cases) | 3.6 mm | 100 %, 1.1 mm / 0.62° | |

- Teacher with only the left chunked (back P2d closed loop): 9000 141 / 160; 9700 140 / 160, with GSR 592 and PSR 582.
  In both, the losses were in Task 4 and 5-2 D0 (learned pre-push / pre-push-in arrivals).

### 3.2 Back (`results/skills/reach_back/2026-10-31_1602_1e329219f1_p2fck`, from P2d model_8200 widened)

- Training: 4.8 s / iteration, held goals 0.03-0.2 m/s (deployed at 0.15).
  - Settled 0.00 → 0.15 after 95 iterations → 0.55 after 131 → 0.86-0.90 from ~550.
  - Held drops 0.25 → 0.
- model_8500 battery (CPU, carrot 0.15 / 0.4):
  - All 544 transits: in band 99.5 %, p50 0.9 mm / 0.35°, p95 1.9 / 1.19.
  - Held D1 cases: 0 drops in all seven; drift p95 nominal 0.5, light 0.3, heavy 0.8, uneven 0.7, slippery 0.5,
    high 0.5, combined 0.7 mm (closed-loop P2d: the same to 0.1 mm).
  - Held arrival 5.3 s (P2d closed loop 4.4, planned 10.6); held pinch speed p50 0.20 m/s.
  - The 3 misses are Task 4-3 D1 pre-pushes (seed 9700000, transits 147 / 154 / 158) stuck at the first via-point.
    Closed-loop P2b fails them the same way (its pinch dips to −21 mm), and `teacher_p2f` plans pre-pushes.
- **model_final (9200)**:
  - All 544 transits: in band 99.5 % (the same 3 pre-pushes), **p50 0.6 mm / 0.32°, p95 1.0 / 1.16**.
  - Held D1 cases: 0 drops in all seven, drift p95 ≤ 0.8 mm, held arrival 5.2 s.
  - Teacher (`p4/p2f_both9200_k16_pp`, the `teacher_p2f` settings): TSR 139 / 160, GSR 588. The two lost episodes are
    Task 2-3 left-arm places (`t1_place` at 35 s, one per tier) with the left model unchanged, i.e. coupling through the
    shared CPU scene. Other cells are equal.

### 3.3 Teacher, both chunked (`results/reach/p4/p2f_both_k16_pp`, `p2f_both_k16`)

- Compare: `python3 scripts/reach_p4_compare.py results/reach/tasks/rehearsal_heldout results/reach/p4/p2f_both_k16_pp`.
- **`_pp` (= `teacher_p2f.json`)**: no cell below planned.
  - Tasks 1-3: 8 / 8 everywhere.
  - Calls per episode, D0 / D1 (planned in brackets): 1-2 384 / 340 (313 / 325), 1-3 528 / 710 (569 / 774), 2-2 130 /
    146 (260 / 166), 2-3 244 / 208 (243 / 252), 3-2 112 / 120 (136 / 150), 3-3 196 / 212 (212 / 246).
  - Task 4 identical to planned.
  - Task 5: 5-2 8 / 8 + 5 / 8, 5-3 8 / 8 + 5 / 8 (= planned), calls within +8 %.
- **All learned**: 4-2 8 + 5 (planned 8 + 4), 4-3 5 + 1 (6 + 1), 5-2 6 + 6 (8 + 5), 5-3 8 + 5.
  - Its GSR 596 vs 588 says the learned pre-pushes place / push more goals in D1 while losing whole tasks in D0.
- Task 3-2 (the closed-loop chunking failure: a 2-row limit cycle at the back arm's pre-grasp): 8 / 8, 24.8 s (planned
  37.0).

### 3.4 Student camera loop (`results/reach/student/p2f_t13`, `t13_points.md`)

- The setup: `student_v1_3` (bc_v2_r3 router, kp_v2 on our Isaac renders, the official mirror to 210 s,
  `--official_orient`), the ten D1 cases, seed 7200 × 10 per variant, `configs/reach/teacher_p2f.json`.
- **Tasks 1-3: 59 / 60, 33.02 points** (v1.3 Step 7a reference `results/v1_3/camera/r3/t13`, same seeds: 59 / 60,
  32.87). Task 1-3 9 / 10, every other variant 10 / 10.
- The miss (seed 7200001, lightweight) is goal 0's first place, judged on the final arrangement:
  - the grasp rotated 2.8°, the held transit arrived 3.6 mm / 1.39° from the pre-place goal, and the place ended at
    yaw 4.99°;
  - the reference run on that seed arrived identically with the planned transit (3.5 mm / 1.36°: the goal, not the
    reacher) and placed at 4.69°, just inside the limit.
  - The reference's own miss was Task 2-2 high_friction.
- The router (trained with planned transits) handled learned, chunked transits without a DAgger round. No reworks.
- Runtime cost: 24.2 ms per call vs 18.8 ms. Calls per mirrored episode are unchanged (614 vs 618: idle chunks to
  210 s dominate). Batches take 305-435 s vs 250-300 s: the shared GPU, with the server's perception, during the
  evaluation.
- **Task 5** (`results/reach/student/p2f_t5`, seed 7400 × 10 per variant, no orientation flag, as `r3/t5_cand`):
  **5-2 8 / 10 (14.45 points), 5-3 5 / 10 (12.10); 26.55 points vs the reference's 23.34** (8 / 10, 2 / 10).
  - The 5-3 gain is the first block. The back arm's learned held pre-insert arrives in 4.8 s instead of 12.4 at the
    same accuracy, and the first insert goes deeper (along p50 −91.2 vs −87.6 mm). Its push-in then fails in 1
    episode instead of 5.
  - The second block's push-in fails alike (6 vs 7).
  - 5-2's push-in misses differ by episode only (its pre-push-in is planned in both runs).
  - The privileged teacher showed no 5-3 difference (5 / 8 both).
  - **Not confirmed on seed 7500** (`results/reach/student/{p2f,ref}_t53_7500`, same code): the reacher config scored
    4 / 10 (10.94 points) and the unchanged bundle 7 / 10 (13.71).
  - Over the two seeds: 9 / 20 each, 23.04 vs 22.71 points. The bundle's first insert was no deeper (−88.2 mm) and
    still succeeded, so insert depth was not the cause.
- On the seeds both configs ran, the reacher is not worse anywhere: Tasks 1-3 59 / 60 vs 59 / 60, Task 5-2 8 / 10 vs
  8 / 10, Task 5-3 9 / 20 vs 9 / 20.

---

## 4. Models and overrides

| What | Path |
|---|---|
| Left chunked (gate passed, in `teacher_p2f`) | `results/skills/reach_left/2026-10-31_1452_989779b262_p2fck/model_final.pt` → `results/reach/export/reach_left_p2fck9700.pt` |
| Left 9000 (also passed) | `.../model_9000.pt` → `results/reach/export/reach_left_p2fck9000.pt` |
| Back chunked (gate passed, in `teacher_p2f`) | `results/skills/reach_back/2026-10-31_1602_1e329219f1_p2fck/model_8500.pt` → `results/reach/export/reach_back_p2fck8500.pt` |
| Back final (also passed; teacher 139 / 160) | `.../model_final.pt` (9200) → `results/reach/export/reach_back_p2fck9200.pt` |
| Widened starting points | `results/reach/ckpt/left_p2d8700_chunk.pt`, `back_p2d8200_chunk.pt` (73 1 17) |
| Overrides | `results/reach/override_p2f_chunk.json` (left), `override_p2f_chunk_back.json` (held goals 0.03-0.2 m/s) |
| Teacher config | `configs/reach/teacher_p2f.json` (`rehearse.py --graph_set @configs/reach/teacher_p2f.json`) |

Any check of these models needs their override, for the input layout.

---

## 5. Uncommitted tower_rl changes from this session (for the user's commit)

- **Modified**: `tower_rl/skills/reach_env.py`, `tower_rl/skills/reach_policy.py`, `configs/skills/reach.yaml`,
  `tower_rl/graph/executor.py`, `scripts/reach_task_check.py`, `tests/test_reach_policy.py`, `BUILDLOG.md`.
- **New**: `configs/reach/teacher_p2f.json`, `results/reach/override_p2f_chunk*.json`, `results/reach/chunk/*.sh`.
- Pre-session copies of the six code files: `results/reach/chunk/backup_1031_pre_chunktrain/`.
- Sessions 8-9's reacher work is still uncommitted too (`Reacher-V2.MD` §5).

---

## 6. Known issues

- **Learned pre-push / pre-push-in arrivals** cost Task 4 / Task 5-2 episodes (§0.3.3), so they stay planned.
  Matching the planned transit's arrival posture would be needed to use them.
- **Server perception under GPU load** (§0.3.5): Task 4-2 calls up to 2.4 s while training ran. Nothing failed, but
  the evaluator waits for each call.
- The left arm still carries at 0.05 m/s; the back arm's chunked held carries take 5.3 s vs 4.4 s closed loop.
- Task 1-2 D0 calls are +23 % on planned (384 vs 313), though all cells together are −0.8 %.
- Cross-env coupling in CPU rehearsals still applies: read one-episode differences per cell as noise.

---

## 7. Next steps (in suggested order)

1. The student needs nothing more for the reacher. On every camera check it is not worse than the v1.3 bundle (§3.4).
   Task 4 needs no camera check while its transits stay planned.
2. A switcher DAgger round with the reacher (`bc_v2_r3` → r4) is optional: the camera loops showed no router errors.
   Do it only if a later check does (gates as G10 in `Student-distillation-plan.MD`).
3. **The bundle and server path** — done 2026-11-01: `student_v1_4` (`corl_challenge/Student-v1.4.MD`):
   - Put the exports (with their JSON) and the `reach` block into a bundle (`scripts/freeze_student_bundle.py`
     `--extra reach/<name>.pt=<file>` and `--graph_set`).
   - `executor.reach_policy()` loads export paths relative to the working directory. It needs a fallback to the
     bundle's directory, and `Executor.from_bundle` does not record that yet.
   - Run the websocket end-to-end test (`results/v1_3/gpu_ws_e2e_v1_3.sh` on a spare port).
   - Check server time per call: the reacher adds a few ms per row and 16 rows per call.
4. **Back 9200 vs 8500**: 9200 is more accurate but got 139 vs 141 in one teacher run (coupling noise). A 16-episode
   A/B on Tasks 1-3 would settle it. (Done 11-01, `Student-v1.4.MD` §6: no difference. 491 vs 493 / 512 paired
   episodes over Tasks 1-3 and 5, McNemar p = 0.73. In the tasks, the back arm's arrival errors are equal, because the
   IK finish sets them.)
5. Left carry speed, learned pre-push arrival postures, recovery pushes (`Reacher-V1.MD` §11.3).

## 8. Open questions for the user

1. Commit the reacher work (sessions 8-10) and make `configs/reach/teacher_p2f.json` the reacher's default config?
2. Is a future bundle with the reacher (v1.4) wanted, or does this stay research beside `student_v1_3`? (Answered
   2026-11-01: built as `student_v1_4`, see `Student-v1.4.MD`; not served.)
3. For future evaluations: is GPU training beside the server still wanted, given the 2-3x slower perception (§0.3.5),
   or only the keeper's lighter stand-in?
4. `Reacher-V1.MD` §12's questions are still open.
