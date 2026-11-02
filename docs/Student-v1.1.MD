# Student v1.1 (`student_v1_1`), v1.2 (`student_v1_2`) and the recovery node

*Written overnight 2026-10-27/28 and on the morning of 10-28 by Claude Code (Opus 5.5), working through
`Plan-v1.1.MD`: Step 1 (orientation and the official mirror), `student_v1_1` for submission on 10-28, then improvement
2 (the recovery node: teacher first, then the student's router, then the student's own rule node: `student_v1_2`). Paths are relative to `~/Documents/tower_rl` unless absolute; every number has its run in
`BUILDLOG.md` "session 7".*

**Status (2026-10-28 15:35): `student_v1_2` is queued for the official evaluation (all ten tasks, 10 Hz).** The
site's Check endpoint passed at 15:31 (ten variants, one blank-image call each, 0 errors). All three endpoints stay
up; `student_v1_1` remains the fallback. Earlier status (11:15): two submission candidates were served, both passed
the qualification-like probe 10 / 10 locally and through the tunnel:
- `student_v1_1` at `wss://epson-korea-analog-patients.trycloudflare.com` (port 9002), as planned last night.
- **`student_v1_2` at `wss://windsor-lighting-lean-iii.trycloudflare.com` (port 9003): `student_v1_1` + the
  recovery node's end-of-plan re-push on Task 4** (§0 item 7, §3.4). Recommended: identical to v1.1 until the Task 4
  plan is done, then it re-pushes only a block seen leaning on the base (a failing block); about +1.3 points on
  official run 1's states.

---

## 0. Summary

1. **`student_v1_1` = `student_v1` (same networks) + the per-episode camera orientation + four opt-in runtime flags
   set in the bundle's graphs.**
   - **Orientation:** per episode, the fixed cameras are turned by image correlation with the dataset's first frame.
     125 / 125 official first frames chosen right; the adapter sees every block where the evaluator rendered them.
   - **Finished guard:** once every goal ended ok, a goal is reworked only if its block moved > 10 mm for 2 s.
     **This turned out to be essential:** with working perception the v1 switcher reworks finished goals in the 100-160
     s after the plan. Tasks 1-3 under the estimator noise model, run to 210 s: **1 / 120 successes without the guard,
     120 / 120 with it.** Official run 1 never showed it because Tasks 1-3 were blind.
   - **Idle holds of 16 rows** (official episodes run to 210 s) and the **blind guard** (no movable block ever seen:
     run the graph's plan on the nominal poses, no retries on blind place verdicts).
   - **`push_settle`, on Task 4-3 only:** v1's push until its radial test, then hold still until the block is
     confirmed >= 2 mm clear of the foundation footprint and at rest. On simulator poses it helped both variants (two
     seeds x 24: 4-2 TSR 50 -> 67 %, LHPR 86.8 -> 92.8 %; 4-3 LHPR 67.4 -> 72.7 %), but **in the camera loop it hurt
     4-2** (TSR 6 -> 4 / 10, LHPR 90.0 -> 81.7 %: the confirmation trusts the estimate at the base edge, and more blocks
     ended leaning) and helped 4-3 (LHPR 66.7 -> 71.1 %). So it is set in 4-3's graph only.
2. **The local evaluator now mirrors the official one** (`--mirror`): every episode to 210 s, success on the final
   arrangement, Task 4 pushes judged on where the block comes to rest. Re-scoring official run 1 with it reproduces the
   site (4-2 LHPR 73.3 % vs 73.3 %, 4-3 53.3 % vs 56.7 %; TSR within one episode).
3. **The recovery node is built** (contracts as code, recovery rules on truth for the teacher and on estimates for the
   student, an abstention log, a re-push recovery option), **and it finds little to recover today:** on truth it adds
   +0.4 points (Tasks 4-5). The remaining failures are prevention problems: 4-3 bottom-layer blocks landing on the top
   layer's debris, blocks tipping onto their side, and Task 5 structure breaks (10-23 mm) during the insert / push-in.
4. **The recovery node distilled into the switcher works:** features v2 (the contracts in the switcher's tokens, a
   `repush` head), labels from the state oracle v2, round 0 + one DAgger round (1,800 episodes): `bc_v2_r1` scores
   79.62 points vs `bc_r2`'s 78.50 in the same closed loop (noise model, mirror, ten cases, 12 per variant), with a
   fifth of the data (a second DAgger round, `bc_v2_r2`, 77.67: parity within noise). It is not in `student_v1_1`, and
   **it did not pass its camera-loop check** on Task 4 (18.18 vs `bc_r2`'s 21.17 points; 4-2 TSR 2 vs 6 / 10). Much of
   that is the camera loop's noise, but its re-pushes also did nothing there (§3.3).
5. **v1.1 vs v1 on the mirror** (noise model, ten cases, same seeds): `student_v1` as submitted 68.90 points,
   `student_v1_1` 78.50 (+9.6, nearly all from the finished guard on Tasks 1-3; that run had `push_settle` on 4-2 as
   well, which the final build drops: about +0.2 on 4-2 under this noise model).
6. **Why 4-2 blocks end leaning on the base (camera loop, 80 episodes):** when the camera loses a moving block, the
   estimator coasts it on (up to ~0.15 m), and the push node's radial test fired on that coasted pose while the block
   had stopped at the base edge. 10 of the 12 blocks that ended leaning after their push were such pushes; pushes ended
   on a confident estimate had truly departed 188 / 189 times. **Official run 1 shows the same:** 16 of its 58 4-2
   pushes ended on an unconfident estimate, and only 7 of those blocks came to rest flat clear (vs 36 / 42 after a
   confident end); all 3 official leaning blocks were among them. The learned push was trained to push on through these
   losses (its training ends on the true departure), but pushing on blind does not work either: graph flag
   `depart_seen` (the departure test only on seen rows) fixed 2 leaning cases in its 4-2 camera A/B and pushed a block
   that had truly left off the table edge (TSR 3 vs 6 / 10; not adopted). The fix has to look before it pushes again,
   which is what the recovery node's R4.1 re-push does; its student-side gate could not see such blocks (§3.4).
7. **`student_v1_2` = `student_v1_1` + the recovery node in the student, on Task 4 only** (graph flags
   `{"recovery": {"seen_s": 15.0, "at_end": true}, "repush_v2": {"max_total_m": 0.1}}`). Once the plan is done and
   nothing is left on the tower, a block seen leaning on the base (R4.1: pushed >= 0.10 m, its near end raised) is
   re-pushed by the planned push, aimed at its last confident pose, at most 0.10 m, until it is seen flat clear.
   Camera loop: the 3 end-of-plan re-pushes all ended flat clear, and all 4 rule decisions were on truly leaning
   blocks. Official run 1 replayed: it would act on 5 blocks (4-2: 4, 4-3: 1), turning one 4-2 episode into a success,
   about +1.3 points if they land. Its 10-episode camera A/B cannot show that: it fired once (a success) and the rest
   differs by the camera loop's noise, which reaches ~14 pp LHPR between identical configurations in 4-3. End to end
   through its server: 0 errors.

## 1. What you need to do this morning

Both candidate endpoints are up (`student_v1_1` restarted 2026-10-28 05:23 with its final bundle, an earlier URL from
02:35 is gone; `student_v1_2` started 11:09); each passed the qualification-like probe 10 / 10 locally and through the
tunnel:

| | `official_1` (unchanged) | `official_v1_1` | `official_v1_2` (new, 11:09) |
|---|---|---|---|
| Bundle, identity | `student_v1`, `sha256:01ab88da...4bccdd` | `student_v1_1`, `sha256:41fddf48...218a` | `student_v1_2`, `sha256:f726c71a453da23943757cf9fb8506d101f83b196b22770f966073a5c3279aa9` |
| Server code | snapshot `server_v1` | snapshot `server_v1_1` | snapshot `server_v1_2` |
| Port, tmux session | 9001, `tower` | 9002, `tower_v1_1` | 9003, `tower_v1_2` |
| Policy id | `tower-student-v1` | `tower-student-v1-1` (plugin version 2) | `tower-student-v1-2` (plugin version 2) |
| Model URL | `wss://mention-sampling-prove-neo.trycloudflare.com` | `wss://epson-korea-analog-patients.trycloudflare.com` | **`wss://windsor-lighting-lean-iii.trycloudflare.com`** |
| Logs | `results/student/server/official_1/` | `results/student/server/official_v1_1/` | `results/student/server/official_v1_2/` |

On the site:
1. **Save model** with the URL of the candidate you submit (a new model entry, e.g. "student_v1_2"), then **Check
   endpoint**. The qualification sends blank images; the probe of the same check passed locally and through the tunnel
   for both.
2. **Queue evaluation:** all ten tasks, 10 Hz (Plan-v1.1 §3.3: one evaluation runs per team at a time; queue as early
   as possible). My recommendation is `student_v1_2`; `student_v1_1` is the conservative choice (the two differ only
   after the Task 4 plan, and only on a block seen leaning on the base).
3. Keep the endpoints up: `official_1` only matters if the organizers retry run 1, and the candidate you do not queue
   can be stopped. Stop one with `SESSION=tower_v1_2 results/student/stop_official.sh` (or `tower_v1_1`, `tower`),
   never `tmux kill-server`.

While it evaluates:
- `WATCH_NAME=official_v1_2 WATCH_QUIET_START=1 python3 -u scripts/official/watch_official.py <queue time ISO> 600 600`
  (Lessons-v1 §9.2), `python3 scripts/official/call_pace.py 300 official_v1_2` (or `official_v1_1`).
- Every episode's first-call record holds the orientation choice and the summary `orient`, `blind`, `vetoed`; v1.2's
  summaries also hold `recovery` (the rule's decisions and abstentions).
- Leave `results/bundles/student_v1_1` and `student_v1_2` untouched while serving (every episode reloads the bundle).
- Background GPU jobs run under `scripts/eval_guard.py --name <endpoint>`, which pauses them when a session opens.

---

## 2. `student_v1_1` in detail

### 2.1 Changes (all opt-in; the teacher, `student_v1` and the parity tests are unchanged)

| Change | Where | Flag | Why |
|---|---|---|---|
| Fixed-camera orientation per episode | `perception/orient.py`, `student/adapter.py` | the bundle's `perception/orient_refs.npz` | Tasks 1-3 were blind in official run 1 |
| Blind guard | `student/adapter.py`, `StudentRun.set_blind` | graph `blind_guard` | blind retries hurt Task 1 (ODR 0.79) |
| 16-row idle holds | `IdleNode` | graph `idle_chunk: 16` | one-row holds cost ~25 min of evaluator wall time per episode |
| Finished guard | `StudentRun._vetoed` | graph `finished_guard` | post-success rework (§0) |
| `push_settle` (4-3's graph only) | `PushNode._settle_step` | graph `push_settle` | the radial departure test fires while side-row blocks still overhang the base |

Fake-world parity 80 / 80 with the flags off; the full unit suite (290 tests) passes; new tests cover each change.

### 2.2 Validation

| Test | Result |
|---|---|
| Orientation on every official first frame (`scripts/build_orient_refs.py --check`) | 125 / 125 by correlation; right orientation >= 0.86, margin >= 0.89 |
| The v1.1 adapter on the official frames (`scripts/official/adapter_replay.py`) | blocks rendered: rotated as the task requires, all movable blocks confident, no holds; blank: three holds, then BLIND |
| Mirror calibration on official run 1 | 4-2 LHPR 73.3 / 73.3 %, 4-3 53.3 / 56.7 %; TSR 1 / 10 and 0 / 10 vs 0 and 0 |
| Finished guard, CPU, noise model, mirror, ten cases, 20 per variant | Tasks 1-3: 120 / 120 on vs 1 / 120 off (33.2 vs 23.4 points) |
| `push_settle`, CPU, simulator poses, mirror, ten cases, 24 per cell, seeds 7100 and 9100 | 21.92 vs 19.04 and 22.10 vs 20.85 points on Task 4 (§0) |
| The finished guard on Task 5 (noise model, 20 per variant) | identical on and off (25.78 points): harmless there |
| `student_v1` as submitted, noise model, mirror, ten cases, 12 per variant (seed 7700) | 68.90 points; Tasks 1-3 22.9 of 33.3 (TSR 0 in five variants: post-success rework) |
| Post-success, camera loop (our renders, unrotated fixed cameras, mirror, ten cases, 10 per variant) | **60 / 60** (33.23 of 33.3 points); orientation rotated by correlation in 60 / 60; estimates p90 0.15-0.41 mm |
| Task 4 camera loop, `push_settle` off vs on (10 per cell, same seeds) | 4-2: off TSR 6 / 10, LHPR 90.0 % (13.28 points) vs on 4 / 10, 81.7 % (11.27); 4-3: off 0, 66.7 % (7.37) vs on 0, 71.1 % (7.89). Hence 4-3 only |
| End to end through the server (`ws_loop.py --official_orient --mirror`, Tasks 1-2, 4-3, 5-3) | Task 1-2 2 / 2, Task 4-3 0 / 2, Task 5-3 1 / 2; no adapter error, hold or clipped value; orientation right per task; never blind; the finished guard vetoed 64 post-completion reworks per Task 1-2 episode |

### 2.3 What `push_settle` does not fix

4-3's bottom layer lands on the top layer's debris (pushes 6-8 end "on a block" in 10-15 of 24 episodes each): both
layers are pushed +x by the left arm. The demo generator pushes the top layer 0.25 m at up to 0.18 m/s; its blocks stand
on end, then topple outward to 316-347 mm, leaving the bottom layer's landing zone (~161 mm) clear. A planned
follow-through on the top layer made 4-3 worse: continuing the stroke drags our gripper through the rest of the top
layer (the first push failed its structure check in 8 / 8 episodes, 19-87 mm), because our pinch rides only ~0-2 mm
above the top surface at a 45 deg tilt, while the demos' tool has a 10 deg forward pitch with its pad 15 mm above the
block centre; raising the pusher 12 mm during the follow-through did not help (ODR ~0.6). Pushing slower near the edge
(x0.5) or faster (x1.5) was worse too (§3.2). Sliding a block off the
debris afterwards is out of the left arm's reach (it holds the pusher pose only to ~0.25 m out). Open: it needs a raised,
more upright pusher for the top layer, i.e. a skill change.

---

## 3. The recovery node (improvement 2)

### 3.1 What was built

- **Contracts** (`tower_rl/recovery/contracts.py`): per goal, its postcondition and the structure rule, judged on
  truth (teacher) or the rest estimates (student): Task 4 the rest kind of `tower_rl/eval/mirror.py`; Task 5 the
  insert's offset from its slot goal along / across / in height ("home", "short", "out"); Tasks 1-3 the goal
  tolerance.
- **Rules** (`tower_rl/recovery/rules.py`): R4.1 re-push a block leaning on the foundation's edge (a new `repush`
  option: the planned straight push from behind its raised end, confirmed clear); R5.1 retry an insert that failed
  while the block is still held; R5.2 push home an insert left short on the slot line with the structure intact.
  Preconditions keep the remaining structure out of reach; violations no rule covers are logged as abstentions.
- **Teacher:** `oracle_state_v2` = the state oracle + the rules on truth (`rehearse.py --student oracle_state_v2`).
  Its reopen rule for finished goals is aligned with the student's finished guard (moved > 10 mm for 2 s), so its
  labels are ones the student can see.
- **Student, rules:** graph flag `recovery` runs the same rules on the student's estimates, gated by perception health.
- **Student, distilled:** features v2 (`student/tokens.py`): the v1 tokens + per goal the contract kind and
  measurements, done / moved since / how long ago, recovery attempts; globally perception health, BLIND, all goals
  done, the structure's largest move. An 8th option (`repush`), masked by R4.1's preconditions.
- **Review sheet** (`scripts/recovery_review.py`): recovery decisions by rule and episode outcome, abstentions by cause,
  failures no rule touched, with seeds and cases to replay. This is the input for the next round of rules (the
  Claude-authored step of Plan-v1.1 §5.3).

### 3.2 What it found

**On truth (the teacher), Tasks 4-5, `push_settle`, mirror, ten cases, 20 episodes per cell
(`results/v1_1/teacher_recovery/r1`):**

| Variant | state oracle: TSR / LHPR | state oracle v2 (recovery): TSR / LHPR |
|---|---|---|
| 4-2 | 55 / 91.7 % | 55 / 90.8 % |
| 4-3 | 0 / 70.0 % | 5 / 71.7 % |
| 5-2 | 70 / 85.0 % | 70 / 85.0 % |
| 5-3 | 60 / 86.2 % | 60 / 86.2 % |
| points (four variants) | 46.81 | 47.19 |

The review sheet (`scripts/recovery_review.py`) explains why the gain is small:
- **4-3:** the failing rest states are bottom-layer blocks on top-layer debris (pushes 6-8: 29), blocks tipped onto their
  side (12) and off the table (3). None is a re-push case: a block on debris or on its side needs a different skill
  (slide off, flip), and the landing zone needs prevention (§2.3).
- **5-3:** every failure is a structure break during the bottom-layer insert (12-23 mm) or the upper push-in (10-16
  mm). The rules abstain: by the time a node notices, the structure has already moved past the official 10 mm.
- **What the first re-push rule did wrong** (now fixed): it fired on blocks lying flat against the base (their
  pre-push pose is inside the foundation: the transit timed out) and, in a heavily injected test, on a block pushed only
  38 mm with the layer above still on the tower (structure failures). R4.1 now requires the push to have ended ok, the
  block 0.10 m out, its near end >= 45 mm high and no block of a higher layer on the tower.

So, as Plan-v1.1 §5.1 expected, recovery is the backstop and the points are in prevention: the top layer's landing
zone in 4-3, tipping, and guarded insertion in Task 5. Two prevention probes on Task 4 tonight: slowing the push near
the edge made it worse (17.7 vs 21.9 points: blocks drop closer to the base); speeding it up (x1.5 from 50 mm out) left 4-2 unchanged and made 4-3 worse (tipped 29 vs 10; 20.8
points). Plain `push_settle` stays.

### 3.3 The distilled router

**How it was trained** (`results/v1_1/cpu_collect_v2.sh`, `cpu_v2_pipeline.sh`):
- **Round 0:** the state oracle v2 acts (truth decisions + the recovery rules), 10 % random exploration (never a
  recovery option), failure injection 15 % per goal option including a new "short push" (the push ends at 0.125-0.15 m
  radius, leaving the block over the base: the re-push's training states), the estimator noise model v1, the mirror
  (episodes to 210 s), the ten cases, the v1.1 flags; 120 episodes per variant (1,200), rows with features v2.
- **BC** (`bc_v2_r0`), then **DAgger round 1** (`bc_v2_r0` acting, the oracle with probability 0.5, 60 episodes per
  variant), then **BC on both rounds** (`bc_v2_r1`).
- **Comparison:** `bc_v2_r1` vs `bc_r2` (features v1), both with the v1.1 flags, noise model v1, mirror, ten cases,
  12 episodes per variant.

**Closed loop** (`results/v1_1/v2_eval`; noise model v1, mirror, ten cases, seed 7700, 12 per variant, the same seeds
for all three):

| | Tasks 1-3 | Task 4 (4-2 TSR / LHPR, 4-3 LHPR) | Task 5 (5-2, 5-3 TSR) | points |
|---|---|---|---|---|
| `student_v1` as submitted | 22.94 (TSR 0 in five variants) | 19.89 (50 / 88.9 %, 65.7 %) | 26.08 (67, 67 %) | 68.90 |
| `student_v1_1` (`bc_r2` + flags) | 32.41 | 20.02 (41.7 / 91.7 %, 68.5 %) | 26.08 (67, 67 %) | 78.50 |
| `bc_v2_r1` + the same flags | 32.46 | 19.75 (41.7 / 88.9 %, 68.5 %) | 27.42 (67, 83 %) | 79.62 |

Offline: `bc_v2_r1` validation accuracy 0.965 (nominal 0.9985, non-nominal 0.957), progress AUC 0.999 (the "is goal k
done" verifier of Plan-v1.1 §5.4, trained on the mirror's contract labels; computed but not yet used at run time). The
router holds `bc_r2`'s level; the recovery option itself is rare (68 + 30 `repush` labels), so the gain the features
can show is bounded by how few recoverable states the scene produces (§3.2). A second DAgger round (`bc_v2_r1` acting alone; `bc_v2_r2`, 2,400 episodes in all) scored 77.67 on the same seeds:
the three routers are within the noise of 12-episode cells.

**Camera loop (the gate before it could replace `bc_r2`; `results/v1_1/t4_camera`):** our renders -> kp_v2, mirror,
ten cases, seed 7400, 10 episodes per variant, student_v1_1's configuration (push_settle on 4-3 only), against the
`bc_r2` arms of the same seeds:

| Router | 4-2 TSR / LHPR / points | 4-3 LHPR / points | Task 4 points |
|---|---|---|---|
| `bc_r2` (`student_v1_1`) | 6 / 10, 90.0 %, 13.28 | 71.1 %, 7.89 | 21.17 |
| `bc_v2_r1` | 2 / 10, 83.3 %, 10.33 | 71.1 %, 7.84 | 18.18 |
| `bc_v2_r1` + `repush_v2` | 3 / 10, 78.3 %, 10.30 | 66.7 %, 7.41 | 17.71 |

- **Not passed, and the camera loop is noisy:** in 6 of the 10 4-2 episodes `bc_r2` and `bc_v2_r1` chose the same
  node sequence, and still `bc_r2` succeeded in 4 of them and `bc_v2_r1` in 2 (heavy: three blocks tipped vs none).
  The third row only changes re-push nodes, and it had one re-push in 20 episodes: its differences are noise too.
  10-episode camera A/Bs of Task 4 resolve only large effects.
- **The re-pushes did nothing.** `bc_v2_r1` chose `repush` in 3 of its 4-2 episodes. In two of them the block it chose
  was the episode's only failure (so detection worked), but every re-push ended after 4 rows without moving it: the
  re-push starts on a block already radially out, and the per-row estimate put it 3-5 mm clear while R4.1 (on the rest
  estimates) had it leaning. Fix: graph flag `repush_v2` (push 40 mm past contact first, confirm only "flat clear",
  at most 0.15 m). With it, the one re-push in the next run pushed its whole stroke, behind the wrong place: the
  estimator had coasted the block 145 mm back towards the tower (§3.4). Under `repush_v2` the re-push now acts on the
  last confident pose (unit-tested; not yet seen in the camera loop).

### 3.4 Coasted estimates end pushes early (camera loop, 2026-10-28 morning)

When the camera loses a block that was moving (e.g. behind the pusher), the estimator keeps it moving for up to 5
calls (x 0.8 per call, up to 40 mm per call), reports it with confidence 0.3, and takes the next velocity from the
jump when it finds the block again. In the episode above, the pushed block stopped at the base edge (leaning, 127 mm
from the tower axis), the estimate coasted on, the push's radial test fired on it at 165 mm (the threshold is 160),
and after a re-sighting the estimate coasted 145 mm back towards the tower.

Over all 80 camera-loop Task 4 episodes, each push that ended ok, at the row where its estimate first passed the
departure test (`scripts/radial_truth.py`):

| | estimate confident there | not confident (coasted / stale) |
|---|---|---|
| 4-2: truly departed at that row | 188 / 189 | 23 / 36 |
| 4-2: block ended "on/leaning on base" | 2 / 189 | **10 / 36** (none had truly departed: 107-157 mm out) |
| 4-3: truly departed at that row | 271 / 280 | 17 / 35 |

So 10 of the 12 4-2 blocks left leaning came from a push that stopped on a coasted pose. **Official run 1, the same
tally from the server's call logs** (`scripts/official/push_end_conf.py`; rest states from the last call's estimates,
the rule that reproduced the site's LHPR): 4-2 pushes ended on a confident estimate 42 times (36 flat clear, 86 %) and
on an unconfident one 16 times (7 flat clear, 44 %; 3 leaning on the base, 5 never seen again, 1 tipped). The learned push itself was
trained for this: its training env shows the same coasted pose with `seen` = 0 and ends on the **true** departure.
**First fix tried, graph flag `depart_seen`:** the departure test counts only rows where the target is seen; the actor pushes on
through the loss, and a target lost for 1.5 s after it was last seen >= 60 mm out still ends the push (as before).
Unit tests cover both paths; the full suite passes.

**`depart_seen` camera A/B, 4-2** (`bc_r2` = `student_v1_1` + the flag vs `student_v1_1`, same seeds; run under
`scripts/eval_guard.py`, which pauses it if your evaluation starts): TSR 3 vs 6 / 10, flat clear 50 vs 54 / 60, off the
table 3 vs 0. Per push (`scripts/depart_counterfactual.py`): 13 pushes would have ended on an unconfident row; 2 of
those blocks had not departed and now ended flat clear, 11 had (the coasted pose was right), and pushing on blind for
1.5 s shoved one of them off the table edge. Pushing on cannot tell a block that left from one that stuck, so the flag
stays off.

**Look, then push again: the recovery node in the student** (graph `recovery`, R4.1 on the rest estimates, then the
planned re-push with `repush_v2`, which now aims at the last confident pose). Its gate needed every call of the last
1.0 s confident, but a leaning block was confidently seen on 4 of ~30 idle calls, 1.6 s apart, never two in a row: the
rule could never fire after the last push (likely why it fired once in 20 4-2 episodes in the noise model). New
option `recovery: {"seen_s": 15}`: >= 2 confident sightings within 15 s, all within 4 mm of the rest estimate
(unit-tested, suite passes).

**Camera mechanism test** (4-2, 10 episodes, 24 pushes stopped short by injection so the rule has work): R4.1 fired
3 times, each time on a block truly leaning on the base, and **all three re-pushed blocks ended flat clear**. But one
of the three was made after layer 2 with layer 1 still standing: the layer-2 block had fallen against a layer-1
block's end, and the re-push came in over layer 1 at the leaning block's height and moved two layer-1 blocks
19-22 mm. New option `at_end`: R4.1 only once the plan is done (every goal pushed, nothing left on the tower, the
router idling), which is where the two clean re-pushes happened. Since the mirror's TSR needs the pushes in order, a
re-push at the end rescues the block's rest state (LHPR, 2/3 of the Task 4 score) and the TSR only when the leaning
block was the last push.

**Official run 1, replayed for this rule** (`scripts/official/repush_candidates.py`): 4-2 has four blocks it would
re-push (the three that ended leaning, all confidently seen through the idle phase, plus one seen leaning until
148 s), 4-3 one. In one 4-2 episode (06aba879) the leaning block was the last push and everything else was flat
clear: a re-push that works turns it into a success. If each re-push lands flat clear: 4-2 LHPR 73 -> 80 %, TSR 0 ->
1 / 10, about +1.3 points.

**No-injection camera A/B** (`{"recovery": {"seen_s": 15.0, "at_end": true}, "repush_v2": true}` vs `student_v1_1`'s
arms, same seeds): 4-2 TSR 5 vs 6 / 10, LHPR 85.0 vs 90.0 %; 4-3 LHPR 56.7 vs 71.1 %. The rule fired once (4-2
combined: leaning -> re-pushed 34 mm -> flat clear) and made **no** decision in 4-3, so the 4-3 arm behaved exactly as
v1.1 (checked in the code): its 14-point LHPR gap is the camera loop's run-to-run noise (four episodes cascaded from
a mid-plan structure failure, as other arms do in other episodes). 10-episode camera A/Bs of Task 4 cannot resolve a
1-point effect, so the case for v1.2 rests on the mechanism, the official replay and its construction. One missed
case led to a refinement (the switcher had skipped a push: `at_end` now also accepts a goal whose block is seen off the
tower), and the final build caps the re-push at 0.10 m (the successful ones were confirmed after 29-34 mm).
`student_v1_2` passed the bundle check (networks identical to v1.1, graphs differ only by these flags), the end to end
through its server (4 episodes, 0 errors; the rule's log is in the episode summaries) and the probes, and is served
(§1).

**What was not done:** live Claude consults during rollouts (Plan-v1.1 §5.3 step 2). The hook is the abstention log:
every contract violation no rule covers is recorded with its variant, time, goal, kind and measurements, and
`scripts/recovery_review.py` turns a run into the review sheet that the next rules are written from. A live call to the
Claude API from the rehearsal processes was left out: it sends scene data to an external service (your call), and
tonight's abstentions were all prevention cases (structure already broken, debris, blocks on their side), where an
advisor has no recovery option to choose either.

---

## 4. Next steps

1. **Submit `student_v1_2` (recommended) or `student_v1_1`** (§1). Watch the run with
   `scripts/official/watch_official.py` (Lessons-v1 §9.2) pointed at the endpoint you queued.
2. **After the evaluation:** re-score it with `scripts/official/task4_outcomes.py` / `push_end_conf.py` /
   `repush_candidates.py` (Task 4) and `task5_outcomes.py`; for v1.2 the episode summaries' `recovery` field shows each
   re-push and abstention. If v1.2 is not queued, the same build can follow v1.1's evaluation when the slot frees.
3. **Per-episode results of official run 1 from the site** (still the open human action of Plan-v1.1 §7): they pin
   Task 4's TSR condition (one 4-2 episode passes our rest rule yet failed officially).
4. **Prevention, where the points are** (the recovery node's own finding):
   - 4-3: land the top layer far out (a raised, more upright pusher for layer 3, like the demos'; or a pusher pose that
     follows the tipping block).
   - Task 5: the push-in is the failure point (student, noise model: 6 of 7 failures in 5-3 and 4 of 7 in 5-2 are a
     push-in moving the structure > 10 mm; the rest end short with the structure intact). A guard that only aborts
     earlier would trade "home but structure moved" (P still counted) for "short" (P lost); it needs a push-in that
     drags less (contact height and line, pace under friction), after which R5.2 can retry the short ones.
   - Tipping (4-2 layer 2, 4-3 everywhere): the push's contact height and speed profile near the edge.
   - Pushes ended on a coasted estimate (§3.4): v1.2 re-pushes the leaning block at the plan's end, which rescues its
     rest state but not the TSR when it was not the last push (the next block's departure breaks the order). Finishing
     it inside the push node (after an unconfident departure: retreat, look, and if a confident sighting shows it still
     at the base edge, the planned finish from that pose) would rescue both; it needs the same camera tests as R4.1
     plus care with the blocks still standing (the heavy case of §3.4).
   - A recovery "flip" for a block lying on its side (the other arm pushes its top edge across its thin dimension:
     the back arm pushes +y, the left arm +x, so each can tip the other's blocks flat): ~10 % of official pushes ended on
     their side. Worth building only once the site's per-episode data confirms that "on its side" fails officially
     (our 4-2 calibration says it does; the demo generator counted one as a success).
5. **The router (features v2):** it holds `bc_r2`'s level in the noise model and did not pass the camera check on
   Task 4 (§3.3). Before another check: `repush_v2` in its configuration and the rule node's lessons in its labels
   (re-push only at the plan's end, on sporadic but consistent sightings), a larger camera sample (20+ episodes per
   variant, or per-push tallies such as `scripts/radial_truth.py` / `recovery_events.py` instead of TSR), and more
   DAgger rounds with injections that leave recoverable states.
6. **Docs:** `Student-v1.MD` / `Teacher-v1.MD` sections on the adapter, the runtime flags and the recovery node; the
   figures (`scripts/doc_figures.py`) if the architecture diagrams should show the recovery node.
