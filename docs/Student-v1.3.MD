# Student v1.3 (`student_v1_3`): the learned router and the careful place

*Written 2026-10-29 by Claude Code (Opus 5.5), working through `Studentv1_3_plan.MD` (the analysis behind it:
`Plan-v1.3.MD`). Paths are relative to `~/Documents/tower_rl` unless absolute; every number has its run in
`BUILDLOG.md` "session 8".*

**Status (2026-10-29 20:20): through every gate and served at
`wss://norfolk-charleston-recommended-enough.trycloudflare.com`; ready to queue as the final run (§1).**
`student_v1_2`'s evaluation was cut off by the machine's 16:41 reset (§6); it stays up as the fallback at
`wss://wish-rico-sic-willing.trycloudflare.com`.

---

## 0. Summary

1. **`student_v1_3` = `student_v1_2` with a learned router in place of `bc_r2`, and one place rule.** Skills,
   perception and the orientation references are byte-identical to v1.2 (bundle check). The differences:
   - **Router `bc_v2_r3`** (features v2: the switcher's tokens carry the recovery contracts, per-goal done / moved /
     how long ago, perception health), trained on four DAgger rounds (3,600 episodes, 65,242 decisions) labelled by
     the state oracle v2 with a new teacher fix (failed goals that are now buried are closed, not reworked).
   - **`offcentre_place`** on the Tasks 1-3 graphs: a block gripped >= 4 mm off-centre along its length is placed by
     the planned place (slow descent to the offset-compensated release pose, open there) instead of the learned one.
   - **`router_repush: false`** on Task 4: the router's own re-push head is masked; v1.2's rule-based end-of-plan
     re-push stays.
   - The finished guard and the blind guard stay on as safety nets.
2. **What the router now decides by itself** (CPU, noise model, mirror, same seeds as v1.2's router):
   - **After success it idles:** 0 finished-guard vetoes in 240 episodes, against 8,759 for v1.2's router (which
     only stays idle because the guard stops it). Officially, v1.2 needed the guard 63-83 times per episode.
   - **It leaves buried failed goals alone:** 0 buried reworks (v1.2's router: 1 on the irregular-layer starts, as in
     official 1-3 588e8a2a).
   - **It pushes a Task 5-3 block home again when its push-in ended short:** v1.2's router parks and gives up; v1.3
     wins those episodes (3 of 48 on CPU).
   - It re-placed a failed place at once once where v1.2's router did not (irregular-layer starts).
3. **Off-centre grips:** officially, places after grips >= 4 mm off-centre failed 7 of 12 (others 1 of 110): the
   learned place let go 4-8 cm above the goal or the block slid. On CPU, forcing 4-8 mm grips cut Tasks 1-3 TSR from
   97 % to 81 %; with `offcentre_place` 97 % again (105 of 106 off-centre places ok, release ~5 mm above the goal vs
   ~11 mm). Camera loop with forced off-centre grips: 12 / 12 episodes, 37 / 37 off-centre places ok.
4. **CPU gates** (24 episodes per variant, ten cases, the official mirror): **78.88 vs v1.2's 78.19 points**; no Tasks
   1-3 variant lost an episode; Tasks 4-5 at parity over 48 episodes per variant (+0.16).
5. **Camera checks and end to end through the server: passed** (§3.4, §3.5).
   - Tasks 1-3: 59 / 60.
   - Task 4, per pushed block: better on 4-2 on both seeds (56 vs 51 and 55 vs 51 of 60 ended flat and clear), level
     on 4-3 (55 / 90 each).
   - Task 5: within one success per variant (5-2 8 vs 7, 5-3 2 vs 3), with the same decisions.
   - Through the server: 0 errors, holds, clipped values or vetoes in 8 episodes; call p99 at most 1.34 s.
   - Every family keeps the learned router; the 09:59 freeze is final.
6. **Expected official effect:** small in points (about +0.5 to +1.5, mostly Tasks 1-3 goal-4 places and Task 5-3);
   the value is that decisions moved from hand-written guards into the learned router, and a second official sample.

---

## 1. What you need to do

**Model URL: `wss://norfolk-charleston-recommended-enough.trycloudflare.com`**

| | |
|---|---|
| Policy | `tower-student-v1-3` |
| Artifact identity | `sha256:76ec7f3d35c4da7b41dc74c26ce2f8484287f87cc4023f3820dc4145a0c8263d` |
| Server | port 9004, tmux `tower_v1_3` (snapshot `tower_rl_snapshots/server_v1_3`) |
| Probes | 10 / 10 locally and 10 / 10 through the tunnel (20:17-20:19) |

1. **Before you queue** (the resets came at light load, §6):
   - Disable the deepest CPU idle states until the next reboot:
     `for f in /sys/devices/system/cpu/cpu*/cpuidle/state{3,4}/disable; do echo 1 | sudo tee $f >/dev/null; done`
   - Close Chrome for the evaluation's duration; use Firefox for the challenge site.
2. On the challenge site: **Save model** with the URL above, **Check endpoint**, then **Queue**: all ten tasks,
   10 Hz.
3. During the run: keep VS Code and this terminal open (the endpoint runs in their session), don't reboot, and don't
   restart the tunnel (a new tunnel gets a new URL).
4. If the machine resets again, re-serve from `~/Documents/tower_rl`. Move the old logs aside first: the serve
   script's READY and URL checks would otherwise read the stale ones.
   ```
   A=results/student/server/official_v1_3/attempt1_crash; mkdir -p $A
   mv results/student/server/official_v1_3/{run.json,server.jsonl} results/student/logs/official_v1_3_{server,tunnel}.log $A/
   SW=results/student/models/bc_v2_r3/switcher.pt OC=1 results/v1_3/build_v1_3.sh
   ```
   It prints a new URL, which needs a new Save model. Restart the telemetry logger
   (`setsid python3 scripts/sys_telemetry.py &`); the CSV it wrote keeps the machine's last seconds.
5. The fallback: `student_v1_2` at `wss://wish-rico-sic-willing.trycloudflare.com` (port 9003, tmux `tower_v1_2`).

---

## 2. What changed

| Change | Where | Flag / file | Why |
|---|---|---|---|
| Router `bc_v2_r3` (features v2) | `student/switcher.pt` in the bundle | `results/student/models/bc_v2_r3` | post-success idling, buried goals, re-place / push home after failures, learned |
| Teacher: close buried failed goals | `student/oracle.py` (`_decide_state`, `_end_state`) | graph `oracle_skip_buried` (labels only) | official 588e8a2a: the router reworked a goal under two layers |
| Real off-centre grips as an injection | `student/inject.py`, `LearnedGraspNode` | `acquire: {grasp_along_m, grasp_along_frac}`, `goals: [...]` | DAgger states after off-centre grips; Steps 2-3 |
| Evaluator-edge starts | `configs/tiers.yaml` | tier `D1e` = D1 + `irregular_edge` | official 2-2 / 2-3 starts with layers 15-22 mm apart |
| Careful place after off-centre grips | `graph/executor.py::_make_node` | graph `offcentre_place: {"min_mm": 4.0}` (Tasks 1-3) | §0 item 3 |
| Router re-push masked on Task 4 | `student/tokens.py::_recovery_ok` | graph `router_repush: false` | the v2 re-push labels predate the 10-28 lessons; v1.2's rule node stays |
| Release logging | both place nodes | info `grip_along_mm`, `release_above_mm` | official release heights in the server summaries |
| Per-variant routers (insurance) | `student/switcher.py::VariantDeciders`, adapter, server | graph `"switcher"` key | lets a family keep `bc_r2` if its camera check fails |
| CPU jobs beside the endpoint | `scripts/eval_guard.py` | `--no-pause` | own systemd scope, nice 10, OOM score 1000 |

All opt-in: with the flags off, the full unit suite passes (309 tests; 304 before) and the fake-world parity is
80 / 80.

---

## 3. Evidence

### 3.1 Off-centre grips (Steps 2-3)

CPU, `bc_r2` + v1.1 flags, noise model, mirror, ten cases, Tasks 1-3 / 1-2 / 2-3 x 12, the same seeds per arm
(`results/v1_3/grip_repro`, `scripts/grip_report.py`):

| Arm | TSR | LHPR | places after 4-6 / 6-8 / >= 8 mm grips ok | release above goal (p50) | points (3 variants) |
|---|---|---|---|---|---|
| base | 97.2 % | 100 % | 6 / 6, -, 1 / 1 | 10.8 mm | 16.45 |
| every acquire gripped 4-8 mm off | 80.6 % | 98.1 % | 42 / 42, 40 / 43, 9 / 12 | 10.4-12.5 mm | 15.32 |
| same + `offcentre_place` | 97.2 % | 99.8 % | 50 / 51, 37 / 37, 18 / 18 | 4.9 mm | 16.44 |
| `offcentre_place`, no forced grips | 100 % | 100 % | 8 / 8 | 4.9 mm (planned) | 16.62 |

The CPU loop reproduces the failures (slides and misplacements) but not the official 4-8 cm early releases, which
need the official physics or renders; the flag removes both kinds by construction (it opens only at the release pose).

### 3.2 The router (Steps 4-5)

- DAgger round 3: `bc_v2_r2` acting with beta 0.5, labels `oracle_state_v2` + `oracle_skip_buried`, tier D1e,
  injections 0.15 (short pushes; a third of acquire injections an off-centre grip), 1,200 episodes, 20,051 decisions.
- `bc_v2_r3` on rounds 0-3: validation decision accuracy 0.968 (gate 0.965), non-nominal 0.955 (0.95), progress AUC
  0.9989 (0.995).

### 3.3 CPU gates (Step 6)

REF = v1.2 (`bc_r2` + v1.2's flags), CAND = v1.3; noise model v1, mirror (to 210 s), ten cases, seeds 7700 / 7900 x
12 per variant, the same seeds per arm (`results/v1_3/gates/r3`):

| Variant | REF succ / LHPR / vetoes / points | CAND succ / LHPR / vetoes / points | Diff |
|---|---|---|---|
| 1-2 | 24/24, 100.0, 1460, 5.54 | 24/24, 100.0, 0, 5.54 | +0.00 |
| 1-3 | 20/24, 97.2, 0, 5.12 | 21/24, 99.0, 0, 5.27 | +0.14 |
| 2-2 | 24/24, 100.0, 1887, 5.54 | 24/24, 100.0, 0, 5.54 | +0.00 |
| 2-3 | 21/24, 97.9, 1554, 5.23 | 23/24, 99.3, 0, 5.43 | +0.21 |
| 3-2 | 24/24, 100.0, 2040, 5.54 | 24/24, 100.0, 0, 5.54 | +0.00 |
| 3-3 | 24/24, 100.0, 1818, 5.54 | 24/24, 100.0, 0, 5.54 | +0.00 |
| 4-2 | 10/24, 86.1, 0, 11.84 | 10/24, 86.1, 0, 11.84 | +0.00 |
| 4-3 | 0/24, 68.5, 0, 7.57 | 0/24, 70.4, 0, 7.80 | +0.23 |
| 5-2 | 16/24, 83.3, 0, 12.89 | 15/24, 81.2, 0, 12.47 | -0.42 |
| 5-3 | 16/24, 88.5, 0, 13.39 | 18/24, 89.6, 0, 13.92 | +0.53 |
| **Total** | **78.19** | **78.88** | **+0.69** |

- **Tasks 4-5 over 48 episodes per variant** (two more seeds): 4-2 +0.19, 4-3 -0.21, 5-2 -0.21, 5-3 +0.39 (together
  +0.16). The routers decide alike there: 4-2 45 / 48 and 5-2 47 / 48 episodes have identical node sequences; in 4-3
  the episodes diverge at the same push with another outcome (physics and noise draws), not at a decision. The real
  difference is 5-3's push home after a short push-in (§0 item 2).
- **Injected off-centre grips** (Tasks 1-3, p 0.3, without `offcentre_place`): REF and CAND identical outcomes (69 /
  72); REF needed 4,209 vetoes for it, CAND 0.
- **Evaluator-edge starts** (2-2 / 2-3 x 12, tier D1e): both 20 / 24 (10.12 vs 10.13 points); REF 1 buried rework,
  CAND 0; failed places redone at once REF 0, CAND 1.

### 3.4 Camera checks (Step 7)

Our Isaac renders -> kp_v2 -> the student, the mirror, ten cases, 10 episodes per variant; references on the same
seeds.

| Family | Candidate | Reference (v1.1 / v1.2 router) | Gate | Result |
|---|---|---|---|---|
| Tasks 1-3 (seed 7200, `--official_orient`) | 59 / 60, 32.87 points | `post_success/core_guard` 60 / 60, 33.23 | >= 58 / 60; 0 releases > 25 mm; 0 vetoes; off-centre places ok | **passed** (0 vetoes, 0 releases > 25 mm, 2 / 2 off-centre places ok, 0 buried reworks) |
| Task 4-2 (seed 7400) | 8 / 10, LHPR 91.7, 14.60; pushed blocks flat clear 55 / 60 | `t4_camera/rec_end` 5 / 10, 85.0, 12.22; 51 / 60 | per push not worse | **passed** |
| Task 4 (seed 7500) | 4-2 5 / 10, 93.3, 13.06, 56 / 60; 4-3 0 / 10, 61.1, 6.73, 55 / 90 | `camera/ref/t4ref_7500` 4-2 4 / 10, 85.0, 11.60, 51 / 60; 4-3 0 / 10, 60.0, 6.68, 55 / 90 | per push not worse | **passed** |
| Task 5 (seed 7400) | 5-2 8 / 10 (14.34), 5-3 2 / 10 (9.00): 23.34 | `camera/ref/t5_ref` 5-2 7 / 10, 5-3 3 / 10 (23.88) | not worse by > 1 success per variant | **passed** |

The Tasks 1-3 failure (2-2, high friction): goal 0's learned place (a 2.1 mm grip, so v1.2's skill) opened 14.4 mm
above the goal and the high-friction fingers flung the block ~100 mm sideways out of view; nothing visible was left to
recover it with. The decisions up to that release were the reference's; the camera loop is not deterministic, so this
is a skill event that can happen to v1.2 as well.

- **Task 4:** the candidate's 4-2 pushes ended flat and clear more often on both seeds; 4-3 is level (the
  candidate tipped more, 18 vs 12, and left fewer on a block, 10 vs 18). On 4-2, seed 7500, the rule re-pushed twice;
  both blocks really were leaning, and both ended flat and clear.
- **Task 4-3 on seed 7400 never finished:** the renderer hung twice (GPU idle, the main thread spinning in native
  code), at different points (~5 and ~12 minutes in). The server never renders, so this has no bearing on the
  submission; seed 7500 stands in for it.
- **Task 5:** the two routers made the same decisions up to every failure. The one difference in 5-3 (uneven
  weights): the candidate's last push-in ended 4.0 mm short and failed, the reference's 2.3 mm and succeeded, with
  byte-identical skills, so camera-loop noise. On camera neither router pushed a short insert home (both park after a
  failed push-in); on CPU the candidate did, in 3 of 48 episodes.

### 3.5 End to end through the server (Step 8)

Our Isaac renders -> `ws_loop` -> `serve_student.py` (the `server_v1_3` snapshot, the frozen bundle, port 9005), the
mirror, seed 7600, nominal and combined per variant (`results/v1_3/ws_e2e`, server logs
`results/student/server/e2e_v1_3`):

| Variant | Successes | Calls per episode | Call p99 (client) | Errors / holds / clipped / vetoes |
|---|---|---|---|---|
| 1-3 | 2 / 2 | 462, 444 | 426 ms | 0 / 0 / 0 / 0 |
| 2-2 | 2 / 2 | 287, 287 | 346 ms | 0 / 0 / 0 / 0 |
| 4-2 | 0 / 2 (GSR 0.75) | 338, 342 | 1,335 ms | 0 / 0 / 0 / 0 |
| 5-3 | 0 / 2 (GSR 1.00, PSR 0.25) | 567, 425 | 324 ms | 0 / 0 / 0 / 0 |

- **Passed:** no errors, holds, clipped values or vetoes, not blind, and every p99 under 1.5 s. 4-2's p99 matches
  v1.2's own end-to-end run (1,248 ms): our renderer shares the GPU with the server.
- The summaries carry the new fields: `release_above_mm` on every place (5.0-15.4 mm), `vetoed` / `vetoes`, and Task
  4's `recovery` record. Officially the server is the only GPU job; v1.2's run averaged ~94 ms per call.

---

## 4. What is still hand-written in v1.3

Changes against `Plan-v1.3.MD` §1:
- **Finished guard:** now inert (0 vetoes on CPU); kept as a net. Its official count is the evidence for removing it.
- **Buried-goal rework, re-place, push-home-after-short:** decided by the learned router.
- **Off-centre place:** a new hand-written switch (>= 4 mm -> the planned place). The learned fix is a place
  fine-tuned on such grips, or a grasp that centres better (after the deadline).
- **Task 4 re-push:** still v1.2's rule (R4.1 + planned re-push); the learned head is masked.
- Unchanged: the success verdicts, `push_settle`, the planned transits and parks, the tracker's coasting.

---

## 5. After the deadline

As in `Plan-v1.3.MD` §6: Task 4-3 and Tasks 4-2 / 5-2 skills on the GPU (where most missing points are), learned
termination for the push, the router's progress head as the done judge, a place fine-tuned on off-centre grips, then
the learned transit residual.

---

## 6. The machine resets

The machine has reset itself four times without warning:

| When | What was running | Back up |
|---|---|---|
| 09-07 ~12:35 | Chrome GPU errors at 12:31 | 12:49 |
| 10-18 ~14:36 | idle | 14:37 |
| 10-29 00:10:55 | v1.2's first evaluation, 4 h in | 00:11 |
| 10-29 16:41:15 | v1.2's second evaluation, 35 min in; load ~2 | 16:42 |

- **No trace in any log:** no kernel panic (and panic is set to hang, not reboot), no machine-check or PCIe errors, no
  thermal or power events. Temperatures are normal, and nothing changed in software.
- **None came under sustained heavy load:** the machine ran 39 days straight from 09-07, and today's full-CPU and
  full-GPU jobs ran fine.
- **Suspected cause:** a hardware-level reset at light load. That points at deep CPU idle-state transitions on an
  old BIOS and microcode (ASUS PRIME Z690-A BIOS 2004 of 2022-08, i9-12900KS, microcode 0x2c), or the PSU's
  behaviour at low load.
- **Mitigations for the evaluation:** C8 / C10 disabled, Chrome closed (its GPU process hung the Intel graphics
  twice today).
- **Logging:** `scripts/sys_telemetry.py` writes an fsynced 1 Hz CSV (`results/v1_3/logs/sys_telemetry_*.csv`), so
  a next reset leaves its last seconds on disk.
- **After the deadline:** BIOS update, check MultiCore Enhancement, memtest, PSU check.
