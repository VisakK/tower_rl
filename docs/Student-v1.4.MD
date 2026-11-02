# Student v1.4 (`student_v1_4`): v1.3 with the learned reachers

*Written 2026-11-01 by Claude Code (Opus 5.5), on the request "build student_v1_4". Paths are relative to
`~/Documents/tower_rl` unless absolute; the raw notes are in `BUILDLOG.md` "2026-11-01 (session 11)".*

**Status: built, checked, served, never evaluated.** It was served on 2026-11-01 at 12:37, on your request "lets get
student v1.4 evaluated" (`wss://bodies-vat-document-bottle.trycloudflare.com`, port 9005, tmux `tower_v1_4`). The
site's endpoint check reached it at 13:18 with 0 errors. By then the organizers had closed submissions (deadline
Oct 31), so no official evaluation followed. `student_v1_3`, evaluated on 10-31, stays the final submission. It was
not touched. On your request, both servers (v1.4 and v1.3) and their tunnels were stopped on 11-01 at 20:00, along
with the GPU keeper and the telemetry logger.

---

## 0. Summary

1. **`student_v1_4` = `student_v1_3` + the chunk-trained reachers of `Reacher-V3.MD`.**
   - Unchanged from v1.3, byte for byte (bundle check): the router `bc_v2_r3`, every skill, kp_v2, the orientation
     references, the task specs, the configs and v1.3's graph flags.
   - New in the bundle: `reach/reach_left_p2fck9700.{pt,json}` and `reach/reach_back_p2fck8500.{pt,json}`. Every graph
     also gets the `reach` block of `configs/reach/teacher_p2f.json`, with the model paths pointing inside the bundle.
   - What the reachers drive:
     - Tasks 1-3: every transit, empty and with a block in the fingers (carries at 0.05 m/s left, 0.15 m/s back).
     - Task 5: the pre-grasp and pre-insert transits.
     - Pre-push (all of Task 4) and pre-push-in stay planned, so Task 4 runs exactly as in v1.3.
2. **The bundle behaves exactly like the configuration it freezes.** On CPU with exact replay, the student on v1.4
   and on v1.3 + `--graph_set @configs/reach/teacher_p2f.json` produced identical episode records on 5 variants
   (outcomes, every node event time), apart from wall-clock timing fields. So every session-10 result for that
   configuration carries over to the bundle. On Task 4, v1.4 is identical to v1.3 alone. A negative control (v1.3
   alone on Task 1-3) differs, as it should (§3.2). With camera perception, which is not bit-exact, the bundle's
   Tasks 1-3 loop reproduced session 10's run of that configuration episode for episode: 59 / 60, 33.02 points (§3.4).
3. **The server path works.** The e2e test ran the v1.4 server and the client on 4 variants: 0 errors in 8 episodes,
   with the same successes as v1.3's e2e test on the same seeds wherever the reacher is used. Transit calls fall by
   38-48 % on the variants it drives. Mean policy time per call is unchanged within ~1.5 ms (§3.3).
4. **What it buys:** transits in fewer calls and Task 1-3 plans that finish sooner. It has not improved points so far:
   the Tasks 1-3 camera loop scored 59 / 60 both with and without the reachers (33.02 vs 32.87 points), and the Task 5
   camera checks were equal (`Reacher-V3.MD` §3.4).

---

## 1. Using it

- **Rehearse** (no `--graph_set` needed):
  `rehearse.py --bundle results/bundles/student_v1_4 --student results/bundles/student_v1_4/student/switcher.pt ...`
- **Serve locally** (as the e2e test does): from `~/Documents/tower_rl_snapshots/server_v1_4`,
  `python scripts/serve_student.py --track data/challenge/track.json --bundle <abs path>/results/bundles/student_v1_4
  --name <name> --port <port> --policy_id tower-student-v1-4 --plugin_version 2`.
- **Serve officially:** done 2026-11-01 12:37 with `results/v1_4/serve_v1_4.sh`. It runs
  `PORT=9005 SESSION=tower_v1_4 SNAP=~/Documents/tower_rl_snapshots/server_v1_4
  BUNDLE=/home/visakii/Documents/tower_rl/results/bundles/student_v1_4 EXTRA="--policy_id tower-student-v1-4
  --plugin_version 2" results/student/serve_official.sh official_v1_4`, then probes locally and through the tunnel as
  `results/v1_3/build_v1_3.sh` does (`results/v1_4/official_v1_4_probe_{local,tunnel}.json`: ok).
  - Model URL: `wss://bodies-vat-document-bottle.trycloudflare.com`. On the site: a new model (name
    `tower-student-v1-4`, this URL), "Save model", "Check endpoint", "Queue evaluation".
  - Stop: `SESSION=tower_v1_4 results/student/stop_official.sh`.
  - After a reset or a restart the quick tunnel's URL changes. Move official_v1_4's `run.json`, `server.jsonl` and logs
    aside (a stale READY / URL), run `serve_v1_4.sh` again, and update the URL on the site.
  - The v1.3 endpoint (port 9004, tmux `tower_v1_3`) is still up.
- **Identity:** tar sha256 `78d27952a2b9e9caf74010ca6126304109c3b44a29d71989e295716533b15a60`
  (`results/bundles/student_v1_4.tar.sha256`, the server's `artifact_identity`).
- **Code:** the snapshot `~/Documents/tower_rl_snapshots/server_v1_4` (tower_rl/ identical to the repo at the
  build). The manifest's git commit (f5e8cce, dirty) predates sessions 8-11, which are uncommitted.

---

## 2. What was built

- **Code: model paths that resolve inside a bundle.** `ExecCfg.bundle` is set by `Executor.from_bundle`, and the
  student runtime (the server's executor) inherits it with the cfg. `executor.reach_policy()` now loads a relative
  `reach.policy.<arm>` path from the bundle when the bundle has that file. Otherwise the path is relative to the
  working directory, so `--graph_set @configs/reach/*.json` works on any bundle as before.
  - Test: `tests/test_reach_policy.py::test_reach_policy_bundle_paths`. Suite: 314 passed, 3 skipped (Isaac).
- **Code since `server_v1_3`** (10-29): in the runtime only `tower_rl/graph/executor.py` changed: the transit dispatch
  behind `reach.mode: learned`, and the new `REACH_DEFAULTS`, `reach_policy` and `LearnedTransitNode`. There are also
  the new `tower_rl/skills/reach_{env,policy,tasks}.py` and `configs/reach/`, `configs/skills/reach.yaml`. A graph
  without a `reach` block runs exactly v1.3's code paths.
- **Scripts** (`results/v1_4/`):
  - `build_v1_4.sh`: freeze, bundle check against v1.3, load check from outside the repository, snapshot.
  - `gpu_checks_v1_4.sh`: the GPU queue that runs the three checks below.
  - `gpu_ws_e2e_v1_4.sh`: server e2e test.
  - `cpu_parity_v1_4.sh`: v1.4 vs v1.3 + teacher_p2f.
  - `cpu_t4_same_v1_4.sh`: Task 4, v1.4 vs v1.3.
  - `cpu_negctl_v1_4.sh`: the negative control (v1.3 alone on Task 1-3 must differ from v1.4).
  - `gpu_cam_v1_4.sh`: Tasks 1-3 camera loop from the bundle.
  - `compare_records.py`: episode records compared up to wall-clock timing. The two CPU scripts' own final lines
    compare whole records, timing included, so they print FAILED; use this instead.
  - `compare_camera.py`: two camera-loop runs compared episode by episode (outcome, per-goal results, events, and the
    router's node sequence, which shows retries).

---

## 3. Evidence

### 3.1 Bundle check (`results/v1_4/logs/build_v1_4.log`)

- 63 files byte-identical to `student_v1_3`: switcher, skills, perception, references, specs, configs.
- The 10 graphs are v1.3's plus exactly the reach block.
- The 4 new files are identical to the exports in `results/reach/export/`.
- MANIFEST verified, and both reachers load from the bundle with the working directory set to `/tmp`.
- 77 files, 106 MB.

### 3.2 CPU parity (`results/v1_4/parity`)

The student (switcher, oracle perception: heatmaps from the true state) ran on seed 7700, D1 nominal + heavy, 2
episodes per variant, with the official mirror (to 210 s).

| Variant | v1.4 vs v1.3 + teacher_p2f | v1.4 vs v1.3 alone | Successes (v1.4) |
|---|---|---|---|
| Task 1-3 | identical | **differs** (negative control) | 2 / 2 (v1.3 alone: 1 / 2) |
| Task 2-3 | identical | - | 2 / 2 |
| Task 3-2 | identical | - | 2 / 2 |
| Task 4-2 | - | identical | 0 / 2 |
| Task 4-3 | identical | identical | 0 / 2 |
| Task 5-2 | identical | - | 1 / 2 |

- "Identical" means equal episode records (outcomes, per-goal grasp / place results, disturbances, every node event
  time, recovery decisions). Only the wall-clock fields `act_ms_mean`, `act_ms_max` and `wall_s` are skipped.
  Results: `results/v1_4/parity/compare_up_to_timing.txt` and `compare_t4_v13_alone.txt`.
- **Task 4 is v1.3's.** All its transits are pre-pushes, which stay planned.
- **The negative control shows the test sees the reacher.** v1.3 alone on Task 1-3 differs from v1.4: v1.4 places the
  last block at 115.4 / 115.6 s, v1.3 alone at 127.4 / 129.9 s. One v1.3-alone episode failed (2 episodes: noise
  level).

### 3.3 End to end through the server (`results/v1_4/ws_e2e`, server logs `results/student/server/e2e_v1_4`)

The v1.4 server ran on port 9005 (camera_source render) with the client `ws_loop.py`. It used the same variants,
cases (D1 nominal + combined) and seed (7600) as v1.3's e2e test (`results/v1_3/ws_e2e`), with the official mirror.

| Variant | v1.4 TSR / GSR / PSR | v1.3 TSR / GSR / PSR | Transit calls per episode (v1.4 / v1.3) | Policy ms per call, mean (v1.4 / v1.3) |
|---|---|---|---|---|
| Task 1-3 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 71, 73 / 117, 116 | 20.8, 20.4 / 19.0, 19.0 |
| Task 2-2 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 21, 22 / 40, 40 | 12.1, 12.0 / 12.2, 11.9 |
| Task 4-2 | 0.5 / 0.83 / 0.83 | 0.0 / 0.75 / 0.75 | 39, 45 / 45, 39 (planned in both) | 7.8, 8.2 / 7.7, 7.6 |
| Task 5-3 | 0.0 / 1.0 / 0.25 | 0.0 / 1.0 / 0.25 | 32, 26 / 53, 47 | 10.1, 10.3 / 10.3, 11.3 |

- **0 errors** in the 8 server episodes: 6 ran to the 210 s limit, and 2 (5-3) were left open by the client at the
  end, as in v1.3's run.
- A reacher transit call costs ~75-80 ms of policy time vs ~45-50 ms planned (16 reacher rows, ~1.7 ms each). There
  are fewer of them, so the mean per call barely moves.
- Server total per call: mean 104-114 ms vs 102-110 ms; p99 216-376 vs 182-335 ms. Most of the difference is
  perception (`est_ms`), which the reacher does not touch.
- Task 4-2 differs from v1.3 (1 / 2 vs 0 / 2) although every Task 4 transit is planned in both. That difference is
  the camera loop's run-to-run noise; the exact check is the CPU one (§3.2).

### 3.4 Camera loop from the bundle (`results/v1_4/camera`)

The student with camera perception (kp_v2) ran Tasks 1-3 from the bundle: seed 7200 x 10 per variant, the ten D1
cases, `--official_orient`, the mirror. The reference is session 10's run of the same configuration, `p2f_t13`
(v1.3 + teacher_p2f, `results/reach/student/p2f_t13`).

| | v1.4 bundle | p2f_t13 (v1.3 + teacher_p2f) | v1.3 alone |
|---|---|---|---|
| Successes | 59 / 60 | 59 / 60 | 59 / 60 |
| Points (mirror, Tasks 1-3) | 33.02 | 33.02 | 32.87 |

- **The same per variant:** Task 1-3 9 / 10 (TSR 0.9, PSR 0.983, LHPR 0.992; the lightweight case fails in both
  runs), the other five variants 10 / 10 (`camera/t13_points.md`).
- **The same per episode:** all 60 episodes have the same outcome, the same per-goal grasp / place results and the
  same order of grasp / place events (`results/v1_4/compare_camera.py`, output `camera/compare_p2f_t13.txt`). Event
  times differ by a median of 0.0-0.1 s per variant, because the camera path is not bit-exact from run to run.
- **Router decisions:** the node sequence is identical in 58 / 60 episodes. The two that differ go one each way:
  - Task 3-2 heavy: p2f_t13's first place failed, so it re-grasped and placed again, 8.9 s later. v1.4 placed first
    time.
  - Task 1-3 lightweight: the first place misses in both runs. Its node reported "ok" in p2f_t13 and "failed" in
    v1.4, and both runs went on to the next block. The episode fails in both.
  - The grasp / place events record only successes, which is why these retries show up only in the node sequence.
- `mirror_points`' column "goal nodes after success" reads 8 vs 9 (Task 1-2) and 4 vs 1 (Task 1-3). That counts only
  the final place node, logged just before or just after the success instant. Neither run started a goal node after
  success.

---

## 4. Uncommitted changes (session 11, for your commit in `~/Documents/tower_rl`)

- `tower_rl/graph/executor.py`: `ExecCfg.bundle`, `from_bundle` sets it, `reach_policy` bundle paths.
- `tests/test_reach_policy.py`: `test_reach_policy_bundle_paths`.
- `BUILDLOG.md`: the session 11 entry.
- `results/v1_4/*`: the scripts above.
- Sessions 8-10's reacher work is uncommitted too (`Reacher-V3.MD` §5).

## 5. Open questions

1. Serve v1.4 (public tunnel) for any evaluation, or keep it local? (Answered 11-01: served for an official
   evaluation, §1.)
2. Commit the reacher work (sessions 8-11)?
3. Back reacher 9200 vs 8500 (`Reacher-V3.MD` §7.4): a v1.5 would only need a re-freeze. (Answered 11-01: "lets do
   back reacher". `student_v1_5` is built and checked on CPU, and the teacher A/B found no difference: §6.)
4. Shelve v1.5 (keep the bundle, skip its ~6 h of GPU checks and don't serve it), or run the GPU checks anyway at a
   time you are away from the screen? The recommendation is to shelve it (§6).

## 6. `student_v1_5`: v1.4 with back reacher 9200 (built 11-01; not served)

- **What it is:** `results/bundles/student_v1_5` = v1.4 with `reach/reach_back_p2fck9200` in place of 8500. Config
  `configs/reach/teacher_p2f_back9200.json`, built by `results/v1_5/build_v1_5.sh`, tar sha256 6a0406e1....
  - Bundle check against v1.4: only the back export and `reach.policy.back` in the ten graphs differ. The other 65
    files are identical.
  - The code is the `server_v1_4` snapshot.
- **CPU parity (`results/v1_5/parity`):** v1.5 behaves exactly like v1.3 + that config on 1-3, 2-3, 3-2 and 5-2, up to
  timing. As a negative control, it differs from v1.4 everywhere the back arm uses the reacher (5-2 is identical: its
  back-arm transits are planned).
- **Teacher A/B, 8500 vs 9200 (`results/v1_5/ab_back/summary.txt`):**
  - Setup: privileged state, CPU, left reacher 9700 in both. Tasks 1-3 and 5, D0 + D1, 8 episodes per tier, four
    seeds (9600, 10000, 10200, 10400). That is 512 episodes per config, paired by episode seed.
  - **Results:**

    | | 8500 (v1.4) | 9200 (v1.5) |
    |---|---|---|
    | Task successes | 493 / 512 | 491 / 512 |
    | Places | 1448 / 1472 | 1445 / 1472 |
    | Mirror points, 8 variants (max 70) | 63.65 | 63.42 |
    | Mirror points, D1 only | 60.85 | 60.55 |

  - 503 of 512 pairs had identical outcomes. Only 8500 succeeded in 5 pairs and only 9200 in 3 (exact McNemar
    p = 0.73).
  - Task 2-3 was 62 vs 60 / 64, the only variant that differs by more than one episode.
    - Its usual loss, the left arm's second place, happens with both models (8500 twice, 9200 three times).
    - 9200 also lost the back arm's last place twice (once here, once in parity). The node reported ok at
      4.1 mm, but the place never counted. 8500 never did this. Two cases are too few to call it an effect.
  - Task 5: 5-2 was identical (57 / 57, the same episodes lost); 5-3 was 55 vs 54.
  - **Why no gain:** in the tasks, the back arm's arrival errors are the same with either model (p50 3.4 vs 3.3 mm,
    places 1.8 vs 1.7 mm). The IK finish and the graph's arrival tolerance set them, so 9200's better battery accuracy
    (0.6 vs 0.9 mm) does not reach the score.
- **GPU checks:** only Task 5-2 on camera (seed 7400) ran: 7 / 10, against 8 / 10 for 8500 (noise level). The rest
  (5-3, Tasks 1-3, the server test) were stopped at 13:18 because the rendering froze your screen.
- **Recommendation:** keep v1.4 as the reacher version and shelve v1.5. The A/B shows nothing that would pay for ~6 h
  of desktop lag. (Submissions closed after the Oct 31 deadline, so neither version will be evaluated.)
