# Review guide

For the TOWER challenge reviewers: what was evaluated, where the exact code is, how to serve each submission, and how this
repository differs from what ran. The technical report, the checkpoints, the evaluation logs and the agent records were
shared with the reviewers privately. Scores are on the challenge site.

## Evaluated submissions

All three ran on the 10 Hz track (`tower-sim-all`, version `20261013-v1-task45-10hz-reconnect`). `student_v1` and `student_v1_3` ran all
ten variants. `student_v1_2`'s attempts were cut off first: attempt 1 ran 1-2, 1-3, 2-2 and 2-3; attempt 2 ran 1-2, and its sessions on 1-3 stayed empty. Dates are
local to the serving machine (UTC-7).

| Model | Policy ID | Plugin | Evaluated code | Evaluation |
|---|---|---|---|---|
| `student_v1` | `tower-student-v1` | 1 | tag `eval/student_v1` | Completed 2026-10-27 |
| `student_v1_2` | `tower-student-v1-2` | 2 | tag `eval/student_v1_2` | Two attempts, both cut off by resets of the serving machine |
| `student_v1_3` | `tower-student-v1-3` | 2 | tag `eval/student_v1_3` | One attempt cut off by a reset; completed 2026-10-31 (the final submission) |

| Model | Artifact identity: sha256 of the bundle tar, advertised by the server | Tag commit |
|---|---|---|
| `student_v1` | `01ab88da17d7f9003bde56ff4db1c71c885298598994eb3dd0f6f031fd4bccdd` | `862244da66ed864c26bb78fc50ac487adbd3d0f6` |
| `student_v1_2` | `f726c71a453da23943757cf9fb8506d101f83b196b22770f966073a5c3279aa9` | `5db8b5dc90b74af086a5750e89b1e29a642174c5` |
| `student_v1_3` | `76ec7f3d35c4da7b41dc74c26ce2f8484287f87cc4023f3820dc4145a0c8263d` | `79203a697ef532edf5079da643ea6e1b35b20c6d` |

Not evaluated: `student_v1_1` was served and never queued. `student_v1_4` and `student_v1_5` were built after the deadline
(§6, item 9).

## 1. Verify the checkpoints and the code

- **Tars:** `sha256sum` must print the identity above. Your logs should show the same identity, because the server sends it in
  its metadata.
- **Files in a bundle:** each `MANIFEST.json` lists every file's sha256, and the server checks them at startup. Copies are in
  [`reproduce/manifests/`](reproduce/manifests).
- **Code:** no commit existed when these ran (§6, item 1). Each `eval/*` tag holds the frozen copy that served that submission.
  `git show eval/student_v1_3:README.md` explains how it was made.
- **What changed since:** `git diff eval/student_v1_3 main -- tower_rl scripts configs`.

## 2. Serve a submission

The server needs Python 3.11, about 2 GB of GPU memory and
[these packages](reproduce/environment/requirements-inference.txt). It does not need Isaac Sim, Isaac Lab or the challenge kit.
You need the bundle tar and the challenge's `track.json`.

```bash
python3.11 -m venv venv && venv/bin/pip install -r reproduce/environment/requirements-inference.txt
PYTHON=$PWD/venv/bin/python reproduce/evaluated/serve_evaluated.sh student_v1_3 student_v1_3.tar.gz track.json
# second terminal: the qualification-like checks (every variant, two resets, zero-valued images)
cd ../tower_rl_eval_student_v1_3
../tower_rl/venv/bin/python scripts/probe_server.py ws://127.0.0.1:9004 --track /path/to/track.json
```

The evaluator needs a public `wss://` URL, for example `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:9004`. To do
the steps by hand, follow the README of the tag. The scripts that served the official runs are in
[`reproduce/evaluated/official/`](reproduce/evaluated/official). Versions: [`reproduce/environment/`](reproduce/environment).

Actions are deterministic given the observations: PPO mean, BC forward pass, masked argmax, no sampling. Replaying the same
observations should repeat the decisions, but our own rendered closed-loop tests vary between runs, so a re-evaluation need not
repeat every episode.

## 3. What is in a bundle

13 of the 14 networks are byte-identical in all three submissions. Only the router differs: `bc_r2` in `student_v1` and
`student_v1_2`, `bc_v2_r3` in `student_v1_3`.

| Used by | Network | Trained by | Rule-based and planned parts around it |
|---|---|---|---|
| All tasks | `perception/kp_v2.pt`: keypoint network, followed by a multi-view pose fit and a tracker | Supervised | The pose fit and the tracker |
| All tasks | `student/switcher.pt`: the router (option, goal, arm) | Behaviour cloning + DAgger on a simulated oracle | Masks, finished and blind guards (flags from v1.1 on) |
| Tasks 1-3 | `skills/grasp_lift_{left,back}.pt`, `skills/place_release_{left,back}.pt` | PPO | Planned transits and parks; from v1.3, a planned place after off-centre grips |
| Task 4 | `skills/push_{left,back}.pt` | PPO | Planned transits and parks; straight-push handover; from v1.2, a re-push rule; `push_settle` |
| Task 5 | `skills/grasp_t5_bc_*.pt`, `skills/insert_bc_*.pt`, `skills/push_in_bc_*.pt` | Behaviour cloning of planned controllers (DART) | Planned transits and parks; planned release, retreat and verdicts |

The bundle's `graphs/` hold the per-variant task graph and its flags. `student_v1_1` and later add `perception/orient_refs.npz`,
reference images from the public dataset that choose the fixed cameras' orientation per episode.

## 4. The organizers' rules: where to check

| Rule | Where to check |
|---|---|
| The same model for the whole evaluation | Each completed evaluation ran under one server process and one artifact identity (`run.json`, `server.jsonl` in the logs) |
| Only the provided observations | `tower_rl/server/handler.py`, `tower_rl/student/adapter.py`. The server imports no simulator code, and the variant comes from the protocol's `X-Tower-Task-Variant` header. The privileged oracle exists only in training |
| No human intervention | We took no action on the policy or the server during an evaluation. The server was only restarted after machine resets, between attempts. During `student_v1`'s evaluation we used the site's "Retry failed episodes" action |
| No hard-coding from test layouts | `configs/layouts/` and the graphs come from the public dataset (`scripts/build_layouts.py`). Seed and trial are only validated, echoed and logged. Official feedback did shape later versions (§6, item 7) |

## 5. What Claude did

Claude Code wrote the code, the task graphs and the recovery rules offline. Nothing at inference, training or data collection calls
Claude or any other language model. See [`reproduce/README.md`](reproduce/README.md) §4.1. The build specification and the student
plan are in [`docs/`](docs).

## 6. Differences between these materials and what ran

1. **No commit at evaluation time.** The servers ran from frozen copies of the code outside git. Bundle MANIFESTs record
   `git_commit: f5e8cce` with `git_dirty: true`. The `eval/*` tags were made from those copies on 2026-11-05, so their dates are later
   than the evaluations.
2. **The tags add three files to the frozen trees:** `assets_m/tower_station_m.json`, `.gitignore`, `README.md`. The copies' symlinks
   to `data`, `assets_m` and `results` are not included. Everything else is byte-identical.
3. **`main` differs from what ran.**
   - Against `student_v1_3`: `tower_rl/graph/executor.py`, `scripts/rehearse.py`, `scripts/train_skill.py` and
     `scripts/widen_push_checkpoint.py` changed, `README.md` and `.gitignore` were updated, and files for learned reaching and machine monitoring, tests, documentation and the reproduction package were added. A graph without a `reach`
     block takes the same code paths as before. Replaying the `student_v1_3` bundle on both versions (ten variants, CPU, oracle perception) gave
     identical episode records apart from timing fields.
   - Against `student_v1` and `student_v1_2`: more has changed, with no such check, so serve those from their tags.
4. **Runtime configs come from the code, not the bundle.** The server reads `configs/` from its code directory and only hash-checks
   the bundle's copies. In all three submissions those copies lack `configs/student/server.yaml` and lack the `release_min_along_m` key
   in `configs/skills/insert.yaml`, which only the unused PPO insert node reads. In `student_v1_3` they also lack tier D1e in
   `configs/tiers.yaml`, which the server does not read.
5. **Two runtime inputs sit outside the bundle:** `track.json`, passed as an argument, and `assets_m/tower_station_m.json` (the camera
   calibration section of a file generated from TOWER-Assets), which the tags and `main` include.
6. **Restarts between attempts.** Each restart after a machine reset served the same code and the same artifact identity, through a
   new tunnel URL. The site's endpoint check was repeated before each new evaluation.
7. **The policy changed between submissions, using official feedback.** Official results, replays, server logs and saved evaluator images
   of earlier submissions shaped later versions:
   - at run time: the per-episode camera orientation and its fallback, 16-row idle holds, the finished and blind guards, `push_settle`,
     `offcentre_place`, the Task 4 re-push rule and the rest-state check behind it (`tower_rl/eval/mirror.py`, calibrated against the site's
     Task 4 results);
   - in training the `bc_v2_r3` router: 210 s episodes judged by that check, injected off-centre grips, an oracle rule that leaves buried
     failed goals alone, and the wider tier D1e.

   Among the networks, only that router was trained after the official runs. No official frame trained any network, no test layout is stored,
   and no training script reads server logs.
8. **Other GPU jobs ran during `student_v1_3`'s evaluation:** reacher training, Isaac camera-loop rehearsals, and a steady GPU load kept on
   purpose, because the machine's earlier resets had come at light load. Perception slowed by about 2-3x. The simulation pauses during inference,
   and the server logged 0 errors.
9. **Later versions were never evaluated.** `student_v1_4` and `student_v1_5` (learned reaching, researched from 2026-10-30 and frozen after the
   deadline) are in this repository and in our blog post. `student_v1_4` passed an endpoint check on 2026-11-01, after submissions had closed.
10. **The blog post reports local results and local timing.** For the evaluated versions, the technical report is authoritative.
11. **`teacher_t45_push9a`'s MANIFEST is stale.** That intermediate bundle was assembled by copying, which affects lineage only: see
    [`reproduce/README.md`](reproduce/README.md) §6.
12. **Dependency versions differ from the kit's pins** (numpy, msgpack), and torch and torchvision come from different CUDA indexes. See
    [`reproduce/environment/versions.md`](reproduce/environment/versions.md).
13. **A full rebuild gives an equivalent policy, not identical bytes.** `reproduce/` retrains from the public downloads in about 40 hours.
    That pipeline has not been rerun end to end. Freezes from the original checkpoints are byte-exact (`reproduce/README.md` §10).
