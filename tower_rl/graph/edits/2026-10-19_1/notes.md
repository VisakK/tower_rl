# Edit iteration 1: outcome (2026-10-19): REVERTED

Edit: every `place_release` node `on_fail: skip_if_close` (15 mm); graphs version 1 -> 2 (`diff.patch`).
Reverted: graphs version 3 = version 1 content (default `retry`).

## Measurements (D1, 8 episodes, seed 1000, CPU station scene)

Executor versions: **E0** = at the "before" runs; **E1** = E0 + `release_pose()` fix + `on_fail` support;
**E2** = E1 + the executor fixes found while diagnosing this edit (below).

| run | file | TSR | GSR | PSR | LHPR | ODR |
|---|---|---|---|---|---|---|
| Task 1-3 planned, graphs v1, E0 | `before_task1-3_D1_planned.json` | 0.875 | 0.958 | 0.875 | 0.917 | 0.093 |
| Task 1-3 planned, graphs v2, E1 | `after_E1_task1-3_D1_planned_graphs_v2.json` | 0.75 | 0.896 | 0.771 | 0.833 | 0.102 |
| Task 1-3 planned, graphs v1, E2 | `ab_E2_task1-3_D1_planned_graphs_v1.json` | **0.875** | 1.0 | 0.938 | 0.969 | 0.021 |
| Task 1-3 planned, graphs v2, E2 | `ab_E2_task1-3_D1_planned_graphs_v2.json` | 0.75 | 1.0 | 0.896 | 0.948 | 0.041 |
| Task 2-2 left grasp it. 200, graphs v1, E0 | `before_task2-2_D1_learned.json` | 0.875 | 0.938 | 0.875 | 0.906 | 0.030 |
| Task 2-2 left grasp it. 200, graphs v2, E1 | `after_E1_task2-2_D1_gl200_graphs_v2.json` | 0.875 | 1.0 | 0.875 | 0.938 | 0.0 |
| Task 2-2 left grasp it. 200, graphs v1, E2 | `ab_E2_task2-2_D1_gl200_graphs_v1.json` | **1.0** | 1.0 | 1.0 | 1.0 | 0.0 |
| Task 2-2 left grasp it. 200, graphs v2, E2 | `ab_E2_task2-2_D1_gl200_graphs_v2.json` | 0.875 | 1.0 | 0.938 | 0.969 | 0.0 |

## What the diagnosis found (`scripts/trace_episode.py`, per-call and per-substep traces)

1. **Executor bug: a skip jumped over the arm-switch park.** `_advance` jumped to the next transfer's
   `first_of_transfer` node, but the park that clears the failing arm belongs to the next transfer and precedes that
   node. After a skipped left-arm place the left arm stayed hovering over pad B, and the back arm's held block ran into
   it (pre-place transit timeouts, then re-grasp timeouts). This was the real cause of the cascades that motivated the
   edit (the original seed 1100001 cascade: retries exhausted -> skip -> same bug). Fixed in E2: a skip continues at
   the next node of a later transfer; a timed-out park no longer skips the following transfer.
2. Near-miss places come from a rare **pop on release** (Task 2-2 seed 1000003: the released block fell 5 mm, landed
   on its low end, pivoted, penetrated the layer below by 1.6 mm and was ejected at 0.6 m/s, landing 24 mm away; the
   retreating finger dragged it back to 8.9 mm). A free-drop test (`scripts/drop_test.py`, 320 drops of 0-8 mm on CPU
   and on GPU) never pops, so the opening fingers are involved; rate in rehearsals ~2-4 % of planned places.
3. The planned grasp lets heavy + slippery blocks ("combined") slide 14 mm in the fingers during its 10 cm/s lift and
   drop them in transit (seed 1100001: the block fell off the table). E2: a held transit fails as soon as the measured
   opening drops below the block width ("dropped"), and a transfer whose block fell off the table (z < -2 cm) is
   skipped instead of retried. (The learned grasp holds that block: TSR 1.0 on that seed.)

## Decision

With E2, retrying a near-miss place (re-grasp from the destination stack, re-place) recovers both affected episodes
(Task 1-3 seed 1000001: re-placed at 1.4 mm; Task 2-2 seed 1000003: re-placed at 0.5 mm on the second attempt), while
`skip_if_close` gives up the P and with it the task. The edit regresses TSR on both variants under the same executor
-> revert (protocol: keep an edit only if it improves or fixes a named signature without regressions). The executor's
`on_fail` support stays (default `retry`), as does `skip` for exhausted retries.
