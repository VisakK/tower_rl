# Edit iteration 1 (2026-10-19, Claude Code, Opus 5.5, in the build session)

## Input

Rehearsal feedback (copied here): `feedback_task2-2_D1_before.md` (learned left grasp v3 iteration 200, planned
back grasp / places) and `feedback_task1-3_D1_before.md` (planned pipeline). Both fail only on seed 1100001
("combined" case) and the planned Task 1-3 D1 run once more, with the same signature:

| node | status | count |
|---|---|---|
| t0_place | failed (place_xy 5.9 mm vs tol 5 mm) | 1 |
| t0_grasp | timeout (retry: re-grasp from the destination stack) | 1 |
| t0/t1_to_pregrasp | timeout | 3-4 |

## Reasoning

A near-miss place (block resting on the stack 5-15 mm from its goal) triggers the default retry, which re-grasps the
block from the destination pad: for the left arm that pose is outside its demonstrated workspace (the learned grasp
was never trained there; the planned transit times out), the retries consume the time budget and the following
transfers never run. Keeping the near-miss block and continuing loses at most that one P (the evaluator's tolerance
is unknown; 5.9 mm may even pass) while preserving every later G/P (LHPR) and the time budget.

## Edit (1 of max 5)

All `place_release` nodes of all six graphs: `on_fail: skip_if_close`, `skip_if_close_m: 0.015` (executor support for
`on_fail` added in the same session; default behaviour of other nodes unchanged). Graph version 1 -> 2.

## Acceptance

Re-rehearse Task 2-2 D1 (learned left grasp) and Task 1-3 D1 (planned) with the same seeds (1000, 1100): keep the edit
if seed 1100001 no longer cascades and nothing else regresses.
