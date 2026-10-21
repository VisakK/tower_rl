# Edit iteration 2 (2026-10-19, Claude Code, Opus 5.5, in the build session)

## Input

Rehearsal of Task 1-3 D1, 48 episodes, seed 9000, bundle `probe_v4grasps` (learned grasps left v4q / back v2q,
learned left place v1 it. 400, planned back place), graphs v3, CPU station scene (`feedback_task1-3_D1_before.md`,
`before_task1-3_D1.json`): TSR 0.83 (Wilson 0.70-0.91), GSR 0.98, PSR 0.93, ODR 0.07. Failure signatures:

| node | status | count |
|---|---|---|
| place | failed (6-17 mm) | 16 |
| to_pregrasp (re-grasp of a missed block) | failed: pregrasp unreachable | 10 |
| to_preplace (held transit) | failed: dropped | 6 |
| grasp | timeout | 6 |
| grasp_alt (arm-switch re-grasp) | timeout | 5 |

The failing episodes are the heavy / uneven-weight cases (block mass up to 2x).

## Reasoning

Six blocks fell out of the gripper during held transits, and the misses concentrate on heavy blocks: the held block
moves in the fingers during transport. Held transits run at 0.12 m/s; the 10 Hz track holds every row for 0.1 s, so
each row is a 12 mm step of the joint targets, which excites the soft arm (BUILDLOG: replayed demos drop blocks at
10 Hz, not at 30 Hz). A block that shifts or rotates in the fingers is then released from a pose computed for its
grasp-time offset (placement misses), or slips out (drops); the retries re-grasp from the destination stack, which
fails (unreachable / out-of-distribution re-grasps). Slower held transport makes the steps smaller (8 mm per row).

## Edit (1 of max 5)

Every held transit (`skill: transit`, `held: true`) of all six graphs: `v_max_held: 0.08` (was the 0.12 m/s default;
the executor reads `v_max`, `v_max_held`, `a_max` from transit nodes since this iteration). Graph version 3 -> 4.

## Acceptance

Re-rehearse Task 1-3 D1 with the same 48 seeds and bundle (only the graphs changed): keep if TSR improves beyond its
Wilson interval or the drop / miss signatures disappear without regressions; watch the episode duration (210 s
budget).
