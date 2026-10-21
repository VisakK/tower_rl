# Edit iteration 2: outcome (2026-10-19): KEPT

Edit: held transits `v_max_held` 0.12 -> 0.08 m/s (graphs v3 -> v4, `diff.patch`).

| Task 1-3 D1, 48 episodes, seed 9000, same bundle | TSR (Wilson 95 %) | GSR | PSR | LHPR | ODR | mean duration |
|---|---|---|---|---|---|---|
| graphs v3 (`before_task1-3_D1.json`) | 0.833 (0.70-0.91) | 0.983 | 0.931 | 0.957 | 0.067 | 128.6 s |
| graphs v4 (`after_task1-3_D1.json`) | **1.000** (0.93-1.00) | 1.000 | 1.000 | 1.000 | 0.000 | 132.6 s |

Failure signatures after the edit: one place miss in 288 places (re-placed successfully); no drops, no grasp
timeouts, no re-grasps. The improvement is far outside the Wilson intervals; the episodes take 4 s longer (budget
210 s). The slower transport removed not only the drops but most placement misses: the block no longer shifts in the
fingers between the grasp (where its offset is measured) and the release.

Follow-ups: the same mechanism argues for the 30 Hz track (rows of 1/30 s, smaller steps) and for checking whether
even slower held transits help the heavy cases further; a 0.08 m/s held speed is now the default in every graph.
