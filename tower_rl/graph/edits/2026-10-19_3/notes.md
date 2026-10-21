# Edit iteration 3: outcome (2026-10-19): KEPT

Edit: top-level `"perception": {"jaw_to_goal": 0.5}` in all six graphs (graphs v4 -> v5, `diff.patch`); the executor
moves half of the held block's perceived jaw offset from the block to the place policy's goal (`LearnedPlaceNode`).

Evidence (per-place diagnostics, `*.jsonl` here, summarized with `python -I scripts/summarize_noise_diag.py <file>`;
teacher_v5, D1, seed 6000, 8 episodes; true error at the release, along the gripper's jaw):

| noise model | w = 0 | w = 1 |
|---|---|---|
| realistic (independent 1 mm / 1 deg + shared 3 mm / 3 deg, jitter, dropouts, outliers) | Task 3-3 sd 3.0 mm, places > 5 mm 15 %; Task 2-3 back arm > 5 mm 7 % | sd 1.2 mm, 4 %; back arm 0 % |
| build file (independent 3 mm / 3 deg per block, same jitter / dropouts / outliers) | Task 3-3 sd 1.4 mm, TSR 2 / 8 | sd 3.4 mm, TSR 2 / 8 |

(The w = 1 realistic runs, `*_v5_jaw1.jsonl`, used this option's first name, `jaw_bias_to_goal`; same arithmetic.)
`*_v5_rel1.jsonl`: moving the whole in-hand offset (jaw and block axis) instead; rejected (Task 2-3 back / left places
> 5 mm 0.15 / 0.50 against 0.07 / 0.30 without).

0.5 is a hedge between the two models: noise-free it is a no-op (true jaw offsets ~0.1 mm). Bundle `teacher_v6` with
graphs v5: Step 7 acceptance N = 100 per cell, all 12 pass (1182 / 1200); realistic noise D1 TSR 0.29-0.88 (BUILDLOG
"Step 8.1 round 2"). Recalibrate w once the pose estimator exists: Cov(goal error, held-block jaw error) /
Var(held-block jaw error) along the jaw, from its validation renders; `jaw_to_goal` may also be one weight per arm.
