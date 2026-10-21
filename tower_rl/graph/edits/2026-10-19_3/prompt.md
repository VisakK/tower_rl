# Edit iteration 3 (2026-10-19, Claude Code, Opus 5.5, in the build session)

## Input

Not a rehearsal `feedback.md`: per-place diagnostics from `scripts/noise_diag.py` (executor estimates against
simulator truth under perception noise), teacher_v5, D1, seed 6000, 8 episodes per run, Task 3-3 and Task 2-3
(`*.jsonl` here; summary: `python -I scripts/summarize_noise_diag.py <file>`). Noise models:

- "realistic": independent 1 mm / 1 deg per block + 3 mm / 3 deg shared by all blocks of the episode, 1 mm / 0.5 deg
  jitter, 2 % dropouts of 0.5 s, 0.5 % outliers of 1 cm (`--noise '{"bias_pos_m": 0.001, "bias_yaw_deg": 1.0,
  "shared_pos_m": 0.003, "shared_yaw_deg": 3.0}'`);
- "build file" (Step 8.1 provisional): independent 3 mm / 3 deg per block, same jitter / dropouts / outliers
  (`--noise '{}'`).

Finding (after the executor fixes of the same day: robust rest-pose estimates, always-attach, windowed place check):
along the gripper's jaw axis the true placement error equals the goal's perception error (correlation 0.96, Task 3-3,
realistic model). The place policies ignore the held block's perceived offset along the jaw (in training the fingers
centre the block, so that offset is pure observation noise) and put the pinch over the perceived goal: an error the
held block shares with the destination foundation does not cancel along the jaw. A probe with a fixed +3 mm bias on
the held blocks only (`--probe_bias`) confirmed it for both arms: true error ~0 along the jaw, ~ -3 mm along the block.

## Edit (1 of max 5)

All six graphs: top-level `"perception": {"jaw_to_goal": 0.5}` (graphs v4 -> v5). The executor
(`LearnedPlaceNode`) then moves half of the held block's perceived jaw offset from the block to the place policy's goal
(same block-to-goal error; the place check still compares the perceived block with the perceived goal).

## Reasoning for 0.5

w = 1 cancels errors shared by all blocks; w = 0 is best for independent per-block errors (the jaw offset of the held
block says nothing about the foundation's error). The optimum is Cov(goal error, held-block jaw error) / Var(held-block
jaw error) along the jaw, a property of the future pose estimator. Measured (true error at the release, along the jaw,
sd): realistic model 3.0 mm (w = 0, Task 3-3 places > 5 mm: 15 %) -> 1.2 mm (w = 1, 4 %); build-file model 1.4 mm
(w = 0) -> 3.4 mm (w = 1), where the independent 3 mm error along the block (sd 3.9 mm) fails 2 / 8 episodes at either
setting. 0.5 roughly halves the worse case of both models; recalibrate once the estimator's errors are measured.
Moving the whole in-hand offset (jaw and block axis) to the goal was tried and rejected (Task 2-3 places > 5 mm
0.07 / 0.30 (back / left) -> 0.15 / 0.50): the policies use the offset along the block.
