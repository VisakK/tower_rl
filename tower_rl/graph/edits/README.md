# Offline Claude edit loop (Step 7.6)

The graph JSONs (`tower_rl/graph/graphs/*.json`) and the skill YAMLs are the only things the loop may change; the
executor code is fixed. Every edit is reproducible from this directory (the rules require prompts and diffs for anything
an LLM authored).

## Protocol

1. Rehearse the current teacher: `python scripts/rehearse.py --headless --device cpu --variant <v> --tiers D0 D1
   --episodes 24 [--bundle bundle]` -> `results/rehearsal/<v>/<tier>/feedback.md`.
2. Run Claude Code on the feedback with permission to edit only:
   - graph JSON node fields (tolerances, heights, timeouts, retries, chunk sizes, node order within a transfer),
   - `configs/skills/*.yaml` tolerances / limits / reward weights (reward edits require retraining the skill).
   At most 5 edits per iteration, at most 10 iterations; stop when TSR and ODR plateau.
3. Record each iteration in `edits/<YYYY-MM-DD>_<n>/`: `prompt.md` (the exact prompt + feedback.md used),
   `diff.patch` (`git diff` of the edited files), `before.json` / `after.json` (rehearsal metrics), `notes.md`.
4. Re-rehearse with the same seeds; keep an edit only if the metric improves beyond its Wilson interval or fixes a
   named failure signature without regressions.

## Iterations

- `2026-10-19_1`: near-miss places skip instead of retrying -> **reverted** (graphs v3 = v1). The cascades it targeted
  were an executor bug (a skip jumped over the arm-switch park), fixed in the executor; with the fix, retrying a
  near-miss place recovers the episode (see its `notes.md`).
- `2026-10-19_2`: held transits 0.12 -> 0.08 m/s -> **kept** (graphs v4): Task 1-3 D1 (48 episodes) TSR 0.83 -> 1.00,
  drops and most placement misses gone (the block shifted in the fingers during fast transport), +4 s per episode.

- `2026-10-19_3`: `perception.jaw_to_goal` 0.5 -> **kept** (graphs v5): under a perception error shared by the held
  block and the foundation the place policies (which ignore the held block's perceived jaw offset) placed the error;
  w = 1 halves the jaw error there, w = 0 is best for independent errors; 0.5 until the pose estimator is measured.
  Input: `scripts/noise_diag.py` diagnostics rather than a rehearsal feedback.md (see its `prompt.md`).

## Open candidates (from the 2026-10-19 rehearsals)

- Rare pops on release (~2-4 % of planned places; the block lands on its low end, pivots and is ejected). A pause
  before opening had no effect; the release harness (block at rest in the hand, no transport) never pops, so the
  likely cause is the block shifting in the fingers during the fast held transit, which edit 2 slowed down: re-check
  the planned place under graphs v4 before spending more on it.
- Placement precision is bounded by perception: with independent 3 mm per-block errors PSR drops to 0.42 (Task 2-3 D1).
- Near misses (5-15 mm) of left-arm places on pad B cannot be re-grasped by the left arm (its pre-grasp there is out
  of reach; the executor now gives up at once). The back arm reaches pad B with both learned skills in distribution:
  an "arm-switch re-place" (park left, back arm re-grasps and re-places, park back) would recover those P's. Needs
  executor support for inserted node sequences; worth it if the final place still misses on pad B.
