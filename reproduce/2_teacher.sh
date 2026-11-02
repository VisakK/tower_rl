#!/usr/bin/env bash
# Stage 2 (reproduce/README.md §4): the teacher's PPO skills and the teacher bundle teacher_t45_final, as built for student_v1_3
# (docs/Teacher-v1.MD §4). GPU, the light one-arm skill scene, one job at a time; each run ~30-80 min on an RTX A5000
# (4096 envs for grasp / place, 3072 for push from configs/skills/push.yaml). Stage 4 fine-tunes the two pushes.
# The chains are the original ones (reproduce/README.md §4.3), with two substitutions:
#   - runs that were stopped by hand at a checkpoint get the --max_iterations that ends there (rsl_rl resumes at the
#     checkpoint's iteration and saves model_<it> every 100 iterations): v2 (101), v3 (201), v1_from_left300 (201),
#     place v1 (401);
#   - runs made before a config key existed switch it off by --override (same behaviour, a different run-dir hash):
#     the grasp quality gate ("quality"), the place's wide held offsets and precision bonus, v2's original reward.
# Not bit-exact: GPU PhysX; and the runs before 2026-10-19 06:21 drew the observation jitter twice per step (no switch
# for that in the code). Judge a rerun by the gates in reproduce/README.md §4.3.
# Output: results/skills/<skill>_<arm>/<date>_<time>_<hash>_<tag>/, results/bundles/teacher_t45_final/
# Usage: reproduce/2_teacher.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
T=("$PY" scripts/train_skill.py --headless)
# the Step 8.1 perception-noise model on the actor's inputs (v5n / v3n)
NOISE='{"obs_noise": {"episode_pos_m": 0.003, "episode_yaw_deg": 3.0, "step_pos_m": 0.001, "step_yaw_deg": 0.5, "dropout_p": 0.02, "dropout_steps": 5, "outlier_p": 0.005, "outlier_m": 0.01}}'

# grasp_lift, left: v2 -> v3 -> v4q_from300 -> v5n_noise (bundle)
run "${T[@]}" --skill grasp_lift --arm left --num_envs 4096 --max_iterations 101 --tag v2 \
    --override '{"reward": {"success_per_remaining_step": 3.0}, "quality": null}'
run "${T[@]}" --skill grasp_lift --arm left --num_envs 4096 --max_iterations 201 --tag v3 --override '{"quality": null}' \
    --resume "$(latest_run grasp_lift_left v2)/model_100.pt"
run "${T[@]}" --skill grasp_lift --arm left --num_envs 4096 --max_iterations 300 --tag v4q_from300 \
    --resume "$(latest_run grasp_lift_left v3)/model_300.pt"
run "${T[@]}" --skill grasp_lift --arm left --num_envs 4096 --max_iterations 250 --tag v5n_noise --override "$NOISE" \
    --resume "$(latest_run grasp_lift_left v4q_from300)/model_final.pt"
# grasp_lift, back: from the left arm's iteration 300 -> v2q_from500 -> v3n_noise (bundle)
run "${T[@]}" --skill grasp_lift --arm back --num_envs 4096 --max_iterations 201 --tag v1_from_left300 \
    --override '{"quality": null}' --resume "$(latest_run grasp_lift_left v3)/model_300.pt"
run "${T[@]}" --skill grasp_lift --arm back --num_envs 4096 --max_iterations 300 --tag v2q_from500 \
    --resume "$(latest_run grasp_lift_back v1_from_left300)/model_500.pt"
run "${T[@]}" --skill grasp_lift --arm back --num_envs 4096 --max_iterations 250 --tag v3n_noise --override "$NOISE" \
    --resume "$(latest_run grasp_lift_back v2q_from500)/model_final.pt"
# place_release: left v1 (bundle: model_400), back v1_scratch (bundle)
run "${T[@]}" --skill place_release --arm left --num_envs 4096 --max_iterations 401 --tag v1 \
    --override '{"reset": {"grasp_offset_wide": null}, "reward": {"success_precision": 0.0}}'
run "${T[@]}" --skill place_release --arm back --num_envs 4096 --max_iterations 600 --tag v1_scratch
# push (Task 4): left v3; back v1 -> v1c (the bases of stage 4's fine-tunes)
run "${T[@]}" --skill push --arm left --max_iterations 300 --tag v3
run "${T[@]}" --skill push --arm back --max_iterations 300 --tag v1
run "${T[@]}" --skill push --arm back --max_iterations 250 --tag v1c --resume "$(latest_run push_back v1)/model_final.pt"

# gates (500 episodes; student_v1_3's skills: grasp left D0 / D1 100 / 98.8 %, back 99.2 / 100 %; place left 99.6 /
# 99.0 %, back 99.8 / 100 % without perception noise), e.g.
#   python scripts/evaluate_skill.py --headless --skill grasp_lift --arm left --checkpoint <run dir> --tier D1 --episodes 500

# the teacher bundle: the committed graphs (tower_rl/graph/graphs/), the configs, the skills as TorchScript; insert
# and push-in stay planned (planned_fallbacks)
run "$PY" scripts/freeze_bundle.py --out results/bundles/teacher_t45_final \
    --skill "grasp_lift:left=$(latest_run grasp_lift_left v5n_noise)/model_final.pt" \
    --skill "grasp_lift:back=$(latest_run grasp_lift_back v3n_noise)/model_final.pt" \
    --skill "place_release:left=$(latest_run place_release_left v1)/model_400.pt" \
    --skill "place_release:back=$(latest_run place_release_back v1_scratch)/model_final.pt" \
    --skill "push:left=$(latest_run push_left v3)/model_final.pt" \
    --skill "push:back=$(latest_run push_back v1c)/model_final.pt"
# optional: the Step 7 acceptance rehearsal of the whole teacher (CPU station scene; reproduce/README.md §4.4)
#   python scripts/rehearse_many.py --bundle results/bundles/teacher_t45_final --episodes 100 --parallel 4 \
#       --out_root results/rehearsal_final/teacher_t45_final
