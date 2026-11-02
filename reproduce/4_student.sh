#!/usr/bin/env bash
# Stage 4 (reproduce/README.md §6): distil the teacher into the student, as built for student_v1_3 (docs/Student-v1.MD,
# docs/Student-v1.1.MD, docs/Student-v1.3.MD). Needs stages 2-3 (teacher_t45_final, the push bases, kp_v2).
# Steps, in order (run a subset by naming them: reproduce/4_student.sh distill router_v2):
#   noise       install the fitted estimator-noise models (reproduce/noise/) into results/student/noise/
#   push9a      Task 4 pushes fine-tuned under the estimator's failure model (GPU, ~35 min per arm and stage) and the
#               teacher bundle teacher_t45_push9a
#   distill     Task 5 grasp / insert / push-in: DART data from the planned controllers (CPU, ~2.3 h with 4
#               processes), behaviour cloning (~2 min)
#   router_v1   the first router bc_r2 (features v1: oracle round + 2 DAgger rounds, CPU ~10 h). It only fills
#               student_v1's switcher slot: nothing below reads it (the v2 rounds pass --student; v1.3 replaces it)
#   student_v1  freeze student_v1 = teacher_t45_push9a + the Task 5 BC skills + kp_v2 + bc_r2, graphs "distilled"
#   router_v2   the v1.3 router bc_v2_r3 (features v2, labels from the recovery node oracle_state_v2): an oracle round
#               and three DAgger rounds, each followed by training (CPU, ~6 h collection + ~1 h training)
# Then reproduce/5_build_v1_3.sh freezes student_v1_3. Env: P = rehearsal processes at a time (default 10; the results
# do not depend on it), DRY_RUN=1. A round's output directory must not exist (shards append to its index).
# Not bit-exact: GPU PPO; and every DAgger round acts with the previous round's switcher, so a retrained switcher
# changes the later rounds' data.
set -euo pipefail
source "$(dirname "$0")/env.sh"
P=${P:-10}
T=("$PY" scripts/train_skill.py --headless)
NV1=$R/student/noise/noise_v1_0034.yaml  # the estimator noise model v1: the switchers' training noise
ALL="task1-2_tower_transfer_2 task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 task2-3_cross_tower_transfer_3
task3-2_vertical_stacking_2 task3-3_vertical_stacking_3 task4-2_full_tower_pushdown_2 task4-3_full_tower_pushdown_3
task5-2_tower_insertion_2 task5-3_tower_insertion_3"
CASES="nominal lightweight heavy uneven_weights slippery high_friction shifted_assembly irregular_layers lighting combined"
M=$R/student/models
fresh() { if [ "${DRY_RUN:-0}" != 1 ] && [ -e "$1" ]; then echo "$1 exists: remove it to collect again"; exit 1; fi; }
jobs_run() { while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done; }

step_noise() {
  # fitted from the per-call perception logs of camera-loop rehearsals with kp_v2 (scripts/fit_estimator_noise.py
  # --min_calls 40 [--out_bank ...]); v1 (2026-10-22 00:34) trains every switcher, v2 (06:59, configs/student/noise.yaml)
  # the Task 4 pushes. Their trace banks are recorded data, shipped so that the rounds below can be rerun
  run mkdir -p "$R/student/noise"
  run cp reproduce/noise/noise_v1_0034.yaml reproduce/noise/traces.npz reproduce/noise/traces_v2.npz "$R/student/noise/"
}

step_push9a() {
  # the push target is lost with the fitted P(lost | tilt, travel) and coasts; the actor gets seen / since_seen
  # (65 -> 67 inputs: the base checkpoints are widened first)
  local OV='{"obs_noise": {"estimator": {"noise_yaml": "student/noise"}, "dropout_p": 0.015, "dropout_steps": 2}}'
  local OV_V1='{"obs_noise": {"estimator": {"noise_yaml": "results/student/noise/noise_v1_0034.yaml"}, "dropout_p": 0.015, "dropout_steps": 2}}'
  run mkdir -p "$R/student/push9a"
  run "$PY" scripts/widen_push_checkpoint.py "$(latest_run push_left v3)/model_final.pt" "$R/student/push9a/push_left_v3_widened.pt"
  run "$PY" scripts/widen_push_checkpoint.py "$(latest_run push_back v1c)/model_final.pt" "$R/student/push9a/push_back_v1c_widened.pt"
  # left: 202 iterations under the v1 model (the original's refit had not landed yet), then 150 under v2, optimizer kept
  run "${T[@]}" --skill push --arm left --max_iterations 202 --tag 9a_est --override "$OV_V1" \
      --resume "$R/student/push9a/push_left_v3_widened.pt" --resume_no_optimizer
  run "${T[@]}" --skill push --arm left --max_iterations 150 --tag 9a_est_v2 --override "$OV" \
      --resume "$(latest_run push_left 9a_est)/model_500.pt"
  run "${T[@]}" --skill push --arm back --max_iterations 250 --tag 9a_est_v2 --override "$OV" \
      --resume "$R/student/push9a/push_back_v1c_widened.pt" --resume_no_optimizer
  # gates (student_v1_3's pushes: D1 under the v2 model, left 98.8 %, back 95.4 %)
  for arm in left back; do
    run "$PY" scripts/evaluate_skill.py --headless --skill push --arm $arm --checkpoint "$(latest_run push_$arm 9a_est_v2)" \
        --tier D1 --episodes 500 --override "$OV" --out "$R/student/push9a/eval_${arm}_D1_est_v2.json"
  done
  # teacher_t45_push9a = teacher_t45_final with these pushes. The original was a copy of teacher_t45_final with the
  # two exports written over it (its MANIFEST still lists the old pushes); a fresh freeze gives the same graphs and
  # weights (TorchScript bytes differ) and a correct MANIFEST
  run "$PY" scripts/freeze_bundle.py --out "$R/bundles/teacher_t45_push9a" \
      --skill "grasp_lift:left=$(latest_run grasp_lift_left v5n_noise)/model_final.pt" \
      --skill "grasp_lift:back=$(latest_run grasp_lift_back v3n_noise)/model_final.pt" \
      --skill "place_release:left=$(latest_run place_release_left v1)/model_400.pt" \
      --skill "place_release:back=$(latest_run place_release_back v1_scratch)/model_final.pt" \
      --skill "push:left=$(latest_run push_left 9a_est_v2)/model_final.pt" \
      --skill "push:back=$(latest_run push_back 9a_est_v2)/model_final.pt"
}

step_distill() {
  # DART: OU noise on the planned controllers' rows (scale 1.5), the clean action as the label; 300 episodes per
  # variant (5-2, 5-3) and tier (D0, D1) in 3 chunks (seeds seed + 10000 k); then one small MLP per skill and arm
  # (v2: the previous action masked from the inputs)
  fresh "$R/student/distill/data/c1"
  run "$PY" scripts/collect_distill.py --bundle "$R/bundles/teacher_t45_final" --out "$R/student/distill/data/c1" \
      --episodes 300 --chunks 3 --parallel 4 --seed 200000
  run "$PY" scripts/train_distill.py --data "$R/student/distill/data/c1" --out "$R/student/distill/models/v2"
}

step_router_v1() {
  # the oracle (oracle_state) acts with exploration 0.1 and failure injection 0.15; DAgger rounds: the previous
  # switcher acts (the oracle with probability beta), the oracle labels; the teacher bundle, the estimator noise v1
  local m=("$PY" scripts/rehearse_many.py --bundle "$R/bundles/teacher_t45_final" --variants $ALL --num_envs 4 --parallel "$P")
  fresh "$R/student/data/r0"
  local X="--collect $R/student/data/r0 --inject 0.15 --explore 0.1 --est_noise $NV1"
  run "${m[@]}" --tiers D1 --episodes 200 --seed 105000 --student oracle_state --extra "$X" --out_root "$R/student/collect/r0"
  run "${m[@]}" --tiers D0 --episodes 100 --seed 100000 --student oracle_state --extra "$X" --out_root "$R/student/collect/r0"
  run "$PY" scripts/train_switcher.py --data "$R/student/data/r0" --out "$M/bc_r0" --device cpu --progress_frac 0.3 --threads 8
  local prev=bc_r0 data="$R/student/data/r0" rnd beta seed
  for rnd in r1 r2; do
    if [ $rnd = r1 ]; then beta=0.5; seed=110000; else beta=0.0; seed=120000; fi
    fresh "$R/student/data/$rnd"
    X="--collect $R/student/data/$rnd --inject 0.1 --beta $beta --est_noise $NV1"
    run "${m[@]}" --tiers D1 --episodes 100 --seed $((seed + 5000)) --student "$M/$prev/switcher.pt" --extra "$X" \
        --out_root "$R/student/collect/$rnd"
    run "${m[@]}" --tiers D0 --episodes 50 --seed $seed --student "$M/$prev/switcher.pt" --extra "$X" \
        --out_root "$R/student/collect/$rnd"
    data="$data $R/student/data/$rnd"
    prev=bc_$rnd
    run "$PY" scripts/train_switcher.py --data $data --out "$M/$prev" --device cpu --progress_frac 0.3 --threads 8
  done
}

step_student_v1() {
  local D=$R/student/distill/models/v2 S=()
  for k in grasp_t5_bc insert_bc push_in_bc; do
    for arm in left back; do S+=(--skill "${k}_$arm=$D/${k}_$arm.pt"); done
  done
  run "$PY" scripts/freeze_student_bundle.py --teacher "$R/bundles/teacher_t45_push9a" --switcher "$M/bc_r2/switcher.pt" \
      --name student_v1 --perception "$R/perception/models/kp_v2/model_last.pt" "${S[@]}" --graph_set '{"distilled": true}'
}

collect_v2() {  # collect_v2 <round> <episodes per variant> <seed> <decider> [beta]: one process per variant
  local rnd=$1 n=$2 seed=$3 dec=$4 beta=${5:-} out=$R/student/collect_v2/$1 v gs tier=D1 cases=$CASES act
  local core='"idle_chunk": 16, "blind_guard": true, "finished_guard": true'
  local inj='{"push_off": {"stop_r_m": [0.125, 0.15], "short_frac": 0.35}}'
  if [ -n "$beta" ]; then act="--beta $beta"; else act="--explore 0.1"; fi
  if [ $rnd = r3 ]; then  # round 3: tier D1e (+ the irregular_edge case), off-centre grips injected, buried goals skipped
    tier=D1e; cases="$CASES irregular_edge"
    inj='{"push_off": {"stop_r_m": [0.125, 0.15], "short_frac": 0.35}, "acquire": {"grasp_along_m": [0.004, 0.008], "grasp_along_frac": 0.34}}'
  fi
  fresh "$out"
  run mkdir -p "$R/student/collect_v2/logs"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  for v in $ALL; do
    gs="{$core}"
    case $v in
      task4*) gs="{$core, \"push_settle\": true}";;
      task5*) ;;
      *) if [ $rnd = r3 ]; then gs="{$core, \"oracle_skip_buried\": true}"; fi;;
    esac
    local cmd=(nice -n 8 "$PY" scripts/rehearse.py --headless --device cpu --variant $v --tiers $tier --case $cases
               --episodes $n --num_envs 4 --seed $seed --bundle "$R/bundles/student_v1" --student "$dec" --est_noise "$NV1"
               --mirror --graph_set "$gs" --collect "$out" --features v2 --inject 0.15 --inject_set "$inj" $act
               --out_root "$R/student/collect_v2/${rnd}_reports")
    local log=$R/student/collect_v2/logs/${rnd}_$v.log
    echo "+ ${cmd[*]} > $log"
    if [ "${DRY_RUN:-0}" != 1 ]; then
      jobs_run
      "${cmd[@]}" < /dev/null > "$log" 2>&1 &
    fi
  done
  wait
}

step_router_v2() {
  local d=$R/student/collect_v2
  train() { run "$PY" scripts/train_switcher.py --data "${@:2}" --out "$M/$1" --features v2 --device cpu --threads 12 --progress_frac 0.3; }
  collect_v2 r0 120 8000 oracle_state_v2
  train bc_v2_r0 $d/r0
  collect_v2 r1 60 8100 "$M/bc_v2_r0/switcher.pt" 0.5
  train bc_v2_r1 $d/r0 $d/r1
  collect_v2 r2 60 8200 "$M/bc_v2_r1/switcher.pt" 0
  train bc_v2_r2 $d/r0 $d/r1 $d/r2
  collect_v2 r3 120 8300 "$M/bc_v2_r2/switcher.pt" 0.5
  train bc_v2_r3 $d/r0 $d/r1 $d/r2 $d/r3  # student_v1_3's router (best epoch 7, validation accuracy 0.968)
}

steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(noise push9a distill router_v1 student_v1 router_v2)
for s in "${steps[@]}"; do
  echo "== $s"
  "step_$s"
done
