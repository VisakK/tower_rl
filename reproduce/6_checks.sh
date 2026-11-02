#!/usr/bin/env bash
# Stage 6 (reproduce/README.md §8): check a frozen student bundle the way student_v1_3 was checked before its submission. The
# numbers in brackets are student_v1_3's; a retrained bundle should land near them, not on them (the camera loop is
# noisy: 10 episodes per variant resolve about one point).
#   cpu     the student on CPU with the estimator noise model v1 instead of the camera (no rendering), the official
#           mirror (episodes to 210 s, success on the final arrangement): all ten variants, the ten D1 cases,
#           12 episodes per variant and seed (mirror points: seed 7700 [80.09], seed 7900 [77.56] of 100). ~40 min
#   camera  the camera loop on the GPU: Isaac renders the four cameras, kp_v2 estimates the poses, the student acts
#           (Tasks 1-3, 10 episodes per variant, seed 7200 [59 / 60 task successes]; Task 4 and Task 5, seed 7400).
#           ~4 h; the rendering keeps the GPU busy (the desktop lags)
#   e2e     through the challenge protocol: the server (reproduce/7_serve.sh's, with rendered images) and a WebSocket
#           client driving the station scene, 4 variants x 2 episodes [0 errors]. ~25 min
# Env: BUNDLE (default results/bundles/student_v1_3), P (CPU processes, default 10), OUT (default results/checks/<bundle>).
# Usage: reproduce/6_checks.sh cpu [camera] [e2e]
set -euo pipefail
source "$(dirname "$0")/env.sh"
B=${BUNDLE:-$R/bundles/student_v1_3}
SW=$B/student/switcher.pt
O=${OUT:-$R/checks/$(basename "$B")}
P=${P:-10}
CASES="nominal lightweight heavy uneven_weights slippery high_friction shifted_assembly irregular_layers lighting combined"
ALL="task1-2_tower_transfer_2 task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 task2-3_cross_tower_transfer_3
task3-2_vertical_stacking_2 task3-3_vertical_stacking_3 task4-2_full_tower_pushdown_2 task4-3_full_tower_pushdown_3
task5-2_tower_insertion_2 task5-3_tower_insertion_3"
jobs_run() { while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done; }
run mkdir -p "$O/logs"

step_cpu() {
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  local seed v
  for seed in 7700 7900; do
    for v in $ALL; do
      local cmd=(nice -n 8 "$PY" scripts/rehearse.py --headless --device cpu --variant $v --tiers D1 --case $CASES
                 --episodes 12 --num_envs 4 --seed $seed --bundle "$B" --student "$SW"
                 --est_noise "$R/student/noise/noise_v1_0034.yaml" --mirror --out_root "$O/cpu_$seed")
      echo "+ ${cmd[*]}"
      if [ "${DRY_RUN:-0}" != 1 ]; then jobs_run; "${cmd[@]}" < /dev/null > "$O/logs/cpu_${seed}_$v.log" 2>&1 & fi
    done
  done
  wait
  for seed in 7700 7900; do run "$PY" scripts/mirror_points.py "$O/cpu_$seed"; done
}

cam() {  # cam <out dir> <seed> <variant> [extra rehearse.py args]
  run "$PY" scripts/rehearse.py --headless --device cpu --variant $3 --tiers D1 --case $CASES --episodes 10 --num_envs 2 \
      --seed $2 --bundle "$B" --student "$SW" --perception "$B/perception/kp_v2.pt" --perception_device cuda --mirror \
      --out_root "$1" "${@:4}"
}

step_camera() {
  local v
  # Tasks 1-3: the evaluator delivers the fixed cameras unrotated there (--official_orient)
  for v in task1-2_tower_transfer_2 task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 task2-3_cross_tower_transfer_3 \
           task3-2_vertical_stacking_2 task3-3_vertical_stacking_3; do
    cam "$O/camera_t13" 7200 $v --official_orient
  done
  run "$PY" scripts/mirror_points.py "$O/camera_t13" --by_case
  for v in task4-2_full_tower_pushdown_2 task4-3_full_tower_pushdown_3 task5-2_tower_insertion_2 task5-3_tower_insertion_3; do
    cam "$O/camera_t45" 7400 $v
  done
  run "$PY" scripts/mirror_points.py "$O/camera_t45" --by_case
}

step_e2e() {
  local port=${PORT:-9006} log=$O/logs/e2e_server.log v
  echo "+ serve_student.py on port $port > $log"
  [ "${DRY_RUN:-0}" = 1 ] && return
  "$PY" scripts/serve_student.py --track data/challenge/track.json --bundle "$B" --name "e2e_$(basename "$B")" \
      --port $port --camera_source render --policy_id tower-student-check --plugin_version 2 > "$log" 2>&1 &
  local server=$!
  for _ in $(seq 1 180); do grep -q READY "$log" 2>/dev/null && break; sleep 1; done
  grep -q READY "$log" || { echo "server not ready (see $log)"; kill $server; exit 1; }
  for v in task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 task4-2_full_tower_pushdown_2 task5-3_tower_insertion_3; do
    run "$PY" scripts/ws_loop.py --headless --device cpu --variant $v --tiers D1 --case nominal combined --episodes 2 \
        --num_envs 2 --seed 7600 --track data/challenge/track.json --url ws://127.0.0.1:$port --official_orient --mirror \
        --out_root "$O/e2e" < /dev/null
  done
  kill -INT $server; sleep 10; kill -9 $server 2>/dev/null || true
  run "$PY" scripts/server_episodes.py "$R/student/server/e2e_$(basename "$B")" --episodes
}

[ $# -gt 0 ] || { echo "usage: reproduce/6_checks.sh cpu|camera|e2e ..."; exit 1; }
for s in "$@"; do
  echo "== $s"
  "step_$s"
done
