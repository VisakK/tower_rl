#!/usr/bin/env bash
# Freeze student_v1_3, the evaluated submission: student_v1's skills and pose estimator, the router bc_v2_r3, the
# orientation references and v1.3's graph flags (reproduce/README.md §7). The same freeze as the original build
# (results/v1_3/build_v1_3.sh with SW=bc_v2_r3 OC=1), without its comparison against student_v1_2 and its serving.
# Output: results/bundles/$NAME/ (+ .tar.gz, .tar.sha256 = the server's artifact_identity).
# Env (defaults: this pipeline's outputs): STUDENT_V1, SWITCHER, ORIENT_REFS, NAME (student_v1_3).
# Usage: reproduce/5_build_v1_3.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
B=${STUDENT_V1:-$R/bundles/student_v1}
SW=${SWITCHER:-$R/student/models/bc_v2_r3/switcher.pt}
REFS=${ORIENT_REFS:-$R/perception/orient_refs.npz}
NAME=${NAME:-student_v1_3}
T4='"recovery": {"seen_s": 15.0, "at_end": true}, "repush_v2": {"max_total_m": 0.1}, "router_repush": false'
OC=()  # the careful place for off-centre grips, on the Tasks 1-3 graphs (the place node it switches exists only there)
for v in task1-2_tower_transfer_2 task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 task2-3_cross_tower_transfer_3 \
         task3-2_vertical_stacking_2 task3-3_vertical_stacking_3; do
  OC+=(--graph_set_variant "$v={\"offcentre_place\": {\"min_mm\": 4.0}}")
done
run "$PY" scripts/freeze_student_bundle.py --teacher "$B" --switcher "$SW" --perception "$B/perception/kp_v2.pt" \
    --name "$NAME" --graph_set '{"idle_chunk": 16, "blind_guard": true, "finished_guard": true}' \
    --graph_set_variant "task4-2_full_tower_pushdown_2={$T4}" \
    --graph_set_variant "task4-3_full_tower_pushdown_3={$T4, \"push_settle\": true}" \
    --graph_set_variant "task5-2_tower_insertion_2={}" --graph_set_variant "task5-3_tower_insertion_3={}" \
    "${OC[@]}" --extra "perception/orient_refs.npz=$REFS" \
    --note "student_v1_3: switcher $(basename "$(dirname "$SW")") (T4 cand, T5 cand), OC 1"
[ "${DRY_RUN:-0}" = 1 ] || run "$PY" reproduce/compare_bundle.py "$R/bundles/$NAME" reproduce/manifests/student_v1_3.json
