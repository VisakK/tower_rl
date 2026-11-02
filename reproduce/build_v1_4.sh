#!/usr/bin/env bash
# Freeze student_v1_4 (v1.3 + the learned reachers; built after the deadline, never evaluated) or, with
# REACH_CFG=configs/reach/teacher_p2f_back9200.json NAME=student_v1_5, student_v1_5 (back reacher 9200 instead of 8500).
# student_v1_3's files unchanged, the two reacher exports (with their JSON) in reach/, and the config's reach block
# (model paths made bundle-relative) merged into every graph. The same freeze as results/v1_4/build_v1_4.sh and
# results/v1_5/build_v1_5.sh, without their checks against the other bundles. Reacher training: docs/Reacher-V3.MD.
# Env: STUDENT_V1_3 (default results/bundles/student_v1_3), REACH_CFG (default configs/reach/teacher_p2f.json),
#      NAME (default student_v1_4).
# Usage: reproduce/build_v1_4.sh    |    REACH_CFG=configs/reach/teacher_p2f_back9200.json NAME=student_v1_5 reproduce/build_v1_4.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
A=${STUDENT_V1_3:-$R/bundles/student_v1_3}
CFG=${REACH_CFG:-configs/reach/teacher_p2f.json}
NAME=${NAME:-student_v1_4}
GSET=$("$PY" - "$CFG" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())["reach"]
r["policy"] = {arm: "reach/" + Path(p).name for arm, p in r["policy"].items()}
print(json.dumps({"reach": r}))
PY
)
POL=$("$PY" -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['reach']['policy'].values()))" "$CFG")
EXTRA=(--extra "perception/orient_refs.npz=$A/perception/orient_refs.npz")
for p in $POL; do
  for f in "$p" "${p%.pt}.json"; do EXTRA+=(--extra "reach/$(basename "$f")=$REPO/$f"); done
done
run "$PY" scripts/freeze_student_bundle.py --teacher "$A" --switcher "$A/student/switcher.pt" \
    --perception "$A/perception/kp_v2.pt" --name "$NAME" --graph_set "$GSET" "${EXTRA[@]}" \
    --note "$NAME: student_v1_3 + the chunk-trained reachers ($(basename "$CFG"): ${POL// /, })"
REF=reproduce/manifests/$NAME.json
if [ "${DRY_RUN:-0}" != 1 ] && [ -f "$REF" ]; then run "$PY" reproduce/compare_bundle.py "$R/bundles/$NAME" "$REF"; fi
