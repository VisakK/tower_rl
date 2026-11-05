#!/usr/bin/env bash
# Studentv1_3_plan Step 8: freeze student_v1_3 = student_v1's skills and perception + the candidate switcher + v1.3's
# graph flags, check it against student_v1_2 (skills / perception / orientation references byte-identical; only the
# switcher files and the listed graph flags differ), then (unless --no-serve) serve it as official_v1_3: snapshot
# server_v1_3, port 9004, tmux tower_v1_3, its own quick tunnel; probes locally and through the URL.
# Env: SW=<candidate switcher.pt> (required); OC=1 adds offcentre_place (Tasks 1-3); T4=cand|bc_r2 and T5=cand|bc_r2
# choose the router of those families (bc_r2 = v1.2's, through a per-variant "switcher" key).
# official_v1_2 (9003, tower_v1_2) is not touched.
# Usage: SW=... [OC=1] [T4=bc_r2] [T5=bc_r2] build_v1_3.sh [--no-serve]
set -eu
T=/home/visakii/Documents/tower_rl
R=$T/results
SNAP=/home/visakii/Documents/tower_rl_snapshots/server_v1_3
B=$R/bundles/student_v1
SW=${SW:?SW=<candidate switcher.pt>}
cd $T
source ~/Documents/moves/env_isaaclab/bin/activate
CORE='"idle_chunk": 16, "blind_guard": true, "finished_guard": true'
OCV=()  # offcentre_place on the Tasks 1-3 graphs only (the place node it switches exists only there; as gated)
if [ "${OC:-0}" = 1 ]; then
  for v in task1-2_tower_transfer_2 task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 task2-3_cross_tower_transfer_3 \
           task3-2_vertical_stacking_2 task3-3_vertical_stacking_3; do
    OCV+=(--graph_set_variant "$v={\"offcentre_place\": {\"min_mm\": 4.0}}")
  done
fi
T4F='"recovery": {"seen_s": 15.0, "at_end": true}, "repush_v2": {"max_total_m": 0.1}, "router_repush": false'
OLD='"switcher": "student/switcher_bc_r2.pt"'
V42="{$T4F}"; V43="{$T4F, \"push_settle\": true}"; V5="{}"
EXTRA=(--extra perception/orient_refs.npz=$R/perception/orient_refs.npz)
if [ "${T4:-cand}" = bc_r2 ]; then V42="{$T4F, $OLD}"; V43="{$T4F, \"push_settle\": true, $OLD}"; fi
if [ "${T5:-cand}" = bc_r2 ]; then V5="{$OLD}"; fi
if [ "${T4:-cand}" = bc_r2 ] || [ "${T5:-cand}" = bc_r2 ]; then
  EXTRA+=(--extra student/switcher_bc_r2.pt=$B/student/switcher.pt --extra student/switcher_bc_r2.pt.json=$B/student/switcher.pt.json)
fi
[ -d $SNAP ] || scripts/make_snapshot.sh server_v1_3
if [ ! -d $R/bundles/student_v1_3 ]; then
  python scripts/freeze_student_bundle.py --teacher $B --switcher $SW --perception $B/perception/kp_v2.pt \
      --name student_v1_3 --graph_set "{$CORE}" \
      --graph_set_variant "task4-2_full_tower_pushdown_2=$V42" --graph_set_variant "task4-3_full_tower_pushdown_3=$V43" \
      --graph_set_variant "task5-2_tower_insertion_2=$V5" --graph_set_variant "task5-3_tower_insertion_3=$V5" \
      "${OCV[@]}" "${EXTRA[@]}" --note "student_v1_3: switcher $(basename $(dirname $SW)) (T4 ${T4:-cand}, T5 ${T5:-cand}), OC ${OC:-0}"
fi
cat $R/bundles/student_v1_3.tar.sha256
python - <<'PY'
import filecmp, json, pathlib
root = pathlib.Path("/home/visakii/Documents/tower_rl/results/bundles")
a, b = root / "student_v1_2", root / "student_v1_3"
fa = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
fb = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
sw = {p for p in fa ^ fb} | {p for p in fa & fb if p.parts[0] == "student" and p.name.startswith("switcher")}
assert all(p.parts[0] == "student" and p.name.startswith("switcher") for p in fa ^ fb), fa ^ fb
diff = {}
for rel in sorted(fa & fb):
    if rel.parts[0] == "graphs":
        ga, gb = json.loads((a / rel).read_text()), json.loads((b / rel).read_text())
        d = {k: gb.get(k) for k in set(ga) | set(gb) if ga.get(k) != gb.get(k)}
        if d:
            diff[rel.stem] = d
    elif rel.name != "MANIFEST.json" and rel not in sw:
        assert filecmp.cmp(a / rel, b / rel, shallow=False), rel
print("bundle check ok: skills / perception / references identical to student_v1_2; switcher files:",
      sorted(str(p) for p in sw))
for v, d in diff.items():
    print("  graph", v, "differs by", json.dumps(d))
PY
[ "${1:-}" = "--no-serve" ] && exit 0
PORT=9004 SESSION=tower_v1_3 SNAP=$SNAP BUNDLE=$R/bundles/student_v1_3 \
    EXTRA="--policy_id tower-student-v1-3 --plugin_version 2" $R/student/serve_official.sh official_v1_3
URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' $R/student/logs/official_v1_3_tunnel.log | head -1)
cd $SNAP
python scripts/probe_server.py ws://127.0.0.1:9004 --track data/challenge/track.json --out $R/v1_3/official_v1_3_probe_local.json \
    > /dev/null 2>&1 && echo "local probe ok" || echo "local probe FAILED"
python scripts/probe_server.py "wss://${URL#https://}" --track data/challenge/track.json \
    --out $R/v1_3/official_v1_3_probe_tunnel.json > /dev/null 2>&1 && echo "tunnel probe ok" || echo "tunnel probe FAILED (retry later)"
echo "MODEL_URL wss://${URL#https://}"
