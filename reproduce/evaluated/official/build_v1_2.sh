#!/usr/bin/env bash
# student_v1_2 = student_v1_1 + graph flags T4 on the two Task 4 graphs (default: the student's recovery node with
# the sporadic-sighting gate and the fixed re-push, BUILDLOG 2026-10-28). Same networks, orientation references and
# v1.1 flags; push_settle stays on 4-3 only. Freeze, check against student_v1_1 (networks byte-identical, graphs differ
# only by the T4 flags on Task 4), then serve beside the other endpoints: server code snapshot server_v1_2, port 9003,
# tmux session tower_v1_2, its own quick tunnel; the qualification-like probe locally and through the URL.
# official_1 (9001, tower) and official_v1_1 (9002, tower_v1_1) are not touched.
# Usage: [T4='<JSON>'] build_v1_2.sh [--no-serve]
set -eu
T=/home/visakii/Documents/tower_rl
R=$T/results
SNAP=/home/visakii/Documents/tower_rl_snapshots/server_v1_2
cd $T
source ~/Documents/moves/env_isaaclab/bin/activate
B=$R/bundles/student_v1
T4=${T4:-'{"recovery": {"seen_s": 15.0, "at_end": true}, "repush_v2": {"max_total_m": 0.1}}'}
T43=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['push_settle']=True; print(json.dumps(d))" "$T4")
[ -d $SNAP ] || scripts/make_snapshot.sh server_v1_2
if [ ! -d $R/bundles/student_v1_2 ]; then
  python scripts/freeze_student_bundle.py --teacher $B --switcher $B/student/switcher.pt --perception $B/perception/kp_v2.pt \
      --name student_v1_2 --graph_set '{"idle_chunk": 16, "blind_guard": true, "finished_guard": true}' \
      --graph_set_variant "task4-2_full_tower_pushdown_2=$T4" \
      --graph_set_variant "task4-3_full_tower_pushdown_3=$T43" \
      --extra perception/orient_refs.npz=$R/perception/orient_refs.npz \
      --note "student_v1_2: student_v1_1 + $T4 on the Task 4 graphs"
fi
cat $R/bundles/student_v1_2.tar.sha256
T4="$T4" python - <<'EOF'
import filecmp, json, os, pathlib
a, b = (pathlib.Path("/home/visakii/Documents/tower_rl/results/bundles") / n for n in ("student_v1_1", "student_v1_2"))
fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
assert fa == fb, ("file lists differ", set(fa) ^ set(fb))
for rel in fa:
    if rel.parts[0] == "graphs":
        ga, gb = json.loads((a / rel).read_text()), json.loads((b / rel).read_text())
        extra = {k: gb[k] for k in gb if ga.get(k) != gb.get(k)}
        missing = [k for k in ga if k not in gb]
        want = json.loads(os.environ["T4"]) if rel.stem.startswith("task4") else {}
        assert extra == want and not missing, (rel, extra, missing)
    elif rel.name != "MANIFEST.json":
        assert filecmp.cmp(a / rel, b / rel, shallow=False), rel
print("bundle check ok: networks / perception identical to student_v1_1; graphs differ only by", os.environ["T4"], "on Task 4")
EOF
[ "${1:-}" = "--no-serve" ] && exit 0
PORT=9003 SESSION=tower_v1_2 SNAP=$SNAP BUNDLE=$R/bundles/student_v1_2 \
    EXTRA="--policy_id tower-student-v1-2 --plugin_version 2" $R/student/serve_official.sh official_v1_2
URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' $R/student/logs/official_v1_2_tunnel.log | head -1)
cd $SNAP
python scripts/probe_server.py ws://127.0.0.1:9003 --track data/challenge/track.json --out $R/v1_2/official_v1_2_probe_local.json \
    > /dev/null 2>&1 && echo "local probe ok" || echo "local probe FAILED"
sleep 20  # the quick tunnel's DNS
python scripts/probe_server.py "wss://${URL#https://}" --track data/challenge/track.json \
    --out $R/v1_2/official_v1_2_probe_tunnel.json > /dev/null 2>&1 && echo "tunnel probe ok" || echo "tunnel probe FAILED (retry later)"
echo "MODEL_URL wss://${URL#https://}"
