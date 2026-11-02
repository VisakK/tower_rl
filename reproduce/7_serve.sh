#!/usr/bin/env bash
# Serve a frozen bundle on the challenge protocol (tower-infer-v1, the challenge kit's WebSocket server) at
# ws://127.0.0.1:$PORT, the way student_v1_3 was served for its evaluation (reproduce/README.md §9). The challenge needs a
# public wss:// URL on port 443, e.g. a Cloudflare quick tunnel in a second terminal:
#   cloudflared tunnel --no-autoupdate --url http://127.0.0.1:$PORT
# Check it as the challenge does (every variant, resets, blank images) with
#   python scripts/probe_server.py ws://127.0.0.1:$PORT --track $TRACK --out results/student/probe.json
# Env: BUNDLE (default results/bundles/student_v1_3), TRACK (default data/challenge/track.json, downloaded from the
#      challenge site after sign-in: all ten tasks, 10 Hz), PORT (9004), POLICY_ID (tower-student-v1-3),
#      NAME (log dir under results/student/server/, default local_<bundle>).
# The artifact_identity is the bundle tar's sha256 (results/bundles/<bundle>.tar.sha256).
set -euo pipefail
source "$(dirname "$0")/env.sh"
BUNDLE=${BUNDLE:-$R/bundles/student_v1_3}
run "$PY" scripts/serve_student.py --track "${TRACK:-data/challenge/track.json}" --bundle "$BUNDLE" \
    --name "${NAME:-local_$(basename "$BUNDLE")}" --port "${PORT:-9004}" --policy_id "${POLICY_ID:-tower-student-v1-3}" \
    --plugin_version 2
