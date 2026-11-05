#!/usr/bin/env bash
# Serve one evaluated submission the way it ran: its tagged code, its bundle tar, its recorded flags. The code comes from
# a git worktree of the tag beside this repository, so nothing in this checkout changes. The server does not need Isaac.
# Usage: reproduce/evaluated/serve_evaluated.sh <student_v1|student_v1_2|student_v1_3> <bundle tar> <track.json>
# Env: PYTHON   interpreter with reproduce/environment/requirements-inference.txt (default: python on PATH)
#      PORT     default: the port of the official run
#      WORK     the worktree, default ../tower_rl_eval_<submission>
#      NAME     log directory under the worktree's results/student/server/, default review
#      DRY_RUN  1: check the tar, print the commands, run nothing
# Then, from the worktree in a second terminal, the qualification-like checks:
#      python scripts/probe_server.py ws://127.0.0.1:<port> --track <track.json>
set -euo pipefail
S=${1:?usage: serve_evaluated.sh <student_v1|student_v1_2|student_v1_3> <bundle tar> <track.json>}
TAR=$(readlink -f "${2:?bundle tar}")
TRACK=$(readlink -f "${3:?track.json}")
case "$S" in
  student_v1)   SHA=01ab88da17d7f9003bde56ff4db1c71c885298598994eb3dd0f6f031fd4bccdd; POLICY=tower-student-v1;   PLUGIN=1; DEF_PORT=9001 ;;
  student_v1_2) SHA=f726c71a453da23943757cf9fb8506d101f83b196b22770f966073a5c3279aa9; POLICY=tower-student-v1-2; PLUGIN=2; DEF_PORT=9003 ;;
  student_v1_3) SHA=76ec7f3d35c4da7b41dc74c26ce2f8484287f87cc4023f3820dc4145a0c8263d; POLICY=tower-student-v1-3; PLUGIN=2; DEF_PORT=9004 ;;
  *) echo "unknown submission $S"; exit 2 ;;
esac
source "$(dirname "$(readlink -f "$0")")/../env.sh"
got=$(sha256sum "$TAR" | cut -c1-64)
[ "$got" = "$SHA" ] || { echo "$TAR: sha256 $got, expected $SHA"; exit 1; }
echo "tar ok: $SHA"
WORK=$(readlink -m "${WORK:-$REPO/../tower_rl_eval_$S}")
if [ ! -e "$WORK" ]; then
  git rev-parse -q --verify "refs/tags/eval/$S" > /dev/null || run git fetch origin --tags
  run git worktree add --detach "$WORK" "eval/$S"
fi
run mkdir -p "$WORK/results/bundles"
run tar xzf "$TAR" -C "$WORK/results/bundles"
if [ "${DRY_RUN:-0}" != 1 ]; then cd "$WORK"; fi
run "$PY" scripts/serve_student.py --track "$TRACK" --bundle "results/bundles/$S" --name "${NAME:-review}" \
    --port "${PORT:-$DEF_PORT}" --policy_id "$POLICY" --plugin_version "$PLUGIN" --artifact_identity "sha256:$SHA"
