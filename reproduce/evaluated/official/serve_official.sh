#!/usr/bin/env bash
# The official evaluation's endpoint (student plan Step 11, G11b): the student server (a frozen code snapshot, the
# downloaded track.json, camera_source evaluator) and a Cloudflare quick tunnel to it, each in a window of a tmux
# session. systemd-inhibit keeps the machine from sleeping while the server runs; the tunnel does not auto-update (a
# restart would change its URL). Prints the model URL to register on the challenge site.
# Usage: results/student/serve_official.sh <name>
#   defaults = official_1's endpoint (student_v1); a second endpoint beside it (Plan-v1.1 §3.1) sets, e.g.:
#   PORT=9002 SESSION=tower_v1_1 SNAP=~/Documents/tower_rl_snapshots/server_v1_1 BUNDLE=.../results/bundles/student_v1_1 \
#     EXTRA="--policy_id tower-student-v1-1 --plugin_version 2" results/student/serve_official.sh official_v1_1
#   logs: results/student/server/<name>/ (server.jsonl, per-episode call logs), results/student/logs/<name>_{server,tunnel}.log
#   watch: tmux attach -t <session> (detach: Ctrl-b d); stop: SESSION=<session> results/student/stop_official.sh
set -u
NAME=${1:?usage: serve_official.sh <name>}
R=/home/visakii/Documents/tower_rl/results
PORT=${PORT:-9001}
SESSION=${SESSION:-tower}
SNAP=${SNAP:-/home/visakii/Documents/tower_rl_snapshots/server_v1}
BUNDLE=${BUNDLE:-$R/bundles/student_v1}
EXTRA=${EXTRA:-}
L=$R/student/logs
CF=/home/visakii/.local/opt/cloudflared/cloudflared
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session '$SESSION' already exists (tmux attach -t $SESSION)"; exit 1; fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "port $PORT is in use"; exit 1; fi
tmux new-session -d -s "$SESSION" -n server "cd $SNAP && source ~/Documents/moves/env_isaaclab/bin/activate && \
  systemd-inhibit --what=sleep:idle --who=$SESSION --why='TOWER evaluation' \
  python scripts/serve_student.py --track data/challenge/track.json --bundle $BUNDLE --name $NAME --port $PORT $EXTRA \
  2>&1 | tee $L/${NAME}_server.log; exec bash"
for i in $(seq 1 120); do
  grep -q READY $L/${NAME}_server.log 2>/dev/null && break
  if grep -qE "Traceback|SystemExit|Error:" $L/${NAME}_server.log 2>/dev/null; then echo "server failed:"; tail -20 $L/${NAME}_server.log; exit 1; fi
  sleep 1
done
grep READY $L/${NAME}_server.log || { echo "server not ready after 120 s"; exit 1; }
tmux new-window -t "$SESSION" -n tunnel "$CF tunnel --no-autoupdate --url http://127.0.0.1:$PORT 2>&1 | tee $L/${NAME}_tunnel.log; exec bash"
for i in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' $L/${NAME}_tunnel.log 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "no tunnel URL after 60 s:"; tail -20 $L/${NAME}_tunnel.log; exit 1; }
echo "model URL: wss://${URL#https://}"
