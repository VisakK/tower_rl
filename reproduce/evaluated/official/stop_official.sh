#!/usr/bin/env bash
# Stop an official endpoint (serve_official.sh): Ctrl-C to the server (it writes its episode summaries and exits) and to
# the tunnel, then close its tmux session (default "tower" = official_1; SESSION=<name> for another endpoint). Only
# after its evaluation shows "evaluated" on the challenge site and no retry is pending.
SESSION=${SESSION:-tower}
tmux has-session -t "$SESSION" 2>/dev/null || { echo "no tmux session '$SESSION'"; exit 0; }
tmux send-keys -t "$SESSION":server C-c
tmux send-keys -t "$SESSION":tunnel C-c
sleep 5
tmux kill-session -t "$SESSION"
echo "stopped $SESSION"
