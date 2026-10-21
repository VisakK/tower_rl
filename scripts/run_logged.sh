#!/usr/bin/env bash
# Run a command with its output in <log> and GPU telemetry (power, temperature, memory, clocks) sampled every
# second into <log>.gpu.csv, so a hard reset leaves a trace of what the GPU was doing.
#
# Usage: scripts/run_logged.sh <log file> <command> [args...]
set -u
log="$1"; shift
mkdir -p "$(dirname "$log")"
nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks.sm \
    --format=csv,noheader -l 1 >> "${log}.gpu.csv" 2>/dev/null &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
"$@" > "$log" 2>&1
status=$?
echo "[run_logged] exit status $status" >> "$log"
exit $status
