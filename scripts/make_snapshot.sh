#!/usr/bin/env bash
# A frozen code copy for long jobs and servers (each Isaac process reads the code at its start, and lazily imported
# modules would pick up later edits): tower_rl/, scripts/, configs/ and pytest.ini copied; data/, assets_m/ and results/
# symlinked into the repository. Usage: scripts/make_snapshot.sh <name>  ->  ~/Documents/tower_rl_snapshots/<name>
set -eu
NAME=${1:?usage: make_snapshot.sh <name>}
SRC=/home/visakii/Documents/tower_rl
DST=/home/visakii/Documents/tower_rl_snapshots/$NAME
[ -e "$DST" ] && { echo "$DST exists"; exit 1; }
mkdir -p "$DST"
cp -r "$SRC/tower_rl" "$SRC/scripts" "$SRC/configs" "$DST/"
cp "$SRC/pytest.ini" "$DST/" 2>/dev/null || true
find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} +
for d in data assets_m results; do ln -s "$SRC/$d" "$DST/$d"; done
echo "$DST"
