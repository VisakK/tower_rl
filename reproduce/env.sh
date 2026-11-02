# Sourced by every reproduce/ script: the repository root, the results directory and the Python interpreter.
#   PYTHON   the interpreter of your Isaac Sim 5.1 + Isaac Lab 2.3 environment (default: `python` on PATH, i.e. run the
#            scripts with that environment activated)
#   DRY_RUN  1: print the commands instead of running them
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=$REPO/results
PY=${PYTHON:-python}
cd "$REPO"

run() {
  echo "+ $*"
  if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi
}

# newest run dir of a skill whose name ends in _<tag> (train_skill.py names runs <date>_<time>_<config hash>_<tag>)
latest_run() {
  ls -d "$R/skills/$1"/*_"$2" 2>/dev/null | sort | tail -1
}
