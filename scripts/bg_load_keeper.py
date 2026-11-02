#!/usr/bin/env python3
"""Keep a steady GPU load running beside the official endpoint through the evaluation (user, 10-30: every hard reset
came at light or changing load, none in weeks of training). The load is the PPO reaching training when it runs
(its launcher writes its pid to ``results/reach/train.pid``); otherwise scripts/gpu_steady_load.py, started under
scripts/eval_guard.py --no-pause (own scope, memory cap, nice 10, first to be OOM-killed) and stopped as soon as a
training is alive again. Checks every 15 s; ``touch results/reach/STOP_BG`` stops the stand-in and the keeper.

  setsid nohup python3 scripts/bg_load_keeper.py [--endpoint official_v1_3] [--duty 0.5] &
"""
import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REACH = REPO / "results" / "reach"
PY = os.path.expanduser("~/Documents/moves/env_isaaclab/bin/python")

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="official_v1_3")
ap.add_argument("--duty", type=float, default=0.5)
ap.add_argument("--poll_s", type=float, default=15.0)
args = ap.parse_args()
(REACH / "logs").mkdir(parents=True, exist_ok=True)
LOG = open(REACH / "logs" / "bg_load_keeper.log", "a", buffering=1)


def say(msg):
    LOG.write(f"[keeper] {time.strftime('%m-%d %H:%M:%S')} {msg}\n")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def training_pid():
    """The training's pid once it has run for 2 min (Isaac's start-up barely loads the GPU), else None."""
    f = REACH / "train.pid"
    try:
        pid = int(f.read_text().split()[0])
        age = time.time() - f.stat().st_mtime
    except (OSError, ValueError, IndexError):
        return None
    return pid if alive(pid) and age >= 120 else None


standin = None


def start_standin():
    global standin
    cmd = ["python3", str(REPO / "scripts" / "eval_guard.py"), "--name", args.endpoint, "--no-pause", "--mem", "4G",
           "--", PY, str(REPO / "scripts" / "gpu_steady_load.py"), "--duty", str(args.duty)]
    out = open(REACH / "logs" / "gpu_steady_load.log", "a")
    standin = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    say(f"stand-in load started (guard pid {standin.pid}, duty {args.duty})")


def stop_standin(why):
    global standin
    if standin is None:
        return
    if standin.poll() is None:
        os.killpg(standin.pid, signal.SIGINT)  # the guard passes it on; a second one kills the group
        for _ in range(20):
            if standin.poll() is not None:
                break
            time.sleep(0.5)
        if standin.poll() is None:
            os.killpg(standin.pid, signal.SIGKILL)
    say(f"stand-in load stopped ({why})")
    standin = None


say(f"start (endpoint {args.endpoint})")
last = None
while True:
    if (REACH / "STOP_BG").exists():
        stop_standin("STOP_BG")
        say("exit (STOP_BG)")
        break
    tp = training_pid()
    if tp:
        if last != "training":
            say(f"training pid {tp} is the load")
        stop_standin(f"training pid {tp} running")
        last = "training"
    else:
        if standin is None or standin.poll() is not None:
            if standin is not None:
                say(f"stand-in exited with {standin.returncode}; restarting")
                standin = None
            start_standin()
        last = "standin"
    time.sleep(args.poll_s)
