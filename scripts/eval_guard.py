"""Run an experiment next to the official endpoint (``serve_official.sh``) without putting the evaluation at risk.

The server and the tunnel live in VS Code's cgroup (the scope of the shell that started them), and systemd-oomd kills
a whole cgroup under memory pressure, so a job started from a VS Code terminal or Claude Code would take the endpoint
down with it. The command therefore runs in its own systemd user scope with a memory cap and no swap (the kernel then
kills it, not the endpoint, when it outgrows the cap), at nice 10 and with the highest OOM-kill priority.

While the evaluation runs, the command is paused: SIGSTOP to its process group as soon as a session opens on the
server or a status line shows a connected session or calls, SIGCONT once that has stayed quiet for ``--idle-min``.
The server needs room on the GPU for its first real inference and a paused process keeps its memory, so the command
is killed if free GPU memory falls below ``--min-free-gb`` while the server runs (and not started if it already is).
With the endpoint down it only runs the command in its scope. Pauses count against wall-clock timeouts inside it.
If the guard itself is killed while the command is paused, ``kill -CONT -<pgid>`` (the pid it printed) resumes it.

  python3 scripts/eval_guard.py [--name official_1] [--mem 36G] [--idle-min 20] [--min-free-gb 2] [--no-pause] -- <command ...>

``--no-pause`` (CPU-only jobs that must keep going through an evaluation): never pause; the command still gets its own
scope with the memory cap, nice 10 and the highest OOM-kill priority, so memory pressure cannot take the endpoint down.

Ctrl-C resumes the command and passes the signal on; a second Ctrl-C kills its process group. Lines go to stdout
and to ``results/student/logs/eval_guard.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATUS = re.compile(r"\[server\] \S+ sessions \d+ \(connected (\d+)\) calls \+(\d+)")


class Endpoint:
    """The official server as its logs show it (read only)."""

    def __init__(self, results: Path, name: str):
        self.status_log = results / f"student/logs/{name}_server.log"
        self.events = results / f"student/server/{name}/server.jsonl"
        run = results / f"student/server/{name}/run.json"
        self.pid = json.loads(run.read_text()).get("pid") if run.exists() else None
        self.pos = self.events.stat().st_size if self.events.exists() else 0  # sessions before the guard don't count

    def up(self) -> bool:
        return self.pid is not None and Path(f"/proc/{self.pid}").exists()

    def opened(self) -> bool:
        """Whether a session opened (or an episode reset) since the last call."""
        if not self.events.exists():
            return False
        seen = False
        with open(self.events, "rb") as f:
            f.seek(self.pos)
            for line in f:
                if not line.endswith(b"\n"):  # a row being written
                    break
                self.pos += len(line)
                try:
                    seen |= json.loads(line).get("kind") in ("session_open", "reset")
                except ValueError:
                    pass
        return seen

    def busy(self) -> bool:
        """Whether the latest status line (one a minute) shows a connected session or calls."""
        try:
            size = self.status_log.stat().st_size
            with open(self.status_log, "rb") as f:
                f.seek(max(0, size - 16384))
                tail = f.read().decode(errors="replace")
        except OSError:
            return False
        m = STATUS.findall(tail)
        return bool(m) and (int(m[-1][0]) > 0 or int(m[-1][1]) > 0)


def gpu_free_gb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return min(float(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="official_1", help="the endpoint (serve_official.sh <name>)")
    ap.add_argument("--mem", default="36G", help="memory cap of the command's scope (MemoryMax)")
    ap.add_argument("--idle-min", type=float, default=20.0, help="resume after this long without evaluation activity")
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="GPU memory the server must keep free (0: off)")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--no-pause", action="store_true", help="never pause the command (CPU-only jobs)")
    ap.add_argument("--results", default=str(REPO / "results"), help="results directory (tests use a fake one)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command ...>")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command (put it after --)")
    results = Path(args.results)
    logf = open(results / "student/logs/eval_guard.log", "a")

    def log(msg):
        line = f"[guard {os.getpid()}] {datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    ep = Endpoint(results, args.name)
    free = gpu_free_gb() if args.min_free_gb > 0 and ep.up() else None
    if free is not None and free < args.min_free_gb:
        log(f"free GPU memory {free:.1f} GB < {args.min_free_gb:g} GB already: not starting {' '.join(cmd)}")
        sys.exit(1)

    def lowest_priority():
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # a guard started in the background inherits SIGINT ignored
        os.nice(10)
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")

    unit = f"guarded-{os.getpid()}"
    scope = ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}",
             "-p", f"MemoryMax={args.mem}", "-p", "MemorySwapMax=0", "--"]
    child = subprocess.Popen(scope + cmd, start_new_session=True, preexec_fn=lowest_priority)
    pgid = child.pid

    def to_group(sig):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:  # the command has just exited
            pass

    log(f"started pid {pgid} in {unit}.scope (MemoryMax {args.mem}): {' '.join(cmd)}; "
        f"endpoint {args.name} {'up (pid %d)' % ep.pid if ep.up() else 'down'}" + ("; no pause" if args.no_pause else ""))

    paused, quiet_since, interrupts, last_gpu = False, None, 0, 0.0

    def on_signal(sig, _frame):
        nonlocal paused, interrupts
        interrupts += 1
        if interrupts > 1:
            log("second interrupt: killing the process group")
            to_group(signal.SIGKILL)
            return
        log(f"{signal.Signals(sig).name}: resuming and passing it on")
        to_group(signal.SIGCONT)
        paused = False
        to_group(sig)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    while child.poll() is None:
        now = time.time()
        opened, busy = ep.opened(), ep.busy()
        if not args.no_pause and interrupts == 0 and ep.up() and (opened or busy):
            quiet_since = None
            if not paused:
                to_group(signal.SIGSTOP)
                paused = True
                log("evaluation activity (" + ("a session opened" if opened else "connected sessions") + "): paused")
        elif paused:
            quiet_since = quiet_since or now
            if now - quiet_since >= args.idle_min * 60:
                to_group(signal.SIGCONT)
                paused = False
                log(f"no evaluation activity for {args.idle_min:g} min"
                    + ("" if ep.up() else " (endpoint down)") + ": resumed")
        if args.min_free_gb > 0 and ep.up() and now - last_gpu >= 15:
            last_gpu = now
            free = gpu_free_gb()
            if free is not None and free < args.min_free_gb:
                log(f"free GPU memory {free:.1f} GB < {args.min_free_gb:g} GB: killing the process group")
                to_group(signal.SIGKILL)
                child.wait()
                break
        time.sleep(args.poll_s)
    log(f"command exited with {child.returncode}")
    sys.exit(child.returncode if child.returncode >= 0 else 128 - child.returncode)


if __name__ == "__main__":
    main()
