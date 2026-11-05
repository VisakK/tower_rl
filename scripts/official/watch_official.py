#!/usr/bin/env python3
"""Watch the official_1 server during the evaluation; one stdout line per event worth acting on.

Emits: the evaluation's first session; every error-kind event (error, adapter_error, prepare_failed, summary_failed,
duplicate_request, an evaluation session's expiry); tunnel (un)registrations and errors; tracebacks in the server
log; the server process dying; a progress line every PERIOD s while the evaluation runs; and a line when no session
has been connected for IDLE s after the evaluation started (the likely end).

  python3 watch_official.py [START_ISO] [PERIOD_S] [IDLE_S] [first]
  "first": exit after the first line (a one-shot wait). Env: WATCH_NAME=<endpoint> (default official_1; the v1.1
  endpoint is official_v1_1; its tunnel's metrics port is read from its tunnel log), WATCH_QUIET_START=1 skips the
  start line and earlier error events (when re-armed; they are still counted),
  WATCH_RESUME=1 reports the first live session (a paused run resuming). Do not pipe it through grep: grep here is
  ugrep 7.8.4, which held lines back even with --line-buffered.
"""
import faulthandler
import json
import signal
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/visakii/Documents/tower_rl/scripts")
from server_episodes import episodes as server_episodes  # noqa: E402

R = Path("/home/visakii/Documents/tower_rl/results/student")
NAME = os.environ.get("WATCH_NAME", "official_1")
JSONL = R / f"server/{NAME}/server.jsonl"
TUNNEL = R / f"logs/{NAME}_tunnel.log"
SLOG = R / f"logs/{NAME}_server.log"
PID = json.loads((R / f"server/{NAME}/run.json").read_text())["pid"]
_m = re.search(r"metrics server on (127\.0\.0\.1:\d+)", TUNNEL.read_text()) if TUNNEL.exists() else None
METRICS = f"http://{_m.group(1) if _m else '127.0.0.1:20241'}/ready"
START = sys.argv[1] if len(sys.argv) > 1 else "2026-10-24T11:10:00"  # earlier events: our probe and the check
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 900
IDLE = float(sys.argv[3]) if len(sys.argv) > 3 else 600
BAD = {"error", "adapter_error", "prepare_failed", "summary_failed", "duplicate_request"}
STATUS = re.compile(r"\[server\] (\S+) sessions (\d+) \(connected (\d+)\) calls \+(\d+) errors (\d+) mean call (\S+) ms")


def tunnel_ready():
    try:
        with urllib.request.urlopen(METRICS, timeout=5) as r:
            return json.loads(r.read()).get("readyConnections", 0) > 0
    except Exception:  # noqa: BLE001
        return False


def tally():
    """Server-side outcomes so far: early / limit (210 s) / open, per task and in all (officially not success or
    failure: official episodes run to the limit even after success)."""
    try:
        eps = server_episodes(R / f"server/{NAME}", START, 2100, 0.1)
    except Exception as ex:  # noqa: BLE001
        return f"tally failed: {ex!r}"
    per = {}
    for e in eps:
        per.setdefault(e["variant"][4:7], Counter())[e["end"]] += 1
    c = Counter(e["end"] for e in eps)
    return (f"early {c['early']} / limit {c['limit']} / open {c['open']} / empty {c['empty']}; "
            + " ".join(f"{v} {x['early']}/{x['limit']}/{x['open']}" for v, x in sorted(per.items())))


FIRST = len(sys.argv) > 4 and sys.argv[4] == "first"  # exit after the first event (a one-shot wait)


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
    if FIRST:
        sys.exit(0)


class Tail:
    def __init__(self, path, from_start=False):
        self.path, self.pos, self.buf = path, 0 if from_start else (path.stat().st_size if path.exists() else 0), ""

    def lines(self):
        if not self.path.exists():
            return []
        with open(self.path, errors="replace") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        self.buf += data
        *out, self.buf = self.buf.split("\n")
        return out


T0 = time.strftime("%Y-%m-%dT%H:%M:%S")  # live events are newer than this
faulthandler.register(signal.SIGUSR1)  # kill -USR1 <pid>: stack to stderr
ev, tun, slog = Tail(JSONL, from_start=True), Tail(TUNNEL), Tail(SLOG)
opened, closed = {}, set()  # evaluation sessions -> open time
n = {"session_open": 0, "reset": 0, "episode_end": 0, "disconnect": 0, "bad": 0, "expired": 0}
calls, connected, mean_ms, last_status = 0, 0, "-", None
started, last_progress, idle_since, idle_said = False, 0.0, None, False
ep_calls = []
ready, last_ready_check = True, 0.0
probes, probe_said = [], False
last_status_seen = time.time()
tb_until = [0.0]


def handshake_tb(line):
    """Traceback lines that belong to a refused handshake (they follow 'opening handshake failed')."""
    return time.time() < tb_until[0]


while True:
    for line in ev.lines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("time", "") < START:
            continue
        k = r.get("kind")
        # re-armed (WATCH_QUIET_START): count earlier events, but report only live ones
        live = r.get("time", "") >= T0 or not os.environ.get("WATCH_QUIET_START")
        if k == "session_open":
            opened[r["session"]] = r["time"]
            n[k] += 1
            if os.environ.get("WATCH_RESUME") and r["time"] >= T0:  # a live session while waiting for a paused run
                say(f"RESUMED: session {r['time'][11:19]} variant binding={r.get('binding')}")
            if not started:
                started, last_progress = True, time.time()
                if not os.environ.get("WATCH_QUIET_START"):  # set when re-arming: the start was reported already
                    say(f"EVALUATION STARTED: first session {r['time'][11:19]} binding={r.get('binding')}")
        elif k in ("reset", "disconnect"):
            n[k] += 1
            if k == "disconnect":
                closed.add(r.get("session"))
        elif k == "episode_end":
            n[k] += 1
            ep_calls.append(r.get("calls", 0))
            if r.get("errors") and live:
                say(f"episode with errors: {r.get('variant')} trial={r.get('trial')} calls={r.get('calls')} errors={r.get('errors')} holds={r.get('holds')}")
        elif k == "session_expired" and r.get("session") in opened:
            n["expired"] += 1
            if r.get("session") not in closed and live:  # expiry after a disconnect is the normal cleanup
                say(f"EVALUATION SESSION EXPIRED WITHOUT DISCONNECT {r['session']} (opened {opened[r['session']][11:19]})")
        if k in BAD:
            n["bad"] += 1
            if live and (n["bad"] <= 20 or n["bad"] % 50 == 0):
                say(f"SERVER {k.upper()} #{n['bad']}: " + json.dumps({a: b for a, b in r.items() if a not in ("trace", "time", "kind")})[:400])
    for line in tun.lines():
        if re.search(r"icmp|ping_group|receive buffer", line, re.I):
            continue
        if "Unable to reach the origin service" in line or "Request failed" in line:
            # a non-WebSocket request (e.g. an HTTP HEAD probe) that the server refused; counted, reported if frequent
            probes.append(time.time())
            probes[:] = [x for x in probes if time.time() - x < 600]
            if len(probes) >= 20 and not probe_said:
                probe_said = True
                say(f"TUNNEL origin errors: {len(probes)} in 10 min (refused non-WebSocket requests?) last: {line[:200]}")
            continue
        if re.search(r" ERR |Unregistered|Registered tunnel|Retrying|terminated|shutting down|Lost connection", line):
            say("TUNNEL " + line[:300])
    for line in slog.lines():
        if "opening handshake failed" in line:
            tb_until[0] = time.time() + 10.0
        m = STATUS.search(line)
        if m:
            last_status_seen = time.time()
            connected, mean_ms, last_status = int(m.group(3)), m.group(6), m.group(1)
            if started:
                calls += int(m.group(4))
            continue
        if "opening handshake failed" in line or "did not receive a valid HTTP request" in line or "unsupported HTTP method" in line:
            continue  # a refused non-WebSocket request: websockets logs a traceback; harmless
        if re.search(r"Traceback|Exception|out of memory|CUDA error|Killed|Segmentation", line):
            if handshake_tb(line):
                continue
            say("SERVER LOG " + line[:300])
    if time.time() - last_status_seen > 200:
        say(f"SERVER STATUS STALE: no status line for {time.time() - last_status_seen:.0f} s (hung?)")
        last_status_seen = time.time()
    alive = os.path.exists(f"/proc/{PID}")
    if not alive:
        say(f"SERVER PROCESS {PID} IS GONE")
        sys.exit(1)
    now = time.time()
    if now - last_ready_check >= 60:
        last_ready_check = now
        r = tunnel_ready()
        if r != ready:
            say("TUNNEL READY again" if r else "TUNNEL NOT READY (cloudflared /ready: no connection)")
            ready = r
    if started:
        if connected == 0:
            idle_since = idle_since or now
            if now - idle_since >= IDLE and not idle_said:
                idle_said = True
                say(f"IDLE: no session connected for {IDLE / 60:.0f} min (sessions {n['session_open']}, resets {n['reset']}, "
                    f"episodes ended {n['episode_end']}, calls {calls}, error events {n['bad']}) | {tally()}")
        else:
            if idle_said:
                say("ACTIVE again: sessions connected")
            idle_since, idle_said = None, False
        if now - last_progress >= PERIOD:
            last_progress = now
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True,
                                     text=True, timeout=30).stdout.strip()
            except subprocess.TimeoutExpired:
                gpu = "nvidia-smi timed out"
            try:
                rss = int(open(f"/proc/{PID}/status").read().split("VmRSS:")[1].split()[0]) // 1024
            except (OSError, IndexError, ValueError):
                rss = -1
            say(f"progress: sessions {n['session_open']}, resets {n['reset']}, episodes ended {n['episode_end']}, "
                f"calls {calls}, connected now {connected}, mean call {mean_ms} ms (status {last_status}), "
                f"disconnects {n['disconnect']}, error events {n['bad']}, GPU {gpu}, RSS {rss} MB | {tally()}")
    time.sleep(5)
