#!/usr/bin/env python3
"""A steady, moderate GPU load beside the official endpoint (10-30: every hard reset came at light or changing load,
none in weeks of training): keeps the GPU busy without idle gaps long enough for a power-state change, until the
PPO reaching training takes over (scripts/bg_load_keeper.py starts and stops it).

Each 20 ms window runs fp32 matmuls for ``duty`` of the window and sleeps the rest (gaps of a few ms, far below the
driver's P-state switching); it synchronizes after every matmul so the server's kernels never queue behind a deep
backlog. One line per minute: iterations and achieved busy fraction.

  python scripts/gpu_steady_load.py [--n 2048] [--duty 0.6] [--log results/reach/logs/gpu_steady_load.log]
"""
import argparse
import signal
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2048, help="matrix size (one matmul ~1 ms at 2048 on an RTX A5000)")
ap.add_argument("--duty", type=float, default=0.6, help="busy fraction of each 20 ms window")
ap.add_argument("--window_s", type=float, default=0.02)
args = ap.parse_args()

stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
a = torch.randn(args.n, args.n, device="cuda")
b = torch.randn(args.n, args.n, device="cuda")
busy_s, iters, t_report = 0.0, 0, time.time()
print(f"[steady_load] start n={args.n} duty={args.duty}", flush=True)
while not stop:
    w0 = time.time()
    while time.time() - w0 < args.duty * args.window_s:
        torch.mm(a, b)
        torch.cuda.synchronize()
        iters += 1
    busy_s += time.time() - w0
    rest = args.window_s - (time.time() - w0)
    if rest > 0:
        time.sleep(rest)
    if time.time() - t_report >= 60:
        span = time.time() - t_report
        print(f"[steady_load] {time.strftime('%H:%M:%S')} iters {iters} busy {busy_s / span:.2f}", flush=True)
        busy_s, iters, t_report = 0.0, 0, time.time()
print("[steady_load] stopped", flush=True)
sys.exit(0)
