#!/usr/bin/env python3
"""Crash-proof system telemetry: one CSV line per second, flushed and fsynced, so a hard reset (three since 10-18, no
kernel trace) leaves the machine's last seconds on disk. Columns: time, load1, CPU package / hottest core temperature,
mean / max core MHz, share of the last second cpu0 spent in each idle state, available memory, GPU power / temperature
/ utilisation / SM clock / P-state / throttle reasons (one persistent nvidia-smi), enp6s0 rx / tx rates.

  python3 scripts/sys_telemetry.py [out.csv]   (default results/v1_3/logs/sys_telemetry_<date>.csv)
"""
import glob
import os
import subprocess
import sys
import threading
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else time.strftime(
    "/home/visakii/Documents/tower_rl/results/v1_3/logs/sys_telemetry_%Y%m%d_%H%M.csv")
NIC = "enp6s0"


def coretemps():
    pkg, hot = None, 0.0
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(f"{h}/name").read().strip() != "coretemp":
                continue
        except OSError:
            continue
        for t in glob.glob(f"{h}/temp*_input"):
            v = int(open(t).read()) / 1000
            label = open(t.replace("_input", "_label")).read().strip()
            if label.startswith("Package"):
                pkg = v
            else:
                hot = max(hot, v)
    return pkg, hot


def mhz():
    v = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
    return sum(v) / len(v), max(v)


IDLE = sorted(glob.glob("/sys/devices/system/cpu/cpu0/cpuidle/state*"))
IDLE_NAMES = [open(f"{s}/name").read().strip() for s in IDLE]


def idle_us():
    return [int(open(f"{s}/time").read()) for s in IDLE]


def nic():
    return (int(open(f"/sys/class/net/{NIC}/statistics/rx_bytes").read()),
            int(open(f"/sys/class/net/{NIC}/statistics/tx_bytes").read()))


def mem_avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1e6
    return float("nan")


gpu = {"line": ""}


def gpu_reader():
    q = "power.draw,temperature.gpu,utilization.gpu,clocks.sm,pstate,clocks_event_reasons.active"
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms", "1000"],
                         stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        gpu["line"] = line.strip().replace(", ", ",")


threading.Thread(target=gpu_reader, daemon=True).start()
new = not os.path.exists(OUT)
f = open(OUT, "a")
if new:
    f.write("time,load1,cpu_pkg_c,cpu_hot_c,mhz_mean,mhz_max," + ",".join(f"cpu0_{n}" for n in IDLE_NAMES)
            + ",mem_avail_gb,gpu_w,gpu_c,gpu_util,gpu_sm_mhz,gpu_pstate,gpu_reasons,rx_mbs,tx_mbs\n")
prev_idle, prev_nic, prev_t = idle_us(), nic(), time.time()
while True:
    time.sleep(1.0)
    now = time.time()
    dt = max(now - prev_t, 1e-3)
    cur_idle, cur_nic = idle_us(), nic()
    share = [(b - a) / (dt * 1e6) for a, b in zip(prev_idle, cur_idle)]
    pkg, hot = coretemps()
    mean_mhz, max_mhz = mhz()
    rx = (cur_nic[0] - prev_nic[0]) / dt / 1e6
    tx = (cur_nic[1] - prev_nic[1]) / dt / 1e6
    f.write(time.strftime("%H:%M:%S", time.localtime(now)) + f",{os.getloadavg()[0]:.2f},{pkg},{hot},{mean_mhz:.0f},"
            f"{max_mhz:.0f}," + ",".join(f"{s:.2f}" for s in share) + f",{mem_avail_gb():.1f},{gpu['line']},"
            f"{rx:.2f},{tx:.2f}\n")
    f.flush()
    os.fsync(f.fileno())
    prev_idle, prev_nic, prev_t = cur_idle, cur_nic, now
