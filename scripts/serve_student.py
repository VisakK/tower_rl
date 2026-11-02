"""Serve the frozen student bundle to the challenge evaluator (``tower-infer-v1``; student plan Step 11).

    python scripts/serve_student.py --track data/challenge/track.json --bundle results/bundles/student_v1 --name official_1

The models are loaded (bundle files checked against its MANIFEST) and warmed up before the port opens, so the metadata
goes out at once. Each evaluation session gets its own estimator and option runtime, cleared at every reset; model calls
run one at a time on the GPU. ``artifact_identity`` defaults to the bundle tar's sha256 (``<bundle>.tar.sha256``).
Logs: ``results/student/server/<name>/`` (``server.jsonl`` events, ``episodes/<variant>/<trial>.jsonl`` per call,
``*_summary.json`` per episode, a few JPEG images per episode), and a status line on the console every minute.
Expose it with ``cloudflared tunnel --url http://127.0.0.1:9001`` and register ``wss://<name>.trycloudflare.com`` on the
challenge site. Long runs: start it from a frozen code snapshot (``tower_rl_snapshots/``), since lazily imported
modules would otherwise pick up later edits mid-evaluation. Defaults: ``configs/student/server.yaml``.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.paths import RESULTS_DIR  # noqa: E402

cfg = load_yaml("student/server")
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--track", required=True, help="track.json downloaded from the challenge site (all ten tasks, 10 Hz)")
parser.add_argument("--bundle", default=str(RESULTS_DIR / "bundles" / "student_v1"))
parser.add_argument("--name", default=None, help="log directory name under results/student/server/ (default: date_time)")
parser.add_argument("--host", default=cfg["host"])
parser.add_argument("--port", type=int, default=cfg["port"])
parser.add_argument("--policy_id", default=cfg["policy_id"])
parser.add_argument("--artifact_identity", default=None, help="sha256:<64 hex> (default: the bundle tar's sha256)")
parser.add_argument("--plugin_version", default=str(cfg["plugin_version"]))
parser.add_argument("--camera_source", choices=["evaluator", "render"], default=cfg["camera_source"])
parser.add_argument("--device", default="cuda")
parser.add_argument("--save_images_every", type=int, default=cfg["save_images_every"])
parser.add_argument("--no_reconnect", action="store_true", help="do not advertise tower-reconnect-v1")
parser.add_argument("--no_verify", action="store_true", help="skip the MANIFEST check of the bundle files")
args = parser.parse_args()

import torch  # noqa: E402

# the runtime's IK / FK / policy calls are tiny tensors (as in rehearse.py: one intra-op thread)
torch.set_num_threads(1)
from websockets.asyncio.server import serve  # noqa: E402

from tower_rl.server.handler import handler_factory  # noqa: E402
from tower_rl.server.student import EventLog, SessionModel, StudentModels, artifact_identity, check_track  # noqa: E402

track = json.loads(Path(args.track).read_text())
check_track(track)
bundle = Path(args.bundle).resolve()
identity = args.artifact_identity or artifact_identity(bundle)
name = args.name or datetime.now().strftime("%Y-%m-%d_%H%M")
log_dir = RESULTS_DIR / "student" / "server" / name
log_dir.mkdir(parents=True, exist_ok=True)
events = EventLog(log_dir / "server.jsonl")

t0 = time.time()
models = StudentModels(bundle, track, device=args.device, camera_source=args.camera_source,
                       hold_until_confident_calls=cfg["hold_until_confident_calls"], verify=not args.no_verify)
models.warm_up()
run = {"started": datetime.now().isoformat(timespec="seconds"), "pid": os.getpid(), "argv": sys.argv, "bundle": str(bundle),
       "artifact_identity": identity, "policy_id": args.policy_id, "plugin_version": args.plugin_version,
       "camera_source": args.camera_source, "reconnect": not args.no_reconnect, "track": args.track,
       "track_version": track.get("version"), "inference_timeout_s": track["inference_timeout_s"],
       "episode_wall_timeout_s": track["episode_wall_timeout_s"], "load_s": round(time.time() - t0, 1)}
(log_dir / "run.json").write_text(json.dumps(run, indent=1))
events("start", **{k: v for k, v in run.items() if k != "argv"})


def make_model(session_id, variant):
    return SessionModel(models, session_id, variant, log_dir, events, save_images_every=args.save_images_every)


handle = handler_factory(track, args.policy_id, identity, make_model, plugin=cfg["plugin"],
                         plugin_version=args.plugin_version, reconnect=not args.no_reconnect, log=events)


async def status():
    """A console line a minute: sessions (connected), calls, errors and the mean call time since the last line."""
    last = {}
    while True:
        await asyncio.sleep(60)
        live = [s for s in handle.sessions.values() if s["model"] is not None]
        calls = {id(s): s["model"].adapter.calls for s in live}
        new = sum(c - last.get(k, 0) for k, c in calls.items() if c >= last.get(k, 0))
        ms = [x for s in live for x in s["model"].adapter.timing[-50:]]
        print(f"[server] {datetime.now():%H:%M:%S} sessions {len(live)} (connected "
              f"{sum(s['connections'] > 0 for s in handle.sessions.values())}) calls +{new} errors "
              f"{sum(s['model'].adapter.errors for s in live)} mean call "
              f"{1000 * sum(ms) / max(1, len(ms)):.0f} ms", flush=True)
        last = calls


async def main():
    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set_result, None)
    async with serve(handle, args.host, args.port, max_size=64 * 1024 * 1024, compression=None,
                     # The worker enforces the release's inference deadline. Keep pings
                     # enabled without closing a busy image exchange after 20 seconds.
                     ping_timeout=None):
        print(f"[server] READY ws://{args.host}:{args.port} policy_id={args.policy_id} artifact_identity={identity} "
              f"camera_source={args.camera_source} reconnect={not args.no_reconnect} logs={log_dir} "
              f"(loaded in {run['load_s']} s)", flush=True)
        task = asyncio.create_task(status())
        await stop
        task.cancel()
    for s in list(handle.sessions.values()):
        if s["model"] is not None:
            s["model"].close()
    events("stop")
    print("[server] stopped", flush=True)


asyncio.run(main())
os._exit(0)
