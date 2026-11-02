"""Probe a running ``tower-infer-v1`` server as the evaluation worker would (student plan Step 11, G11a), before the
site's "Check endpoint".

  --check   (default) qualification-like probes: one session per track variant, all at once, each with two fresh
            resets and a few infer calls on zero-valued images with joint positions drawn within the action bounds.
  --replay  organizer episodes (TOWER-SimData HDF5: the evaluator's renderer, real images in the delivered orientation,
            recorded joints and widths) streamed at 10 Hz through the server, one session per file, all at once. The
            server's per-call log (``--server_log``) then gives the estimated block poses, compared with the recorded
            ones: the evaluator camera path end to end (decoding, orientation, camera model, joint mapping). The
            replayed world does not follow the returned actions, so this checks perception, protocol and timing only.

Every reply is validated like the worker: identity echoes, seed fields and counters, ``dt_s``, plugin fields, chunk
dtype / shape / finiteness / bounds; the metadata against the track's specs. Prints a JSON report (``--out`` too).

  python scripts/probe_server.py ws://127.0.0.1:9001 --track data/challenge/track.json
  python scripts/probe_server.py ws://127.0.0.1:9001 --track data/challenge/track.json --replay \
      data/episodes_full/data/task2-2_cross_tower_transfer_2/*.hdf5 --server_log results/student/server/<name>
"""
import argparse
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.server.tower_protocol import action_spec, observation_spec, pack, sample_seed, unpack  # noqa: E402

CAMERAS = ("left_back_camera", "right_front_camera", "left_wrist_camera", "back_wrist_camera")


class Fail(Exception):
    pass


def check_metadata(m: dict, track: dict):
    want = {"protocol": "tower-infer-v1", "type": "metadata", "ready": True, "seed_scheme": "tower-seed-stream-v1",
            "supports_seed_reset": True}
    for k, v in want.items():
        if m.get(k) != v:
            raise Fail(f"metadata {k} = {m.get(k)!r}, expected {v!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", str(m.get("policy_id"))):
        raise Fail(f"metadata policy_id {m.get('policy_id')!r}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(m.get("artifact_identity"))):
        raise Fail(f"metadata artifact_identity {m.get('artifact_identity')!r}")
    if m.get("seed_mode") not in ("sample_seed", "deterministic"):
        raise Fail(f"metadata seed_mode {m.get('seed_mode')!r}")
    if m.get("observation_spec") != json.loads(json.dumps(observation_spec(track))):
        raise Fail("metadata observation_spec differs from the track's")
    if m.get("action_spec") != json.loads(json.dumps(action_spec(track))):
        raise Fail("metadata action_spec differs from the track's")
    rc = m.get("reconnect")
    if rc is not None and (rc.get("protocol") != "tower-reconnect-v1" or rc.get("retention_s", 0) < 60 + track["inference_timeout_s"]):
        raise Fail(f"metadata reconnect {rc}")


def check_reply(res: dict, req: dict, meta: dict, track: dict, seed: int, index: int | None):
    if res.get("type") == "error":
        raise Fail(f"error reply {res.get('error')}")
    for k in ("protocol", "session_id", "trial_id", "request_id", "policy_id"):
        if res.get(k) != req[k]:
            raise Fail(f"{res.get('type')} {k} = {res.get(k)!r}, expected {req[k]!r}")
    if index is None:  # reset_ack
        if (res.get("type"), res.get("base_seed"), res.get("next_sample_index"), res.get("seed_scheme")) != \
                ("reset_ack", seed, 0, "tower-seed-stream-v1"):
            raise Fail(f"reset_ack {res}")
        return None
    if res.get("type") != "infer_result" or res.get("base_seed") != seed or res.get("sample_index") != index or \
            res.get("sample_seed") != sample_seed(seed, index) or res.get("seed_scheme") != "tower-seed-stream-v1":
        raise Fail(f"infer_result seed fields {[res.get(k) for k in ('type', 'base_seed', 'sample_index', 'sample_seed')]}")
    if res.get("dt_s") != track["action_dt_s"] or res.get("plugin") != meta.get("plugin") or \
            res.get("plugin_version") != meta.get("plugin_version"):
        raise Fail(f"infer_result dt_s / plugin {res.get('dt_s')} {res.get('plugin')} {res.get('plugin_version')}")
    a = res.get("actions")
    if not isinstance(a, np.ndarray) or a.dtype != np.dtype("<f4") or a.ndim != 2 or a.shape[1] != 16 or \
            not 1 <= a.shape[0] <= track["max_chunk_actions"]:
        raise Fail(f"actions {type(a).__name__} {getattr(a, 'dtype', None)} {getattr(a, 'shape', None)}")
    if not np.isfinite(a).all():
        raise Fail("actions not finite")
    lo, hi = np.asarray(track["action_lower"], dtype=np.float64), np.asarray(track["action_upper"], dtype=np.float64)
    a64 = a.astype(np.float64)
    if (a64 < lo).any() or (a64 > hi).any():
        raise Fail(f"actions outside the bounds: {np.argwhere((a64 < lo) | (a64 > hi))[:4].tolist()}")
    return a


class Session:
    def __init__(self, url: str, track: dict, variant: str, policy_id: str | None):
        from websockets.sync.client import connect

        headers = {"X-Tower-Task-Variant": variant, "X-Tower-Track-Version": track.get("policy_track_version") or track["version"],
                   "X-Tower-Evaluation-Version": track["version"]}
        t0 = time.perf_counter()
        self.ws = connect(url, additional_headers=headers, compression=None, max_size=64 * 1024 * 1024, open_timeout=30)
        self.meta = unpack(self.ws.recv(timeout=10))
        self.meta_s = time.perf_counter() - t0
        check_metadata(self.meta, track)
        if policy_id and self.meta["policy_id"] != policy_id:
            raise Fail(f"policy_id {self.meta['policy_id']} != {policy_id}")
        self.track, self.variant, self.sid = track, variant, str(uuid.uuid4())
        self.latency: list[float] = []
        self.rows: list[int] = []

    def request(self, body: dict, seed: int, index: int | None):
        req = {"protocol": "tower-infer-v1", "session_id": self.sid, "trial_id": self.trial, "request_id": str(uuid.uuid4()),
               "policy_id": self.meta["policy_id"], **body}
        t0 = time.perf_counter()
        self.ws.send(pack(req))
        res = unpack(self.ws.recv(timeout=self.track["inference_timeout_s"] + 5))
        if index is not None:
            self.latency.append(time.perf_counter() - t0)
        a = check_reply(res, req, self.meta, self.track, seed, index)
        if a is not None:
            self.rows.append(a.shape[0])
        return a

    def reset(self, seed: int):
        self.trial, self.seed, self.index = str(uuid.uuid4()), seed, 0
        self.request({"type": "reset", "seed": seed}, seed, None)

    def infer(self, obs: dict):
        a = self.request({"type": "infer", "observation": obs}, self.seed, self.index)
        self.index += 1
        return a


def instruction(track: dict, variant: str) -> str:
    return next(v["instruction"] for v in track["variants"] if v["id"] == variant)


def zero_obs(track, variant, rng, step):
    lo, hi = np.asarray(track["action_lower"]), np.asarray(track["action_upper"])
    q = rng.uniform(np.r_[lo[0:7], lo[8:15]], np.r_[hi[0:7], hi[8:15]]).reshape(2, 7).astype("<f4")
    w = rng.uniform([lo[7], lo[15]], [hi[7], hi[15]]).astype("<f4")
    return {"schema": track["observation_schema"], "step_index": step, "sim_time_s": round(0.1 * step, 6), "task_variant": variant,
            "instruction": instruction(track, variant), "joint_positions": q, "gripper_widths_m": w,
            "images": {c: np.zeros((*track["cameras"][c], 3), dtype=np.uint8) for c in CAMERAS}}


def run_check(url, track, policy_id, calls, results):
    def one(variant, k):
        out = {"variant": variant}
        try:
            s = Session(url, track, variant, policy_id)
            rng = np.random.default_rng(k)
            for r in range(2):  # two fresh resets
                s.reset(int(rng.integers(0, 2**32)))
                for i in range(calls):
                    s.infer(zero_obs(track, variant, rng, i))
            s.ws.close()
            out.update(ok=True, metadata_s=round(s.meta_s, 3), latency_ms=_ms(s.latency))
        except Exception as ex:  # noqa: BLE001
            out.update(ok=False, error=repr(ex))
        results.append(out)

    th = [threading.Thread(target=one, args=(v["id"], k)) for k, v in enumerate(track["variants"])]
    for t in th:
        t.start()
    for t in th:
        t.join()


def _ms(x):
    if not x:
        return {}
    a = 1000 * np.asarray(x)
    return {"n": len(a), "mean": round(float(a.mean()), 1), "p50": round(float(np.percentile(a, 50)), 1),
            "p99": round(float(np.percentile(a, 99)), 1), "max": round(float(a.max()), 1)}


def run_replay(url, track, policy_id, path, stride, max_calls, server_log, corrections, results):
    import cv2
    import h5py

    variant = Path(path).parent.name
    out = {"variant": variant, "episode": Path(path).stem}
    try:
        s = Session(url, track, variant, policy_id)
        s.reset(12345)
        with h5py.File(path, "r") as f:
            qpos = f["observations/qpos"][()]
            T = len(qpos)
            frames = list(range(0, T, stride))[:max_calls]
            for k, t in enumerate(frames):
                imgs = {n: np.ascontiguousarray(cv2.imdecode(np.frombuffer(bytes(f[f"observations/images/{n}"][t]), np.uint8),
                                                             cv2.IMREAD_COLOR)[..., ::-1]) for n in CAMERAS}
                obs = {"schema": track["observation_schema"], "step_index": t // 3, "sim_time_s": round(t / 30.0, 6),
                       "task_variant": variant, "instruction": instruction(track, variant),
                       "joint_positions": np.stack([qpos[t, 0:7], qpos[t, 8:15]]).astype("<f4"),
                       "gripper_widths_m": np.array([qpos[t, 7], qpos[t, 15]], dtype="<f4"), "images": imgs}
                s.infer(obs)
            ids = json.loads(f["sim"].attrs["block_ids"])
            bp = f["sim/block_poses"][()]
        s.ws.close()
        out.update(ok=True, calls=len(frames), latency_ms=_ms(s.latency), rows_mean=round(float(np.mean(s.rows)), 2))
        if server_log:
            out["estimation"] = compare(Path(server_log), variant, s.trial, frames, ids, bp, corrections.get(f"{variant}/{Path(path).stem}", {}))
    except Exception as ex:  # noqa: BLE001
        out.update(ok=False, error=repr(ex))
    results.append(out)


def compare(log_dir: Path, variant: str, trial: str, frames, ids, bp, corr: dict) -> dict:
    """Estimated (the server's call log) vs recorded block poses, per call: position error of confident estimates."""
    from tower_rl.data.layouts import Layout
    from tower_rl.perception.estimator import block_order

    lay = Layout.load(variant)
    order_ids = block_order(lay)
    order = [ids.index(b) for b in order_ids]
    fixed = np.array([bool(lay.fixed[b]) for b in order_ids])
    bp = bp.copy()
    for pad, c in corr.items():
        for b in lay.pads[pad]["foundation_blocks"]:
            bp[:, ids.index(b), 0] += c["dx_mm"] / 1000
            bp[:, ids.index(b), 1] += c["dy_mm"] / 1000
    p = log_dir / "episodes" / variant / f"{trial}.jsonl"
    for _ in range(20):  # the last row may still be in flight
        if p.is_file() and sum(1 for _ in open(p)) >= len(frames):
            break
        time.sleep(0.25)
    rows = [json.loads(x) for x in open(p)]
    err_m, err_f, conf_ok, holds = [], [], [], 0
    for r, t in zip(rows, frames):
        if "pos" not in r:
            holds += 1
            continue
        e = 1000 * np.linalg.norm(np.asarray(r["pos"]) - bp[t, order, :3], axis=-1)
        c = np.asarray(r["conf"]) >= 0.5
        conf_ok.append(c.mean())
        err_m += e[c & ~fixed].tolist()
        err_f += e[c & fixed].tolist()

    def st(e):
        if not e:
            return {}
        e = np.asarray(e)
        return {"n": len(e), "median_mm": round(float(np.median(e)), 2), "p90_mm": round(float(np.percentile(e, 90)), 2),
                "p99_mm": round(float(np.percentile(e, 99)), 2), "frac_gt_5mm": round(float((e > 5).mean()), 4)}

    return {"rows": len(rows), "holds": holds, "frac_confident": round(float(np.mean(conf_ok)), 4) if conf_ok else None,
            "movable": st(err_m), "foundations": st(err_f), "foundation_correction": corr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--track", required=True)
    ap.add_argument("--policy_id", default=None)
    ap.add_argument("--calls", type=int, default=3, help="--check: infer calls after each reset")
    ap.add_argument("--replay", nargs="*", default=None, help="organizer episode HDF5 files")
    ap.add_argument("--stride", type=int, default=3, help="--replay: 30 Hz frames per call (3: 10 Hz)")
    ap.add_argument("--max_calls", type=int, default=400)
    ap.add_argument("--server_log", default=None, help="--replay: the server's log dir (for the estimation errors)")
    ap.add_argument("--foundation_corrections", nargs="*", default=["data/perception/dataset_val/foundation_corrections.json",
                                                                    "data/perception/dataset_train/foundation_corrections.json"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    track = json.loads(Path(args.track).read_text())
    results: list[dict] = []
    t0 = time.time()
    if args.replay is None:
        run_check(args.url, track, args.policy_id, args.calls, results)
    else:
        corr = {}
        for p in args.foundation_corrections:
            if Path(p).is_file():
                corr.update(json.loads(Path(p).read_text()))
        th = [threading.Thread(target=run_replay, args=(args.url, track, args.policy_id, p, args.stride, args.max_calls,
                                                         args.server_log, corr, results)) for p in args.replay]
        for t in th:
            t.start()
        for t in th:
            t.join()
    results.sort(key=lambda r: r["variant"])
    report = {"url": args.url, "mode": "replay" if args.replay is not None else "check", "ok": all(r.get("ok") for r in results),
              "wall_s": round(time.time() - t0, 1), "sessions": results}
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
