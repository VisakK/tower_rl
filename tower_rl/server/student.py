"""The student behind the ``tower-infer-v1`` handler (student plan Step 11): models loaded once and shared, one
``SessionModel`` per evaluation session, and the server's logs.

- ``StudentModels``: the frozen bundle (every file checked against its MANIFEST), kp_v2 on the GPU, the switcher, and
  the track's action bounds. ``camera_source`` is "evaluator" for the challenge's images and "render" only for our own
  Isaac renders (``scripts/ws_loop.py``): the two camera models differ in the principal point
  (``tower_rl.perception.cameras``).
- ``SessionModel``: a ``StudentAdapter`` plus the episode log. ``start(trial_id, seed)`` (every reset) clears the adapter
  and builds the estimator and runtime for the session's variant; ``infer(observation, sample_index)`` returns the chunk.
  Per episode it writes ``<log_dir>/episodes/<variant>/<trial>.jsonl`` (one row per call: time, step, hold reason,
  node, timings, estimated poses and confidences), ``<trial>_images/`` (the four images of the first call and of every
  ``save_images_every``-th call, JPEG) and ``<trial>_summary.json`` (counts, timing percentiles, the runtime's node
  events and options).
- ``EventLog``: ``<log_dir>/server.jsonl``, one line per session event (open, reset, error, disconnect, expiry), and
  the console lines.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

CAMERAS = ("left_back_camera", "right_front_camera", "left_wrist_camera", "back_wrist_camera")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_bundle(bundle: Path) -> dict:
    """The bundle's MANIFEST with every listed file's sha256 re-checked (raises on a mismatch or a missing file)."""
    man = json.loads((bundle / "MANIFEST.json").read_text())
    bad = [f for f, d in man["files"].items() if f != "MANIFEST.json" and
           (not (bundle / f).is_file() or sha256_file(bundle / f) != d)]
    if bad:
        raise SystemExit(f"{bundle}: {len(bad)} files differ from MANIFEST.json, e.g. {bad[:3]}")
    return man


def artifact_identity(bundle: Path) -> str:
    """``sha256:<hex>`` of the bundle's tar (``freeze_student_bundle.py`` writes ``<name>.tar.sha256``)."""
    p = bundle.parent / f"{bundle.name}.tar.sha256"
    digest = p.read_text().split()[0]
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), p
    return "sha256:" + digest


def check_track(track: dict):
    """The student serves the 10 Hz track with the four 480 x 640 cameras (its skills and runtime are 10 Hz only)."""
    if abs(track["action_dt_s"] - 0.1) > 1e-9:
        raise SystemExit(f"track action_dt_s {track['action_dt_s']}: the student is 10 Hz only")
    for c in CAMERAS:
        if list(track["cameras"].get(c, [])) != [480, 640]:
            raise SystemExit(f"track camera {c}: {track['cameras'].get(c)}, expected [480, 640]")
    if len(track["action_lower"]) != 16 or len(track["action_upper"]) != 16 or track["max_chunk_actions"] < 16:
        raise SystemExit("track: expected 16-D action bounds and chunks of up to 16 actions")


class EventLog:
    ECHO = ("session_open", "reset", "prepare_failed", "error", "adapter_error", "session_expired", "duplicate_request")

    def __init__(self, path: Path | None, echo=lambda line: print(line, flush=True)):
        self.lock = threading.Lock()
        self.f = open(path, "a") if path is not None else None
        self.echo = echo

    def __call__(self, kind: str, **kw):
        row = {"time": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, **kw}
        with self.lock:
            if self.f is not None:
                self.f.write(json.dumps(row, default=str) + "\n")
                self.f.flush()
        if kind in self.ECHO and self.echo is not None:
            self.echo(f"[server] {row['time'][11:]} {kind} " + " ".join(f"{k}={v}" for k, v in kw.items() if k != "trace"))


class StudentModels:
    def __init__(self, bundle: str | Path, track: dict, device: str = "cuda", camera_source: str = "evaluator",
                 hold_until_confident_calls: int = 3, verify: bool = True):
        import torch

        from tower_rl.perception.estimator import load_net
        from tower_rl.student.switcher import SwitcherDecider

        self.bundle = Path(bundle).resolve()
        self.manifest = verify_bundle(self.bundle) if verify else None
        self.track = track
        self.device = device
        self.camera_source = camera_source
        self.hold_calls = hold_until_confident_calls
        self.net = load_net(str(self.bundle / "perception" / "kp_v2.pt"), torch.device(device))
        self.decider = SwitcherDecider.load(self.bundle / "student" / "switcher.pt")
        per = {}  # v1.3: a graph's "switcher" (a path in the bundle) overrides the bundle's switcher for that variant
        for g in sorted((self.bundle / "graphs").glob("*.json")):
            rel = json.loads(g.read_text()).get("switcher")
            if rel:
                per[g.stem] = SwitcherDecider.load(self.bundle / rel)
        if per:
            from tower_rl.student.switcher import VariantDeciders

            self.decider = VariantDeciders(self.decider, per)
        # student_v1_1: the fixed cameras' orientation is chosen per episode against these references (absent in
        # student_v1, whose adapter uses the images as delivered)
        from tower_rl.perception.orient import load_refs

        self.orient_refs = load_refs(self.bundle / "perception" / "orient_refs.npz") or None
        self.bounds = (np.asarray(track["action_lower"], dtype=np.float64), np.asarray(track["action_upper"], dtype=np.float64))

    def estimator(self):
        from tower_rl.perception.estimator import PoseEstimator

        return PoseEstimator(None, self.device, source=self.camera_source, net=self.net)

    def adapter(self, recorder=None, log=print):
        from tower_rl.student.adapter import StudentAdapter

        return StudentAdapter(str(self.bundle), self.decider, self.estimator, bounds=self.bounds,
                              hold_until_confident_calls=self.hold_calls, log=log, recorder=recorder,
                              orient_refs=self.orient_refs)

    def warm_up(self, variant: str = "task2-2_cross_tower_transfer_2"):
        """One network pass and one runtime build before the port opens (CUDA kernels, lazy imports): the first
        evaluation call then costs what every call costs."""
        import torch

        est = self.estimator()
        est.reset(variant)
        with torch.no_grad():
            est.heatmaps({c: np.zeros((480, 640, 3), dtype=np.uint8) for c in CAMERAS})
        a = self.adapter(log=lambda *x: None)
        a.prepare(variant)
        return a


def _pct(x: list[float]) -> dict:
    if not x:
        return {}
    a = np.asarray(x, dtype=np.float64)
    return {"n": len(a), "mean": round(float(a.mean()), 1), "p50": round(float(np.percentile(a, 50)), 1),
            "p99": round(float(np.percentile(a, 99)), 1), "max": round(float(a.max()), 1)}


def _jsonable(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if hasattr(x, "value") and type(x).__module__ != "builtins":  # enums (node status)
        return x.value
    return x


class SessionModel:
    def __init__(self, models: StudentModels, session_id: str, variant: str | None, log_dir: Path | None,
                 events: EventLog, save_images_every: int = 100):
        self.models, self.session, self.variant = models, session_id, variant
        self.log_dir, self.events, self.every = log_dir, events, save_images_every
        self._lock = threading.Lock()
        self.adapter = models.adapter(recorder=self._record, log=self._adapter_error)
        self.trial, self.seed, self._f, self._index = None, None, None, None
        self._stats = {"est_ms": [], "policy_ms": [], "total_ms": []}
        self._holds: dict[str, int] = {}

    # ------------------------------------------------------------------ the handler's interface
    def start(self, trial_id: str, seed: int):
        with self._lock:
            self._finish()
            self.adapter.reset()
            self.trial, self.seed = trial_id, seed
            self._stats = {"est_ms": [], "policy_ms": [], "total_ms": []}
            self._holds = {}
            if self.log_dir is not None:
                d = self.log_dir / "episodes" / (self.variant or "unknown_variant")
                d.mkdir(parents=True, exist_ok=True)
                self._f = open(d / f"{trial_id}.jsonl", "a")
            try:
                self.adapter.prepare(self.variant)
            except Exception as ex:  # noqa: BLE001 (the first call builds it again, and holds if that fails too)
                self.events("prepare_failed", session=self.session, variant=self.variant, error=repr(ex))
                self.adapter.reset()

    def infer(self, observation: dict, sample_index: int) -> np.ndarray:
        with self._lock:
            self._index = sample_index
            if self.variant is None:
                self.variant = observation.get("task_variant")
            return self.adapter.infer(observation)

    def flush(self):
        """The episode summary so far (on a disconnect; skipped while a call is running: it is written later)."""
        if self._lock.acquire(blocking=False):
            try:
                self._write_summary()
            finally:
                self._lock.release()

    # ------------------------------------------------------------------ logs
    def _adapter_error(self, msg: str):
        self.events("adapter_error", session=self.session, variant=self.variant, trial=self.trial,
                    error=msg.splitlines()[0] if msg else "", trace=msg)

    def _record(self, rec: dict, images: dict):
        rec = {"sample_index": self._index, **rec}
        for k in ("est_ms", "policy_ms", "total_ms"):
            if k in rec:
                self._stats[k].append(rec[k])
        if rec.get("hold"):
            self._holds[rec["hold"]] = self._holds.get(rec["hold"], 0) + 1
        if self._f is not None:
            self._f.write(json.dumps(_jsonable(rec)) + "\n")
            self._f.flush()
        n = rec.get("call", 0)
        if self.log_dir is not None and images and self.every and (n == 1 or n % self.every == 0):
            import cv2

            d = self.log_dir / "episodes" / (self.variant or "unknown_variant") / f"{self.trial}_images"
            d.mkdir(parents=True, exist_ok=True)
            for c, img in images.items():
                cv2.imwrite(str(d / f"{n:05d}_{c}.jpg"), np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _summary(self) -> dict:
        a = self.adapter
        out = {"session": self.session, "trial": self.trial, "seed": self.seed, "variant": self.variant,
               "written": datetime.now().isoformat(timespec="seconds"), "calls": a.calls, "errors": a.errors,
               "clipped_values": a.clipped, "holds": self._holds, "timing_ms": {k: _pct(v) for k, v in self._stats.items()},
               "orient": a.orient_info, "blind": a.blind, "confident_calls": a.confident_calls}
        if a.policy is not None:
            m = a.policy.memory(0)
            out.update(decisions=m.decisions, masked=m.masked, vetoed=getattr(m, "vetoed", 0),
                       events=_jsonable(a.policy.episode_log(0)), options=_jsonable(m.option_log),
                       vetoes=_jsonable(getattr(m, "vetoes", [])))
            extra = a.policy.episode_extra(0) if hasattr(a.policy, "episode_extra") else {}
            if extra.get("recovery") is not None:  # graph "recovery": the rule node's decisions and abstentions
                out["recovery"] = _jsonable(extra["recovery"])
        return out

    def _write_summary(self):
        """Never raises (it runs in the handler's ``finally``): a failure is logged."""
        if self.trial is None or self.log_dir is None:
            return
        try:
            d = self.log_dir / "episodes" / (self.variant or "unknown_variant")
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{self.trial}_summary.json").write_text(json.dumps(self._summary(), indent=1, default=str))
        except Exception as ex:  # noqa: BLE001
            self.events("summary_failed", session=self.session, trial=self.trial, error=repr(ex))

    def _finish(self):
        """The previous episode of this session: its summary, its call log closed."""
        if self.trial is not None:
            self._write_summary()
            a = self.adapter
            self.events("episode_end", session=self.session, trial=self.trial, variant=self.variant, calls=a.calls,
                        errors=a.errors, holds=self._holds, total_ms=_pct(self._stats["total_ms"]),
                        orient=(a.orient_info or {}).get("rotate"), blind=a.blind)
        if self._f is not None:
            self._f.close()
            self._f = None

    def close(self):
        with self._lock:
            self._finish()
