"""Policy-server adapter (student plan Step 11.2): one per evaluation session (the server keeps them by session id).

Each ``infer``: the observation -> joint angles, gripper widths, images, time; the pose estimator (shared network, own
tracking state) focused on the block of the runtime's running option -> ``State`` -> the student runtime (option
switcher + the executor's node classes) -> a little-endian float32 ``[H, 16]`` chunk. Safety (plan §11.4): a hold chunk
(measured joints and widths) while the images are blank or, on the first calls, no movable block is confidently seen;
any exception in the session returns a hold chunk and is logged; rows clipped to the action bounds; NaN guarded.
``reset`` clears the runtime and the estimator (even when the seed repeats); ``prepare(variant)`` builds them ahead of
the first call (the server does it at the episode's reset, from the worker's ``X-Tower-Task-Variant`` header).
``parse_observation`` reads the ``observation`` object of a ``tower-infer-v1`` infer request (schema
``tower-challenge-observation-v1``; the starter kit's ``observation_spec``, 2026-10-24: ``joint_positions`` <f4 [2, 7]
left / back, radians; ``gripper_widths_m`` <f4 [2] full openings in metres; ``images`` by camera name, |u1 [480, 640, 3]
RGB; ``task_variant``; ``instruction``; ``sim_time_s``; ``step_index`` = control actions already executed).
``recorder(record, images)``, if given, gets one dict per call (timings, hold reason, node, confidences, the estimated
poses) and the call's images (the server's session log).
"""

from __future__ import annotations

import time
import traceback

import numpy as np

from tower_rl.graph.interface import State
from tower_rl.perception.cameras import CAMERA_IDS
from tower_rl.student.runtime import StudentExecutor

CLIP_TOL = 1e-6  # clips beyond this are counted (Isaac reports an open gripper at 0.10000002 m: not a mapping error)


def parse_observation(obs: dict) -> dict:
    """The fields the adapter needs, from one infer request's ``observation`` object (live docs keys)."""
    imgs = obs.get("images") or {}
    return {"images": {c: np.asarray(imgs[c], dtype=np.uint8) for c in CAMERA_IDS if c in imgs},
            "q": np.asarray(obs["joint_positions"], dtype=np.float64).reshape(2, 7),
            "widths": np.asarray(obs["gripper_widths_m"], dtype=np.float64).reshape(2),
            "variant": obs.get("task_variant"), "instruction": obs.get("instruction", ""),
            "t": float(obs.get("sim_time_s", 0.0)), "step": int(obs.get("step_index", 0))}


def clip32(chunk: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, int]:
    """``chunk`` clipped to [lo, hi] as little-endian float32 that stays inside the float64 bounds after rounding (a
    bound such as 6.2832 rounds up in float32), and the number of values moved by more than ``CLIP_TOL``."""
    c = np.asarray(chunk, dtype=np.float64)
    lo, hi = np.broadcast_to(np.asarray(lo, dtype=np.float64), c.shape), np.broadcast_to(np.asarray(hi, dtype=np.float64), c.shape)
    clipped = np.clip(c, lo, hi)
    moved = int((np.abs(clipped - c) > CLIP_TOL).sum())
    out = clipped.astype("<f4")
    up, down = out.astype(np.float64) > hi, out.astype(np.float64) < lo
    out[up] = np.nextafter(out[up], np.float32(-np.inf))
    out[down] = np.nextafter(out[down], np.float32(np.inf))
    return np.ascontiguousarray(out), moved


class StudentAdapter:
    def __init__(self, bundle: str, decider, make_estimator, bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 hold_until_confident_calls: int = 3, log=print, recorder=None):
        """``decider``: the switcher (``SwitcherDecider``) or an oracle, shared by sessions (stateless per call);
        ``make_estimator()``: a fresh ``PoseEstimator`` (sharing the loaded network); ``bounds``: the track's action
        lower / upper rows ``[16]`` (default: the runtime's joint / gripper limits)."""
        self.bundle, self.decider, self.make_estimator = bundle, decider, make_estimator
        self.bounds = bounds
        self.hold_calls = hold_until_confident_calls
        self.log = log
        self.recorder = recorder
        self.reset()

    def reset(self):
        self.variant = None
        self.est = None
        self.policy = None
        self.calls = 0
        self.errors = 0
        self.clipped = 0
        self.timing: list[float] = []

    def prepare(self, variant: str | None):
        """Build the estimator and runtime for ``variant`` now (after ``reset``), so the first call does not load them."""
        if variant and self.variant is None:
            self._init(variant)

    def _init(self, variant: str):
        self.variant = variant
        self.est = self.make_estimator()
        self.est.reset(variant)
        self.policy = StudentExecutor.from_bundle(self.bundle, variant, device="cpu", rate_hz=10, decider=self.decider)
        self.policy.reset(1, variant, [0])

    @staticmethod
    def hold(q: np.ndarray, widths: np.ndarray) -> np.ndarray:
        return np.concatenate([q[0], widths[:1], q[1], widths[1:2]]).astype("<f4")[None]

    def infer(self, obs: dict) -> np.ndarray:
        t0 = time.time()
        o = None
        rec = {"call": self.calls + 1, "hold": None}
        try:
            o = parse_observation(obs)
            rec.update(t=round(o["t"], 3), step=o["step"])
            if self.variant is None:
                self._init(o["variant"])
            elif o["variant"] and o["variant"] != self.variant:
                raise ValueError(f"task_variant changed within a session: {self.variant} -> {o['variant']}")
            self.calls += 1
            imgs = o["images"]
            if len(imgs) < len(CAMERA_IDS) or all(not img.any() for img in imgs.values()):
                rec["hold"] = "blank"
                return self._done(self.hold(o["q"], o["widths"]), t0, rec, o)
            fb = self.policy.focus_block(0)
            focus = [self.est.block_ids.index(fb)] if fb in self.est.block_ids else None
            t1 = time.time()
            pos, quat, conf = self.est(imgs, o["q"], o["widths"], focus=focus)
            rec["est_ms"] = round(1000 * (time.time() - t1), 1)
            fixed = np.asarray(self.est.fixed.cpu().numpy(), dtype=bool)
            rec.update(pos=np.round(np.asarray(pos, dtype=np.float64), 5).tolist(),
                       quat=np.round(np.asarray(quat, dtype=np.float64), 5).tolist(), conf=np.asarray(conf).tolist())
            if self.calls <= self.hold_calls and not (conf[~fixed] >= 0.5).any():
                rec["hold"] = "unconfident"
                return self._done(self.hold(o["q"], o["widths"]), t0, rec, o)
            s = State(t=o["t"], step=o["step"], variant=self.variant, instruction=o["instruction"], q=o["q"].astype(np.float32),
                      width=o["widths"].astype(np.float32), block_ids=list(self.est.block_ids), block_pos=np.asarray(pos),
                      block_quat=np.asarray(quat), block_conf=np.asarray(conf))
            t1 = time.time()
            chunk = np.asarray(self.policy.act([s], [0])[0], dtype=np.float64)
            rec["policy_ms"] = round(1000 * (time.time() - t1), 1)
            rec["node"] = self.policy.node_info(0)
            return self._done(chunk, t0, rec, o)
        except Exception as ex:  # noqa: BLE001 (a session error never ends the episode: hold and log)
            self.errors += 1
            self.log(f"[adapter] {self.variant} call {self.calls}: {ex!r}\n{traceback.format_exc()}")
            if o is None:
                raise
            rec["hold"], rec["error"] = "error", repr(ex)
            return self._done(self.hold(o["q"], o["widths"]), t0, rec, o)

    def _done(self, chunk: np.ndarray, t0: float, rec: dict, o: dict) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 16 or not 1 <= chunk.shape[0] or not np.isfinite(chunk).all():
            self.errors += 1
            rec["hold"], rec["error"] = "bad_chunk", f"shape {list(chunk.shape)}"
            chunk = self.hold(o["q"], o["widths"]).astype(np.float64)
        chunk = chunk[:16]
        if self.bounds is not None:
            out, moved = clip32(chunk, self.bounds[0], self.bounds[1])
        else:
            out, moved = np.ascontiguousarray(chunk.astype("<f4")), 0
        self.clipped += moved
        self.timing.append(time.time() - t0)
        rec.update(rows=int(out.shape[0]), clipped=moved, total_ms=round(1000 * self.timing[-1], 1))
        if self.recorder is not None:
            try:
                self.recorder(rec, o["images"])
            except Exception as ex:  # noqa: BLE001 (logging never fails a call)
                self.log(f"[adapter] recorder: {ex!r}")
        return out
