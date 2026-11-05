"""The ``tower-infer-v1`` WebSocket handler: the starter kit's reference ``server.py`` (2026-10-24, sha256 405ad123...,
``data/challenge/kit/``) with per-session model state (student plan Step 11.1).

The kit's ``predict(observation, *, prompt, seed)`` gets no session, but the student keeps per-episode state (the
estimator's tracking, the option runtime). So every session in the kit's registry also holds a model object made by
``make_model(session_id, variant)``: ``start(trial_id, seed)`` clears and rebuilds its episode state at every reset
(before the ``reset_ack``, even when the seed repeats), and ``infer(observation)`` returns the ``<f4 [H, 16]`` chunk. The
object lives in the session record, so it survives reconnects exactly like the kit's seed counter and cached response
(``tower-reconnect-v1``: a duplicate request gets the committed bytes, the model is not called again). Model calls run
in worker threads, one at a time across sessions (``model_lock``: one kp_v2 network on the GPU, the runtime on CPU), so
the event loop keeps answering new connections within the 10 s metadata deadline. Everything else (identity checks,
session binding to the worker's headers, request dedupe, error frames, close codes) is the kit's code; the changes are
marked ``# student:``.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from math import ceil
from time import monotonic

from websockets.exceptions import ConnectionClosed

from tower_rl.server.tower_protocol import action_spec, observation_spec, pack, sample_seed, unpack


def handler_factory(track, policy_id, artifact_identity, make_model, *, plugin="tower-rl-student", plugin_version="1",
                    reconnect=True, log=None):
    # Shared across sockets, separate for every episode. A production adapter
    # must retain its own recurrent/chunk state with this session too.
    sessions = {}
    retention = max(300, ceil(track["inference_timeout_s"] + 60))
    model_lock = threading.Lock()  # student: one model instance serves every session, one call at a time
    log = log or (lambda kind, **kw: None)  # student: server event log

    def session_for(request, ws):
        now = monotonic()
        expired = [
            key
            for key, value in sessions.items()
            if value["connections"] == 0 and now - value["detached_at"] > retention
        ]
        for key in expired:
            del sessions[key]
            log("session_expired", session=key)  # student
        session_id = request["session_id"]
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise ValueError("invalid_session")
        binding = tuple(
            ws.request.headers.get(name)
            for name in (
                "X-Tower-Task-Variant",
                "X-Tower-Track-Version",
                "X-Tower-Evaluation-Version",
            )
        )
        state = sessions.get(session_id)
        if state is None:
            # An infer on a new socket is never permission to reset lost state.
            if request["type"] != "reset":
                raise ValueError("session_expired")
            if len(sessions) >= 64:
                raise ValueError("session_capacity")
            state = dict(
                lock=asyncio.Lock(),
                binding=binding,
                connections=0,
                detached_at=now,
                trial=None,
                trials=set(),
                seed=None,
                index=0,
                seen=set(),
                last_id=None,
                last_digest=None,
                response=None,
                invalid=False,
                model=None,  # student: this session's model state (estimator tracking, option runtime)
            )
            sessions[session_id] = state
            log("session_open", session=session_id, binding=list(binding))  # student
        if state["binding"] != binding:
            raise ValueError("session_binding_mismatch")
        return state

    def locked(fn, *args):  # student
        t0 = time.perf_counter()
        with model_lock:
            t1 = time.perf_counter()
            out = fn(*args)
        return out, 1000 * (t1 - t0), 1000 * (time.perf_counter() - t1)

    async def process(r, raw, state):
        if state["invalid"]:
            raise ValueError("session_unavailable")
        base = {k: r[k] for k in ("protocol", "session_id", "trial_id", "request_id", "policy_id")}
        request_id = r["request_id"]
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("invalid_request_id")
        content = hashlib.sha256(raw).digest()
        if request_id == state["last_id"]:
            if content != state["last_digest"]:
                raise ValueError("request_payload_mismatch")
            log("duplicate_request", session=r["session_id"], request=request_id)  # student
            return state["response"]
        if request_id in state["seen"]:
            raise ValueError("stale_request")
        if len(state["seen"]) >= 100000:
            raise ValueError("session_request_limit")
        if r["type"] == "reset":
            if r["trial_id"] in state["trials"]:
                raise ValueError("repeated_trial")
            seed = r["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("invalid_seed")
            # student: clear and rebuild this session's episode state before acknowledging (the model loads in a
            # worker thread: the event loop stays free for other connections)
            if state["model"] is None:
                state["model"] = make_model(r["session_id"], state["binding"][0])
            t0 = time.perf_counter()
            await asyncio.to_thread(state["model"].start, r["trial_id"], seed)
            log("reset", session=r["session_id"], trial=r["trial_id"], seed=seed, variant=state["binding"][0],
                ms=round(1000 * (time.perf_counter() - t0), 1))
            response = dict(
                **base,
                type="reset_ack",
                base_seed=seed,
                seed_scheme="tower-seed-stream-v1",
                next_sample_index=0,
            )
            state.update(trial=r["trial_id"], seed=seed, index=0)
            state["trials"].add(r["trial_id"])
        elif r["type"] == "infer":
            if state["trial"] is None or r["trial_id"] != state["trial"]:
                raise ValueError("reset_required")
            observation = r["observation"]
            prompt = observation["instruction"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("invalid_instruction")
            seed, index = state["seed"], state["index"]
            # student: the session's model instead of the kit's stateless predict (deterministic: prompt and sample
            # seed are not used by the student; the seed fields are still echoed)
            actions, wait_ms, infer_ms = await asyncio.to_thread(locked, state["model"].infer, observation, index)
            response = dict(
                **base,
                type="infer_result",
                plugin=plugin,
                plugin_version=plugin_version,
                base_seed=seed,
                sample_index=index,
                sample_seed=sample_seed(seed, index),
                seed_scheme="tower-seed-stream-v1",
                actions=actions,
                dt_s=track["action_dt_s"],
                action_metadata={},
                server_timing={"infer_ms": round(infer_ms, 1), "queue_ms": round(wait_ms, 1)},  # student: measured
            )
            state["index"] += 1
        else:
            raise ValueError("unsupported_request")
        # Commit the counter and response before sending: if that send is lost,
        # the reconnecting socket gets exactly these bytes without re-inference.
        encoded = pack(response)
        state["seen"].add(request_id)
        state.update(last_id=request_id, last_digest=content, response=encoded)
        return encoded

    async def handle(ws):
        state = None
        session_id = None
        try:
            meta = dict(
                type="metadata",
                protocol="tower-infer-v1",
                ready=True,
                policy_id=policy_id,
                plugin=plugin,
                plugin_version=plugin_version,
                plugin_api_version=1,
                seed_scheme="tower-seed-stream-v1",
                seed_mode="deterministic",
                supports_seed_reset=True,
                artifact_identity=artifact_identity,
                observation_spec=observation_spec(track),
                action_spec=action_spec(track),
                policy_metadata={},
                config_metadata={},
            )
            if reconnect:  # student: optional (--no_reconnect)
                meta["reconnect"] = dict(
                    protocol="tower-reconnect-v1",
                    idempotent_requests=True,
                    retention_s=retention,
                )
            await ws.send(pack(meta))
            async for raw in ws:
                r = unpack(raw)
                try:
                    if r.get("protocol") != "tower-infer-v1" or r.get("policy_id") != policy_id:
                        raise ValueError("identity_mismatch")
                    if state is None:
                        state = session_for(r, ws)
                        session_id = r["session_id"]
                        state["connections"] += 1
                    if r["session_id"] != session_id:
                        raise ValueError("session_mismatch")
                    # A reconnect can arrive while the old handler is still
                    # computing. Serialize against it, then reuse its response.
                    async with state["lock"]:
                        try:
                            response = await process(r, raw, state)
                        except BaseException:
                            # A failed/cancelled model call may have mutated its
                            # state. Never execute that uncertain request again.
                            state["invalid"] = True
                            raise
                    await ws.send(response)
                except (ValueError, KeyError) as exc:
                    log("error", session=session_id, request=r.get("request_id"), code=str(exc))  # student
                    await ws.send(
                        pack(
                            dict(
                                protocol="tower-infer-v1",
                                type="error",
                                request_id=r.get("request_id"),
                                error={"code": str(exc), "message": "Protocol request rejected"},
                            )
                        )
                    )
                    await ws.close(code=1008)
                    return
        except ConnectionClosed:
            pass  # Retain committed state for a reconnect, including a lost response.
        finally:
            if state is not None:
                state["connections"] -= 1
                state["detached_at"] = monotonic()
                log("disconnect", session=session_id, connections=state["connections"])  # student
                if state["model"] is not None and hasattr(state["model"], "flush"):
                    state["model"].flush()  # student: write the episode's summary so far

    handle.sessions = sessions  # student: for tests and status
    return handle
