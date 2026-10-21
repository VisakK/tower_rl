"""When do the recorded success / terminal flags change, relative to the operations? (Step 1, item 4)."""

from __future__ import annotations

import numpy as np

from tower_rl.data.episode import Episode
from tower_rl.data.events import operations


def first_true(x: np.ndarray) -> int:
    idx = np.nonzero(x)[0]
    return int(idx[0]) if len(idx) else -1


def episode_semantics(ep: Episode) -> dict:
    ops = operations(ep)
    T = ep.num_frames
    t = ep.time
    s, term = first_true(ep.success), first_true(ep.terminal)
    last_open = max((o.open_frame for o in ops), default=-1)
    complete = [e["time"] for e in ep.events if e.get("event") == "transfer_complete"]
    done = [e["time"] for e in ep.events if e.get("event") == "phase" and e.get("phase") == "done"]
    progress = ep.arrays.get("sim__task_progress")
    steps = []
    if progress is not None:
        d = np.nonzero(np.diff(progress[:, 0]) != 0)[0]
        steps = [(round(float(t[i + 1]), 2), round(float(progress[i + 1, 0]), 3)) for i in d]
    return {
        "episode": ep.name,
        "variant": ep.variant,
        "frames": T,
        "duration_s": round(float(t[-1]), 2),
        "success_first_frame": s,
        "terminal_first_frame": term,
        "success_at_last_frame_only": s == T - 1,
        "success_final": bool(ep.success[-1]),
        "terminal_before_success": term >= 0 and (s < 0 or term < s),
        "last_open_frame": last_open,
        "success_delay_after_last_open_s": round(float(t[s] - t[last_open]), 2) if s >= 0 and last_open >= 0 else None,
        "transfer_complete_s": [round(x, 2) for x in complete],
        "done_phase_s": [round(x, 2) for x in done],
        "progress_steps": steps,
        "collapse_any": bool(ep.arrays["sim__collapse"].any()) if "sim__collapse" in ep.arrays else None,
        "outcome": ep.outcome,
    }


def summarize(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    out = {}
    for v, rs in sorted(by.items()):
        delays = [r["success_delay_after_last_open_s"] for r in rs if r["success_delay_after_last_open_s"] is not None]
        out[v] = {
            "episodes": len(rs),
            "all_success_final": all(r["success_final"] for r in rs),
            "success_only_on_last_frame": all(r["success_at_last_frame_only"] for r in rs),
            "terminal_equals_success_frame": all(r["terminal_first_frame"] == r["success_first_frame"] for r in rs),
            "any_terminal_before_success": any(r["terminal_before_success"] for r in rs),
            "success_delay_after_last_open_s": [round(min(delays), 2), round(float(np.mean(delays)), 2), round(max(delays), 2)] if delays else None,
            "any_collapse": any(r["collapse_any"] for r in rs),
            "duration_s": [min(r["duration_s"] for r in rs), max(r["duration_s"] for r in rs)],
        }
    return out
