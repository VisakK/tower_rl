"""Failure injection for data collection (student plan Step 6): DART-style perturbations of option execution, so the
logged states include real drops, misplacements, knocks and shoves to recover from.

With probability ``p_inject`` per goal option, one perturbation (``configs/student/collect.yaml`` ``inject``) is written
into every node spec of the option as ``spec["inject"]``; the node classes apply it through opt-in hooks
(``tower_rl.graph.executor``: shifted pre-grasp / pre-push, a drop part-way up the lift or during the held transit, a
shifted or yawed goal, an entry offset at the slot). The decision itself is the expert's; its row's source is
"inject".
"""

from __future__ import annotations

import math

import numpy as np

from tower_rl.student.spec import Decision


def _horizontal(rng, lo_hi) -> list[float]:
    a = rng.uniform(0, 2 * math.pi)
    r = rng.uniform(*lo_hi)
    return [r * math.cos(a), r * math.sin(a), 0.0]


def _across(rng, direction, lo_hi) -> list[float]:
    d = np.asarray(direction, dtype=float)
    lat = np.array([-d[1], d[0], 0.0])
    lat /= max(np.linalg.norm(lat), 1e-9)
    return (lat * rng.uniform(*lo_hi) * rng.choice([-1.0, 1.0])).tolist()


class Injector:
    def __init__(self, cfg: dict, p_inject: float, rng: np.random.Generator):
        self.cfg, self.p, self.rng = cfg, p_inject, rng

    def maybe(self, d: Decision, nodes: list[dict]) -> dict | None:
        """Perturb option ``d`` (its node specs, in place) with probability ``p_inject``; returns the perturbation."""
        c = self.cfg.get(d.option)
        if c is None or not nodes or self.rng.random() >= self.p:
            return None
        rng = self.rng
        if d.option == "acquire":
            inj = {"target_shift": _horizontal(rng, c["lateral_m"])} if rng.random() < 0.5 else \
                {"drop_at_lift_m": float(rng.uniform(*c["drop_at_lift_m"]))}
        elif d.option == "deliver":
            u = rng.random()
            if u < 0.4:
                inj = {"goal_shift": _horizontal(rng, c["goal_shift_m"])}
            elif u < 0.7:
                inj = {"goal_yaw_deg": float(rng.uniform(*c["goal_yaw_deg"]) * rng.choice([-1.0, 1.0]))}
            else:
                inj = {"drop_after_rows": int(rng.integers(c["drop_after_rows"][0], c["drop_after_rows"][1] + 1))}
        elif d.option == "push_off":
            direction = nodes[0].get("direction", [1.0, 0.0, 0.0])
            if "stop_r_m" in c and rng.random() < c.get("short_frac", 0.0):
                # a push that stops short (the block left over the foundation footprint: the recovery node's re-push)
                inj = {"stop_r_m": float(rng.uniform(*c["stop_r_m"]))}
            else:
                inj = {"target_shift": _across(rng, direction, c["lateral_m"])} if rng.random() < 0.6 else \
                    {"target_shift": [0.0, 0.0, -float(c["low_m"])]}
        elif d.option in ("insert", "push_home"):
            direction = nodes[0].get("direction", [1.0, 0.0, 0.0])
            inj = {"slot_shift": _across(rng, direction, c["across_m"])} if rng.random() < 0.6 else \
                {"slot_shift": [0.0, 0.0, -float(c["low_m"])]}
        else:
            return None
        for n in nodes:
            n["inject"] = inj
        return inj
