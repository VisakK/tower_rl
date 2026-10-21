"""Challenge metrics (docs "Results and videos", rules v1.3) from per-episode operation outcomes (pure python).

Per episode with G planned grasps and P planned places:
  TSR  : episodes with task_success / evaluated episodes
  GSR  : successful G operations / planned G operations
  PSR  : successful P operations / planned P operations
  LHPR : successful G + P operations / planned G + P (successes after an earlier failure still count)
  ODR  : disturbed non-target block-occurrences / counted steps, each episode contributing G + P + 1 - task_success
         steps (a block counts once per operation; current target, foundations and tools excluded).
Counts are summed over episodes before dividing (docs: "Counts are summed before division"). Planned totals include
skipped and unstarted operations. 95 % Wilson intervals are reported for the rates.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class EpisodeOutcome:
    variant: str
    planned_g: int
    planned_p: int
    grasp_ok: list[bool]            # per planned transfer, in plan order
    place_ok: list[bool]
    disturbed_per_op: list[int]     # distinct disturbed non-target blocks per counted step (ops + final window)
    task_success: bool
    duration_s: float = 0.0
    error: str | None = None        # policy error ends the episode as a failure
    extra: dict = field(default_factory=dict)

    @property
    def counted_steps(self) -> int:
        return self.planned_g + self.planned_p + 1 - int(self.task_success)

    def to_dict(self) -> dict:
        return asdict(self)


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def aggregate(outcomes: list[EpisodeOutcome]) -> dict:
    n = len(outcomes)
    g_ok = sum(sum(o.grasp_ok) for o in outcomes)
    p_ok = sum(sum(o.place_ok) for o in outcomes)
    g_pl = sum(o.planned_g for o in outcomes)
    p_pl = sum(o.planned_p for o in outcomes)
    succ = sum(int(o.task_success) for o in outcomes)
    dist = sum(sum(o.disturbed_per_op) for o in outcomes)
    steps = sum(o.counted_steps for o in outcomes)

    def rate(k, m):
        return {"value": round(k / m, 4) if m else 0.0, "k": k, "n": m, "wilson95": [round(x, 4) for x in wilson(k, m)]}

    return {
        "episodes": n,
        "TSR": rate(succ, n),
        "GSR": rate(g_ok, g_pl),
        "PSR": rate(p_ok, p_pl),
        "LHPR": rate(g_ok + p_ok, g_pl + p_pl),
        "ODR": {"value": round(dist / steps, 4) if steps else 0.0, "disturbed": dist, "steps": steps},
        "policy_errors": sum(o.error is not None for o in outcomes),
        "mean_duration_s": round(sum(o.duration_s for o in outcomes) / n, 2) if n else 0.0,
    }


def aggregate_by_variant(outcomes: list[EpisodeOutcome]) -> dict:
    by = {}
    for o in outcomes:
        by.setdefault(o.variant, []).append(o)
    out = {v: aggregate(os_) for v, os_ in sorted(by.items())}
    out["all"] = aggregate(outcomes)
    return out
