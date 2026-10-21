"""Rehearsal reports (Step 7.5): per-node pass counts, failure signatures and error distributions -> feedback.md.

Input: episode outcomes from ``tower_rl.eval.runner.run_batch`` with the executor's node events in
``extra["policy_log"]`` (``(t, node id, status, info)``). The markdown is what the offline Claude edit loop reads.
"""

from __future__ import annotations

import datetime
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tower_rl.eval.metrics import EpisodeOutcome, aggregate


def _pct(vals, qs=(50, 90, 100)):
    if not vals:
        return "-"
    a = np.asarray(vals, dtype=float)
    return " / ".join(f"{np.percentile(a, q):.1f}" for q in qs)


def node_stats(outcomes: list[EpisodeOutcome]) -> dict:
    st = defaultdict(lambda: {"attempts": 0, "ok": 0, "failed": 0, "timeout": 0, "dur": [], "info": defaultdict(list), "why": Counter()})
    for o in outcomes:
        log = o.extra.get("policy_log", [])
        prev_t = 0.0
        for t, nid, status, info in log:
            s = st[nid]
            s["attempts"] += 1
            s[status] += 1
            s["dur"].append(t - prev_t)
            prev_t = t
            for k, v in (info or {}).items():
                if isinstance(v, (int, float)):
                    s["info"][k].append(v)
                else:
                    s["why"][str(v)] += 1
    return st


def failure_signatures(outcomes: list[EpisodeOutcome], top: int = 12) -> list[tuple]:
    c = Counter()
    examples = {}
    for o in outcomes:
        for t, nid, status, info in o.extra.get("policy_log", []):
            if status != "ok":
                key = (nid, status, (info or {}).get("why", ""))
                c[key] += 1
                examples.setdefault(key, (o.extra.get("seed"), o.extra.get("case"), round(t, 1)))
        if o.error:
            key = ("<policy>", "error", o.error[:80])
            c[key] += 1
            examples.setdefault(key, (o.extra.get("seed"), o.extra.get("case"), o.duration_s))
    return [(k, n, examples[k]) for k, n in c.most_common(top)]


def write_report(outcomes: list[EpisodeOutcome], out_dir: str | Path, title: str, config: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = aggregate(outcomes)
    lines = [f"# Rehearsal feedback: {title}", "",
             f"{datetime.datetime.now().isoformat(timespec='minutes')} | episodes {len(outcomes)} | config `{json.dumps(config, sort_keys=True)}`", "",
             "## Metrics", "", "| metric | value | k / n | Wilson 95 % |", "|---|---|---|---|"]
    for k in ("TSR", "GSR", "PSR", "LHPR"):
        v = m[k]
        lines.append(f"| {k} | {v['value']:.3f} | {v['k']} / {v['n']} | {v['wilson95'][0]:.2f}-{v['wilson95'][1]:.2f} |")
    lines.append(f"| ODR | {m['ODR']['value']:.3f} | {m['ODR']['disturbed']} / {m['ODR']['steps']} | |")
    tc = [o.extra["task_success_collapse_only"] for o in outcomes if "task_success_collapse_only" in o.extra]
    if tc:  # Task 4 / 5: success under the generator's collapse rule only (no 10 mm structure rule)
        lines.append(f"| TSR collapse-only | {np.mean(tc):.3f} | {sum(tc)} / {len(tc)} | |")
    act = [o.extra.get("act_ms_mean") for o in outcomes if o.extra.get("act_ms_mean") is not None]
    actmax = [o.extra.get("act_ms_max") for o in outcomes if o.extra.get("act_ms_max") is not None]
    calls = [o.extra.get("act_calls", 0) for o in outcomes]
    lines += ["", f"policy errors {m['policy_errors']}, mean duration {m['mean_duration_s']} s; executor wall time per call "
              f"mean {np.mean(act) if act else 0:.1f} ms, max {np.max(actmax) if actmax else 0:.1f} ms; "
              f"inference calls per episode mean {np.mean(calls) if calls else 0:.0f}"]
    gaps = [o.extra["min_arm_gap_m"] for o in outcomes if o.extra.get("min_arm_gap_m") is not None]
    if gaps:  # Step 8.2: closest approach of the two arms (elbow to pinch points) per episode
        lines.append(f"closest arm-arm approach per episode (m): min {np.min(gaps):.3f}, p10 {np.percentile(gaps, 10):.3f}, "
                     f"median {np.median(gaps):.3f}")
    lines.append("")
    by_case = defaultdict(list)
    for o in outcomes:
        by_case[o.extra.get("case", "?")].append(o)
    if len(by_case) > 1:
        lines += ["## By robustness case", "", "| case | n | TSR | GSR | PSR | ODR |", "|---|---|---|---|---|---|"]
        for c, os_ in sorted(by_case.items()):
            a = aggregate(os_)
            lines.append(f"| {c} | {len(os_)} | {a['TSR']['value']:.2f} | {a['GSR']['value']:.2f} | {a['PSR']['value']:.2f} | {a['ODR']['value']:.2f} |")
        lines.append("")
    st = node_stats(outcomes)
    lines += ["## Per-node outcomes", "", "| node | attempts | ok | failed | timeout | duration p50/p90/max (s) | measurements p50/p90/max |",
              "|---|---|---|---|---|---|---|"]
    for nid, s in st.items():
        meas = "; ".join(f"{k} {_pct(v)}" for k, v in s["info"].items())
        lines.append(f"| {nid} | {s['attempts']} | {s['ok']} | {s['failed']} | {s['timeout']} | {_pct(s['dur'])} | {meas} |")
    lines += ["", "## Failure signatures (most frequent first)", "", "| node | status | why | count | example (seed, case, t) |", "|---|---|---|---|---|"]
    for (nid, status, why), n, ex in failure_signatures(outcomes):
        lines.append(f"| {nid} | {status} | {why} | {n} | {ex} |")
    dist = Counter()
    for o in outcomes:
        for i, d in enumerate(o.disturbed_per_op):
            if d:
                dist[i] += d
    lines += ["", "## Disturbances by operation window (index: G0, P0, G1, P1, ..., final)", "",
              ", ".join(f"{i}: {n}" for i, n in sorted(dist.items())) or "none", ""]
    fails = [o for o in outcomes if not o.task_success]
    lines += ["## Failed episodes (seeds for videos)", "",
              ", ".join(f"{o.extra.get('seed')}({o.extra.get('case')})" for o in fails[:20]) or "none", ""]
    path = out_dir / "feedback.md"
    path.write_text("\n".join(lines))
    (out_dir / "metrics.json").write_text(json.dumps({"config": config, "metrics": m}, indent=1))
    (out_dir / "episodes.json").write_text(json.dumps([o.to_dict() for o in outcomes], indent=1, default=str))
    return path
