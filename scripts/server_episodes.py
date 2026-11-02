"""Student plan Step 11: the episodes a policy server logged, summarised from the server's side, for the official
evaluation (whose outcomes arrive only on the challenge site).

``serve_student.py --name N`` writes ``results/student/server/N/``: ``server.jsonl`` (session events) and per episode
``episodes/<variant>/<trial>.jsonl`` (one row per call) and ``<trial>_summary.json``. Per variant this prints the
episodes and how they ended, next to a local run's successes (``--compare``, e.g. the Step 10 camera loop); with
``--episodes`` also one row per episode (calls, simulated time reached, the runtime's last node and events, holds).

How an episode ended, from the server's side alone: "limit" when its last chunk reaches the track's simulation limit;
"early" when the evaluator stopped it before that (our local evaluator stops on success; the official one does not:
official_1 scored up to 90 % TSR with every scored episode at the limit, and its three early ends were evaluator-side
and re-run, e.g. `1e6d738a` cut at 164 s). So officially neither "limit" nor "early" says anything about success;
"open" while neither a later reset of its session nor a disconnect after its last call has been logged (a disconnect
counts only once it is RECONNECT_GRACE_S old: official sessions drop and reconnect, e.g. over a tunnel blip, and the
episode goes on); "empty" when its session closed without a single call (the worker reconnects and starts the episode
over in a new session).

  python scripts/server_episodes.py results/student/server/official_1 --since 2026-10-24T11:10 \
      --track data/challenge/track.json --compare results/student/closed_loop/student_v1_kp_v2 [--episodes] [--json out.json]
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def read_jsonl(p: Path) -> list[dict]:
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except ValueError:  # a row being written
            pass
    return out


def pct(x: list[float], q: float):
    if not x:
        return None
    s = sorted(x)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


RECONNECT_GRACE_S = 120  # a dropped session reconnected within ~30 s in official run 1


def episodes(log_dir: Path, since: str, limit_steps: int, dt: float) -> list[dict]:
    ev = read_jsonl(log_dir / "server.jsonl")
    settled = (datetime.now() - timedelta(seconds=RECONNECT_GRACE_S)).isoformat(timespec="milliseconds")
    resets = {r["trial"]: r for r in ev if r.get("kind") == "reset"}
    per_session = defaultdict(list)
    for r in ev:
        if r.get("kind") in ("reset", "disconnect"):
            per_session[r["session"]].append(r)
    out = []
    for f in sorted((log_dir / "episodes").glob("*/*.jsonl")):
        rs = resets.get(f.stem)
        if rs is None or rs["time"] < since:
            continue
        rows = read_jsonl(f)
        last = rows[-1] if rows else {"step": 0, "rows": 0, "node": {}}
        written = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="milliseconds")
        closed = any(r["time"] > rs["time"] and (r["kind"] == "reset" or written <= r["time"] <= settled)
                     for r in per_session[rs["session"]])
        reach = last["step"] + last["rows"]
        sp = f.with_name(f.stem + "_summary.json")
        summ = json.loads(sp.read_text()) if sp.exists() else {}
        evs = summ.get("events") or []
        node = last.get("node") or {}
        confs = [min(r["conf"]) for r in rows if r.get("conf")]
        out.append({
            "variant": f.parent.name, "trial": f.stem, "session": rs["session"], "seed": rs.get("seed"), "reset": rs["time"],
            "calls": len(rows), "t_end": round(reach * dt, 1),
            "end": "open" if not closed else "empty" if not rows else "limit" if reach >= limit_steps - 1 else "early",
            "last_node": node.get("node") or "-", "events": len(evs), "last_event": f"{evs[-1][1]}:{evs[-1][2]}" if evs else "-",
            "not_ok": [f"{e[0]}:{e[1]}:{e[2]}" for e in evs if e[2] != "ok"], "decisions": summ.get("decisions"),
            "holds": dict(Counter(r["hold"] for r in rows if r.get("hold"))), "errors": summ.get("errors", 0),
            "clipped": sum(r.get("clipped", 0) for r in rows), "min_conf": round(min(confs), 3) if confs else None,
            "est_ms_p50": pct([r["est_ms"] for r in rows if "est_ms" in r], 0.5),
            "total_ms_p99": pct([r["total_ms"] for r in rows if "total_ms" in r], 0.99)})
    return out


def local_results(root: Path) -> dict:
    """Per variant: (successes, episodes, the successes' median duration) of a local run's ``episodes.json`` files."""
    res = {}
    for v in sorted(p for p in root.iterdir() if p.is_dir()):
        eps = [e for f in sorted(v.glob("*/episodes.json")) for e in json.loads(f.read_text())]
        ok = [e["duration_s"] for e in eps if e["task_success"]]
        res[v.name] = (len(ok), len(eps), round(statistics.median(ok), 1) if ok else None)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--since", default="", help="only trials reset at or after this local ISO time")
    ap.add_argument("--track", default="data/challenge/track.json")
    ap.add_argument("--compare", default=None, help="a local run's root (<variant>/<tier>/episodes.json)")
    ap.add_argument("--episodes", action="store_true", help="also one row per episode")
    ap.add_argument("--json", default=None, help="write the episode records here")
    args = ap.parse_args()
    track = json.loads(Path(args.track).read_text())
    dt = track["action_dt_s"]
    eps = episodes(Path(args.log_dir), args.since, round(track["simulation_limit_s"] / dt), dt)
    local = local_results(Path(args.compare)) if args.compare else {}
    by_v = defaultdict(list)
    for e in eps:
        by_v[e["variant"]].append(e)
    print("| variant | episodes | early | limit | open | empty | early end t median s | calls mean | holds | errors | local successes | local t median s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = Counter()
    for v in sorted(set(by_v) | set(local)):
        es = by_v.get(v, [])
        c = Counter(e["end"] for e in es)
        tot.update(c)
        tot["episodes"] += len(es)
        early_t = [e["t_end"] for e in es if e["end"] == "early"]
        holds = Counter()
        for e in es:
            holds.update(e["holds"])
        ls, ln, lt = local.get(v, (None, None, None))
        tot["local_ok"] += ls or 0
        tot["local_n"] += ln or 0
        print(f"| {v} | {len(es)} | {c['early']} | {c['limit']} | {c['open']} | {c['empty']} | "
              f"{round(statistics.median(early_t), 1) if early_t else '-'} | "
              f"{round(sum(e['calls'] for e in es) / len(es)) if es else '-'} | {dict(holds) or '-'} | "
              f"{sum(e['errors'] for e in es)} | {f'{ls}/{ln}' if ln else '-'} | {lt or '-'} |")
    print(f"| all | {tot['episodes']} | {tot['early']} | {tot['limit']} | {tot['open']} | {tot['empty']} | | | | | "
          f"{tot['local_ok']}/{tot['local_n']} | |")
    if args.episodes:
        print("\n| variant | reset | seed | calls | t end s | end | last node | events | last event | not ok | holds | min conf | est p50 ms | total p99 ms |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in sorted(eps, key=lambda e: (e["variant"], e["reset"])):
            print(f"| {e['variant']} | {e['reset'][11:19]} | {e['seed']} | {e['calls']} | {e['t_end']} | {e['end']} | {e['last_node']} | "
                  f"{e['events']} | {e['last_event']} | {' '.join(e['not_ok'][:4]) or '-'} | {e['holds'] or '-'} | {e['min_conf']} | "
                  f"{e['est_ms_p50']} | {e['total_ms_p99']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(eps, indent=1))


if __name__ == "__main__":
    main()
