"""Step 1: fetch a subset of TOWER-SimData episodes (non-image arrays only) for the six Task1-3 variants.

Per variant, ``--per_variant`` episodes are picked from ``MANIFEST.jsonl`` spread over the collection dates
(deterministic: evenly spaced in manifest order within each date). Arrays are read with HTTP range requests
(``tower_rl.data.remote``), so the JPEG streams are never downloaded. Output: ``data/subset/<variant>/<stem>.npz`` and
``<stem>_attrs.json`` plus ``data/subset/fetch_report.json``.

Usage: python -I scripts/fetch_subset.py --per_variant 10 --workers 6
(-I: the script parses downloaded files; it puts the repository on sys.path itself.)
"""
import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.data.remote import fetch_arrays  # noqa: E402
from tower_rl.paths import DATA_DIR  # noqa: E402

VARIANTS = ["task1-2_tower_transfer_2", "task1-3_tower_transfer_3", "task2-2_cross_tower_transfer_2",
            "task2-3_cross_tower_transfer_3", "task3-2_vertical_stacking_2", "task3-3_vertical_stacking_3"]


def pick(rows: list[dict], n: int) -> list[dict]:
    """``n`` rows spread over source dates (round-robin), evenly spaced within each date."""
    by_date = defaultdict(list)
    for r in rows:
        by_date[r["source_date"]].append(r)
    dates = sorted(by_date)
    quota = {d: n // len(dates) + (1 if i < n % len(dates) else 0) for i, d in enumerate(dates)}
    out = []
    for d in dates:
        rs = by_date[d]
        k = min(quota[d], len(rs))
        idx = [round(i * (len(rs) - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [len(rs) // 2]
        out += [rs[i] for i in sorted(set(idx))]
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DATA_DIR / "SimData-meta" / "MANIFEST.jsonl"))
    ap.add_argument("--out_dir", default=str(DATA_DIR / "subset"))
    ap.add_argument("--per_variant", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    jobs = []
    for v in args.variants:
        for r in pick([r for r in rows if r["task"] == v], args.per_variant):
            out = Path(args.out_dir) / v / (Path(r["path"]).stem + ".npz")
            if not out.is_file():
                jobs.append((r["path"], out))
    print(f"[fetch] {len(jobs)} episodes to fetch ({args.per_variant} per variant, {len(args.variants)} variants)", flush=True)
    report_path = Path(args.out_dir) / "fetch_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_arrays, rel, str(out)) for rel, out in jobs]
        for i, fu in enumerate(as_completed(futs)):
            res = fu.result()
            report[res["path"]] = res
            print(f"[fetch] {i + 1}/{len(jobs)} {res['path']} " + (res.get("error") or f"{res['frames']} frames {res['seconds']} s"), flush=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    n_err = sum("error" in r for r in report.values())
    print(f"[fetch] done: {len(report)} episodes recorded, {n_err} errors", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
