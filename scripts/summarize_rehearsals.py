"""Collect results/rehearsal/<variant>/<tier>/metrics.json into one markdown table (results/rehearsal/summary.md).

Usage: python -I scripts/summarize_rehearsals.py [--root results/rehearsal]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.paths import RESULTS_DIR  # noqa: E402


def step7(variant: str, tier: str, tsr: float, odr: float) -> str:
    """Build-file Step 7 acceptance: TSR >= 0.8 / 0.6 on D0 and >= 0.6 / 0.4 on D1 (2- / 3-layer variants), ODR < 0.1."""
    base = tier.split("_")[0]
    if base not in ("D0", "D1"):
        return ""
    three = variant.split("_")[-1] == "3"
    target = {("D0", False): 0.8, ("D0", True): 0.6, ("D1", False): 0.6, ("D1", True): 0.4}[(base, three)]
    ok = tsr >= target and odr < 0.1
    return f"{'pass' if ok else 'FAIL'} (TSR >= {target}, ODR < 0.1)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RESULTS_DIR / "rehearsal"))
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for f in sorted(root.glob("*/*/metrics.json")):
        d = json.loads(f.read_text())
        m, c = d["metrics"], d["config"]
        rows.append((f.parent.parent.name, f.parent.name, m["episodes"], m["TSR"]["value"], m["TSR"]["wilson95"], m["GSR"]["value"],
                     m["PSR"]["value"], m["LHPR"]["value"], m["ODR"]["value"], m["mean_duration_s"], c.get("bundle") or "planned"))
    lines = ["| variant | tier | n | TSR | TSR 95 % | GSR | PSR | LHPR | ODR | mean s | skills | Step 7 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4][0]:.2f}-{r[4][1]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:.2f} | {r[9]:.0f} | {Path(str(r[10])).name} | {step7(r[0], r[1], r[3], r[8])} |")
    out = root / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
