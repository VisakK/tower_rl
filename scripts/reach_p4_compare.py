"""P4 gate table (Reach-PPO-plan.MD): teacher rehearsals with the learned reacher against the planned transit on the
same seeds, per variant / tier: TSR, GSR, PSR, LHPR (k / n), disturbed steps, mean episode time and policy calls.

  python3 scripts/reach_p4_compare.py results/reach/tasks/rehearsal_heldout results/reach/p4/p2b
"""
import json
import sys
from pathlib import Path

base, new = Path(sys.argv[1]), Path(sys.argv[2])
KEYS = ("TSR", "GSR", "PSR", "LHPR")


def load(d):
    m = json.loads((d / "metrics.json").read_text())["metrics"]
    eps = json.loads((d / "episodes.json").read_text())
    calls = sum(e["extra"].get("act_calls", 0) for e in eps) / max(len(eps), 1)
    return m, calls


tot = {k: [0, 0, 0, 0] for k in KEYS}
worse = []
print(f"{'variant/tier':38s} " + " ".join(f"{k:>13s}" for k in KEYS) + "   ODR p/l   time s p/l   calls p/l")
for d in sorted(new.glob("*/D*")):
    b = base / d.parent.name / d.name
    if not (d / "metrics.json").exists() or not (b / "metrics.json").exists():
        continue
    (mb, cb), (mn, cn) = load(b), load(d)
    cells = []
    for k in KEYS:
        kb, nb, kn, nn = mb[k]["k"], mb[k]["n"], mn[k]["k"], mn[k]["n"]
        tot[k] = [tot[k][0] + kb, tot[k][1] + nb, tot[k][2] + kn, tot[k][3] + nn]
        cells.append(f"{kb:3d}/{nb:<3d}{kn:3d}/{nn:<3d}")
        if k == "TSR" and nb and nn and kn / nn < kb / nb - 1 / 24:
            worse.append(f"{d.parent.name}/{d.name} TSR {kb}/{nb} -> {kn}/{nn}")
    print(f"{d.parent.name + '/' + d.name:38s} " + " ".join(cells)
          + f"   {mb['ODR']['disturbed']:2d}/{mn['ODR']['disturbed']:<2d}   {mb['mean_duration_s']:5.1f}/{mn['mean_duration_s']:<5.1f}"
          + f"   {cb:4.0f}/{cn:<4.0f}")
print(f"{'all':38s} " + " ".join(f"{v[0]:3d}/{v[1]:<3d}{v[2]:3d}/{v[3]:<3d}" for v in tot.values()))
print("cells worse than planned by more than 1/24:", worse or "none")
