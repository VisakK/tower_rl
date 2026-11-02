"""Dips on recorded reacher paths (``reach_task_check.py --dump ... --traj``): how far the pinch went below the lower of
its start and goal heights, per transit target, goals needing <= 90 deg of turn (the rest go to the planned transit).
10-31: on CPU PhysX the GPU-trained left reacher overshot pre-grasp descents by up to 7 cm.

  python scripts/reach_dips.py results/reach/tasks/check_left_<tag>.json [...]
"""
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.kinematics import ArmBase, ArmKinematics  # noqa: E402
from tower_rl.skills.reach_tasks import load_goals  # noqa: E402

G = {g['id']: g for g in load_goals(sorted(glob.glob('results/reach/tasks/*_*.jsonl')))}
for f in sys.argv[1:]:
    arm = 'left' if '_left_' in f else 'back'
    kin, base = ArmKinematics(device='cpu').set_arm(arm), ArmBase(arm, device='cpu')
    R = json.load(open(f))
    if 'q_path' not in R[0]: print(f, 'no traj'); continue
    by = {}
    for r in R:
        g = G[r['id']]
        if g.get('start_rot_deg', 0) > 90: continue
        pp = base.to_world(*kin.fk(torch.tensor(r['q_path']), 'pinch'))[0]
        z = pp[:, 2]
        # dip below the lowest of start / goal, and where (row) the min happened
        d = max(0.0, min(float(z[0]), g['goal_pos'][2]) - float(z.min())) * 1000
        k = int(z.argmin())
        tg = g['target'] + ('_held' if g['held'] else '')
        by.setdefault(tg, []).append((d, k, len(z)))
    allv = [x for v in by.values() for x in v]
    def s(v):
        u = sorted(x[0] for x in v); q = lambda p: u[int(p * (len(u) - 1))]
        late = sum(1 for x in v if x[0] > 10 and x[1] > 3) / len(v)
        return f"n {len(v):3d} p90 {q(.9):5.1f} max {max(u):5.1f} >10mm {sum(x>10 for x in u)/len(u):.3f} (after row 3: {late:.3f})"
    print(f.split('check_')[1][:-5].ljust(24), 'all', s(allv))
    for tg in sorted(by): print(' ' * 24, tg.ljust(14), s(by[tg]))
