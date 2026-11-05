"""Camera loop, an arm run with graph depart_seen: every push where the v1.1 rule would have ended the push on an
unconfident row (the estimate passed push.departed with confidence < 0.5). At that row: was the block truly departed?
How did the push end, and where did the block come to rest?

  python3 scripts/depart_counterfactual.py <arm dir under results/v1_1/t4_camera> [variant 4-2|4-3]
"""
import glob
import json
import pickle
import re
import sys
import collections

import numpy as np

R = '/home/visakii/Documents/tower_rl'
arm, v = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '4-2')
var = {'4-2': 'task4-2_full_tower_pushdown_2', '4-3': 'task4-3_full_tower_pushdown_3'}[v]
g = json.load(open(f'{R}/results/bundles/student_v1_1/graphs/{var}.json'))
blk = {n['id']: n['block'] for n in g['nodes'] if n.get('skill') == 'push'}
order = [n['block'] for n in g['nodes'] if n.get('skill') == 'push']
d, = glob.glob(f'{R}/results/v1_1/t4_camera/{arm}/{var}/*/')
eps = json.load(open(d + 'episodes.json'))
pl = pickle.load(open(d + 'perception_log.pkl', 'rb'))
ids = pl['block_ids']


def dep(p, txy):
    r = np.linalg.norm(p[:2] - txy)
    return bool(r >= 0.16 or (r >= 0.09 and p[2] <= 0.02)), r


tab = collections.Counter()
for ep in eps:
    x = ep['extra']
    L = [e for e in pl['log'] if e['seed'] == x['seed']]
    txy = np.mean([L[0]['true_pos'][ids.index(f)][:2] for f in g['tower']['foundation']], 0)
    t_prev = 0.0
    for t, name, res, info in x['policy_log']:
        m = re.match(r'(p\d+)_push$', name)
        if m:
            i = ids.index(blk[m.group(1) + '_push'])
            win = [e for e in L if t_prev < e['t'] <= t + 1e-6]
            hit = next((e for e in win if e['conf'][i] < 0.5 and dep(e['pos'][i], txy)[0]), None)
            if hit is not None:
                td, tr = dep(hit['true_pos'][i], txy)
                rest = x['rest'][order.index(blk[m.group(1) + '_push'])][0]
                end = f'{res}:{info.get("why", "departed")}'
                tab[('truly departed' if td else 'NOT departed', end, rest)] += 1
                print(f"  {x['case']:17s} {name} at {hit['t']:5.1f}: true r {tr*1000:4.0f} ({'departed' if td else 'NOT departed'}), "
                      f"push ended {end} at {t:5.1f}, rest {rest}")
        t_prev = t
print()
for k, n in sorted(tab.items()):
    print(n, k)
